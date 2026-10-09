"""Fresh R005 native97 FP32 stage: sequential TF + 0.1 * BID reconstruction.

No change to the existing TF trajectory, masks, optimizer or resume contract.
This entry reuses its checked lineage/model/precision helpers. The new stage
has a distinct checkpoint identity and a fixed20-step/160-window trajectory.
GPU launch is function-only through the shared leased/watchdog worker.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

import train_action_teacher as action_entry
import train_causal_teacher_forcing as tf_entry
import train_causal_r005_probe as tf_probe
from training.causal_moba import (
    CausalMoBAConfig, backward_microbatch, load_moba_config,
    verify_bidirectional_interface,
)
from training.data.action_resampled import sampling_contract
from training.models.lora import load_trainable_state_dict, trainable_state_dict
from training.runtime import CheckpointManager, append_jsonl, load_checkpoint, seed_everything

STAGE_NAME = "causal_r005_native97_fp32_bid_v1"
FACTORY = tf_probe.FACTORY
_require = tf_probe._require


def validate_probe_config(config):
    _require(isinstance(config, CausalMoBAConfig), "BID stage requires its explicit regularization configuration")
    tf_probe.validate_probe_config(config)
    _require(config.regularization.bidirectional_weight == 0.1, "paired BID stage fixes auxiliary weight0.1")


def load_config(path):
    config = load_moba_config(path)
    validate_probe_config(config)
    return config


def method_contract():
    result = tf_probe.method_contract()
    result.update(method=STAGE_NAME, objective="causal_TF_plus_0.1_bidirectional_flow_matching",
        bidirectional_auxiliary=True, bidirectional_weight=0.1,
        backward_order=["causal_teacher_forcing", "bidirectional_auxiliary"],
        same_batch_noise_actions_prompt=True, bidirectional_clean_context=False,
        bidirectional_kv_cache=False, first_latent_in_loss=False,
        prompt_policy="scene_static_only_v1", prompt_storage="inline_97_frame_feature_shards",
        simultaneous_tf_bid_graphs=False, packed_moba_mask=False)
    return result


def load_parent(config, config_path):
    validate_probe_config(config)
    # This preserves the exact original R005 pin, cache receipt, architecture
    # names/shapes and weights-only initialization. No causal11 optimizer enters.
    return tf_probe.load_parent(config, config_path)


def validate_resume(path, *, config, hashes, lineage, saved_tensors_on_cpu):
    validate_probe_config(config)
    payload = load_checkpoint(path, expected_manifest_hashes=hashes, map_location="cpu")
    _require(payload.get("format_version") == 1 and payload.get("stage") == STAGE_NAME,
             "resume requires this TF+BID FP32 stage, not the pure-TF trajectory")
    _require(payload.get("config") == config.to_dict() and payload.get("initialization") == lineage
             and payload.get("method_contract") == method_contract(), "strict BID resume contract changed")
    _require(payload.get("sampling_contract") == sampling_contract(config.training), "BID resume sampling changed")
    _require(payload.get("saved_tensors_on_cpu") is saved_tensors_on_cpu, "BID resume saved-tensor policy changed")
    step, micro = payload.get("step"), payload.get("micro_batches_consumed")
    _require(type(step) is int and 0 < step <= 20 and type(micro) is int and micro == step * 8,
             "BID resume optimizer/data cursor is inconsistent")
    _require(all(name in payload for name in ("optimizer", "torch_rng_state", "cuda_rng_state_all",
                                              "python_rng_state", "numpy_rng_state")), "BID resume lacks optimizer/RNG state")
    _require(all(key in payload.get("metrics", {}) for key in ("loss_tf", "loss_bid", "loss_total")),
             "BID checkpoint lacks separate objective measurements")
    action_entry._validate_lora_state_dtype(payload.get("trainable_model"), "float32")
    return payload


def launch(config, config_path, *, shared_lease, shared_max_seconds,
           shared_request_sha256, source_revision, stop_after_step=2, resume=None,
           saved_tensors_on_cpu=True):
    validate_probe_config(config)
    _require(type(stop_after_step) is int and 1 <= stop_after_step <= 20, "BID execution cap must be1..20")
    _require(type(saved_tensors_on_cpu) is bool, "saved_tensors_on_cpu must be an explicit boolean")
    _require(isinstance(source_revision, str) and bool(source_revision.strip()), "deployed source revision required")
    output = Path(config.training.output_dir)
    _require(resume is not None or not output.exists() or not any(output.iterdir()), "fresh BID output must be empty")
    if resume:
        _require(output.resolve() in Path(resume).resolve().parents, "BID resume checkpoint is outside this run")
    parent, lineage, hashes = load_parent(config, config_path)
    restored = validate_resume(resume, config=config, hashes=hashes, lineage=lineage,
                               saved_tensors_on_cpu=saved_tensors_on_cpu) if resume else None
    start = restored["step"] if restored else 0
    _require(start < stop_after_step, "BID execution cap must exceed restored step")
    gpu = action_entry._shared_training_gpu(shared_lease, max_seconds=shared_max_seconds,
                                            request_sha256=shared_request_sha256)
    _require(torch.cuda.is_available(), "leased BID launch requires CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(config.training.seed)
    output.mkdir(parents=True, exist_ok=True)
    policy = sampling_contract(config.training)
    metadata = dict(stage=STAGE_NAME, config=config.to_dict(), manifest_hashes=hashes,
        initialization=lineage, method_contract=method_contract(), source_revision=source_revision,
        sampling_contract=policy, command=sys.argv, gpu_state=gpu.as_dict(),
        execution_stop_after_step=stop_after_step, resume=resume,
        saved_tensors_on_cpu=saved_tensors_on_cpu, started_unix=time.time())
    # Preserve this new stage's initial execution receipt across strict resumes.
    if not resume:
        (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    append_jsonl(output / "executions.jsonl", metadata)
    try:
        model, summary = tf_entry._build_model(config, parent, device)
        del parent
        action_entry._apply_lora_parameter_dtype(model, "float32")
        verify_bidirectional_interface(model)
        optimizer = tf_entry._build_optimizer(model, config)
        micro = start * 8
        if restored:
            load_trainable_state_dict(model, restored["trainable_model"])
            optimizer.load_state_dict(restored["optimizer"])
            action_entry._restore_rng_state(restored, require_complete=True)
            del restored
        loader = tf_entry._import_factory(FACTORY)(config=config.data, training=config.training)
        _, iterator = tf_entry._resume_iterator_preserving_rng(loader, micro)
        model.zero_grad(set_to_none=True)
        # This finite stage is capped at20; preserve all endpoint evidence.
        manager = CheckpointManager(output, keep_last=20)
        began = time.monotonic()
        latest = {}
        for step in range(start + 1, stop_after_step + 1):
            torch.cuda.synchronize(device)
            step_began = time.monotonic()
            before = {name: value.clone() for name, value in trainable_state_dict(model).items()}
            sums = dict(loss_tf=0.0, loss_bid=0.0, loss_total=0.0)
            for _ in range(8):
                values = backward_microbatch(model, next(iterator), config, device,
                                             saved_tensors_on_cpu=saved_tensors_on_cpu)
                _require(values["bid_evaluated"] is True, "BID objective was not evaluated")
                for key in sums:
                    sums[key] += float(values[key])
                micro += 1
            precision = action_entry._lora_gradient_precision(model)
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.optimizer.max_grad_norm)
            _require(bool(torch.isfinite(norm)), "nonfinite TF+BID gradient norm")
            optimizer.step()
            evidence = tf_probe._update_evidence(model, before)
            del before
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            latest = dict(**{key: value / 8 for key, value in sums.items()}, loss=sums["loss_total"] / 8,
                grad_norm=float(norm), precision=precision, **evidence,
                optimizer_step_seconds=time.monotonic() - step_began,
                peak_vram_bytes=int(torch.cuda.max_memory_allocated(device)),
                peak_reserved_vram_bytes=int(torch.cuda.max_memory_reserved(device)),
                samples_per_second=8 / (time.monotonic() - step_began), bidirectional_weight=0.1)
            append_jsonl(output / "metrics.jsonl", dict(step=step, micro_batches_consumed=micro, **latest))
            print(json.dumps(dict(step=step, **latest)), flush=True)
            if step % 20 == 0 or step == stop_after_step:
                payload = dict(format_version=1, stage=STAGE_NAME, step=step, micro_batches_consumed=micro,
                    config=config.to_dict(), manifest_hashes=hashes, initialization=lineage,
                    method_contract=method_contract(), saved_tensors_on_cpu=saved_tensors_on_cpu,
                    sampling_contract=policy, trainable_model=trainable_state_dict(model),
                    optimizer=optimizer.state_dict(), metrics=latest,
                    torch_rng_state=torch.get_rng_state(), cuda_rng_state_all=torch.cuda.get_rng_state_all(),
                    python_rng_state=random.getstate(), numpy_rng_state=np.random.get_state())
                manager.save(payload, step=step, metric=latest["loss_total"])
        result = dict(status="completed", stage=STAGE_NAME, step=stop_after_step,
            micro_batches_consumed=micro, resume_step=start, elapsed_seconds=time.monotonic() - began,
            trainable_parameters=summary.trainable_parameters, **latest,
            checkpoint_path=str((output / "checkpoints" / f"step-{stop_after_step:07d}.pt").resolve()),
            quality_improvement_verified=False)
        (output / "execution_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    except BaseException as exc:
        failure = dict(status="failed", stage=STAGE_NAME, failure_type=type(exc).__name__, error=str(exc),
            peak_vram_bytes=int(torch.cuda.max_memory_allocated(device)))
        (output / "failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-parent", action="store_true", help="CPU-only cache and R005 lineage verification")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    report = tf_entry.cpu_dry_run(config, args.config)
    report.update(stage=STAGE_NAME, method_contract=method_contract(),
                  sampling_contract=sampling_contract(config.training), launch="shared leased worker only")
    if args.verify_parent:
        parent, lineage, hashes = load_parent(config, args.config)
        report.update(initialization=lineage, manifest_hashes=hashes, parent_tensor_count=len(parent["trainable_model"]))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
