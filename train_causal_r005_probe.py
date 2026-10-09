"""Bounded R005 -> native97 causal TF, FP32 LoRA, BF16 compute.

The new stage reuses the verified 49->97 weights-only migration. It does not
change the legacy teacher-lineage checks, perform BID/CD/DMD, or train from
generated history. Launch is function-only through the shared leased worker.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import fields
import json
from pathlib import Path
import random
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch

import train_action_teacher as action_entry
import train_causal_teacher_forcing as tf_entry
from training.causal_tf import artifact_hashes, causal_teacher_forcing_loss, load_causal_config
from training.config import ActionTeacherConfig, ModelConfig, DataConfig, OptimizerConfig, TrainingConfig
from training.models.lora import load_trainable_state_dict, trainable_state_dict
from training.runtime import CheckpointManager, append_jsonl, load_checkpoint, seed_everything

STAGE_NAME = "causal_r005_native97_fp32_v1"
FACTORY = "training.data.action_resampled:build_resampled_action_teacher_dataloader"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_probe_config(config):
    config.validate()
    _require(config.data.num_frames == 97 and config.data.data_factory == FACTORY
             and config.data.prompt_cache_path is None, "probe requires verified native97 inline-static resampled data")
    _require(not config.error_recycling.enabled, "this stage requires clean completed-block teacher forcing")
    _require(config.lineage.checkpoint_sha256 == action_entry.FP32_LORA_PARENT_SHA,
             "this stage requires the original SHA-pinned R005 step1040")
    _require(config.model.action_scale == 0.03 and config.model.lora_dropout == 0
             and config.model.gradient_checkpointing, "probe requires R005 action scale and gradient checkpointing")
    _require(config.optimizer.adapter_lr == 0 and config.optimizer.lora_lr == 2e-6,
             "probe fixes adapter_lr=0 and LoRA LR2e-6")
    _require(config.training.max_steps == 20 and config.training.checkpoint_every == 20,
             "bounded probe trajectory has max_steps20 and checkpoint_every20")


def method_contract():
    return dict(method=STAGE_NAME, objective="causal_completed_clean_blocks_flow_matching",
        latent_frames=25, independent_first_frame=True, future_latent_frames_per_block=3,
        completed_history="ground_truth_clean_latents_at_t0", kv_inference_history="generated_latents_at_t0",
        exposure_gap_closed=False, fp32_lora=True, action_adapter_lr=0.0,
        precision_transition=action_entry.fp32_lora_precision_transition(),
        packed_teacher_forcing=True, bidirectional_auxiliary=False,
        consistency_distillation=False, distribution_matching_distillation=False,
        few_step_quality_validated=False, quality_validated=False)


def load_parent(config, config_path):
    """Validate the existing cache transition, then declare the causal change.

The intermediate configuration is validation-only: no action-model training,
fake checkpoint, optimizer state, or GT-prefix sampling is introduced.
"""
    validate_probe_config(config)
    hashes = artifact_hashes(config, config_path)
    source = torch.load(config.lineage.checkpoint_path, map_location="cpu", weights_only=False)
    _require(isinstance(source, dict) and isinstance(source.get("config"), dict), "invalid R005 source payload")
    raw = copy.deepcopy(source["config"])
    raw["model"]["base_model_path"] = config.model.base_model_path
    raw["data"].update(num_frames=97, manifest_path=config.data.manifest_path,
        manifest_sha256=config.data.manifest_sha256, prompt_cache_path=None, data_factory=FACTORY)
    raw["optimizer"]["adapter_lr"] = 0.0
    raw["training"].update(max_steps=config.training.max_steps, output_dir=config.training.output_dir)
    migration = ActionTeacherConfig(model=ModelConfig(**raw["model"]), data=DataConfig(**raw["data"]),
        optimizer=OptimizerConfig(**raw["optimizer"]), training=TrainingConfig(**raw["training"]))
    migration.validate()
    # No undocumented optimizer, shared model, worker or training-field drift.
    for name in (item.name for item in fields(ModelConfig)):
        _require(getattr(migration.model, name) == getattr(config.model, name), f"R005 inherited model field changed: {name}")
    for name in (item.name for item in fields(OptimizerConfig)):
        _require(getattr(migration.optimizer, name) == getattr(config.optimizer, name), f"R005 inherited optimizer field changed: {name}")
    for name in (item.name for item in fields(TrainingConfig) if item.name != "lora_parameter_dtype"):
        _require(getattr(migration.training, name) == getattr(config.training, name), f"R005 inherited training field changed: {name}")
    _require(migration.data.num_workers == config.data.num_workers, "R005 worker configuration changed")
    migration_hashes = {key: hashes[key] for key in
        ("dataset_manifest", "feature_index", "feature_receipt", "training_config")}
    del source
    state, lineage = action_entry._load_length_transition_checkpoint(
        config.lineage.checkpoint_path, expected_sha256=config.lineage.checkpoint_sha256,
        config=migration, current_manifest_hashes=migration_hashes,
        base_model_path_relocated=True, resampled=True, freeze_action_adapter=True)
    action_entry._validate_lora_state_dtype(state, "bfloat16")
    _require(all(bool(torch.isfinite(value).all()) for value in state.values()), "nonfinite R005 parent state")
    lineage.update(mode="r005_native97_causal_fp32_weights_only",
        cache_transition_validation="existing_length_resampled_transition_helper",
        attention_transition=dict(source="bidirectional_action_teacher", target="block_causal_teacher_forcing",
                                  independent_first_frame=True, future_latent_frames_per_block=3),
        precision_transition=action_entry.fp32_lora_precision_transition())
    return {"trainable_model": state}, lineage, hashes


def validate_resume(path, *, config, hashes, lineage, saved_tensors_on_cpu):
    payload = load_checkpoint(path, expected_manifest_hashes=hashes, map_location="cpu")
    _require(payload.get("format_version") == 1 and payload.get("stage") == STAGE_NAME, "resume requires this causal FP32 stage")
    _require(payload.get("config") == config.to_dict() and payload.get("initialization") == lineage
             and payload.get("method_contract") == method_contract(), "strict causal resume contract changed")
    _require(payload.get("saved_tensors_on_cpu") is saved_tensors_on_cpu, "strict resume saved-tensor policy changed")
    step, micro = payload.get("step"), payload.get("micro_batches_consumed")
    _require(type(step) is int and 0 < step <= 20 and type(micro) is int and micro == step * 8,
             "resume optimizer/data cursor is inconsistent")
    _require(all(name in payload for name in ("optimizer", "torch_rng_state", "cuda_rng_state_all",
                                              "python_rng_state", "numpy_rng_state")), "resume lacks optimizer/RNG state")
    action_entry._validate_lora_state_dtype(payload.get("trainable_model"), "float32")
    return payload


def _update_evidence(model, before):
    """Measure real changes over every LoRA tensor, and zero adapter changes."""
    after = trainable_state_dict(model)
    changed, elements, squared_delta = 0, 0, 0.0
    for name, value in after.items():
        _require(bool(torch.isfinite(value).all()), f"nonfinite updated parameter: {name}")
        if "act_control_adapter." in name:
            _require(torch.equal(value, before[name]), "zero-LR action adapter unexpectedly changed")
            continue
        difference = value.float() - before[name].float()
        changed += int(torch.count_nonzero(difference))
        elements += difference.numel()
        squared_delta += float(difference.double().square().sum())
    _require(changed > 0, "optimizer produced no representable LoRA update")
    return dict(lora_changed_elements=changed, lora_elements=elements,
                lora_update_l2=squared_delta ** 0.5, action_adapter_unchanged=True)


def launch(config, config_path, *, shared_lease, shared_max_seconds,
           shared_request_sha256, source_revision, stop_after_step=2, resume=None,
           saved_tensors_on_cpu=True):
    validate_probe_config(config)
    _require(type(stop_after_step) is int and 1 <= stop_after_step <= 20, "execution cap must be1..20")
    _require(type(saved_tensors_on_cpu) is bool, "saved_tensors_on_cpu must be an explicit boolean")
    _require(isinstance(source_revision, str) and bool(source_revision.strip()), "deployed source revision required")
    output = Path(config.training.output_dir)
    _require(resume is not None or not output.exists() or not any(output.iterdir()), "fresh output must be empty")
    if resume:
        _require(output.resolve() in Path(resume).resolve().parents, "resume checkpoint is outside this run")
    parent, lineage, hashes = load_parent(config, config_path)
    restored = validate_resume(resume, config=config, hashes=hashes, lineage=lineage,
                               saved_tensors_on_cpu=saved_tensors_on_cpu) if resume else None
    start = restored["step"] if restored else 0
    _require(start < stop_after_step, "execution cap must exceed restored step")
    gpu = action_entry._shared_training_gpu(shared_lease, max_seconds=shared_max_seconds,
                                            request_sha256=shared_request_sha256)
    _require(torch.cuda.is_available(), "leased launch requires CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(config.training.seed)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(stage=STAGE_NAME, config=config.to_dict(), manifest_hashes=hashes,
        initialization=lineage, method_contract=method_contract(), source_revision=source_revision,
        command=sys.argv, gpu_state=gpu.as_dict(), execution_stop_after_step=stop_after_step,
        resume=resume, saved_tensors_on_cpu=saved_tensors_on_cpu, started_unix=time.time())
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        model, summary = tf_entry._build_model(config, parent, device)
        del parent
        action_entry._apply_lora_parameter_dtype(model, "float32")
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
            losses = []
            for _ in range(8):
                batch = next(iterator)
                context = torch.autograd.graph.save_on_cpu(pin_memory=True) if saved_tensors_on_cpu else nullcontext()
                with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = causal_teacher_forcing_loss(model, batch, config, device)
                _require(bool(torch.isfinite(loss)), "nonfinite causal teacher-forcing loss")
                (loss / 8).backward()
                losses.append(float(loss.detach()))
                del loss
                micro += 1
            precision = action_entry._lora_gradient_precision(model)
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.optimizer.max_grad_norm)
            _require(bool(torch.isfinite(norm)), "nonfinite causal gradient norm")
            optimizer.step()
            evidence = _update_evidence(model, before)
            del before
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            latest = dict(loss=sum(losses) / 8, grad_norm=float(norm), precision=precision,
                **evidence, optimizer_step_seconds=time.monotonic() - step_began,
                peak_vram_bytes=int(torch.cuda.max_memory_allocated(device)),
                peak_reserved_vram_bytes=int(torch.cuda.max_memory_reserved(device)),
                samples_per_second=8 / (time.monotonic() - step_began))
            append_jsonl(output / "metrics.jsonl", dict(step=step, micro_batches_consumed=micro, **latest))
            print(json.dumps(dict(step=step, **latest)), flush=True)
            if step % 20 == 0 or step == stop_after_step:
                payload = dict(format_version=1, stage=STAGE_NAME, step=step, micro_batches_consumed=micro,
                    config=config.to_dict(), manifest_hashes=hashes, initialization=lineage,
                    method_contract=method_contract(), saved_tensors_on_cpu=saved_tensors_on_cpu,
                    trainable_model=trainable_state_dict(model), optimizer=optimizer.state_dict(), metrics=latest,
                    torch_rng_state=torch.get_rng_state(), cuda_rng_state_all=torch.cuda.get_rng_state_all(),
                    python_rng_state=random.getstate(), numpy_rng_state=np.random.get_state())
                manager.save(payload, step=step, metric=latest["loss"])
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
    parser.add_argument("--verify-parent", action="store_true", help="CPU-only hash/cache/R005 lineage verification")
    args = parser.parse_args(argv)
    config = load_causal_config(args.config)
    validate_probe_config(config)
    report = tf_entry.cpu_dry_run(config, args.config)
    report.update(stage=STAGE_NAME, method_contract=method_contract(), launch="shared leased worker only")
    if args.verify_parent:
        parent, lineage, hashes = load_parent(config, args.config)
        report.update(initialization=lineage, manifest_hashes=hashes, parent_tensor_count=len(parent["trainable_model"]))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
