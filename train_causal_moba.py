"""Independent MoBA-inspired sequential regularization stage; CPU plan by default.

Not official packed MoBA, CD, DMD, or a real-time/long-horizon quality claim.
CUDA requires --launch and the inherited fresh dedicated-GPU authorization gate.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

import train_causal_teacher_forcing as tf_entry
from training.causal_tf import artifact_hashes, load_teacher_checkpoint
from training.causal_moba import (
    STAGE_NAME, CausalMoBAConfig, backward_microbatch, load_moba_config,
    method_contract, verify_bidirectional_interface,
)
from training.gpu_gate import query_dedicated_gpu, validate_confirmation
from training.models.lora import load_trainable_state_dict
from training.runtime import (
    CheckpointManager, ThroughputTracker, append_jsonl, git_revision,
    load_checkpoint, peak_vram_bytes, seed_everything, sha256_file,
)

DEFAULT_CONFIG = Path(__file__).parent / "configs/train/causal_moba_example.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="CPU-only configuration and TF mask check")
    mode.add_argument("--launch", action="store_true", help="explicitly permit CUDA training after the fresh GPU gate")
    parser.add_argument("--resume", help=f"strict {STAGE_NAME} checkpoint only")
    parser.add_argument("--allow-horizon-extension", action="store_true",
                        help="opt in to resume with only max_steps increased; keep the original YAML")
    parser.add_argument("--max-steps", type=int, help="explicit optimizer-step cap")
    parser.add_argument("--output-dir", help="explicit run directory override")
    parser.add_argument("--confirmed-gpu-index", type=int)
    parser.add_argument("--confirmed-gpu-uuid")
    parser.add_argument("--confirmed-at-utc")
    parser.add_argument("--allocation-profile")
    return parser.parse_args(argv)


def _sampling_contract(config: CausalMoBAConfig) -> dict[str, Any]:
    if config.data.data_factory == "training.data.action_resampled:build_resampled_action_teacher_dataloader":
        from training.data.action_resampled import sampling_contract
        return {"factory": config.data.data_factory, **sampling_contract(config.training)}
    return {"factory": config.data.data_factory, "training_seed": config.training.seed}


def validation_report(config: CausalMoBAConfig, config_path: str | Path, *, dry_run=False):
    config.validate()
    report = (tf_entry.cpu_dry_run if dry_run else tf_entry.validation_report)(config, config_path)
    report.update(stage=STAGE_NAME, method_contract=method_contract(config),
                  sampling_contract=_sampling_contract(config),
                  config=config.to_dict(), gpu_method_validated=False,
                  initialization_mode="action_teacher_weights_only_fresh_optimizer_rng_step0")
    return report


def _checkpoint_payload(model, optimizer, *, config: CausalMoBAConfig, continuations=None, **kwargs):
    payload = tf_entry._checkpoint_payload(model, optimizer, config=config, **kwargs)
    payload.update(stage=STAGE_NAME, method_contract=method_contract(config),
                   sampling_contract=_sampling_contract(config),
                   initialization_mode="action_teacher_weights_only_fresh_optimizer_rng_step0")
    if continuations is not None:
        payload["continuations"] = copy.deepcopy(continuations)
    return payload


def _canonical_config(config):
    return json.loads(json.dumps(config))


def validate_resume_checkpoint(path, *, config: CausalMoBAConfig,
                               hashes: dict[str, str], teacher_lineage: dict[str, Any],
                               allow_horizon_extension: bool = False):
    """CPU-only resume validation; never relabel or rewrite the source payload.

    The opt-in permits exactly one configuration difference: a larger max_steps.
    Original source-YAML/artifact hashes remain exact. Resampled data is required
    because its absolute sample indices do not depend on the configured horizon.
    Returns the loaded payload and an extension record (None for strict resume).
    """
    checkpoint = load_checkpoint(path, expected_manifest_hashes=hashes, map_location="cpu")
    if checkpoint.get("format_version") != 1 or checkpoint.get("stage") != STAGE_NAME:
        raise ValueError(f"resume requires an exact {STAGE_NAME} checkpoint")
    if checkpoint.get("method_contract") != method_contract(config):
        raise ValueError("resume method/auxiliary-weight contract changed")
    if checkpoint.get("parent_teacher") != teacher_lineage:
        raise ValueError("resume parent-teacher lineage changed")
    saved = checkpoint.get("config")
    if not isinstance(saved, dict) or not isinstance(saved.get("training"), dict):
        raise ValueError("resume serialized configuration changed")
    source_max = saved["training"].get("max_steps")
    if isinstance(source_max, bool) or not isinstance(source_max, int) or source_max <= 0:
        raise ValueError("resume source max_steps is invalid")
    expected_source = copy.deepcopy(config)
    if allow_horizon_extension:
        if config.data.data_factory != "training.data.action_resampled:build_resampled_action_teacher_dataloader":
            raise ValueError("horizon extension requires absolute-index resampled data")
        if config.training.max_steps <= source_max:
            raise ValueError("horizon extension must strictly increase max_steps")
        expected_source.training.max_steps = source_max
    if _canonical_config(saved) != _canonical_config(expected_source.to_dict()):
        raise ValueError("resume serialized configuration changed")
    if checkpoint.get("sampling_contract") != _sampling_contract(expected_source):
        raise ValueError("resume sampling contract changed")
    if checkpoint.get("initialization_mode") != "action_teacher_weights_only_fresh_optimizer_rng_step0":
        raise ValueError("resume initialization mode changed")
    step = checkpoint.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= source_max:
        raise ValueError("resume optimizer step is invalid")
    if allow_horizon_extension and step == 0:
        raise ValueError("horizon extension requires a trained checkpoint, not a fresh restart")
    micro = checkpoint.get("micro_batches_consumed")
    if isinstance(micro, bool) or not isinstance(micro, int) or micro != step * config.training.gradient_accumulation_steps:
        raise ValueError("resume micro-batch position is inconsistent")
    for key in ("trainable_model", "optimizer", "torch_rng_state", "cuda_rng_state_all"):
        if key not in checkpoint:
            raise ValueError(f"resume checkpoint missing {key}")
    if not isinstance(checkpoint.get("continuations", []), list):
        raise ValueError("resume continuation history must be a list")
    record = None
    if allow_horizon_extension:
        record = {
            "version": 1, "mode": "max_steps_only_horizon_extension_v1",
            "source_checkpoint": str(Path(path).resolve()), "source_checkpoint_sha256": sha256_file(path),
            "source_step": step, "micro_batches_consumed": micro,
            "next_absolute_sample_index": micro * config.training.micro_batch_size,
            "source_max_steps": source_max, "target_max_steps": config.training.max_steps,
            "source_config": copy.deepcopy(saved), "target_config": config.to_dict(),
            "source_sampling_contract": copy.deepcopy(checkpoint["sampling_contract"]),
            "target_sampling_contract": _sampling_contract(config),
            "manifest_hashes": dict(hashes), "method_contract": method_contract(config),
            "parent_teacher": copy.deepcopy(teacher_lineage),
            "restore_policy": {"optimizer": "checkpoint", "torch_rng": "checkpoint",
                               "cuda_rng": "checkpoint", "micro_batches": "checkpoint", "fresh_restart": False},
        }
    return checkpoint, record


def _restore_validated_state(checkpoint, model, optimizer):
    load_trainable_state_dict(model, checkpoint["trainable_model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    return checkpoint["step"], checkpoint["micro_batches_consumed"]


def _restore_resume(path, *, config: CausalMoBAConfig, model, optimizer,
                    hashes: dict[str, str], teacher_lineage: dict[str, Any],
                    allow_horizon_extension: bool = False):
    checkpoint, _ = validate_resume_checkpoint(
        path, config=config, hashes=hashes, teacher_lineage=teacher_lineage,
        allow_horizon_extension=allow_horizon_extension,
    )
    return _restore_validated_state(checkpoint, model, optimizer)


def _resume_iterator(loader, micro):
    # DataLoader iterator construction/skip consumes a CPU base_seed. Preserve
    # restored training RNG while rebuilding the stateless absolute data cursor.
    with torch.random.fork_rng(devices=[]):
        return tf_entry._iterator_at_micro_batch(loader, micro)


def launch(config: CausalMoBAConfig, config_path: str | Path, resume: str | None, *,
           confirmed_gpu_index: int, confirmed_gpu_uuid: str,
           confirmed_at_utc: str, allocation_profile: str,
           allow_horizon_extension: bool = False) -> None:
    config.validate()
    if allow_horizon_extension and resume is None:
        raise ValueError("horizon extension requires --resume; no fresh restart")
    output = Path(config.training.output_dir)
    if resume is None and output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing fresh launch into non-empty output directory: {output}")
    if resume is not None and output.resolve() not in Path(resume).resolve().parents:
        raise RuntimeError("resume checkpoint must belong to configured output directory")
    hashes = artifact_hashes(config, config_path)
    teacher_payload, lineage = load_teacher_checkpoint(config, hashes)
    resume_payload, extension = None, None
    continuations = []
    if resume is not None:
        # Reject invalid resume/extension before GPU inventory or model loading.
        resume_payload, extension = validate_resume_checkpoint(
            resume, config=config, hashes=hashes, teacher_lineage=lineage.as_dict(),
            allow_horizon_extension=allow_horizon_extension,
        )
        continuations = copy.deepcopy(resume_payload.get("continuations", []))
        if extension is not None:
            continuations.append(extension)
    validate_confirmation(confirmed_at_utc)
    snapshot = query_dedicated_gpu(confirmed_index=confirmed_gpu_index,
                                 confirmed_uuid=confirmed_gpu_uuid, profile=allocation_profile)
    if not torch.cuda.is_available():
        raise RuntimeError("--launch requested but CUDA is unavailable")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(config.training.seed)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "command": sys.argv, "stage": STAGE_NAME,
        "git_revision": git_revision(Path(__file__).parent),
        "config": config.to_dict(), "manifest_hashes": hashes,
        "parent_teacher": lineage.as_dict(), "seed": config.training.seed,
        "method_contract": method_contract(config), "sampling_contract": _sampling_contract(config),
        "initialization_mode": "action_teacher_weights_only_fresh_optimizer_rng_step0",
        "execution_mode": "horizon_extension_resume" if extension else ("strict_resume" if resume else "fresh_stage"),
        "continuations": continuations,
        "resume_path": resume, "device": torch.cuda.get_device_name(device),
        "gpu_state": snapshot.as_dict(), "started_unix": time.time(),
        "conditioning_modules": "precomputed; no VAE/T5 or second teacher model loaded",
    }
    # Preserve fresh-run metadata on resume; append a separate execution record.
    append_jsonl(output / "executions.jsonl", metadata)
    if resume is None:
        (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        model, summary = tf_entry._build_model(config, teacher_payload, device)
        del teacher_payload
        if config.regularization.bidirectional_weight > 0:
            verify_bidirectional_interface(model)
        optimizer = tf_entry._build_optimizer(model, config)
        loader = tf_entry._import_factory(config.data.data_factory)(config=config.data, training=config.training)
        step, micro = 0, 0
        if resume:
            step, micro = _restore_validated_state(resume_payload, model, optimizer)
            del resume_payload
        manager = CheckpointManager(output, keep_last=config.training.keep_last)
        print(json.dumps({"status": "launching", "stage": STAGE_NAME,
                          "method_contract": method_contract(config), "resume_step": step,
                          "resume_micro_batches": micro,
                          "next_absolute_sample_index": micro * config.training.micro_batch_size,
                          "execution_mode": metadata["execution_mode"],
                          "teacher_step": lineage.step, "trainable_parameters": summary.trainable_parameters}), flush=True)
        model.zero_grad(set_to_none=True)
        _, iterator = (_resume_iterator(loader, micro) if resume
                       else tf_entry._iterator_at_micro_batch(loader, micro))
        tracker = ThroughputTracker.start()
        accumulated = {key: 0.0 for key in ("loss", "loss_tf", "loss_bid", "loss_total")}
        latest: dict[str, Any] = {"loss": float("inf")}
        last_saved = step
        while step < config.training.max_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                try:
                    batch = next(iterator)
                except StopIteration as exc:
                    raise RuntimeError("data factory returned an empty iterable") from exc
            values = backward_microbatch(model, batch, config, device)
            for key in accumulated:
                accumulated[key] += float(values[key])
            micro += 1
            tracker.update(config.training.micro_batch_size)
            if micro % config.training.gradient_accumulation_steps:
                continue
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            norm = torch.nn.utils.clip_grad_norm_(trainable, config.optimizer.max_grad_norm,
                                                  error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            latest = {key: value / config.training.gradient_accumulation_steps
                      for key, value in accumulated.items()}
            accumulated = {key: 0.0 for key in accumulated}
            latest.update(grad_norm=float(norm), samples_per_second=tracker.samples_per_second,
                          peak_vram_bytes=peak_vram_bytes(device),
                          bid_evaluated=config.regularization.bidirectional_weight > 0,
                          bidirectional_weight=config.regularization.bidirectional_weight)
            if step % config.training.log_every == 0:
                record = {"step": step, **latest, "unix": time.time()}
                append_jsonl(output / "metrics.jsonl", record)
                print(json.dumps(record, sort_keys=True), flush=True)
            if step % config.training.checkpoint_every == 0:
                manager.save(_checkpoint_payload(
                    model, optimizer, config=config, step=step, micro_batches_consumed=micro,
                    metrics=latest, manifest_hashes=hashes, teacher_lineage=lineage.as_dict(),
                    continuations=continuations,
                ), step=step, metric=latest["loss"])
                last_saved = step
        if step > 0 and last_saved != step:
            manager.save(_checkpoint_payload(
                model, optimizer, config=config, step=step, micro_batches_consumed=micro,
                metrics=latest, manifest_hashes=hashes, teacher_lineage=lineage.as_dict(),
                continuations=continuations,
            ), step=step, metric=latest["loss"])
        print(json.dumps({"status": "complete", "stage": STAGE_NAME, "step": step,
                          "output_dir": str(output.resolve()), "peak_vram_bytes": peak_vram_bytes(device)}), flush=True)
    except Exception as exc:
        append_jsonl(output / "failures.jsonl", {
            "stage": STAGE_NAME, "failure_type": type(exc).__name__, "error": str(exc),
            "unix": time.time(), "peak_vram_bytes": peak_vram_bytes(device),
        })
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.allow_horizon_extension and (not args.resume or args.max_steps is None):
        raise SystemExit("--allow-horizon-extension requires --resume and --max-steps; keep the original YAML")
    config = load_moba_config(args.config)
    tf_entry._apply_overrides(config, args)
    if not args.launch:
        print(json.dumps(validation_report(config, args.config, dry_run=args.dry_run), indent=2, sort_keys=True))
        return 0
    required = {"confirmed_gpu_index": args.confirmed_gpu_index,
                "confirmed_gpu_uuid": args.confirmed_gpu_uuid,
                "confirmed_at_utc": args.confirmed_at_utc, "allocation_profile": args.allocation_profile}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit("--launch requires fresh GPU confirmation fields: " + ", ".join(missing))
    launch(config, args.config, args.resume, allow_horizon_extension=args.allow_horizon_extension, **required)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
