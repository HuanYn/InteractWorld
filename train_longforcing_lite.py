"""Train the stage-3 LongForcing-lite causal student.

The default invocation is a CPU-only configuration report.  ``--launch`` is
accepted only with fresh dedicated-GPU confirmation and explicit long-window
data/backend factories.  This stage is endpoint distillation, not DMD.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from training.gpu_gate import query_dedicated_gpu, validate_confirmation
from training.causal_tf import _prompt_contract
from training.longforcing_lite import (
    METHOD_NAME,
    STAGE_NAME,
    LongForcingConfig,
    artifact_hashes,
    curriculum_depth,
    initialize_trainable_from_checkpoint,
    is_flowmatch_replay,
    load_longforcing_config,
    load_parent_checkpoints,
    longforcing_lite_loss,
    rgb_frames_for_blocks,
    validate_backend,
)
from training.models.lora import load_trainable_state_dict, trainable_state_dict
from training.models.action_adapter import validate_action_scale
from training.paths import project_root
from training.runtime import (
    CheckpointManager,
    ThroughputTracker,
    append_jsonl,
    git_revision,
    load_checkpoint,
    peak_vram_bytes,
    seed_everything,
    sha256_file,
)

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "train" / "longforcing_lite_v1.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="CPU-only contract validation")
    mode.add_argument("--launch", action="store_true", help="load models and train on CUDA")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--confirmed-gpu-index", type=int)
    parser.add_argument("--confirmed-gpu-uuid")
    parser.add_argument("--confirmed-at-utc")
    parser.add_argument("--allocation-profile")
    return parser.parse_args(argv)


def _apply_overrides(config: LongForcingConfig, args: argparse.Namespace) -> None:
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        config.training.max_steps = args.max_steps
    if args.output_dir is not None:
        config.training.output_dir = args.output_dir
    config.validate()


def validation_report(config: LongForcingConfig, config_path: str | Path) -> dict[str, Any]:
    paths = {
        "dataset_manifest": Path(config.data.manifest_path),
        "feature_index": Path(config.data.feature_index_path),
        "feature_receipt": Path(config.data.feature_receipt_path),
        "long_feature_index": Path(config.data.long_feature_index_path),
        "long_feature_receipt": Path(config.data.long_feature_receipt_path),
        "teacher_checkpoint": Path(config.lineage.teacher_checkpoint_path),
        "causal_checkpoint": Path(config.lineage.causal_checkpoint_path),
    }
    missing_factories = [
        name
        for name, value in (
            ("data.data_factory", config.data.data_factory),
            ("rollout.backend_factory", config.rollout.backend_factory),
        )
        if not value
    ]
    return {
        "status": "dry_run_valid",
        "mode": "cpu_dry_run",
        "stage": STAGE_NAME,
        "method": METHOD_NAME,
        "is_dmd": False,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": sha256_file(config_path),
        "git_revision": git_revision(Path(__file__).parent),
        "artifact_exists": {name: path.is_file() for name, path in paths.items()},
        "available_artifact_sha256": {
            name: sha256_file(path) for name, path in paths.items() if path.is_file()
        },
        "student_steps": config.rollout.student_steps,
        "teacher_steps": config.rollout.teacher_steps,
        "action_scale": config.model.action_scale,
        "teacher_stop_gradient": config.rollout.teacher_stop_gradient,
        "only_last_block_backward": config.rollout.only_last_block_backward,
        "flowmatch_replay_fraction": config.rollout.flowmatch_replay_fraction,
        "curriculum": [
            {
                "start_step": start,
                "depth_blocks": depth,
                "rgb_frames": rgb_frames_for_blocks(depth, config),
            }
            for start, depth in zip(
                config.rollout.curriculum_start_steps, config.rollout.curriculum_depths
            )
        ],
        "demo_duration_seconds": (config.data.demo_rollout_frames - 1) / config.data.fps,
        "launch_ready": not missing_factories,
        "launch_blockers": missing_factories,
        "cuda_queried": False,
        "weights_loaded": False,
    }


def _require_launch_factories(config: LongForcingConfig) -> tuple[str, str]:
    missing = []
    if not config.data.data_factory:
        missing.append("data.data_factory")
    if not config.rollout.backend_factory:
        missing.append("rollout.backend_factory")
    if missing:
        raise RuntimeError(
            "refusing LongForcing-lite launch before GPU inspection; explicit real "
            "interfaces are unset: " + ", ".join(missing)
        )
    return config.data.data_factory, config.rollout.backend_factory


def _import_factory(spec: str):
    if ":" not in spec:
        raise ValueError("factory must use the form 'module:function'")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, function_name)
    except AttributeError as exc:
        raise RuntimeError(f"factory {spec!r} does not exist") from exc


def _move(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        dtype = torch.bfloat16 if value.is_floating_point() else value.dtype
        return value.to(device=device, dtype=dtype, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


def _build_optimizer(model: torch.nn.Module, config: LongForcingConfig):
    adapter, lora, unexpected = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "act_control_adapter" in name:
            adapter.append(parameter)
        elif ".lora_a." in name or ".lora_b." in name:
            lora.append(parameter)
        else:
            unexpected.append(name)
    if unexpected or not adapter or not lora:
        raise RuntimeError(
            "student trainables must be non-empty LoRA/action-adapter groups only; "
            f"unexpected={unexpected[:8]}"
        )
    if config.optimizer.name != "adamw8bit":
        raise ValueError("LongForcing-lite v1 supports only adamw8bit")
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise RuntimeError("--launch requires bitsandbytes AdamW8bit") from exc
    return bnb.optim.AdamW8bit(
        [
            {"params": adapter, "lr": config.optimizer.adapter_lr, "name": "action_adapter"},
            {"params": lora, "lr": config.optimizer.lora_lr, "name": "lora"},
        ],
        betas=config.optimizer.betas,
        weight_decay=config.optimizer.weight_decay,
    )


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    config: LongForcingConfig,
    step: int,
    micro_batches_consumed: int,
    metrics: dict[str, Any],
    hashes: dict[str, str],
    teacher_lineage: dict[str, Any],
    causal_lineage: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "stage": STAGE_NAME,
        "method": METHOD_NAME,
        "is_dmd": False,
        "step": step,
        "micro_batches_consumed": micro_batches_consumed,
        "replay_phase": micro_batches_consumed % 4,
        "curriculum_depth": curriculum_depth(step, config),
        "trainable_model": trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
        "metrics": metrics,
        "manifest_hashes": hashes,
        "parent_teacher": teacher_lineage,
        "parent_causal": causal_lineage,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }


def _restore_resume(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    hashes: dict[str, str],
    teacher_lineage: dict[str, Any],
    causal_lineage: dict[str, Any],
    config: LongForcingConfig,
) -> tuple[int, int]:
    checkpoint = load_checkpoint(path, expected_manifest_hashes=hashes, map_location="cpu")
    if checkpoint.get("stage") != STAGE_NAME or checkpoint.get("is_dmd") is not False:
        raise ValueError(f"resume checkpoint is not from {STAGE_NAME}")
    if checkpoint.get("parent_teacher") != teacher_lineage:
        raise ValueError("resume teacher lineage changed")
    if checkpoint.get("parent_causal") != causal_lineage:
        raise ValueError("resume causal lineage changed")
    # Older LongForcing checkpoints omitted the formerly hard-coded scale.
    saved_scale = validate_action_scale(
        checkpoint.get("config", {}).get("model", {}).get("action_scale", 1.0)
    )
    if saved_scale != validate_action_scale(config.model.action_scale):
        raise ValueError("resume action_scale changed; use a new calibrated run")
    saved_data = checkpoint.get("config", {}).get("data", {})
    if not isinstance(saved_data, Mapping):
        raise ValueError("resume has invalid prompt configuration")
    if _prompt_contract(saved_data, checkpoint["manifest_hashes"]) != _prompt_contract(config.to_dict()["data"], hashes):
        raise ValueError("resume prompt policy/path/hash contract changed; use a new run")
    load_trainable_state_dict(model, checkpoint["trainable_model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    step = int(checkpoint["step"])
    consumed = int(checkpoint.get("micro_batches_consumed", -1))
    if consumed != step * config.training.gradient_accumulation_steps:
        raise ValueError("checkpoint micro-batch position is not an optimizer boundary")
    if checkpoint.get("replay_phase") != consumed % 4:
        raise ValueError("checkpoint replay phase is inconsistent")
    if checkpoint.get("curriculum_depth") != curriculum_depth(step, config):
        raise ValueError("checkpoint curriculum state is inconsistent")
    return step, consumed


def _iterator_at_micro_batch(loader: Iterable[Any], consumed: int):
    if consumed < 0:
        raise ValueError("consumed micro-batches cannot be negative")
    try:
        length = len(loader)  # type: ignore[arg-type]
    except TypeError:
        length = None
    skip = consumed if length is None else consumed % length
    iterator = iter(loader)
    for _ in range(skip):
        try:
            next(iterator)
        except StopIteration:
            iterator = iter(loader)
            try:
                next(iterator)
            except StopIteration as exc:
                raise RuntimeError("data factory returned an empty iterable") from exc
    return iterator


def _objective_counts_at_micro_batch(consumed: int) -> dict[str, int]:
    """Restore cumulative counters for the fixed one-in-four replay schedule."""
    if consumed < 0:
        raise ValueError("consumed micro-batches cannot be negative")
    replay = consumed // 4
    return {"longforcing": consumed - replay, "flowmatch_replay": replay}


def _prepare_resume_attempt(output_dir: Path, resume: str | Path, step: int) -> dict[str, Any]:
    """Preserve prior attempt evidence before resuming its saved optimizer step.

    The active metric series follows the checkpoint's lineage, while the full
    failed-attempt series remains byte-for-byte available in the archive.
    Checkpoint files, including the historical loss-selected best, are untouched.
    """
    output_dir = output_dir.resolve()
    checkpoint = Path(resume).resolve()
    if project_root().resolve() not in output_dir.parents:
        raise ValueError("resume evidence must stay below the configured project root")
    if checkpoint.parent != output_dir / "checkpoints" or not checkpoint.is_file():
        raise ValueError("resume checkpoint must be a file in this run's checkpoints directory")
    metadata_path = output_dir / "run_metadata.json"
    metrics_path = output_dir / "metrics.jsonl"
    attempts = output_dir / "attempts"
    if step < 0 or not metadata_path.is_file():
        raise ValueError("resume requires a nonnegative step and existing run_metadata.json")
    if any(path.is_symlink() for path in (metadata_path, metrics_path, attempts)):
        raise ValueError("refusing symlinked resume evidence paths")

    retained: list[bytes] = []
    removed_steps: list[int] = []
    previous = 0
    if metrics_path.exists():
        for line in metrics_path.read_bytes().splitlines(keepends=True):
            if not line.strip():
                continue
            record = json.loads(line)
            record_step = record.get("step")
            if type(record_step) is not int or record_step <= previous:
                raise ValueError("resume metrics must have strictly increasing positive steps")
            previous = record_step
            if record_step <= step:
                retained.append(line)
            else:
                removed_steps.append(record_step)

    stamp = time.time_ns()
    archive = attempts / f"before-resume-step-{step:07d}-{stamp}"
    archive.mkdir(parents=True, exist_ok=False)
    archived_files = {}
    for path in (metadata_path, metrics_path):
        if path.is_file():
            target = archive / path.name
            shutil.copy2(path, target)
            archived_files[path.name] = {"path": str(target), "sha256": sha256_file(target)}
    receipt = {
        "schema_version": 1,
        "resume_checkpoint": str(checkpoint),
        "resume_step": step,
        "archive_dir": str(archive),
        "archived_files": archived_files,
        "steps_removed_from_active_metrics": removed_steps,
        "checkpoint_files_modified": False,
        "created_unix": time.time(),
    }
    (archive / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    if removed_steps:
        temporary = output_dir / f"metrics.resume-{stamp}.tmp"
        with temporary.open("xb") as stream:
            stream.write(b"".join(retained))
        temporary.replace(metrics_path)
    return receipt


def launch(
    config: LongForcingConfig,
    config_path: str | Path,
    resume: str | None,
    *,
    confirmed_gpu_index: int,
    confirmed_gpu_uuid: str,
    confirmed_at_utc: str,
    allocation_profile: str,
) -> None:
    data_spec, backend_spec = _require_launch_factories(config)
    hashes = artifact_hashes(config, config_path)
    teacher_payload, teacher_lineage, causal_payload, causal_lineage = load_parent_checkpoints(
        config, hashes
    )
    validate_confirmation(confirmed_at_utc)
    gpu_snapshot = query_dedicated_gpu(
        confirmed_index=confirmed_gpu_index,
        confirmed_uuid=confirmed_gpu_uuid,
        profile=allocation_profile,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("--launch was requested but CUDA is unavailable")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(config.training.seed)

    output_dir = Path(config.training.output_dir)
    if resume is None and output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing a fresh launch into non-empty directory: {output_dir}")
    if resume is not None and output_dir.resolve() not in Path(resume).resolve().parents:
        raise RuntimeError("resume checkpoint must belong to the configured output directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = _import_factory(backend_spec)(config=config, device=device)
    validate_backend(backend)
    student_summary = initialize_trainable_from_checkpoint(
        backend.student, causal_payload, config, frozen_after_load=False
    )
    initialize_trainable_from_checkpoint(
        backend.teacher, teacher_payload, config, frozen_after_load=True
    )
    if config.model.gradient_checkpointing:
        enable = getattr(backend.student, "enable_gradient_checkpointing", None)
        if not callable(enable):
            raise RuntimeError("student wrapper cannot enable required gradient checkpointing")
        enable()
    backend.student.to(device=device, dtype=torch.bfloat16).train()
    if config.rollout.teacher_cpu_offload:
        backend.teacher.to(device=torch.device("cpu"), dtype=torch.bfloat16).eval()
    else:  # guarded by the v1 config validator
        backend.teacher.to(device=device, dtype=torch.bfloat16).eval()
    optimizer = _build_optimizer(backend.student, config)
    loader = _import_factory(data_spec)(config=config.data, training=config.training)

    teacher_dict, causal_dict = teacher_lineage.as_dict(), causal_lineage.as_dict()
    step = consumed = 0
    resume_attempt = None
    if resume:
        step, consumed = _restore_resume(
            resume,
            model=backend.student,
            optimizer=optimizer,
            hashes=hashes,
            teacher_lineage=teacher_dict,
            causal_lineage=causal_dict,
            config=config,
        )
        resume_attempt = _prepare_resume_attempt(output_dir, resume, step)
    metadata = {
        "command": sys.argv,
        "stage": STAGE_NAME,
        "method": METHOD_NAME,
        "is_dmd": False,
        "git_revision": git_revision(Path(__file__).parent),
        "config": config.to_dict(),
        "manifest_hashes": hashes,
        "parent_teacher": teacher_dict,
        "parent_causal": causal_dict,
        "gpu_state": gpu_snapshot.as_dict(),
        "started_unix": time.time(),
        "resume_attempt": resume_attempt,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "launching",
                "stage": STAGE_NAME,
                "trainable_parameters": student_summary.trainable_parameters,
                "resume_step": step,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    manager = CheckpointManager(output_dir, keep_last=config.training.keep_last)
    iterator = _iterator_at_micro_batch(loader, consumed)
    tracker = ThroughputTracker.start()
    backend.student.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    objective_counts = _objective_counts_at_micro_batch(consumed)
    last_saved_step = step
    latest_metrics: dict[str, Any] = {"loss": float("inf")}
    while step < config.training.max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise RuntimeError("data factory returned an empty iterable") from exc
        batch = _move(batch, device)
        depth = curriculum_depth(step, config)
        replay = is_flowmatch_replay(consumed)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if replay:
                short_window = batch.get("short_window") if isinstance(batch, Mapping) else None
                if not isinstance(short_window, Mapping):
                    raise ValueError("replay micro-batch requires a short_window mapping")
                loss = backend.flowmatch_replay_loss(short_window)
                details = {"objective": "flowmatch_replay", "depth_blocks": depth}
                objective_counts["flowmatch_replay"] += 1
            else:
                loss, details = longforcing_lite_loss(backend, batch, config, depth=depth)
                objective_counts["longforcing"] += 1
        if not torch.is_tensor(loss) or loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError("LongForcing-lite objective returned a non-finite scalar")
        (loss / config.training.gradient_accumulation_steps).backward()
        accumulated_loss += float(loss.detach())
        consumed += 1
        tracker.update(config.training.micro_batch_size)
        if consumed % config.training.gradient_accumulation_steps:
            continue

        trainable = [p for p in backend.student.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, config.optimizer.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        mean_loss = accumulated_loss / config.training.gradient_accumulation_steps
        accumulated_loss = 0.0
        latest_metrics = {
            "loss": mean_loss,
            "grad_norm": float(grad_norm),
            "samples_per_second": tracker.samples_per_second,
            "peak_vram_bytes": peak_vram_bytes(device),
            "depth_blocks": depth,
            "last_objective": details["objective"],
            **objective_counts,
        }
        if step % config.training.log_every == 0:
            record = {"step": step, **latest_metrics, "unix": time.time()}
            append_jsonl(output_dir / "metrics.jsonl", record)
            print(json.dumps(record, sort_keys=True), flush=True)
        if step % config.training.checkpoint_every == 0:
            manager.save(
                _checkpoint_payload(
                    backend.student,
                    optimizer,
                    config=config,
                    step=step,
                    micro_batches_consumed=consumed,
                    metrics=latest_metrics,
                    hashes=hashes,
                    teacher_lineage=teacher_dict,
                    causal_lineage=causal_dict,
                ),
                step=step,
                metric=mean_loss,
            )
            last_saved_step = step

    if step and last_saved_step != step:
        manager.save(
            _checkpoint_payload(
                backend.student,
                optimizer,
                config=config,
                step=step,
                micro_batches_consumed=consumed,
                metrics=latest_metrics,
                hashes=hashes,
                teacher_lineage=teacher_dict,
                causal_lineage=causal_dict,
            ),
            step=step,
            metric=float(latest_metrics["loss"]),
        )
    print(
        json.dumps(
            {"status": "complete", "stage": STAGE_NAME, "step": step, "output_dir": str(output_dir)},
            sort_keys=True,
        ),
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_longforcing_config(args.config)
    _apply_overrides(config, args)
    if not args.launch:
        print(json.dumps(validation_report(config, args.config), indent=2, sort_keys=True))
        return 0
    required = {
        "--confirmed-gpu-index": args.confirmed_gpu_index,
        "--confirmed-gpu-uuid": args.confirmed_gpu_uuid,
        "--confirmed-at-utc": args.confirmed_at_utc,
        "--allocation-profile": args.allocation_profile,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit("--launch requires fresh GPU confirmation fields: " + ", ".join(missing))
    launch(
        config,
        args.config,
        args.resume,
        confirmed_gpu_index=args.confirmed_gpu_index,
        confirmed_gpu_uuid=args.confirmed_gpu_uuid,
        confirmed_at_utc=args.confirmed_at_utc,
        allocation_profile=args.allocation_profile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
