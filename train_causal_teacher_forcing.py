"""Stage-2 block-causal teacher-forcing training for ABot-style control.

Without ``--launch`` this entry point performs CPU-only validation.  It does
not import the Wan wrapper, inspect CUDA, load model weights, or download any
artifact.  A launch requires the same fresh dedicated-GPU confirmation gate as
stage 1.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
import numpy as np

from training.causal_tf import (
    STAGE_NAME,
    CausalTeacherForcingConfig,
    artifact_hashes,
    assert_no_future_leakage,
    causal_teacher_forcing_loss,
    construct_causal_wrapper,
    frame_blocks,
    initialize_from_action_teacher,
    latent_shape,
    load_causal_config,
    load_teacher_checkpoint,
    teacher_forcing_visibility,
    stage_name,
    error_recycling_contract,
)
from training.error_recycling import ContextErrorRecycling
from training.gpu_gate import query_dedicated_gpu, validate_confirmation
from training.models.lora import load_trainable_state_dict, trainable_state_dict
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

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "train" / "causal_teacher_forcing_v1.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="CPU-only contract validation; does not construct the Wan model",
    )
    mode.add_argument(
        "--launch",
        action="store_true",
        help="explicitly allow CUDA model loading and training",
    )
    parser.add_argument("--resume", default=None, help="stage-2 resumable checkpoint path")
    parser.add_argument("--max-steps", type=int, default=None, help="explicit optimizer-step cap")
    parser.add_argument("--stop-after-step", type=int, default=None,
                        help="absolute execution segment stop; leaves the configured max_steps and hashes unchanged")
    parser.add_argument("--output-dir", default=None, help="explicit run directory override")
    parser.add_argument("--confirmed-gpu-index", type=int)
    parser.add_argument("--confirmed-gpu-uuid")
    parser.add_argument("--confirmed-at-utc")
    parser.add_argument("--allocation-profile")
    return parser.parse_args(argv)


def _apply_overrides(config: CausalTeacherForcingConfig, args: argparse.Namespace) -> None:
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        config.training.max_steps = args.max_steps
    if args.output_dir is not None:
        config.training.output_dir = args.output_dir
    stop = getattr(args, "stop_after_step", None)
    if stop is not None and not 1 <= stop <= config.training.max_steps:
        raise ValueError("--stop-after-step must be between 1 and configured max_steps")
    config.validate()


def validation_report(
    config: CausalTeacherForcingConfig,
    config_path: str | Path,
) -> dict[str, Any]:
    paths = {
        "dataset_manifest": Path(config.data.manifest_path),
        "feature_index": Path(config.data.feature_index_path),
        "feature_receipt": Path(config.data.feature_receipt_path),
        "teacher_checkpoint": Path(config.lineage.checkpoint_path),
    }
    report: dict[str, Any] = {
        "status": "configuration_valid",
        "mode": "cpu_validate",
        "stage": stage_name(config),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": sha256_file(config_path),
        "git_revision": git_revision(Path(__file__).parent),
        "artifact_exists": {name: path.is_file() for name, path in paths.items()},
        "latent_shape_bfchw": list(latent_shape(config)),
        "latent_blocks": [list(block) for block in frame_blocks(
            latent_shape(config)[1],
            frames_per_block=config.model.num_frame_per_block,
            independent_first_frame=config.model.independent_first_frame,
        )],
        "teacher_checkpoint_path": config.lineage.checkpoint_path,
        "teacher_checkpoint_expected_sha256": config.lineage.checkpoint_sha256,
        "causal_model": "CausalWanModel",
        "clean_x": True,
        "aug_t": config.model.teacher_aug_t,
        "cuda_queried": False,
        "weights_loaded": False,
    }
    # Hash only artifacts that already exist.  Missing stage-1 output is normal
    # while preparing stage 2 and becomes a hard failure only on --launch.
    report["available_artifact_sha256"] = {
        name: sha256_file(path) for name, path in paths.items() if path.is_file()
    }
    if config.error_recycling.enabled:
        report["method_contract"] = error_recycling_contract(config)
    return report


def cpu_dry_run(
    config: CausalTeacherForcingConfig,
    config_path: str | Path,
) -> dict[str, Any]:
    report = validation_report(config, config_path)
    visibility = teacher_forcing_visibility(
        latent_shape(config)[1],
        frames_per_block=config.model.num_frame_per_block,
        independent_first_frame=config.model.independent_first_frame,
    )
    assert_no_future_leakage(
        visibility,
        num_frames=latent_shape(config)[1],
        frames_per_block=config.model.num_frame_per_block,
        independent_first_frame=config.model.independent_first_frame,
    )
    report.update(
        status="dry_run_valid",
        mode="cpu_dry_run",
        teacher_forcing_visibility_shape=list(visibility.shape),
        teacher_forcing_visible_edges=int(visibility.sum().item()),
        future_leakage=False,
    )
    return report


def _import_factory(spec: str):
    if ":" not in spec:
        raise ValueError("data_factory must be 'module:function'")
    module_name, function_name = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(f"cannot import data module {module_name!r}") from exc
    try:
        return getattr(module, function_name)
    except AttributeError as exc:
        raise RuntimeError(f"data factory {spec!r} does not exist") from exc


def _build_optimizer(model: torch.nn.Module, config: CausalTeacherForcingConfig):
    adapter_parameters = []
    lora_parameters = []
    unexpected = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "act_control_adapter" in name:
            adapter_parameters.append(parameter)
        elif ".lora_a." in name or ".lora_b." in name:
            lora_parameters.append(parameter)
        else:
            unexpected.append(name)
    if unexpected:
        raise RuntimeError(f"unexpected trainable base-model parameters: {unexpected[:16]}")
    if not adapter_parameters or not lora_parameters:
        raise RuntimeError("both inherited action-adapter and LoRA groups must be non-empty")
    if config.optimizer.name != "adamw8bit":
        raise ValueError("causal v1 supports only adamw8bit")
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise RuntimeError(
            "--launch requires bitsandbytes for AdamW8bit in the configured project environment"
        ) from exc
    return bnb.optim.AdamW8bit(
        [
            {
                "params": adapter_parameters,
                "lr": config.optimizer.adapter_lr,
                "name": "action_adapter",
            },
            {"params": lora_parameters, "lr": config.optimizer.lora_lr, "name": "lora"},
        ],
        betas=config.optimizer.betas,
        weight_decay=config.optimizer.weight_decay,
    )


def _build_model(
    config: CausalTeacherForcingConfig,
    teacher_payload: Mapping[str, Any],
    device: torch.device,
):
    # Delayed import prevents CPU validation from importing optional CUDA
    # kernels or materializing the 5B checkpoint.
    from utils.wan_wrapper import WanDiffusionWrapper

    wrapper = construct_causal_wrapper(config, WanDiffusionWrapper)
    if config.model.gradient_checkpointing:
        wrapper.enable_gradient_checkpointing()
    summary = initialize_from_action_teacher(wrapper, teacher_payload, config)
    wrapper.to(device=device, dtype=torch.bfloat16)
    wrapper.train()
    return wrapper, summary


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    config: CausalTeacherForcingConfig,
    step: int,
    micro_batches_consumed: int,
    metrics: dict[str, Any],
    manifest_hashes: dict[str, str],
    teacher_lineage: dict[str, Any],
    recycler: ContextErrorRecycling | None = None,
) -> dict[str, Any]:
    if config.error_recycling.enabled != (recycler is not None):
        raise ValueError("checkpoint requires the matching error-recycling runtime state")
    payload = {
        "format_version": 1,
        "stage": stage_name(config),
        "step": step,
        "micro_batches_consumed": micro_batches_consumed,
        "trainable_model": trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
        "metrics": metrics,
        "manifest_hashes": manifest_hashes,
        "parent_teacher": teacher_lineage,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }
    if config.data.num_frames == 97 or recycler is not None:
        payload.update(python_rng_state=random.getstate(), numpy_rng_state=np.random.get_state())
    if recycler is not None:
        if recycler.metrics()["observations"] != micro_batches_consumed:
            raise ValueError("error buffer observations differ from checkpoint data cursor")
        payload.update(method_contract=error_recycling_contract(config),
                       error_recycling_state=recycler.state_dict())
    return payload


def _restore_resume(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    hashes: dict[str, str],
    teacher_lineage: dict[str, Any],
    gradient_accumulation_steps: int,
    config: CausalTeacherForcingConfig | None = None,
    recycler: ContextErrorRecycling | None = None,
) -> tuple[int, int]:
    checkpoint = load_checkpoint(path, expected_manifest_hashes=hashes, map_location="cpu")
    expected_stage = stage_name(config) if config is not None else STAGE_NAME
    if checkpoint.get("stage") != expected_stage:
        raise ValueError(f"resume checkpoint is not from {expected_stage}; method changes require a fresh stage")
    if checkpoint.get("parent_teacher") != teacher_lineage:
        raise ValueError("resume checkpoint parent-teacher lineage changed")
    enabled = config is not None and config.error_recycling.enabled
    if enabled != (recycler is not None):
        raise ValueError("resume requires the matching error-recycling runtime state")
    strict = enabled or (config is not None and config.data.num_frames == 97)
    if strict:
        if checkpoint.get("config") != config.to_dict():
            raise ValueError("strict causal resume configuration changed")
        required = ("python_rng_state", "numpy_rng_state",
                    "micro_batches_consumed", "optimizer", "torch_rng_state", "cuda_rng_state_all")
        if any(name not in checkpoint for name in required):
            raise ValueError("strict causal resume is missing a resumable state field")
    if enabled:
        if "error_recycling_state" not in checkpoint:
            raise ValueError("error-recycling resume is missing its resumable buffer state")
        if checkpoint.get("method_contract") != error_recycling_contract(config):
            raise ValueError("strict error-recycling resume method contract changed")
    step = int(checkpoint["step"])
    micro_batches_consumed = int(
        checkpoint.get("micro_batches_consumed", step * gradient_accumulation_steps)
    )
    if micro_batches_consumed != step * gradient_accumulation_steps:
        raise ValueError("checkpoint micro-batch position is inconsistent with optimizer step")
    if recycler is not None:
        if checkpoint["error_recycling_state"].get("observations") != micro_batches_consumed:
            raise ValueError("resume error buffer observations differ from data cursor")
        recycler.load_state_dict(checkpoint["error_recycling_state"])
    load_trainable_state_dict(model, checkpoint["trainable_model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    if strict:
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
    return step, micro_batches_consumed


def _iterator_at_micro_batch(
    loader: Iterable[Mapping[str, Any]], micro_batches_consumed: int
) -> tuple[Iterable[Mapping[str, Any]], Any]:
    """Recreate the deterministic data position recorded by a checkpoint."""

    if micro_batches_consumed < 0:
        raise ValueError("micro_batches_consumed cannot be negative")
    skip = micro_batches_consumed
    try:
        length = len(loader)  # type: ignore[arg-type]
    except TypeError:
        length = None
    if length is not None:
        if length <= 0:
            raise RuntimeError("data factory returned an empty iterable")
        skip %= length
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
    return loader, iterator


def _resume_iterator_preserving_rng(loader, micro_batches_consumed):
    """Rebuild/skip the stateless data cursor without consuming restored RNG."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            return _iterator_at_micro_batch(loader, micro_batches_consumed)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def launch(
    config: CausalTeacherForcingConfig,
    config_path: str | Path,
    resume: str | None,
    *,
    confirmed_gpu_index: int,
    confirmed_gpu_uuid: str,
    confirmed_at_utc: str,
    allocation_profile: str,
    stop_after_step: int | None = None,
) -> None:
    # All CPU lineage checks happen before model construction.  CUDA inventory
    # is touched only inside the explicit --launch path.
    target_step = config.training.max_steps if stop_after_step is None else stop_after_step
    if not 1 <= target_step <= config.training.max_steps:
        raise ValueError("execution stop must be between 1 and configured max_steps")
    hashes = artifact_hashes(config, config_path)
    teacher_payload, teacher_lineage = load_teacher_checkpoint(config, hashes)
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
        raise RuntimeError(
            f"refusing a fresh launch into non-empty output directory: {output_dir}"
        )
    if resume is not None:
        resolved_resume = Path(resume).resolve()
        resolved_output = output_dir.resolve()
        if resolved_output not in resolved_resume.parents:
            raise RuntimeError("resume checkpoint must belong to the configured output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "command": sys.argv,
        "stage": stage_name(config),
        "git_revision": git_revision(Path(__file__).parent),
        "config": config.to_dict(),
        "manifest_hashes": hashes,
        "parent_teacher": teacher_lineage.as_dict(),
        "seed": config.training.seed,
        "device": torch.cuda.get_device_name(device),
        "gpu_state": {
            "prelaunch_snapshot": gpu_snapshot.as_dict(),
            "allocated_bytes_at_start": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes_at_start": int(torch.cuda.memory_reserved(device)),
        },
        "conditioning_modules": "precomputed; VAE and T5 are not loaded in this process",
        "started_unix": time.time(),
        "execution_stop_after_step": target_step,
        "configured_max_steps": config.training.max_steps,
    }
    if config.error_recycling.enabled:
        metadata.update(method_contract=error_recycling_contract(config),
                        execution_mode="strict_resume" if resume else "fresh_stage_weights_only_parent",
                        error_recycling_restore_status="pending" if resume else "fresh_empty_buffer")
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    model, summary = _build_model(config, teacher_payload, device)
    del teacher_payload
    optimizer = _build_optimizer(model, config)
    factory = _import_factory(config.data.data_factory)
    loader: Iterable[Mapping[str, Any]] = factory(config=config.data, training=config.training)
    lineage_dict = teacher_lineage.as_dict()
    start_step = 0
    micro_batches_consumed = 0
    recycler = ContextErrorRecycling(config.error_recycling) if config.error_recycling.enabled else None
    if resume:
        start_step, micro_batches_consumed = _restore_resume(
            resume,
            model=model,
            optimizer=optimizer,
            hashes=hashes,
            teacher_lineage=lineage_dict,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            config=config,
            recycler=recycler,
        )
        if start_step >= target_step:
            raise ValueError("execution stop must be greater than the resumed optimizer step")
    if recycler is not None:
        metadata["error_recycling_restore_status"] = "restored_exact_state" if resume else "fresh_empty_buffer"
        metadata["error_recycling_initial_metrics"] = recycler.metrics()
        (output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    manager = CheckpointManager(output_dir, keep_last=config.training.keep_last)

    print(json.dumps({
        "status": "launching",
        "stage": stage_name(config),
        "trainable_parameters": summary.trainable_parameters,
        "total_parameters": summary.total_parameters,
        "lora_layers": len(summary.replaced_linear_layers),
        "teacher_step": teacher_lineage.step,
        "resume_step": start_step,
    }, sort_keys=True))
    model.zero_grad(set_to_none=True)
    iterator_factory = (_resume_iterator_preserving_rng
                        if resume and (recycler is not None or config.data.num_frames == 97)
                        else _iterator_at_micro_batch)
    _, iterator = iterator_factory(loader, micro_batches_consumed)
    tracker = ThroughputTracker.start()
    accumulated_loss = 0.0
    micro_step = micro_batches_consumed
    global_step = start_step
    last_saved_step = start_step
    latest_metrics: dict[str, Any] = {"loss": float("inf")}
    step_started = time.perf_counter()

    while global_step < target_step:
        if micro_step % config.training.gradient_accumulation_steps == 0:
            torch.cuda.synchronize(device)
            step_started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise RuntimeError("data factory returned an empty iterable") from exc
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = causal_teacher_forcing_loss(model, batch, config, device, recycler=recycler)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite causal teacher-forcing loss: {loss.item()}")
        (loss / config.training.gradient_accumulation_steps).backward()
        accumulated_loss += float(loss.detach())
        micro_step += 1
        tracker.update(config.training.micro_batch_size)
        if micro_step % config.training.gradient_accumulation_steps:
            continue

        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, config.optimizer.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        mean_loss = accumulated_loss / config.training.gradient_accumulation_steps
        accumulated_loss = 0.0
        latest_metrics = {
            "loss": mean_loss,
            "grad_norm": float(grad_norm),
            "samples_per_second": tracker.samples_per_second,
            "peak_vram_bytes": peak_vram_bytes(device),
        }
        torch.cuda.synchronize(device)
        latest_metrics["optimizer_step_seconds"] = time.perf_counter() - step_started
        if recycler is not None:
            latest_metrics.update(error_recycling=recycler.metrics(),
                                  error_recycling_restore_status=metadata["error_recycling_restore_status"])
        if global_step % config.training.log_every == 0:
            record = {"step": global_step, **latest_metrics, "unix": time.time()}
            append_jsonl(output_dir / "metrics.jsonl", record)
            print(json.dumps(record, sort_keys=True), flush=True)
        if global_step % config.training.checkpoint_every == 0:
            payload = _checkpoint_payload(
                model,
                optimizer,
                config=config,
                step=global_step,
                micro_batches_consumed=micro_step,
                metrics=latest_metrics,
                manifest_hashes=hashes,
                teacher_lineage=lineage_dict,
                recycler=recycler,
            )
            manager.save(payload, step=global_step, metric=mean_loss)
            last_saved_step = global_step

    if global_step > 0 and last_saved_step != global_step:
        payload = _checkpoint_payload(
            model,
            optimizer,
            config=config,
            step=global_step,
            micro_batches_consumed=micro_step,
            metrics=latest_metrics,
            manifest_hashes=hashes,
            teacher_lineage=lineage_dict,
            recycler=recycler,
        )
        manager.save(payload, step=global_step, metric=float(latest_metrics["loss"]))
    execution_result = {
        "status": "complete" if global_step == config.training.max_steps else "segment_complete",
        "stop_reason": "configured_max_steps" if global_step == config.training.max_steps else "execution_step_cap",
        "stage": stage_name(config),
        "step": global_step,
        "micro_batches_consumed": micro_step,
        "configured_max_steps": config.training.max_steps,
        "execution_stop_after_step": target_step,
        "resume_step": start_step,
        "checkpoint_path": str((output_dir / "checkpoints" / f"step-{global_step:07d}.pt").resolve()),
        "output_dir": str(output_dir.resolve()),
        "peak_vram_bytes": peak_vram_bytes(device),
        **({"error_recycling": recycler.metrics(),
            "error_recycling_restore_status": metadata["error_recycling_restore_status"]}
           if recycler is not None else {}),
    }
    (output_dir / "execution_result.json").write_text(
        json.dumps(execution_result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(execution_result, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_causal_config(args.config)
    _apply_overrides(config, args)
    if not args.launch:
        report = cpu_dry_run(config, args.config) if args.dry_run else validation_report(
            config, args.config
        )
        print(json.dumps(report, indent=2, sort_keys=True))
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
        stop_after_step=args.stop_after_step,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
