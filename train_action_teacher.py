"""Stage-1 ABot-inspired action teacher training entry point.

Without ``--launch`` this command performs CPU-only configuration validation.
It never downloads weights.  CUDA model construction and training are gated
behind the explicit ``--launch`` flag.

Expected data-factory contract
------------------------------
The configured ``module:function`` is called as ``factory(config=cfg.data,
training=cfg.training)`` and must return an iterable of mappings containing
precomputed ``noisy_latents``, ``target_flow``, ``timesteps``,
``prompt_embeds``, and ``actions``. The original 49-RGB-frame contract uses
actions ``[B,48,8]`` and latents ``[B,13,48,30,52]`` (B,F,C,H,W). The explicit
97-frame contract uses ``[B,96,8]`` and ``[B,25,48,30,52]`` independently
encoded from continuous RGB windows. Optional Wan conditions (``y``,
``clip_fea``) are forwarded unchanged.

The repository does not publish ABot's teacher-training loader or optimizer
state, so those are intentionally explicit integration boundaries rather than
fabricated compatibility claims.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import re
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import numpy as np

from training.config import ActionTeacherConfig, load_config
from training.gpu_gate import query_dedicated_gpu, validate_confirmation
from training.models.action_adapter import build_action_context, pack_canonical_actions
from training.models.lora import (
    configure_action_teacher,
    load_trainable_state_dict,
    trainable_state_dict,
)
from training.runtime import (
    CheckpointManager,
    OptimizerStepProfiler,
    ThroughputTracker,
    append_jsonl,
    collect_manifest_hashes,
    git_revision,
    load_checkpoint,
    peak_vram_bytes,
    seed_everything,
    sha256_file,
)

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "train" / "action_teacher_lora_v1.yaml"
GATE_COMPLETION_STEP = 20
SHARED_DATA_HASH_KEYS = ("dataset_manifest", "feature_index", "feature_receipt")
PROMPT_CACHE_HASH_KEYS = ("prompt_cache", "prompt_cache_receipt")
LEGACY_ACTION_FACTORY = "training.data.action_dataset:build_action_teacher_dataloader"
RESAMPLED_ACTION_FACTORY = "training.data.action_resampled:build_resampled_action_teacher_dataloader"
LENGTH_TRANSITION_SOURCE_STEP = 1040
LENGTH_TRANSITION_DATA_FIELDS = frozenset({
    "num_frames", "manifest_path", "manifest_sha256", "prompt_cache_path", "data_factory",
})


def _canonical_data_contract(value: Any) -> dict[str, Any]:
    """The absent legacy optional prompt path means the original feature text."""
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint has no serialized data configuration")
    result = dict(value)
    result.setdefault("prompt_cache_path", None)
    return result


def _prompt_contract(data: Mapping[str, Any], hashes: Mapping[str, str]) -> dict[str, Any]:
    path = data.get("prompt_cache_path")
    if path is not None and any(not hashes.get(key) for key in PROMPT_CACHE_HASH_KEYS):
        raise ValueError("static prompt contract is missing prompt cache artifact hashes")
    if path is None and any(key in hashes for key in PROMPT_CACHE_HASH_KEYS):
        raise ValueError("legacy prompt contract unexpectedly contains prompt cache hashes")
    return {
        "policy": "scene_static_only_v1" if path is not None else "original_feature_cache_prompt",
        "prompt_cache_path": path,
        "artifact_hashes": {key: hashes[key] for key in PROMPT_CACHE_HASH_KEYS if key in hashes},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="CPU-only shape/config validation; does not construct the Wan model",
    )
    mode.add_argument(
        "--launch",
        action="store_true",
        help="explicitly allow CUDA model loading and training",
    )
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--resume", default=None, help="same-run resumable checkpoint path")
    checkpoint.add_argument(
        "--initialize-from",
        default=None,
        help="completed gate20 checkpoint used to continue in a fresh full-run directory",
    )
    checkpoint.add_argument(
        "--warm-start-from",
        default=None,
        help="action-teacher weights only; new optimizer, RNG, step zero, and output directory",
    )
    checkpoint.add_argument(
        "--sampling-transition-from",
        default=None,
        help="explicit legacy-to-resampled data-factory transition; identical model/data/optimizer, weights only and fresh step zero",
    )
    checkpoint.add_argument(
        "--length-transition-from", default=None,
        help="explicit 49-to-97-frame parent step1040 weights only; new optimizer and step zero",
    )
    parser.add_argument(
        "--length-transition-sha256", default=None,
        help="required exact parent checkpoint SHA256 for --length-transition-from",
    )
    parser.add_argument(
        "--stop-after-step", type=int, default=None,
        help="absolute optimizer-step execution cap; preserves fixed config/max_steps for strict resume",
    )
    parser.add_argument(
        "--profile-runtime", action="store_true",
        help="record synchronized optimizer-step/checkpoint and sampled driver memory measurements (automatic for97 frames)",
    )
    parser.add_argument("--confirmed-gpu-index", type=int)
    parser.add_argument("--confirmed-gpu-uuid")
    parser.add_argument("--confirmed-at-utc")
    parser.add_argument("--allocation-profile")
    args = parser.parse_args(argv)
    if bool(args.length_transition_from) != bool(args.length_transition_sha256):
        parser.error("--length-transition-from and --length-transition-sha256 must be supplied together")
    if args.length_transition_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", args.length_transition_sha256):
        parser.error("--length-transition-sha256 must be 64 lowercase hexadecimal characters")
    if args.stop_after_step is not None and args.stop_after_step <= 0:
        parser.error("--stop-after-step must be positive")
    return args


def _latent_shape(config: ActionTeacherConfig) -> tuple[int, int, int, int, int]:
    model, data, train = config.model, config.data, config.training
    latent_frames = 1 + (data.num_frames - 1) // model.temporal_compression
    return (
        train.micro_batch_size,
        latent_frames,
        model.latent_channels,
        data.height // model.spatial_compression,
        data.width // model.spatial_compression,
    )


def validation_report(config: ActionTeacherConfig, config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    manifest = Path(config.data.manifest_path)
    report: dict[str, Any] = {
        "status": "configuration_valid",
        "mode": "cpu_validate",
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "git_revision": git_revision(Path(__file__).parent),
        "manifest_path": str(manifest),
        "manifest_exists": manifest.is_file(),
        "rgb_shape": [config.training.micro_batch_size, config.data.num_frames, 3, config.data.height, config.data.width],
        "latent_shape_bfchw": list(_latent_shape(config)),
        "future_action_shape": [config.training.micro_batch_size, config.data.num_frames - 1, 8],
        "packed_action_shape": [
            config.training.micro_batch_size,
            (config.data.num_frames - 1) // config.data.rgb_frames_per_action_token,
            config.model.action_dim,
        ],
        "cuda_queried": False,
        "weights_loaded": False,
    }
    if manifest.is_file():
        report["manifest_sha256"] = sha256_file(manifest)
        expected = config.data.manifest_sha256
        report["manifest_hash_matches_config"] = expected is None or expected == report["manifest_sha256"]
    return report


def cpu_dry_run(config: ActionTeacherConfig, config_path: str | Path) -> dict[str, Any]:
    report = validation_report(config, config_path)
    actions = torch.zeros(
        config.training.micro_batch_size,
        config.data.num_frames - 1,
        8,
        dtype=torch.float32,
    )
    packed = pack_canonical_actions(actions)
    report.update(
        status="dry_run_valid",
        mode="cpu_dry_run",
        packed_action_tensor_shape=list(packed.shape),
        action_context_shapes=[
            [
                config.model.action_dim,
                packed.shape[1],
                config.data.height,
                config.data.width,
            ]
            for _ in range(config.training.micro_batch_size)
        ],
    )
    return report


def _import_factory(spec: str):
    if ":" not in spec:
        raise ValueError("data_factory must be 'module:function'")
    module_name, function_name = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"data integration boundary is not implemented: cannot import {module_name!r}"
        ) from exc
    try:
        return getattr(module, function_name)
    except AttributeError as exc:
        raise RuntimeError(f"data factory {spec!r} does not exist") from exc


def _manifest_hashes(config: ActionTeacherConfig, config_path: str | Path) -> dict[str, str]:
    from training.data.action_dataset import cache_index_path, validate_feature_cache_binding

    feature_index = cache_index_path(config.data.manifest_path)
    feature_receipt = feature_index.with_suffix(feature_index.suffix + ".receipt.json")
    receipt = validate_feature_cache_binding(feature_index, config.data.manifest_path)
    if config.data.num_frames == 97:
        from training.data.action_dataset import validate_window97_contract

        validate_window97_contract(receipt)
    hashes = collect_manifest_hashes(
        {
            "dataset_manifest": config.data.manifest_path,
            "feature_index": feature_index,
            "feature_receipt": feature_receipt,
            "training_config": config_path,
        }
    )
    expected = config.data.manifest_sha256
    if expected is not None and hashes["dataset_manifest"] != expected:
        raise ValueError(
            f"dataset manifest hash mismatch: expected {expected}, got {hashes['dataset_manifest']}"
        )
    if config.data.prompt_cache_path is not None:
        from training.data.action_dataset import validate_scene_static_prompt_cache_binding

        prompt_path = Path(config.data.prompt_cache_path)
        prompt_receipt_path = prompt_path.with_suffix(prompt_path.suffix + ".receipt.json")
        prompt_receipt = validate_scene_static_prompt_cache_binding(
            prompt_path, index_path=feature_index, manifest_path=config.data.manifest_path,
        )
        # The binding validator hashes the full tensor sidecar; do not read its
        # several gigabytes a second time solely for the training hash ledger.
        hashes["prompt_cache"] = prompt_receipt["cache_sha256"]
        hashes["prompt_cache_receipt"] = sha256_file(prompt_receipt_path)
    return hashes


def _build_model(config: ActionTeacherConfig, device: torch.device):
    # Delayed import is intentional: CPU validation must not import optional
    # CUDA kernels or materialize the 5B checkpoint.
    from utils.wan_wrapper import WanDiffusionWrapper

    wrapper = WanDiffusionWrapper(
        model_name=config.model.base_model_path,
        timestep_shift=config.model.timestep_shift,
        is_causal=False,
        model_type=config.model.model_type,
    )
    if config.model.gradient_checkpointing:
        wrapper.enable_gradient_checkpointing()
    summary = configure_action_teacher(
        wrapper.model,
        rank=config.model.lora_rank,
        alpha=config.model.lora_alpha,
        dropout=config.model.lora_dropout,
    )
    wrapper.to(device=device, dtype=torch.bfloat16)
    wrapper.train()
    return wrapper, summary


def _build_optimizer(model: torch.nn.Module, config: ActionTeacherConfig):
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
        raise RuntimeError(f"unexpected trainable backbone parameters: {unexpected[:16]}")
    if not adapter_parameters or not lora_parameters:
        raise RuntimeError("both official action-adapter and LoRA parameter groups must be non-empty")
    groups = [
        {"params": adapter_parameters, "lr": config.optimizer.adapter_lr, "name": "action_adapter"},
        {"params": lora_parameters, "lr": config.optimizer.lora_lr, "name": "lora"},
    ]
    if config.optimizer.name != "adamw8bit":
        raise ValueError("v1 supports only adamw8bit")
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise RuntimeError(
            "--launch requires bitsandbytes for AdamW8bit; install it in the configured project environment"
        ) from exc
    return bnb.optim.AdamW8bit(
        groups,
        betas=config.optimizer.betas,
        weight_decay=config.optimizer.weight_decay,
    )


def _validate_batch(batch: Mapping[str, Any], config: ActionTeacherConfig) -> None:
    required = {"noisy_latents", "target_flow", "timesteps", "prompt_embeds", "actions"}
    missing = required.difference(batch)
    if missing:
        raise ValueError(f"training batch is missing keys: {sorted(missing)}")
    noisy = batch["noisy_latents"]
    target = batch["target_flow"]
    actions = batch["actions"]
    expected = _latent_shape(config)
    if tuple(noisy.shape) != expected or tuple(target.shape) != expected:
        raise ValueError(
            f"noisy_latents/target_flow must both be {expected} (B,F,C,H,W), "
            f"got {tuple(noisy.shape)} and {tuple(target.shape)}"
        )
    expected_actions = (config.training.micro_batch_size, config.data.num_frames - 1, 8)
    if tuple(actions.shape) != expected_actions:
        raise ValueError(f"actions must be {expected_actions}, got {tuple(actions.shape)}")


def _move(value: Any, device: torch.device, *, dtype: torch.dtype | None = None):
    if torch.is_tensor(value):
        if value.is_floating_point() and dtype is not None:
            return value.to(device=device, dtype=dtype, non_blocking=True)
        return value.to(device=device, non_blocking=True)
    if isinstance(value, list):
        return [_move(item, device, dtype=dtype) for item in value]
    return value


def _forward_loss(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    config: ActionTeacherConfig,
    device: torch.device,
) -> torch.Tensor:
    _validate_batch(batch, config)
    noisy = _move(batch["noisy_latents"], device, dtype=torch.bfloat16)
    target = _move(batch["target_flow"], device, dtype=torch.bfloat16)
    timestep = _move(batch["timesteps"], device)
    prompt_embeds = _move(batch["prompt_embeds"], device, dtype=torch.bfloat16)
    if torch.is_tensor(prompt_embeds):
        prompt_embeds = [sample for sample in prompt_embeds]
    actions = _move(batch["actions"], device, dtype=torch.bfloat16)
    conditions: dict[str, Any] = {
        "prompt_embeds": prompt_embeds,
        "act_context": build_action_context(
            actions,
            height=config.data.height,
            width=config.data.width,
            device=device,
            dtype=torch.bfloat16,
        ),
        "act_context_scale": config.model.action_scale,
    }
    for key in ("y", "clip_fea"):
        if key in batch:
            conditions[key] = _move(batch[key], device, dtype=torch.bfloat16)
    model_output = model(
        noisy,
        conditional_dict=conditions,
        timestep=timestep,
        # TI2V keeps latent frame zero clean.  Without this flag the wrapper's
        # uniform-timestep path would reduce [B,F] to frame zero (t=0) and
        # accidentally train the entire clip at zero noise.
        replace_first_timestep_and_noise_latents=True,
    )
    prediction = model_output[0] if isinstance(model_output, tuple) else model_output
    if prediction.shape != target.shape:
        raise RuntimeError(
            f"Wan prediction shape {tuple(prediction.shape)} != target {tuple(target.shape)}"
        )
    # Frame zero is a clean TI2V condition, not a denoising target.  Excluding
    # it avoids training the model to emit an artificial zero-flow prediction.
    return F.mse_loss(prediction[:, 1:].float(), target[:, 1:].float())


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    config: ActionTeacherConfig,
    step: int,
    micro_batches_consumed: int,
    metrics: dict[str, Any],
    manifest_hashes: dict[str, str],
    initialization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 1,
        "stage": "action_teacher_lora_v1",
        "step": step,
        "micro_batches_consumed": micro_batches_consumed,
        "trainable_model": trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
        "metrics": metrics,
        "manifest_hashes": manifest_hashes,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "rng_contract_version": 2,
    }
    if initialization is not None:
        payload["initialization"] = dict(initialization)
    return payload


def _iterator_at_micro_batch(
    loader: Iterable[Mapping[str, Any]], micro_batches_consumed: int
) -> tuple[Iterable[Mapping[str, Any]], Any]:
    """Recreate the deterministic loader position used by a resumed run."""
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


def _load_gate_checkpoint(
    path: str | Path,
    *,
    config: ActionTeacherConfig,
    current_manifest_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Validate the exact gate20 continuation contract.

    A normal ``--resume`` remains strict about the training-config hash and
    output directory.  Gate20 intentionally uses a different max-step and
    output directory, so this narrow initializer compares the shared data,
    model, data, and optimizer contracts while requiring the exact completed
    gate step.
    """

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0 compatibility
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("gate checkpoint must contain a mapping")
    if payload.get("format_version") != 1 or payload.get("stage") != "action_teacher_lora_v1":
        raise ValueError("gate checkpoint is not an action_teacher_lora_v1 format_version=1 artifact")
    if payload.get("step") != GATE_COMPLETION_STEP:
        raise ValueError(
            f"gate checkpoint must be exactly step {GATE_COMPLETION_STEP}, got {payload.get('step')!r}"
        )
    source_hashes = payload.get("manifest_hashes")
    if not isinstance(source_hashes, Mapping):
        raise ValueError("gate checkpoint has no manifest hashes")
    for key in (*SHARED_DATA_HASH_KEYS, *PROMPT_CACHE_HASH_KEYS):
        if source_hashes.get(key) != current_manifest_hashes.get(key):
            raise ValueError(
                f"gate checkpoint {key} mismatch: "
                f"gate={source_hashes.get(key)!r}, current={current_manifest_hashes.get(key)!r}"
            )

    source_config = payload.get("config")
    if not isinstance(source_config, Mapping):
        raise ValueError("gate checkpoint has no serialized configuration")
    current_config = config.to_dict()
    for section in ("model", "data", "optimizer"):
        source_section = source_config.get(section)
        current_section = current_config[section]
        if section == "data":
            source_section = _canonical_data_contract(source_section)
            current_section = _canonical_data_contract(current_section)
        if source_section != current_section:
            raise ValueError(f"gate checkpoint {section} contract differs from the full run")
    _prompt_contract(_canonical_data_contract(source_config.get("data")), source_hashes)
    _prompt_contract(current_config["data"], current_manifest_hashes)
    source_training = source_config.get("training")
    if not isinstance(source_training, Mapping):
        raise ValueError("gate checkpoint has no serialized training configuration")
    if source_training.get("max_steps") != GATE_COMPLETION_STEP:
        raise ValueError("gate checkpoint was not produced by the 20-step gate configuration")
    gate_training_contract = dict(source_training)
    full_training_contract = dict(current_config["training"])
    for field in ("max_steps", "output_dir"):
        gate_training_contract.pop(field, None)
        full_training_contract.pop(field, None)
    if gate_training_contract != full_training_contract:
        raise ValueError("gate checkpoint training contract differs from the full run")
    micro_batches = payload.get("micro_batches_consumed")
    expected_micro_batches = GATE_COMPLETION_STEP * config.training.gradient_accumulation_steps
    if micro_batches != expected_micro_batches:
        raise ValueError(
            "gate checkpoint micro-batch position is inconsistent with the full-run contract"
        )
    for key in ("trainable_model", "optimizer", "torch_rng_state", "cuda_rng_state_all"):
        if key not in payload:
            raise ValueError(f"gate checkpoint is missing resumable field {key!r}")
    if not isinstance(payload["trainable_model"], Mapping) or not payload["trainable_model"]:
        raise ValueError("gate checkpoint has no trainable model state")
    if not isinstance(payload["optimizer"], Mapping):
        raise ValueError("gate checkpoint optimizer state must be a mapping")
    if GATE_COMPLETION_STEP >= config.training.max_steps:
        raise ValueError("full action run must extend beyond the completed gate step")
    return payload


def _load_warm_start_checkpoint(
    path: str | Path,
    *,
    config: ActionTeacherConfig,
    current_manifest_hashes: Mapping[str, str],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load only compatible Action weights, never resumable training state.

    The same base, LoRA, tensor layout, conditioning, and data contracts are
    required. Action scale, activation checkpointing, and the optional static
    prompt sidecar may change, as may the optimizer and fresh-run training
    settings. Prompt transitions are recorded, never treated as resumes.
    Exact tensor names/shapes
    are additionally checked by ``load_trainable_state_dict`` on the model.
    """
    source = Path(path).resolve()
    source_sha256 = sha256_file(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if sha256_file(source) != source_sha256:
        raise ValueError("warm-start checkpoint changed while loading")
    if not isinstance(payload, Mapping):
        raise ValueError("warm-start checkpoint must contain a mapping")
    if payload.get("format_version") != 1 or payload.get("stage") != "action_teacher_lora_v1":
        raise ValueError("warm-start requires an action_teacher_lora_v1 format_version=1 checkpoint")
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("warm-start checkpoint must have a positive optimizer step")
    source_hashes = payload.get("manifest_hashes")
    if not isinstance(source_hashes, Mapping):
        raise ValueError("warm-start checkpoint has no manifest hashes")
    for key in SHARED_DATA_HASH_KEYS:
        expected = current_manifest_hashes.get(key)
        if not expected or source_hashes.get(key) != expected:
            raise ValueError(f"warm-start checkpoint {key} mismatch")
    source_config = payload.get("config")
    if not isinstance(source_config, Mapping):
        raise ValueError("warm-start checkpoint has no serialized configuration")
    source_model = source_config.get("model")
    if not isinstance(source_model, Mapping):
        raise ValueError("warm-start checkpoint has no serialized model configuration")
    current_config = config.to_dict()
    source_model_contract = dict(source_model)
    current_model_contract = dict(current_config["model"])
    for field in ("action_scale", "gradient_checkpointing"):
        source_model_contract.pop(field, None)
        current_model_contract.pop(field, None)
    if source_model_contract != current_model_contract:
        raise ValueError("warm-start checkpoint model architecture/conditioning contract differs")
    source_data = _canonical_data_contract(source_config.get("data"))
    current_data = _canonical_data_contract(current_config["data"])
    source_data_contract = dict(source_data)
    current_data_contract = dict(current_data)
    source_data_contract.pop("prompt_cache_path")
    current_data_contract.pop("prompt_cache_path")
    if source_data_contract != current_data_contract:
        raise ValueError("warm-start checkpoint data contract differs")
    source_prompt = _prompt_contract(source_data, source_hashes)
    current_prompt = _prompt_contract(current_data, current_manifest_hashes)
    state = payload.get("trainable_model")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("warm-start checkpoint has no trainable model state")
    if any(not isinstance(name, str) or not torch.is_tensor(value) for name, value in state.items()):
        raise ValueError("warm-start trainable model must map parameter names to tensors")
    initialization = {
        "mode": "warm_start_weights_only",
        "path": str(source),
        "sha256": source_sha256,
        "source_stage": payload["stage"],
        "source_step": step,
        "source_config": dict(source_config),
        "source_manifest_hashes": dict(source_hashes),
        "target_manifest_hashes": dict(current_manifest_hashes),
        "prompt_transition": {
            "changed": source_prompt != current_prompt,
            "source": source_prompt,
            "target": current_prompt,
            "mode": "weights_only_new_training_contract",
        },
        "source_initialization": payload.get("initialization"),
        "optimizer_restored": False,
        "rng_restored": False,
        "start_step": 0,
        "micro_batches_consumed": 0,
        "seed": config.training.seed,
    }
    # Returning only weights and provenance lets the unused old optimizer/RNG
    # tensors be released; neither can accidentally enter the new run.
    return dict(state), initialization


def _load_length_transition_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    config: ActionTeacherConfig,
    current_manifest_hashes: Mapping[str, str],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Pin one parent and allow only the documented 49-to-97 cache transition.

    Target feature receipts are validated by ``_manifest_hashes`` before this
    function. They bind independently encoded continuous 97-frame RGB windows
    and inline scene-static text. No parent optimizer, RNG, or data position is
    returned: this is a new optimization stage, never source step 1041.
    """
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("length-transition requires an exact lowercase SHA256 pin")
    source = Path(path).resolve()
    if sha256_file(source) != expected_sha256:
        raise ValueError("length-transition parent SHA256 does not match the required pin")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if sha256_file(source) != expected_sha256:
        raise ValueError("length-transition checkpoint changed while loading")
    if not isinstance(payload, Mapping):
        raise ValueError("length-transition checkpoint must contain a mapping")
    if payload.get("format_version") != 1 or payload.get("stage") != "action_teacher_lora_v1":
        raise ValueError("length-transition requires an action_teacher_lora_v1 format_version=1 checkpoint")
    if type(payload.get("step")) is not int or payload["step"] != LENGTH_TRANSITION_SOURCE_STEP:
        raise ValueError("length-transition requires the pinned completed parent step1040")
    source_config = payload.get("config")
    source_hashes = payload.get("manifest_hashes")
    if not isinstance(source_config, Mapping) or not isinstance(source_hashes, Mapping):
        raise ValueError("length-transition requires source configuration and artifact hashes")
    target_config = config.to_dict()
    if set(source_config) != set(target_config):
        raise ValueError("length-transition configuration sections differ")
    for section in ("model", "optimizer"):
        if source_config.get(section) != target_config[section]:
            raise ValueError(f"length-transition {section} contract must remain identical")
    source_data = _canonical_data_contract(source_config.get("data"))
    target_data = _canonical_data_contract(target_config["data"])
    if source_data.get("num_frames") != 49 or target_data.get("num_frames") != 97:
        raise ValueError("length-transition permits only 49-to-97 RGB frames")
    if source_data.get("data_factory") != LEGACY_ACTION_FACTORY or target_data.get("data_factory") != LEGACY_ACTION_FACTORY:
        raise ValueError("length-transition requires the same verified action_dataset factory")
    for side in (source_data, target_data):
        if (side.get("height"), side.get("width"), side.get("rgb_frames_per_action_token")) != (480, 832, 4):
            raise ValueError("length-transition requires 480x832 and four RGB frames per action token")
    if {key: value for key, value in source_data.items() if key not in LENGTH_TRANSITION_DATA_FIELDS} != {
        key: value for key, value in target_data.items() if key not in LENGTH_TRANSITION_DATA_FIELDS
    }:
        raise ValueError("length-transition data contract differs beyond length/cache fields")
    if source_data.get("manifest_path") == target_data.get("manifest_path"):
        raise ValueError("length-transition requires a dedicated new 97-frame manifest")
    source_prompt = _prompt_contract(source_data, source_hashes)
    if source_prompt["policy"] != "scene_static_only_v1" or target_data.get("prompt_cache_path") is not None:
        raise ValueError("length-transition requires static sidecar-to-inline feature prompts")
    allowed_source_hashes = {*SHARED_DATA_HASH_KEYS, *PROMPT_CACHE_HASH_KEYS, "training_config"}
    allowed_target_hashes = {*SHARED_DATA_HASH_KEYS, "training_config"}
    if set(source_hashes) != allowed_source_hashes or set(current_manifest_hashes) != allowed_target_hashes:
        raise ValueError("length-transition artifact hash keys differ from the documented cache transition")
    for key in allowed_source_hashes:
        if not source_hashes.get(key):
            raise ValueError(f"length-transition source is missing {key} hash")
    for key in allowed_target_hashes:
        if not current_manifest_hashes.get(key) or source_hashes.get(key) == current_manifest_hashes[key]:
            raise ValueError(f"length-transition requires a new bound {key} hash")
    source_training = source_config.get("training")
    if not isinstance(source_training, Mapping):
        raise ValueError("length-transition requires source training configuration")
    allowed_training_fields = {"max_steps", "output_dir"}
    if {key: value for key, value in source_training.items() if key not in allowed_training_fields} != {
        key: value for key, value in target_config["training"].items() if key not in allowed_training_fields
    }:
        raise ValueError("length-transition training contract differs beyond max_steps/output_dir")
    source_output = source_training.get("output_dir")
    if not isinstance(source_output, str) or Path(source_output).resolve() == Path(config.training.output_dir).resolve():
        raise ValueError("length-transition requires a different fresh output directory")
    if payload.get("micro_batches_consumed") != LENGTH_TRANSITION_SOURCE_STEP * config.training.gradient_accumulation_steps:
        raise ValueError("length-transition parent micro-batch position is inconsistent")
    state = payload.get("trainable_model")
    if not isinstance(state, Mapping) or not state or any(
        not isinstance(name, str) or not torch.is_tensor(value) for name, value in state.items()
    ):
        raise ValueError("length-transition trainable model must map parameter names to tensors")
    changed_fields = sorted(key for key in source_data if source_data[key] != target_data.get(key))
    return dict(state), {
        "mode": "length_transition_weights_only", "path": str(source), "sha256": expected_sha256,
        "source_stage": payload["stage"], "source_step": payload["step"],
        "source_config": dict(source_config), "source_manifest_hashes": dict(source_hashes),
        "target_manifest_hashes": dict(current_manifest_hashes),
        "source_initialization": payload.get("initialization"),
        "length_transition": {
            "version": 1, "source_rgb_frames": 49, "target_rgb_frames": 97,
            "source_latent_frames": 13, "target_latent_frames": 25,
            "source_future_actions": 48, "target_future_actions": 96,
            "changed_data_fields": changed_fields,
            "encoding_policy": "independent_continuous_rgb_window_vae_v1",
            "target_cache_kind": "action_teacher_window97_features_v1",
            "model_contract_unchanged": True, "optimizer_config_unchanged": True,
            "source_position_restored": False,
        },
        "prompt_transition": {
            "policy_changed": False, "storage_changed": True, "policy": "scene_static_only_v1",
            "source": source_prompt, "target": {"storage": "inline_97_frame_feature_shards", "policy": "scene_static_only_v1"},
        },
        "optimizer_restored": False, "rng_restored": False, "start_step": 0,
        "micro_batches_consumed": 0, "seed": config.training.seed,
    }


def _execution_limit(max_steps: int, stop_after_step: int | None, start_step: int = 0) -> int:
    if stop_after_step is not None and (type(stop_after_step) is not int or stop_after_step <= 0):
        raise ValueError("stop_after_step must be a positive absolute optimizer step")
    limit = max_steps if stop_after_step is None else min(max_steps, stop_after_step)
    if start_step >= limit:
        raise ValueError("execution limit must extend beyond the resumed optimizer step")
    return limit


def _restore_rng_state(checkpoint: Mapping[str, Any], *, require_complete: bool) -> None:
    if require_complete and any(key not in checkpoint for key in (
        "python_rng_state", "numpy_rng_state", "torch_rng_state", "cuda_rng_state_all",
    )):
        raise ValueError("97-frame strict resume requires complete Python/NumPy/Torch/CUDA RNG state")
    torch.set_rng_state(checkpoint["torch_rng_state"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])


def _load_sampling_transition_checkpoint(
    path: str | Path,
    *,
    config: ActionTeacherConfig,
    current_manifest_hashes: Mapping[str, str],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """One allowlisted sampler conversion, not a resume or broad warm start.

    Unlike ordinary warm starts this preserves the complete model, optimizer,
    prompt and data contracts except the exact factory transition below. Only
    max_steps/output_dir change in training. Original SHA/configs stay intact.
    """
    source = Path(path).resolve()
    source_sha256 = sha256_file(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if sha256_file(source) != source_sha256:
        raise ValueError("sampling-transition checkpoint changed while loading")
    if not isinstance(payload, Mapping):
        raise ValueError("sampling-transition checkpoint must contain a mapping")
    if payload.get("format_version") != 1 or payload.get("stage") != "action_teacher_lora_v1":
        raise ValueError("sampling-transition requires an action_teacher_lora_v1 format_version=1 checkpoint")
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("sampling-transition source must have a positive optimizer step")
    source_config = payload.get("config")
    source_hashes = payload.get("manifest_hashes")
    if not isinstance(source_config, Mapping) or not isinstance(source_hashes, Mapping):
        raise ValueError("sampling-transition requires serialized source configuration and manifest hashes")
    current_config = config.to_dict()
    for section in ("model", "optimizer"):
        if not isinstance(source_config.get(section), Mapping) or source_config[section] != current_config[section]:
            raise ValueError(f"sampling-transition {section} contract must remain identical")
    source_data = _canonical_data_contract(source_config.get("data"))
    current_data = _canonical_data_contract(current_config["data"])
    if source_data.get("data_factory") != LEGACY_ACTION_FACTORY or current_data.get("data_factory") != RESAMPLED_ACTION_FACTORY:
        raise ValueError("sampling-transition permits only the exact legacy-to-resampled factory pair")
    source_data_contract = {key: value for key, value in source_data.items() if key != "data_factory"}
    current_data_contract = {key: value for key, value in current_data.items() if key != "data_factory"}
    if source_data_contract != current_data_contract:
        raise ValueError("sampling-transition data contract differs beyond the factory")
    for key in SHARED_DATA_HASH_KEYS:
        expected = current_manifest_hashes.get(key)
        if not expected or source_hashes.get(key) != expected:
            raise ValueError(f"sampling-transition {key} mismatch")
    source_prompt = _prompt_contract(source_data, source_hashes)
    target_prompt = _prompt_contract(current_data, current_manifest_hashes)
    if source_prompt != target_prompt:
        raise ValueError("sampling-transition prompt contract/artifacts must remain identical")
    source_data_hashes = {key: value for key, value in source_hashes.items() if key != "training_config"}
    target_data_hashes = {key: value for key, value in current_manifest_hashes.items() if key != "training_config"}
    if source_data_hashes != target_data_hashes:
        raise ValueError("sampling-transition cannot change other artifact hashes")
    if not source_hashes.get("training_config") or not current_manifest_hashes.get("training_config"):
        raise ValueError("sampling-transition requires both original and target training-config hashes")
    if source_hashes["training_config"] == current_manifest_hashes["training_config"]:
        raise ValueError("sampling-transition must record the new configuration hash, not reuse the source hash")
    source_training = source_config.get("training")
    if not isinstance(source_training, Mapping):
        raise ValueError("sampling-transition requires serialized source training configuration")
    source_training_contract = {key: value for key, value in source_training.items() if key not in ("max_steps", "output_dir")}
    target_training_contract = {key: value for key, value in current_config["training"].items() if key not in ("max_steps", "output_dir")}
    if source_training_contract != target_training_contract:
        raise ValueError("sampling-transition training contract differs beyond max_steps/output_dir")
    source_output = source_training.get("output_dir")
    if not isinstance(source_output, str) or Path(source_output).resolve() == Path(config.training.output_dir).resolve():
        raise ValueError("sampling-transition requires a different fresh output directory")
    if payload.get("micro_batches_consumed") != step * config.training.gradient_accumulation_steps:
        raise ValueError("sampling-transition source micro-batch position is inconsistent")
    state = payload.get("trainable_model")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("sampling-transition checkpoint has no trainable model state")
    if any(not isinstance(name, str) or not torch.is_tensor(value) for name, value in state.items()):
        raise ValueError("sampling-transition trainable model must map parameter names to tensors")
    initialization = {
        "mode": "sampling_transition_weights_only",
        "path": str(source), "sha256": source_sha256,
        "source_stage": payload["stage"], "source_step": step,
        "source_config": dict(source_config),
        "source_manifest_hashes": dict(source_hashes),
        "target_manifest_hashes": dict(current_manifest_hashes),
        "source_initialization": payload.get("initialization"),
        "sampling_transition": {
            "version": 1, "source_factory": LEGACY_ACTION_FACTORY,
            "target_factory": RESAMPLED_ACTION_FACTORY,
            "target_sampling_namespace": "action_resampled_absolute_v1",
            "target_virtual_sample_count": config.training.max_steps * config.training.gradient_accumulation_steps * config.training.micro_batch_size,
            "target_absolute_sample_start": 0,
            "changed_data_fields": ["data_factory"],
            "model_contract_unchanged": True, "optimizer_config_unchanged": True,
            "prompt_contract_unchanged": True,
            "source_micro_batches_consumed": payload["micro_batches_consumed"],
            "source_position_restored": False,
        },
        "prompt_transition": {"changed": False, "source": source_prompt, "target": target_prompt},
        "optimizer_restored": False, "rng_restored": False,
        "start_step": 0, "micro_batches_consumed": 0, "seed": config.training.seed,
    }
    return dict(state), initialization


def launch(
    config: ActionTeacherConfig,
    config_path: str | Path,
    resume: str | None,
    initialize_from: str | None,
    *,
    confirmed_gpu_index: int,
    confirmed_gpu_uuid: str,
    confirmed_at_utc: str,
    allocation_profile: str,
    warm_start_from: str | None = None,
    sampling_transition_from: str | None = None,
    length_transition_from: str | None = None,
    length_transition_sha256: str | None = None,
    stop_after_step: int | None = None,
    profile_runtime: bool = False,
) -> None:
    launch_started = time.perf_counter()
    if sum(value is not None for value in (resume, initialize_from, warm_start_from, sampling_transition_from, length_transition_from)) > 1:
        raise ValueError("resume and checkpoint initialization modes are mutually exclusive")
    if bool(length_transition_from) != bool(length_transition_sha256):
        raise ValueError("length-transition path and exact SHA256 pin must be supplied together")
    _execution_limit(config.training.max_steps, stop_after_step)
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
    manifest_hashes = _manifest_hashes(config, config_path)
    gate_checkpoint = None
    if initialize_from is not None:
        gate_checkpoint = _load_gate_checkpoint(
            initialize_from,
            config=config,
            current_manifest_hashes=manifest_hashes,
        )
    warm_start_state = None
    initialization = (
        {
            "mode": "continue_completed_gate20",
            "path": str(Path(initialize_from).resolve()),
            "sha256": sha256_file(initialize_from),
            "step": GATE_COMPLETION_STEP,
        }
        if initialize_from is not None
        else {"mode": "same_run_resume" if resume is not None else "fresh_base"}
    )
    if warm_start_from is not None:
        warm_start_state, initialization = _load_warm_start_checkpoint(
            warm_start_from, config=config, current_manifest_hashes=manifest_hashes,
        )
    if sampling_transition_from is not None:
        warm_start_state, initialization = _load_sampling_transition_checkpoint(
            sampling_transition_from, config=config, current_manifest_hashes=manifest_hashes,
        )
    if length_transition_from is not None:
        warm_start_state, initialization = _load_length_transition_checkpoint(
            length_transition_from, expected_sha256=length_transition_sha256,
            config=config, current_manifest_hashes=manifest_hashes,
        )
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
        "git_revision": git_revision(Path(__file__).parent),
        "config": config.to_dict(),
        "manifest_hashes": manifest_hashes,
        "initialization": initialization,
        "seed": config.training.seed,
        "device": torch.cuda.get_device_name(device),
        "gpu_state": {
            "prelaunch_snapshot": gpu_snapshot.as_dict(),
            "allocated_bytes_at_start": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes_at_start": int(torch.cuda.memory_reserved(device)),
        },
        "conditioning_modules": "precomputed; VAE and T5 are frozen and not loaded into the training process",
        "started_unix": time.time(),
        "execution": {"stop_after_step": stop_after_step, "configured_max_steps": config.training.max_steps},
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    profiler = OptimizerStepProfiler(
        device, warmup_steps=5, output_dir=output_dir / "profile_output", gpu_uuid=confirmed_gpu_uuid,
        enabled=profile_runtime or config.data.num_frames == 97,
    )
    try:
        _execute_training(
            config=config, device=device, output_dir=output_dir, manifest_hashes=manifest_hashes,
            metadata=metadata, initialization=initialization, resume=resume,
            gate_checkpoint=gate_checkpoint, warm_start_state=warm_start_state,
            stop_after_step=stop_after_step, profiler=profiler, launch_started=launch_started,
        )
    except BaseException as exc:
        profiler.close(error=exc)
        failure = {
            "status": "failed", "failure_type": type(exc).__name__, "message": str(exc),
            "timestamp_unix": time.time(), "execution": metadata["execution"],
            "profile": profiler.summary(),
        }
        (output_dir / "failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
        raise
    finally:
        profiler.close()


def _execute_training(
    *, config: ActionTeacherConfig, device: torch.device, output_dir: Path,
    manifest_hashes: dict[str, str], metadata: dict[str, Any], initialization: Mapping[str, Any],
    resume: str | None, gate_checkpoint: Mapping[str, Any] | None,
    warm_start_state: Mapping[str, torch.Tensor] | None, stop_after_step: int | None,
    profiler: OptimizerStepProfiler, launch_started: float,
) -> None:
    with profiler.phase("model_load"):
        model, summary = _build_model(config, device)
    with profiler.phase("optimizer_build"):
        optimizer = _build_optimizer(model, config)
    with profiler.phase("data_loader"):
        factory = _import_factory(config.data.data_factory)
        loader: Iterable[Mapping[str, Any]] = factory(config=config.data, training=config.training)
    start_step = 0
    micro_batches_consumed = 0
    restore_rng_checkpoint = None
    profiler.set_phase("restore")
    if resume:
        checkpoint = load_checkpoint(
            resume,
            expected_manifest_hashes=manifest_hashes,
            map_location="cpu",
        )
        if config.data.num_frames == 97:
            if (checkpoint.get("format_version"), checkpoint.get("stage")) != (1, "action_teacher_lora_v1"):
                raise ValueError("97-frame strict resume requires the original action-teacher checkpoint format")
            if checkpoint.get("config") != config.to_dict():
                raise ValueError("97-frame strict resume serialized configuration differs")
            if type(checkpoint.get("step")) is not int or checkpoint["step"] <= 0:
                raise ValueError("97-frame strict resume optimizer step must be positive")
            if type(checkpoint.get("micro_batches_consumed")) is not int:
                raise ValueError("97-frame strict resume requires an explicit integer data position")
        # Preserve a repaired run's original weight lineage across strict resumes.
        initialization = checkpoint.get("initialization", initialization)
        if not isinstance(initialization, Mapping):
            raise ValueError("resume checkpoint initialization provenance must be a mapping")
        metadata["initialization"] = dict(initialization)
        metadata["resumed_from"] = {
            "path": str(Path(resume).resolve()),
            "sha256": sha256_file(resume),
            "step": int(checkpoint["step"]),
        }
        (output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        load_trainable_state_dict(model, checkpoint["trainable_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng_checkpoint = checkpoint
        start_step = int(checkpoint["step"])
        micro_batches_consumed = int(
            checkpoint.get(
                "micro_batches_consumed",
                start_step * config.training.gradient_accumulation_steps,
            )
        )
        if micro_batches_consumed != start_step * config.training.gradient_accumulation_steps:
            raise ValueError("checkpoint micro-batch position is inconsistent with optimizer step")
    elif gate_checkpoint is not None:
        load_trainable_state_dict(model, gate_checkpoint["trainable_model"])
        optimizer.load_state_dict(gate_checkpoint["optimizer"])
        restore_rng_checkpoint = gate_checkpoint
        start_step = GATE_COMPLETION_STEP
        micro_batches_consumed = int(gate_checkpoint["micro_batches_consumed"])
    elif warm_start_state is not None:
        load_trainable_state_dict(model, warm_start_state)
        del warm_start_state
    execution_limit = _execution_limit(config.training.max_steps, stop_after_step, start_step)
    checkpoint_manager = CheckpointManager(output_dir, keep_last=config.training.keep_last)

    print(
        json.dumps(
            {
                "status": "launching",
                "trainable_parameters": summary.trainable_parameters,
                "total_parameters": summary.total_parameters,
                "lora_layers": len(summary.replaced_linear_layers),
                "resume_step": start_step,
                "execution_limit": execution_limit,
                "configured_max_steps": config.training.max_steps,
            },
            sort_keys=True,
        )
    )
    model.zero_grad(set_to_none=True)
    with profiler.phase("iterator_restore"):
        _, iterator = _iterator_at_micro_batch(loader, micro_batches_consumed)
        # Loader construction/skipping may consume global RNG. Restore only
        # after reaching the saved position, before the next real micro-batch.
        if restore_rng_checkpoint is not None:
            _restore_rng_state(restore_rng_checkpoint, require_complete=config.data.num_frames == 97)
            del restore_rng_checkpoint
    metadata["startup_seconds"] = time.perf_counter() - launch_started
    metadata["execution"]["resume_step"] = start_step
    metadata["execution"]["actual_stop_limit"] = execution_limit
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    tracker = ThroughputTracker.start()
    accumulated_loss = 0.0
    micro_step = micro_batches_consumed
    global_step = start_step
    last_saved_step = start_step
    while global_step < execution_limit:
        if micro_step % config.training.gradient_accumulation_steps == 0:
            profiler.begin_step(global_step + 1)
        profiler.set_phase("data_wait")
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise RuntimeError("data factory returned an empty iterable") from exc
        profiler.set_phase("forward")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = _forward_loss(model, batch, config, device)
            scaled_loss = loss / config.training.gradient_accumulation_steps
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError(f"non-finite loss at optimizer step {global_step + 1}")
        profiler.set_phase("backward")
        scaled_loss.backward()
        accumulated_loss += float(loss.detach())
        micro_step += 1
        tracker.update(config.training.micro_batch_size)
        if micro_step % config.training.gradient_accumulation_steps:
            continue

        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        profiler.set_phase("gradient_clip")
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, config.optimizer.max_grad_norm)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(f"non-finite gradient at optimizer step {global_step + 1}")
        profiler.set_phase("optimizer_update")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        step_profile = profiler.end_step(global_step)
        mean_loss = accumulated_loss / config.training.gradient_accumulation_steps
        accumulated_loss = 0.0
        metrics = {
            "step": global_step,
            "loss": mean_loss,
            "grad_norm": float(grad_norm),
            "samples_per_second": tracker.samples_per_second,
            "peak_vram_bytes": peak_vram_bytes(device),
            "elapsed_seconds": tracker.elapsed_seconds,
            "timestamp_unix": time.time(),
            "step_profile": step_profile,
        }
        if global_step % config.training.log_every == 0:
            append_jsonl(output_dir / "metrics.jsonl", metrics)
            print(json.dumps(metrics, sort_keys=True))
        if global_step % config.training.checkpoint_every == 0:
            profiler.begin_checkpoint(global_step)
            saved_path = checkpoint_manager.save(
                _checkpoint_payload(
                    model,
                    optimizer,
                    config=config,
                    step=global_step,
                    micro_batches_consumed=micro_step,
                    metrics=metrics,
                    manifest_hashes=manifest_hashes,
                    initialization=initialization,
                ),
                step=global_step,
                metric=mean_loss,
            )
            profiler.end_checkpoint(global_step, checkpoint_path=str(saved_path))
            last_saved_step = global_step

    if global_step != last_saved_step:
        profiler.begin_checkpoint(global_step)
        saved_path = checkpoint_manager.save(
            _checkpoint_payload(
                model,
                optimizer,
                config=config,
                step=global_step,
                micro_batches_consumed=micro_step,
                metrics=metrics,
                manifest_hashes=manifest_hashes,
                initialization=initialization,
            ),
            step=global_step,
            metric=mean_loss,
        )
        profiler.end_checkpoint(global_step, checkpoint_path=str(saved_path))
    completion = {
        "status": "completed_execution_segment", "step": global_step,
        "micro_batches_consumed": micro_step, "configured_max_steps": config.training.max_steps,
        "execution_stop_after_step": stop_after_step,
        "stop_reason": "configured_max_steps" if global_step == config.training.max_steps else "execution_step_limit",
        "checkpoint_path": str(output_dir / "checkpoints" / f"step-{global_step:07d}.pt"),
        "startup_seconds": metadata["startup_seconds"],
        "profile": profiler.summary(), "timestamp_unix": time.time(),
    }
    (output_dir / "execution_result.json").write_text(json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8")
    append_jsonl(output_dir / "execution_segments.jsonl", completion)
    print(json.dumps(completion, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if not args.launch:
        report = cpu_dry_run(config, args.config) if args.dry_run else validation_report(config, args.config)
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
        raise ValueError(f"--launch requires fresh GPU authorization fields: {missing}")
    launch(
        config,
        args.config,
        args.resume,
        args.initialize_from,
        confirmed_gpu_index=args.confirmed_gpu_index,
        confirmed_gpu_uuid=args.confirmed_gpu_uuid,
        confirmed_at_utc=args.confirmed_at_utc,
        allocation_profile=args.allocation_profile,
        warm_start_from=args.warm_start_from,
        sampling_transition_from=args.sampling_transition_from,
        length_transition_from=args.length_transition_from,
        length_transition_sha256=args.length_transition_sha256,
        stop_after_step=args.stop_after_step,
        profile_runtime=args.profile_runtime,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
