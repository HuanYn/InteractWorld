"""Contracts and lineage helpers for causal teacher-forcing training.

This module deliberately stays importable on a CPU-only machine.  The real
Wan wrapper is imported by the launch entry point only after the explicit GPU
authorization gate has passed.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from training.models.action_adapter import (
    ACTION_DIM,
    CANONICAL_ACTION_KEYS,
    RGB_FRAMES_PER_ACTION_TOKEN,
    build_action_context,
    validate_action_scale,
)
from training.models.lora import (
    TrainableSummary,
    configure_action_teacher,
    load_trainable_state_dict,
)
from training.runtime import collect_manifest_hashes, sha256_file
from training.paths import is_pinned_base_model

STAGE_NAME = "causal_teacher_forcing_v1"
PARENT_STAGE_NAME = "action_teacher_lora_v1"
PINNED_BASE_MODEL = (
    "/path/to/interactworld/models/"
    "Wan2.2-TI2V-5B@921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
)
SHARED_DATA_HASH_KEYS = ("dataset_manifest", "feature_index", "feature_receipt")


@dataclass
class CausalModelConfig:
    base_model_path: str = PINNED_BASE_MODEL
    model_type: str = "ci2v"
    timestep_shift: float = 5.0
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    action_dim: int = ACTION_DIM
    action_scale: float = 1.0
    gradient_checkpointing: bool = True
    latent_channels: int = 48
    temporal_compression: int = 4
    spatial_compression: int = 16
    downscale_factor_control_adapter: int = 16
    num_frame_per_block: int = 3
    independent_first_frame: bool = True
    local_attn_size: int = -1
    teacher_aug_t: float = 0.0


@dataclass
class CausalDataConfig:
    manifest_path: str = "/path/to/interactworld/data/manifests/train.jsonl"
    manifest_sha256: str | None = None
    feature_index_path: str = (
        "/path/to/interactworld/data/features/train.features.jsonl"
    )
    feature_receipt_path: str = (
        "/path/to/interactworld/data/features/train.features.jsonl.receipt.json"
    )
    data_factory: str = "training.data.action_dataset:build_action_teacher_dataloader"
    precomputed_latents: bool = True
    precomputed_text_embeddings: bool = True
    canonical_action_keys: tuple[str, ...] = CANONICAL_ACTION_KEYS
    rgb_frames_per_action_token: int = RGB_FRAMES_PER_ACTION_TOKEN
    height: int = 480
    width: int = 832
    num_frames: int = 49
    num_workers: int = 4


@dataclass
class TeacherLineageConfig:
    checkpoint_path: str = (
        "/path/to/interactworld/runs/action_teacher_lora_v1/checkpoints/best.pt"
    )
    checkpoint_sha256: str | None = None
    expected_stage: str = PARENT_STAGE_NAME


@dataclass
class CausalOptimizerConfig:
    name: str = "adamw8bit"
    adapter_lr: float = 5e-5
    lora_lr: float = 2e-5
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0


@dataclass
class CausalTrainingConfig:
    precision: str = "bf16"
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_steps: int = 1000
    checkpoint_every: int = 50
    keep_last: int = 2
    keep_best: int = 1
    seed: int = 42
    output_dir: str = "/path/to/interactworld/runs/causal_teacher_forcing_v1"
    log_every: int = 1


@dataclass
class CausalTeacherForcingConfig:
    model: CausalModelConfig = field(default_factory=CausalModelConfig)
    data: CausalDataConfig = field(default_factory=CausalDataConfig)
    lineage: TeacherLineageConfig = field(default_factory=TeacherLineageConfig)
    optimizer: CausalOptimizerConfig = field(default_factory=CausalOptimizerConfig)
    training: CausalTrainingConfig = field(default_factory=CausalTrainingConfig)

    def validate(self) -> None:
        errors: list[str] = []
        try:
            validate_action_scale(self.model.action_scale)
        except ValueError as exc:
            errors.append(str(exc))
        if not is_pinned_base_model(self.model.base_model_path):
            errors.append("base_model_path must be the pinned Wan2.2-TI2V-5B revision")
        if self.model.model_type != "ci2v":
            errors.append("model_type must be ci2v so the official action adapter is materialized")
        if self.model.timestep_shift != 5.0:
            errors.append("causal v1 requires the same timestep shift (5.0) as the teacher")
        if self.model.lora_rank != 16:
            errors.append("causal v1 fixes LoRA rank to 16 for teacher-checkpoint compatibility")
        if self.model.action_dim != ACTION_DIM:
            errors.append("action_dim must be 32 (8 canonical keys * 4 RGB frames)")
        if self.model.downscale_factor_control_adapter != 16:
            errors.append("causal v1 action adapter downscale must be 16")
        if self.model.num_frame_per_block != 3:
            errors.append("causal v1 requires exactly 3 latent frames per block")
        if not self.model.independent_first_frame:
            errors.append("causal v1 requires an independent clean first latent frame")
        if self.model.local_attn_size != -1:
            errors.append("causal v1 uses full completed-block history (local_attn_size=-1)")
        if self.model.teacher_aug_t != 0.0:
            errors.append("clean teacher-forcing latents must use aug_t=0")
        if tuple(self.data.canonical_action_keys) != CANONICAL_ACTION_KEYS:
            errors.append(f"canonical_action_keys must be exactly {CANONICAL_ACTION_KEYS}")
        if self.data.rgb_frames_per_action_token != RGB_FRAMES_PER_ACTION_TOKEN:
            errors.append("rgb_frames_per_action_token must be 4")
        if not self.data.precomputed_latents or not self.data.precomputed_text_embeddings:
            errors.append("causal training requires precomputed VAE and text features")
        expected_index = (
            Path(self.data.manifest_path).parent.parent
            / "features"
            / f"{Path(self.data.manifest_path).stem}.features.jsonl"
        )
        if Path(self.data.feature_index_path) != expected_index:
            errors.append(
                "feature_index_path must match the index consumed by the configured v1 data factory"
            )
        if Path(self.data.feature_receipt_path) != expected_index.with_suffix(
            expected_index.suffix + ".receipt.json"
        ):
            errors.append("feature_receipt_path must bind the consumed feature index receipt")
        if (self.data.num_frames, self.data.height, self.data.width) != (49, 480, 832):
            errors.append("causal v1 is fixed to 49x480x832 RGB windows")
        latent_frames = 1 + (self.data.num_frames - 1) // self.model.temporal_compression
        latent_height = self.data.height // self.model.spatial_compression
        latent_width = self.data.width // self.model.spatial_compression
        if (latent_frames, self.model.latent_channels, latent_height, latent_width) != (
            13,
            48,
            30,
            52,
        ):
            errors.append("49x480x832 RGB must map to latent [13,48,30,52]")
        if (latent_frames - 1) % self.model.num_frame_per_block:
            errors.append("the 12 future latent frames must form complete 3-frame blocks")
        if self.lineage.expected_stage != PARENT_STAGE_NAME:
            errors.append(f"expected parent stage must be {PARENT_STAGE_NAME!r}")
        if not self.lineage.checkpoint_path:
            errors.append("a stage-1 action-teacher checkpoint is required")
        if self.training.precision != "bf16":
            errors.append("causal v1 requires bf16")
        if self.training.micro_batch_size != 1:
            errors.append("micro_batch_size must be 1")
        if self.training.gradient_accumulation_steps != 8:
            errors.append("gradient_accumulation_steps must be 8")
        if not 20 <= self.training.checkpoint_every <= 50:
            errors.append("checkpoint_every must be between 20 and 50")
        if (self.training.keep_best, self.training.keep_last) != (1, 2):
            errors.append("checkpoint retention must be best 1 + last 2")
        if self.training.max_steps <= 0:
            errors.append("max_steps must be positive")
        if self.training.seed != 42:
            errors.append("causal v1 reproducibility seed is fixed to 42")
        if errors:
            raise ValueError("invalid causal teacher-forcing configuration:\n- " + "\n- ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tuple_fields(raw: dict[str, Any]) -> dict[str, Any]:
    raw = dict(raw)
    data = dict(raw.get("data", {}))
    if "canonical_action_keys" in data:
        data["canonical_action_keys"] = tuple(data["canonical_action_keys"])
    raw["data"] = data
    optimizer = dict(raw.get("optimizer", {}))
    if "betas" in optimizer:
        optimizer["betas"] = tuple(optimizer["betas"])
    raw["optimizer"] = optimizer
    return raw


def load_causal_config(path: str | Path) -> CausalTeacherForcingConfig:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - declared project dependency
        raise RuntimeError("PyYAML is required to load the training configuration") from exc
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    allowed = {"model", "data", "lineage", "optimizer", "training"}
    unknown = set(raw).difference(allowed)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    raw = _tuple_fields(raw)
    config = CausalTeacherForcingConfig(
        model=CausalModelConfig(**raw.get("model", {})),
        data=CausalDataConfig(**raw.get("data", {})),
        lineage=TeacherLineageConfig(**raw.get("lineage", {})),
        optimizer=CausalOptimizerConfig(**raw.get("optimizer", {})),
        training=CausalTrainingConfig(**raw.get("training", {})),
    )
    config.validate()
    return config


def latent_shape(config: CausalTeacherForcingConfig) -> tuple[int, int, int, int, int]:
    frames = 1 + (config.data.num_frames - 1) // config.model.temporal_compression
    return (
        config.training.micro_batch_size,
        frames,
        config.model.latent_channels,
        config.data.height // config.model.spatial_compression,
        config.data.width // config.model.spatial_compression,
    )


def frame_blocks(
    num_frames: int,
    *,
    frames_per_block: int,
    independent_first_frame: bool,
) -> tuple[tuple[int, int], ...]:
    """Return the same frame grouping used by ``CausalWanModel``."""

    if num_frames <= 0 or frames_per_block <= 0:
        raise ValueError("num_frames and frames_per_block must be positive")
    blocks: list[tuple[int, int]] = []
    start = 0
    if independent_first_frame:
        blocks.append((0, 1))
        start = 1
    while start < num_frames:
        end = min(start + frames_per_block, num_frames)
        blocks.append((start, end))
        start = end
    return tuple(blocks)


def teacher_forcing_visibility(
    num_frames: int = 13,
    *,
    tokens_per_frame: int = 1,
    frames_per_block: int = 3,
    independent_first_frame: bool = True,
) -> torch.Tensor:
    """Materialize the official clean/noisy teacher-forcing mask contract.

    The result is for tests and audits, not the 5B training forward.  Clean
    queries can see only their current-or-earlier clean block.  A noisy query
    can see the clean *completed* blocks and its own noisy block, never the
    clean target from its current/future block or a future noisy block.
    """

    if tokens_per_frame <= 0:
        raise ValueError("tokens_per_frame must be positive")
    tokens = num_frames * tokens_per_frame
    visible = torch.zeros((tokens * 2, tokens * 2), dtype=torch.bool)
    for frame_start, frame_end in frame_blocks(
        num_frames,
        frames_per_block=frames_per_block,
        independent_first_frame=independent_first_frame,
    ):
        start = frame_start * tokens_per_frame
        end = frame_end * tokens_per_frame
        visible[start:end, :end] = True
        noisy_start = tokens + start
        noisy_end = tokens + end
        visible[noisy_start:noisy_end, :start] = True
        visible[noisy_start:noisy_end, noisy_start:noisy_end] = True
    visible.fill_diagonal_(True)
    return visible


def assert_no_future_leakage(
    visibility: torch.Tensor,
    *,
    num_frames: int = 13,
    tokens_per_frame: int = 1,
    frames_per_block: int = 3,
    independent_first_frame: bool = True,
) -> None:
    expected = teacher_forcing_visibility(
        num_frames,
        tokens_per_frame=tokens_per_frame,
        frames_per_block=frames_per_block,
        independent_first_frame=independent_first_frame,
    )
    if visibility.dtype != torch.bool or visibility.shape != expected.shape:
        raise ValueError(
            f"teacher-forcing visibility must be bool {tuple(expected.shape)}, "
            f"got {visibility.dtype} {tuple(visibility.shape)}"
        )
    leaks = visibility & ~expected
    if leaks.any():
        query, key = leaks.nonzero(as_tuple=False)[0].tolist()
        raise ValueError(f"teacher-forcing mask leaks future token: query={query}, key={key}")
    missing = expected & ~visibility
    if missing.any():
        query, key = missing.nonzero(as_tuple=False)[0].tolist()
        raise ValueError(f"teacher-forcing mask misses required token: query={query}, key={key}")


def artifact_hashes(
    config: CausalTeacherForcingConfig,
    config_path: str | Path,
) -> dict[str, str]:
    hashes = collect_manifest_hashes(
        {
            "dataset_manifest": config.data.manifest_path,
            "feature_index": config.data.feature_index_path,
            "feature_receipt": config.data.feature_receipt_path,
            "training_config": config_path,
            "teacher_checkpoint": config.lineage.checkpoint_path,
        }
    )
    if config.data.manifest_sha256 is not None:
        if hashes["dataset_manifest"] != config.data.manifest_sha256:
            raise ValueError(
                "dataset manifest hash mismatch: "
                f"expected {config.data.manifest_sha256}, got {hashes['dataset_manifest']}"
            )
    if config.lineage.checkpoint_sha256 is not None:
        if hashes["teacher_checkpoint"] != config.lineage.checkpoint_sha256:
            raise ValueError(
                "teacher checkpoint hash mismatch: "
                f"expected {config.lineage.checkpoint_sha256}, "
                f"got {hashes['teacher_checkpoint']}"
            )
    return hashes


@dataclass(frozen=True)
class TeacherCheckpointLineage:
    path: str
    sha256: str
    stage: str
    step: int
    source_manifest_hashes: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_named_teacher_parameter(name: str) -> bool:
    return (
        "act_control_adapter." in name
        or ".lora_a." in name
        or ".lora_b." in name
    )


def load_teacher_checkpoint(
    config: CausalTeacherForcingConfig,
    hashes: Mapping[str, str],
) -> tuple[dict[str, Any], TeacherCheckpointLineage]:
    """Load and validate the exact stage-1 parent without remapping names."""

    path = Path(config.lineage.checkpoint_path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0 compatibility
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("teacher checkpoint must contain a mapping")
    if payload.get("format_version") != 1:
        raise ValueError("teacher checkpoint format_version must be 1")
    stage = payload.get("stage")
    if stage != config.lineage.expected_stage:
        raise ValueError(
            f"teacher checkpoint stage must be {config.lineage.expected_stage!r}, got {stage!r}"
        )
    step = payload.get("step")
    if not isinstance(step, int) or step <= 0:
        raise ValueError("teacher checkpoint must come from a completed positive optimizer step")
    state = payload.get("trainable_model")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("teacher checkpoint has no named trainable_model state")
    invalid = sorted(name for name in state if not _is_named_teacher_parameter(str(name)))
    if invalid:
        raise ValueError(f"teacher checkpoint contains non-LoRA/adapter tensors: {invalid[:8]}")
    names = tuple(str(name) for name in state)
    if not any("act_control_adapter." in name for name in names):
        raise ValueError("teacher checkpoint has no action-adapter tensors")
    if not any(".lora_a." in name for name in names) or not any(".lora_b." in name for name in names):
        raise ValueError("teacher checkpoint has incomplete named LoRA tensors")

    source_hashes = payload.get("manifest_hashes")
    if not isinstance(source_hashes, dict):
        raise ValueError("teacher checkpoint has no source manifest hashes")
    for key in SHARED_DATA_HASH_KEYS:
        if source_hashes.get(key) != hashes.get(key):
            raise ValueError(
                f"teacher checkpoint {key} lineage mismatch: "
                f"teacher={source_hashes.get(key)!r}, current={hashes.get(key)!r}"
            )
    source_config = payload.get("config")
    if not isinstance(source_config, dict):
        raise ValueError("teacher checkpoint has no serialized configuration")
    source_model = source_config.get("model")
    if not isinstance(source_model, dict):
        raise ValueError("teacher checkpoint has no serialized model configuration")
    if source_model.get("base_model_path") != config.model.base_model_path:
        raise ValueError("teacher and causal stages do not use the same pinned base model")
    inherited_contract = {
        "model_type": config.model.model_type,
        "lora_rank": config.model.lora_rank,
        "lora_alpha": config.model.lora_alpha,
        "lora_dropout": config.model.lora_dropout,
        "action_dim": config.model.action_dim,
    }
    for key, expected in inherited_contract.items():
        if source_model.get(key) != expected:
            raise ValueError(
                f"teacher and causal stages disagree on inherited model field {key}: "
                f"teacher={source_model.get(key)!r}, current={expected!r}"
            )

    actual_hash = sha256_file(path)
    if actual_hash != hashes.get("teacher_checkpoint"):
        raise ValueError("teacher checkpoint changed after artifact hashes were collected")
    lineage = TeacherCheckpointLineage(
        path=str(path.resolve()),
        sha256=actual_hash,
        stage=str(stage),
        step=step,
        source_manifest_hashes={str(key): str(value) for key, value in source_hashes.items()},
    )
    return payload, lineage


def verify_official_causal_interface(wrapper: nn.Module) -> None:
    """Fail before training if the checked upstream causal interface changed."""

    if getattr(wrapper, "uniform_timestep", None) is not False:
        raise RuntimeError("Wan wrapper is not in causal/per-frame-timestep mode")
    backbone = getattr(wrapper, "model", None)
    if backbone is None or backbone.__class__.__name__ != "CausalWanModel":
        actual = type(backbone).__name__ if backbone is not None else None
        raise RuntimeError(f"is_causal=True did not construct CausalWanModel (got {actual!r})")
    for method in (
        "_forward_train",
        "_prepare_teacher_forcing_mask",
        "_maybe_build_block_mask",
    ):
        if not callable(getattr(backbone, method, None)):
            raise RuntimeError(f"CausalWanModel is missing required upstream method {method}")
    wrapper_params = inspect.signature(wrapper.forward).parameters
    for parameter in ("clean_x", "aug_t"):
        if parameter not in wrapper_params:
            raise RuntimeError(f"Wan wrapper forward no longer accepts {parameter}")
    train_params = inspect.signature(backbone._forward_train).parameters
    for parameter in ("clean_x", "aug_t", "act_context"):
        if parameter not in train_params:
            raise RuntimeError(f"CausalWanModel._forward_train no longer accepts {parameter}")


def construct_causal_wrapper(config: CausalTeacherForcingConfig, wrapper_factory):
    """Construct the checked upstream wrapper with causal mode explicit."""

    wrapper = wrapper_factory(
        model_name=config.model.base_model_path,
        timestep_shift=config.model.timestep_shift,
        is_causal=True,
        local_attn_size=config.model.local_attn_size,
        model_type=config.model.model_type,
        num_frame_per_block=config.model.num_frame_per_block,
        downscale_factor_control_adapter=config.model.downscale_factor_control_adapter,
    )
    backbone = getattr(wrapper, "model", None)
    if backbone is None:
        raise RuntimeError("Wan wrapper did not expose its backbone as .model")
    backbone.num_frame_per_block = config.model.num_frame_per_block
    # CausalWanModel initializes this to False even when block size is passed.
    backbone.independent_first_frame = config.model.independent_first_frame
    verify_official_causal_interface(wrapper)
    return wrapper


def initialize_from_action_teacher(
    wrapper: nn.Module,
    teacher_payload: Mapping[str, Any],
    config: CausalTeacherForcingConfig,
) -> TrainableSummary:
    """Freeze the base, inject matching LoRA modules, and load names exactly."""

    summary = configure_action_teacher(
        wrapper.model,
        rank=config.model.lora_rank,
        alpha=config.model.lora_alpha,
        dropout=config.model.lora_dropout,
    )
    state = teacher_payload["trainable_model"]
    expected = {name for name, parameter in wrapper.named_parameters() if parameter.requires_grad}
    actual = set(state)
    if expected != actual:
        missing = sorted(expected.difference(actual))
        unexpected = sorted(actual.difference(expected))
        raise ValueError(
            "named teacher state is not architecture-compatible with CausalWanModel: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    if any(not _is_named_teacher_parameter(name) for name in expected):
        raise RuntimeError("causal setup unexpectedly enabled a frozen base-model parameter")
    load_trainable_state_dict(wrapper, dict(state))
    return summary


def clean_latents_from_batch(
    batch: Mapping[str, Any],
    config: CausalTeacherForcingConfig,
) -> torch.Tensor:
    """Return clean latents, exactly reconstructing them for the v1 cache format."""

    noisy = batch["noisy_latents"].float()
    target = batch["target_flow"].float()
    timesteps = batch["timesteps"].float()
    if "clean_latents" in batch:
        clean = batch["clean_latents"].float()
    else:
        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(0)
        sigma = (timesteps / 1000.0).view(timesteps.shape[0], timesteps.shape[1], 1, 1, 1)
        # The feature dataset uses x_t = x_0 + sigma * (noise - x_0).
        clean = noisy - sigma * target
    expected = latent_shape(config)
    if tuple(clean.shape) != expected:
        raise ValueError(f"clean_latents must be {expected}, got {tuple(clean.shape)}")
    return clean


def validate_causal_batch(
    batch: Mapping[str, Any],
    config: CausalTeacherForcingConfig,
) -> torch.Tensor:
    required = {"noisy_latents", "target_flow", "timesteps", "prompt_embeds", "actions"}
    missing = required.difference(batch)
    if missing:
        raise ValueError(f"training batch is missing keys: {sorted(missing)}")
    expected = latent_shape(config)
    noisy = batch["noisy_latents"]
    target = batch["target_flow"]
    if tuple(noisy.shape) != expected or tuple(target.shape) != expected:
        raise ValueError(
            f"noisy_latents/target_flow must both be {expected} (B,F,C,H,W), "
            f"got {tuple(noisy.shape)} and {tuple(target.shape)}"
        )
    expected_actions = (config.training.micro_batch_size, config.data.num_frames - 1, 8)
    if tuple(batch["actions"].shape) != expected_actions:
        raise ValueError(
            f"actions must be {expected_actions}, got {tuple(batch['actions'].shape)}"
        )
    timesteps = batch["timesteps"]
    if timesteps.ndim == 1:
        timesteps = timesteps.unsqueeze(0)
    if tuple(timesteps.shape) != expected[:2]:
        raise ValueError(f"timesteps must be {expected[:2]}, got {tuple(timesteps.shape)}")
    clean = clean_latents_from_batch(batch, config)
    if not torch.allclose(timesteps[:, 0].float(), torch.zeros_like(timesteps[:, 0].float())):
        raise ValueError("the independent first latent frame must have timestep zero")
    if torch.count_nonzero(target[:, 0]).item() != 0:
        raise ValueError("the independent first latent frame must have zero flow target")
    if not torch.allclose(noisy[:, 0].float(), clean[:, 0], atol=1e-6, rtol=0):
        raise ValueError("the independent first latent frame must remain clean")
    return clean


def _move(value: Any, device: torch.device, *, dtype: torch.dtype | None = None):
    if torch.is_tensor(value):
        if dtype is not None and value.is_floating_point():
            return value.to(device=device, dtype=dtype, non_blocking=True)
        return value.to(device=device, non_blocking=True)
    if isinstance(value, list):
        return [_move(item, device, dtype=dtype) for item in value]
    return value


def causal_teacher_forcing_loss(
    model: nn.Module,
    batch: Mapping[str, Any],
    config: CausalTeacherForcingConfig,
    device: torch.device,
) -> torch.Tensor:
    """Run the checked ``clean_x`` tuple/tensor flow-prediction contract."""

    clean_cpu = validate_causal_batch(batch, config)
    noisy = _move(batch["noisy_latents"], device, dtype=torch.bfloat16)
    target = _move(batch["target_flow"], device, dtype=torch.bfloat16)
    clean = _move(clean_cpu, device, dtype=torch.bfloat16)
    timestep = _move(batch["timesteps"], device)
    if timestep.ndim == 1:
        timestep = timestep.unsqueeze(0)
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
        clean_x=clean,
        aug_t=torch.full_like(timestep, config.model.teacher_aug_t),
        replace_first_timestep_and_noise_latents=True,
    )
    prediction = model_output[0] if isinstance(model_output, tuple) else model_output
    if tuple(prediction.shape) != tuple(target.shape):
        raise RuntimeError(
            f"causal Wan prediction shape {tuple(prediction.shape)} != target {tuple(target.shape)}"
        )
    # Latent frame zero is a clean condition, not a denoising target.
    return torch.nn.functional.mse_loss(prediction[:, 1:].float(), target[:, 1:].float())
