"""Stage-3 LongForcing-lite contracts and differentiable rollout core.

LongForcing-lite is a supervised endpoint-distillation recipe.  It is
deliberately **not DMD**: there is no discriminator, distribution-matching
gradient, or score-difference objective.  A causal student self-rolls with a
four-step Euler solver, while a frozen action-teacher supplies a stop-gradient
40-step endpoint for the final block.

The module is CPU importable and has no CUDA, network, or Wan imports.  The
real Wan bridge is an explicit factory contract in the launch entry point; an
unknown bridge therefore fails closed instead of guessing an upstream API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

import torch
from torch import nn

from training.models.action_adapter import ACTION_DIM, CANONICAL_ACTION_KEYS
from training.models.action_adapter import build_action_context, validate_action_scale
from training.models.lora import (
    TrainableSummary,
    configure_action_teacher,
    load_trainable_state_dict,
)
from training.runtime import collect_manifest_hashes, sha256_file
from training.paths import is_pinned_base_model

STAGE_NAME = "longforcing_lite_v1"
METHOD_NAME = "LongForcing-lite"
TEACHER_STAGE = "action_teacher_lora_v1"
CAUSAL_STAGE = "causal_teacher_forcing_v1"
PINNED_BASE_MODEL = (
    "/path/to/interactworld/models/"
    "Wan2.2-TI2V-5B@921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
)
SHARED_DATA_HASH_KEYS = ("dataset_manifest", "feature_index", "feature_receipt")


@dataclass
class LongForcingModelConfig:
    base_model_path: str = PINNED_BASE_MODEL
    model_type: str = "ci2v"
    timestep_shift: float = 5.0
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    action_dim: int = ACTION_DIM
    action_scale: float = 1.0
    latent_channels: int = 48
    temporal_compression: int = 4
    spatial_compression: int = 16
    num_frame_per_block: int = 3
    rgb_frames_per_latent: int = 4
    downscale_factor_control_adapter: int = 16
    local_attn_size: int = -1
    gradient_checkpointing: bool = True
    gradient_checkpointing_mode: str = "non_reentrant"


@dataclass
class LongForcingDataConfig:
    manifest_path: str = "/path/to/interactworld/data/manifests/train.jsonl"
    manifest_sha256: str | None = None
    feature_index_path: str = (
        "/path/to/interactworld/data/features/train.features.jsonl"
    )
    feature_receipt_path: str = (
        "/path/to/interactworld/data/features/train.features.jsonl.receipt.json"
    )
    long_feature_index_path: str = (
        "/path/to/interactworld/data/features/train.long241.features.jsonl"
    )
    long_feature_receipt_path: str = (
        "/path/to/interactworld/data/features/train.long241.features.jsonl.receipt.json"
    )
    data_factory: str | None = (
        "training.data.longforcing_dataset:build_longforcing_dataloader"
    )
    precomputed_latents: bool = True
    precomputed_text_embeddings: bool = True
    canonical_action_keys: tuple[str, ...] = CANONICAL_ACTION_KEYS
    height: int = 480
    width: int = 832
    fps: int = 16
    short_window_frames: int = 49
    demo_rollout_frames: int = 241
    num_workers: int = 4


@dataclass
class LongForcingLineageConfig:
    teacher_checkpoint_path: str = (
        "/path/to/interactworld/runs/action_teacher_lora_v1/checkpoints/best.pt"
    )
    teacher_checkpoint_sha256: str | None = None
    causal_checkpoint_path: str = (
        "/path/to/interactworld/runs/causal_teacher_forcing_v1/checkpoints/best.pt"
    )
    causal_checkpoint_sha256: str | None = None
    expected_teacher_stage: str = TEACHER_STAGE
    expected_causal_stage: str = CAUSAL_STAGE


@dataclass
class LongForcingRolloutConfig:
    backend_factory: str | None = "training.longforcing_lite:wan_longforcing_backend_factory"
    student_steps: int = 4
    teacher_steps: int = 40
    curriculum_depths: tuple[int, ...] = (1, 4, 8, 20)
    curriculum_start_steps: tuple[int, ...] = (0, 100, 300, 600)
    flowmatch_replay_fraction: float = 0.25
    only_last_block_backward: bool = True
    teacher_stop_gradient: bool = True
    teacher_cpu_offload: bool = True
    solver: str = "euler"
    is_dmd: bool = False


@dataclass
class LongForcingOptimizerConfig:
    name: str = "adamw8bit"
    adapter_lr: float = 2e-5
    lora_lr: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0


@dataclass
class LongForcingTrainingConfig:
    precision: str = "bf16"
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_steps: int = 1000
    checkpoint_every: int = 50
    keep_last: int = 2
    keep_best: int = 1
    seed: int = 42
    output_dir: str = "/path/to/interactworld/runs/longforcing_lite_v1"
    log_every: int = 1


@dataclass
class LongForcingConfig:
    model: LongForcingModelConfig = field(default_factory=LongForcingModelConfig)
    data: LongForcingDataConfig = field(default_factory=LongForcingDataConfig)
    lineage: LongForcingLineageConfig = field(default_factory=LongForcingLineageConfig)
    rollout: LongForcingRolloutConfig = field(default_factory=LongForcingRolloutConfig)
    optimizer: LongForcingOptimizerConfig = field(default_factory=LongForcingOptimizerConfig)
    training: LongForcingTrainingConfig = field(default_factory=LongForcingTrainingConfig)

    def validate(self) -> None:
        errors: list[str] = []
        try:
            validate_action_scale(self.model.action_scale)
        except ValueError as exc:
            errors.append(str(exc))
        if not is_pinned_base_model(self.model.base_model_path):
            errors.append("base_model_path must be the pinned Wan2.2-TI2V-5B revision")
        if self.model.model_type != "ci2v":
            errors.append("model_type must be ci2v")
        if self.model.timestep_shift != 5.0:
            errors.append("the student and endpoint teacher require timestep_shift=5.0")
        if self.model.lora_rank != 16 or self.model.action_dim != ACTION_DIM:
            errors.append("the causal student requires rank-16 LoRA and the 32-D action adapter")
        if self.model.num_frame_per_block != 3 or self.model.rgb_frames_per_latent != 4:
            errors.append("one rollout block must be 3 latent / 12 RGB frames")
        if self.model.downscale_factor_control_adapter != 16:
            errors.append("the action adapter downscale must be 16")
        if self.model.local_attn_size != -1:
            errors.append("v1 uses full attention inside each 49-frame sliding window")
        if self.model.gradient_checkpointing_mode not in ("non_reentrant", "reentrant"):
            errors.append("gradient_checkpointing_mode must be non_reentrant or reentrant")
        if tuple(self.data.canonical_action_keys) != CANONICAL_ACTION_KEYS:
            errors.append(f"canonical_action_keys must be exactly {CANONICAL_ACTION_KEYS}")
        if (self.data.height, self.data.width, self.data.fps) != (480, 832, 16):
            errors.append("v1 is fixed to 480x832 at 16 fps")
        if self.data.short_window_frames != rgb_frames_for_blocks(4, self):
            errors.append("the short training window must be 49 RGB frames / 4 blocks")
        if self.data.demo_rollout_frames != rgb_frames_for_blocks(20, self):
            errors.append("the demo rollout must be 241 RGB frames / exactly 15 seconds")
        expected_feature_index = (
            Path(self.data.manifest_path).parent.parent
            / "features"
            / f"{Path(self.data.manifest_path).stem}.features.jsonl"
        )
        if Path(self.data.feature_index_path) != expected_feature_index:
            errors.append("feature_index_path must match the selected manifest")
        if Path(self.data.feature_receipt_path) != expected_feature_index.with_suffix(
            expected_feature_index.suffix + ".receipt.json"
        ):
            errors.append("feature_receipt_path must bind the consumed feature index")
        expected_long_index = expected_feature_index.with_name(
            f"{Path(self.data.manifest_path).stem}.long241.features.jsonl"
        )
        if Path(self.data.long_feature_index_path) != expected_long_index:
            errors.append("long_feature_index_path must be the manifest's long241 cache")
        if Path(self.data.long_feature_receipt_path) != expected_long_index.with_suffix(
            expected_long_index.suffix + ".receipt.json"
        ):
            errors.append("long_feature_receipt_path must bind the long241 cache index")
        if not self.data.precomputed_latents or not self.data.precomputed_text_embeddings:
            errors.append("LongForcing-lite requires precomputed VAE and text features")
        if self.lineage.expected_teacher_stage != TEACHER_STAGE:
            errors.append(f"teacher parent stage must be {TEACHER_STAGE}")
        if self.lineage.expected_causal_stage != CAUSAL_STAGE:
            errors.append(f"causal parent stage must be {CAUSAL_STAGE}")
        if self.rollout.student_steps != 4 or self.rollout.teacher_steps != 40:
            errors.append("v1 requires a 4-step student and 40-step teacher endpoint")
        if tuple(self.rollout.curriculum_depths) != (1, 4, 8, 20):
            errors.append("rollout depth curriculum must be 1 -> 4 -> 8 -> 20 blocks")
        starts = tuple(self.rollout.curriculum_start_steps)
        if len(starts) != 4 or starts[0] != 0 or any(a >= b for a, b in zip(starts, starts[1:])):
            errors.append("curriculum_start_steps must be four increasing values beginning at zero")
        if self.rollout.flowmatch_replay_fraction != 0.25:
            errors.append("short-window FlowMatch replay fraction must be exactly 0.25")
        if not self.rollout.only_last_block_backward:
            errors.append("only the last rollout block may receive gradients")
        if not self.rollout.teacher_stop_gradient:
            errors.append("the 40-step teacher endpoint must be stop-gradient")
        if not self.rollout.teacher_cpu_offload:
            errors.append("v1 requires exclusive teacher CPU offload for a 32GB GPU")
        if self.rollout.solver != "euler":
            errors.append("the checked v1 rollout solver is Euler")
        if self.rollout.is_dmd:
            errors.append("LongForcing-lite is endpoint distillation, not DMD")
        if self.training.precision != "bf16" or self.training.micro_batch_size != 1:
            errors.append("v1 requires bf16 with micro_batch_size=1")
        if self.training.gradient_accumulation_steps != 8:
            errors.append("gradient_accumulation_steps must be 8")
        if self.training.max_steps <= 0 or self.training.seed != 42:
            errors.append("max_steps must be positive and seed is fixed to 42")
        if not 20 <= self.training.checkpoint_every <= 50:
            errors.append("checkpoint_every must be between 20 and 50")
        if (self.training.keep_best, self.training.keep_last) != (1, 2):
            errors.append("checkpoint retention must be best 1 + last 2")
        if errors:
            raise ValueError("invalid LongForcing-lite configuration:\n- " + "\n- ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _convert_tuples(raw: dict[str, Any]) -> dict[str, Any]:
    result = dict(raw)
    data = dict(result.get("data", {}))
    if "canonical_action_keys" in data:
        data["canonical_action_keys"] = tuple(data["canonical_action_keys"])
    result["data"] = data
    rollout = dict(result.get("rollout", {}))
    for key in ("curriculum_depths", "curriculum_start_steps"):
        if key in rollout:
            rollout[key] = tuple(rollout[key])
    result["rollout"] = rollout
    optimizer = dict(result.get("optimizer", {}))
    if "betas" in optimizer:
        optimizer["betas"] = tuple(optimizer["betas"])
    result["optimizer"] = optimizer
    return result


def load_longforcing_config(path: str | Path) -> LongForcingConfig:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load the training configuration") from exc
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    return longforcing_config_from_dict(raw)


def longforcing_config_from_dict(raw: Mapping[str, Any]) -> LongForcingConfig:
    """Use the same validated config for a saved checkpoint and training YAML."""
    allowed = {"model", "data", "lineage", "rollout", "optimizer", "training"}
    unknown = set(raw).difference(allowed)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    raw = _convert_tuples(raw)
    config = LongForcingConfig(
        model=LongForcingModelConfig(**raw.get("model", {})),
        data=LongForcingDataConfig(**raw.get("data", {})),
        lineage=LongForcingLineageConfig(**raw.get("lineage", {})),
        rollout=LongForcingRolloutConfig(**raw.get("rollout", {})),
        optimizer=LongForcingOptimizerConfig(**raw.get("optimizer", {})),
        training=LongForcingTrainingConfig(**raw.get("training", {})),
    )
    config.validate()
    return config


def rgb_frames_for_blocks(depth: int, config: LongForcingConfig) -> int:
    if depth <= 0:
        raise ValueError("rollout depth must be positive")
    return 1 + depth * config.model.num_frame_per_block * config.model.rgb_frames_per_latent


def curriculum_depth(optimizer_step: int, config: LongForcingConfig) -> int:
    if optimizer_step < 0:
        raise ValueError("optimizer_step cannot be negative")
    depth = config.rollout.curriculum_depths[0]
    for start, candidate in zip(
        config.rollout.curriculum_start_steps, config.rollout.curriculum_depths
    ):
        if optimizer_step < start:
            break
        depth = candidate
    return int(depth)


def is_flowmatch_replay(micro_batch_index: int) -> bool:
    """Deterministic one-in-four schedule, stable across exact resume."""

    if micro_batch_index < 0:
        raise ValueError("micro_batch_index cannot be negative")
    return micro_batch_index % 4 == 3


def euler_sigmas(
    steps: int,
    *,
    shift: float = 5.0,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("solver steps must be positive")
    from utils.scheduler import FlowMatchScheduler

    scheduler = FlowMatchScheduler(
        num_inference_steps=steps,
        shift=shift,
        sigma_min=0.0,
        extra_one_step=True,
    )
    return torch.cat(
        (
            scheduler.sigmas.to(device=device, dtype=dtype),
            torch.zeros(1, device=device, dtype=dtype),
        )
    )


class LongForcingBackend(Protocol):
    student: nn.Module
    teacher: nn.Module

    def student_velocity(self, **kwargs: Any) -> torch.Tensor: ...

    def teacher_velocity(self, **kwargs: Any) -> torch.Tensor: ...

    def flowmatch_replay_loss(self, short_window: Mapping[str, Any]) -> torch.Tensor: ...

    def activate_student(self, device: torch.device) -> None: ...

    def activate_teacher(self, device: torch.device) -> None: ...


def validate_backend(backend: Any) -> None:
    if not isinstance(getattr(backend, "student", None), nn.Module):
        raise TypeError("LongForcing backend must expose student: nn.Module")
    if not isinstance(getattr(backend, "teacher", None), nn.Module):
        raise TypeError("LongForcing backend must expose teacher: nn.Module")
    for name in (
        "student_velocity",
        "teacher_velocity",
        "flowmatch_replay_loss",
        "activate_student",
        "activate_teacher",
    ):
        if not callable(getattr(backend, name, None)):
            raise TypeError(f"LongForcing backend is missing callable {name}")


def _named_trainable_state(payload: Mapping[str, Any], *, stage: str) -> Mapping[str, Any]:
    if payload.get("format_version") != 1 or payload.get("stage") != stage:
        raise ValueError(f"checkpoint must be format_version=1 from stage {stage!r}")
    if not isinstance(payload.get("step"), int) or int(payload["step"]) <= 0:
        raise ValueError(f"{stage} checkpoint must have a positive completed step")
    state = payload.get("trainable_model")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{stage} checkpoint has no named trainable_model state")
    invalid = [
        str(name)
        for name in state
        if not (
            "act_control_adapter." in str(name)
            or ".lora_a." in str(name)
            or ".lora_b." in str(name)
        )
    ]
    if invalid:
        raise ValueError(f"{stage} checkpoint contains unexpected trainable tensors: {invalid[:8]}")
    return state


@dataclass(frozen=True)
class ParentLineage:
    path: str
    sha256: str
    stage: str
    step: int
    source_manifest_hashes: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def artifact_hashes(config: LongForcingConfig, config_path: str | Path) -> dict[str, str]:
    hashes = collect_manifest_hashes(
        {
            "dataset_manifest": config.data.manifest_path,
            "feature_index": config.data.feature_index_path,
            "feature_receipt": config.data.feature_receipt_path,
            "long_feature_index": config.data.long_feature_index_path,
            "long_feature_receipt": config.data.long_feature_receipt_path,
            "training_config": config_path,
            "teacher_checkpoint": config.lineage.teacher_checkpoint_path,
            "causal_checkpoint": config.lineage.causal_checkpoint_path,
        }
    )
    expected = {
        "dataset_manifest": config.data.manifest_sha256,
        "teacher_checkpoint": config.lineage.teacher_checkpoint_sha256,
        "causal_checkpoint": config.lineage.causal_checkpoint_sha256,
    }
    for name, digest in expected.items():
        if digest is not None and hashes[name] != digest:
            raise ValueError(f"{name} hash mismatch: expected {digest}, got {hashes[name]}")
    return hashes


def _torch_load(path: str | Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - old PyTorch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must contain a mapping: {path}")
    return payload


def load_parent_checkpoints(
    config: LongForcingConfig, hashes: Mapping[str, str]
) -> tuple[dict[str, Any], ParentLineage, dict[str, Any], ParentLineage]:
    """Load both parents and require an unbroken teacher -> causal lineage."""

    teacher = _torch_load(config.lineage.teacher_checkpoint_path)
    if sha256_file(config.lineage.teacher_checkpoint_path) != hashes.get("teacher_checkpoint"):
        raise ValueError("teacher checkpoint changed after artifact hashes were collected")
    teacher_state = _named_trainable_state(teacher, stage=config.lineage.expected_teacher_stage)
    teacher_hashes = teacher.get("manifest_hashes")
    if not isinstance(teacher_hashes, dict):
        raise ValueError("teacher checkpoint has no source manifest hashes")
    for key in SHARED_DATA_HASH_KEYS:
        if teacher_hashes.get(key) != hashes.get(key):
            raise ValueError(f"teacher checkpoint {key} lineage mismatch")
    teacher_model = teacher.get("config", {}).get("model", {})
    if teacher_model.get("base_model_path") != config.model.base_model_path:
        raise ValueError("teacher checkpoint uses a different base model")
    teacher_lineage = ParentLineage(
        path=str(Path(config.lineage.teacher_checkpoint_path).resolve()),
        sha256=hashes["teacher_checkpoint"],
        stage=config.lineage.expected_teacher_stage,
        step=int(teacher["step"]),
        source_manifest_hashes={str(k): str(v) for k, v in teacher_hashes.items()},
    )

    causal = _torch_load(config.lineage.causal_checkpoint_path)
    if sha256_file(config.lineage.causal_checkpoint_path) != hashes.get("causal_checkpoint"):
        raise ValueError("causal checkpoint changed after artifact hashes were collected")
    causal_state = _named_trainable_state(causal, stage=config.lineage.expected_causal_stage)
    causal_hashes = causal.get("manifest_hashes")
    if not isinstance(causal_hashes, dict):
        raise ValueError("causal checkpoint has no source manifest hashes")
    for key in SHARED_DATA_HASH_KEYS:
        if causal_hashes.get(key) != hashes.get(key):
            raise ValueError(f"causal checkpoint {key} lineage mismatch")
    if causal_hashes.get("teacher_checkpoint") != hashes["teacher_checkpoint"]:
        raise ValueError("causal checkpoint was not trained from the selected teacher checkpoint")
    if causal.get("parent_teacher") != teacher_lineage.as_dict():
        raise ValueError("causal checkpoint parent_teacher lineage is not exact")
    causal_model = causal.get("config", {}).get("model", {})
    if causal_model.get("base_model_path") != config.model.base_model_path:
        raise ValueError("causal checkpoint uses a different base model")
    if set(teacher_state) != set(causal_state):
        raise ValueError("teacher and causal named trainable state schemas differ")
    causal_lineage = ParentLineage(
        path=str(Path(config.lineage.causal_checkpoint_path).resolve()),
        sha256=hashes["causal_checkpoint"],
        stage=config.lineage.expected_causal_stage,
        step=int(causal["step"]),
        source_manifest_hashes={str(k): str(v) for k, v in causal_hashes.items()},
    )
    return teacher, teacher_lineage, causal, causal_lineage


def initialize_trainable_from_checkpoint(
    wrapper: nn.Module,
    payload: Mapping[str, Any],
    config: LongForcingConfig,
    *,
    frozen_after_load: bool,
) -> TrainableSummary:
    backbone = getattr(wrapper, "model", None)
    if not isinstance(backbone, nn.Module):
        raise TypeError("Wan wrapper must expose its backbone as .model")
    summary = configure_action_teacher(
        backbone,
        rank=config.model.lora_rank,
        alpha=config.model.lora_alpha,
        dropout=config.model.lora_dropout,
    )
    state = payload["trainable_model"]
    expected = {name for name, parameter in wrapper.named_parameters() if parameter.requires_grad}
    if expected != set(state):
        raise ValueError("checkpoint trainable names do not exactly match the constructed Wan model")
    load_trainable_state_dict(wrapper, dict(state))
    if frozen_after_load:
        wrapper.requires_grad_(False)
        wrapper.eval()
    return summary


def validate_rollout_batch(
    batch: Mapping[str, Any], config: LongForcingConfig, depth: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    required = {"initial_latent", "rollout_noise", "block_actions"}
    missing = required.difference(batch)
    if missing:
        raise ValueError(f"long-rollout batch is missing keys: {sorted(missing)}")
    initial = batch["initial_latent"]
    noise = batch["rollout_noise"]
    actions = batch["block_actions"]
    if not all(torch.is_tensor(value) for value in (initial, noise, actions)):
        raise TypeError("initial_latent, rollout_noise, and block_actions must be tensors")
    batch_size = config.training.micro_batch_size
    c, h, w = (
        config.model.latent_channels,
        config.data.height // config.model.spatial_compression,
        config.data.width // config.model.spatial_compression,
    )
    expected_initial = (batch_size, 1, c, h, w)
    expected_noise_tail = (config.model.num_frame_per_block, c, h, w)
    expected_actions_tail = (
        config.model.num_frame_per_block * config.model.rgb_frames_per_latent,
        len(CANONICAL_ACTION_KEYS),
    )
    if tuple(initial.shape) != expected_initial:
        raise ValueError(f"initial_latent must be {expected_initial}, got {tuple(initial.shape)}")
    if noise.ndim != 6 or tuple(noise.shape[:1]) != (batch_size,) or tuple(noise.shape[2:]) != expected_noise_tail:
        raise ValueError("rollout_noise must be [B,blocks,3,C,H,W]")
    if actions.ndim != 4 or tuple(actions.shape[:1]) != (batch_size,) or tuple(actions.shape[2:]) != expected_actions_tail:
        raise ValueError("block_actions must be [B,blocks,12,8]")
    if noise.shape[1] < depth or actions.shape[1] < depth:
        raise ValueError(f"batch provides fewer than the required {depth} rollout blocks")
    conditions = batch.get("conditions", {})
    return initial, noise, actions, conditions


def _solve_block(
    velocity_fn,
    *,
    initial_noise: torch.Tensor,
    history: torch.Tensor,
    action_block: torch.Tensor,
    block_index: int,
    conditions: Any,
    steps: int,
    timestep_shift: float,
) -> torch.Tensor:
    state = initial_noise
    sigmas = euler_sigmas(
        steps,
        shift=timestep_shift,
        device=state.device,
        dtype=state.dtype,
    )
    for index in range(steps):
        sigma = sigmas[index].expand(state.shape[0])
        velocity = velocity_fn(
            noisy_block=state,
            history=history,
            action_block=action_block,
            sigma=sigma,
            timestep=sigma * 1000.0,
            block_index=block_index,
            conditions=conditions,
        )
        if not torch.is_tensor(velocity) or tuple(velocity.shape) != tuple(state.shape):
            raise RuntimeError("rollout velocity must match the noisy block shape")
        state = state + (sigmas[index + 1] - sigmas[index]) * velocity
    return state


def longforcing_lite_loss(
    backend: LongForcingBackend,
    batch: Mapping[str, Any],
    config: LongForcingConfig,
    *,
    depth: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Endpoint MSE with student gradients only through the final block."""

    validate_backend(backend)
    initial, noise, actions, conditions = validate_rollout_batch(batch, config, depth)
    if isinstance(conditions, Mapping):
        conditions = dict(conditions)
    else:
        raise TypeError("conditions must be a mapping")
    conditions["all_block_actions"] = actions
    backend.activate_student(initial.device)
    history = initial
    for block_index in range(depth - 1):
        with torch.no_grad():
            endpoint = _solve_block(
                backend.student_velocity,
                initial_noise=noise[:, block_index],
                history=history,
                action_block=actions[:, block_index],
                block_index=block_index,
                conditions=conditions,
                steps=config.rollout.student_steps,
                timestep_shift=config.model.timestep_shift,
            )
        history = torch.cat((history, endpoint.detach()), dim=1).detach()

    final_index = depth - 1
    history = history.detach()
    # The endpoint teacher must run before the differentiable student endpoint.
    # This leaves no live student autograd graph while the two 5B backbones are
    # swapped, allowing them to use the GPU exclusively on a 32GB card.
    backend.activate_teacher(initial.device)
    try:
        with torch.no_grad():
            teacher_endpoint = _solve_block(
                backend.teacher_velocity,
                initial_noise=noise[:, final_index].detach(),
                history=history,
                action_block=actions[:, final_index],
                block_index=final_index,
                conditions=conditions,
                steps=config.rollout.teacher_steps,
                timestep_shift=config.model.timestep_shift,
            ).detach()
    finally:
        backend.activate_student(initial.device)
    student_endpoint = _solve_block(
        backend.student_velocity,
        initial_noise=noise[:, final_index],
        history=history,
        action_block=actions[:, final_index],
        block_index=final_index,
        conditions=conditions,
        steps=config.rollout.student_steps,
        timestep_shift=config.model.timestep_shift,
    )
    loss = torch.nn.functional.mse_loss(student_endpoint.float(), teacher_endpoint.float())
    return loss, {
        "objective": "teacher_endpoint_mse",
        "method": METHOD_NAME,
        "is_dmd": False,
        "depth_blocks": depth,
        "student_steps": config.rollout.student_steps,
        "teacher_steps": config.rollout.teacher_steps,
        "rollout_rgb_frames": rgb_frames_for_blocks(depth, config),
    }


class WanLongForcingWindowStudent:
    """Shared window-recompute student used by training and stage-3 inference."""

    def __init__(self, *, student: nn.Module, config: LongForcingConfig):
        self.student = student
        self.config = config

    @staticmethod
    def _prompt_list(conditions: Mapping[str, Any]) -> list[torch.Tensor]:
        prompt = conditions.get("prompt_embeds")
        if torch.is_tensor(prompt) and prompt.ndim == 3:
            return [sample for sample in prompt]
        if isinstance(prompt, list) and prompt and all(torch.is_tensor(item) for item in prompt):
            return prompt
        raise ValueError("rollout conditions require prompt_embeds as [B,L,D] or a tensor list")

    def _window_inputs(
        self,
        *,
        noisy_block: torch.Tensor,
        history: torch.Tensor,
        block_index: int,
        conditions: Mapping[str, Any],
    ) -> tuple[torch.Tensor, list[torch.Tensor], int]:
        all_actions = conditions.get("all_block_actions")
        if not torch.is_tensor(all_actions) or all_actions.ndim != 4:
            raise ValueError("all_block_actions must be [B,blocks,12,8]")
        if block_index < 4:
            history_window = history
            first_block = 0
            global_latent_start = 0
        else:
            history_window = history[:, -9:]
            first_block = block_index - 3
            global_latent_start = 1 + first_block * self.config.model.num_frame_per_block
        full_window = torch.cat((history_window, noisy_block), dim=1)
        if full_window.shape[1] > 13:
            raise RuntimeError("sliding Wan window exceeded the 49-frame / 13-latent contract")
        action_frames = all_actions[:, first_block : block_index + 1].flatten(1, 2)
        cache = conditions.get("_longforcing_action_context_cache")
        if cache is None:
            cache = {}
            if isinstance(conditions, dict):
                conditions["_longforcing_action_context_cache"] = cache
        cache_key = (first_block, block_index, str(action_frames.device), str(action_frames.dtype))
        if cache_key not in cache:
            # A spatial action tensor is up to ~0.29 GiB at 480x832.  Retaining
            # one per rollout block would pin ~5.3 GiB at depth 20, so keep only
            # the current four-block window (all solver steps reuse this entry).
            cache.clear()
            cache[cache_key] = build_action_context(
                action_frames,
                height=self.config.data.height,
                width=self.config.data.width,
                device=action_frames.device,
                dtype=action_frames.dtype,
            )
        return full_window, cache[cache_key], global_latent_start

    def student_velocity(self, **kwargs: Any) -> torch.Tensor:
        conditions = kwargs["conditions"]
        full, action_context, global_start = self._window_inputs(
            noisy_block=kwargs["noisy_block"],
            history=kwargs["history"],
            block_index=int(kwargs["block_index"]),
            conditions=conditions,
        )
        batch, frames, _, height, width = full.shape
        timestep = torch.zeros((batch, frames), device=full.device, dtype=torch.float32)
        timestep[:, -self.config.model.num_frame_per_block :] = kwargs["timestep"].float()[:, None]
        patch_h, patch_w = self.student.model.patch_size[1:]
        tokens_per_frame = (height // patch_h) * (width // patch_w)
        output = self.student.model(
            full.permute(0, 2, 1, 3, 4),
            t=timestep,
            context=self._prompt_list(conditions),
            seq_len=None,
            act_context=action_context,
            act_context_scale=self.config.model.action_scale,
            current_start=global_start * tokens_per_frame,
        )
        if isinstance(output, tuple):
            output = output[0]
        output = output.permute(0, 2, 1, 3, 4)
        return output[:, -self.config.model.num_frame_per_block :]


class WanLongForcingBackend(WanLongForcingWindowStudent):
    """Checked Wan2.2 implementation of the rollout protocol.

    Each model call is limited to the latest four causal blocks (49 RGB frames
    for the initial window, 48 thereafter).  Long-range state is represented by
    the student's detached generated latents, so a 20-block rollout does not
    materialize a 241-frame attention graph.
    """

    def __init__(self, *, student: nn.Module, teacher: nn.Module, config: LongForcingConfig):
        super().__init__(student=student, config=config)
        self.teacher = teacher
        self._active_role = "student"
        self._validate_wan_models()
        from training.causal_tf import CausalTeacherForcingConfig

        replay = CausalTeacherForcingConfig()
        replay.model.base_model_path = config.model.base_model_path
        replay.model.action_scale = config.model.action_scale
        replay.model.latent_channels = config.model.latent_channels
        replay.model.spatial_compression = config.model.spatial_compression
        replay.data.height = config.data.height
        replay.data.width = config.data.width
        replay.data.num_frames = config.data.short_window_frames
        replay.training.micro_batch_size = config.training.micro_batch_size
        self._replay_config = replay

    def _validate_wan_models(self) -> None:
        student_backbone = getattr(self.student, "model", None)
        teacher_backbone = getattr(self.teacher, "model", None)
        if type(student_backbone).__name__ != "CausalWanModel":
            raise RuntimeError("student factory did not construct CausalWanModel")
        if type(teacher_backbone).__name__ != "WanModel":
            raise RuntimeError("endpoint teacher factory did not construct bidirectional WanModel")
        if getattr(self.student, "uniform_timestep", None) is not False:
            raise RuntimeError("causal student must preserve per-frame timesteps")
        if getattr(self.teacher, "uniform_timestep", None) is not True:
            raise RuntimeError("endpoint teacher must be the non-causal Wan wrapper")
        for wrapper in (self.student, self.teacher):
            backbone = wrapper.model
            adapter = getattr(backbone, "act_control_adapter", None)
            if not isinstance(adapter, nn.Module):
                raise RuntimeError("Wan ci2v model is missing its action adapter")
            if int(getattr(backbone, "in_dim", -1)) != self.config.model.latent_channels:
                raise RuntimeError("Wan latent channels do not match the cached latent contract")

    def _activate_exclusive(self, role: str, device: torch.device) -> None:
        if role not in {"student", "teacher"}:
            raise ValueError(f"unknown LongForcing model role: {role}")
        active = self.student if role == "student" else self.teacher
        inactive = self.teacher if role == "student" else self.student
        if (
            self.config.rollout.teacher_cpu_offload
            and device.type == "cuda"
            and role != self._active_role
        ):
            inactive.to(device=torch.device("cpu"), dtype=torch.bfloat16)
            torch.cuda.empty_cache()
            active.to(device=device, dtype=torch.bfloat16)
        self._active_role = role
        if role == "student":
            self.student.train()
        else:
            self.teacher.eval()

    def activate_student(self, device: torch.device) -> None:
        self._activate_exclusive("student", device)

    def activate_teacher(self, device: torch.device) -> None:
        self._activate_exclusive("teacher", device)

    def teacher_velocity(self, **kwargs: Any) -> torch.Tensor:
        conditions = kwargs["conditions"]
        full, action_context, _ = self._window_inputs(
            noisy_block=kwargs["noisy_block"],
            history=kwargs["history"],
            block_index=int(kwargs["block_index"]),
            conditions=conditions,
        )
        batch, frames = full.shape[:2]
        timestep = torch.zeros((batch, frames), device=full.device, dtype=torch.float32)
        timestep[:, -self.config.model.num_frame_per_block :] = kwargs["timestep"].float()[:, None]
        output = self.teacher.model(
            full.permute(0, 2, 1, 3, 4),
            t=timestep,
            context=self._prompt_list(conditions),
            seq_len=None,
            act_context=action_context,
            act_context_scale=self.config.model.action_scale,
        )
        if isinstance(output, tuple):
            output = output[0]
        output = output.permute(0, 2, 1, 3, 4)
        return output[:, -self.config.model.num_frame_per_block :]

    def flowmatch_replay_loss(self, short_window: Mapping[str, Any]) -> torch.Tensor:
        from training.causal_tf import causal_teacher_forcing_loss

        noisy_latents = short_window.get("noisy_latents")
        if not torch.is_tensor(noisy_latents):
            raise ValueError("flowmatch replay requires noisy_latents")
        self.activate_student(noisy_latents.device)
        device = next(self.student.parameters()).device
        return causal_teacher_forcing_loss(
            self.student,
            short_window,
            self._replay_config,
            device,
        )


def wan_longforcing_backend_factory(
    *, config: LongForcingConfig, device: torch.device
) -> WanLongForcingBackend:
    """Construct the real stage-3 Wan teacher/student pair.

    Checkpoint trainables are intentionally loaded by the launch entry point
    after this returns, so both parent lineages are verified first.
    """

    del device  # construction is on CPU; the launcher moves checked models once.
    from utils.wan_wrapper import WanDiffusionWrapper

    common = {
        "model_name": config.model.base_model_path,
        "timestep_shift": config.model.timestep_shift,
        "local_attn_size": config.model.local_attn_size,
        "model_type": config.model.model_type,
        "num_frame_per_block": config.model.num_frame_per_block,
        "downscale_factor_control_adapter": config.model.downscale_factor_control_adapter,
    }
    student = WanDiffusionWrapper(is_causal=True, **common)
    student.model.independent_first_frame = True
    student.model.gradient_checkpointing_mode = config.model.gradient_checkpointing_mode
    teacher = WanDiffusionWrapper(is_causal=False, **common)
    return WanLongForcingBackend(student=student, teacher=teacher, config=config)
