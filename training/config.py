"""Typed configuration for the action-teacher stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .models.action_adapter import ACTION_DIM, CANONICAL_ACTION_KEYS, RGB_FRAMES_PER_ACTION_TOKEN
from .paths import is_pinned_base_model


@dataclass
class ModelConfig:
    base_model_path: str = (
        "/path/to/interactworld/models/"
        "Wan2.2-TI2V-5B@921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
    )
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
    num_frame_per_block: int = 3
    independent_first_frame: bool = True


@dataclass
class DataConfig:
    manifest_path: str = "/path/to/interactworld/data/manifests/train.jsonl"
    manifest_sha256: str | None = None
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
class OptimizerConfig:
    name: str = "adamw8bit"
    adapter_lr: float = 1e-4
    lora_lr: float = 5e-5
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0


@dataclass
class TrainingConfig:
    precision: str = "bf16"
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_steps: int = 2000
    checkpoint_every: int = 50
    keep_last: int = 2
    keep_best: int = 1
    seed: int = 42
    output_dir: str = "/path/to/interactworld/runs/action_teacher_lora_v1"
    log_every: int = 1


@dataclass
class ActionTeacherConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        errors: list[str] = []
        if not is_pinned_base_model(self.model.base_model_path):
            errors.append("base_model_path must be an absolute path to the pinned Wan2.2 revision")
        if tuple(self.data.canonical_action_keys) != CANONICAL_ACTION_KEYS:
            errors.append(f"canonical_action_keys must be exactly {CANONICAL_ACTION_KEYS}")
        if self.data.rgb_frames_per_action_token != RGB_FRAMES_PER_ACTION_TOKEN:
            errors.append("rgb_frames_per_action_token must be 4 for the released ABot interface")
        if self.model.action_dim != ACTION_DIM:
            errors.append("action_dim must be 32 (8 canonical keys * 4 RGB frames)")
        if self.model.model_type != "ci2v":
            errors.append("model_type must be ci2v so the official action adapter is materialized")
        if self.model.lora_rank != 16:
            errors.append("the v1 action-teacher contract fixes LoRA rank to 16")
        if self.training.precision != "bf16":
            errors.append("the v1 action-teacher contract requires bf16")
        if self.training.micro_batch_size != 1:
            errors.append("micro_batch_size must be 1 on the 32 GB RTX 5090")
        if self.training.gradient_accumulation_steps != 8:
            errors.append("gradient_accumulation_steps must be 8")
        if (self.data.num_frames - 1) % self.data.rgb_frames_per_action_token:
            errors.append("num_frames - 1 must be divisible by 4 (first frame is the clean condition)")
        expected_latent_frames = 1 + (self.data.num_frames - 1) // self.model.temporal_compression
        expected_latent_height = self.data.height // self.model.spatial_compression
        expected_latent_width = self.data.width // self.model.spatial_compression
        if (expected_latent_frames, self.model.latent_channels, expected_latent_height, expected_latent_width) != (
            13,
            48,
            30,
            52,
        ):
            errors.append("49x480x832 RGB must map to latent [13,48,30,52]")
        if self.model.num_frame_per_block != 3 or not self.model.independent_first_frame:
            errors.append("teacher profile requires independent first latent frame plus 3-frame blocks")
        if (expected_latent_frames - 1) % self.model.num_frame_per_block:
            errors.append("future latent frames must divide into complete 3-frame blocks")
        if (self.data.height, self.data.width) != (480, 832):
            errors.append("the v1 profile is fixed to 480x832")
        if self.training.checkpoint_every < 20 or self.training.checkpoint_every > 50:
            errors.append("checkpoint_every must be between 20 and 50")
        if (self.training.keep_best, self.training.keep_last) != (1, 2):
            errors.append("checkpoint retention must be best 1 + last 2")
        if self.training.seed != 42:
            errors.append("the v1 reproducibility seed is fixed to 42")
        if errors:
            raise ValueError("invalid action-teacher configuration:\n- " + "\n- ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_tuple_fields(data: dict[str, Any]) -> dict[str, Any]:
    data = dict(data)
    section = dict(data.get("data", {}))
    if "canonical_action_keys" in section:
        section["canonical_action_keys"] = tuple(section["canonical_action_keys"])
    data["data"] = section
    optimizer = dict(data.get("optimizer", {}))
    if "betas" in optimizer:
        optimizer["betas"] = tuple(optimizer["betas"])
    data["optimizer"] = optimizer
    return data


def load_config(path: str | Path) -> ActionTeacherConfig:
    """Load YAML without silently accepting unknown top-level sections."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - declared transitively by OmegaConf
        raise RuntimeError("PyYAML is required to load the training configuration") from exc
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    allowed = {"model", "data", "optimizer", "training"}
    unknown = set(raw).difference(allowed)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    raw = _coerce_tuple_fields(raw)
    config = ActionTeacherConfig(
        model=ModelConfig(**raw.get("model", {})),
        data=DataConfig(**raw.get("data", {})),
        optimizer=OptimizerConfig(**raw.get("optimizer", {})),
        training=TrainingConfig(**raw.get("training", {})),
    )
    config.validate()
    return config
