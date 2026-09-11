"""Fail-closed 15-second causal rollout evaluation and demo gallery.

The v1 timing contract follows the training cache exactly: an independent
initial RGB frame plus 20 causal chunks x 3 latent frames x 4 RGB frames gives
241 total RGB frames.  At 16 fps, frame 240 is exactly 15.0 seconds after the
initial frame.  An adapter is started once per rollout and must return a linked
cursor for every chunk; independent clip generation cannot satisfy the API.
"""

from __future__ import annotations

import hashlib
import html
import importlib
import json
import math
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from training.runtime import git_revision, sha256_file
from training.paths import is_pinned_base_model

FPS = 16
NUM_CHUNKS = 20
LATENT_FRAMES_PER_CHUNK = 3
RGB_FRAMES_PER_LATENT = 4
RGB_FRAMES_PER_CHUNK = LATENT_FRAMES_PER_CHUNK * RGB_FRAMES_PER_LATENT
FUTURE_RGB_FRAMES = NUM_CHUNKS * RGB_FRAMES_PER_CHUNK
TOTAL_RGB_FRAMES = 1 + FUTURE_RGB_FRAMES
DURATION_SECONDS = (TOTAL_RGB_FRAMES - 1) / FPS
ACTION_KEYS = ("W", "A", "S", "D", "I", "J", "K", "L")
CAUSAL_STAGE = "causal_teacher_forcing_v1"
MOBA_STAGE = "causal_moba_regularized_v1"
LONGFORCING_STAGE = "longforcing_lite_v1"
EXPECTED_STAGE = CAUSAL_STAGE  # Backward-compatible name used by the adapter.
EXPECTED_PARENT_STAGE = "action_teacher_lora_v1"
PINNED_BASE_MODEL = (
    "/path/to/interactworld/models/"
    "Wan2.2-TI2V-5B@921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
)
CAUSAL_LINEAGE_KEYS = (
    "dataset_manifest",
    "feature_index",
    "feature_receipt",
    "training_config",
    "teacher_checkpoint",
)
LONGFORCING_LINEAGE_KEYS = (
    *CAUSAL_LINEAGE_KEYS,
    "causal_checkpoint",
    "long_feature_index",
    "long_feature_receipt",
)
PROMPT_LINEAGE_KEYS = ("prompt_cache", "prompt_cache_receipt")


@dataclass(frozen=True)
class ActionSegment:
    frames: int
    keys: tuple[str, ...]


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    prompt: str
    initial_frame_path: str
    reference_frames_path: str
    seed: int
    action_segments: tuple[ActionSegment, ...]
    source_episode_id: str | None = None


@dataclass(frozen=True)
class AnchorSpec:
    frame_index: int
    weight: float


@dataclass(frozen=True)
class LineageSpec:
    checkpoint_path: str
    checkpoint_sha256: str | None
    expected_stage: str
    expected_base_model_path: str
    artifact_paths: dict[str, str]


@dataclass(frozen=True)
class Rollout15sConfig:
    version: int
    run_id: str
    output_root: str
    adapter_factory: str
    width: int
    height: int
    fps: int
    num_chunks: int
    latent_frames_per_chunk: int
    rgb_frames_per_latent: int
    total_rgb_frames: int
    late_anchors: tuple[AnchorSpec, ...]
    lineage: LineageSpec
    scenes: tuple[SceneSpec, ...]

    def validate(self) -> None:
        errors: list[str] = []
        fixed = (
            self.version == 1,
            self.fps == FPS,
            self.num_chunks == NUM_CHUNKS,
            self.latent_frames_per_chunk == LATENT_FRAMES_PER_CHUNK,
            self.rgb_frames_per_latent == RGB_FRAMES_PER_LATENT,
            self.total_rgb_frames == TOTAL_RGB_FRAMES,
        )
        if not all(fixed):
            errors.append("v1 timing must be 20x3x4 future + 1 initial = 241 RGB frames at 16 fps")
        if self.width <= 0 or self.height <= 0:
            errors.append("width and height must be positive")
        if not 1 <= len(self.scenes) <= 3 or len({scene.scene_id for scene in self.scenes}) != len(self.scenes):
            errors.append("v1 requires one to three uniquely named scenes")
        if not self.run_id or Path(self.run_id).name != self.run_id:
            errors.append("run_id must be one safe path component")
        if ":" not in self.adapter_factory:
            errors.append("adapter_factory must be module:function")
        if self.lineage.expected_stage not in (CAUSAL_STAGE, MOBA_STAGE, LONGFORCING_STAGE):
            errors.append(f"expected_stage must be {CAUSAL_STAGE}, {MOBA_STAGE} or {LONGFORCING_STAGE}")
        if not is_pinned_base_model(self.lineage.expected_base_model_path):
            errors.append("expected_base_model_path must be the pinned Wan2.2 revision")
        required_lineage = (
            LONGFORCING_LINEAGE_KEYS
            if self.lineage.expected_stage == LONGFORCING_STAGE
            else CAUSAL_LINEAGE_KEYS
        )
        lineage_names = set(self.lineage.artifact_paths)
        if not set(required_lineage).issubset(lineage_names) or not lineage_names.issubset(
            (*LONGFORCING_LINEAGE_KEYS, *PROMPT_LINEAGE_KEYS)
        ):
            errors.append(f"artifact_paths must contain {required_lineage} and no unknown keys")
        prompt_keys = lineage_names.intersection(PROMPT_LINEAGE_KEYS)
        if prompt_keys and prompt_keys != set(PROMPT_LINEAGE_KEYS):
            errors.append("static prompt lineage requires both cache and receipt")
        if self.lineage.expected_stage == MOBA_STAGE:
            if lineage_names != set((*CAUSAL_LINEAGE_KEYS, *PROMPT_LINEAGE_KEYS)):
                errors.append("MoBA evaluation requires exact Action-parent and scene-static prompt artifacts")
            if self.adapter_factory != "training.eval.wan_causal_adapter:create_wan_causal_adapter":
                errors.append("MoBA evaluation requires the explicit causal full-history adapter")
        indices = [anchor.frame_index for anchor in self.late_anchors]
        if indices != sorted(indices) or not indices or indices[0] < 120 or indices[-1] != 240:
            errors.append("late anchors must be sorted, start at frame >=120, and include frame 240")
        if not math.isclose(sum(anchor.weight for anchor in self.late_anchors), 1.0, abs_tol=1e-9):
            errors.append("late-anchor weights must sum to 1")
        if any(anchor.weight <= 0 for anchor in self.late_anchors):
            errors.append("late-anchor weights must be positive")
        opposing = ({"W", "S"}, {"A", "D"}, {"I", "K"}, {"J", "L"})
        for scene in self.scenes:
            if prompt_keys and not scene.source_episode_id:
                errors.append(f"static-prompt scene {scene.scene_id!r} needs source_episode_id")
            if not scene.scene_id or not scene.prompt or scene.seed < 0:
                errors.append(f"scene {scene.scene_id!r} has incomplete fixed metadata")
            if sum(segment.frames for segment in scene.action_segments) != FUTURE_RGB_FRAMES:
                errors.append(f"scene {scene.scene_id!r} action script must cover 240 future frames")
            for segment in scene.action_segments:
                keys = set(segment.keys)
                if segment.frames <= 0 or not keys.issubset(ACTION_KEYS):
                    errors.append(f"scene {scene.scene_id!r} has an invalid action segment")
                if any(pair.issubset(keys) for pair in opposing):
                    errors.append(f"scene {scene.scene_id!r} has opposing simultaneous actions")
        if errors:
            raise ValueError("invalid rollout15s configuration:\n- " + "\n- ".join(errors))


@dataclass
class RolloutCursor:
    """Opaque adapter state linked across one continuous causal session."""

    session_id: str
    next_chunk_index: int
    state_token: str
    last_frame_sha256: str
    opaque: Any = None


@dataclass
class CausalChunk:
    parent_state_token: str
    cursor: RolloutCursor
    frames: np.ndarray


class CausalRolloutAdapter(Protocol):
    def begin(
        self,
        *,
        scene: SceneSpec,
        initial_frame: np.ndarray,
        variant: str,
        seed: int,
    ) -> RolloutCursor: ...

    def generate_next(
        self,
        *,
        cursor: RolloutCursor,
        actions: np.ndarray,
        chunk_index: int,
        latent_frames: int,
        rgb_frames: int,
    ) -> CausalChunk: ...


class VideoWriter(Protocol):
    def write(self, frames: np.ndarray) -> None: ...

    def close(self) -> None: ...

    def abort(self) -> None: ...


def _segments(raw: list[Mapping[str, Any]]) -> tuple[ActionSegment, ...]:
    return tuple(
        ActionSegment(frames=int(item["frames"]), keys=tuple(str(key) for key in item.get("keys", [])))
        for item in raw
    )


def load_rollout_config(path: str | Path) -> Rollout15sConfig:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    geometry = raw.get("geometry", {})
    lineage_raw = raw.get("lineage", {})
    config = Rollout15sConfig(
        version=int(raw.get("version", 0)),
        run_id=str(raw.get("run_id", "")),
        output_root=str(raw.get("output_root", "")),
        adapter_factory=str(raw.get("adapter_factory", "")),
        width=int(geometry.get("width", 0)),
        height=int(geometry.get("height", 0)),
        fps=int(geometry.get("fps", 0)),
        num_chunks=int(geometry.get("num_chunks", 0)),
        latent_frames_per_chunk=int(geometry.get("latent_frames_per_chunk", 0)),
        rgb_frames_per_latent=int(geometry.get("rgb_frames_per_latent", 0)),
        total_rgb_frames=int(geometry.get("total_rgb_frames", 0)),
        late_anchors=tuple(
            AnchorSpec(frame_index=int(item["frame_index"]), weight=float(item["weight"]))
            for item in raw.get("late_anchors", [])
        ),
        lineage=LineageSpec(
            checkpoint_path=str(lineage_raw.get("checkpoint_path", "")),
            checkpoint_sha256=lineage_raw.get("checkpoint_sha256"),
            expected_stage=str(lineage_raw.get("expected_stage", "")),
            expected_base_model_path=str(lineage_raw.get("expected_base_model_path", "")),
            artifact_paths={str(k): str(v) for k, v in lineage_raw.get("artifact_paths", {}).items()},
        ),
        scenes=tuple(
            SceneSpec(
                scene_id=str(item["scene_id"]),
                prompt=str(item["prompt"]),
                initial_frame_path=str(item["initial_frame_path"]),
                reference_frames_path=str(item["reference_frames_path"]),
                seed=int(item["seed"]),
                action_segments=_segments(item.get("action_segments", [])),
                source_episode_id=item.get("source_episode_id"),
            )
            for item in raw.get("scenes", [])
        ),
    )
    config.validate()
    return config


def action_script(scene: SceneSpec) -> np.ndarray:
    rows: list[np.ndarray] = []
    for segment in scene.action_segments:
        row = np.zeros(len(ACTION_KEYS), dtype=np.float32)
        for key in segment.keys:
            row[ACTION_KEYS.index(key)] = 1.0
        rows.extend([row.copy() for _ in range(segment.frames)])
    actions = np.stack(rows)
    if actions.shape != (FUTURE_RGB_FRAMES, len(ACTION_KEYS)):
        raise ValueError(f"action script shape must be (240, 8), got {actions.shape}")
    return actions


def action_variants(scene: SceneSpec) -> dict[str, np.ndarray]:
    correct = action_script(scene)
    zero = np.zeros_like(correct)
    chunks = correct.reshape(NUM_CHUNKS, RGB_FRAMES_PER_CHUNK, len(ACTION_KEYS))
    permutation = np.random.default_rng(scene.seed + 15_000).permutation(NUM_CHUNKS)
    if np.array_equal(permutation, np.arange(NUM_CHUNKS)):
        permutation = np.roll(permutation, 1)
    shuffled = chunks[permutation].reshape(correct.shape)
    if np.array_equal(shuffled, correct):
        raise ValueError(f"scene {scene.scene_id!r} has no meaningful shuffled-action baseline")
    return {"correct": correct, "zero": zero, "shuffled": shuffled}


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def default_image_loader(path: str | Path) -> np.ndarray:
    source = Path(path)
    if source.suffix.lower() == ".npy":
        frame = np.load(source, allow_pickle=False)
    else:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Pillow is required for PNG/JPEG evaluation frames") from exc
        with Image.open(source) as image:
            frame = np.asarray(image.convert("RGB"))
    return np.asarray(frame)


def default_reference_loader(path: str | Path, anchors: tuple[AnchorSpec, ...]) -> np.ndarray:
    source = Path(path)
    if source.suffix.lower() == ".npy":
        frames = np.load(source, allow_pickle=False)
        if frames.shape[0] == TOTAL_RGB_FRAMES:
            frames = frames[[anchor.frame_index for anchor in anchors]]
        return np.asarray(frames)
    if source.suffix.lower() != ".npz":
        raise ValueError("reference_frames_path must be .npy or .npz (decoded RGB, not a video clip)")
    with np.load(source, allow_pickle=False) as payload:
        frames = np.asarray(payload["frames"])
        indices = tuple(int(value) for value in payload["frame_indices"].tolist())
    expected = tuple(anchor.frame_index for anchor in anchors)
    if indices != expected:
        raise ValueError(f"reference anchor indices {indices} do not match configured {expected}")
    return frames


class FFmpegVideoWriter:
    def __init__(self, path: Path, *, width: int, height: int, fps: int) -> None:
        if path.exists():
            raise FileExistsError(path)
        self.path = path
        # Keep an .mp4 suffix so ffmpeg can select the muxer before the file is
        # atomically promoted to its public name.
        self.partial = path.with_name(path.stem + ".partial.mp4")
        self.process = subprocess.Popen(
            [
                "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.partial),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frames: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        self.process.stdin.write(np.ascontiguousarray(frames).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        code = self.process.wait(timeout=120)
        if code != 0:
            self.partial.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed with exit code {code}")
        self.partial.replace(self.path)

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=10)
        self.partial.unlink(missing_ok=True)


def ffmpeg_writer_factory(path: Path, width: int, height: int, fps: int) -> VideoWriter:
    return FFmpegVideoWriter(path, width=width, height=height, fps=fps)


def build_plan(config: Rollout15sConfig, config_path: str | Path) -> dict[str, Any]:
    paths = {
        "checkpoint": config.lineage.checkpoint_path,
        **config.lineage.artifact_paths,
        **{f"{scene.scene_id}:initial": scene.initial_frame_path for scene in config.scenes},
        **{f"{scene.scene_id}:reference": scene.reference_frames_path for scene in config.scenes},
    }
    return {
        "status": "plan_valid",
        "mode": "cpu_plan",
        "cuda_queried": False,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": sha256_file(config_path),
        "timing": {
            "initial_frames": 1,
            "future_frames": FUTURE_RGB_FRAMES,
            "total_frames": TOTAL_RGB_FRAMES,
            "fps": FPS,
            "duration_seconds": DURATION_SECONDS,
            "continuous_causal_chunks": NUM_CHUNKS,
        },
        "scenes": [scene.scene_id for scene in config.scenes],
        "artifact_exists": {name: Path(path).is_file() for name, path in paths.items()},
        "checkpoint_sha256_pinned": bool(config.lineage.checkpoint_sha256),
        "adapter_configured": config.adapter_factory != "training.eval.rollout15s:unconfigured_adapter_factory",
        "output_dir": str((Path(config.output_root) / config.run_id).resolve()),
    }


def _canonical_contract(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def reject_moba_stage_alias(payload: Mapping[str, Any]) -> None:
    """A BID checkpoint cannot become a legacy stage by relabeling its header."""
    saved = payload.get("config")
    method = payload.get("method_contract")
    if payload.get("stage") != MOBA_STAGE and (
        isinstance(saved, dict) and "regularization" in saved
        or isinstance(method, dict) and method.get("method") == "moba_inspired_sequential_bid_v1"
    ):
        raise ValueError("MoBA regularization checkpoint must retain its own distinct stage")


def validate_moba_rollout_payload(payload: Mapping[str, Any]):
    """Validate the new stage as itself, without importing GPU/model code.

    This metadata check is repeated by the factory after the checkpoint hash
    gate. ``verify_checkpoint_lineage`` additionally binds the actual source
    YAML, data files and Action parent; metadata alone is not provenance.
    """
    from dataclasses import fields
    import torch
    from training.causal_moba import CausalMoBAConfig, method_contract
    from training.causal_tf import _is_named_teacher_parameter
    from train_causal_moba import _sampling_contract

    if payload.get("format_version") != 1 or payload.get("stage") != MOBA_STAGE:
        raise ValueError("MoBA rollout requires the exact causal_moba_regularized_v1 stage")
    raw = payload.get("config")
    config = CausalMoBAConfig()
    sections = {field.name for field in fields(config)}
    if not isinstance(raw, dict) or set(raw) != sections:
        raise ValueError("MoBA checkpoint requires a complete serialized configuration")
    for name in sections:
        values, defaults = raw[name], getattr(config, name)
        if not isinstance(values, dict) or set(values) != {field.name for field in fields(defaults)}:
            raise ValueError(f"MoBA serialized {name} configuration has missing or unknown fields")
        setattr(config, name, type(defaults)(**values))
    config.validate()
    if config.data.data_factory not in (
        "training.data.action_dataset:build_action_teacher_dataloader",
        "training.data.action_resampled:build_resampled_action_teacher_dataloader",
    ):
        raise ValueError("MoBA evaluation does not recognize the serialized sampling factory")
    if _canonical_contract(payload.get("method_contract")) != _canonical_contract(method_contract(config)):
        raise ValueError("MoBA method/regularization contract disagrees with serialized configuration")
    if _canonical_contract(payload.get("sampling_contract")) != _canonical_contract(_sampling_contract(config)):
        raise ValueError("MoBA sampling contract disagrees with serialized configuration")
    if payload.get("initialization_mode") != "action_teacher_weights_only_fresh_optimizer_rng_step0":
        raise ValueError("MoBA checkpoint has an unexpected initialization mode")
    maximum, step, micro = config.training.max_steps, payload.get("step"), payload.get("micro_batches_consumed")
    if (isinstance(maximum, bool) or not isinstance(maximum, int)
            or isinstance(step, bool) or not isinstance(step, int) or not 0 < step <= maximum):
        raise ValueError("MoBA checkpoint must contain a completed positive optimizer step within its budget")
    if (isinstance(micro, bool) or not isinstance(micro, int)
            or micro != step * config.training.gradient_accumulation_steps):
        raise ValueError("MoBA checkpoint micro-batch position disagrees with its completed optimizer step")
    state = payload.get("trainable_model")
    if (not isinstance(state, dict) or not state
            or any(not isinstance(name, str) or not _is_named_teacher_parameter(name) for name in state)
            or not all(any(part in name for name in state)
                       for part in ("act_control_adapter.", ".lora_a.", ".lora_b."))):
        raise ValueError("MoBA checkpoint must contain named action-adapter and complete LoRA state")
    for name, tensor in state.items():
        if (not torch.is_tensor(tensor) or not tensor.is_floating_point()
                or not bool(torch.isfinite(tensor).all())):
            raise ValueError(f"MoBA checkpoint contains invalid trainable weights: {name}")
    if payload.get("parent_causal") is not None:
        raise ValueError("MoBA stage must retain its direct Action parent, not a causal-parent substitution")
    return config


def _verify_moba_source_lineage(payload: Mapping[str, Any], spec: LineageSpec, config) -> dict[str, Any]:
    import torch
    from training.causal_moba import load_moba_config, method_contract
    from training.causal_tf import load_teacher_checkpoint

    paths = spec.artifact_paths
    hashes = payload["manifest_hashes"]
    if set(paths) != set((*CAUSAL_LINEAGE_KEYS, *PROMPT_LINEAGE_KEYS)):
        raise ValueError("MoBA evaluation requires exact source and static-prompt artifact paths")
    consumed = {
        "dataset_manifest": config.data.manifest_path,
        "feature_index": config.data.feature_index_path,
        "feature_receipt": config.data.feature_receipt_path,
        "teacher_checkpoint": config.lineage.checkpoint_path,
        "prompt_cache": config.data.prompt_cache_path,
        "prompt_cache_receipt": str(Path(config.data.prompt_cache_path).with_suffix(".pt.receipt.json")),
    }
    if any(Path(path).resolve() != Path(paths[name]).resolve() for name, path in consumed.items()):
        raise ValueError("MoBA serialized source paths differ from the pinned evaluation artifacts")
    if config.data.manifest_sha256 not in (None, hashes["dataset_manifest"]):
        raise ValueError("MoBA serialized manifest SHA-256 differs from its source")
    if config.lineage.checkpoint_sha256 not in (None, hashes["teacher_checkpoint"]):
        raise ValueError("MoBA serialized Action-parent SHA-256 differs from its source")
    source = load_moba_config(paths["training_config"])
    saved_dict, source_dict = config.to_dict(), source.to_dict()
    # These are the two explicit CLI overrides in the training entry point.
    # All model/data/optimizer/regularization fields remain source-YAML bound.
    for value in (saved_dict, source_dict):
        for name in ("max_steps", "output_dir"):
            value["training"].pop(name)
    if _canonical_contract(saved_dict) != _canonical_contract(source_dict):
        raise ValueError("MoBA serialized configuration differs from the hashed source training config")
    parent_payload, parent_lineage = load_teacher_checkpoint(config, hashes)
    if isinstance(parent_payload.get("step"), bool):
        raise ValueError("MoBA Action parent must have a positive integer optimizer step")
    if _canonical_contract(payload["parent_teacher"]) != _canonical_contract(parent_lineage.as_dict()):
        raise ValueError("MoBA parent-teacher record differs from the actual Action checkpoint")
    state, parent_state = payload["trainable_model"], parent_payload["trainable_model"]
    if set(state) != set(parent_state) or any(
        not torch.is_tensor(parent_state[name])
        or tuple(state[name].shape) != tuple(parent_state[name].shape) for name in state
    ):
        raise ValueError("MoBA trainable weights do not preserve the Action-parent parameter names/shapes")
    return {
        "method_contract": method_contract(config),
        "sampling_contract": dict(payload["sampling_contract"]),
        "initialization_mode": payload["initialization_mode"],
        "regularization": dict(payload["config"]["regularization"]),
        "action_scale": config.model.action_scale,
        "inference_contract": {
            "model_class": "CausalWanModel", "training_attention_mode": "causal",
            "bidirectional_enabled": False, "context_mode": "full_history_kv",
            "denoising_steps": 40, "streaming_solver": "flow_euler",
            "timestep_shift": 5.0, "realtime": False,
        },
    }


def verify_checkpoint_lineage(config: Rollout15sConfig) -> dict[str, Any]:
    spec = config.lineage
    if spec.expected_stage not in (CAUSAL_STAGE, MOBA_STAGE, LONGFORCING_STAGE):
        raise ValueError("evaluation requires an explicitly supported expected_stage")
    if not spec.checkpoint_sha256 or len(spec.checkpoint_sha256) != 64:
        raise ValueError("launch requires an exact checkpoint_sha256")
    checkpoint_path = Path(spec.checkpoint_path)
    actual_checkpoint_hash = sha256_file(checkpoint_path)
    if actual_checkpoint_hash != spec.checkpoint_sha256.lower():
        raise ValueError("checkpoint SHA-256 does not match the pinned evaluation config")
    actual_artifacts = {name: sha256_file(path) for name, path in spec.artifact_paths.items()}
    try:
        import torch
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("checkpoint must be a format_version=1 mapping")
    reject_moba_stage_alias(payload)
    if (payload.get("stage") != spec.expected_stage or isinstance(payload.get("step"), bool)
            or not isinstance(payload.get("step"), int) or payload["step"] <= 0):
        raise ValueError(f"checkpoint is not a completed {spec.expected_stage} step")
    if not isinstance(payload.get("trainable_model"), Mapping) or not payload["trainable_model"]:
        raise ValueError("checkpoint has no named trainable model state")
    moba_config = validate_moba_rollout_payload(payload) if spec.expected_stage == MOBA_STAGE else None
    hashes = payload.get("manifest_hashes")
    expected_keys = set(
        LONGFORCING_LINEAGE_KEYS
        if spec.expected_stage == LONGFORCING_STAGE
        else CAUSAL_LINEAGE_KEYS
    )
    if "prompt_cache" in spec.artifact_paths:
        expected_keys.update(PROMPT_LINEAGE_KEYS)
    if not isinstance(hashes, dict) or set(hashes) != expected_keys:
        raise ValueError("checkpoint has an unexpected artifact-lineage schema")
    if hashes != {key: actual_artifacts[key] for key in expected_keys}:
        raise ValueError("checkpoint artifact lineage does not match the exact current files")
    parent = payload.get("parent_teacher")
    if not isinstance(parent, dict) or parent.get("stage") != EXPECTED_PARENT_STAGE:
        raise ValueError("checkpoint has no valid action-teacher parent lineage")
    if parent.get("sha256") != hashes.get("teacher_checkpoint"):
        raise ValueError("parent teacher SHA-256 disagrees with causal checkpoint lineage")
    parent_sources = parent.get("source_manifest_hashes")
    inherited_keys = ("dataset_manifest", "feature_index", "feature_receipt", *(
        PROMPT_LINEAGE_KEYS if "prompt_cache" in expected_keys else ()
    ))
    for key in inherited_keys:
        if not isinstance(parent_sources, dict) or parent_sources.get(key) != hashes.get(key):
            raise ValueError(f"parent teacher {key} lineage mismatch")
    from training.causal_tf import _prompt_contract
    saved_data = payload.get("config", {}).get("data", {})
    prompt_contract = _prompt_contract(saved_data, hashes)
    prompt_path = spec.artifact_paths.get("prompt_cache")
    if prompt_contract["prompt_cache_path"] != prompt_path:
        raise ValueError("evaluation prompt cache path differs from saved training configuration")
    if prompt_path is not None:
        from training.data.action_dataset import validate_scene_static_prompt_cache_binding
        if Path(spec.artifact_paths["prompt_cache_receipt"]).resolve() != Path(prompt_path).with_suffix(".pt.receipt.json").resolve():
            raise ValueError("evaluation prompt receipt is not the bound cache sidecar")
        receipt = validate_scene_static_prompt_cache_binding(prompt_path,
            index_path=spec.artifact_paths["feature_index"], manifest_path=spec.artifact_paths["dataset_manifest"])
        for scene in config.scenes:
            binding = receipt["episodes"].get(scene.source_episode_id)
            if not binding or binding["split"] != "dev" or scene.prompt != binding["prompt"]:
                raise ValueError(f"evaluation prompt does not match held-out training-cache binding: {scene.scene_id}")
    model = payload.get("config", {}).get("model", {})
    if model.get("base_model_path") != spec.expected_base_model_path:
        raise ValueError("checkpoint base-model revision differs from the pinned evaluation base")
    if model.get("num_frame_per_block") != 3:
        raise ValueError("checkpoint does not use the required 3-latent causal block contract")
    moba_metadata = {}
    if spec.expected_stage in (CAUSAL_STAGE, MOBA_STAGE):
        if model.get("independent_first_frame") is not True:
            raise ValueError("causal checkpoint did not train with an independent first frame")
        if spec.expected_stage == MOBA_STAGE:
            moba_metadata = _verify_moba_source_lineage(payload, spec, moba_config)
    else:
        if payload.get("method") != "LongForcing-lite" or payload.get("is_dmd") is not False:
            raise ValueError("stage-3 checkpoint is not the approved LongForcing-lite objective")
        if model.get("rgb_frames_per_latent") != 4:
            raise ValueError("LongForcing checkpoint does not use four RGB frames per latent")
        causal_parent = payload.get("parent_causal")
        if not isinstance(causal_parent, dict) or causal_parent.get("stage") != CAUSAL_STAGE:
            raise ValueError("LongForcing checkpoint has no causal parent lineage")
        if causal_parent.get("sha256") != hashes.get("causal_checkpoint"):
            raise ValueError("LongForcing causal-parent SHA-256 mismatch")
        causal_sources = causal_parent.get("source_manifest_hashes")
        for key in inherited_keys:
            if not isinstance(causal_sources, dict) or causal_sources.get(key) != hashes.get(key):
                raise ValueError(f"LongForcing causal parent {key} lineage mismatch")
        try:
            causal_payload = torch.load(
                spec.artifact_paths["causal_checkpoint"],
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:  # pragma: no cover
            causal_payload = torch.load(spec.artifact_paths["causal_checkpoint"], map_location="cpu")
        causal_model = causal_payload.get("config", {}).get("model", {})
        from training.models.action_adapter import validate_action_scale
        if validate_action_scale(causal_model.get("action_scale", 1.0)) != validate_action_scale(model.get("action_scale", 1.0)):
            raise ValueError("LongForcing inference and causal parent action_scale disagree")
        if _prompt_contract(causal_payload.get("config", {}).get("data", {}), causal_payload.get("manifest_hashes", {})) != prompt_contract:
            raise ValueError("LongForcing inference and causal parent prompt contract disagree")
        if (
            causal_payload.get("stage") != CAUSAL_STAGE
            or causal_payload.get("parent_teacher") != parent
            or causal_model.get("base_model_path") != spec.expected_base_model_path
            or causal_model.get("num_frame_per_block") != 3
            or causal_model.get("independent_first_frame") is not True
        ):
            raise ValueError("LongForcing causal parent checkpoint contract is not exact")
    return {
        "path": str(checkpoint_path.resolve()),
        "sha256": actual_checkpoint_hash,
        "stage": payload["stage"],
        "step": payload["step"],
        "manifest_hashes": dict(hashes),
        "parent_teacher": dict(parent),
        "parent_causal": payload.get("parent_causal"),
        "prompt_contract": prompt_contract,
        **moba_metadata,
    }


def resolve_adapter_factory(spec: str) -> Callable[..., CausalRolloutAdapter]:
    if spec == "training.eval.rollout15s:unconfigured_adapter_factory":
        raise RuntimeError("real causal rollout adapter is not configured")
    module_name, name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), name)
    if not callable(factory):
        raise TypeError(f"adapter factory is not callable: {spec}")
    return factory


def unconfigured_adapter_factory(**_: Any) -> CausalRolloutAdapter:
    raise RuntimeError("real causal rollout adapter is not configured")


def _validate_frame_batch(frames: np.ndarray, config: Rollout15sConfig) -> np.ndarray:
    frames = np.asarray(frames)
    expected = (RGB_FRAMES_PER_CHUNK, config.height, config.width, 3)
    if frames.dtype != np.uint8 or frames.shape != expected:
        raise ValueError(f"adapter frames must be uint8 {expected}, got {frames.dtype} {frames.shape}")
    return frames


def _weighted_mse(predictions: Mapping[int, np.ndarray], references: np.ndarray, anchors: tuple[AnchorSpec, ...]) -> float:
    expected = (len(anchors), *next(iter(predictions.values())).shape)
    if references.dtype != np.uint8 or references.shape != expected:
        raise ValueError(f"reference frames must be uint8 {expected}, got {references.dtype} {references.shape}")
    value = 0.0
    for target, anchor in zip(references, anchors, strict=True):
        difference = predictions[anchor.frame_index].astype(np.float32) - target.astype(np.float32)
        value += anchor.weight * float(np.mean(np.square(difference / 255.0)))
    return value


def _run_variant(
    *,
    config: Rollout15sConfig,
    scene: SceneSpec,
    variant: str,
    actions: np.ndarray,
    initial: np.ndarray,
    adapter: CausalRolloutAdapter,
    output_path: Path,
    writer_factory: Callable[[Path, int, int, int], VideoWriter],
) -> dict[int, np.ndarray]:
    cursor = adapter.begin(scene=scene, initial_frame=initial, variant=variant, seed=scene.seed)
    if not isinstance(cursor, RolloutCursor) or cursor.next_chunk_index != 0:
        raise TypeError("adapter.begin must return a RolloutCursor at chunk zero")
    if cursor.last_frame_sha256 != _array_sha256(initial) or not cursor.session_id or not cursor.state_token:
        raise ValueError("initial rollout cursor is not linked to the supplied first frame")
    writer = writer_factory(output_path, config.width, config.height, config.fps)
    predictions: dict[int, np.ndarray] = {}
    written = 0
    try:
        writer.write(initial[None])
        written = 1
        for chunk_index in range(NUM_CHUNKS):
            start = chunk_index * RGB_FRAMES_PER_CHUNK
            chunk_actions = actions[start : start + RGB_FRAMES_PER_CHUNK]
            result = adapter.generate_next(
                cursor=cursor,
                actions=chunk_actions,
                chunk_index=chunk_index,
                latent_frames=LATENT_FRAMES_PER_CHUNK,
                rgb_frames=RGB_FRAMES_PER_CHUNK,
            )
            if not isinstance(result, CausalChunk) or not isinstance(result.cursor, RolloutCursor):
                raise TypeError("adapter.generate_next must return CausalChunk with RolloutCursor")
            frames = _validate_frame_batch(result.frames, config)
            next_cursor = result.cursor
            if (
                result.parent_state_token != cursor.state_token
                or next_cursor.session_id != cursor.session_id
                or next_cursor.next_chunk_index != chunk_index + 1
                or not next_cursor.state_token
                or next_cursor.state_token == cursor.state_token
                or next_cursor.last_frame_sha256 != _array_sha256(frames[-1])
            ):
                raise ValueError("causal cursor chain broke; refusing independently generated clip")
            writer.write(frames)
            for anchor in config.late_anchors:
                local = anchor.frame_index - written
                if 0 <= local < len(frames):
                    predictions[anchor.frame_index] = frames[local].copy()
            written += len(frames)
            cursor = next_cursor
        if written != TOTAL_RGB_FRAMES or set(predictions) != {a.frame_index for a in config.late_anchors}:
            raise RuntimeError("rollout did not produce the exact 241-frame/late-anchor contract")
        writer.close()
    except BaseException:
        writer.abort()
        raise
    return predictions


def _write_gallery(stage: Path, receipt: dict[str, Any]) -> None:
    rows: list[str] = []
    for scene in receipt["scenes"]:
        cells = "".join(
            f'<td><video controls preload="metadata" src="{html.escape(scene["videos"][variant])}"></video><br>{variant}</td>'
            for variant in ("correct", "zero", "shuffled")
        )
        rows.append(
            f'<tr><th>{html.escape(scene["scene_id"])}</th>{cells}'
            f'<td>{scene["weighted_mse"]["correct"]:.6f} &lt; '
            f'{scene["weighted_mse"]["zero"]:.6f}, {scene["weighted_mse"]["shuffled"]:.6f}: '
            f'{"PASS" if scene["passed"] else "FAIL"}</td></tr>'
        )
    document = (
        "<!doctype html><meta charset=\"utf-8\"><title>InteractWorld rollout15s</title>"
        "<style>body{font:14px sans-serif;margin:24px;background:#111;color:#eee}"
        "table{border-collapse:collapse}td,th{border:1px solid #555;padding:8px}"
        "video{width:300px}</style>"
        f'<h1>15-second continuous rollout: {"PASS" if receipt["passed"] else "FAIL"}</h1>'
        "<p>241 frames at 16 fps: one initial frame plus 20 linked causal chunks.</p>"
        "<table><tr><th>Scene</th><th>Correct</th><th>Zero</th><th>Shuffled</th><th>Late-anchor gate</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    (stage / "index.html").write_text(document, encoding="utf-8")


def run_rollout_suite(
    config: Rollout15sConfig,
    *,
    lineage: dict[str, Any],
    adapter_factory: Callable[..., CausalRolloutAdapter],
    device: str,
    image_loader: Callable[[str | Path], np.ndarray] = default_image_loader,
    reference_loader: Callable[[str | Path, tuple[AnchorSpec, ...]], np.ndarray] = default_reference_loader,
    writer_factory: Callable[[Path, int, int, int], VideoWriter] = ffmpeg_writer_factory,
    config_sha256: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    config.validate()
    final = (Path(config.output_root) / config.run_id).resolve()
    if final.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = final.parent / f".{final.name}.tmp-{uuid.uuid4().hex}"
    stage.mkdir()
    receipt: dict[str, Any] = {
        "format_version": 1,
        "stage": "rollout15s_v1",
        "config_sha256": config_sha256,
        "git_revision": git_revision(Path(__file__).parents[2]),
        "lineage": lineage,
        "timing": {
            "fps": FPS,
            "initial_frames": 1,
            "future_frames": FUTURE_RGB_FRAMES,
            "total_frames": TOTAL_RGB_FRAMES,
            "duration_seconds": DURATION_SECONDS,
            "continuous_causal_chunks": NUM_CHUNKS,
        },
        "scenes": [],
    }
    try:
        adapter_kwargs = {"expected_stage": MOBA_STAGE} if config.lineage.expected_stage == MOBA_STAGE else {}
        adapter = adapter_factory(
            checkpoint_path=config.lineage.checkpoint_path,
            checkpoint_sha256=config.lineage.checkpoint_sha256,
            base_model_path=config.lineage.expected_base_model_path,
            device=device,
            **adapter_kwargs,
        )
        for scene in config.scenes:
            initial = np.asarray(image_loader(scene.initial_frame_path))
            expected_initial = (config.height, config.width, 3)
            if initial.dtype != np.uint8 or initial.shape != expected_initial:
                raise ValueError(f"initial frame must be uint8 {expected_initial}")
            references = np.asarray(reference_loader(scene.reference_frames_path, config.late_anchors))
            metrics: dict[str, float] = {}
            videos: dict[str, str] = {}
            for variant, actions in action_variants(scene).items():
                relative = f"{scene.scene_id}-{variant}.mp4"
                predictions = _run_variant(
                    config=config,
                    scene=scene,
                    variant=variant,
                    actions=actions,
                    initial=initial,
                    adapter=adapter,
                    output_path=stage / relative,
                    writer_factory=writer_factory,
                )
                metrics[variant] = _weighted_mse(predictions, references, config.late_anchors)
                videos[variant] = relative
            passed = metrics["correct"] < metrics["zero"] and metrics["correct"] < metrics["shuffled"]
            receipt["scenes"].append(
                {
                    "scene_id": scene.scene_id,
                    "seed": scene.seed,
                    "action_script_sha256": _array_sha256(action_script(scene)),
                    "weighted_mse": metrics,
                    "passed": passed,
                    "videos": videos,
                }
            )
        aggregate = {
            variant: float(np.mean([scene["weighted_mse"][variant] for scene in receipt["scenes"]]))
            for variant in ("correct", "zero", "shuffled")
        }
        receipt["aggregate_weighted_mse"] = aggregate
        receipt["passed"] = bool(
            all(scene["passed"] for scene in receipt["scenes"])
            and aggregate["correct"] < aggregate["zero"]
            and aggregate["correct"] < aggregate["shuffled"]
        )
        _write_gallery(stage, receipt)
        metrics_tmp = stage / "metrics.json.tmp"
        metrics_tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        metrics_tmp.replace(stage / "metrics.json")
        os.replace(stage, final)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return final, receipt
