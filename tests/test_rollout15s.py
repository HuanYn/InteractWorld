from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from training.eval.rollout15s import (
    NUM_CHUNKS,
    TOTAL_RGB_FRAMES,
    CausalChunk,
    RolloutCursor,
    action_script,
    build_plan,
    load_rollout_config,
    run_rollout_suite,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "eval" / "rollout15s_v1.yaml"


def _digest(frame: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def test_default_is_cpu_plan_for_real_adapter_without_gpu_or_artifacts() -> None:
    config = load_rollout_config(CONFIG)
    plan = build_plan(config, CONFIG)
    assert plan["mode"] == "cpu_plan"
    assert plan["cuda_queried"] is False
    assert plan["adapter_configured"] is True
    assert plan["timing"] == {
        "initial_frames": 1,
        "future_frames": 240,
        "total_frames": 241,
        "fps": 16,
        "duration_seconds": 15.0,
        "continuous_causal_chunks": 20,
    }


class _Writer:
    def __init__(self, path: Path) -> None:
        self.path, self.frames = path, 0

    def write(self, frames: np.ndarray) -> None:
        self.frames += len(frames)

    def close(self) -> None:
        assert self.frames == TOTAL_RGB_FRAMES
        self.path.write_bytes(b"mp4")

    def abort(self) -> None:
        self.path.unlink(missing_ok=True)


class _LinkedAdapter:
    calls = 0

    def begin(self, *, scene, initial_frame, variant, seed):
        return RolloutCursor(
            session_id=f"{scene.scene_id}:{variant}:{seed}",
            next_chunk_index=0,
            state_token="s0",
            last_frame_sha256=_digest(initial_frame),
            opaque={"value": 0},
        )

    def generate_next(self, *, cursor, actions, chunk_index, latent_frames, rgb_frames):
        assert (latent_frames, rgb_frames) == (3, 12)
        self.calls += 1
        value = cursor.opaque["value"]
        weights = np.arange(1, 9, dtype=np.float32)
        frames = []
        for row in actions:
            value = (value + int(row @ weights)) % 251
            frames.append(np.full((2, 2, 3), value, dtype=np.uint8))
        frames = np.stack(frames)
        return CausalChunk(
            parent_state_token=cursor.state_token,
            cursor=RolloutCursor(
                session_id=cursor.session_id,
                next_chunk_index=chunk_index + 1,
                state_token=f"s{chunk_index + 1}:{value}",
                last_frame_sha256=_digest(frames[-1]),
                opaque={"value": value},
            ),
            frames=frames,
        )


def _reference(actions: np.ndarray, anchors) -> np.ndarray:
    value = 0
    frames = {}
    weights = np.arange(1, 9, dtype=np.float32)
    for index, row in enumerate(actions, start=1):
        value = (value + int(row @ weights)) % 251
        frames[index] = np.full((2, 2, 3), value, dtype=np.uint8)
    return np.stack([frames[anchor.frame_index] for anchor in anchors])


def test_one_linked_fake_pipeline_generates_gallery_and_passes_action_gate(tmp_path: Path) -> None:
    base = load_rollout_config(CONFIG)
    config = replace(base, output_root=str(tmp_path), run_id="fixed", width=2, height=2)
    adapter = _LinkedAdapter()
    initial = np.zeros((2, 2, 3), dtype=np.uint8)
    references = {scene.reference_frames_path: scene for scene in config.scenes}
    output, receipt = run_rollout_suite(
        config,
        lineage={"stage": "longforcing_lite_v1", "sha256": "a" * 64},
        adapter_factory=lambda **_kwargs: adapter,
        device="cpu",
        image_loader=lambda _path: initial,
        reference_loader=lambda path, anchors: _reference(
            action_script(references[str(path)]), anchors
        ),
        writer_factory=lambda path, _width, _height, _fps: _Writer(path),
    )
    assert receipt["passed"] is True
    assert adapter.calls == 3 * 3 * NUM_CHUNKS
    assert len(list(output.glob("*.mp4"))) == 9
    assert (output / "metrics.json").is_file() and (output / "index.html").is_file()
    with pytest.raises(FileExistsError, match="overwrite"):
        run_rollout_suite(
            config,
            lineage={},
            adapter_factory=lambda **_kwargs: adapter,
            device="cpu",
        )
