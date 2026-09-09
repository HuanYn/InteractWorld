"""Strict canonical action schema and temporal eligibility checks."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

ACTION_KEYS: tuple[str, ...] = ("W", "A", "S", "D", "I", "J", "K", "L")
ACTION_KEY_SET = frozenset(ACTION_KEYS)
# The released annotations contain these controls in addition to the model's
# canonical eight keys. They are safe to drop only when inactive.
INACTIVE_ONLY_KEYS = frozenset(("Q", "E", "SPACE"))
SUPPORTED_WINDOWS: tuple[int, ...] = (49, 241)
OFFICIAL_SOURCE_FPS = 30
TRAINING_OUTPUT_FPS = 16
_FRAME_ID_SUFFIX = re.compile(r"(?:^|_)([0-9]+)$")


class ActionSchemaError(ValueError):
    """Raised when an action file cannot be represented without information loss."""


@dataclass(frozen=True)
class CanonicalActionFrame:
    frame_id: int
    keys: tuple[int, ...]
    timestamp: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "keys": dict(zip(ACTION_KEYS, self.keys, strict=True)),
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class CanonicalActionSequence:
    fps: int
    frames: tuple[CanonicalActionFrame, ...]

    @property
    def total_frames(self) -> int:
        return len(self.frames)

    def eligible_window_counts(
        self, window_sizes: Sequence[int] = SUPPORTED_WINDOWS
    ) -> dict[int, int]:
        """Count all windows fully contained in a run of consecutive frame ids."""
        sizes = tuple(int(size) for size in window_sizes)
        if any(size <= 0 for size in sizes):
            raise ValueError("window sizes must be positive")
        if not self.frames:
            return {size: 0 for size in sizes}

        run_lengths: list[int] = []
        run_length = 1
        expected_period = 1.0 / self.fps
        timestamp_tolerance = max(1e-3, expected_period * 0.10)
        for previous, current in zip(self.frames, self.frames[1:]):
            timestamps_contiguous = (
                previous.timestamp is None
                or current.timestamp is None
                or abs((current.timestamp - previous.timestamp) - expected_period)
                <= timestamp_tolerance
            )
            if current.frame_id == previous.frame_id + 1 and timestamps_contiguous:
                run_length += 1
            else:
                run_lengths.append(run_length)
                run_length = 1
        run_lengths.append(run_length)
        return {
            size: sum(max(0, length - size + 1) for length in run_lengths)
            for size in sizes
        }

    def eligible_window_starts(self, window_size: int) -> tuple[int, ...]:
        """Return sequence indices at which a fully continuous window begins."""
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if window_size > len(self.frames):
            return ()
        expected_period = 1.0 / self.fps
        tolerance = max(1e-3, expected_period * 0.10)
        starts: list[int] = []
        run_start = 0
        for index in range(1, len(self.frames) + 1):
            at_end = index == len(self.frames)
            if not at_end:
                previous = self.frames[index - 1]
                current = self.frames[index]
                timestamp_ok = (
                    previous.timestamp is None
                    or current.timestamp is None
                    or abs((current.timestamp - previous.timestamp) - expected_period) <= tolerance
                )
                if current.frame_id == previous.frame_id + 1 and timestamp_ok:
                    continue
            run_length = index - run_start
            starts.extend(range(run_start, run_start + max(0, run_length - window_size + 1)))
            run_start = index
        return tuple(starts)

    def resampled_offsets(
        self,
        output_frames: int,
        *,
        output_fps: int = TRAINING_OUTPUT_FPS,
    ) -> tuple[int, ...]:
        """Map an output-rate clip to nearest source-frame offsets.

        Official Explorer episodes are 30 FPS while the training contract is
        16 FPS.  Keeping this mapping integer and explicit prevents a 49-frame
        clip from silently shrinking from three seconds to 1.6 seconds.
        """
        if output_frames <= 0 or output_fps <= 0:
            raise ValueError("output_frames and output_fps must be positive")
        return tuple(
            (2 * index * self.fps + output_fps) // (2 * output_fps)
            for index in range(output_frames)
        )

    def eligible_resampled_window_starts(
        self,
        output_frames: int,
        *,
        output_fps: int = TRAINING_OUTPUT_FPS,
    ) -> tuple[int, ...]:
        """Return source indices whose resampled output window is continuous."""
        offsets = self.resampled_offsets(output_frames, output_fps=output_fps)
        source_span = offsets[-1] + 1
        starts: list[int] = []
        for raw_start in self.eligible_window_starts(source_span):
            # eligible_window_starts already proves the complete source span;
            # every selected offset is therefore present and timestamp-aligned.
            starts.append(raw_start)
        return tuple(starts)

    def eligible_resampled_window_counts(
        self,
        window_sizes: Sequence[int] = SUPPORTED_WINDOWS,
        *,
        output_fps: int = TRAINING_OUTPUT_FPS,
    ) -> dict[int, int]:
        return {
            int(size): len(
                self.eligible_resampled_window_starts(int(size), output_fps=output_fps)
            )
            for size in window_sizes
        }


def _binary_switch(value: object, *, key: str, frame_id: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    raise ActionSchemaError(
        f"frame {frame_id}: action {key!r} must be bool or integer 0/1, "
        f"got {value!r}"
    )


def _frame_id(raw: object, fallback: int) -> int:
    if raw is None:
        return fallback
    if isinstance(raw, bool):
        raise ActionSchemaError("frame_id cannot be boolean")
    if isinstance(raw, int) and raw >= 0:
        return raw
    if isinstance(raw, str) and raw.isascii():
        match = _FRAME_ID_SUFFIX.search(raw)
        if match:
            return int(match.group(1))
    raise ActionSchemaError(f"invalid frame_id: {raw!r}")


def _timestamp(raw: object, *, frame_id: int) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ActionSchemaError(f"frame {frame_id}: invalid timestamp {raw!r}")
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ActionSchemaError(f"frame {frame_id}: invalid timestamp {raw!r}")
    return value


def canonicalize_keys(keys: Mapping[str, object], *, frame_id: int) -> tuple[int, ...]:
    """Return canonical W/A/S/D/I/J/K/L values, rejecting every unknown key."""
    if not isinstance(keys, Mapping):
        raise ActionSchemaError(f"frame {frame_id}: keys must be an object")
    normalized: dict[str, object] = {}
    for raw_key, value in keys.items():
        if not isinstance(raw_key, str):
            raise ActionSchemaError(f"frame {frame_id}: action key must be a string")
        key = raw_key.upper()
        if key in INACTIVE_ONLY_KEYS:
            if _binary_switch(value, key=key, frame_id=frame_id):
                raise ActionSchemaError(
                    f"frame {frame_id}: unsupported active action key {raw_key!r}"
                )
            continue
        if key not in ACTION_KEY_SET:
            raise ActionSchemaError(f"frame {frame_id}: unknown action key {raw_key!r}")
        if key in normalized:
            raise ActionSchemaError(f"frame {frame_id}: duplicate action key {key!r}")
        normalized[key] = value
    return tuple(
        _binary_switch(normalized.get(key, False), key=key, frame_id=frame_id)
        for key in ACTION_KEYS
    )


def parse_action_document(document: Mapping[str, object]) -> CanonicalActionSequence:
    """Validate an official-style ``action.json`` document.

    Gaps are retained so callers can reject long windows without silently treating a
    missing frame as a no-op. Frame order must be strictly increasing.
    """
    if not isinstance(document, Mapping):
        raise ActionSchemaError("action document must be an object")
    raw_fps = document.get("fps")
    if (
        isinstance(raw_fps, bool)
        or not isinstance(raw_fps, (int, float))
        or raw_fps <= 0
        or not float(raw_fps).is_integer()
    ):
        raise ActionSchemaError(f"fps must be a positive whole number, got {raw_fps!r}")
    fps = int(raw_fps)
    raw_frames = document.get("frames")
    if not isinstance(raw_frames, list):
        raise ActionSchemaError("frames must be a list")
    declared_total = document.get("total_frames", len(raw_frames))
    if (
        isinstance(declared_total, bool)
        or not isinstance(declared_total, int)
        or declared_total != len(raw_frames)
    ):
        raise ActionSchemaError(
            f"total_frames={declared_total!r} does not match {len(raw_frames)} frames"
        )

    frames: list[CanonicalActionFrame] = []
    previous_id = -1
    previous_timestamp: float | None = None
    for index, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, Mapping):
            raise ActionSchemaError(f"frame {index}: must be an object")
        current_id = _frame_id(raw_frame.get("frame_id"), index)
        if current_id <= previous_id:
            raise ActionSchemaError("frame_id values must be strictly increasing")
        current_timestamp = _timestamp(raw_frame.get("timestamp"), frame_id=current_id)
        if (
            current_timestamp is not None
            and previous_timestamp is not None
            and current_timestamp <= previous_timestamp
        ):
            raise ActionSchemaError("timestamp values must be strictly increasing")
        frames.append(
            CanonicalActionFrame(
                frame_id=current_id,
                keys=canonicalize_keys(raw_frame.get("keys", {}), frame_id=current_id),
                timestamp=current_timestamp,
            )
        )
        previous_id = current_id
        previous_timestamp = current_timestamp
    return CanonicalActionSequence(fps=fps, frames=tuple(frames))
