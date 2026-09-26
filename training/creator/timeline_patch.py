"""Small deterministic control-track editor (CPU only, no model trust required).

Intervals are half open: [start_seconds, end_seconds). A replacement edits one
key's entire track, including clearing its old intervals. Every other key is
copied exactly; this says nothing about pixel preservation in regenerated video.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation


FPS = 16
TOTAL_FRAMES = 240
KEYS = ("W", "A", "S", "D", "I", "J", "K", "L")
OPPOSING = (("W", "S"), ("A", "D"), ("I", "K"), ("J", "L"))


def seconds_to_frame(value) -> int:
    """Require an exact, finite, nonnegative 16-FPS time; never round."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("seconds must be a finite number or decimal string")
    representation = str(value)
    if len(representation) > 128:
        raise ValueError("seconds representation is too long")
    try:
        seconds = Decimal(representation)
    except (InvalidOperation, ValueError):
        raise ValueError("invalid seconds") from None
    if not seconds.is_finite() or seconds < 0 or seconds > 15:
        raise ValueError("seconds must be finite and within 0..15")
    if abs(seconds.as_tuple().exponent) > 128:
        raise ValueError("seconds precision is excessive")
    # Decimal multiplication uses the active context precision and could round
    # a tiny nonzero tail away. Integer arithmetic avoids accepting that tail.
    numerator, denominator = seconds.as_integer_ratio()
    numerator *= FPS
    if numerator % denominator:
        raise ValueError("seconds must align exactly to a 1/16-second frame boundary")
    return numerator // denominator


def frame_seconds(frame: int):
    # Multiples of 1/16 are exactly representable in IEEE doubles and JSON.
    return frame // FPS if frame % FPS == 0 else frame / FPS


def intervals_from_rows(rows, key):
    intervals, start = [], None
    for index in range(TOTAL_FRAMES + 1):
        active = index < TOTAL_FRAMES and key in rows[index]
        if active and start is None:
            start = index
        if not active and start is not None:
            intervals.append((start, index))
            start = None
    return intervals


def edits_from_intervals(intervals_by_key):
    return [dict(op="replace_intervals", key=key, intervals=[
        dict(start_seconds=frame_seconds(start), end_seconds=frame_seconds(end))
        for start, end in intervals_by_key[key]
    ]) for key in KEYS if key in intervals_by_key]


def compile_edits(edits, original=None):
    """Return canonical rows and metadata; reject unknown fields and conflicts."""
    if not isinstance(edits, list) or not 1 <= len(edits) <= len(KEYS):
        raise ValueError("edits must contain 1..8 replace_intervals operations")
    if original is not None and len(original) != TOTAL_FRAMES:
        raise ValueError("original timeline must have exactly 240 frames")
    rows = [set(row) for row in original] if original is not None else [set() for _ in range(TOTAL_FRAMES)]
    intervals_by_key = {}
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {"op", "key", "intervals"}:
            raise ValueError("invalid edit fields")
        if edit["op"] != "replace_intervals" or edit["key"] not in KEYS:
            raise ValueError("unsupported edit operation or control key")
        key = edit["key"]
        if key in intervals_by_key:
            raise ValueError("duplicate key edits are not supported")
        values = edit["intervals"]
        if not isinstance(values, list) or len(values) > TOTAL_FRAMES:
            raise ValueError("invalid intervals")
        intervals = []
        for interval in values:
            if not isinstance(interval, dict) or set(interval) != {"start_seconds", "end_seconds"}:
                raise ValueError("invalid interval fields")
            start = seconds_to_frame(interval["start_seconds"])
            end = seconds_to_frame(interval["end_seconds"])
            if end <= start:
                raise ValueError("interval end must be later than start")
            if intervals and start < intervals[-1][1]:
                raise ValueError("intervals must be ordered and non-overlapping")
            # Adjacent intervals have the same control meaning; normalize them.
            if intervals and start == intervals[-1][1]:
                intervals[-1] = (intervals[-1][0], end)
            else:
                intervals.append((start, end))
        intervals_by_key[key] = intervals
        for row in rows:
            row.discard(key)
        for start, end in intervals:
            for row in rows[start:end]:
                row.add(key)
    if any(a in row and b in row for row in rows for a, b in OPPOSING):
        raise ValueError("edit creates opposing simultaneous controls")
    canonical = [tuple(key for key in KEYS if key in row) for row in rows]
    metadata = dict(schema_version=1, fps=FPS, total_frames=TOTAL_FRAMES,
                    edits=edits_from_intervals(intervals_by_key),
                    protected_keys=[key for key in KEYS if key not in intervals_by_key])
    return canonical, metadata
