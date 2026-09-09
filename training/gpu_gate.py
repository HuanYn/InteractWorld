"""Fail-closed authorization gate for the dedicated local RTX 5090."""

from __future__ import annotations

import datetime as dt
import os
import subprocess
from dataclasses import asdict, dataclass

MAX_CONFIRMATION_AGE_SECONDS = 900
MAX_IDLE_DISPLAY_MEMORY_MIB = 4096
DEDICATED_PROFILE = "dedicated_local_single_gpu"


class GpuGateError(RuntimeError):
    pass


@dataclass(frozen=True)
class GpuSnapshot:
    index: int
    uuid: str
    memory_used_mib: int
    compute_pids: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalized_uuid(value: str) -> str:
    text = value.strip().lower()
    if not text.startswith("gpu-"):
        text = f"gpu-{text}"
    body = text[4:]
    compact = body.replace("-", "")
    if len(compact) != 32 or any(char not in "0123456789abcdef" for char in compact):
        raise GpuGateError(f"invalid GPU UUID: {value!r}")
    return f"gpu-{body}"


def validate_confirmation(confirmed_at_utc: str, *, now: dt.datetime | None = None) -> None:
    try:
        confirmed = dt.datetime.fromisoformat(confirmed_at_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GpuGateError("confirmed-at-utc must be ISO-8601 UTC") from exc
    if confirmed.tzinfo is None or confirmed.utcoffset() != dt.timedelta(0):
        raise GpuGateError("confirmed-at-utc must include UTC timezone")
    current = now or dt.datetime.now(dt.timezone.utc)
    age = (current - confirmed).total_seconds()
    if age < -60 or age > MAX_CONFIRMATION_AGE_SECONDS:
        raise GpuGateError(
            f"GPU confirmation is not fresh: age={age:.1f}s, "
            f"allowed <= {MAX_CONFIRMATION_AGE_SECONDS}s"
        )


def _run_nvidia_smi(args: list[str]) -> str:
    completed = subprocess.run(
        ["nvidia-smi", *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return completed.stdout


def query_dedicated_gpu(
    *,
    confirmed_index: int,
    confirmed_uuid: str,
    profile: str,
) -> GpuSnapshot:
    if profile != DEDICATED_PROFILE:
        raise GpuGateError(f"allocation profile must be {DEDICATED_PROFILE!r}")
    rows = [
        line.strip()
        for line in _run_nvidia_smi(
            ["--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"]
        ).splitlines()
        if line.strip()
    ]
    if len(rows) != 1:
        raise GpuGateError(f"dedicated profile requires exactly one visible physical GPU, got {len(rows)}")
    parts = [part.strip() for part in rows[0].split(",")]
    if len(parts) != 3:
        raise GpuGateError("unexpected nvidia-smi GPU inventory format")
    try:
        index, memory = int(parts[0]), int(parts[2])
    except ValueError as exc:
        raise GpuGateError("invalid numeric field in GPU inventory") from exc
    uuid = _normalized_uuid(parts[1])
    expected_uuid = _normalized_uuid(confirmed_uuid)
    if index != confirmed_index or uuid != expected_uuid:
        raise GpuGateError("visible GPU index/UUID differs from the fresh confirmation")

    compute_rows = [
        line.strip()
        for line in _run_nvidia_smi(
            ["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"]
        ).splitlines()
        if line.strip()
    ]
    pids: list[int] = []
    for row in compute_rows:
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 2:
            raise GpuGateError("unexpected nvidia-smi compute inventory format")
        if _normalized_uuid(fields[0]) == uuid:
            try:
                pids.append(int(fields[1]))
            except ValueError as exc:
                raise GpuGateError("invalid compute PID in GPU inventory") from exc
    if pids:
        raise GpuGateError(f"dedicated GPU already has compute processes: {sorted(pids)}")
    if memory >= MAX_IDLE_DISPLAY_MEMORY_MIB:
        raise GpuGateError(
            f"dedicated GPU memory.used={memory} MiB is not below "
            f"{MAX_IDLE_DISPLAY_MEMORY_MIB} MiB"
        )
    snapshot = GpuSnapshot(index=index, uuid=uuid, memory_used_mib=memory, compute_pids=())
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # CUDA accepts the canonical, case-sensitive ``GPU-`` UUID prefix.  Keep
    # comparisons normalized, but expose the UUID in CUDA's canonical form.
    os.environ["CUDA_VISIBLE_DEVICES"] = f"GPU-{uuid[4:]}"
    return snapshot
