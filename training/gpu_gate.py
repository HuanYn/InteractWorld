"""Fail-closed physical GPU gates for explicit dedicated/shared allocations."""

from __future__ import annotations

import datetime as dt
import os
import subprocess
from dataclasses import asdict, dataclass

MAX_CONFIRMATION_AGE_SECONDS = 900
MAX_IDLE_DISPLAY_MEMORY_MIB = 4096
MAX_SHARED_MEMORY_MIB = 500
DEDICATED_PROFILE = "dedicated_local_single_gpu"
SHARED_PROFILE = "shared_server_single_gpu"


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
    if not isinstance(value, str):
        raise GpuGateError("GPU UUID must be a string")
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


def _gpu_inventory() -> list[tuple[int, str, int]]:
    rows = [
        line.strip()
        for line in _run_nvidia_smi(
            ["--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"]
        ).splitlines()
        if line.strip()
    ]
    inventory = []
    for row in rows:
        parts = [part.strip() for part in row.split(",")]
        if len(parts) != 3:
            raise GpuGateError("unexpected nvidia-smi GPU inventory format")
        try:
            index, memory = int(parts[0]), int(parts[2])
        except ValueError as exc:
            raise GpuGateError("invalid numeric field in GPU inventory") from exc
        if index < 0 or memory < 0:
            raise GpuGateError("negative numeric field in GPU inventory")
        inventory.append((index, _normalized_uuid(parts[1]), memory))
    if len({row[0] for row in inventory}) != len(inventory) or len({row[1] for row in inventory}) != len(inventory):
        raise GpuGateError("duplicate index/UUID in GPU inventory")
    return inventory


def _idle_snapshot(index: int, uuid: str, memory: int, *, threshold: int, label: str) -> GpuSnapshot:
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
        raise GpuGateError(f"{label} GPU already has compute processes: {sorted(pids)}")
    if memory >= threshold:
        raise GpuGateError(
            f"{label} GPU memory.used={memory} MiB is not below {threshold} MiB"
        )
    snapshot = GpuSnapshot(index=index, uuid=uuid, memory_used_mib=memory, compute_pids=())
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # CUDA accepts the canonical, case-sensitive ``GPU-`` UUID prefix.  Keep
    # comparisons normalized, but expose the UUID in CUDA's canonical form.
    os.environ["CUDA_VISIBLE_DEVICES"] = f"GPU-{uuid[4:]}"
    return snapshot


def query_dedicated_gpu(
    *, confirmed_index: int, confirmed_uuid: str, profile: str,
    memory_max_exclusive: int | None = None,
) -> GpuSnapshot:
    """Keep the dedicated contract: exactly one physical GPU must be visible."""
    if profile != DEDICATED_PROFILE:
        raise GpuGateError(f"allocation profile must be {DEDICATED_PROFILE!r}")
    if type(confirmed_index) is not int or confirmed_index < 0:
        raise GpuGateError("confirmed GPU index must be a nonnegative integer")
    expected_uuid = _normalized_uuid(confirmed_uuid)
    threshold = MAX_IDLE_DISPLAY_MEMORY_MIB if memory_max_exclusive is None else memory_max_exclusive
    if type(threshold) is not int or not 1 <= threshold <= MAX_IDLE_DISPLAY_MEMORY_MIB:
        raise GpuGateError("invalid dedicated memory ceiling")
    inventory = _gpu_inventory()
    if len(inventory) != 1:
        raise GpuGateError(f"dedicated profile requires exactly one visible physical GPU, got {len(inventory)}")
    index, uuid, memory = inventory[0]
    if index != confirmed_index or uuid != expected_uuid:
        raise GpuGateError("visible GPU index/UUID differs from the fresh confirmation")
    return _idle_snapshot(index, uuid, memory, threshold=threshold, label="dedicated")


def query_shared_gpu(
    *, confirmed_index: int, confirmed_uuid: str, profile: str,
    memory_max_exclusive: int = MAX_SHARED_MEMORY_MIB,
) -> GpuSnapshot:
    """Select one explicitly authorized card on a server with multiple GPUs.

    Other cards may be occupied. The selected physical index AND UUID must match,
    its memory must be below 500 MiB (or a stricter authorized threshold), and it
    must have no compute process. This function never chooses a free card itself.
    """
    if profile != SHARED_PROFILE:
        raise GpuGateError(f"allocation profile must be {SHARED_PROFILE!r}")
    if type(confirmed_index) is not int or confirmed_index < 0:
        raise GpuGateError("confirmed GPU index must be a nonnegative integer")
    expected_uuid = _normalized_uuid(confirmed_uuid)
    if type(memory_max_exclusive) is not int or not 1 <= memory_max_exclusive <= MAX_SHARED_MEMORY_MIB:
        raise GpuGateError("shared memory ceiling must be an integer in 1..500 MiB")
    selected = [row for row in _gpu_inventory() if row[0] == confirmed_index]
    if len(selected) != 1 or selected[0][1] != expected_uuid:
        raise GpuGateError("selected GPU index/UUID differs from the explicit authorization")
    index, uuid, memory = selected[0]
    return _idle_snapshot(index, uuid, memory, threshold=memory_max_exclusive, label="shared")
