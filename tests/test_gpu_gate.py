from __future__ import annotations

import datetime as dt
import os

import pytest

from training import gpu_gate

UUID = "GPU-00000000-0000-0000-0000-000000000001"


def test_fresh_confirmation_and_idle_single_gpu_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    now = dt.datetime(2026, 9, 6, 12, 0, tzinfo=dt.timezone.utc)
    gpu_gate.validate_confirmation("2026-09-06T11:50:01Z", now=now)
    outputs = iter([f"0, {UUID}, 623\n", ""])
    monkeypatch.setattr(gpu_gate, "_run_nvidia_smi", lambda _args: next(outputs))
    snapshot = gpu_gate.query_dedicated_gpu(
        confirmed_index=0,
        confirmed_uuid=UUID,
        profile=gpu_gate.DEDICATED_PROFILE,
    )
    assert snapshot.memory_used_mib == 623
    assert snapshot.compute_pids == ()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == UUID


def test_stale_confirmation_is_rejected() -> None:
    now = dt.datetime(2026, 9, 6, 12, 0, tzinfo=dt.timezone.utc)
    with pytest.raises(gpu_gate.GpuGateError, match="not fresh"):
        gpu_gate.validate_confirmation("2026-09-06T11:44:59Z", now=now)


@pytest.mark.parametrize(
    ("gpu_rows", "compute_rows", "message"),
    [
        ("0, GPU-00000000-0000-0000-0000-000000000001, 4096\n", "", "memory.used"),
        (
            "0, GPU-00000000-0000-0000-0000-000000000001, 623\n",
            "GPU-00000000-0000-0000-0000-000000000001, 1234\n",
            "compute processes",
        ),
        (
            "0, GPU-00000000-0000-0000-0000-000000000001, 623\n"
            "1, GPU-11111111-1111-1111-1111-111111111111, 1\n",
            "",
            "exactly one",
        ),
    ],
)
def test_busy_or_multi_gpu_inventory_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    gpu_rows: str,
    compute_rows: str,
    message: str,
) -> None:
    outputs = iter([gpu_rows, compute_rows])
    monkeypatch.setattr(gpu_gate, "_run_nvidia_smi", lambda _args: next(outputs))
    with pytest.raises(gpu_gate.GpuGateError, match=message):
        gpu_gate.query_dedicated_gpu(
            confirmed_index=0,
            confirmed_uuid=UUID,
            profile=gpu_gate.DEDICATED_PROFILE,
        )
