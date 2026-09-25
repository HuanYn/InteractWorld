from __future__ import annotations

import datetime as dt
import os

import pytest

from training import gpu_gate

UUID = "GPU-00000000-0000-0000-0000-000000000001"


def test_fresh_confirmation_and_idle_single_gpu_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    now = dt.datetime(2026, 9, 6, 12, 0, tzinfo=dt.timezone.utc)
    gpu_gate.validate_confirmation("2026-09-06T11:50:01Z", now=now)
    # The public overlay intentionally has a stricter idle-memory threshold.
    idle_memory = gpu_gate.MAX_IDLE_DISPLAY_MEMORY_MIB - 1
    outputs = iter([f"0, {UUID}, {idle_memory}\n", ""])
    monkeypatch.setattr(gpu_gate, "_run_nvidia_smi", lambda _args: next(outputs))
    snapshot = gpu_gate.query_dedicated_gpu(
        confirmed_index=0,
        confirmed_uuid=UUID,
        profile=gpu_gate.DEDICATED_PROFILE,
    )
    assert snapshot.memory_used_mib == idle_memory
    assert snapshot.compute_pids == ()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == UUID


def test_idle_memory_threshold_is_exclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = iter([f"0, {UUID}, {gpu_gate.MAX_IDLE_DISPLAY_MEMORY_MIB}\n", ""])
    monkeypatch.setattr(gpu_gate, "_run_nvidia_smi", lambda _args: next(outputs))
    with pytest.raises(gpu_gate.GpuGateError, match="memory.used"):
        gpu_gate.query_dedicated_gpu(
            confirmed_index=0, confirmed_uuid=UUID, profile=gpu_gate.DEDICATED_PROFILE,
        )


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


OTHER_UUID = "GPU-11111111-1111-1111-1111-111111111111"


def test_shared_selects_fixed_card_while_other_cards_are_busy(monkeypatch):
    outputs = iter([f"0, {OTHER_UUID}, 22000\n2, {UUID}, 499\n", f"{OTHER_UUID}, 9123\n"])
    monkeypatch.setattr(gpu_gate, "_run_nvidia_smi", lambda _args: next(outputs))
    snapshot = gpu_gate.query_shared_gpu(confirmed_index=2, confirmed_uuid=UUID, profile=gpu_gate.SHARED_PROFILE)
    assert snapshot.index == 2 and snapshot.memory_used_mib == 499
    assert snapshot.compute_pids == () and os.environ["CUDA_VISIBLE_DEVICES"] == UUID


@pytest.mark.parametrize("rows,compute,index,uuid,threshold,message", [
    (f"2, {UUID}, 500\n", "", 2, UUID, 500, "memory.used"),
    (f"2, {UUID}, 10\n", f"{UUID}, 123\n", 2, UUID, 500, "compute processes"),
    (f"2, {UUID}, 10\n", "", 1, UUID, 500, "index/UUID"),
    (f"2, {UUID}, 10\n", "", 2, OTHER_UUID, 500, "index/UUID"),
    (f"2, {UUID}, 10\n", "", -1, UUID, 500, "nonnegative integer"),
    (f"2, {UUID}, 10\n", "", True, UUID, 500, "nonnegative integer"),
    (f"2, {UUID}, 10\n", "", 2, UUID, 501, "memory ceiling"),
    (f"2, {UUID}, 400\n", "", 2, UUID, 400, "memory.used"),
    (f"2, {UUID}, -1\n", "", 2, UUID, 500, "negative"),
    (f"2, {UUID}, 10\n2, {OTHER_UUID}, 0\n", "", 2, UUID, 500, "duplicate"),
])
def test_shared_fails_closed_without_changing_cuda_selection(monkeypatch, rows, compute, index, uuid, threshold, message):
    outputs = iter([rows, compute])
    monkeypatch.setattr(gpu_gate, "_run_nvidia_smi", lambda _args: next(outputs))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "unchanged-on-rejection")
    with pytest.raises(gpu_gate.GpuGateError, match=message):
        gpu_gate.query_shared_gpu(confirmed_index=index, confirmed_uuid=uuid,
                                 profile=gpu_gate.SHARED_PROFILE, memory_max_exclusive=threshold)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "unchanged-on-rejection"


def test_allocation_profiles_cannot_cross_gate_boundaries(monkeypatch):
    monkeypatch.setattr(gpu_gate, "_run_nvidia_smi", lambda _args: pytest.fail("invalid profile must fail before GPU queries"))
    for query, wrong_profile in ((gpu_gate.query_dedicated_gpu, gpu_gate.SHARED_PROFILE),
                                 (gpu_gate.query_shared_gpu, gpu_gate.DEDICATED_PROFILE)):
        with pytest.raises(gpu_gate.GpuGateError, match="allocation profile"):
            query(confirmed_index=0, confirmed_uuid=UUID, profile=wrong_profile)


def test_per_call_dedicated_threshold_does_not_modify_global_default(monkeypatch):
    original = gpu_gate.MAX_IDLE_DISPLAY_MEMORY_MIB
    outputs = iter([f"0, {UUID}, 499\n", ""])
    monkeypatch.setattr(gpu_gate, "_run_nvidia_smi", lambda _args: next(outputs))
    gpu_gate.query_dedicated_gpu(confirmed_index=0, confirmed_uuid=UUID,
                                profile=gpu_gate.DEDICATED_PROFILE, memory_max_exclusive=500)
    assert gpu_gate.MAX_IDLE_DISPLAY_MEMORY_MIB == original
