"""Small CPU-only invariants; no models, video caches, or CUDA are loaded."""

import copy

import pytest
import torch

from training.error_recycling import ContextErrorRecycling, ErrorRecyclingConfig


def case(frames=13, sigma_value=0.5):
    clean = torch.arange(frames * 8, dtype=torch.float32).reshape(1, frames, 2, 2, 2) / 100
    target = torch.full_like(clean, 0.25)
    target[:, 0] = 0
    sigma = torch.full((1, frames), sigma_value)
    sigma[:, 0] = 0
    noisy = clean + sigma[..., None, None, None] * target
    prediction = (target + 0.5).requires_grad_()
    return clean, sigma, noisy, target, prediction


def recycling(**kwargs):
    options = dict(enabled=True, warmup_observations=0, context_inject_prob=1,
                   clean_prob=0, error_scale=1)
    options.update(kwargs)
    return ContextErrorRecycling(ErrorRecyclingConfig(**options))


def finish(bank, values):
    clean, sigma, noisy, target, prediction = values
    return bank.observe(noisy, target, prediction, sigma, clean=clean)


@pytest.mark.parametrize("options", [{"enabled": False}, {"error_scale": 0},
                                     {"context_inject_prob": 0}, {"clean_prob": 1}])
def test_zero_perturbation_matches_clean_without_input_or_rng_changes(options):
    bank, values = recycling(**options), case()
    clean, sigma, noisy, target, prediction = values
    originals = [tensor.detach().clone() for tensor in values]
    global_rng = torch.get_rng_state().clone()
    for _ in range(3):
        context, receipt = bank.prepare_context(clean, sigma)
        assert torch.equal(context, clean) and not receipt["injected"]
        finish(bank, values)
    assert all(torch.equal(actual, expected) for actual, expected in zip(values, originals))
    assert torch.equal(torch.get_rng_state(), global_rng)


@pytest.mark.parametrize("frames", [13, 25])
def test_old_errors_only_correct_sign_protected_first_and_detached_cpu_storage(frames):
    bank, values = recycling(), case(frames)
    clean, sigma, noisy, target, prediction = values
    context, receipt = bank.prepare_context(clean.requires_grad_(), sigma)
    assert torch.equal(context, clean) and not context.requires_grad
    assert not receipt["injected"]
    finish(bank, values)
    state = bank.state_dict()
    assert state["observations"] == 1
    for bucket in state["buckets"]:
        for entry in bucket:
            error = entry["error"]
            assert error.device.type == "cpu" and error.dtype == torch.bfloat16
            assert not error.requires_grad and error.grad_fn is None
            assert tuple(error.shape) == (3, 2, 2, 2)
            assert torch.allclose(error.float(), torch.full_like(error.float(), -0.25), atol=0.002)
    replayed, receipt = bank.prepare_context(clean, sigma)
    assert receipt["injected"] and receipt["injected_blocks"] == (frames - 1) // 3
    assert all(entry_id < state["next_id"] for entry_id in receipt["source_entry_ids"])
    assert torch.equal(replayed[:, 0], clean[:, 0])
    assert torch.allclose(replayed[:, 1:], clean[:, 1:] - 0.25, atol=0.002)
    assert not replayed.requires_grad and replayed.grad_fn is None
    finish(bank, values)
    assert clean.grad is None and prediction.grad is None
    assert bank.metrics()["total_injected_forwards"] == 1
    assert bank.metrics()["total_injected_blocks"] == (frames - 1) // 3


def test_implicit_gt_reconstruction_matches_explicit_clean():
    first, second, values = recycling(), recycling(), case()
    clean, sigma, noisy, target, prediction = values
    first.prepare_context(clean, sigma)
    second.prepare_context(clean, sigma)
    first.observe(noisy, target, prediction, sigma)
    second.observe(noisy, target, prediction, sigma, clean=clean)
    left, _ = first.prepare_context(clean, sigma)
    right, _ = second.prepare_context(clean, sigma)
    assert torch.equal(left, right)


def test_checkpoint_replays_identical_contexts_and_private_rng():
    bank, values = recycling(context_inject_prob=0.7, clean_prob=0.2), case()
    for _ in range(3):
        bank.prepare_context(*values[:2])
        finish(bank, values)
    state = bank.state_dict()
    resumed = recycling(context_inject_prob=0.7, clean_prob=0.2)
    resumed.load_state_dict(state)
    global_rng = torch.get_rng_state().clone()
    for _ in range(12):
        expected, expected_receipt = bank.prepare_context(*values[:2])
        actual, actual_receipt = resumed.prepare_context(*values[:2])
        assert torch.equal(expected, actual) and expected_receipt == actual_receipt
        assert finish(bank, values) == finish(resumed, values)
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert torch.equal(bank.state_dict()["rng_state"], resumed.state_dict()["rng_state"])


def test_warmup_is_observation_count_and_checkpoint_does_not_alias_buffer():
    bank, values = recycling(warmup_observations=2), case()
    for _ in range(2):
        context, receipt = bank.prepare_context(*values[:2])
        assert receipt["reason"] == "warmup" and torch.equal(context, values[0])
        finish(bank, values)
    state = bank.state_dict()
    for bucket in state["buckets"]:
        for entry in bucket:
            entry["error"].zero_()
    _, receipt = bank.prepare_context(*values[:2])
    assert receipt["injected"] and receipt["error_rms"] > 0
    finish(bank, values)


def test_byte_and_bucket_limits_apply_across_observations_and_restore():
    # Every 3-frame BF16 item is 48 bytes: at most two items in the live bank.
    bank = recycling(max_buffer_bytes=96, entries_per_bucket=1)
    for sigma in (0.1, 0.3, 0.5, 0.7, 0.9):
        values = case(sigma_value=sigma)
        bank.prepare_context(*values[:2])
        metrics = finish(bank, values)
        state = bank.state_dict()
        assert metrics["buffer_bytes"] <= 96
        assert metrics["buffer_entries"] <= 2
        assert all(len(bucket) <= 1 for bucket in state["buckets"])
    assert metrics["total_dropped"] > 0
    restored = recycling(max_buffer_bytes=96, entries_per_bucket=1)
    restored.load_state_dict(state)
    assert restored.metrics() == bank.metrics()


def test_oversize_item_is_skipped_without_breaking_hard_bound():
    bank, values = recycling(max_buffer_bytes=1), case()
    bank.prepare_context(*values[:2])
    result = finish(bank, values)
    assert result["buffer_bytes"] == result["buffer_entries"] == 0
    assert result["total_dropped"] == 4


def test_prepare_observe_order_and_partial_checkpoint_are_rejected():
    bank, values = recycling(), case()
    with pytest.raises(RuntimeError, match="prepare_context must precede"):
        finish(bank, values)
    bank.prepare_context(*values[:2])
    with pytest.raises(RuntimeError, match="previous prepare_context"):
        bank.prepare_context(*values[:2])
    with pytest.raises(RuntimeError, match="incomplete"):
        bank.state_dict()
    finish(bank, values)
    assert bank.state_dict()["observations"] == 1


@pytest.mark.parametrize("mutation", ["config", "nan", "grad", "bytes", "rng"])
def test_invalid_restore_is_rejected_without_mutating_live_state(mutation):
    bank, values = recycling(max_buffer_bytes=192), case()
    bank.prepare_context(*values[:2])
    finish(bank, values)
    before = bank.metrics()
    state = copy.deepcopy(bank.state_dict())
    bucket = next(bucket for bucket in state["buckets"] if bucket)
    if mutation == "config":
        state["config"]["seed"] += 1
    elif mutation == "nan":
        bucket[0]["error"].fill_(float("nan"))
    elif mutation == "grad":
        bucket[0]["error"].requires_grad_()
    elif mutation == "bytes":
        state["next_id"] += 1
        bucket.append({"id": state["next_id"] - 1, "error": bucket[0]["error"].clone()})
    else:
        state["rng_state"] = "invalid"
    with pytest.raises(ValueError):
        bank.load_state_dict(state)
    assert bank.metrics() == before


def test_raw_timesteps_and_nonfinite_errors_are_rejected():
    bank, values = recycling(), case()
    with pytest.raises(ValueError, match="sigma must be finite"):
        bank.prepare_context(values[0], values[1] * 1000)
    bank.prepare_context(*values[:2])
    bad_pred = values[4].detach().clone()
    bad_pred[:, 1] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        bank.observe(values[2], values[3], bad_pred, values[1])
    assert bank.metrics()["buffer_bytes"] == 0
