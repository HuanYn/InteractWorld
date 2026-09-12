"""Small CPU integration checks; no model weights, VAE, CUDA, or videos."""
from __future__ import annotations

import contextlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from training.causal_tf import (
    CausalTeacherForcingConfig, ERROR_RECYCLING_STAGE_NAME, STAGE_NAME,
    assert_no_future_leakage, causal_teacher_forcing_loss, error_recycling_contract,
    latent_shape, load_causal_config, stage_name, teacher_forcing_visibility,
)
from training.error_recycling import ContextErrorRecycling, ErrorRecyclingConfig
from train_causal_teacher_forcing import (
    _apply_overrides, _checkpoint_payload, _restore_resume, _resume_iterator_preserving_rng, parse_args,
)


def config_for(frames=97, enabled=True):
    config = CausalTeacherForcingConfig()
    config.data.num_frames = frames
    if frames == 97:
        config.data.data_factory = "training.data.action_resampled:build_resampled_action_teacher_dataloader"
    config.training.checkpoint_every = 20
    config.error_recycling = ErrorRecyclingConfig(
        enabled=enabled, warmup_observations=0, context_inject_prob=1,
        clean_prob=0, error_scale=0.25,
    )
    return config


def tiny_batch(config):
    config.data.height = config.data.width = 32
    config.model.latent_channels = 4
    rng = torch.Generator().manual_seed(17)
    shape = latent_shape(config)
    clean = torch.randn(shape, generator=rng)
    target = torch.randn(shape, generator=rng)
    target[:, 0] = 0
    timestep = torch.full(shape[:2], 500.0)
    timestep[:, 0] = 0
    return {
        "noisy_latents": clean + (timestep / 1000)[..., None, None, None] * target,
        "target_flow": target, "timesteps": timestep,
        "prompt_embeds": torch.zeros(1, 2, 16),
        "actions": torch.zeros(1, config.data.num_frames - 1, 8),
    }


class Flow(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.inputs = []

    def forward(self, noisy, **kwargs):
        self.inputs.append((noisy.detach().clone(), kwargs))
        prediction = torch.ones_like(noisy) * self.weight
        # A conspicuously wrong first-frame prediction must remain excluded.
        prediction = torch.cat([prediction[:, :1] + 100, prediction[:, 1:]], dim=1)
        return prediction, None


def test_off_serialization_and_method_are_legacy_compatible(tmp_path):
    import yaml
    config = config_for(49, False)
    assert stage_name(config) == STAGE_NAME
    assert "error_recycling" not in config.to_dict()
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
    restored = load_causal_config(path)
    assert not restored.error_recycling.enabled
    assert restored.to_dict() == config.to_dict()


@pytest.mark.parametrize("frames,latents", [(49, 13), (97, 25)])
def test_explicit_supported_geometry_and_no_future_mask_leakage(frames, latents):
    config = config_for(frames)
    config.validate()
    assert latent_shape(config)[1] == latents
    assert stage_name(config) == ERROR_RECYCLING_STAGE_NAME
    mask = teacher_forcing_visibility(latents)
    assert_no_future_leakage(mask, num_frames=latents)
    for frame in range(1, latents):
        start = 1 + ((frame - 1) // 3) * 3
        assert not mask[latents + frame, start:latents].any()


def test_97_requires_resampled_factory_and_twenty_step_checkpoints():
    config = config_for()
    config.data.data_factory = "training.data.action_dataset:build_action_teacher_dataloader"
    with pytest.raises(ValueError, match="absolute-index resampled"):
        config.validate()
    config = config_for()
    config.training.checkpoint_every = 50
    with pytest.raises(ValueError, match="every 20"):
        config.validate()


@pytest.mark.parametrize("frames", [49, 97])
def test_real_recycling_changes_only_old_context_and_preserves_loss_inputs(frames):
    config = config_for(frames)
    batch = tiny_batch(config)
    pristine = {name: tensor.clone() for name, tensor in batch.items()}
    model = Flow()
    recycler = ContextErrorRecycling(config.error_recycling)
    global_rng = torch.get_rng_state().clone()
    first_metrics, second_metrics = {}, {}
    first_loss = causal_teacher_forcing_loss(
        model, batch, config, torch.device("cpu"), recycler=recycler, recycling_metrics=first_metrics)
    second_loss = causal_teacher_forcing_loss(
        model, batch, config, torch.device("cpu"), recycler=recycler, recycling_metrics=second_metrics)
    assert not first_metrics["prepare"]["injected"]
    assert second_metrics["prepare"]["injected"]
    assert recycler.metrics()["observations"] == 2
    assert torch.equal(global_rng, torch.get_rng_state())
    assert torch.equal(first_loss, second_loss)
    second_loss.backward()
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad)
    noisy_first, first = model.inputs[0]
    noisy_second, second = model.inputs[1]
    assert torch.equal(noisy_first, noisy_second)
    assert torch.equal(first["clean_x"][:, :1], second["clean_x"][:, :1])
    assert not torch.equal(first["clean_x"][:, 1:], second["clean_x"][:, 1:])
    assert torch.equal(first["timestep"], second["timestep"])
    assert torch.equal(first["aug_t"], second["aug_t"])
    for key in ("act_context", "prompt_embeds"):
        assert all(torch.equal(a, b) for a, b in zip(
            first["conditional_dict"][key], second["conditional_dict"][key], strict=True))
    assert all(torch.equal(batch[name], pristine[name]) for name in batch)


def test_enabled_runtime_cannot_silently_skip_recycling():
    config = config_for()
    batch = tiny_batch(config)
    with pytest.raises(ValueError, match="enabled together"):
        causal_teacher_forcing_loss(Flow(), batch, config, torch.device("cpu"))


def checkpoint_fixture(tmp_path):
    config = config_for()
    batch = tiny_batch(config)
    recycler, model = ContextErrorRecycling(config.error_recycling), Flow()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    for _ in range(8):
        loss = causal_teacher_forcing_loss(model, batch, config, torch.device("cpu"), recycler=recycler)
        (loss / 8).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    payload = _checkpoint_payload(model, optimizer, config=config, step=1, micro_batches_consumed=8,
        metrics={"loss": float(loss.detach())}, manifest_hashes={"training_config": "frozen"},
        teacher_lineage={"stage": "action_teacher_lora_v1", "step": 120}, recycler=recycler)
    path = tmp_path / "resume.pt"
    torch.save(payload, path)
    return config, batch, recycler, model, optimizer, payload, path


def test_resume_restores_buffer_optimizer_global_rng_and_next_context_exactly(tmp_path):
    config, batch, recycler, model, optimizer, payload, path = checkpoint_fixture(tmp_path)
    clean = batch["noisy_latents"] - batch["timesteps"][..., None, None, None] / 1000 * batch["target_flow"]
    expected_context, expected_receipt = recycler.prepare_context(clean, batch["timesteps"] / 1000)
    expected_random = (random.random(), np.random.random(), torch.rand(2))
    target, restored = Flow(), ContextErrorRecycling(config.error_recycling)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.7, momentum=0.9)
    result = _restore_resume(path, model=target, optimizer=target_optimizer,
        hashes=payload["manifest_hashes"], teacher_lineage=payload["parent_teacher"],
        gradient_accumulation_steps=8, config=config, recycler=restored)
    actual_context, actual_receipt = restored.prepare_context(clean, batch["timesteps"] / 1000)
    assert result == (1, 8)
    assert torch.equal(target.weight, model.weight)
    assert target_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert torch.equal(actual_context, expected_context) and actual_receipt == expected_receipt
    assert random.random() == expected_random[0]
    assert np.random.random() == expected_random[1]
    assert torch.equal(torch.rand(2), expected_random[2])
    assert payload["method_contract"] == error_recycling_contract(config)


@pytest.mark.parametrize("change", ["legacy_stage", "missing_state", "cursor", "config", "method"])
def test_strict_resume_rejects_incomplete_or_relabelled_state(tmp_path, change):
    config, _, _, _, _, payload, path = checkpoint_fixture(tmp_path)
    if change == "legacy_stage":
        payload["stage"] = STAGE_NAME
    elif change == "missing_state":
        del payload["error_recycling_state"]
    elif change == "cursor":
        payload["error_recycling_state"]["observations"] = 7
    elif change == "config":
        payload["config"]["training"]["max_steps"] += 1
    else:
        payload["method_contract"]["full_self_rollout"] = True
    torch.save(payload, path)
    model = Flow()
    with pytest.raises(ValueError):
        _restore_resume(path, model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            hashes=payload["manifest_hashes"], teacher_lineage=payload["parent_teacher"],
            gradient_accumulation_steps=8, config=config, recycler=ContextErrorRecycling(config.error_recycling))


def test_cursor_reconstruction_preserves_restored_global_rng():
    class ConsumingLoader:
        def __len__(self):
            return 5

        def __iter__(self):
            for index in range(5):
                random.random(), np.random.random(), torch.rand(2)
                yield {"index": index}

    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    _, iterator = _resume_iterator_preserving_rng(ConsumingLoader(), 3)
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert next(iterator)["index"] == 3


def test_segment_cap_does_not_change_configured_sampling_horizon():
    config = config_for()
    config.training.max_steps = 200
    before = config.to_dict()
    args = parse_args(["--stop-after-step", "20"])
    _apply_overrides(config, args)
    assert config.to_dict() == before
    with pytest.raises(ValueError, match="configured max_steps"):
        _apply_overrides(config, parse_args(["--stop-after-step", "201"]))


@pytest.mark.parametrize("enabled", [False, True])
def test_cpu_mocked_segment_and_strict_resume_write_real_cursor_and_timing(tmp_path, monkeypatch, enabled):
    """Exercise the actual loop with tiny CPU stand-ins, never a CUDA model."""
    import train_causal_teacher_forcing as entry

    config = config_for(enabled=enabled)
    batch = tiny_batch(config)
    config.training.max_steps = 3
    config.training.output_dir = str(tmp_path / "run")
    lineage = SimpleNamespace(step=120, as_dict=lambda: {"stage": "action_teacher_lora_v1", "step": 120})
    monkeypatch.setattr(entry, "artifact_hashes", lambda *args: {"training_config": "frozen"})
    monkeypatch.setattr(entry, "load_teacher_checkpoint", lambda *args: ({}, lineage))
    monkeypatch.setattr(entry, "validate_confirmation", lambda *args: None)
    monkeypatch.setattr(entry, "query_dedicated_gpu", lambda **kwargs: SimpleNamespace(as_dict=lambda: {}))
    monkeypatch.setattr(entry, "_build_model", lambda *args: (
        Flow(), SimpleNamespace(trainable_parameters=1, total_parameters=1, replaced_linear_layers=[])))
    monkeypatch.setattr(entry, "_build_optimizer", lambda model, config: torch.optim.SGD(model.parameters(), lr=0.01))
    monkeypatch.setattr(entry, "_import_factory", lambda spec: (
        lambda config, training: [batch] * (training.max_steps * training.gradient_accumulation_steps)))
    monkeypatch.setattr(entry, "causal_teacher_forcing_loss", lambda model, batch, config, device, **kwargs:
        causal_teacher_forcing_loss(model, batch, config, torch.device("cpu"), **kwargs))
    monkeypatch.setattr(entry, "peak_vram_bytes", lambda *args: 0)
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "CPU_TEST_STAND_IN")
    for name in ("set_device", "reset_peak_memory_stats", "synchronize", "manual_seed_all", "set_rng_state_all"):
        monkeypatch.setattr(torch.cuda, name, lambda *args, **kwargs: None)
    for name in ("memory_allocated", "memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: 0)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    options = dict(confirmed_gpu_index=0, confirmed_gpu_uuid="CPU_TEST_STAND_IN",
                   confirmed_at_utc="CPU_TEST_STAND_IN", allocation_profile="CPU_TEST_STAND_IN")
    entry.launch(config, tmp_path / "unused.yaml", None, stop_after_step=1, **options)
    result_path = tmp_path / "run" / "execution_result.json"
    first = json.loads(result_path.read_text())
    assert first["status"] == "segment_complete" and first["stop_reason"] == "execution_step_cap"
    assert (first["step"], first["micro_batches_consumed"], first["configured_max_steps"]) == (1, 8, 3)
    checkpoint = torch.load(first["checkpoint_path"], weights_only=False)
    assert checkpoint["config"]["training"]["max_steps"] == 3
    assert checkpoint["metrics"]["optimizer_step_seconds"] > 0
    entry.launch(config, tmp_path / "unused.yaml", first["checkpoint_path"], **options)
    final = json.loads(result_path.read_text())
    assert (final["status"], final["step"], final["micro_batches_consumed"]) == ("complete", 3, 24)
    assert final["resume_step"] == 1
    checkpoint = torch.load(final["checkpoint_path"], weights_only=False)
    assert checkpoint["metrics"]["optimizer_step_seconds"] > 0
    if enabled:
        assert checkpoint["error_recycling_state"]["observations"] == 24
        assert final["error_recycling_restore_status"] == "restored_exact_state"
    else:
        assert "error_recycling_state" not in checkpoint
