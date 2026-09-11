"""Bounded CPU tests for the new stage; no Wan weights or CUDA kernels loaded."""

import copy
import json
import weakref
from pathlib import Path

import pytest
import torch
from torch import nn

from training.causal_tf import causal_teacher_forcing_loss, load_teacher_checkpoint
from training.causal_moba import (
    METHOD_NAME, STAGE_NAME, CausalMoBAConfig, MoBARegularizationConfig,
    backward_microbatch, bidirectional_flow_loss, load_moba_config,
    method_contract, verify_bidirectional_interface,
)
from training.runtime import sha256_file
from train_causal_moba import (
    DEFAULT_CONFIG, _checkpoint_payload, _restore_resume, _resume_iterator,
    _sampling_contract, validate_resume_checkpoint, main,
)
from test_action_resampled import fake_cache


def _small_case(weight=0.1):
    config = load_moba_config(DEFAULT_CONFIG)
    config.regularization.bidirectional_weight = weight
    config.data.height = config.data.width = 32
    config.model.latent_channels = 2
    clean = torch.arange(104, dtype=torch.float32).reshape(1, 13, 2, 2, 2) / 104
    target = torch.full_like(clean, 0.25)
    target[:, 0] = 0
    timesteps = torch.tensor([[0.0] + [500.0] * 12])
    batch = {"noisy_latents": clean + (timesteps / 1000).view(1, 13, 1, 1, 1) * target,
             "target_flow": target, "timesteps": timesteps,
             "actions": torch.ones(1, 48, 8), "prompt_embeds": torch.full((1, 2, 16), 0.5)}
    return config, batch


class TinyFlow(nn.Module):
    def __init__(self, *, require_sequential=False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.4))
        self.first_only = nn.Parameter(torch.tensor(123.0))
        self.calls = []
        self.previous_tf_prediction = None
        self.require_sequential = require_sequential

    def forward(self, noisy, conditional_dict, timestep, clean_x=None, aug_t=None,
                training_attention_mode="causal", **kwargs):
        bid = training_attention_mode == "bidirectional"
        if bid:
            assert clean_x is None and aug_t is None
            assert kwargs == {}
            assert torch.count_nonzero(timestep[:, 0]) == 0
            if self.require_sequential:
                assert self.weight.grad is not None  # TF backward already happened.
                assert self.previous_tf_prediction() is None  # TF graph released.
        else:
            assert clean_x is not None and torch.count_nonzero(aug_t) == 0
            assert set(kwargs) == {"replace_first_timestep_and_noise_latents"}
            assert kwargs["replace_first_timestep_and_noise_latents"] is True
        self.calls.append({"mode": training_attention_mode, "noisy": noisy.detach().clone(),
                           "timestep": timestep.detach().clone(),
                           "actions": conditional_dict["act_context"][0].detach().clone(),
                           "prompt": conditional_dict["prompt_embeds"][0].detach().clone(),
                           "action_scale": conditional_dict["act_context_scale"]})
        prediction = noisy.float() * self.weight * (1.5 if bid else 1.0)
        prediction[:, 0] = self.first_only
        if not bid:
            self.previous_tf_prediction = weakref.ref(prediction)
        return prediction, torch.zeros_like(prediction)


def test_weight_zero_exactly_matches_tf_loss_and_gradient_without_bid():
    config, batch = _small_case(0)
    actual, expected = TinyFlow(), TinyFlow()
    reference = causal_teacher_forcing_loss(expected, batch, config, torch.device("cpu"))
    (reference / 8).backward()
    metrics = backward_microbatch(actual, batch, config, torch.device("cpu"))
    assert [call["mode"] for call in actual.calls] == ["causal"]
    assert torch.equal(actual.weight.grad, expected.weight.grad)
    assert actual.first_only.grad.item() == 0
    assert metrics == {"loss_tf": reference.item(), "loss_bid": 0.0,
                       "loss": reference.item(), "loss_total": reference.item(), "bid_evaluated": False}


def test_weight_point_one_real_loss_gradients_add_sequentially_same_inputs():
    config, batch = _small_case(0.1)
    actual, expected = TinyFlow(require_sequential=True), TinyFlow()
    tf = causal_teacher_forcing_loss(expected, batch, config, torch.device("cpu"))
    bid = bidirectional_flow_loss(expected, batch, config, torch.device("cpu"))
    ((tf + 0.1 * bid) / 8).backward()
    original = {key: value.clone() for key, value in batch.items()}
    metrics = backward_microbatch(actual, batch, config, torch.device("cpu"))
    assert torch.allclose(actual.weight.grad, expected.weight.grad, atol=1e-7, rtol=1e-6)
    assert actual.first_only.grad.item() == 0
    assert metrics["loss_tf"] == tf.item() and metrics["loss_bid"] == bid.item()
    assert metrics["loss"] == pytest.approx(tf.item() + 0.1 * bid.item())
    assert metrics["bid_evaluated"] is True
    first, second = actual.calls
    assert (first["mode"], second["mode"]) == ("causal", "bidirectional")
    for key in ("noisy", "timestep", "actions", "prompt"):
        assert torch.equal(first[key], second[key])
    assert first["action_scale"] == second["action_scale"] == 0.03
    assert all(torch.equal(batch[key], value) for key, value in original.items())


def test_microbatch_accumulation_does_not_zero_or_step_optimizer():
    config, batch = _small_case()
    model = TinyFlow()
    backward_microbatch(model, batch, config, torch.device("cpu"))
    single = model.weight.grad.clone()
    backward_microbatch(model, batch, config, torch.device("cpu"))
    assert torch.allclose(model.weight.grad, 2 * single)
    assert model.weight.item() == pytest.approx(0.4)


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_weights_fail_before_forward(weight):
    config, batch = _small_case()
    config.regularization.bidirectional_weight = weight
    model = TinyFlow()
    with pytest.raises(ValueError, match="bidirectional_weight"):
        backward_microbatch(model, batch, config, torch.device("cpu"))
    assert model.calls == [] and model.weight.grad is None


def test_method_static_prompt_and_bid_interface_fail_closed():
    config = load_moba_config(DEFAULT_CONFIG)
    config.regularization.method = "official_packed_MoBA"
    with pytest.raises(ValueError, match="method"):
        config.validate()
    config.regularization.method = METHOD_NAME
    config.data.prompt_cache_path = None
    with pytest.raises(ValueError, match="scene-static"):
        config.validate()
    with pytest.raises(RuntimeError, match="training_attention_mode"):
        verify_bidirectional_interface(nn.Linear(2, 2))
    with pytest.raises(RuntimeError, match="backbone"):
        verify_bidirectional_interface(TinyFlow())


def test_default_plan_identifies_new_stage_without_cuda_weights_or_output(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU plan must not query CUDA or load weights")
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    assert main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["stage"] == STAGE_NAME
    assert report["method_contract"]["method"] == METHOD_NAME
    assert not report["cuda_queried"] and not report["weights_loaded"]
    assert not report["method_contract"]["packed_moba_mask"]
    assert report["sampling_contract"]["namespace"] == "action_resampled_absolute_v1"
    assert "/home/" not in json.dumps(report["config"])
    with pytest.raises(SystemExit, match="fresh GPU confirmation"):
        main(["--launch"])


def _lineage_case(tmp_path):
    from test_causal_tf import _write_static_lineage_fixture
    legacy, model, teacher, hashes = _write_static_lineage_fixture(tmp_path)
    config = CausalMoBAConfig(**{name: getattr(legacy, name) for name in
                              ("model", "data", "lineage", "optimizer", "training")})
    return config, model, teacher, hashes


def test_action_parent_stage_hash_prompt_and_scale_checks_are_inherited(tmp_path):
    config, _, teacher, hashes = _lineage_case(tmp_path)
    payload, lineage = load_teacher_checkpoint(config, hashes)
    assert lineage.stage == "action_teacher_lora_v1"
    assert lineage.sha256 == sha256_file(teacher)
    for key in ("prompt_cache", "feature_index"):
        changed = dict(hashes, **{key: "changed"})
        with pytest.raises(ValueError):
            load_teacher_checkpoint(config, changed)
    for field, value in (("stage", "causal_teacher_forcing_v1"), ("stage", "lingbot"), ("step", 0)):
        changed = copy.deepcopy(payload)
        changed[field] = value
        torch.save(changed, teacher)
        with pytest.raises(ValueError):
            load_teacher_checkpoint(config, dict(hashes, teacher_checkpoint=sha256_file(teacher)))


def test_checkpoint_records_new_stage_and_strict_resume_restores_state(tmp_path):
    config, model, _, hashes = _lineage_case(tmp_path)
    _, lineage = load_teacher_checkpoint(config, hashes)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    kwargs = dict(config=config, step=2, micro_batches_consumed=16,
                  metrics={"loss": 1.1, "loss_tf": 1.0, "loss_bid": 1.0},
                  manifest_hashes=hashes, teacher_lineage=lineage.as_dict())
    checkpoint = _checkpoint_payload(model, optimizer, **kwargs)
    assert checkpoint["stage"] == STAGE_NAME
    assert checkpoint["method_contract"] == method_contract(config)
    assert checkpoint["initialization_mode"].endswith("step0")
    path = tmp_path / "resume.pt"
    torch.save(checkpoint, path)
    assert _restore_resume(path, config=config, model=model, optimizer=optimizer,
                           hashes=hashes, teacher_lineage=lineage.as_dict()) == (2, 16)
    for field, value in (("stage", "causal_teacher_forcing_v1"),
                         ("method_contract", {}), ("sampling_contract", {}),
                         ("parent_teacher", {}), ("micro_batches_consumed", 15),
                         ("config", {}), ("initialization_mode", "resume_action_teacher")):
        bad = dict(checkpoint, **{field: value})
        torch.save(bad, path)
        with pytest.raises(ValueError):
            _restore_resume(path, config=config, model=model, optimizer=optimizer,
                            hashes=hashes, teacher_lineage=lineage.as_dict())
    torch.save(checkpoint, path)
    config.regularization.bidirectional_weight = 0.0
    with pytest.raises(ValueError, match="method/auxiliary-weight"):
        _restore_resume(path, config=config, model=model, optimizer=optimizer,
                        hashes=hashes, teacher_lineage=lineage.as_dict())


def test_nonfinite_bid_loss_raises_after_tf_without_optimizer_update(monkeypatch):
    import training.causal_moba as moba
    config, batch = _small_case()
    model = TinyFlow()
    monkeypatch.setattr(moba, "bidirectional_flow_loss",
                        lambda *args: model.weight * float("nan"))
    with pytest.raises(FloatingPointError, match="bidirectional"):
        backward_microbatch(model, batch, config, torch.device("cpu"))
    assert model.weight.grad is not None and model.weight.item() == pytest.approx(0.4)


def _extension_case(tmp_path):
    config, model, _, hashes = _lineage_case(tmp_path)
    config.training.max_steps = 40
    config.data.data_factory = "training.data.action_resampled:build_resampled_action_teacher_dataloader"
    config.data.num_workers = 0
    _, lineage = load_teacher_checkpoint(config, hashes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(42)
    payload = _checkpoint_payload(
        model, optimizer, config=config, step=40, micro_batches_consumed=320,
        metrics={"loss": 0.2}, manifest_hashes=hashes, teacher_lineage=lineage.as_dict(),
    )
    path = tmp_path / "step-0000040.pt"
    torch.save(payload, path)
    extended = copy.deepcopy(config)
    extended.training.max_steps = 260
    return config, extended, model, optimizer, hashes, lineage.as_dict(), path, payload


def test_explicit_horizon_extension_restores_optimizer_rng_and_absolute_sample_320(tmp_path, fake_cache):
    from training.data.action_resampled import build_resampled_action_teacher_dataloader
    config, extended, model, optimizer, hashes, parent, path, payload = _extension_case(tmp_path)
    original_sha = sha256_file(path)
    with pytest.raises(ValueError, match="serialized configuration"):
        _restore_resume(path, config=extended, model=model, optimizer=optimizer, hashes=hashes, teacher_lineage=parent)
    for parameter in model.parameters():
        parameter.data.add_(10)
    restored_optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    torch.manual_seed(9999)
    assert _restore_resume(path, config=extended, model=model, optimizer=restored_optimizer,
                           hashes=hashes, teacher_lineage=parent, allow_horizon_extension=True) == (40, 320)
    assert torch.equal(torch.get_rng_state(), payload["torch_rng_state"])
    assert restored_optimizer.state_dict()["param_groups"] == payload["optimizer"]["param_groups"]
    for key, state in payload["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, restored_optimizer.state_dict()["state"][key][name])
    for name, parameter in model.named_parameters():
        if name in payload["trainable_model"]:
            assert torch.equal(parameter, payload["trainable_model"][name])
    _, record = validate_resume_checkpoint(path, config=extended, hashes=hashes,
                                           teacher_lineage=parent, allow_horizon_extension=True)
    assert record["source_checkpoint_sha256"] == original_sha == sha256_file(path)
    assert record["source_max_steps"] == 40 and record["target_max_steps"] == 260
    assert record["next_absolute_sample_index"] == 320
    assert record["source_sampling_contract"]["virtual_samples"] == 320
    assert record["target_sampling_contract"]["virtual_samples"] == 2080
    assert record["manifest_hashes"] == hashes
    assert record["restore_policy"]["fresh_restart"] is False
    loader = build_resampled_action_teacher_dataloader(config=extended.data, training=extended.training)
    rng = torch.get_rng_state().clone()
    _, iterator = _resume_iterator(loader, 320)
    assert torch.equal(torch.get_rng_state(), rng)
    expected, actual = loader.dataset[320], next(iterator)
    assert loader.dataset.sampling_spec(320)["absolute_sample_index"] == 320
    assert all(torch.equal(actual[key][0], value) for key, value in expected.items())
    original = build_resampled_action_teacher_dataloader(config=config.data, training=config.training)
    assert all(torch.equal(value, loader.dataset[319][key]) for key, value in original.dataset[319].items())
    # The new checkpoint records new horizon + lineage while retaining source hashes.
    result = _checkpoint_payload(model, restored_optimizer, config=extended, step=60,
                                 micro_batches_consumed=480, metrics={"loss": 0.1}, manifest_hashes=hashes,
                                 teacher_lineage=parent, continuations=[record])
    assert result["continuations"] == [record] and result["config"]["training"]["max_steps"] == 260
    assert result["manifest_hashes"] == payload["manifest_hashes"]
    assert result["sampling_contract"] == _sampling_contract(extended)
    assert result["initialization_mode"] == payload["initialization_mode"]
    later = tmp_path / "step-0000060.pt"
    torch.save(result, later)
    loaded, no_new_extension = validate_resume_checkpoint(later, config=extended, hashes=hashes, teacher_lineage=parent)
    assert loaded["continuations"] == [record] and no_new_extension is None
    assert sha256_file(path) == original_sha


@pytest.mark.parametrize("change", ["seed", "output", "lr", "weight", "action_scale", "factory", "method", "prompt", "hash"])
def test_horizon_extension_does_not_allow_any_other_change(tmp_path, change):
    _, config, _, _, hashes, parent, path, _ = _extension_case(tmp_path)
    if change == "seed": config.training.seed += 1
    elif change == "output": config.training.output_dir += "-other"
    elif change == "lr": config.optimizer.adapter_lr *= 2
    elif change == "weight": config.regularization.bidirectional_weight = 0
    elif change == "action_scale": config.model.action_scale = 1
    elif change == "factory": config.data.data_factory = "training.data.action_dataset:build_action_teacher_dataloader"
    elif change == "method": config.regularization.method = "different"
    elif change == "prompt": config.data.prompt_cache_path += ".changed"
    elif change == "hash": hashes = dict(hashes, training_config="different YAML")
    with pytest.raises(ValueError):
        validate_resume_checkpoint(path, config=config, hashes=hashes, teacher_lineage=parent, allow_horizon_extension=True)


def test_extension_requires_increase_valid_source_contract_and_explicit_resume(tmp_path):
    config, extended, _, _, hashes, parent, path, payload = _extension_case(tmp_path)
    with pytest.raises(ValueError, match="strictly increase"):
        validate_resume_checkpoint(path, config=config, hashes=hashes, teacher_lineage=parent, allow_horizon_extension=True)
    for key, value in [("sampling_contract", {}), ("micro_batches_consumed", 319), ("step", 41)]:
        torch.save(dict(payload, **{key: value}), path)
        with pytest.raises(ValueError):
            validate_resume_checkpoint(path, config=extended, hashes=hashes, teacher_lineage=parent, allow_horizon_extension=True)
    for args in [["--allow-horizon-extension"], ["--allow-horizon-extension", "--resume", "checkpoint.pt"]]:
        with pytest.raises(SystemExit, match="requires --resume and --max-steps"):
            main(args)
