"""CPU contracts only: these tests are not a 5090 training feasibility result."""
from __future__ import annotations

import copy
import json
import random
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import train_action_teacher as entry
from training.config import load_config
from training.runtime import load_checkpoint, sha256_file

CONFIG = Path(__file__).parents[1] / "configs/train/action_teacher_5090_repair_r005.yaml"


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.act_control_adapter = nn.Linear(1, 1)


def transition_fixture(tmp_path):
    cfg = load_config(CONFIG)
    cfg.training.output_dir = str(tmp_path / "window97")
    cfg.training.max_steps = 24
    source_config = cfg.to_dict()
    source_config["training"].update(max_steps=1040, output_dir=str(tmp_path / "source"))
    cfg.data.num_frames = 97
    cfg.data.manifest_path = str(tmp_path / "train97.jsonl")
    cfg.data.prompt_cache_path = None
    hashes = {key: "target-" + key for key in (*entry.SHARED_DATA_HASH_KEYS, "training_config")}
    source_hashes = {key: "source-" + key for key in (*entry.SHARED_DATA_HASH_KEYS, *entry.PROMPT_CACHE_HASH_KEYS, "training_config")}
    state = {name: torch.full_like(value, 0.25) for name, value in TinyTeacher().state_dict().items()}
    payload = dict(format_version=1, stage="action_teacher_lora_v1", step=1040,
                   micro_batches_consumed=8320, config=source_config, manifest_hashes=source_hashes,
                   trainable_model=state, optimizer={"must_not_restore": True},
                   torch_rng_state=torch.tensor([255], dtype=torch.uint8), cuda_rng_state_all=[],
                   initialization={"mode": "source-lineage"})
    path = tmp_path / "source1040.pt"
    torch.save(payload, path)
    return cfg, hashes, path, payload


def load_transition(cfg, hashes, path, pin=None):
    return entry._load_length_transition_checkpoint(path, expected_sha256=pin or sha256_file(path),
                                                    config=cfg, current_manifest_hashes=hashes)


def test_cli_requires_explicit_pin_and_independent_execution_limit():
    args = entry.parse_args(["--length-transition-from", "parent.pt", "--length-transition-sha256", "9" * 64,
                             "--stop-after-step", "20"])
    assert args.stop_after_step == 20 and args.length_transition_from == "parent.pt"
    for args in (["--length-transition-from", "parent.pt"], ["--length-transition-sha256", "9" * 64],
                 ["--stop-after-step", "0"], ["--stop-after-step", "-1"],
                 ["--length-transition-from", "parent.pt", "--length-transition-sha256", "not-a-hash"]):
        with pytest.raises(SystemExit):
            entry.parse_args(args)


@pytest.mark.parametrize("other", ["--resume", "--initialize-from", "--warm-start-from", "--sampling-transition-from"])
def test_cli_transition_modes_are_mutually_exclusive(other):
    with pytest.raises(SystemExit):
        entry.parse_args(["--length-transition-from", "parent.pt", "--length-transition-sha256", "9" * 64, other, "other.pt"])


def test_pinned_length_migration_retains_only_weights_and_truthful_lineage(tmp_path):
    cfg, hashes, path, payload = transition_fixture(tmp_path)
    original_config = copy.deepcopy(cfg.to_dict())
    state, lineage = load_transition(cfg, hashes, path)
    assert state.keys() == payload["trainable_model"].keys()
    assert all(torch.equal(state[key], payload["trainable_model"][key]) for key in state)
    assert lineage["sha256"] == sha256_file(path)
    assert lineage["source_step"] == 1040 and lineage["start_step"] == lineage["micro_batches_consumed"] == 0
    assert not lineage["optimizer_restored"] and not lineage["rng_restored"]
    assert lineage["prompt_transition"]["policy_changed"] is False
    assert lineage["prompt_transition"]["storage_changed"] is True
    assert lineage["length_transition"]["target_latent_frames"] == 25
    assert lineage["length_transition"]["target_future_actions"] == 96
    assert cfg.to_dict() == original_config


def test_wrong_pin_rejected_before_deserializing(tmp_path, monkeypatch):
    cfg, hashes, path, _ = transition_fixture(tmp_path)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("must verify pin before torch.load"))
    with pytest.raises(ValueError, match="SHA256"):
        load_transition(cfg, hashes, path, "0" * 64)


@pytest.mark.parametrize(("section", "field", "value"), [
    ("model", "action_scale", 0.5), ("model", "lora_alpha", 32),
    ("model", "base_model_path", "/another-base"), ("model", "gradient_checkpointing", False),
    ("optimizer", "adapter_lr", 1e-3), ("optimizer", "lora_lr", 1e-3),
    ("data", "canonical_action_keys", tuple("LKJIDSAW")), ("data", "num_workers", 0),
    ("training", "gradient_accumulation_steps", 4), ("training", "checkpoint_every", 50),
])
def test_unrelated_training_changes_are_not_whitelisted(tmp_path, section, field, value):
    cfg, hashes, path, _ = transition_fixture(tmp_path)
    setattr(getattr(cfg, section), field, value)
    with pytest.raises(ValueError, match="contract"):
        load_transition(cfg, hashes, path)


@pytest.mark.parametrize(("field", "value"), [("step", 1020), ("step", True), ("micro_batches_consumed", 0),
                                                ("stage", "causal_teacher_forcing_v1"), ("trainable_model", {})])
def test_invalid_parent_is_rejected_even_with_matching_pin(tmp_path, field, value):
    cfg, hashes, path, payload = transition_fixture(tmp_path)
    payload[field] = value
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_transition(cfg, hashes, path)


@pytest.mark.parametrize("key", [*entry.SHARED_DATA_HASH_KEYS, "training_config"])
def test_new_length_cache_cannot_reuse_parent_artifact_hashes(tmp_path, key):
    cfg, hashes, path, payload = transition_fixture(tmp_path)
    hashes[key] = payload["manifest_hashes"][key]
    with pytest.raises(ValueError, match="new bound"):
        load_transition(cfg, hashes, path)


def test_fixed_config_resume_and_ordinary_warm_start_stay_strict(tmp_path):
    cfg, hashes, path, _ = transition_fixture(tmp_path)
    with pytest.raises(ValueError, match="mismatch"):
        entry._load_warm_start_checkpoint(path, config=cfg, current_manifest_hashes=hashes)
    with pytest.raises(ValueError, match="manifest hashes"):
        load_checkpoint(path, expected_manifest_hashes=hashes)
    assert entry._execution_limit(2000, 20) == 20
    assert entry._execution_limit(2000, 40, 20) == 40
    assert entry._execution_limit(2000, None, 20) == 2000
    with pytest.raises(ValueError, match="beyond"):
        entry._execution_limit(2000, 20, 20)


def test_batch_contract_accepts_25_latents_and_96_actions_only(tmp_path):
    cfg, _, _, _ = transition_fixture(tmp_path)
    batch = {
        "noisy_latents": torch.empty((1, 25, 48, 30, 52), device="meta"),
        "target_flow": torch.empty((1, 25, 48, 30, 52), device="meta"),
        "timesteps": torch.empty((1, 25), device="meta"),
        "prompt_embeds": torch.empty((1, 3, 4096), device="meta"),
        "actions": torch.empty((1, 96, 8), device="meta"),
    }
    entry._validate_batch(batch, cfg)
    batch["actions"] = torch.empty((1, 48, 8), device="meta")
    with pytest.raises(ValueError, match="actions must"):
        entry._validate_batch(batch, cfg)


class FakeProfiler:
    instances = []

    def __init__(self, *args, **kwargs):
        self.events = []
        self.instances.append(self)

    def phase(self, name):
        self.set_phase(name)
        return nullcontext()

    def set_phase(self, name):
        self.events.append(("phase", name))

    def begin_step(self, step):
        self.events.append(("begin", step))

    def end_step(self, step):
        self.events.append(("end", step))
        return {"optimizer_step": step, "seconds": 0.1, "cpu_standin": True}

    def begin_checkpoint(self, step):
        self.events.append(("save_begin", step))

    def end_checkpoint(self, step, checkpoint_path=None):
        self.events.append(("save_end", step))
        return {"checkpoint_path": checkpoint_path}

    def summary(self):
        return {"cpu_standin": True}

    def close(self, error=None):
        self.events.append(("close",))


def cpu_launch_standin(monkeypatch, hashes, seen_batches):
    monkeypatch.setattr(entry, "OptimizerStepProfiler", FakeProfiler)
    monkeypatch.setattr(entry, "validate_confirmation", lambda *a: None)
    monkeypatch.setattr(entry, "query_dedicated_gpu", lambda **k: SimpleNamespace(as_dict=lambda: {}))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name in ("set_device", "reset_peak_memory_stats", "set_rng_state_all", "manual_seed_all"):
        monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    for name in ("memory_allocated", "memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "CPU stand-in")
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(entry, "peak_vram_bytes", lambda *a: 0)
    monkeypatch.setattr(entry, "_manifest_hashes", lambda *a: hashes)
    monkeypatch.setattr(entry, "_build_model", lambda *a: (TinyTeacher(), SimpleNamespace(
        trainable_parameters=2, total_parameters=2, replaced_linear_layers=[])))
    monkeypatch.setattr(entry, "_build_optimizer", lambda model, *a: torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9))
    monkeypatch.setattr(entry, "_import_factory", lambda *a: lambda **k: list(range(1000)))
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: nullcontext())

    def loss(model, batch, *args):
        seen_batches.append(batch)
        factor = random.random() + float(np.random.rand()) + float(torch.rand(()))
        return sum(p.square().sum() for p in model.parameters()) * factor

    monkeypatch.setattr(entry, "_forward_loss", loss)


def test_twenty_real_updates_resume_without_config_change_or_rng_drift(tmp_path, monkeypatch):
    cfg, hashes, path, payload = transition_fixture(tmp_path)
    seen = []
    cpu_launch_standin(monkeypatch, hashes, seen)
    kwargs = dict(confirmed_gpu_index=0, confirmed_gpu_uuid="CPU", confirmed_at_utc="CPU", allocation_profile="CPU")
    original_config = copy.deepcopy(cfg.to_dict())
    entry.launch(cfg, CONFIG, None, None, length_transition_from=str(path),
                 length_transition_sha256=sha256_file(path), stop_after_step=20, **kwargs)
    output = Path(cfg.training.output_dir)
    checkpoint20 = output / "checkpoints/step-0000020.pt"
    first = torch.load(checkpoint20, weights_only=False)
    assert first["step"] == 20 and first["micro_batches_consumed"] == 160
    assert first["config"] == original_config and first["config"]["training"]["max_steps"] == 24
    assert first["optimizer"]["state"]
    assert first["rng_contract_version"] == 2
    assert seen == list(range(160))
    assert any(not torch.equal(value, payload["trainable_model"][name]) for name, value in first["trainable_model"].items())
    result = json.loads((output / "execution_result.json").read_text())
    assert result["stop_reason"] == "execution_step_limit"
    events = FakeProfiler.instances[-1].events
    assert [event for event in events if event[0] == "begin"] == [("begin", i) for i in range(1, 21)]
    assert events.index(("end", 20)) < events.index(("save_begin", 20)) < events.index(("save_end", 20))
    entry.launch(cfg, CONFIG, str(checkpoint20), None, **kwargs)
    final = torch.load(output / "checkpoints/step-0000024.pt", weights_only=False)
    assert seen == list(range(192))
    assert final["step"] == 24 and final["initialization"]["source_step"] == 1040
    assert json.loads((output / "execution_result.json").read_text())["stop_reason"] == "configured_max_steps"

    # The resumed trajectory equals an uninterrupted CPU reference, including
    # optimizer momentum and random draws from all three host RNG sources.
    cfg.training.output_dir = str(tmp_path / "uninterrupted")
    entry.launch(cfg, CONFIG, None, None, length_transition_from=str(path),
                 length_transition_sha256=sha256_file(path), **kwargs)
    uninterrupted = torch.load(Path(cfg.training.output_dir) / "checkpoints/step-0000024.pt", weights_only=False)
    assert all(torch.equal(value, uninterrupted["trainable_model"][name]) for name, value in final["trainable_model"].items())
    assert torch.equal(final["torch_rng_state"], uninterrupted["torch_rng_state"])


def test_97_resume_rejects_incomplete_rng_contract(monkeypatch):
    monkeypatch.setattr(torch, "set_rng_state", lambda *a: pytest.fail("must reject before restore"))
    with pytest.raises(ValueError, match="complete"):
        entry._restore_rng_state({"torch_rng_state": torch.get_rng_state(), "cuda_rng_state_all": []}, require_complete=True)


def test_failure_is_recorded_and_profiler_closed(tmp_path, monkeypatch):
    cfg, hashes, path, _ = transition_fixture(tmp_path)
    cpu_launch_standin(monkeypatch, hashes, [])
    monkeypatch.setattr(entry, "_forward_loss", lambda *a: torch.tensor(float("nan")))
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        entry.launch(cfg, CONFIG, None, None, length_transition_from=str(path), length_transition_sha256=sha256_file(path),
                     stop_after_step=20, confirmed_gpu_index=0, confirmed_gpu_uuid="CPU", confirmed_at_utc="CPU", allocation_profile="CPU")
    failure = json.loads((Path(cfg.training.output_dir) / "failure.json").read_text())
    assert failure["failure_type"] == "FloatingPointError"
    assert FakeProfiler.instances[-1].events[-1] == ("close",)
