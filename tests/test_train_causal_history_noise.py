"""Minimal CPU new-stage/cursor contracts, not GPU or quality validation."""
import copy
import json
from pathlib import Path

import pytest
import torch

import train_causal_history_noise as entry
from training.causal_tf import CausalTeacherForcingConfig
from training.history_noise import HistoryNoiseConfig
from training.runtime import sha256_file


def test_archive_revision_comes_only_from_authorizer(monkeypatch):
    monkeypatch.setattr(entry, "git_revision", lambda root: "unknown")
    bound = entry.source_provenance({"source_revision": "a" * 40})
    assert bound["git_revision"] == "a" * 40
    assert bound["git_query_revision"] == "unknown"
    assert bound["source_revision_origin"] == "verified_external_authorizer"
    assert entry.source_provenance({})["git_revision"] == "unknown"
    monkeypatch.setattr(entry, "git_revision", lambda root: "b" * 40)
    with pytest.raises(ValueError, match="differs"):
        entry.source_provenance({"source_revision": "a" * 40})


def stage_fixture(tmp_path):
    import yaml
    config = CausalTeacherForcingConfig()
    config.data.num_frames = 97
    config.data.data_factory = entry.FACTORY
    config.training.max_steps = 200
    config.training.checkpoint_every = 20
    config.model.action_scale = .03
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
    stage = {
        "schema_version": 1, "stage": entry.STAGE_NAME, "base_config_sha256": sha256_file(base),
        "parent_checkpoint_path": "/p/clean60.pt", "parent_checkpoint_sha256": "e" * 64,
        "parent_expected_stage": "causal_teacher_forcing_v1", "parent_expected_step": 60,
        "data_start_absolute_index": 480, "max_steps": 200, "output_dir": "/p/new-stage",
        "history_noise": HistoryNoiseConfig().to_dict(),
    }
    path = tmp_path / "stage.json"
    path.write_text(json.dumps(stage), encoding="utf-8")
    effective, loaded = entry.load_stage(base, path)
    return effective, loaded, base, path


def test_effective_config_preserves_original_base_and_pair_toggle(tmp_path):
    config, stage, base, path = stage_fixture(tmp_path)
    before = base.read_bytes()
    assert config.training.output_dir == stage["output_dir"]
    stage["history_noise"]["enabled"] = False
    path.write_text(json.dumps(stage), encoding="utf-8")
    control, loaded = entry.load_stage(base, path)
    assert control.to_dict() == config.to_dict()
    assert entry.method_contract(loaded)["history_noise"]["enabled"] is False
    assert base.read_bytes() == before
    assert entry.method_contract(loaded)["teacher_model_loaded"] is False


def test_absolute_dataset_offset_not_fake_resume(tmp_path, monkeypatch):
    config, stage, _, _ = stage_fixture(tmp_path)
    captured = {}
    class FakeDataset:
        def __init__(self, index_path, **kwargs):
            captured.update(kwargs)
    import training.data.action_resampled as source
    monkeypatch.setattr(source, "ResampledActionDataset", FakeDataset)
    loader = entry.build_offset_dataloader(config, stage, consumed_micro=160)
    assert captured["num_samples"] == 2080
    assert loader.dataset.indices.start == 640
    assert loader.dataset.indices.stop == 2080


def test_strict_resume_binds_stage_parent_all_rng_and_absolute_cursor(tmp_path):
    config, stage, base, path = stage_fixture(tmp_path)
    hashes = {"base_training_config": sha256_file(base), "stage_training_config": sha256_file(path)}
    payload = {
        "format_version": 1, "stage": entry.STAGE_NAME, "step": 20, "micro_batches_consumed": 160,
        "absolute_micro_batches_consumed": 640, "next_absolute_sample_index": 640,
        "data_start_absolute_index": 480, "config": config.to_dict(), "stage_config": stage,
        "base_config_sha256": hashes["base_training_config"], "stage_config_sha256": hashes["stage_training_config"],
        "manifest_hashes": hashes, "method_contract": entry.method_contract(stage),
        "initialization_mode": entry.INITIALIZATION,
        "parent_causal": {"kind": "causal_weights_only_parent", "sha256": "e" * 64,
                          "stage": "causal_teacher_forcing_v1", "step": 60},
        "trainable_model": {}, "optimizer": {}, "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": [], "python_rng_state": (), "numpy_rng_state": (),
    }
    checkpoint = tmp_path / "step20.pt"
    torch.save(payload, checkpoint)
    assert entry.validate_resume_checkpoint(checkpoint, config=config, stage=stage, hashes=hashes)["step"] == 20
    for change in ({"stage": "causal_teacher_forcing_v1"}, {"next_absolute_sample_index": 160},
                   {"micro_batches_consumed": 480}, {"parent_teacher": {}}):
        bad = copy.deepcopy(payload)
        bad.update(change)
        torch.save(bad, checkpoint)
        with pytest.raises(ValueError):
            entry.validate_resume_checkpoint(checkpoint, config=config, stage=stage, hashes=hashes)
    torch.save(payload, checkpoint)
    changed_stage = copy.deepcopy(stage)
    changed_stage["history_noise"]["enabled"] = False
    with pytest.raises(ValueError):
        entry.validate_resume_checkpoint(checkpoint, config=config, stage=changed_stage, hashes=hashes)
