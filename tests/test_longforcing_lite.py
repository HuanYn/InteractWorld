from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from train_longforcing_lite import (
    _checkpoint_payload,
    _objective_counts_at_micro_batch,
    _prepare_resume_attempt,
    _require_launch_factories,
    _restore_resume,
    validation_report,
)
from training.longforcing_lite import (
    METHOD_NAME,
    PINNED_BASE_MODEL,
    STAGE_NAME,
    ParentLineage,
    WanLongForcingBackend,
    artifact_hashes,
    curriculum_depth,
    is_flowmatch_replay,
    load_longforcing_config,
    load_parent_checkpoints,
    longforcing_lite_loss,
    rgb_frames_for_blocks,
)
from training.runtime import append_jsonl, sha256_file

ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "train" / "longforcing_lite_v1.yaml"


def test_config_encodes_lite_not_dmd_and_15_second_contract() -> None:
    config = load_longforcing_config(CONFIG_PATH)
    assert METHOD_NAME == "LongForcing-lite"
    assert config.rollout.is_dmd is False
    assert config.rollout.student_steps == 4
    assert config.rollout.teacher_steps == 40
    assert config.rollout.curriculum_depths == (1, 4, 8, 20)
    assert config.rollout.flowmatch_replay_fraction == 0.25
    assert config.rollout.teacher_cpu_offload is True
    assert config.data.short_window_frames == rgb_frames_for_blocks(4, config) == 49
    assert config.data.demo_rollout_frames == rgb_frames_for_blocks(20, config) == 241
    assert (config.data.demo_rollout_frames - 1) / config.data.fps == 15.0


def test_curriculum_and_replay_schedule_are_exact_and_resume_stable() -> None:
    config = load_longforcing_config(CONFIG_PATH)
    starts = config.rollout.curriculum_start_steps
    assert [curriculum_depth(step, config) for step in starts] == [1, 4, 8, 20]
    assert curriculum_depth(starts[1] - 1, config) == 1
    assert [is_flowmatch_replay(index) for index in range(8)] == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]


class _Student(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))


class _FakeBackend:
    def __init__(self) -> None:
        self.student = _Student()
        self.teacher = nn.Identity()
        self.student_calls: list[tuple[int, bool, int]] = []
        self.teacher_calls: list[tuple[int, bool]] = []
        self.activations: list[str] = []

    def activate_student(self, device):
        self.activations.append("student")

    def activate_teacher(self, device):
        self.activations.append("teacher")

    def student_velocity(self, **kwargs):
        self.student_calls.append(
            (
                kwargs["block_index"],
                torch.is_grad_enabled(),
                kwargs["history"].shape[1],
            )
        )
        return torch.ones_like(kwargs["noisy_block"]) * self.student.scale

    def teacher_velocity(self, **kwargs):
        self.teacher_calls.append((kwargs["block_index"], torch.is_grad_enabled()))
        return torch.zeros_like(kwargs["noisy_block"])

    def flowmatch_replay_loss(self, short_window):
        return self.student.scale.square()


def _tiny_rollout(config, depth: int):
    config.model.latent_channels = 2
    config.data.height = 32
    config.data.width = 32
    return {
        "initial_latent": torch.zeros(1, 1, 2, 2, 2),
        "rollout_noise": torch.zeros(1, depth, 3, 2, 2, 2),
        "block_actions": torch.zeros(1, depth, 12, 8),
        "conditions": {"prompt": "fake"},
    }


def test_four_step_self_roll_only_backprops_last_block_and_teacher_is_stop_grad() -> None:
    config = load_longforcing_config(CONFIG_PATH)
    backend = _FakeBackend()
    loss, details = longforcing_lite_loss(backend, _tiny_rollout(config, 4), config, depth=4)
    loss.backward()

    assert len(backend.student_calls) == 4 * 4
    assert all(not enabled for _, enabled, _ in backend.student_calls[:-4])
    assert all(enabled for _, enabled, _ in backend.student_calls[-4:])
    assert {history for _, _, history in backend.student_calls[-4:]} == {10}
    assert len(backend.teacher_calls) == 40
    assert all(not enabled for _, enabled in backend.teacher_calls)
    assert backend.activations == ["student", "teacher", "student"]
    assert backend.student.scale.grad is not None
    assert backend.student.scale.grad.abs().item() > 0
    assert details["objective"] == "teacher_endpoint_mse"
    assert details["is_dmd"] is False
    assert details["rollout_rgb_frames"] == 49


class _FakeWanBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_dim = 2
        self.patch_size = (1, 2, 2)
        self.act_control_adapter = nn.Identity()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, x, **kwargs):
        self.calls.append((tuple(x.shape), kwargs))
        return torch.zeros_like(x) + self.anchor


CausalWanModel = type("CausalWanModel", (_FakeWanBackbone,), {})
WanModel = type("WanModel", (_FakeWanBackbone,), {})


class _FakeWanWrapper(nn.Module):
    def __init__(self, model, *, causal: bool) -> None:
        super().__init__()
        self.model = model
        self.uniform_timestep = not causal


def test_real_backend_uses_aligned_four_block_sliding_window() -> None:
    config = load_longforcing_config(CONFIG_PATH)
    config.model.latent_channels = 2
    config.data.height = 32
    config.data.width = 32
    student = _FakeWanWrapper(CausalWanModel(), causal=True)
    teacher = _FakeWanWrapper(WanModel(), causal=False)
    backend = WanLongForcingBackend(student=student, teacher=teacher, config=config)
    conditions = {
        "prompt_embeds": torch.zeros(1, 2, 4),
        "all_block_actions": torch.zeros(1, 5, 12, 8),
    }
    kwargs = {
        "noisy_block": torch.zeros(1, 3, 2, 2, 2),
        "history": torch.zeros(1, 13, 2, 2, 2),
        "block_index": 4,
        "conditions": conditions,
        "timestep": torch.tensor([500.0]),
    }
    assert backend.student_velocity(**kwargs).shape == (1, 3, 2, 2, 2)
    assert backend.teacher_velocity(**kwargs).shape == (1, 3, 2, 2, 2)
    student_shape, student_kwargs = student.model.calls[-1]
    assert student_shape == (1, 2, 12, 2, 2)
    assert student_kwargs["current_start"] == 4
    assert student_kwargs["t"].shape == (1, 12)
    assert torch.count_nonzero(student_kwargs["t"][:, :-3]).item() == 0
    assert len(conditions["_longforcing_action_context_cache"]) == 1

    earlier = dict(kwargs, block_index=3, history=torch.zeros(1, 10, 2, 2, 2))
    backend.student_velocity(**earlier)
    assert len(conditions["_longforcing_action_context_cache"]) == 1


def _parent_fixture(tmp_path: Path):
    config = load_longforcing_config(CONFIG_PATH)
    manifest = tmp_path / "train.jsonl"
    index = tmp_path / "train.features.jsonl"
    receipt = tmp_path / "train.features.jsonl.receipt.json"
    long_index = tmp_path / "train.long241.features.jsonl"
    long_receipt = tmp_path / "train.long241.features.jsonl.receipt.json"
    run_config = tmp_path / "longforcing.yaml"
    manifest.write_text('{"sample":"one"}\n', encoding="utf-8")
    index.write_text('{"sample":"one"}\n', encoding="utf-8")
    receipt.write_text('{"schema_version":1}\n', encoding="utf-8")
    long_index.write_text('{"sample":"one"}\n', encoding="utf-8")
    long_receipt.write_text('{"schema_version":1}\n', encoding="utf-8")
    run_config.write_text("stage: longforcing\n", encoding="utf-8")
    config.data.manifest_path = str(manifest)
    config.data.feature_index_path = str(index)
    config.data.feature_receipt_path = str(receipt)
    config.data.long_feature_index_path = str(long_index)
    config.data.long_feature_receipt_path = str(long_receipt)
    shared = {
        "dataset_manifest": sha256_file(manifest),
        "feature_index": sha256_file(index),
        "feature_receipt": sha256_file(receipt),
    }
    state = {
        "model.act_control_adapter.weight": torch.ones(1),
        "model.block.self_attn.q.lora_a.weight": torch.ones(1),
        "model.block.self_attn.q.lora_b.weight": torch.ones(1),
    }
    teacher_path = tmp_path / "teacher.pt"
    teacher_payload = {
        "format_version": 1,
        "stage": "action_teacher_lora_v1",
        "step": 50,
        "trainable_model": state,
        "manifest_hashes": {**shared, "training_config": "teacher-config"},
        "config": {"model": {"base_model_path": PINNED_BASE_MODEL}},
    }
    torch.save(teacher_payload, teacher_path)
    teacher_hash = sha256_file(teacher_path)
    teacher_lineage = ParentLineage(
        path=str(teacher_path.resolve()),
        sha256=teacher_hash,
        stage="action_teacher_lora_v1",
        step=50,
        source_manifest_hashes=teacher_payload["manifest_hashes"],
    )
    causal_path = tmp_path / "causal.pt"
    torch.save(
        {
            "format_version": 1,
            "stage": "causal_teacher_forcing_v1",
            "step": 100,
            "trainable_model": state,
            "manifest_hashes": {
                **shared,
                "training_config": "causal-config",
                "teacher_checkpoint": teacher_hash,
            },
            "parent_teacher": teacher_lineage.as_dict(),
            "config": {"model": {"base_model_path": PINNED_BASE_MODEL}},
        },
        causal_path,
    )
    config.lineage.teacher_checkpoint_path = str(teacher_path)
    config.lineage.causal_checkpoint_path = str(causal_path)
    hashes = artifact_hashes(config, run_config)
    return config, run_config, causal_path, hashes


def test_parent_hashes_and_teacher_to_causal_lineage_are_strict(tmp_path: Path) -> None:
    config, _, causal_path, hashes = _parent_fixture(tmp_path)
    teacher, teacher_lineage, causal, causal_lineage = load_parent_checkpoints(config, hashes)
    assert teacher_lineage.sha256 == hashes["teacher_checkpoint"]
    assert causal_lineage.sha256 == hashes["causal_checkpoint"]
    assert causal["parent_teacher"] == teacher_lineage.as_dict()
    assert teacher["stage"] == "action_teacher_lora_v1"

    causal["parent_teacher"] = {**causal["parent_teacher"], "step": 49}
    torch.save(causal, causal_path)
    changed_hashes = dict(hashes)
    changed_hashes["causal_checkpoint"] = sha256_file(causal_path)
    with pytest.raises(ValueError, match="parent_teacher lineage is not exact"):
        load_parent_checkpoints(config, changed_hashes)


def test_default_cpu_report_never_loads_weights_or_queries_cuda(monkeypatch) -> None:
    config = load_longforcing_config(CONFIG_PATH)

    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not touch checkpoint or CUDA inventory")

    monkeypatch.setattr(torch, "load", forbidden)
    report = validation_report(config, CONFIG_PATH)
    assert report["stage"] == STAGE_NAME
    assert report["cuda_queried"] is False
    assert report["weights_loaded"] is False
    assert report["launch_ready"] is True
    config.rollout.backend_factory = None
    with pytest.raises(RuntimeError, match="before GPU inspection"):
        _require_launch_factories(config)


class _ResumeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.act_control_adapter = nn.Linear(1, 1)


def test_checkpoint_resume_restores_exact_microbatch_and_schedule_state(tmp_path: Path) -> None:
    config = load_longforcing_config(CONFIG_PATH)
    model = _ResumeModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    hashes = {"all": "fixed"}
    teacher = {"sha256": "teacher"}
    causal = {"sha256": "causal"}
    payload = _checkpoint_payload(
        model,
        optimizer,
        config=config,
        step=2,
        micro_batches_consumed=16,
        metrics={"loss": 1.0},
        hashes=hashes,
        teacher_lineage=teacher,
        causal_lineage=causal,
    )
    path = tmp_path / "resume.pt"
    torch.save(payload, path)
    step, consumed = _restore_resume(
        path,
        model=model,
        optimizer=optimizer,
        hashes=hashes,
        teacher_lineage=teacher,
        causal_lineage=causal,
        config=config,
    )
    assert (step, consumed) == (2, 16)
    payload["replay_phase"] = 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="replay phase"):
        _restore_resume(
            path,
            model=model,
            optimizer=optimizer,
            hashes=hashes,
            teacher_lineage=teacher,
            causal_lineage=causal,
            config=config,
        )


def test_resume_objective_counters_continue_from_consumed_micro_batches() -> None:
    assert _objective_counts_at_micro_batch(0) == {"longforcing": 0, "flowmatch_replay": 0}
    counts = _objective_counts_at_micro_batch(160)
    assert counts == {"longforcing": 120, "flowmatch_replay": 40}
    for consumed in range(160, 168):
        counts["flowmatch_replay" if is_flowmatch_replay(consumed) else "longforcing"] += 1
    assert counts == _objective_counts_at_micro_batch(168)
    assert counts == {"longforcing": 126, "flowmatch_replay": 42}


def test_resume_archives_full_attempt_and_trims_only_active_metric_tail(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("INTERACTWORLD_ROOT", str(tmp_path))
    tmp_path = tmp_path / "run"
    tmp_path.mkdir()
    metadata = b'{"started_unix": 123, "attempt": "old"}'
    original = b"".join(json.dumps({"step": step, "loss": 1 / step}).encode() + b"\n"
                        for step in range(1, 26))
    (tmp_path / "run_metadata.json").write_bytes(metadata)
    (tmp_path / "metrics.jsonl").write_bytes(original)
    checkpoint = tmp_path / "checkpoints" / "step-0000020.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"saved step 20")
    best = checkpoint.with_name("best.pt")
    best.write_bytes(b"historical best step 20")
    rng = torch.get_rng_state().clone()
    result = _prepare_resume_attempt(tmp_path, checkpoint, 20)
    assert torch.equal(torch.get_rng_state(), rng)
    archive = Path(result["archive_dir"])
    assert (archive / "run_metadata.json").read_bytes() == metadata
    assert (archive / "metrics.jsonl").read_bytes() == original
    assert (tmp_path / "run_metadata.json").read_bytes() == metadata
    active = (tmp_path / "metrics.jsonl").read_bytes()
    assert active == b"".join(original.splitlines(keepends=True)[:20])
    assert result["steps_removed_from_active_metrics"] == [21, 22, 23, 24, 25]
    assert result["checkpoint_files_modified"] is False
    assert checkpoint.read_bytes() == b"saved step 20"
    assert best.read_bytes() == b"historical best step 20"
    for entry in result["archived_files"].values():
        assert sha256_file(entry["path"]) == entry["sha256"]
    second = _prepare_resume_attempt(tmp_path, checkpoint, 20)
    assert second["archive_dir"] != result["archive_dir"]
    assert second["steps_removed_from_active_metrics"] == []
    assert (archive / "metrics.jsonl").read_bytes() == original
    for step in range(21, 26):
        append_jsonl(tmp_path / "metrics.jsonl", {"step": step, "loss": 0.1})
    active_steps = [json.loads(line)["step"] for line in
                    (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert active_steps == list(range(1, 26))
    assert (archive / "metrics.jsonl").read_bytes() == original


def test_resume_refuses_ambiguous_metric_order_before_changing_evidence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("INTERACTWORLD_ROOT", str(tmp_path))
    tmp_path = tmp_path / "run"
    tmp_path.mkdir()
    (tmp_path / "run_metadata.json").write_text("{}", encoding="utf-8")
    original = b'{"step":20}\n{"step":21}\n{"step":20}\n'
    (tmp_path / "metrics.jsonl").write_bytes(original)
    checkpoint = tmp_path / "checkpoints" / "step-0000020.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"saved step 20")
    with pytest.raises(ValueError, match="strictly increasing"):
        _prepare_resume_attempt(tmp_path, checkpoint, 20)
    assert (tmp_path / "metrics.jsonl").read_bytes() == original
    assert not (tmp_path / "attempts").exists()
    monkeypatch.setenv("INTERACTWORLD_ROOT", str(tmp_path / "another-project"))
    with pytest.raises(ValueError, match="configured project root"):
        _prepare_resume_attempt(tmp_path, checkpoint, 20)
    assert not (tmp_path / "attempts").exists()
