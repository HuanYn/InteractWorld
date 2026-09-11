"""CPU-only lineage/factory contracts for the distinct MoBA candidate stage.

Tiny serialized checkpoints are fixtures, never evidence of learned quality.
No model weights are downloaded and no GPU or production service is used.
"""

from dataclasses import replace
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from test_demo_prompt_contract import _fixture as _old_fixture
from test_rollout15s import _LinkedAdapter, _Writer, _reference
from training.causal_moba import CausalMoBAConfig, METHOD_NAME, method_contract
from training.causal_tf import load_teacher_checkpoint
from training.eval.rollout15s import (
    CAUSAL_STAGE, MOBA_STAGE, LONGFORCING_STAGE, action_script,
    run_rollout_suite, verify_checkpoint_lineage,
)
from training.eval.wan_causal_adapter import (
    WanCausalRolloutAdapter, create_wan_causal_adapter, _denoising_steps,
)
from training.models.lora import LoRALinear, trainable_state_dict
from training.runtime import sha256_file
from train_causal_moba import _checkpoint_payload, _sampling_contract, validate_resume_checkpoint


def _tiny_model():
    model = nn.Module()
    model.model = nn.Module()
    model.model.act_control_adapter = nn.Linear(2, 2)
    block = nn.Module()
    block.self_attn = nn.Module()
    block.self_attn.q = LoRALinear(nn.Linear(2, 2), rank=16, alpha=16.0)
    model.model.blocks = nn.ModuleList([block])
    return model


def _fixture(tmp_path, monkeypatch, *, weight=0.1, resampled=True):
    evaluation, previous, _, _, training_path = _old_fixture(tmp_path, stage="causal", static=True)
    config = CausalMoBAConfig()
    for name, value in previous["config"].items():
        setattr(config, name, type(getattr(config, name))(**value))
    config.training.max_steps = 40
    config.regularization.bidirectional_weight = weight
    if resampled:
        config.data.data_factory = "training.data.action_resampled:build_resampled_action_teacher_dataloader"
    model = _tiny_model()
    teacher_path = Path(config.lineage.checkpoint_path)
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
    teacher["trainable_model"] = trainable_state_dict(model)
    torch.save(teacher, teacher_path)
    config.lineage.checkpoint_sha256 = sha256_file(teacher_path)
    training_path.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
    artifacts = evaluation.lineage.artifact_paths
    hashes = {name: sha256_file(path) for name, path in artifacts.items()}
    _, parent = load_teacher_checkpoint(config, hashes)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    payload = _checkpoint_payload(
        model, torch.optim.AdamW(model.parameters()), config=config, step=20,
        micro_batches_consumed=160, metrics={"loss": 0.2}, manifest_hashes=hashes,
        teacher_lineage=parent.as_dict(),
    )
    evaluation = replace(evaluation, lineage=replace(evaluation.lineage, expected_stage=MOBA_STAGE))
    return _save(evaluation, payload), payload, config


def _save(evaluation, payload):
    path = Path(evaluation.lineage.checkpoint_path)
    torch.save(payload, path)
    return replace(evaluation, lineage=replace(evaluation.lineage, checkpoint_sha256=sha256_file(path)))


@pytest.mark.parametrize("weight,resampled", [(0.0, True), (0.1, True), (0.1, False)])
def test_actual_serializer_positive_new_stage_static_parent_and_method(tmp_path, monkeypatch, weight, resampled):
    evaluation, payload, config = _fixture(tmp_path, monkeypatch, weight=weight, resampled=resampled)
    evaluation.validate()
    result = verify_checkpoint_lineage(evaluation)
    assert result["stage"] == MOBA_STAGE and result["stage"] != CAUSAL_STAGE
    assert result["step"] == 20 and result["parent_teacher"]["stage"] == "action_teacher_lora_v1"
    assert result["method_contract"] == method_contract(config)
    assert result["sampling_contract"] == _sampling_contract(config)
    assert result["regularization"]["bidirectional_weight"] == weight
    assert result["prompt_contract"]["policy"] == "scene_static_only_v1"
    assert result["action_scale"] == 0.03
    assert result["inference_contract"] == dict(
        model_class="CausalWanModel", training_attention_mode="causal", bidirectional_enabled=False,
        context_mode="full_history_kv", denoising_steps=40, streaming_solver="flow_euler",
        timestep_shift=5.0, realtime=False,
    )


@pytest.mark.parametrize("change", [
    "weight", "missing_regularization", "method", "sampling", "sampling_seed", "sampling_length",
    "factory", "step_zero", "step_bool", "step_over_cap", "micro", "initialization",
    "missing_lora", "weight_shape", "weight_nan", "unexpected_weight", "parent_step",
    "parent_path", "parent_hash", "parent_causal", "source_path", "source_config", "truncated_history",
    "dynamic_prompt", "action_scale",
])
def test_new_stage_rejects_rehashed_metadata_weights_and_lineage_tampering(tmp_path, monkeypatch, change):
    evaluation, payload, _ = _fixture(tmp_path, monkeypatch)
    if change == "weight":
        payload["config"]["regularization"]["bidirectional_weight"] = 0.2
    elif change == "missing_regularization":
        payload["config"].pop("regularization")
    elif change == "method":
        payload["method_contract"]["distribution_matching_distillation"] = True
    elif change == "sampling":
        payload["sampling_contract"]["namespace"] = "legacy-replay"
    elif change == "sampling_seed":
        payload["sampling_contract"]["data_seed"] += 1
    elif change == "sampling_length":
        payload["sampling_contract"]["virtual_samples"] += 8
    elif change == "factory":
        payload["config"]["data"]["data_factory"] = "custom:unknown_factory"
    elif change in ("step_zero", "step_bool", "step_over_cap"):
        payload["step"] = {"step_zero": 0, "step_bool": True, "step_over_cap": 41}[change]
    elif change == "micro":
        payload["micro_batches_consumed"] -= 1
    elif change == "initialization":
        payload["initialization_mode"] = "strict_resume_causal_tf"
    elif change == "missing_lora":
        payload["trainable_model"] = {name: value for name, value in payload["trainable_model"].items() if ".lora_b." not in name}
    elif change == "weight_shape":
        payload["trainable_model"]["model.act_control_adapter.weight"] = torch.ones(3, 3)
    elif change == "weight_nan":
        payload["trainable_model"]["model.act_control_adapter.weight"][0, 0] = float("nan")
    elif change == "unexpected_weight":
        payload["trainable_model"]["model.blocks.0.unapproved.weight"] = torch.ones(1)
    elif change in ("parent_step", "parent_path", "parent_hash"):
        key, value = {"parent_step": ("step", 21), "parent_path": ("path", "other.pt"),
                      "parent_hash": ("sha256", "0" * 64)}[change]
        payload["parent_teacher"][key] = value
    elif change == "parent_causal":
        payload["parent_causal"] = {"stage": CAUSAL_STAGE}
    elif change == "source_path":
        payload["config"]["lineage"]["checkpoint_path"] = str(tmp_path / "other.pt")
    elif change == "source_config":
        # Internally consistent metadata still must match the original hashed YAML.
        payload["config"]["regularization"]["bidirectional_weight"] = 0.2
        payload["method_contract"]["bidirectional_weight"] = 0.2
    elif change == "truncated_history":
        payload["config"]["model"]["local_attn_size"] = 12
    elif change == "dynamic_prompt":
        payload["config"]["data"]["prompt_cache_path"] = None
    else:
        payload["config"]["model"]["action_scale"] = 1.0
    evaluation = _save(evaluation, payload)
    with pytest.raises(ValueError):
        verify_checkpoint_lineage(evaluation)


def test_unrepinned_weight_change_fails_checkpoint_sha_gate(tmp_path, monkeypatch):
    evaluation, payload, _ = _fixture(tmp_path, monkeypatch)
    payload["trainable_model"]["model.act_control_adapter.weight"][0, 0] += 0.1
    _save(evaluation, payload)
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        verify_checkpoint_lineage(evaluation)


@pytest.mark.parametrize("field", ["action_scale", "prompt"])
def test_actual_parent_condition_tamper_rejected_even_with_consistent_hash_records(tmp_path, monkeypatch, field):
    evaluation, payload, config = _fixture(tmp_path, monkeypatch)
    teacher_path = Path(config.lineage.checkpoint_path)
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
    if field == "action_scale":
        teacher["config"]["model"]["action_scale"] = 1.0
    else:
        teacher["config"]["data"]["prompt_cache_path"] = None
    torch.save(teacher, teacher_path)
    digest = sha256_file(teacher_path)
    payload["manifest_hashes"]["teacher_checkpoint"] = digest
    payload["parent_teacher"]["sha256"] = digest
    payload["config"]["lineage"]["checkpoint_sha256"] = digest
    training = Path(evaluation.lineage.artifact_paths["training_config"])
    training.write_text(yaml.safe_dump(payload["config"]), encoding="utf-8")
    payload["manifest_hashes"]["training_config"] = sha256_file(training)
    with pytest.raises(ValueError, match=field):
        verify_checkpoint_lineage(_save(evaluation, payload))


@pytest.mark.parametrize("stage", [CAUSAL_STAGE, LONGFORCING_STAGE, "unknown"])
def test_new_checkpoint_cannot_pass_false_expected_stage(tmp_path, monkeypatch, stage):
    evaluation, _, _ = _fixture(tmp_path, monkeypatch)
    evaluation = replace(evaluation, lineage=replace(evaluation.lineage, expected_stage=stage))
    with pytest.raises(ValueError, match="stage|completed"):
        verify_checkpoint_lineage(evaluation)


def test_relabeling_moba_as_legacy_causal_is_not_a_stage_conversion(tmp_path, monkeypatch):
    evaluation, payload, _ = _fixture(tmp_path, monkeypatch)
    payload["stage"] = CAUSAL_STAGE
    evaluation = replace(_save(evaluation, payload), lineage=replace(
        evaluation.lineage, expected_stage=CAUSAL_STAGE,
        checkpoint_sha256=sha256_file(evaluation.lineage.checkpoint_path),
    ))
    with pytest.raises(ValueError, match="distinct stage"):
        verify_checkpoint_lineage(evaluation)
    with pytest.raises(ValueError, match="distinct stage"):
        create_wan_causal_adapter(**_factory_args(evaluation))


@pytest.mark.parametrize("change", ["prompt_pair", "factory", "causal_artifact"])
def test_new_eval_config_requires_explicit_full_history_factory_and_static_artifacts(tmp_path, monkeypatch, change):
    evaluation, _, _ = _fixture(tmp_path, monkeypatch)
    if change == "factory":
        evaluation = replace(evaluation, adapter_factory="other:factory")
    else:
        paths = dict(evaluation.lineage.artifact_paths)
        if change == "prompt_pair":
            paths.pop("prompt_cache")
            paths.pop("prompt_cache_receipt")
        else:
            paths["causal_checkpoint"] = "old-causal.pt"
        evaluation = replace(evaluation, lineage=replace(evaluation.lineage, artifact_paths=paths))
    with pytest.raises(ValueError, match="MoBA"):
        evaluation.validate()


def test_saved_cli_overrides_are_recorded_but_not_misidentified_as_source_tampering(tmp_path, monkeypatch):
    evaluation, payload, config = _fixture(tmp_path, monkeypatch)
    config.training.max_steps = 80
    config.training.output_dir = str(tmp_path / "explicit-cli-output")
    payload["config"] = config.to_dict()
    payload["sampling_contract"] = _sampling_contract(config)
    result = verify_checkpoint_lineage(_save(evaluation, payload))
    assert result["sampling_contract"]["virtual_samples"] == 640


def test_real_extension_serializer_remains_bound_to_original_40step_yaml(tmp_path, monkeypatch):
    evaluation, payload, config = _fixture(tmp_path, monkeypatch)
    payload.update(step=40, micro_batches_consumed=320)
    evaluation = _save(evaluation, payload)
    source_sha = evaluation.lineage.checkpoint_sha256
    original_yaml_sha = sha256_file(evaluation.lineage.artifact_paths["training_config"])
    config.training.max_steps = 260
    _, record = validate_resume_checkpoint(
        evaluation.lineage.checkpoint_path, config=config, hashes=payload["manifest_hashes"],
        teacher_lineage=payload["parent_teacher"], allow_horizon_extension=True,
    )
    model = _tiny_model()
    extended = _checkpoint_payload(model, torch.optim.AdamW(model.parameters()), config=config,
                                   step=60, micro_batches_consumed=480, metrics={"loss": 0.1},
                                   manifest_hashes=payload["manifest_hashes"], teacher_lineage=payload["parent_teacher"],
                                   continuations=[record])
    result = verify_checkpoint_lineage(_save(evaluation, extended))
    assert result["step"] == 60 and result["sampling_contract"]["virtual_samples"] == 2080
    assert record["source_checkpoint_sha256"] == source_sha
    assert extended["manifest_hashes"]["training_config"] == original_yaml_sha
    assert sha256_file(evaluation.lineage.artifact_paths["training_config"]) == original_yaml_sha


def _fake_runtime(monkeypatch, *, truncated=False):
    def namespace(value):
        return SimpleNamespace(**{key: namespace(item) for key, item in value.items()}) if isinstance(value, dict) else value

    class CausalWanModel:
        local_attn_size = 12 if truncated else -1
        num_frame_per_block = 3

    class Component:
        def __init__(self):
            self.model = CausalWanModel()

        def to(self, **kwargs):
            return self

    class Pipeline:
        def __init__(self, config, device):
            self.args = config
            self.generator, self.vae, self.text_encoder = Component(), Component(), Component()

        def requires_grad_(self, value):
            return self

        def eval(self):
            return self

    pipeline_module, config_module = ModuleType("pipeline.causal_inference"), ModuleType("omegaconf")
    pipeline_module.CausalInferencePipeline = Pipeline
    config_module.OmegaConf = SimpleNamespace(create=namespace)
    monkeypatch.setitem(__import__("sys").modules, "pipeline.causal_inference", pipeline_module)
    monkeypatch.setitem(__import__("sys").modules, "omegaconf", config_module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)  # No actual CUDA operation.
    monkeypatch.setattr("training.models.lora.configure_action_teacher", lambda *a, **kw: None)


def _factory_args(evaluation):
    return dict(checkpoint_path=evaluation.lineage.checkpoint_path,
                checkpoint_sha256=evaluation.lineage.checkpoint_sha256,
                base_model_path=evaluation.lineage.expected_base_model_path, device="cuda")


@pytest.mark.parametrize("expected", [None, CAUSAL_STAGE, LONGFORCING_STAGE])
def test_factory_never_auto_enables_moba_for_legacy_web_calls(tmp_path, monkeypatch, expected):
    evaluation, _, _ = _fixture(tmp_path, monkeypatch)
    def no_cuda():
        raise AssertionError("must reject before CUDA query or heavy imports")
    monkeypatch.setattr(torch.cuda, "is_available", no_cuda)
    with pytest.raises(ValueError, match="expected_stage"):
        create_wan_causal_adapter(**_factory_args(evaluation), expected_stage=expected)


def test_factory_explicit_moba_uses_causal_full61_kv_40euler_and_real_stage(tmp_path, monkeypatch):
    evaluation, payload, _ = _fixture(tmp_path, monkeypatch)
    _fake_runtime(monkeypatch)
    with patch("training.models.lora.load_trainable_state_dict") as load:
        adapter = create_wan_causal_adapter(**_factory_args(evaluation), expected_stage=MOBA_STAGE)
    assert type(adapter) is WanCausalRolloutAdapter
    assert adapter.checkpoint_stage == MOBA_STAGE and adapter.context_mode == "full_history_kv"
    assert adapter.checkpoint_method_contract == payload["method_contract"]
    assert adapter.pipeline.args.model_kwargs.local_attn_size == -1
    assert adapter.pipeline.args.image_or_video_shape[1] == 61
    assert adapter.pipeline.args.streaming_solver == "flow_euler"
    assert adapter.pipeline.args.denoising_step_list == list(range(1000, 0, -25))
    assert adapter.action_scale == 0.03
    assert "training_attention_mode" not in vars(adapter.pipeline.args)
    assert load.call_count == 1
    assert set(load.call_args.args[1]) == set(payload["trainable_model"])


def test_factory_rejects_actual_truncated_backbone(tmp_path, monkeypatch):
    evaluation, _, _ = _fixture(tmp_path, monkeypatch)
    _fake_runtime(monkeypatch, truncated=True)
    with pytest.raises(RuntimeError, match="actual full-history CausalWanModel"):
        create_wan_causal_adapter(**_factory_args(evaluation), expected_stage=MOBA_STAGE)


def test_new_suite_receipt_preserves_method_and_explicit_factory_opt_in(tmp_path, monkeypatch):
    evaluation, _, _ = _fixture(tmp_path, monkeypatch)
    lineage = verify_checkpoint_lineage(evaluation)
    evaluation = replace(evaluation, width=2, height=2, run_id="moba-cpu-fixture")
    references = {scene.reference_frames_path: scene for scene in evaluation.scenes}
    kwargs_seen = []

    def factory(**kwargs):
        kwargs_seen.append(kwargs)
        return _LinkedAdapter()

    output, receipt = run_rollout_suite(
        evaluation, lineage=lineage, adapter_factory=factory, device="cpu",
        image_loader=lambda path: np.zeros((2, 2, 3), dtype=np.uint8),
        reference_loader=lambda path, anchors: _reference(action_script(references[str(path)]), anchors),
        writer_factory=lambda path, width, height, fps: _Writer(path),
    )
    assert kwargs_seen[0]["expected_stage"] == MOBA_STAGE
    saved = json.loads((output / "metrics.json").read_text())
    assert saved["lineage"]["stage"] == MOBA_STAGE
    assert saved["lineage"]["method_contract"]["method"] == METHOD_NAME
    assert saved["lineage"]["inference_contract"]["training_attention_mode"] == "causal"
    assert saved["lineage"]["method_contract"]["quality_validated"] is False


def test_old_sampler_contracts_are_unchanged():
    assert _denoising_steps(CAUSAL_STAGE) == list(range(1000, 0, -25))
    assert _denoising_steps(LONGFORCING_STAGE) == [1000, 750, 500, 250]
    assert _denoising_steps(MOBA_STAGE) == _denoising_steps(CAUSAL_STAGE)
    with pytest.raises(ValueError, match="unsupported"):
        _denoising_steps("unknown")


def test_longforcing_parent_allowlist_is_not_widened_to_moba(tmp_path):
    evaluation, payload, _, _, _ = _old_fixture(tmp_path, stage="longforcing", static=True)
    payload["parent_causal"]["stage"] = MOBA_STAGE
    with pytest.raises(ValueError, match="no causal parent lineage"):
        verify_checkpoint_lineage(_save(evaluation, payload))

def test_cli_stage_override_is_explicit_and_does_not_enable_launch():
    from scripts.evaluate_rollout15s import parse_args
    args = parse_args(["--expected-stage", "causal_moba_regularized_v1"])
    assert args.expected_stage == "causal_moba_regularized_v1"
    assert args.launch is False
