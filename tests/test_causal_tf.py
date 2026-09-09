from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from training.causal_tf import (
    PINNED_BASE_MODEL,
    STAGE_NAME,
    artifact_hashes,
    assert_no_future_leakage,
    causal_teacher_forcing_loss,
    construct_causal_wrapper,
    frame_blocks,
    initialize_from_action_teacher,
    latent_shape,
    load_causal_config,
    load_teacher_checkpoint,
    teacher_forcing_visibility,
)
from training.models.lora import configure_action_teacher, trainable_state_dict
from training.runtime import sha256_file
from train_causal_teacher_forcing import (
    _checkpoint_payload, _iterator_at_micro_batch, _restore_resume, validation_report,
)

ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "train" / "causal_teacher_forcing_v1.yaml"


def _load_production_simple_adapter():
    """Load the exact adapter classes without importing optional CUDA kernels."""

    path = ROOT / "wan" / "modules" / "model.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in {"ResidualBlock", "SimpleAdapter"}
    ]
    namespace = {"nn": nn}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["SimpleAdapter"]


SimpleAdapter = _load_production_simple_adapter()


class _Attention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_attn = _Attention(dim)
        self.cross_attn = _Attention(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))


class CausalWanModel(nn.Module):
    """Tiny structural stand-in for upstream interface tests only."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(8)])
        self.act_control_adapter = SimpleAdapter(
            32,
            8,
            kernel_size=(2, 2),
            stride=(2, 2),
            downscale_factor=16,
        )
        self.num_frame_per_block = 3
        self.independent_first_frame = False

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        act_context=None,
    ):
        return x

    def _prepare_teacher_forcing_mask(self):
        return None

    def _maybe_build_block_mask(self):
        return None


class _TinyCausalWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = CausalWanModel()
        self.uniform_timestep = False

    def forward(self, noisy, conditional_dict, timestep, clean_x=None, aug_t=None, **kwargs):
        return noisy


def test_config_fixes_causal_geometry_and_training_contract() -> None:
    config = load_causal_config(CONFIG_PATH)
    assert config.model.base_model_path == PINNED_BASE_MODEL
    assert config.model.model_type == "ci2v"
    assert config.model.num_frame_per_block == 3
    assert config.model.independent_first_frame is True
    assert config.model.teacher_aug_t == 0.0
    assert config.model.downscale_factor_control_adapter == 16
    assert latent_shape(config) == (1, 13, 48, 30, 52)
    assert frame_blocks(13, frames_per_block=3, independent_first_frame=True) == (
        (0, 1),
        (1, 4),
        (4, 7),
        (7, 10),
        (10, 13),
    )
    assert config.training.precision == "bf16"
    assert config.training.micro_batch_size == 1
    assert config.training.gradient_accumulation_steps == 8
    assert (config.training.keep_best, config.training.keep_last) == (1, 2)


def test_teacher_forcing_mask_has_no_future_block_or_clean_target_leakage() -> None:
    mask = teacher_forcing_visibility()
    assert mask.shape == (26, 26)
    assert_no_future_leakage(mask)

    # Clean frame 4 is in block [4,7): it can see its block through frame 6,
    # but not clean frame 7 or anything in the noisy branch.
    assert mask[4, 6]
    assert not mask[4, 7]
    assert not mask[4, 13]

    # The noisy query for frame 4 sees completed clean frames [0,4) and its
    # own noisy block [4,7), never clean frame 4 (the current target).
    noisy_query = 13 + 4
    assert mask[noisy_query, 3]
    assert not mask[noisy_query, 4]
    assert mask[noisy_query, 13 + 6]
    assert not mask[noisy_query, 13 + 3]
    assert not mask[noisy_query, 13 + 7]

    leaked = mask.clone()
    leaked[noisy_query, 4] = True
    with pytest.raises(ValueError, match="leaks future token"):
        assert_no_future_leakage(leaked)


def test_constructor_explicitly_selects_causal_model_and_independent_first_frame() -> None:
    config = load_causal_config(CONFIG_PATH)
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return _TinyCausalWrapper()

    wrapper = construct_causal_wrapper(config, factory)
    assert captured["is_causal"] is True
    assert captured["model_type"] == "ci2v"
    assert captured["num_frame_per_block"] == 3
    assert captured["downscale_factor_control_adapter"] == 16
    assert wrapper.uniform_timestep is False
    assert wrapper.model.independent_first_frame is True


def test_real_action_adapter_matches_patch_grid_and_inherits_exact_state() -> None:
    adapter = SimpleAdapter(
        32,
        8,
        kernel_size=(2, 2),
        stride=(2, 2),
        downscale_factor=16,
    )
    action = torch.zeros(1, 32, 2, 32, 32)
    assert adapter(action).shape == (1, 8, 2, 1, 1)
    assert adapter.conv.in_channels == 32 * 16 * 16

    config = load_causal_config(CONFIG_PATH)
    source = _TinyCausalWrapper()
    target = _TinyCausalWrapper()
    configure_action_teacher(source.model, rank=16, alpha=16.0)
    state = trainable_state_dict(source)
    initialize_from_action_teacher(target, {"trainable_model": state}, config)
    assert target.model.act_control_adapter.conv.weight.shape == source.model.act_control_adapter.conv.weight.shape


class _TupleFlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kwargs = None

    def forward(self, noisy, **kwargs):
        self.kwargs = kwargs
        prediction = torch.zeros_like(noisy)
        prediction[:, 0].fill_(123)  # the clean condition must not enter the loss
        return prediction, torch.ones_like(noisy)


def _small_batch(config):
    config.data.height = 32
    config.data.width = 32
    config.model.latent_channels = 4
    shape = latent_shape(config)
    clean = torch.randn(shape)
    target = torch.zeros(shape)
    timesteps = torch.tensor([[0.0] + [500.0] * 12])
    noisy = clean + (timesteps / 1000.0).view(1, 13, 1, 1, 1) * target
    return {
        "noisy_latents": noisy,
        "target_flow": target,
        "timesteps": timesteps,
        "prompt_embeds": torch.zeros(1, 2, 16),
        "actions": torch.zeros(1, 48, 8),
    }, clean


def test_tuple_flow_uses_clean_x_zero_aug_t_and_excludes_first_frame_loss() -> None:
    config = load_causal_config(CONFIG_PATH)
    batch, clean = _small_batch(config)
    model = _TupleFlow()
    loss = causal_teacher_forcing_loss(model, batch, config, torch.device("cpu"))
    assert loss.item() == 0.0
    assert model.kwargs is not None
    assert model.kwargs["replace_first_timestep_and_noise_latents"] is True
    assert torch.count_nonzero(model.kwargs["aug_t"]).item() == 0
    assert model.kwargs["clean_x"].shape == clean.shape
    assert torch.equal(model.kwargs["clean_x"][:, 0], batch["noisy_latents"].bfloat16()[:, 0])
    assert model.kwargs["conditional_dict"]["act_context"][0].shape == (32, 12, 32, 32)


@pytest.mark.parametrize("field", ["timestep", "target"])
def test_independent_first_frame_contract_is_fail_closed(field: str) -> None:
    config = load_causal_config(CONFIG_PATH)
    batch, _ = _small_batch(config)
    if field == "timestep":
        batch["timesteps"][:, 0] = 1
        expected = "timestep zero"
    else:
        batch["target_flow"][:, 0].fill_(1)
        expected = "zero flow target"
    with pytest.raises(ValueError, match=expected):
        causal_teacher_forcing_loss(_TupleFlow(), batch, config, torch.device("cpu"))


def _write_lineage_fixture(tmp_path: Path):
    config = load_causal_config(CONFIG_PATH)
    manifest = tmp_path / "train.jsonl"
    feature_index = tmp_path / "train.features.jsonl"
    feature_receipt = tmp_path / "train.features.jsonl.receipt.json"
    train_config = tmp_path / "causal.yaml"
    manifest.write_text('{"episode_id":"one"}\n', encoding="utf-8")
    feature_index.write_text('{"episode_id":"one"}\n', encoding="utf-8")
    feature_receipt.write_text('{"schema_version":1}\n', encoding="utf-8")
    train_config.write_text("stage: causal\n", encoding="utf-8")
    config.data.manifest_path = str(manifest)
    config.data.feature_index_path = str(feature_index)
    config.data.feature_receipt_path = str(feature_receipt)

    source = _TinyCausalWrapper()
    configure_action_teacher(source.model, rank=16, alpha=16.0)
    with torch.no_grad():
        for index, parameter in enumerate(
            parameter for parameter in source.parameters() if parameter.requires_grad
        ):
            parameter.fill_(0.01 * (index + 1))
    shared = {
        "dataset_manifest": sha256_file(manifest),
        "feature_index": sha256_file(feature_index),
        "feature_receipt": sha256_file(feature_receipt),
    }
    checkpoint = tmp_path / "teacher.pt"
    torch.save(
        {
            "format_version": 1,
            "stage": "action_teacher_lora_v1",
            "step": 50,
            "trainable_model": trainable_state_dict(source),
            "manifest_hashes": {**shared, "training_config": "teacher-config-hash"},
            "config": {
                "model": {
                    "base_model_path": PINNED_BASE_MODEL,
                    "model_type": "ci2v",
                    "lora_rank": 16,
                    "lora_alpha": 16.0,
                    "lora_dropout": 0.0,
                    "action_dim": 32,
                }
            },
        },
        checkpoint,
    )
    config.lineage.checkpoint_path = str(checkpoint)
    hashes = artifact_hashes(config, train_config)
    return config, source, checkpoint, hashes


def test_teacher_checkpoint_is_loaded_by_exact_names_and_bound_into_lineage(tmp_path: Path) -> None:
    config, source, checkpoint, hashes = _write_lineage_fixture(tmp_path)
    payload, lineage = load_teacher_checkpoint(config, hashes)
    assert lineage.stage == "action_teacher_lora_v1"
    assert lineage.step == 50
    assert lineage.sha256 == sha256_file(checkpoint)

    target = _TinyCausalWrapper()
    initialize_from_action_teacher(target, payload, config)
    source_state = trainable_state_dict(source)
    target_state = trainable_state_dict(target)
    assert source_state.keys() == target_state.keys()
    assert all(torch.equal(source_state[name], target_state[name]) for name in source_state)
    assert all(
        "act_control_adapter" in name or ".lora_a." in name or ".lora_b." in name
        for name, parameter in target.named_parameters()
        if parameter.requires_grad
    )


def test_teacher_checkpoint_rejects_changed_feature_lineage(tmp_path: Path) -> None:
    config, _, _, hashes = _write_lineage_fixture(tmp_path)
    hashes["feature_receipt"] = "changed"
    with pytest.raises(ValueError, match="feature_receipt lineage mismatch"):
        load_teacher_checkpoint(config, hashes)


def test_cpu_validation_reports_missing_outputs_without_loading_weights(monkeypatch) -> None:
    config = load_causal_config(CONFIG_PATH)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU validation must not load a checkpoint")

    monkeypatch.setattr(torch, "load", forbidden)
    report = validation_report(config, CONFIG_PATH)
    assert report["stage"] == STAGE_NAME
    assert report["cuda_queried"] is False
    assert report["weights_loaded"] is False


def test_resume_iterator_returns_to_exact_deterministic_position() -> None:
    loader = [{"index": value} for value in range(5)]
    _, iterator = _iterator_at_micro_batch(loader, 8)
    assert next(iterator)["index"] == 3


@pytest.mark.parametrize("path", ["relative.pt", "/cache/prompts.json", "/cache/../prompts.pt", "", 123,
                                  "C:relative.pt", "C:/cache/../prompts.pt"])
def test_causal_prompt_cache_config_rejects_invalid_paths(path):
    config = load_causal_config(CONFIG_PATH)
    config.data.prompt_cache_path = path
    with pytest.raises(ValueError, match="prompt_cache_path"):
        config.validate()


@pytest.mark.parametrize("path", [None, "/project/features/static.pt", "Z:/example/features/static.pt"])
def test_causal_prompt_cache_config_preserves_explicit_and_legacy_values(tmp_path, path):
    import yaml
    config = load_causal_config(CONFIG_PATH)
    config.data.prompt_cache_path = path
    config.validate()
    serialized = config.to_dict()
    if path is None:
        serialized["data"].pop("prompt_cache_path")
    file = tmp_path / "roundtrip.yaml"
    file.write_text(yaml.safe_dump(serialized), encoding="utf-8")
    assert load_causal_config(file).data.prompt_cache_path == path


def _write_static_lineage_fixture(tmp_path):
    config, source, checkpoint, _ = _write_lineage_fixture(tmp_path)
    manifest = Path(config.data.manifest_path)
    index = Path(config.data.feature_index_path)
    feature_receipt = Path(config.data.feature_receipt_path)
    row = {"episode_id": "one", "split": "train"}
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    index.write_text(json.dumps(row) + "\n", encoding="utf-8")
    feature_receipt.write_text(json.dumps({"schema_version": 1, "index": str(index), "manifest": str(manifest),
        "index_sha256": sha256_file(index), "manifest_sha256": sha256_file(manifest)}), encoding="utf-8")
    prompt = tmp_path / "static.pt"
    torch.save({"schema_version": 1, "kind": "scene_static_prompt_cache",
                "prompt_embeds": {"one": torch.zeros(2, 4096, dtype=torch.bfloat16)}}, prompt)
    caption = "A quiet outdoor environment."
    receipt = {"schema_version": 1, "kind": "scene_static_prompt_cache", "prompt_policy": "scene_static_only_v1",
               "cache_path": str(prompt), "cache_sha256": sha256_file(prompt),
               "manifest_sha256": sha256_file(manifest), "feature_index_sha256": sha256_file(index),
               "feature_receipt_sha256": sha256_file(feature_receipt), "encoder": {"kind": "unit-test-fixture"},
               "episodes": {"one": {"split": "train", "prompt": caption,
                                      "prompt_sha256": hashlib.sha256(caption.encode()).hexdigest()}}}
    prompt.with_suffix(prompt.suffix + ".receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    config.data.prompt_cache_path = str(prompt)
    config.model.action_scale = 0.03
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["config"]["data"] = {"prompt_cache_path": str(prompt)}
    payload["config"]["model"]["action_scale"] = 0.03
    payload["manifest_hashes"].update(
        dataset_manifest=sha256_file(manifest), feature_index=sha256_file(index),
        feature_receipt=sha256_file(feature_receipt), prompt_cache=sha256_file(prompt),
        prompt_cache_receipt=sha256_file(prompt.with_suffix(prompt.suffix + ".receipt.json")),
    )
    torch.save(payload, checkpoint)
    hashes = artifact_hashes(config, tmp_path / "causal.yaml")
    return config, source, checkpoint, hashes


def test_static_causal_artifacts_validate_binding_without_loading_payload_or_cuda(tmp_path, monkeypatch):
    config, _, checkpoint, _ = _write_static_lineage_fixture(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("artifact hashing must not load tensor payloads or query CUDA")
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    hashes = artifact_hashes(config, tmp_path / "causal.yaml")
    assert hashes["prompt_cache"] == sha256_file(config.data.prompt_cache_path)
    assert hashes["teacher_checkpoint"] == sha256_file(checkpoint)
    receipt_path = Path(config.data.prompt_cache_path).with_suffix(".pt.receipt.json")
    assert hashes["prompt_cache_receipt"] == sha256_file(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    receipt["prompt_policy"] = "original_feature_cache_prompt"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="kind/policy mismatch"):
        artifact_hashes(config, tmp_path / "causal.yaml")


def test_teacher_static_prompt_path_and_both_hashes_are_inherited_exactly(tmp_path):
    config, _, _, hashes = _write_static_lineage_fixture(tmp_path)
    payload, lineage = load_teacher_checkpoint(config, hashes)
    assert payload["config"]["data"]["prompt_cache_path"] == config.data.prompt_cache_path
    assert lineage.source_manifest_hashes["prompt_cache"] == hashes["prompt_cache"]
    assert lineage.source_manifest_hashes["prompt_cache_receipt"] == hashes["prompt_cache_receipt"]


@pytest.mark.parametrize("change", ["static_to_narrative", "narrative_to_static", "different_path",
                                   "different_cache_hash", "different_receipt_hash", "missing_hash",
                                   "undeclared_source_hashes", "action_scale"])
def test_causal_parent_rejects_prompt_policy_path_hash_and_scale_changes(tmp_path, change):
    config, _, checkpoint, hashes = _write_static_lineage_fixture(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = "prompt"
    if change == "static_to_narrative":
        config.data.prompt_cache_path = None
        hashes.pop("prompt_cache")
        hashes.pop("prompt_cache_receipt")
    elif change == "narrative_to_static":
        payload["config"]["data"].pop("prompt_cache_path")
        payload["manifest_hashes"].pop("prompt_cache")
        payload["manifest_hashes"].pop("prompt_cache_receipt")
    elif change == "different_path":
        config.data.prompt_cache_path = str(tmp_path / "other.pt")
    elif change == "different_cache_hash":
        payload["manifest_hashes"]["prompt_cache"] = "0" * 64
    elif change == "different_receipt_hash":
        payload["manifest_hashes"]["prompt_cache_receipt"] = "0" * 64
    elif change == "missing_hash":
        payload["manifest_hashes"].pop("prompt_cache_receipt")
    elif change == "undeclared_source_hashes":
        payload["config"]["data"].pop("prompt_cache_path")
    else:
        config.model.action_scale = 1.0
        expected = "action_scale"
    torch.save(payload, checkpoint)
    hashes["teacher_checkpoint"] = sha256_file(checkpoint)
    with pytest.raises(ValueError, match=expected):
        load_teacher_checkpoint(config, hashes)


@pytest.mark.parametrize("use_static", [False, True])
def test_causal_resume_keeps_legacy_none_and_strict_prompt_artifact_binding(tmp_path, use_static):
    fixture = _write_static_lineage_fixture if use_static else _write_lineage_fixture
    config, model, _, hashes = fixture(tmp_path)
    _, lineage = load_teacher_checkpoint(config, hashes)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    payload = _checkpoint_payload(model, optimizer, config=config, step=2, micro_batches_consumed=16,
                                  metrics={"loss": 1.0}, manifest_hashes=hashes, teacher_lineage=lineage.as_dict())
    if not use_static:
        # Old checkpoints omitted this optional field; their exact hashes and
        # narrative semantics must remain valid without rewriting old YAMLs.
        payload["config"]["data"].pop("prompt_cache_path")
    resume = tmp_path / "resume.pt"
    torch.save(payload, resume)
    kwargs = dict(model=model, optimizer=optimizer, teacher_lineage=lineage.as_dict(), gradient_accumulation_steps=8)
    assert _restore_resume(resume, hashes=hashes, **kwargs) == (2, 16)
    changed = dict(hashes)
    if use_static:
        changed.pop("prompt_cache")
        changed.pop("prompt_cache_receipt")
    else:
        changed.update(prompt_cache="0" * 64, prompt_cache_receipt="1" * 64)
    with pytest.raises(ValueError, match="resume manifest hashes do not match"):
        _restore_resume(resume, hashes=changed, **kwargs)
