from __future__ import annotations

import importlib.util
import json
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from torch import nn

from training.config import load_config
from training.models.action_adapter import (
    build_action_context,
    inject_action_features,
    pack_canonical_actions,
)
from training.models.lora import LoRALinear, configure_action_teacher
from training.runtime import CheckpointManager, load_checkpoint, sha256_file
from train_action_teacher import (
    _forward_loss,
    _iterator_at_micro_batch,
    _load_gate_checkpoint,
    _load_warm_start_checkpoint,
    parse_args,
)
from utils.action_alignment import expand_frame_conditioning_to_tokens, reset_missing_action_adapter

ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "train" / "action_teacher_lora_v1.yaml"


def _import_wan_model_without_optional_triton(monkeypatch: pytest.MonkeyPatch):
    """Load the real model files without importing GPU-only sibling modules."""

    wan_package = types.ModuleType("wan")
    wan_package.__path__ = [str(ROOT / "wan")]
    modules_package = types.ModuleType("wan.modules")
    modules_package.__path__ = [str(ROOT / "wan" / "modules")]
    diffusers_package = types.ModuleType("diffusers")
    diffusers_configuration = types.ModuleType("diffusers.configuration_utils")
    diffusers_models = types.ModuleType("diffusers.models")
    diffusers_modeling = types.ModuleType("diffusers.models.modeling_utils")
    diffusers_configuration.ConfigMixin = type("ConfigMixin", (), {})
    diffusers_configuration.register_to_config = lambda function: function
    diffusers_modeling.ModelMixin = nn.Module
    sla_attn = types.ModuleType("wan.modules.sla_attn")
    sla_attn.get_block_map = lambda *args, **kwargs: None
    sla_kernel = types.ModuleType("wan.modules.sla_kernel")
    sla_kernel._attention = types.SimpleNamespace(apply=lambda *args, **kwargs: None)
    for name, module in (
        ("wan", wan_package),
        ("wan.modules", modules_package),
        ("diffusers", diffusers_package),
        ("diffusers.configuration_utils", diffusers_configuration),
        ("diffusers.models", diffusers_models),
        ("diffusers.models.modeling_utils", diffusers_modeling),
        ("wan.modules.sla_attn", sla_attn),
        ("wan.modules.sla_kernel", sla_kernel),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    def load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    wan_attention_module = load(
        "wan.modules.attention", ROOT / "wan" / "modules" / "attention.py"
    )
    wan_model_module = load("wan.modules.model", ROOT / "wan" / "modules" / "model.py")
    return wan_model_module, wan_attention_module


def test_v1_config_has_exact_rgb_latent_and_training_contract() -> None:
    config = load_config(CONFIG_PATH)
    assert (config.data.num_frames, config.data.height, config.data.width) == (49, 480, 832)
    latent_frames = 1 + (config.data.num_frames - 1) // config.model.temporal_compression
    latent_height = config.data.height // config.model.spatial_compression
    latent_width = config.data.width // config.model.spatial_compression
    assert (latent_frames, config.model.latent_channels, latent_height, latent_width) == (
        13,
        48,
        30,
        52,
    )
    assert config.model.independent_first_frame is True
    assert config.model.num_frame_per_block == 3
    assert (latent_frames - 1) // config.model.num_frame_per_block == 4
    assert config.model.lora_rank == 16
    assert config.training.precision == "bf16"
    assert config.training.micro_batch_size == 1
    assert config.training.gradient_accumulation_steps == 8
    assert config.training.checkpoint_every == 50
    assert (config.training.keep_best, config.training.keep_last) == (1, 2)
    assert config.training.seed == 42
    assert config.model.base_model_path == (
        "/path/to/interactworld/models/"
        "Wan2.2-TI2V-5B@921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
    )


def test_missing_action_adapter_is_deterministically_initialized() -> None:
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.act_control_adapter = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))

    first, second = Backbone(), Backbone()
    keys = [f"act_control_adapter.{name}" for name in first.act_control_adapter.state_dict()]
    assert reset_missing_action_adapter(first, keys)
    assert reset_missing_action_adapter(second, keys)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            first.act_control_adapter.state_dict().values(),
            second.act_control_adapter.state_dict().values(),
        )
    )
    with pytest.raises(RuntimeError, match="partial action adapter"):
        reset_missing_action_adapter(first, keys[:1])


class _TupleReturningWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kwargs = None

    def forward(self, noisy, **kwargs):
        self.kwargs = kwargs
        flow = torch.zeros_like(noisy)
        flow[:, 0] = 99
        return flow, torch.ones_like(noisy)


def test_forward_loss_uses_flow_output_and_preserves_per_frame_timestep_contract() -> None:
    config = load_config(CONFIG_PATH)
    model = _TupleReturningWrapper()
    shape = (1, 13, 48, 30, 52)
    batch = {
        "noisy_latents": torch.zeros(shape),
        "target_flow": torch.zeros(shape),
        "timesteps": torch.tensor([[0] + [500] * 12]),
        "prompt_embeds": torch.zeros(1, 2, 16),
        "actions": torch.zeros(1, 48, 8),
    }
    loss = _forward_loss(model, batch, config, torch.device("cpu"))
    assert loss.item() == 0.0
    assert model.kwargs is not None
    assert model.kwargs["replace_first_timestep_and_noise_latents"] is True


def test_per_frame_conditioning_expands_over_spatial_tokens() -> None:
    conditioning = torch.tensor([[[1.0], [2.0], [3.0]]])
    grid = torch.tensor([[3, 2, 2]])
    expanded = expand_frame_conditioning_to_tokens(conditioning, grid, 12)
    assert expanded.shape == (1, 12, 1)
    assert expanded[0, :, 0].tolist() == [1.0] * 4 + [2.0] * 4 + [3.0] * 4


def test_official_action_adapter_downscales_480x832_to_latent_patch_grid() -> None:
    source = (ROOT / "wan" / "modules" / "model.py").read_text(encoding="utf-8")
    assert "downscale_factor_control_adapter=16" in source
    assert "downscale_factor=downscale_factor_control_adapter" in source


def test_noncausal_ci2v_forward_dispatches_attention_to_cpu_sdpa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wan_model, wan_attention = _import_wan_model_without_optional_triton(monkeypatch)
    monkeypatch.setattr(wan_attention, "DEFAULT_ATTN_BACKEND", "sdpa")
    calls: list[str | None] = []
    real_dispatch = wan_attention.attention

    def traced_dispatch(*args, **kwargs):
        calls.append(kwargs.get("backend"))
        return real_dispatch(*args, **kwargs)

    monkeypatch.setattr(wan_model, "attention", traced_dispatch)
    model = wan_model.WanModel(
        model_type="ci2v",
        patch_size=(1, 2, 2),
        text_len=4,
        in_dim=4,
        dim=8,
        ffn_dim=16,
        freq_dim=8,
        text_dim=8,
        out_dim=4,
        num_heads=2,
        num_layers=1,
        act_control_in_dim=32,
        downscale_factor_control_adapter=16,
    ).to(dtype=torch.bfloat16)
    assert isinstance(model.blocks[0].cross_attn, wan_model.WanCrossAttention)
    assert not isinstance(model.blocks[0].cross_attn, wan_model.WanI2VCrossAttention)

    output = model(
        torch.randn(1, 4, 2, 4, 4, dtype=torch.bfloat16),
        t=torch.tensor([[0.0, 500.0]]),
        context=[torch.randn(4, 8, dtype=torch.bfloat16)],
        seq_len=None,
        act_context=[torch.zeros(32, 1, 64, 64, dtype=torch.bfloat16)],
    )
    assert output.shape == (1, 4, 2, 4, 4)
    assert calls == ["auto", "auto"]


def test_resume_iterator_returns_to_exact_deterministic_position() -> None:
    loader = ["a", "b", "c", "d"]
    _, iterator = _iterator_at_micro_batch(loader, 10)
    assert next(iterator) == "c"


def test_four_rgb_actions_pack_to_one_ordered_32d_token() -> None:
    actions = torch.arange(4 * 8, dtype=torch.float32).reshape(1, 4, 8)
    # Packing validates binary values, so use a distinct binary pattern per frame.
    actions = (actions % 3 == 0).float()
    packed = pack_canonical_actions(actions)
    assert packed.shape == (1, 1, 32)
    assert torch.equal(packed[0, 0], actions[0].transpose(0, 1).flatten())


def test_49_rgb_frame_window_builds_12_action_tokens_and_preserves_clean_latent() -> None:
    actions = torch.zeros(1, 48, 8)
    actions[:, :, 0] = 1
    context = build_action_context(actions, height=16, width=32)
    assert len(context) == 1
    assert context[0].shape == (32, 12, 16, 32)

    video = torch.zeros(1, 4, 13, 2, 3)
    action_features = torch.ones(1, 4, 12, 2, 3)
    injected = inject_action_features(video, action_features)
    assert torch.count_nonzero(injected[:, :, :1]) == 0
    assert torch.all(injected[:, :, 1:] == 1)


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


class _TinyWan(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dim = 8
        self.patch_size = (1, 2, 2)
        self.patch_embedding = nn.Conv3d(4, 8, 1)
        self.blocks = nn.ModuleList([_Block(8)])
        # This stands in for the already-materialized official SimpleAdapter.
        self.act_control_adapter = nn.Sequential(nn.Linear(32, 8), nn.SiLU())


def test_only_existing_official_adapter_and_rank16_lora_are_trainable() -> None:
    backbone = _TinyWan()
    original_adapter = backbone.act_control_adapter
    summary = configure_action_teacher(backbone, rank=16, alpha=16.0)
    assert backbone.act_control_adapter is original_adapter
    assert summary.replaced_linear_layers
    assert all(isinstance(block.self_attn.q, LoRALinear) for block in backbone.blocks)
    trainable_names = {name for name, parameter in backbone.named_parameters() if parameter.requires_grad}
    assert trainable_names
    assert all(
        "act_control_adapter" in name or ".lora_a." in name or ".lora_b." in name
        for name in trainable_names
    )
    assert backbone.patch_embedding.weight.requires_grad is False
    assert backbone.blocks[0].self_attn.q.base.weight.requires_grad is False
    assert backbone.blocks[0].self_attn.q.rank == 16


def test_configuration_rejects_wrong_lora_rank(tmp_path: Path) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8").replace("lora_rank: 16", "lora_rank: 8")
    bad = tmp_path / "bad.yaml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="LoRA rank"):
        load_config(bad)


def test_checkpoint_keeps_best_one_and_last_two_and_checks_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"episode":"one"}\n', encoding="utf-8")
    hashes = {"dataset_manifest": sha256_file(manifest)}
    manager = CheckpointManager(tmp_path / "run", keep_last=2)
    for step, metric in ((50, 3.0), (100, 2.0), (150, 2.5)):
        manager.save(
            {"step": step, "manifest_hashes": hashes, "tensor": torch.tensor([step])},
            step=step,
            metric=metric,
        )
    names = {path.name for path in (tmp_path / "run" / "checkpoints").glob("*.pt")}
    assert names == {"best.pt", "step-0000100.pt", "step-0000150.pt"}
    resumed = load_checkpoint(
        tmp_path / "run" / "checkpoints" / "step-0000150.pt",
        expected_manifest_hashes=hashes,
    )
    assert resumed["step"] == 150
    with pytest.raises(ValueError, match="manifest hashes"):
        load_checkpoint(
            tmp_path / "run" / "checkpoints" / "step-0000150.pt",
            expected_manifest_hashes={"dataset_manifest": "wrong"},
        )


def test_completed_gate20_can_initialize_full_run_without_repeating_steps(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "train" / "action_teacher_5090_week.yaml")
    source_config = config.to_dict()
    source_config["training"] = {
        **source_config["training"],
        "max_steps": 20,
        "output_dir": "/path/to/interactworld/runs/abot-week-v1/action-gate20",
    }
    current_hashes = {
        "dataset_manifest": "manifest",
        "feature_index": "index",
        "feature_receipt": "receipt",
        "training_config": "full-config",
    }
    checkpoint = tmp_path / "step-0000020.pt"
    torch.save(
        {
            "format_version": 1,
            "stage": "action_teacher_lora_v1",
            "step": 20,
            "micro_batches_consumed": 160,
            "trainable_model": {"act_control_adapter.weight": torch.zeros(1)},
            "optimizer": {},
            "config": source_config,
            "manifest_hashes": {**current_hashes, "training_config": "gate-config"},
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": [],
        },
        checkpoint,
    )

    loaded = _load_gate_checkpoint(
        checkpoint,
        config=config,
        current_manifest_hashes=current_hashes,
    )
    assert loaded["step"] == 20
    assert loaded["micro_batches_consumed"] == 20 * config.training.gradient_accumulation_steps

    wrong = dict(current_hashes)
    wrong["feature_index"] = "different"
    with pytest.raises(ValueError, match="feature_index mismatch"):
        _load_gate_checkpoint(checkpoint, config=config, current_manifest_hashes=wrong)


def _warm_start_fixture(tmp_path: Path):
    config = load_config(CONFIG_PATH)
    config.training.output_dir = str(tmp_path / "repair")
    source_config = config.to_dict()
    source_config["training"]["output_dir"] = str(tmp_path / "old-run")
    hashes = {
        "dataset_manifest": "manifest",
        "feature_index": "index",
        "feature_receipt": "receipt",
        "training_config": "repair-config",
    }
    model = _TinyWan()
    configure_action_teacher(model)
    state = {
        name: torch.full_like(parameter, 0.25)
        for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    payload = {
        "format_version": 1,
        "stage": "action_teacher_lora_v1",
        "step": 840,
        "micro_batches_consumed": 6720,
        "trainable_model": state,
        "optimizer": {"must_not_be_restored": True},
        "torch_rng_state": torch.tensor([255], dtype=torch.uint8),
        "cuda_rng_state_all": [torch.tensor([255], dtype=torch.uint8)],
        "config": source_config,
        "manifest_hashes": {**hashes, "training_config": "source-config"},
    }
    path = tmp_path / "source-best840.pt"
    torch.save(payload, path)
    return config, hashes, path, payload


@pytest.mark.parametrize("other", ["--resume", "--initialize-from"])
def test_warm_start_cli_is_mutually_exclusive_with_existing_resume_modes(other: str) -> None:
    args = parse_args(["--warm-start-from", "source.pt"])
    assert args.warm_start_from == "source.pt"
    with pytest.raises(SystemExit):
        parse_args(["--warm-start-from", "source.pt", other, "other.pt"])


def test_warm_start_accepts_new_scale_and_optimizer_but_returns_weights_only(tmp_path: Path) -> None:
    config, hashes, path, source = _warm_start_fixture(tmp_path)
    config.model.action_scale = 0.03
    config.optimizer.adapter_lr = 2e-5
    config.optimizer.lora_lr = 0.0
    config.training.max_steps = 240
    state, lineage = _load_warm_start_checkpoint(
        path, config=config, current_manifest_hashes=hashes,
    )
    assert set(state) == set(source["trainable_model"])
    assert all(torch.equal(value, source["trainable_model"][name]) for name, value in state.items())
    assert lineage["sha256"] == sha256_file(path)
    assert lineage["source_step"] == 840
    assert lineage["source_stage"] == "action_teacher_lora_v1"
    assert lineage["source_config"] == source["config"]
    assert lineage["source_manifest_hashes"] == source["manifest_hashes"]
    assert lineage["source_config"]["model"]["action_scale"] == 1.0
    assert lineage["mode"] == "warm_start_weights_only"
    assert lineage["optimizer_restored"] is False
    assert lineage["rng_restored"] is False
    assert lineage["start_step"] == lineage["micro_batches_consumed"] == 0
    assert "optimizer" not in state and "torch_rng_state" not in state


@pytest.mark.parametrize("field", ["base_model_path", "lora_alpha", "action_dim", "timestep_shift"])
def test_warm_start_rejects_changed_model_contract(tmp_path: Path, field: str) -> None:
    config, hashes, path, payload = _warm_start_fixture(tmp_path)
    payload["config"]["model"][field] = "different"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="model architecture/conditioning"):
        _load_warm_start_checkpoint(path, config=config, current_manifest_hashes=hashes)


@pytest.mark.parametrize("field", ["dataset_manifest", "feature_index", "feature_receipt"])
def test_warm_start_rejects_changed_cache_binding(tmp_path: Path, field: str) -> None:
    config, hashes, path, _ = _warm_start_fixture(tmp_path)
    hashes[field] = "different"
    with pytest.raises(ValueError, match=f"{field} mismatch"):
        _load_warm_start_checkpoint(path, config=config, current_manifest_hashes=hashes)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("stage", "causal_teacher_forcing_v1", "action_teacher_lora_v1"),
        ("step", 0, "positive optimizer step"),
        ("step", True, "positive optimizer step"),
        ("trainable_model", {}, "no trainable model state"),
        ("trainable_model", {"weight": "not a tensor"}, "map parameter names to tensors"),
    ],
)
def test_warm_start_rejects_invalid_parent(tmp_path: Path, field, value, message) -> None:
    config, hashes, path, payload = _warm_start_fixture(tmp_path)
    payload[field] = value
    torch.save(payload, path)
    with pytest.raises(ValueError, match=message):
        _load_warm_start_checkpoint(path, config=config, current_manifest_hashes=hashes)


def test_warm_start_launch_starts_at_zero_with_fresh_optimizer_rng_and_saved_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import train_action_teacher as entry

    config, hashes, path, source = _warm_start_fixture(tmp_path)
    config.model.action_scale = 0.03
    config.optimizer.lora_lr = 0.0
    config.training.max_steps = 1
    parent_hash = sha256_file(path)
    model = _TinyWan()
    summary = configure_action_teacher(model)
    seen_batches = []
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)

    def reject_restore(*args, **kwargs):
        raise AssertionError("warm start must not restore parent optimizer/RNG")

    monkeypatch.setattr(optimizer, "load_state_dict", reject_restore)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", reject_restore)
    monkeypatch.setattr(torch, "set_rng_state", reject_restore)
    monkeypatch.setattr(entry, "validate_confirmation", lambda value: None)
    monkeypatch.setattr(entry, "query_dedicated_gpu", lambda **kwargs: types.SimpleNamespace(as_dict=lambda: {}))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name in ("set_device", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *args, **kwargs: None)
    for name in ("memory_allocated", "memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *args, **kwargs: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "CPU test fixture")
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(entry, "peak_vram_bytes", lambda device: 0)
    seeds = []
    monkeypatch.setattr(entry, "seed_everything", seeds.append)
    monkeypatch.setattr(entry, "_manifest_hashes", lambda *args: hashes)
    monkeypatch.setattr(entry, "_build_model", lambda *args: (model, summary))
    monkeypatch.setattr(entry, "_build_optimizer", lambda *args: optimizer)
    monkeypatch.setattr(entry, "_import_factory", lambda spec: lambda **kwargs: list(range(8)))
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())

    def fake_loss(model, batch, config, device):
        seen_batches.append(batch)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert torch.equal(parameter, source["trainable_model"][name])
        return sum(p.square().sum() for p in model.parameters() if p.requires_grad)

    monkeypatch.setattr(entry, "_forward_loss", fake_loss)
    entry.launch(
        config, CONFIG_PATH, None, None,
        confirmed_gpu_index=0, confirmed_gpu_uuid="cpu-fixture",
        confirmed_at_utc="cpu-fixture", allocation_profile="cpu-fixture",
        warm_start_from=str(path),
    )
    assert seeds == [42]
    assert seen_batches == list(range(8))
    saved = torch.load(
        Path(config.training.output_dir) / "checkpoints" / "step-0000001.pt",
        weights_only=False,
    )
    assert saved["step"] == 1
    assert saved["micro_batches_consumed"] == 8
    assert saved["config"]["model"]["action_scale"] == 0.03
    assert saved["optimizer"]["state"] == {}
    assert saved["initialization"]["source_step"] == 840
    assert saved["initialization"]["sha256"] == parent_hash
    assert saved["initialization"]["optimizer_restored"] is False
    assert saved["initialization"]["rng_restored"] is False
    metadata = json.loads((Path(config.training.output_dir) / "run_metadata.json").read_text())
    assert metadata["initialization"] == json.loads(json.dumps(saved["initialization"]))
    assert sha256_file(path) == parent_hash


def test_warm_start_launch_rejects_conflicting_modes_before_gpu_gate() -> None:
    from train_action_teacher import launch

    config = load_config(CONFIG_PATH)
    with pytest.raises(ValueError, match="mutually exclusive"):
        launch(
            config, CONFIG_PATH, "resume.pt", None,
            warm_start_from="parent.pt", confirmed_gpu_index=0,
            confirmed_gpu_uuid="no-gpu", confirmed_at_utc="no-gpu",
            allocation_profile="no-gpu",
        )
