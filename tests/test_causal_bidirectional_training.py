"""CPU checks of actual small Wan blocks/model with reference attention kernels.

Only the optional CUDA import/compiled-kernel boundary is replaced. Production
patch/action embeddings, QKV, RoPE, masks, modulation, blocks, head, dispatch and
checkpoint functions are executed from their source, not structural stand-ins.
These tests do not establish GPU kernel performance or learned video quality.
"""

import ast
import copy
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
import torch.utils.checkpoint


ROOT = Path(__file__).resolve().parents[1]


def _dense_mask(mask, length):
    index = torch.arange(length)
    return mask.mask_mod(torch.tensor(0), torch.tensor(0), index[:, None], index[None, :])


def _cpu_flex(*, query, key, value, block_mask):
    # Dense reference is confined to tiny CPU tests; never installed in runtime.
    assert query.device.type == "cpu" and query.shape[-2] <= 256
    mask = _dense_mask(block_mask, query.shape[-2])
    return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)


def _cpu_flash(q, k, v, k_lens=None, **kwargs):
    assert q.device.type == "cpu" and k_lens is None
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
    ).transpose(1, 2)


def _load_actual_classes():
    namespace = dict(
        torch=torch, nn=nn, math=math, os=os, Any=Any,
        ModelMixin=nn.Module, ConfigMixin=object,
        register_to_config=lambda method: method,
        BlockMask=BlockMask, create_block_mask=create_block_mask,
        attention=_cpu_flash, flash_attention=_cpu_flash, flex_attention=_cpu_flex,
    )
    for relative in ("wan/modules/model.py", "wan/modules/causal_model.py"):
        path = ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                or (isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "WAN_CROSSATTENTION_CLASSES"
                    for target in node.targets))]
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


ACTUAL = _load_actual_classes()
CausalWanModel = ACTUAL["CausalWanModel"]


def _load_wrapper_forward():
    path = ROOT / "utils/wan_wrapper.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "WanDiffusionWrapper")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    namespace = dict(torch=torch, List=List, Optional=Optional)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["forward"]


WRAPPER_FORWARD = _load_wrapper_forward()


@pytest.fixture(autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _model():
    torch.manual_seed(812)
    model = CausalWanModel(
        model_type="t2v", patch_size=(1, 1, 1), text_len=3,
        in_dim=2, dim=12, ffn_dim=24, freq_dim=8, text_dim=8,
        out_dim=2, num_heads=2, num_layers=2, num_frame_per_block=1,
        act_control_in_dim=2, downscale_factor_control_adapter=1,
    )
    model.independent_first_frame = True
    nn.init.normal_(model.head.head.weight, std=0.1)  # Production init is zero.
    model.act_control_adapter.requires_grad_(True)
    return model


def _inputs(batch=1):
    generator = torch.Generator().manual_seed(913)
    return dict(
        x=torch.randn(batch, 2, 4, 2, 2, generator=generator),
        t=torch.tensor([[0., 400., 600., 800.]]).expand(batch, -1).clone(),
        context=torch.randn(batch, 3, 8, generator=generator), seq_len=16,
        act_context=torch.randn(batch, 2, 3, 2, 2, generator=generator),
        act_context_scale=0.03,
    )


def _attention_state(model):
    return [dict(vars(block.self_attn)) for block in model.blocks]


@pytest.mark.parametrize("length", [1, 16, 128, 129, 254])
def test_full_bidirectional_mask_and_padding_self_only(length):
    mask = CausalWanModel._prepare_bidirectional_training_mask("cpu", length)
    padded = math.ceil(length / 128) * 128
    dense = _dense_mask(mask, padded)
    expected = torch.eye(padded, dtype=torch.bool)
    expected[:length, :length] = True
    assert torch.equal(dense, expected)
    assert bool(mask.to_dense().all())  # Every listed block has valid edges.
    assert mask.kv_indices.numel() == (padded // 128) ** 2


def test_mask_constructor_never_builds_dense_token_tensor(monkeypatch):
    shapes = []
    actual_ones = torch.ones

    def checked_ones(shape, **kwargs):
        shapes.append(shape)
        assert math.prod(shape) < 100_000
        return actual_ones(shape, **kwargs)

    monkeypatch.setattr(torch, "ones", checked_ones)
    mask = CausalWanModel._prepare_bidirectional_training_mask("cpu", 23_790)
    assert shapes == [(186, 186)]
    assert mask.kv_indices.numel() == 186 ** 2
    with pytest.raises(ValueError, match="nonempty"):
        CausalWanModel._prepare_bidirectional_training_mask("cpu", 0)


def test_actual_self_attention_matches_unpadded_dense_reference():
    model = _model()
    block = model.blocks[0].self_attn
    value = torch.randn(1, 16, 12)
    grids = torch.tensor([[4, 2, 2]])
    mask = model._prepare_bidirectional_training_mask("cpu", 16)
    actual = block(value, torch.tensor([16]), grids, model.freqs, block_mask=mask)
    q = block.norm_q(block.q(value)).view(1, 16, 2, 6)
    k = block.norm_k(block.k(value)).view(1, 16, 2, 6)
    v = block.v(value).view(1, 16, 2, 6)
    q = ACTUAL["causal_rope_apply"](q, grids, model.freqs).type_as(v)
    k = ACTUAL["causal_rope_apply"](k, grids, model.freqs).type_as(v)
    expected = block.o(_cpu_flash(q, k, v).flatten(2))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("batch", [1, 2])
def test_actual_model_bidirectional_future_attention_and_default_causality(batch):
    model, values = _model(), _inputs(batch)
    changed = dict(values, x=values["x"].clone())
    changed["x"][:, :, -1] += 2.5
    with torch.no_grad():
        default = model(**values)
        explicit = model(**values, training_attention_mode="causal")
        causal_changed = model(**changed)
        bid = model(**values, training_attention_mode="bidirectional")
        bid_changed = model(**changed, training_attention_mode="bidirectional")
    torch.testing.assert_close(default, explicit, rtol=0, atol=0)
    torch.testing.assert_close(default[:, :, :3], causal_changed[:, :, :3], rtol=0, atol=0)
    assert (bid[:, :, :3] - bid_changed[:, :, :3]).abs().max() > 1e-5
    assert torch.isfinite(bid).all()


def test_tf_bid_tf_restores_exact_mask_cache_flags_and_input_values():
    model, values = _model(), _inputs()
    clean = values["x"].clone()
    original_x, original_t = values["x"].clone(), values["t"].clone()
    with torch.no_grad():
        first = model(**values, clean_x=clean)
        mask, cache_key = model.block_mask, model._block_mask_cache_key
        cache = dict(model._block_mask_cache)
        state = _attention_state(model)
        model(**values, training_attention_mode="bidirectional")
        assert model.block_mask is mask and model._block_mask_cache_key == cache_key
        assert model._block_mask_cache == cache
        assert _attention_state(model) == state
        again = model(**values, clean_x=clean)
    torch.testing.assert_close(first, again, rtol=0, atol=0)
    torch.testing.assert_close(values["x"], original_x, rtol=0, atol=0)
    torch.testing.assert_close(values["t"], original_t, rtol=0, atol=0)


def test_bid_training_does_not_change_subsequent_actual_kv_inference():
    model, values = _model(), _inputs()
    baseline = copy.deepcopy(model)
    original_keys = tuple(model.state_dict())
    with torch.no_grad():
        model(**values, training_attention_mode="bidirectional")
    assert tuple(model.state_dict()) == original_keys
    assert model.block_mask is None and not model._block_mask_cache
    assert not hasattr(model.blocks[0].self_attn, "_is_teacher_forcing")

    def infer(net):
        cache = [dict(k=torch.zeros(1, 16, 2, 6), v=torch.zeros(1, 16, 2, 6),
                      global_end_index=torch.tensor(0), local_end_index=torch.tensor(0))
                 for _ in net.blocks]
        parts = []
        with torch.no_grad():
            for start in range(2):
                parts.append(net(
                    x=values["x"][:, :, start:start + 1],
                    t=values["t"][:, start:start + 1], context=values["context"],
                    seq_len=16, kv_cache=cache, current_start=start * 4,
                ))
        return torch.cat(parts, dim=2), cache

    actual, actual_cache = infer(model)
    expected, expected_cache = infer(baseline)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_layer, expected_layer in zip(actual_cache, expected_cache):
        for name in actual_layer:
            torch.testing.assert_close(actual_layer[name], expected_layer[name], rtol=0, atol=0)
        assert actual_layer["global_end_index"].item() == 8


@pytest.mark.parametrize("checkpoint_mode", ["non_reentrant", "reentrant"])
def test_bid_checkpoint_recompute_after_tf_has_real_lora_adapter_gradients(checkpoint_mode):
    from training.models.lora import LoRALinear

    model = _model()
    model.requires_grad_(False)
    for block in model.blocks:
        block.self_attn.q = LoRALinear(block.self_attn.q, rank=2, alpha=2.0)
        nn.init.normal_(block.self_attn.q.lora_b.weight, std=0.03)
    model.act_control_adapter.requires_grad_(True)
    reference = copy.deepcopy(model)
    model.gradient_checkpointing = True
    model.gradient_checkpointing_mode = checkpoint_mode
    values = _inputs()

    def compute(net):
        with torch.no_grad():
            net(**values, clean_x=values["x"])
        output = net(**values, training_attention_mode="bidirectional")
        # Change state before backward; replay must still use BID's RoPE/mask.
        with torch.no_grad():
            net(**values, clean_x=values["x"])
        output.square().mean().backward()
        return output

    actual, expected = compute(model), compute(reference)
    torch.testing.assert_close(actual, expected)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            assert param.grad is None
            continue
        baseline = dict(reference.named_parameters())[name]
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        torch.testing.assert_close(param.grad, baseline.grad, rtol=1e-4, atol=2e-7)
    assert model.blocks[0].self_attn.q.lora_b.weight.grad.abs().sum() > 0
    assert model.act_control_adapter.conv.weight.grad.abs().sum() > 0
    assert model.blocks[0].self_attn._is_teacher_forcing is True


@pytest.mark.parametrize("grad_enabled", [True, False])
@pytest.mark.parametrize("invalid", [
    {"clean_x": torch.zeros(1)}, {"aug_t": torch.zeros(1)},
    {"kv_cache": []}, {"crossattn_cache": {}}, {"ref_latents": torch.zeros(1)},
    {"ref_mask": torch.zeros(1)}, {"history_x": torch.zeros(1)},
    {"current_start": 16}, {"cache_start": 1}, {"updating_cache": True},
])
def test_model_bidirectional_rejects_invalid_state_before_any_dispatch(invalid, grad_enabled):
    model = _model()
    with torch.set_grad_enabled(grad_enabled), pytest.raises(ValueError, match="bidirectional"):
        model(**_inputs(), training_attention_mode="bidirectional", **invalid)
    assert model.block_mask is None and not model._block_mask_cache


@pytest.mark.parametrize("grad_enabled", [True, False])
def test_direct_inference_entry_rejects_bid_even_with_gradients(grad_enabled):
    with torch.set_grad_enabled(grad_enabled), pytest.raises(ValueError, match="inference entry"):
        _model()._forward_inference(**_inputs(), training_attention_mode="bidirectional")


def test_bid_exception_restores_prior_attention_state(monkeypatch):
    model, values = _model(), _inputs()
    with torch.no_grad():
        model(**values, clean_x=values["x"])
    mask, state = model.block_mask, _attention_state(model)

    def fail(*args, **kwargs):
        raise RuntimeError("test failure after flags changed")

    monkeypatch.setattr(model.blocks[0], "forward", fail)
    with pytest.raises(RuntimeError, match="test failure"):
        model(**values, training_attention_mode="bidirectional")
    assert model.block_mask is mask and _attention_state(model) == state


def _wrapper(model=None, is_causal=True):
    return SimpleNamespace(
        is_causal=is_causal, uniform_timestep=False, seq_len=16,
        model=_model() if model is None else model,
        _convert_flow_pred_to_x0=lambda flow_pred, xt, timestep: xt - flow_pred,
    )


def _wrapper_values():
    values = _inputs()
    return dict(
        noisy_image_or_video=values["x"].permute(0, 2, 1, 3, 4),
        conditional_dict={"prompt_embeds": values["context"],
                          "act_context": values["act_context"], "act_context_scale": 0.03},
        timestep=values["t"],
    )


def test_wrapper_passes_bid_to_actual_model_without_modifying_supplied_first_frame():
    wrapper, values = _wrapper(), _wrapper_values()
    original_x = values["noisy_image_or_video"].clone()
    original_t = values["timestep"].clone()
    with torch.no_grad():
        flow, _ = WRAPPER_FORWARD(wrapper, **values, training_attention_mode="bidirectional")
        direct = wrapper.model(**_inputs(), training_attention_mode="bidirectional")
    torch.testing.assert_close(flow, direct.permute(0, 2, 1, 3, 4))
    torch.testing.assert_close(values["noisy_image_or_video"], original_x, rtol=0, atol=0)
    torch.testing.assert_close(values["timestep"], original_t, rtol=0, atol=0)


def test_wrapper_default_keeps_legacy_kwargs_identical():
    calls = []

    def record(x, **kwargs):
        calls.append(kwargs)
        return torch.zeros_like(x)

    wrapper, values = _wrapper(record), _wrapper_values()
    WRAPPER_FORWARD(wrapper, **values)
    WRAPPER_FORWARD(wrapper, **values, training_attention_mode="causal")
    assert "training_attention_mode" not in calls[0]
    assert calls[0].keys() == calls[1].keys()
    assert all(calls[0][key] is calls[1][key] or calls[0][key] == calls[1][key] for key in calls[0])


@pytest.mark.parametrize("invalid", [
    {"clean_x": torch.zeros(1)}, {"aug_t": torch.zeros(1)}, {"kv_cache": []},
    {"crossattn_cache": {}}, {"history_x": torch.zeros(1)}, {"history_y": torch.zeros(1)},
    {"history_act_context": torch.zeros(1)}, {"history_y_action": torch.zeros(1)},
    {"current_start": 1}, {"cache_start": 1}, {"updating_cache": True},
    {"noisy_start_frame": 1}, {"classify_mode": True}, {"concat_time_embeddings": True},
    {"replace_first_timestep_and_noise_latents": True},
])
def test_wrapper_bid_rejects_inference_or_clean_state(invalid):
    with pytest.raises(ValueError, match="bidirectional"):
        WRAPPER_FORWARD(_wrapper(), **_wrapper_values(), training_attention_mode="bidirectional", **invalid)


@pytest.mark.parametrize("field", ["ref_latents", "ref_mask"])
def test_wrapper_bid_rejects_reference_condition(field):
    values = _wrapper_values()
    values["conditional_dict"][field] = torch.zeros(1)
    with pytest.raises(ValueError, match="bidirectional"):
        WRAPPER_FORWARD(_wrapper(), **values, training_attention_mode="bidirectional")


def test_invalid_mode_and_noncausal_wrapper_fail_closed():
    with pytest.raises(ValueError, match="unknown training_attention_mode"):
        _model()(**_inputs(), training_attention_mode="typo")
    with pytest.raises(ValueError, match="unknown training_attention_mode"):
        WRAPPER_FORWARD(_wrapper(), **_wrapper_values(), training_attention_mode="typo")
    with pytest.raises(ValueError, match="existing causal model"):
        WRAPPER_FORWARD(_wrapper(is_causal=False), **_wrapper_values(), training_attention_mode="bidirectional")
