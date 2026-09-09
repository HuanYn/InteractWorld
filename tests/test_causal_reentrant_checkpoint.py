"""Small CPU gradient checks of the actual Wan checkpoint helper."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
import torch.utils.checkpoint

from training.longforcing_lite import load_longforcing_config
from training.models.lora import LoRALinear


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "wan" / "modules" / "causal_model.py"


def _load_helper():
    # Avoid importing CUDA-only Wan/T5 dependencies in a CPU regression test.
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(item for item in tree.body
               if isinstance(item, ast.ClassDef) and item.name == "CausalWanModel")
    method = next(item for item in cls.body
                  if isinstance(item, ast.FunctionDef)
                  and item.name == "_checkpoint_train_block")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[method.name]


CHECKPOINT_BLOCK = _load_helper()


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q = LoRALinear(nn.Linear(4, 4), rank=2, alpha=2.0)
        # Nonzero B also lets the test check the A gradient, not only B.
        nn.init.normal_(self.self_attn.q.lora_b.weight, std=0.05)
        self.self_attn._is_teacher_forcing = False
        self.self_attn._num_ref_tokens = 0

    def forward(self, x, *, e, context, block_mask, current_start):
        factor = (2 if self.self_attn._is_teacher_forcing else 1)
        factor += 0.1 * self.self_attn._num_ref_tokens
        return torch.tanh(self.self_attn.q(x) + e + context) * factor * block_mask + current_start


def _assert_parameter_gradients_equal(actual, expected):
    for name, parameter in actual.named_parameters():
        other = dict(expected.named_parameters())[name]
        if not parameter.requires_grad:
            assert parameter.grad is None and other.grad is None
            continue
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-5, atol=1e-7)


def test_frozen_first_input_still_trains_lora_without_mutating_input():
    torch.manual_seed(12)
    actual = Block()
    expected = copy.deepcopy(actual)
    x = torch.randn(2, 4)
    kwargs = dict(e=torch.randn(2, 4), context=torch.randn(2, 4),
                  block_mask=torch.tensor(0.7), current_start=0.0)
    result = CHECKPOINT_BLOCK(SimpleNamespace(gradient_checkpointing_mode="reentrant"),
                              actual, x, **kwargs)
    baseline = expected(x, **kwargs)
    result.square().mean().backward()
    baseline.square().mean().backward()
    assert x.requires_grad is False
    assert result.requires_grad
    assert actual.self_attn.q.lora_b.weight.grad.abs().sum() > 0
    _assert_parameter_gradients_equal(actual, expected)


def test_four_step_chain_preserves_adapter_kwargs_and_each_block_state():
    torch.manual_seed(13)
    blocks = nn.ModuleList([Block(), Block()])
    reference_blocks = copy.deepcopy(blocks)
    adapter = nn.Linear(4, 4)
    reference_adapter = copy.deepcopy(adapter)
    embedding = nn.Parameter(torch.randn(2, 4))
    reference_embedding = nn.Parameter(embedding.detach().clone())
    noise = torch.randn(2, 4)
    action = torch.randn(2, 4)

    def solve(models, action_adapter, e, checkpoint):
        state = noise.clone()
        for step in range(4):
            value = state + action_adapter(action)
            for index, block in enumerate(models):
                block.self_attn._is_teacher_forcing = bool((step + index) % 2)
                block.self_attn._num_ref_tokens = step + index
                kwargs = dict(e=e * (step + 1), context=state * 0.2,
                              block_mask=torch.tensor(0.4 + index * 0.1),
                              current_start=step * 0.03)
                if checkpoint:
                    value = CHECKPOINT_BLOCK(
                        SimpleNamespace(gradient_checkpointing_mode="reentrant"),
                        block, value, **kwargs)
                    # Must not be read through a shared dict during backward.
                    kwargs["current_start"] = 500.0
                else:
                    value = block(value, **kwargs)
            state = state - 0.25 * value
        for block in models:
            block.self_attn._is_teacher_forcing = "later forward"
            block.self_attn._num_ref_tokens = 123
        return state

    actual = solve(blocks, adapter, embedding, True)
    expected = solve(reference_blocks, reference_adapter, reference_embedding, False)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    _assert_parameter_gradients_equal(blocks, reference_blocks)
    _assert_parameter_gradients_equal(adapter, reference_adapter)
    assert adapter.weight.grad.abs().sum() > 0
    torch.testing.assert_close(embedding.grad, reference_embedding.grad, rtol=1e-5, atol=1e-7)
    for block in blocks:
        assert block.self_attn._is_teacher_forcing == "later forward"
        assert block.self_attn._num_ref_tokens == 123
        assert not hasattr(block.self_attn, "_ref_grid_sizes")


def test_reentrant_restores_state_when_block_raises():
    class FailingBlock(Block):
        def forward(self, *args, **kwargs):
            self.self_attn._is_teacher_forcing = "corrupted"
            self.self_attn._ref_grid_sizes = torch.ones(1)
            raise RuntimeError("deliberate test failure")

    block = FailingBlock()
    with pytest.raises(RuntimeError, match="deliberate"):
        CHECKPOINT_BLOCK(SimpleNamespace(gradient_checkpointing_mode="reentrant"),
                         block, torch.randn(2, 4))
    assert block.self_attn._is_teacher_forcing is False
    assert not hasattr(block.self_attn, "_ref_grid_sizes")


def test_default_keeps_non_reentrant_and_invalid_mode_is_rejected():
    block = Block()
    kwargs = dict(e=torch.zeros(2, 4), context=torch.zeros(2, 4),
                  block_mask=torch.tensor(1.0), current_start=0.0)
    with patch.object(torch.utils.checkpoint, "checkpoint",
                      wraps=torch.utils.checkpoint.checkpoint) as checkpoint:
        CHECKPOINT_BLOCK(SimpleNamespace(), block, torch.ones(2, 4), **kwargs).sum().backward()
    assert checkpoint.call_args.kwargs["use_reentrant"] is False
    assert "determinism_check" not in checkpoint.call_args.kwargs
    with pytest.raises(ValueError, match="unknown gradient_checkpointing_mode"):
        CHECKPOINT_BLOCK(SimpleNamespace(gradient_checkpointing_mode="typo"),
                         block, torch.ones(2, 4), **kwargs)


def test_only_week_longforcing_config_opts_in_and_invalid_config_fails():
    original = load_longforcing_config(ROOT / "configs/train/longforcing_lite_v1.yaml")
    week = load_longforcing_config(ROOT / "configs/train/longforcing_lite_5090_week.yaml")
    assert original.model.gradient_checkpointing_mode == "non_reentrant"
    assert week.model.gradient_checkpointing_mode == "reentrant"
    week.model.gradient_checkpointing_mode = "typo"
    with pytest.raises(ValueError, match="gradient_checkpointing_mode"):
        week.validate()
