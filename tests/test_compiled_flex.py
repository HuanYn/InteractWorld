"""CPU checks of the finite compiled-only policy; no attention/GPU kernels."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import torch._dynamo


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compiled_flex_policy", ROOT / "wan/modules/compiled_flex.py"
)
POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)


def test_compile_policy_is_finite_fullgraph_and_scoped():
    config = torch._dynamo.config
    previous = (config.recompile_limit, config.fail_on_recompile_limit_hit,
                config.suppress_errors, config.accumulated_recompile_limit)
    calls = []

    def compiled(**kwargs):
        calls.append((config.recompile_limit, config.fail_on_recompile_limit_hit,
                      config.suppress_errors, config.accumulated_recompile_limit))
        return kwargs["query"] + 1

    original = Mock()
    with patch.object(torch, "compile", return_value=compiled) as compile_call:
        attention = POLICY.compile_sparse_flex_attention(original)
    compile_call.assert_called_once_with(
        original, dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs"
    )
    torch.testing.assert_close(attention(query=torch.tensor(2.0)), torch.tensor(3.0))
    assert calls == [(32, True, False, previous[3])]
    assert (config.recompile_limit, config.fail_on_recompile_limit_hit,
            config.suppress_errors, config.accumulated_recompile_limit) == previous
    original.assert_not_called()


def test_compiler_failure_propagates_without_eager_retry_and_restores_policy():
    config = torch._dynamo.config
    original = Mock()
    compiled = Mock(side_effect=RuntimeError("compile failed"))
    with patch.object(torch, "compile", return_value=compiled):
        attention = POLICY.compile_sparse_flex_attention(original)
    with config.patch(recompile_limit=3, fail_on_recompile_limit_hit=False, suppress_errors=True):
        with pytest.raises(RuntimeError, match="compile failed"):
            attention(query=torch.tensor(1.0))
        assert config.recompile_limit == 3
        assert config.fail_on_recompile_limit_hit is False
        assert config.suppress_errors is True
    compiled.assert_called_once()
    original.assert_not_called()


def test_smaller_accumulated_budget_fails_without_changing_it():
    compiled = Mock()
    with patch.object(torch, "compile", return_value=compiled):
        attention = POLICY.compile_sparse_flex_attention(Mock())
    with torch._dynamo.config.patch(accumulated_recompile_limit=16):
        with pytest.raises(RuntimeError, match="accumulated_recompile_limit"):
            attention(query=torch.tensor(1.0))
        assert torch._dynamo.config.accumulated_recompile_limit == 16
    compiled.assert_not_called()


def test_real_cpu_dynamo_limit_is_a_hard_failure_not_eager_fallback():
    real_compile = torch.compile
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def cpu_compile(function, **kwargs):
        kwargs.pop("mode")  # A CPU test backend, not a change to production mode.
        return real_compile(function, backend=backend, **kwargs)

    def small_function(x):
        return x.sin() + x

    with patch.object(POLICY, "FLEX_RECOMPILE_LIMIT", 2), patch.object(
        torch, "compile", side_effect=cpu_compile
    ):
        attention = POLICY.compile_sparse_flex_attention(small_function)
        for size in (2, 3):
            value = torch.ones(size, requires_grad=True)
            attention(value).sum().backward()
            torch.testing.assert_close(value.grad, value.detach().cos() + 1)
        with pytest.raises(torch._dynamo.exc.FailOnRecompileLimitHit):
            attention(torch.ones(4, requires_grad=True))
    assert len(graphs) == 2
