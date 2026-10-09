from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask

from training.completion_tf_mask_memory_v1 import (
    build_components, install_scoped, prepare_mask,
)

SOURCE = Path(__file__).parents[1] / "wan/modules/causal_model.py"


def original_class():
    """Execute the actual original methods; only capture create_block_mask."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CausalWanModel")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {
        "_frame_block_token_ranges", "_prepare_teacher_forcing_mask"}]
    node = ast.ClassDef(name="CausalWanModel", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    ns = {"torch": torch, "math": math, "BlockMask": BlockMask,
          "create_block_mask": lambda fn, **kw: (fn, kw)}
    exec(compile(module, str(SOURCE), "exec"), ns)
    return ns["CausalWanModel"]


Original = original_class()


@pytest.mark.parametrize("frames,f,block,first", [
    (n, f, b, first) for n, f in [(1, 1), (2, 3), (4, 17), (5, 127),
                                (4, 128), (5, 129), (2, 390)]
    for b in (1, 3) for first in (False, True)])
def test_exhaustive_native_predicate_and_sparse_coverage(frames, f, block, first):
    visible, pred, info = build_components("cpu", frames, f, block, first)
    oracle, kwargs = Original._prepare_teacher_forcing_mask("cpu", frames, f, block, first)
    size = info["padded_tokens"]
    assert kwargs["Q_LEN"] == size
    k = torch.arange(size)[None, :]
    # Bounded strips also prevent quadratic test memory at larger shapes.
    for start in range(0, size, 128):
        q = torch.arange(start, min(start + 128, size))[:, None]
        expected = oracle(0, 0, q, k)
        actual = pred(0, 0, q, k)
        assert torch.equal(actual, expected)
        assert torch.equal(actual & visible[q // 128, k // 128], expected)
    if info["packed_tokens"] < size:
        q = torch.arange(info["packed_tokens"], size)[:, None]
        assert torch.equal(pred(0, 0, q, k), q == k)


def test_actual_61_latent_geometry_without_dense_token_square():
    visible, pred, info = build_components("cpu", 61, 390, 3, True)
    oracle, _ = Original._prepare_teacher_forcing_mask("cpu", 61, 390, 3, True)
    assert info["padded_tokens"] == 47616
    assert visible.shape == (372, 372)
    assert info["visibility_bytes"] == 138384
    # Every possible query block boundary, plus all frame/clean/noisy edges.
    edges = set(range(0, 47616, 128)) | {0, 390, 23790, 47580, 47616}
    edges |= {offset + frame * 390 for offset in (0, 23790)
              for frame in range(1, 62, 3)}
    points = sorted({x + delta for x in edges for delta in (-1, 0, 1)
                     if 0 <= x + delta < 47616})
    k = torch.tensor(points)[None, :]
    for start in range(0, len(points), 64):
        q = torch.tensor(points[start:start + 64])[:, None]
        expected = oracle(0, 0, q, k)
        assert torch.equal(pred(0, 0, q, k), expected)
        assert torch.equal(expected & visible[q // 128, k // 128], expected)
    mask = prepare_mask("cpu", 61, 390, 3, True)
    assert mask.kv_num_blocks.shape[-1] == 372
    assert mask.full_kv_num_blocks is None and mask.full_kv_indices is None
    assert torch.equal(mask.to_dense()[0, 0].bool(), visible)


def test_instance_scoped_without_class_or_rng_mutation():
    a, b = Original(), Original()
    original = Original._prepare_teacher_forcing_mask
    a.block_mask = object()
    a._block_mask_cache = {"old": object()}
    rng = torch.get_rng_state().clone()
    receipt = install_scoped(SimpleNamespace(model=a))
    assert receipt["status"] == "installed"
    assert Original._prepare_teacher_forcing_mask is original
    assert b._prepare_teacher_forcing_mask is original
    assert a._prepare_teacher_forcing_mask is prepare_mask
    assert a.block_mask is None and a._block_mask_cache == {}
    assert torch.equal(rng, torch.get_rng_state())
    assert install_scoped(a)["already_installed"]


def test_reject_conflicting_instance_override():
    a = Original()
    a._prepare_teacher_forcing_mask = lambda *a, **kw: None
    with pytest.raises(RuntimeError, match="existing instance"):
        install_scoped(a)
    with pytest.raises(TypeError):
        install_scoped(object())


@pytest.mark.parametrize("args", [(0, 390, 3, True), (61, 0, 3, True),
                                   (61, 390, 0, True), (61, 390, 3, 1),
                                   (True, 390, 3, True)])
def test_invalid_geometry_fails_closed(args):
    with pytest.raises(ValueError):
        build_components("cpu", *args)
