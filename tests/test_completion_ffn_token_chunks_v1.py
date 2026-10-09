"""CPU semantic tests; these are not the actual full-Wan CUDA resource gate."""
import ast
import copy
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from training.completion_ffn_token_chunks_v1 import (
    CHUNK_TOKENS, install_scoped, pointwise_chunks, observations,
)
from training.models.lora import LoRALinear


torch.set_num_threads(1)
SOURCE = Path(__file__).parents[1] / "wan/modules/causal_model.py"


def original_forward():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CausalWanAttentionBlock")
    fn = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"))
    for arg in fn.args.args:
        arg.annotation = None
    namespace = {"_dbg_print": lambda *a, **k: None, "_dbg_block_mask": lambda x: x,
                 "_is_checkpoint_stop_signal": lambda error: type(error).__name__ == "_StopRecomputationError"}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["forward"]


class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Linear(dim, dim)
        self._is_teacher_forcing = True
        self.calls = []

    def forward(self, x, seq_lens, grid_sizes, freqs, **kwargs):
        self.calls.append(kwargs)
        return self.projection(x)


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Linear(dim, dim)

    def forward(self, x, context, context_lens, crossattn_cache=None):
        return self.projection(x + context.mean(1, keepdim=True))


class PromotedLayerNorm(nn.LayerNorm):
    """Explicit fixture for the inferred possible full FP32 norm workspace."""
    def forward(self, x):
        return super().forward(x.float())


def make_block(dim=8, *, lora=True, promoted_norm=False):
    cls = type("CausalWanAttentionBlock", (nn.Module,), {"forward": original_forward()})
    block = cls()
    block.norm1 = nn.LayerNorm(dim)
    block.norm2 = (PromotedLayerNorm if promoted_norm else nn.LayerNorm)(dim)
    block.norm3 = nn.LayerNorm(dim)
    block.self_attn = SelfAttention(dim)
    block.cross_attn = CrossAttention(dim)
    first, last = nn.Linear(dim, dim * 4), nn.Linear(dim * 4, dim)
    if lora:
        first, last = (LoRALinear(layer, rank=3, alpha=3, dropout=0) for layer in (first, last))
        for layer in (first, last):
            nn.init.normal_(layer.lora_b.weight, std=.03)
    block.ffn = nn.Sequential(first, nn.GELU(approximate="tanh"), last)
    block.modulation = nn.Parameter(torch.randn(1, 6, dim) * .03)
    return block.eval()


def patch(block):
    model = nn.Module()
    model.blocks = nn.ModuleList([block])
    model.eval()
    return install_scoped(model)


def execute(block, x, e, context, *, outer_checkpoint=False, autocast=False, mask=True, kv=None):
    args = dict(e=e, seq_lens=None, grid_sizes=None, freqs=None, context=context,
                context_lens=None, block_mask=object() if mask else None, kv_cache=kv)
    def call(a, b, c):
        with torch.autocast("cpu", dtype=torch.bfloat16) if autocast else nullcontext():
            return block(a, **dict(args, e=b, context=c))
    # The actual VJP uses save_on_cpu around a native outer non-reentrant block.
    with torch.autograd.graph.save_on_cpu(pin_memory=False):
        output = checkpoint(call, x, e, context, use_reentrant=False) if outer_checkpoint else call(x, e, context)
        params = [(n, p) for n, p in block.named_parameters() if p.requires_grad]
        coeff = torch.linspace(.2, 1.2, output.numel()).reshape_as(output)
        grads = torch.autograd.grad((output.float() * coeff).sum(), [x, e, context] + [p for _, p in params])
    return output, dict(zip(["x", "e", "context"] + [n for n, _ in params], grads))


@pytest.mark.parametrize("token_level", [False, True])
@pytest.mark.parametrize("outer_checkpoint", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("promoted_norm", [False, True])
def test_native_forward_vjp_lora_and_modulation_equal(token_level, outer_checkpoint, autocast, promoted_norm):
    torch.manual_seed(17)
    a = make_block(promoted_norm=promoted_norm)
    b = copy.deepcopy(a)
    receipt = patch(b)
    assert receipt["chunk_tokens"] == 390 and not receipt["actual_gpu_vjp_resource_gate_passed"]
    length, frames = 488, (488 if token_level else 122)
    dtype = torch.bfloat16 if autocast else torch.float32
    values = [torch.randn(1, length, 8, dtype=dtype), torch.randn(1, frames, 6, 8) * .04,
              torch.randn(1, 5, 8, dtype=dtype)]
    left = [v.clone().requires_grad_() for v in values]
    right = [v.clone().requires_grad_() for v in values]
    ya, ga = execute(a, *left, outer_checkpoint=outer_checkpoint, autocast=autocast)
    yb, gb = execute(b, *right, outer_checkpoint=outer_checkpoint, autocast=autocast)
    assert ya.dtype == yb.dtype
    tolerance = dict(rtol=.025, atol=.025) if autocast else dict(rtol=1e-5, atol=3e-5)
    torch.testing.assert_close(ya, yb, **tolerance)
    for name in ga:
        torch.testing.assert_close(ga[name], gb[name], **tolerance)
        assert torch.isfinite(gb[name]).all(), name
    for name in ("ffn.0.lora_a.weight", "ffn.0.lora_b.weight", "ffn.2.lora_a.weight", "ffn.2.lora_b.weight"):
        assert gb[name].abs().max() > 0


def test_actual_122_times_390_tokens_and_clean_noisy_midpoint():
    torch.manual_seed(23)
    a = make_block(dim=4, lora=False)
    b = copy.deepcopy(a)
    patch(b)
    norm_sizes, ffn_sizes = [], []
    b.norm2.register_forward_pre_hook(lambda module, args: norm_sizes.append(args[0].shape[1]))
    b.ffn.register_forward_pre_hook(lambda module, args: ffn_sizes.append(args[0].shape[1]))
    values = [torch.randn(1, 47580, 4), torch.randn(1, 122, 6, 4) * .03, torch.randn(1, 2, 4)]
    ya, ga = execute(a, *[v.clone().requires_grad_() for v in values], outer_checkpoint=True)
    yb, gb = execute(b, *[v.clone().requires_grad_() for v in values], outer_checkpoint=True)
    torch.testing.assert_close(ya, yb, rtol=1e-5, atol=2e-5)
    for key in ("x", "e", "context", "modulation"):
        torch.testing.assert_close(ga[key], gb[key], rtol=2e-5, atol=.002)
    assert norm_sizes and max(norm_sizes) == 390
    assert ffn_sizes and max(ffn_sizes) == 390
    assert ya.shape[1] == 122 * 390 and ya[:, 23790:].shape[1] == 61 * 390
    assert b._completion_ffn_observation["first_chunk"]["scale_shape"] == [1, 1, 4]
    assert b._completion_ffn_observation["max_chunk_tokens"] == 390


def test_actual_bf16_base_with_fp32_lora_banks_under_autocast():
    torch.manual_seed(71)
    a=make_block(promoted_norm=True)
    for layer in (a.ffn[0],a.ffn[2]):
        layer.base.to(dtype=torch.bfloat16)
        assert layer.lora_a.weight.dtype==layer.lora_b.weight.dtype==torch.float32
    b=copy.deepcopy(a);patch(b)
    values=[torch.randn(1,488,8,dtype=torch.bfloat16),torch.randn(1,122,6,8)*.02,
            torch.randn(1,3,8,dtype=torch.bfloat16)]
    ya,ga=execute(a,*[v.clone().requires_grad_() for v in values],outer_checkpoint=True,autocast=True)
    yb,gb=execute(b,*[v.clone().requires_grad_() for v in values],outer_checkpoint=True,autocast=True)
    torch.testing.assert_close(ya,yb,rtol=.025,atol=.025)
    for name in ga:
        torch.testing.assert_close(ga[name],gb[name],rtol=.025,atol=.025)
    for name in ('ffn.0.lora_a.weight','ffn.0.lora_b.weight','ffn.2.lora_a.weight','ffn.2.lora_b.weight'):
        assert gb[name].dtype==torch.float32 and gb[name].abs().max()>0


def test_non_aligned_chunk_tail_and_checkpoint_saved_hidden_not_accumulated():
    torch.manual_seed(5)
    block = make_block(dim=8)
    x = torch.randn(1, 27, 8, requires_grad=True)
    raw = torch.randn(1, 9, 6, 8, requires_grad=True)
    e = (block.modulation.unsqueeze(0) + raw).chunk(6, dim=2)
    mapped = lambda part: part.expand(-1, -1, 3, -1).flatten(1, 2)
    reference = x + block.ffn(block.norm2(x) * (1 + mapped(e[4])) + mapped(e[3])) * mapped(e[5])
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(tuple(t.shape)) or t, lambda t: t):
        actual = pointwise_chunks(block, x, e, chunk_tokens=4)
    torch.testing.assert_close(reference, actual)
    assert not any(len(shape) == 3 and shape[-1] == 32 for shape in saved)
    ga = torch.autograd.grad(reference.sum(), (x, raw), retain_graph=True)
    gb = torch.autograd.grad(actual.sum(), (x, raw))
    for aa, bb in zip(ga, gb):
        torch.testing.assert_close(aa, bb)


def test_aligned_real_geometry_checkpoint_does_not_save_expanded_modulation():
    block = make_block(dim=4)
    x = torch.randn(1, 47580, 4, requires_grad=True)
    raw = torch.randn(1, 122, 6, 4, requires_grad=True)
    e = (block.modulation.unsqueeze(0) + raw).chunk(6, dim=2)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(tuple(t.shape)) or t, lambda t: t):
        result = pointwise_chunks(block, x, e)
    # Per chunk: one x slice plus three [B,1,C] modulation views, no expanded e.
    assert saved.count((1, 390, 4)) == 122
    assert saved.count((1, 1, 4)) == 122 * 3
    assert not any(len(shape) == 3 and shape[-1] == 16 for shape in saved)
    torch.autograd.grad(result.sum(), (x, raw))


@pytest.mark.parametrize("mask,tf,kv,length", [(False, True, None, 488), (True, False, None, 488),
                                               (True, True, {}, 488), (True, True, None, 5)])
def test_non_target_native_paths_delegate_verbatim(mask, tf, kv, length):
    block = make_block()
    patch(block)
    block.self_attn._is_teacher_forcing = tf
    sizes = []
    block.norm2.register_forward_pre_hook(lambda module, args: sizes.append(args[0].shape[1]))
    execute(block, torch.randn(1, length, 8, requires_grad=True),
            torch.randn(1, 1, 6, 8, requires_grad=True), torch.randn(1, 2, 8, requires_grad=True),
            mask=mask, kv=kv)
    assert sizes == [length]


def test_install_keeps_state_parameter_ids_rng_signature_and_class():
    block = make_block()
    before = {n: p.clone() for n, p in block.state_dict().items()}
    ids = {n: id(p) for n, p in block.named_parameters()}
    original = type(block).forward
    rng = torch.get_rng_state().clone()
    patch(block)
    assert original is type(block).forward
    assert ids == {n: id(p) for n, p in block.named_parameters()}
    assert set(before) == set(block.state_dict())
    for name, value in before.items():
        torch.testing.assert_close(value, block.state_dict()[name], rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert patch(block)["new_blocks"] == 0


def test_fail_closed_stochastic_training_or_shape():
    block = make_block()
    block.ffn[0].dropout.p = .1
    with pytest.raises(ValueError, match="dropout"):
        patch(block)
    block = make_block().train()
    model = nn.Module()
    model.blocks = nn.ModuleList([block])
    with pytest.raises(ValueError, match="eval-mode"):
        install_scoped(model)
    block = make_block()
    x = torch.randn(1, 10, 8)
    e = tuple(torch.randn(1, 3, 1, 8) for _ in range(6))
    with pytest.raises(ValueError, match="divisible"):
        pointwise_chunks(block, x, e)
