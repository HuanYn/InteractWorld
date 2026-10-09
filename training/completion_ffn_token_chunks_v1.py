"""Instance-only, token-exact FFN workspace repair for native packed TF.

The attention calls, mask and clean/noisy history are unchanged.  Only the
pointwise norm2 -> modulation -> FFN -> gate -> residual chain is evaluated in
fixed token slices.  Non-reentrant per-slice checkpointing prevents retaining
all FFN hidden activations and remains compatible with the native outer
non-reentrant checkpoint and autograd.grad VJP probe.
"""
from __future__ import annotations

import hashlib
import inspect
import types

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

VERSION = "completion-ffn-token-chunks-v1"
CHUNK_TOKENS = 390  # One actual 30 x 52 / 2 x 2 spatial latent frame. No sweep.


def _piece(self, value, scale, shift, gate):
    # Keep native LayerNorm/autocast and original LoRA/Linear/GELU arithmetic.
    normed = self.norm2(value)
    modulated = normed * (1 + scale) + shift
    y = self.ffn(modulated)
    observation = self.__dict__.get("_completion_ffn_observation")
    if observation is not None:
        observation["max_chunk_tokens"] = max(observation["max_chunk_tokens"], value.shape[1])
        if observation["first_chunk"] is None:
            # Metadata only: no item(), tensor copy, CUDA synchronization or hook.
            observation["first_chunk"] = {
                "input_shape": list(value.shape), "input_dtype": str(value.dtype),
                "norm2_dtype": str(normed.dtype), "modulated_dtype": str(modulated.dtype),
                "ffn_output_dtype": str(y.dtype), "scale_shape": list(scale.shape),
                "scale_dtype": str(scale.dtype), "device_type": value.device.type,
            }
    return value + y * gate


def pointwise_chunks(block, x, e, *, chunk_tokens=CHUNK_TOKENS):
    """Run the complete second pointwise residual, not just self.ffn.

    ``e`` is the native six-element tuple after modulation addition/chunk.
    Frame-to-token mapping uses integer indices, including non-frame-aligned
    chunk boundaries and the clean/noisy midpoint. No full tokenwise e is built.
    """
    if type(chunk_tokens) is not int or chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be a positive integer")
    if x.ndim != 3 or x.shape[1] <= 0 or len(e) != 6:
        raise ValueError("expected nonempty [B,L,C] and native six modulation tensors")
    frames = e[0].shape[1]
    if frames <= 0 or x.shape[1] % frames:
        raise ValueError("native frame/token split is not divisible")
    frame_tokens = x.shape[1] // frames
    expected = (x.shape[0], frames, 1, x.shape[2])
    if any(tuple(part.shape) != expected for part in e):
        raise ValueError("native modulation shape mismatch")
    pieces = []
    for start in range(0, x.shape[1], chunk_tokens):
        stop = min(start + chunk_tokens, x.shape[1])
        if frame_tokens == 1:
            scale, shift, gate = (e[index][:, start:stop, 0, :] for index in (4, 3, 5))
        elif start // frame_tokens == (stop - 1) // frame_tokens:
            # Production 390 tokens == one frame: keep three tiny broadcast
            # views, NOT 3 x full-token modulation saved by nested checkpoint.
            frame = start // frame_tokens
            scale, shift, gate = (e[index][:, frame:frame + 1, 0, :] for index in (4, 3, 5))
        else:
            frame_indices = torch.arange(start, stop, device=x.device) // frame_tokens
            scale, shift, gate = (
                e[index].squeeze(2).index_select(1, frame_indices) for index in (4, 3, 5)
            )
        value = x[:, start:stop]
        if torch.is_grad_enabled():
            # Explicit tensor inputs retain x/e/modulation/LoRA gradients. Unlike
            # reentrant checkpoint, this supports autograd.grad(x, clean_x).
            output = checkpoint(
                lambda v, s, h, g: _piece(block, v, s, h, g),
                value, scale, shift, gate, use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            output = _piece(block, value, scale, shift, gate)
        pieces.append(output)
    return torch.cat(pieces, dim=1)


def _chunked_forward(
    self, x, e, seq_lens, grid_sizes, freqs, context, context_lens,
    block_mask=None, kv_cache=None, crossattn_cache=None, current_start=0,
    cache_start=None,
):
    original = self.__dict__["_completion_ffn_original_forward"]
    chunk_tokens = self.__dict__["_completion_ffn_chunk_tokens"]
    # Preserve rollout/KV/non-TF and small-sequence implementation verbatim.
    targeted = (
        block_mask is not None and kv_cache is None
        and bool(getattr(self.self_attn, "_is_teacher_forcing", False))
        and x.shape[1] > chunk_tokens
    )
    if not targeted:
        return original(
            x, e, seq_lens, grid_sizes, freqs, context, context_lens,
            block_mask=block_mask, kv_cache=kv_cache,
            crossattn_cache=crossattn_cache, current_start=current_start,
            cache_start=cache_start,
        )
    frames = e.shape[1]
    if frames <= 0 or x.shape[1] % frames:
        raise RuntimeError(f"x token length {x.shape[1]} is not divisible by e frames {frames}")
    frame_tokens = x.shape[1] // frames
    token_level = frames == x.shape[1]
    e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)

    def gate(value, modulation):
        if token_level:
            return value * modulation.squeeze(2)
        return (value.unflatten(1, (frames, frame_tokens)) * modulation).flatten(1, 2)

    normed = self.norm1(x)
    if token_level:
        normalized = normed * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
    else:
        normalized = (
            normed.unflatten(1, (frames, frame_tokens)) * (1 + e[1]) + e[0]
        ).flatten(1, 2)
    del normed
    y = self.self_attn(
        normalized, seq_lens, grid_sizes, freqs, block_mask=block_mask,
        kv_cache=kv_cache, current_start=current_start, cache_start=cache_start,
    )
    del normalized
    x = x + gate(y, e[2])
    del y
    x = x + self.cross_attn(
        self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache,
    )
    return pointwise_chunks(self, x, e, chunk_tokens=chunk_tokens)


def install_scoped(model, *, chunk_tokens=CHUNK_TOKENS):
    """Install on one constructed native model; no class/parameter edits.

    Dropout/stochastic blocks are rejected; the actual DMD worker uses eval().
    Installation does not change training flags, state_dict, RNG, dtype or masks.
    """
    if chunk_tokens != CHUNK_TOKENS:
        raise ValueError("production repair uses exactly the fixed 390-token chunk")
    target = getattr(model, "model", model)
    blocks = getattr(target, "blocks", None)
    if not isinstance(blocks, (nn.ModuleList, list, tuple)) or not blocks:
        raise TypeError("expected native causal model blocks")
    checked = []
    original_sha = set()
    for block in blocks:
        if type(block).__name__ != "CausalWanAttentionBlock":
            raise TypeError("unexpected native attention block class")
        if block.training:
            raise ValueError("scoped FFN repair requires the native eval-mode worker")
        existing = block.__dict__.get("_completion_ffn_chunk_version")
        if existing is not None:
            if existing != VERSION or block.__dict__.get("_completion_ffn_chunk_tokens") != chunk_tokens:
                raise RuntimeError("conflicting instance FFN override")
            if getattr(block.forward, "__func__", None) is not _chunked_forward:
                raise RuntimeError("instance FFN override was replaced")
            continue
        if "forward" in block.__dict__:
            raise RuntimeError("refusing to overwrite an existing instance block forward")
        if not isinstance(block.norm2, nn.LayerNorm):
            raise TypeError("expected original tokenwise LayerNorm")
        if not isinstance(block.ffn, nn.Sequential) or len(block.ffn) != 3:
            raise TypeError("expected original two-linear GELU FFN")
        if not isinstance(block.ffn[1], nn.GELU) or block.ffn[1].approximate != "tanh":
            raise TypeError("expected native tanh GELU")
        for layer in (block.ffn[0], block.ffn[2]):
            if not isinstance(layer, nn.Linear) and type(layer).__name__ != "LoRALinear":
                raise TypeError("unexpected FFN linear/LoRA layer")
        if any(isinstance(layer, nn.Dropout) and layer.p != 0 for layer in block.ffn.modules()):
            raise ValueError("nonzero dropout is outside this exact repair")
        original = type(block).forward
        if tuple(inspect.signature(original).parameters) != tuple(inspect.signature(_chunked_forward).parameters):
            raise TypeError("native forward signature changed")
        original_sha.add(hashlib.sha256(inspect.getsource(original).encode()).hexdigest())
        checked.append(block)
    # Validate all instances before mutating any.
    for block in checked:
        block._completion_ffn_original_forward = block.forward
        block._completion_ffn_chunk_tokens = chunk_tokens
        block._completion_ffn_chunk_version = VERSION
        block._completion_ffn_observation = {"max_chunk_tokens": 0, "first_chunk": None}
        block.forward = types.MethodType(_chunked_forward, block)
    return {
        "status": "installed", "version": VERSION, "scope": "model_instance_only",
        "blocks": len(blocks), "new_blocks": len(checked), "chunk_tokens": chunk_tokens,
        "original_forward_source_sha256": sorted(original_sha),
        "target": "teacher_forcing_no_kv_long_tokens_only",
        "chain": "norm2_modulation_ffn_gate_residual",
        "nested_checkpoint": "non_reentrant_preserve_rng_state",
        "attention_mask_history_objective_unchanged": True,
        "actual_gpu_vjp_resource_gate_passed": False,
    }


def observations(model):
    """Small actual dtype/shape receipts; null until this instance executes."""
    target = getattr(model, "model", model)
    return [dict(index=index, **block.__dict__["_completion_ffn_observation"])
            for index, block in enumerate(target.blocks)
            if "_completion_ffn_observation" in block.__dict__]
