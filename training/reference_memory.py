"""Parameter-free primitives used by the opt-in bidirectional teacher path.

Q/K/V use Wan's [batch, tokens, heads, head_dim] layout. Reference outputs
depend only on the supplied reference Q/K/V; callers must construct those
without video/action conditioning. In particular, timestep-zero modulation
must come from the actual model, not a fabricated zero vector. This module
does not detect people, crop images, encode a VAE, or cache learned features.
Integration in WanModel is local/opt-in; its presence does not establish GPU
validation, deployed reference data, or a qualified action teacher.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import NamedTuple

import torch
from torch.nn import functional as F


class ReferenceAttentionOutput(NamedTuple):
    video: torch.Tensor
    reference: torch.Tensor


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate_qkv(q, k, v, name):
    values = (q, k, v)
    _require(all(torch.is_tensor(value) for value in values), f"{name} Q/K/V must be tensors")
    _require(q.ndim == 4 and all(value.shape == q.shape for value in values),
             f"{name} Q/K/V must have the same [B,L,H,D] shape")
    _require(q.shape[0] > 0 and q.shape[2] > 0 and q.shape[3] > 0, f"{name} B/H/D must be positive")
    _require(q.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
             and all(value.dtype == q.dtype for value in values), f"{name} Q/K/V floating dtypes must agree")
    _require(q.device.type in ("cpu", "cuda") and all(value.device == q.device for value in values),
             f"{name} Q/K/V must share a CPU or CUDA device")
    _require(all(value.layout == torch.strided for value in values), f"{name} Q/K/V must be dense tensors")


def _sdpa(q, k, v, scale):
    # No dense attention mask and no optional/custom kernel imports. PyTorch
    # chooses its available SDPA implementation; this does not guarantee its
    # fused CUDA kernel or a particular memory bound on an untested device.
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        dropout_p=0.0, is_causal=False, scale=scale,
    ).transpose(1, 2).contiguous()


def _zero_reference(q, k, v):
    # Empty reductions link zero gradients without touching invalid values:
    # masked references may contain nonzero projection biases or even NaNs.
    empty = q[:, :0].sum() + k[:, :0].sum() + v[:, :0].sum()
    return torch.zeros_like(q) + empty


def split_reference_attention(q_video, k_video, v_video, *, q_ref=None, k_ref=None,
                              v_ref=None, ref_valid=None, scale=None):
    """Full video attention with strictly one-way access to reference memory.

    Ref queries see only valid ref keys; video queries see valid ref keys and
    *all* video keys. ``ref_valid`` is an explicit bool [B,R] tensor on the QKV
    device whenever reference QKV are supplied. It masks both reference queries
    and keys: invalid reference outputs are exactly zero, including after biased
    projections. Valid tokens are selected per example, never zero-filled into
    the softmax. No [L,L] mask is allocated.

    Reference triples may all be None, or have R=0. If every reference is invalid,
    video attention uses the original single batched SDPA call, preserving its
    output and Q/K/V gradients. Reference gradients are zero when their zero
    outputs are used in a loss (otherwise unused inputs may have grad=None).
    Dropout is deliberately disabled for reproducible static-memory semantics.
    """
    _validate_qkv(q_video, k_video, v_video, "video")
    _require(q_video.shape[1] > 0, "video must have at least one token")
    _require(scale is None or (isinstance(scale, Real) and not isinstance(scale, bool)
                              and math.isfinite(scale) and scale > 0), "scale must be None or positive finite")
    supplied = [value is not None for value in (q_ref, k_ref, v_ref)]
    if not any(supplied):
        _require(ref_valid is None, "ref_valid requires the complete reference Q/K/V triple")
        return ReferenceAttentionOutput(_sdpa(q_video, k_video, v_video, scale), q_video.new_zeros(
            q_video.shape[0], 0, q_video.shape[2], q_video.shape[3]))
    _require(all(supplied), "reference Q/K/V must be supplied together")
    _validate_qkv(q_ref, k_ref, v_ref, "reference")
    _require((q_ref.shape[0], *q_ref.shape[2:]) == (q_video.shape[0], *q_video.shape[2:])
             and q_ref.dtype == q_video.dtype and q_ref.device == q_video.device,
             "reference and video B/H/D, dtype and device must agree")
    _require(torch.is_tensor(ref_valid) and ref_valid.dtype == torch.bool
             and tuple(ref_valid.shape) == tuple(q_ref.shape[:2]) and ref_valid.device == q_ref.device
             and ref_valid.layout == torch.strided, "ref_valid must be dense bool [B,R] on the QKV device")
    if not bool(ref_valid.any()):
        return ReferenceAttentionOutput(_sdpa(q_video, k_video, v_video, scale),
                                        _zero_reference(q_ref, k_ref, v_ref))
    videos, references = [], []
    for sample in range(q_video.shape[0]):
        valid = torch.nonzero(ref_valid[sample], as_tuple=False).flatten()
        qr, kr, vr = [value[sample:sample + 1] for value in (q_ref, k_ref, v_ref)]
        qv, kv, vv = [value[sample:sample + 1] for value in (q_video, k_video, v_video)]
        if valid.numel() == 0:
            videos.append(_sdpa(qv, kv, vv, scale))
            references.append(_zero_reference(qr, kr, vr))
            continue
        valid_q, valid_k, valid_v = [value.index_select(1, valid) for value in (qr, kr, vr)]
        ref_output = _sdpa(valid_q, valid_k, valid_v, scale)
        references.append(torch.zeros_like(qr).index_copy(1, valid, ref_output))
        videos.append(_sdpa(qv, torch.cat((valid_k, kv), dim=1), torch.cat((valid_v, vv), dim=1), scale))
    return ReferenceAttentionOutput(torch.cat(videos, dim=0), torch.cat(references, dim=0))


def _integer(value, name, minimum):
    _require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def reference_temporal_positions(num_slots, tokens_per_slot, *, device="cpu"):
    """Match the existing causal reference convention: -K*s, ..., -s."""
    _integer(num_slots, "num_slots", 0)
    _integer(tokens_per_slot, "tokens_per_slot", 1)
    device = torch.device(device)
    _require(device.type in ("cpu", "cuda"), "reference positions require a CPU or CUDA device")
    step = max(tokens_per_slot, 256)
    return -(num_slots - torch.arange(num_slots, device=device, dtype=torch.int64)) * step


def build_reference_rope_frequencies(freqs, *, num_slots, tokens_per_slot, ref_grid, device=None):
    """Negative-time RoPE directly comparable to causal_model._build_ref_freqs.

    ``freqs`` is the original model's complex table, not an embedding to learn.
    Each slot has one negative temporal coordinate; its spatial bands come
    from the original table. Returns complex64 [K*tokens_per_slot,1,D/2]. No
    global cache is used, so a changed table cannot reuse stale frequencies.
    This matches the existing convention for T>1 too: all temporal patches in
    a reference slot share that slot's timestamp. Image references use T=1.
    """
    _integer(num_slots, "num_slots", 0)
    _integer(tokens_per_slot, "tokens_per_slot", 1)
    _require(torch.is_tensor(freqs) and freqs.ndim == 2 and freqs.shape[0] > 0 and freqs.shape[1] > 0
             and freqs.dtype in (torch.complex64, torch.complex128) and not freqs.requires_grad
             and freqs.layout == torch.strided and freqs.device.type in ("cpu", "cuda"),
             "freqs must be a fixed dense complex [positions,D/2] CPU/CUDA table")
    _require(isinstance(ref_grid, (tuple, list)) and len(ref_grid) == 3,
             "ref_grid must be the three-dimensional patch grid")
    t, h, w = [_integer(value, "ref_grid dimension", 1) for value in ref_grid]
    _require(t * h * w == tokens_per_slot, "tokens_per_slot differs from ref_grid product")
    _require(max(h, w) <= freqs.shape[0], "reference spatial grid exceeds the original RoPE table")
    device = freqs.device if device is None else torch.device(device)
    _require(device.type in ("cpu", "cuda"), "reference frequencies require a CPU or CUDA device")
    bands = freqs.shape[1]
    if num_slots == 0:
        return torch.empty((0, 1, bands), dtype=torch.complex64, device=device)
    f_band, h_band, w_band = bands - 2 * (bands // 3), bands // 3, bands // 3
    positions = reference_temporal_positions(num_slots, tokens_per_slot, device=device).to(torch.float64)
    inverse = 1.0 / torch.pow(10000, torch.arange(0, 2 * f_band, 2, dtype=torch.float64, device=device) / (2 * f_band))
    temporal = torch.outer(positions, inverse)
    temporal = torch.polar(torch.ones_like(temporal), temporal)
    _, height, width = freqs.split([f_band, h_band, w_band], dim=1)
    result = torch.cat((
        temporal[:, None, None, None, :].expand(num_slots, t, h, w, f_band),
        height[:h].to(device)[None, None, :, None, :].expand(num_slots, t, h, w, h_band),
        width[:w].to(device)[None, None, None, :, :].expand(num_slots, t, h, w, w_band),
    ), dim=-1)
    return result.reshape(num_slots * tokens_per_slot, 1, bands).to(torch.complex64)


def apply_reference_rope(x, frequencies):
    """Apply reference RoPE in FP32, preserving Q/K dtype and autograd."""
    _require(torch.is_tensor(x) and x.ndim == 4 and x.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
             and x.device.type in ("cpu", "cuda") and x.layout == torch.strided
             and x.shape[0] > 0 and x.shape[2] > 0 and x.shape[3] > 0 and x.shape[3] % 2 == 0,
             "reference Q/K must be floating [B,R,H,even D] on CPU/CUDA")
    _require(torch.is_tensor(frequencies) and frequencies.dtype in (torch.complex64, torch.complex128)
             and frequencies.shape == (x.shape[1], 1, x.shape[3] // 2) and frequencies.device == x.device
             and not frequencies.requires_grad and frequencies.layout == torch.strided,
             "reference frequencies must be fixed complex [R,1,D/2] on the Q/K device")
    pairs = x.float().reshape(*x.shape[:3], x.shape[3] // 2, 2).contiguous()
    rotated = torch.view_as_complex(pairs) * frequencies.to(torch.complex64)
    return torch.view_as_real(rotated).flatten(3).to(x.dtype)
