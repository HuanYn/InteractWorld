"""Opt-in bounded action injection for absolute-position causal windows.

This extends the existing teacher mechanism to a pure-future KV chunk, without
changing that teacher helper's clean-prefix contract or any legacy raw residual.
The bound is a mechanism property, not evidence of learned control or quality.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from utils.controlled_action import inject_controlled_action_features


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def causal_action_start_frame(current_start, frame_seqlen, *, temporal_patch_size=1):
    """Convert an absolute *video-token* offset; references are not counted."""
    _require(type(current_start) is int and current_start >= 0,
             'controlled causal current_start must be a nonnegative integer')
    _require(type(frame_seqlen) is int and frame_seqlen > 0,
             'controlled causal frame_seqlen must be a positive integer')
    _require(temporal_patch_size == 1,
             'controlled causal injection requires temporal patch size 1')
    _require(current_start % frame_seqlen == 0,
             'controlled causal current_start must align to a complete latent frame')
    return current_start // frame_seqlen


def inject_controlled_causal_action_features(
        model, video_features, action_features, *, absolute_start_frame):
    """Inject right-aligned actions only at absolute latent positions >= 1.

At absolute zero the window must contain a clean prefix, exactly as in the
bidirectional teacher. Later windows may contain only future latents (T == A).
No fake prefix, absolute-offset inference, additional action scale, or reference
feature injection is used. Noisy and clean TF features call this separately so
each branch uses its own detached patch-feature RMS.
    """
    _require(type(absolute_start_frame) is int and absolute_start_frame >= 0,
             'controlled causal absolute_start_frame must be a nonnegative integer')
    u, v = video_features, action_features
    _require(torch.is_tensor(u) and torch.is_tensor(v) and u.ndim == v.ndim == 5,
             'controlled causal features must be [B,C,T,H,W]')
    offset = u.shape[2] - v.shape[2]
    _require(offset >= 0 and absolute_start_frame + offset >= 1,
             'controlled causal action must not touch absolute first latent or exceed its window')
    if offset > 0:
        # Identical full-window teacher computation, including its validations.
        return inject_controlled_action_features(model, u, v)

    _require(getattr(model, 'controlled_action_enabled', False) is True,
             'controlled action is not installed')
    _require(u.is_floating_point() and v.is_floating_point(),
             'controlled causal features must be floating point')
    _require(u.shape == v.shape and all(size > 0 for size in u.shape),
             'controlled causal batch/channel/spatial geometry differs or is empty')
    gate = model.controlled_action_gate
    _require(isinstance(gate, nn.Parameter) and gate.dtype == torch.float32
             and gate.ndim == 1 and gate.numel() == u.shape[1],
             'invalid controlled FP32 channel gate')
    _require(gate.device == u.device == v.device,
             'controlled features and gate must share a device')
    _require(bool(torch.isfinite(gate).all()), 'controlled gate contains nonfinite values')
    cap, eps = model.controlled_action_ratio_cap, model.controlled_action_eps
    _require(cap == .1 and math.isfinite(eps) and eps > 0,
             'controlled ratio/epsilon contract changed')
    with torch.autocast(device_type=u.device.type, enabled=False):
        vf = v.float()
        denominator = vf.square().mean(dim=1, keepdim=True).clamp_min(eps * eps).sqrt()
        reference = u.detach().float().square().mean(dim=1, keepdim=True).sqrt()
        delta = cap * reference * gate.tanh().view(1, -1, 1, 1, 1) * (vf / denominator)
    return u + delta.to(dtype=u.dtype)
