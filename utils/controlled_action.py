"""Opt-in bounded action residual for a NEW teacher stage, never a legacy fix.

The FP32 residual obeys RMS(delta) <= ratio_cap * RMS(video), per spatial/
temporal token over channels. Casting and adding in BF16 introduce normal
rounding error; this mechanism guarantee is not a claim of video/control quality.
"""
from __future__ import annotations

import math
from types import MethodType

import torch
from torch import nn


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def install_controlled_action(model: nn.Module, *, ratio_cap=.1, eps=1e-6):
    """Install a zero FP32 channel gate on the actual Wan network exactly once.

An instance-local device/dtype-transfer wrapper keeps the gate and its gradient
FP32 without first rounding through the requested model dtype. Checkpoints must
use state_dict, not pickle the modified module. Install before loading gate state.
    """
    _require(isinstance(model, nn.Module), 'controlled action requires an nn.Module')
    _require(float(ratio_cap) == .1, 'this controlled stage has a fixed ratio_cap=0.1')
    _require(math.isfinite(float(eps)) and float(eps) > 0, 'eps must be positive and finite')
    _require(not hasattr(model, 'controlled_action_gate')
             and not hasattr(model, 'controlled_action_enabled'), 'controlled action already installed')
    _require(isinstance(getattr(model, 'act_control_adapter', None), nn.Module),
             'actual Wan action adapter required')
    dim = getattr(model, 'dim', None)
    _require(type(dim) is int and dim > 0, 'actual Wan channel dimension required')
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device('cpu')
    _require(device.type != 'meta', 'install controlled action after materializing model')
    model.register_parameter('controlled_action_gate', nn.Parameter(torch.zeros(dim, device=device, dtype=torch.float32)))
    model.controlled_action_enabled = True
    model.controlled_action_ratio_cap = float(ratio_cap)
    model.controlled_action_eps = float(eps)
    original_apply = model._apply

    def apply_preserving_gate(self, fn, recurse=True):
        # Temporarily exclude only this parameter from the ordinary dtype cast.
        # Keep identity so an existing optimizer still owns the same parameter.
        gate = self._parameters.pop('controlled_action_gate')
        try:
            with torch.no_grad():
                probe = fn(torch.empty(0, device=gate.device, dtype=torch.float32))
                _require(probe.device.type != 'meta', 'controlled gate cannot migrate to meta')
                original_apply(fn, recurse=recurse)
                gate.data = gate.data.to(device=probe.device, dtype=torch.float32)
                if gate.grad is not None:
                    gate.grad.data = gate.grad.data.to(device=probe.device, dtype=torch.float32)
        finally:
            self._parameters['controlled_action_gate'] = gate
        return self

    model._apply = MethodType(apply_preserving_gate, model)
    return model


def controlled_action_contract(model):
    _require(getattr(model, 'controlled_action_enabled', False) is True, 'controlled action is not installed')
    gate = getattr(model, 'controlled_action_gate', None)
    _require(isinstance(gate, nn.Parameter) and gate.dtype == torch.float32,
             'controlled action gate must remain an FP32 Parameter')
    return dict(kind='zero_gated_relative_rms_action_v1',
        ratio_cap=model.controlled_action_ratio_cap, eps=model.controlled_action_eps,
        gate_parameter='controlled_action_gate', gate_shape=list(gate.shape), gate_dtype=str(gate.dtype),
        normalization='per_B_T_H_W_token_over_channels_FP32',
        reference='RMS_detached_future_video_patch_features',
        gate_function='tanh_channelwise', initial_gate='zero',
        injection='right_aligned_future_slots_only', first_latent_preserved=True,
        legacy_action_scale_applied=False,
        fp32_bound='RMS(delta)<=ratio_cap*RMS(video); dtype cast/add subject to rounding',
        quality_or_control_verified=False)


def inject_controlled_action_features(model, video_features, action_features):
    """Same differentiable injection in train/eval; never changes the clean prefix."""
    _require(getattr(model, 'controlled_action_enabled', False) is True, 'controlled action is not installed')
    u, v = video_features, action_features
    _require(torch.is_tensor(u) and torch.is_tensor(v) and u.ndim == v.ndim == 5,
             'controlled features must be [B,C,T,H,W]')
    _require(u.is_floating_point() and v.is_floating_point(), 'controlled features must be floating point')
    _require(u.shape[:2] == v.shape[:2] and u.shape[-2:] == v.shape[-2:]
             and all(size > 0 for size in u.shape) and v.shape[2] > 0,
             'controlled batch/channel/spatial geometry differs or is empty')
    # This stage is specifically TI2V: at least one clean temporal prefix slot.
    _require(u.shape[2] > v.shape[2], 'controlled action must leave a clean first latent')
    gate = model.controlled_action_gate
    _require(isinstance(gate, nn.Parameter) and gate.dtype == torch.float32
             and gate.ndim == 1 and gate.numel() == u.shape[1], 'invalid controlled FP32 channel gate')
    _require(gate.device == u.device == v.device, 'controlled features and gate must share a device')
    _require(bool(torch.isfinite(gate).all()), 'controlled gate contains nonfinite values')
    cap, eps = model.controlled_action_ratio_cap, model.controlled_action_eps
    _require(cap == .1 and math.isfinite(eps) and eps > 0, 'controlled ratio/epsilon contract changed')
    offset = u.shape[2] - v.shape[2]
    future = u[:, :, offset:]
    # Disable outer autocast explicitly for both normalization and gating.
    with torch.autocast(device_type=u.device.type, enabled=False):
        vf = v.float()
        denominator = vf.square().mean(dim=1, keepdim=True).clamp_min(eps * eps).sqrt()
        reference = future.detach().float().square().mean(dim=1, keepdim=True).sqrt()
        delta = cap * reference * gate.tanh().view(1, -1, 1, 1, 1) * (vf / denominator)
    return torch.cat((u[:, :, :offset], future + delta.to(dtype=u.dtype)), dim=2)
