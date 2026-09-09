# InterActWorld reproduction: adapted from ABot-World; see NOTICE.
"""Tensor-only action-conditioning alignment shared by Wan training paths."""

from __future__ import annotations

import torch


def reset_missing_action_adapter(module: torch.nn.Module, missing_keys) -> bool:
    """Deterministically initialize a complete adapter absent from a base checkpoint."""
    adapter = getattr(module, "act_control_adapter", None)
    if adapter is None:
        return False
    adapter_keys = {f"act_control_adapter.{name}" for name in adapter.state_dict()}
    missing_adapter_keys = adapter_keys.intersection(missing_keys)
    if not missing_adapter_keys:
        return False
    if missing_adapter_keys != adapter_keys:
        raise RuntimeError("checkpoint contains only a partial action adapter")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        for child in adapter.modules():
            reset = getattr(child, "reset_parameters", None)
            if callable(reset):
                reset()
    return True


def expand_frame_conditioning_to_tokens(
    e: torch.Tensor, grid_sizes: torch.Tensor, seq_len: int
) -> torch.Tensor:
    """Expand per-frame conditioning to each spatial patch token."""
    if e.shape[1] == 1:
        return e
    if e.shape[0] != grid_sizes.shape[0]:
        raise ValueError("conditioning batch does not match grid_sizes")
    expanded = []
    for sample, grid in zip(e, grid_sizes):
        frames, height, width = (int(value) for value in grid.tolist())
        if sample.shape[0] != frames:
            raise ValueError(
                f"per-frame conditioning has {sample.shape[0]} frames, expected {frames}"
            )
        tokens = sample.repeat_interleave(height * width, dim=0)
        if tokens.shape[0] > seq_len:
            raise ValueError("expanded conditioning exceeds model sequence length")
        if tokens.shape[0] < seq_len:
            padding = tokens.new_zeros((seq_len - tokens.shape[0], *tokens.shape[1:]))
            tokens = torch.cat([tokens, padding], dim=0)
        expanded.append(tokens)
    return torch.stack(expanded)


def inject_action_features(
    video_features: torch.Tensor,
    action_features: torch.Tensor,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Right-align action features, preserving a clean leading TI2V frame."""
    if video_features.ndim != 5 or action_features.ndim != 5:
        raise ValueError("video_features and action_features must be [B,C,T,H,W]")
    if video_features.shape[:2] != action_features.shape[:2]:
        raise ValueError("batch/channel dimensions of action and video features must match")
    if video_features.shape[-2:] != action_features.shape[-2:]:
        raise ValueError("spatial dimensions of action and video features must match")
    if action_features.shape[2] > video_features.shape[2]:
        raise ValueError("action sequence cannot be longer than video features")
    if action_features.shape[2] == 0:
        return video_features
    offset = video_features.shape[2] - action_features.shape[2]
    return torch.cat(
        [
            video_features[:, :, :offset],
            video_features[:, :, offset:] + action_features * scale,
        ],
        dim=2,
    )
