"""Action packing and the ABot-compatible spatial control adapter.

The released ABot inference path represents every four RGB-frame actions as
32 channels (``8 keys * 4 frames``), broadcasts them spatially, and supplies
the result as ``act_context``.  The Wan models in this repository already add
the adapter output to their video features.  This module makes that contract
explicit and testable without loading model weights.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real

import torch

from utils.action_alignment import inject_action_features

__all__ = [
    "ACTION_DIM",
    "CANONICAL_ACTION_KEYS",
    "RGB_FRAMES_PER_ACTION_TOKEN",
    "build_action_context",
    "inject_action_features",
    "pack_canonical_actions",
    "validate_action_scale",
]

CANONICAL_ACTION_KEYS: tuple[str, ...] = ("W", "A", "S", "D", "I", "J", "K", "L")
RGB_FRAMES_PER_ACTION_TOKEN = 4
ACTION_DIM = len(CANONICAL_ACTION_KEYS) * RGB_FRAMES_PER_ACTION_TOKEN


def validate_action_scale(value: object) -> float:
    """Validate a residual multiplier; zero is diagnostic, not working control."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("action_scale must be a finite nonnegative number")
    scale = float(value)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("action_scale must be a finite nonnegative number")
    return scale


def canonical_action_vector(action: Mapping[str, object]) -> list[float]:
    """Return one strict W/A/S/D/I/J/K/L vector.

    Unknown keys are rejected instead of being silently interpreted as no-op.
    """

    unknown = set(action).difference(CANONICAL_ACTION_KEYS)
    if unknown:
        raise ValueError(f"unknown action keys: {sorted(unknown)}")
    return [float(bool(action.get(key, False))) for key in CANONICAL_ACTION_KEYS]


def actions_to_tensor(
    actions: torch.Tensor | Sequence[Mapping[str, object]],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalize actions to ``[batch, rgb_frames, 8]``."""

    if torch.is_tensor(actions):
        tensor = actions.to(device=device, dtype=dtype)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
    else:
        tensor = torch.tensor(
            [canonical_action_vector(action) for action in actions],
            device=device,
            dtype=dtype,
        ).unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[-1] != len(CANONICAL_ACTION_KEYS):
        raise ValueError(
            "actions must be [batch, rgb_frames, 8] in canonical "
            f"{CANONICAL_ACTION_KEYS} order, got {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError("actions contain non-finite values")
    if not torch.logical_or(tensor == 0, tensor == 1).all():
        raise ValueError("actions must contain only binary 0/1 values")
    return tensor


def pack_canonical_actions(
    actions: torch.Tensor | Sequence[Mapping[str, object]],
    *,
    frames_per_token: int = RGB_FRAMES_PER_ACTION_TOKEN,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Pack four consecutive 8-key actions into a 32-D action token.

    Returns ``[batch, action_tokens, 32]``.  For a 49-frame TI2V clip the
    expected input is the 48 future-frame actions, yielding 12 action tokens;
    the clean first latent frame is intentionally left unconditioned.
    """

    if frames_per_token <= 0:
        raise ValueError("frames_per_token must be positive")
    tensor = actions_to_tensor(actions, device=device, dtype=dtype)
    if tensor.shape[1] % frames_per_token:
        raise ValueError(
            f"action frame count {tensor.shape[1]} is not divisible by {frames_per_token}; "
            "pass future-frame actions only and reject misaligned windows"
        )
    batch, frames, keys = tensor.shape
    # Match the released inference implementation's repeat_interleave(4,
    # dim=channel): W[t0:t4], A[t0:t4], ..., L[t0:t4].
    grouped = tensor.reshape(batch, frames // frames_per_token, frames_per_token, keys)
    return grouped.transpose(-1, -2).reshape(
        batch, frames // frames_per_token, keys * frames_per_token
    )


def build_action_context(
    actions: torch.Tensor | Sequence[Mapping[str, object]],
    *,
    height: int,
    width: int,
    frames_per_token: int = RGB_FRAMES_PER_ACTION_TOKEN,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> list[torch.Tensor]:
    """Build the released Wan/ABot ``act_context`` list.

    Each list item has shape ``[32, action_tokens, height, width]`` and is
    accepted directly by ``WanModel.forward(act_context=...)``.
    """

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    packed = pack_canonical_actions(
        actions, frames_per_token=frames_per_token, device=device, dtype=dtype
    )
    context = packed.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
    context = context.expand(-1, -1, -1, height, width).contiguous()
    return [sample for sample in context]
