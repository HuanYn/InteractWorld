"""Trainable model components for the ABot-inspired training pipeline."""

from .action_adapter import (
    ACTION_DIM,
    CANONICAL_ACTION_KEYS,
    RGB_FRAMES_PER_ACTION_TOKEN,
    build_action_context,
    inject_action_features,
    pack_canonical_actions,
)
from .lora import LoRALinear, TrainableSummary, configure_action_teacher

__all__ = [
    "ACTION_DIM",
    "CANONICAL_ACTION_KEYS",
    "RGB_FRAMES_PER_ACTION_TOKEN",
    "LoRALinear",
    "TrainableSummary",
    "build_action_context",
    "configure_action_teacher",
    "inject_action_features",
    "pack_canonical_actions",
]
