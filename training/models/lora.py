"""Small dependency-free LoRA implementation for Wan attention/FFN layers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

DEFAULT_TARGET_SUFFIXES = (
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "cross_attn.q",
    "cross_attn.k",
    "cross_attn.v",
    "cross_attn.o",
    "ffn.0",
    "ffn.2",
)


class LoRALinear(nn.Module):
    """Frozen linear layer with a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.base.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.lora_b(self.lora_a(self.dropout(inputs))) * self.scaling


@dataclass(frozen=True)
class TrainableSummary:
    replaced_linear_layers: tuple[str, ...]
    trainable_parameters: int
    total_parameters: int

    @property
    def trainable_fraction(self) -> float:
        return self.trainable_parameters / max(1, self.total_parameters)


def _resolve_parent(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    *,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_suffixes: tuple[str, ...] = DEFAULT_TARGET_SUFFIXES,
) -> tuple[str, ...]:
    """Replace matching Wan attention/FFN linear layers with LoRA wrappers."""

    matches = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and any(name.endswith(suffix) for suffix in target_suffixes)
    ]
    for name in matches:
        parent, leaf = _resolve_parent(model, name)
        layer = getattr(parent, leaf)
        setattr(parent, leaf, LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout))
    if not matches:
        raise ValueError(
            "no Wan attention/FFN linear layers matched the LoRA targets; "
            "the upstream model interface may have changed"
        )
    return tuple(matches)


def configure_action_teacher(
    backbone: nn.Module,
    *,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> TrainableSummary:
    """Freeze the Wan backbone, then enable only LoRA and the action adapter."""

    backbone.requires_grad_(False)
    action_adapter = getattr(backbone, "act_control_adapter", None)
    if action_adapter is None:
        raise ValueError(
            "the loaded Wan backbone has no official act_control_adapter; "
            "load it with model_type='ci2v' rather than installing a guessed adapter"
        )
    action_adapter.requires_grad_(True)
    replaced = inject_lora(backbone, rank=rank, alpha=alpha, dropout=dropout)
    trainable = sum(parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in backbone.parameters())
    return TrainableSummary(replaced, trainable, total)


def trainable_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return CPU copies of only the adapter/LoRA tensors."""

    trainable_names = {name for name, parameter in module.named_parameters() if parameter.requires_grad}
    return {
        name: tensor.detach().cpu()
        for name, tensor in module.state_dict().items()
        if name in trainable_names
    }


def load_trainable_state_dict(module: nn.Module, state: dict[str, torch.Tensor]) -> None:
    expected = {name for name, parameter in module.named_parameters() if parameter.requires_grad}
    missing = expected.difference(state)
    unexpected = set(state).difference(expected)
    if missing or unexpected:
        raise ValueError(
            f"trainable checkpoint mismatch: missing={sorted(missing)[:8]}, "
            f"unexpected={sorted(unexpected)[:8]}"
        )
    current = module.state_dict()
    current.update(state)
    module.load_state_dict(current, strict=True)
