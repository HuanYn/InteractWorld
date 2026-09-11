"""MoBA-inspired sequential BID regularization, not the paper's packed mask.

The same causal backbone computes TF loss and immediately backpropagates it,
then computes a full-bidirectional noisy-video auxiliary loss and backpropagates
that separately. No second model or clean future context enters the BID branch.
This implements neither consistency distillation nor DMD nor LingBot weights.
Inspiration: https://arxiv.org/html/2607.07534v1#S3.SS2
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from training.causal_tf import (
    CausalTeacherForcingConfig, CausalModelConfig, CausalDataConfig,
    TeacherLineageConfig, CausalOptimizerConfig, CausalTrainingConfig,
    _tuple_fields, _move, causal_teacher_forcing_loss, validate_causal_batch,
)
from training.models.action_adapter import build_action_context

STAGE_NAME = "causal_moba_regularized_v1"
METHOD_NAME = "moba_inspired_sequential_bid_v1"


@dataclass
class MoBARegularizationConfig:
    method: str = METHOD_NAME
    bidirectional_weight: float = 0.1

    def validate(self) -> None:
        if self.method != METHOD_NAME:
            raise ValueError(f"regularization.method must be {METHOD_NAME!r}")
        weight = self.bidirectional_weight
        if (isinstance(weight, bool) or not isinstance(weight, Real)
                or not math.isfinite(weight) or not 0 <= weight <= 1):
            raise ValueError("bidirectional_weight must be a finite number in [0,1]")


@dataclass
class CausalMoBAConfig(CausalTeacherForcingConfig):
    regularization: MoBARegularizationConfig = field(default_factory=MoBARegularizationConfig)

    def validate(self) -> None:
        super().validate()
        self.regularization.validate()
        if self.data.prompt_cache_path is None:
            raise ValueError("sequential BID requires the same explicit scene-static prompt cache as TF")
        if self.model.lora_dropout != 0:
            raise ValueError("paired sequential BID v1 fixes lora_dropout=0")


def load_moba_config(path: str | Path) -> CausalMoBAConfig:
    import yaml

    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError("MoBA config must contain a mapping")
    allowed = {"model", "data", "lineage", "optimizer", "training", "regularization"}
    if set(raw).difference(allowed):
        raise ValueError(f"unknown config sections: {sorted(set(raw).difference(allowed))}")
    raw = _tuple_fields(raw)
    config = CausalMoBAConfig(
        model=CausalModelConfig(**raw.get("model", {})),
        data=CausalDataConfig(**raw.get("data", {})),
        lineage=TeacherLineageConfig(**raw.get("lineage", {})),
        optimizer=CausalOptimizerConfig(**raw.get("optimizer", {})),
        training=CausalTrainingConfig(**raw.get("training", {})),
        regularization=MoBARegularizationConfig(**raw.get("regularization", {})),
    )
    config.validate()
    return config


def method_contract(config: CausalMoBAConfig) -> dict[str, Any]:
    config.regularization.validate()
    return {
        "method": METHOD_NAME,
        "bidirectional_weight": float(config.regularization.bidirectional_weight),
        "backward_order": ["causal_teacher_forcing"] + (
            ["bidirectional_auxiliary"] if config.regularization.bidirectional_weight > 0 else []
        ),
        "bidirectional_enabled": config.regularization.bidirectional_weight > 0,
        "first_latent_in_loss": False,
        "bidirectional_clean_context": False,
        "bidirectional_kv_cache": False,
        "prompt_policy": "scene_static_only_v1",
        "same_batch_noise_actions_prompt": True,
        "packed_moba_mask": False,
        "consistency_distillation": False,
        "distribution_matching_distillation": False,
        "lingbot_weights_loaded": False,
        "quality_validated": False,
    }


def verify_bidirectional_interface(model: nn.Module) -> None:
    """Fail closed rather than falling back to a second causal/TF forward."""
    if "training_attention_mode" not in inspect.signature(model.forward).parameters:
        raise RuntimeError("wrapper lacks explicit training_attention_mode support")
    backbone = getattr(model, "model", None)
    forward = getattr(backbone, "_forward_train", None)
    if forward is None or "training_attention_mode" not in inspect.signature(forward).parameters:
        raise RuntimeError("causal backbone lacks explicit training_attention_mode support")


def bidirectional_flow_loss(
    model: nn.Module, batch: Mapping[str, Any], config: CausalMoBAConfig,
    device: torch.device,
) -> torch.Tensor:
    """BID sees noisy targets, the clean initial latent, actions and static text.

    Clean latents reconstructed by validation stay on CPU and are discarded;
    they are never passed to this forward. Full attention over noisy target
    latents is a training auxiliary, not a causal inference mode or a claim
    that this branch has no access to future noisy inputs.
    """
    validate_causal_batch(batch, config)
    noisy = _move(batch["noisy_latents"], device, dtype=torch.bfloat16)
    target = _move(batch["target_flow"], device, dtype=torch.bfloat16)
    timestep = _move(batch["timesteps"], device)
    if timestep.ndim == 1:
        timestep = timestep.unsqueeze(0)
    prompt = _move(batch["prompt_embeds"], device, dtype=torch.bfloat16)
    if torch.is_tensor(prompt):
        prompt = [sample for sample in prompt]
    conditions = {
        "prompt_embeds": prompt,
        "act_context": build_action_context(
            _move(batch["actions"], device, dtype=torch.bfloat16),
            height=config.data.height, width=config.data.width,
            device=device, dtype=torch.bfloat16,
        ),
        "act_context_scale": config.model.action_scale,
    }
    for key in ("y", "clip_fea"):
        if key in batch:
            conditions[key] = _move(batch[key], device, dtype=torch.bfloat16)
    output = model(
        noisy, conditional_dict=conditions, timestep=timestep,
        training_attention_mode="bidirectional",
    )
    prediction = output[0] if isinstance(output, tuple) else output
    if tuple(prediction.shape) != tuple(target.shape):
        raise RuntimeError("bidirectional prediction shape differs from target")
    return torch.nn.functional.mse_loss(prediction[:, 1:].float(), target[:, 1:].float())


def backward_microbatch(
    model: nn.Module, batch: Mapping[str, Any], config: CausalMoBAConfig,
    device: torch.device,
) -> dict[str, float | bool]:
    """Accumulate (TF + weight*BID)/accum gradients, freeing each graph in turn.

    Does not zero gradients or step the optimizer. No retained graph is needed.
    Weight zero calls the existing TF loss/backward only, not the BID interface.
    """
    config.regularization.validate()
    accumulation = config.training.gradient_accumulation_steps
    if isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    weight = float(config.regularization.bidirectional_weight)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        loss = causal_teacher_forcing_loss(model, batch, config, device)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite causal teacher-forcing loss")
    tf_value = float(loss.detach())
    (loss / accumulation).backward()
    del loss
    bid_value = 0.0
    if weight > 0:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = bidirectional_flow_loss(model, batch, config, device)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite bidirectional auxiliary loss")
        bid_value = float(loss.detach())
        (loss * (weight / accumulation)).backward()
        del loss
    total = tf_value + weight * bid_value
    return {"loss_tf": tf_value, "loss_bid": bid_value, "loss_total": total,
            "loss": total, "bid_evaluated": weight > 0}
