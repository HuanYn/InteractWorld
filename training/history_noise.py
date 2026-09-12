"""Teacher-free causal history augmentation with matched augmentation timesteps.

The current noisy block, original GT flow target, text, and actions are unchanged.
Only the separate ``clean_x`` context stream is perturbed. The existing causal
teacher-forcing mask still restricts each prediction to completed context blocks.
This is GT-derived noise augmentation, NOT generated-history training, teacher
distillation, error replay, or proof of improved long-video quality.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from training.causal_tf import _move, causal_teacher_forcing_loss, validate_causal_batch
from training.data.action_dataset import _seed64
from training.models.action_adapter import build_action_context

STAGE_NAME = "causal_history_noise_v1"
NOISE_NAMESPACE = "causal_history_noise_absolute_micro_v1"


@dataclass(frozen=True)
class HistoryNoiseConfig:
    schema_version: int = 1
    enabled: bool = True
    seed: int = 1842
    clean_probability: float = 0.5
    sigma_start: float = 0.05
    sigma_end: float = 0.15
    curriculum_steps: int = 80

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("history noise schema_version must be 1")
        if type(self.enabled) is not bool:
            raise ValueError("history noise enabled must be bool")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("history noise seed must be a nonnegative int64")
        if type(self.curriculum_steps) is not int or not 1 <= self.curriculum_steps <= 10000:
            raise ValueError("history noise curriculum_steps must be an integer in [1,10000]")
        for name in ("clean_probability", "sigma_start", "sigma_end"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"history noise {name} must be finite")
        if not 0 <= self.clean_probability <= 1:
            raise ValueError("history noise clean_probability must be in [0,1]")
        if not 0 <= self.sigma_start <= self.sigma_end <= .15:
            raise ValueError("bounded history noise requires 0 <= sigma_start <= sigma_end <= 0.15")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HistoryNoiseConfig":
        if not isinstance(value, Mapping):
            raise ValueError("history noise config must be a mapping")
        unknown = set(value).difference(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown history noise fields: {sorted(unknown)}")
        result = cls(**dict(value))
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def history_noise_contract(config: HistoryNoiseConfig) -> dict[str, Any]:
    config.validate()
    return {
        "stage": STAGE_NAME, "config": config.to_dict(), "sampling_unit": "per_sample",
        "corruption": "context=(1-sigma)*clean+sigma*independent_standard_Gaussian",
        "sigma_distribution": "clean_probability atom at zero; otherwise Uniform(0,curriculum_max_sigma)",
        "curriculum_position": "new_stage_completed_optimizer_steps; 0=start, curriculum_steps=end",
        "noise_seed_policy": "seed64(seed,namespace,absolute_microbatch_index,batch_slot)",
        "noise_namespace": NOISE_NAMESPACE, "stateless_resume": True,
        "first_latent": "bitwise_preserved_and_aug_t_zero", "augmentation_timestep": "1000*sigma_FP32_per_latent",
        "prediction_target": "unchanged_original_dataset_GT_flow_future_latents_only",
        "current_noisy_latents_changed": False, "actions_or_prompt_changed": False,
        "teacher_model": False, "teacher_predictions": False, "student_rollout": False,
        "cross_sample_error_replay": False, "causal_visibility_mask_changed": False,
    }


def max_history_sigma(config: HistoryNoiseConfig, stage_optimizer_step: int) -> float:
    config.validate()
    if type(stage_optimizer_step) is not int or stage_optimizer_step < 0:
        raise ValueError("stage_optimizer_step must be the nonnegative completed optimizer-step count")
    progress = min(stage_optimizer_step, config.curriculum_steps) / config.curriculum_steps
    return config.sigma_start + progress * (config.sigma_end - config.sigma_start)


def prepare_history_context(clean: torch.Tensor, *, noise_config: HistoryNoiseConfig,
                            absolute_microbatch_index: int, stage_optimizer_step: int
                            ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Stateless CPU corruption; never touch global CPU/CUDA/diffusion RNG state."""
    noise_config.validate()
    if type(absolute_microbatch_index) is not int or absolute_microbatch_index < 0:
        raise ValueError("absolute_microbatch_index must be a nonnegative integer")
    cap = max_history_sigma(noise_config, stage_optimizer_step)
    if (not torch.is_tensor(clean) or clean.ndim != 5 or clean.shape[1] < 2
            or any(size < 1 for size in clean.shape) or not clean.is_floating_point()
            or clean.device.type != "cpu" or not bool(torch.isfinite(clean).all())):
        raise ValueError("history augmentation requires finite floating CPU [B,F,C,H,W] clean latents")
    original = clean.detach().float()
    context = original.clone()
    aug_t = torch.zeros(clean.shape[:2], dtype=torch.float32, device="cpu")
    rows = []
    for slot in range(clean.shape[0]):
        seed = _seed64(noise_config.seed, NOISE_NAMESPACE, absolute_microbatch_index, slot)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        selected = noise_config.enabled and float(torch.rand((), generator=generator)) >= noise_config.clean_probability
        sigma = cap * float(torch.rand((), generator=generator)) if selected else 0.0
        if sigma > 0:
            eps = torch.randn(original[slot, 1:].shape, generator=generator, dtype=torch.float32, device="cpu")
            context[slot, 1:] = (1.0 - sigma) * original[slot, 1:] + sigma * eps
            aug_t[slot, 1:] = 1000.0 * sigma
        rows.append({"batch_slot": slot, "noise_seed": seed, "history_noised": sigma > 0,
                     "sigma": sigma, "augmentation_timestep": float(aug_t[slot, 1]),
                     "perturbation_rms": float((context[slot, 1:] - original[slot, 1:]).square().mean().sqrt())})
    if not torch.equal(context[:, :1], original[:, :1]) or bool(aug_t[:, :1].any()):
        raise RuntimeError("history noise changed protected first latent/timestep")
    return context, aug_t, {
        "method": STAGE_NAME, "enabled": noise_config.enabled,
        "absolute_microbatch_index": absolute_microbatch_index,
        "stage_completed_optimizer_steps": stage_optimizer_step, "curriculum_max_sigma": cap,
        "noised_samples": sum(row["history_noised"] for row in rows), "samples": rows,
        "first_latent_unchanged": True, "new_teacher_model": False,
        "context_source": "current_GT_reconstruction_only_not_generated_rollout",
    }


def history_noise_loss(model, batch: Mapping[str, Any], config, device: torch.device, *,
                       noise_config: HistoryNoiseConfig, absolute_microbatch_index: int,
                       stage_optimizer_step: int, metrics: dict[str, Any] | None = None) -> torch.Tensor:
    """Keep original GT flow supervision and causal dispatch; change context only."""
    noise_config.validate()
    if config.error_recycling.enabled:
        raise ValueError("history-noise v1 cannot be combined with the closed error-recycling branch")
    clean_cpu = validate_causal_batch(batch, config)
    context_cpu, aug_t_cpu, receipt = prepare_history_context(clean_cpu,
        noise_config=noise_config, absolute_microbatch_index=absolute_microbatch_index,
        stage_optimizer_step=stage_optimizer_step)
    if metrics is not None:
        metrics.update(receipt)
    if receipt["noised_samples"] == 0:
        # Exact original implementation for disabled controls and the clean half
        # of the mixture; no alternate loss reduction or altered model dispatch.
        if metrics is not None:
            metrics["forward_path"] = "unchanged_original_clean_history_loss"
        return causal_teacher_forcing_loss(model, batch, config, device)

    noisy = _move(batch["noisy_latents"], device, dtype=torch.bfloat16)
    target = _move(batch["target_flow"], device, dtype=torch.bfloat16)
    context = _move(context_cpu, device, dtype=torch.bfloat16)
    timesteps = _move(batch["timesteps"], device)
    if timesteps.ndim == 1:
        timesteps = timesteps.unsqueeze(0)
    prompt = _move(batch["prompt_embeds"], device, dtype=torch.bfloat16)
    if torch.is_tensor(prompt):
        prompt = [sample for sample in prompt]
    actions = _move(batch["actions"], device, dtype=torch.bfloat16)
    conditions = {
        "prompt_embeds": prompt,
        "act_context": build_action_context(actions, height=config.data.height, width=config.data.width,
                                             device=device, dtype=torch.bfloat16),
        "act_context_scale": config.model.action_scale,
    }
    for key in ("y", "clip_fea"):
        if key in batch:
            conditions[key] = _move(batch[key], device, dtype=torch.bfloat16)
    output = model(noisy, conditional_dict=conditions, timestep=timesteps, clean_x=context,
                   aug_t=aug_t_cpu.to(device=device), replace_first_timestep_and_noise_latents=True)
    prediction = output[0] if isinstance(output, tuple) else output
    if tuple(prediction.shape) != tuple(target.shape):
        raise RuntimeError("history-noise causal prediction and unchanged GT target shapes differ")
    if metrics is not None:
        metrics["forward_path"] = "same_causal_GT_flow_with_noised_history_and_matched_aug_t"
    return torch.nn.functional.mse_loss(prediction[:, 1:].float(), target[:, 1:].float())
