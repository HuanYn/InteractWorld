"""Isolated CFG=5 batch candidate; not enabled by any default inference path.

This changes execution batch size, not the number of logical text branches.
CPU contract fixtures cannot establish actual Wan kernel equivalence or speed.
No input/weight mutation, history cache, text cache or LoRA patch is installed.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from training.demo.contracts import require


class BatchedTextCFGFlow:
    """Evaluate [positive, negative] together for one plain noncausal sample."""

    def __init__(self, model, scale, negative):
        require(type(scale) in (int, float) and scale == 5,
                'batched CFG qualification fixes scale=5')
        require(isinstance(model, nn.Module), 'batched CFG requires the original model module')
        self.model, self.scale, self.negative = model, scale, negative
        self.forward_calls = self.logical_branch_evaluations = self.completed_calls = 0
        self._last = dict(last_model_input_shape=None, last_timestep_shape=None,
                          last_action_context_count=None,
                          last_action_context_shared_reference=None)

    @property
    def stats(self):
        return dict(kind='batched_text_cfg_v1', cfg_scale=self.scale,
                    branch_order=['positive', 'negative'],
                    forward_calls=self.forward_calls,
                    logical_branch_evaluations=self.logical_branch_evaluations,
                    completed_calls=self.completed_calls,
                    completed_logical_branch_evaluations=2 * self.completed_calls,
                    action_context_materialized_copies=0,
                    text_kv_cached=False, video_history_cached=False,
                    lora_inplace=False, **self._last)

    def _validate(self, current, conditions, timestep, replace_first):
        require(not torch.is_grad_enabled(), 'batched CFG requires caller no_grad')
        require(getattr(self.model, 'is_causal', None) is False
                and getattr(self.model, 'uniform_timestep', None) is True,
                'batched CFG requires the plain noncausal wrapper')
        require(all(not module.training for module in self.model.modules())
                and all(not parameter.requires_grad for parameter in self.model.parameters()),
                'batched CFG requires all model modules eval and parameters frozen')
        require(all(not hasattr(module, '_studio_text_kv_provider')
                    and 'forward' not in module.__dict__ for module in self.model.modules()),
                'batched CFG cannot combine text-cache or temporary forward overrides')
        require(replace_first is True, 'batched CFG requires the clean first-slot path')
        require(torch.is_tensor(current) and current.ndim == 5 and current.shape[0] == 1
                and current.shape[1] >= 2 and all(size > 0 for size in current.shape[2:])
                and current.shape[-2] % 2 == current.shape[-1] % 2 == 0
                and current.dtype in (torch.float32, torch.bfloat16)
                and not current.requires_grad,
                'batched CFG needs B=1 floating [1,F,C,H,W] with even spatial geometry')
        _, frames, _, height, width = current.shape
        require(torch.is_tensor(timestep) and tuple(timestep.shape) == (1, frames)
                and timestep.dtype == current.dtype and timestep.device == current.device
                and not timestep.requires_grad,
                'batched CFG timestep shape/dtype/device must match [1,F]')
        # Only this small per-slot tensor is inspected; never scan the expanded
        # spatial action context, already built from the worker's legal actions.
        require(bool(torch.isfinite(timestep).all()) and timestep[0, 0].item() == 0
                and bool((timestep[0, 1:] > 0).all())
                and bool((timestep[0, 1:] == timestep[0, 1]).all()),
                'batched CFG requires slot0=0 and a shared positive future timestep')
        require(type(conditions) is dict and set(conditions) ==
                {'prompt_embeds', 'act_context', 'act_context_scale'},
                'batched CFG only accepts prompt_embeds, act_context, act_context_scale')
        prompts, actions = conditions['prompt_embeds'], conditions['act_context']
        require(type(prompts) is list and len(prompts) == 1
                and type(actions) is list and len(actions) == 1,
                'batched CFG requires one prompt and one action context; no implicit zip truncation')
        positive, action = prompts[0], actions[0]
        require(torch.is_tensor(positive) and positive.ndim == 2
                and 1 <= positive.shape[0] <= 512 and positive.shape[1] > 0
                and torch.is_tensor(self.negative) and self.negative.ndim == 2
                and 1 <= self.negative.shape[0] <= 512
                and self.negative.shape[1] == positive.shape[1],
                'batched CFG requires compatible explicit positive and negative text embeddings')
        require(all(value.dtype == current.dtype and value.device == current.device
                    and not value.requires_grad for value in (positive, self.negative)),
                'batched CFG text dtype/device/gradient contract differs')
        require(torch.is_tensor(action) and tuple(action.shape) ==
                (32, frames - 1, height * 16, width * 16)
                and action.dtype == current.dtype and action.device == current.device
                and not action.requires_grad,
                'batched CFG action must be [32,F-1,H*16,W*16] with matching dtype/device')
        require(type(conditions['act_context_scale']) in (int, float)
                and math.isfinite(conditions['act_context_scale']),
                'batched CFG action scale must be an explicit finite scalar')
        return positive, action

    def __call__(self, noisy_image_or_video, *, conditional_dict, timestep,
                 replace_first_timestep_and_noise_latents=True):
        positive, action = self._validate(noisy_image_or_video, conditional_dict, timestep,
                                         replace_first_timestep_and_noise_latents)
        current = torch.cat((noisy_image_or_video, noisy_image_or_video), dim=0)
        times = torch.cat((timestep, timestep), dim=0)
        # Both branches read the same action tensor. Do not materialize another
        # full-resolution action context or combine samples along temporal axes.
        conditions = dict(prompt_embeds=[positive, self.negative],
                          act_context=[action, action],
                          act_context_scale=conditional_dict['act_context_scale'])
        self._last = dict(last_model_input_shape=list(current.shape),
                          last_timestep_shape=list(times.shape),
                          last_action_context_count=len(conditions['act_context']),
                          last_action_context_shared_reference=
                          conditions['act_context'][0] is conditions['act_context'][1])
        self.forward_calls += 1
        self.logical_branch_evaluations += 2
        outputs = self.model(current, conditional_dict=conditions, timestep=times,
                             replace_first_timestep_and_noise_latents=True)
        require(isinstance(outputs, (tuple, list)) and len(outputs) == 2,
                'batched CFG expects the original (velocity, x0) model return')
        velocity = outputs[0]
        require(torch.is_tensor(velocity) and tuple(velocity.shape) == tuple(current.shape)
                and velocity.dtype == current.dtype and velocity.device == current.device
                and not velocity.requires_grad,
                'batched CFG output must be [2,F,C,H,W] with unchanged dtype/device')
        pos, neg = velocity[:1], velocity[1:2]
        guided = neg + self.scale * (pos - neg)
        self.completed_calls += 1
        return guided, None
