"""Opt-in UniPC flow candidate; no model loading and no changes to Euler defaults."""
from __future__ import annotations

from training.demo.contracts import require


def make_scheduler(steps, device):
    import diffusers
    import torch
    from diffusers import UniPCMultistepScheduler

    require(type(steps) is int and steps == 40, 'UniPC quality candidate fixes 40 steps')
    # A fresh instance for every window: neither predictor nor corrector history
    # may leak into the next independently conditioned video segment.
    scheduler = UniPCMultistepScheduler(
        num_train_timesteps=1000, solver_order=2, prediction_type='flow_prediction',
        predict_x0=True, thresholding=False, solver_type='bh2', lower_order_final=True,
        use_flow_sigmas=True, flow_shift=5.0, final_sigmas_type='zero')
    scheduler.set_timesteps(steps, device=device)
    require(len(scheduler.timesteps) == steps and len(scheduler.sigmas) == steps + 1
            and bool(torch.isfinite(scheduler.sigmas).all())
            and bool((scheduler.sigmas[:-1] > scheduler.sigmas[1:]).all())
            and scheduler.sigmas[-1].item() == 0, 'invalid UniPC flow schedule')
    receipt = dict(solver='flow_unipc', scheduler_class=type(scheduler).__name__,
        diffusers_version=diffusers.__version__, scheduler_config=dict(scheduler.config),
        timesteps=scheduler.timesteps.detach().cpu().tolist(),
        sigmas=scheduler.sigmas.detach().cpu().tolist(),
        schedule_source='diffusers flow schedule; NOT the existing BF16 Euler sigma nodes',
        state_dtype='float32', model_timestep_dtype='float32 from scheduler integer timesteps',
        first_latent_policy='fixed; timestep zero; excluded from scheduler state',
        state_scope='fresh predictor/corrector history for this window only')
    return scheduler, receipt


def unipc_rollout(model, *, first, noise, conditions, steps=40, trace=None):
    import torch

    require(torch.is_tensor(noise) and noise.ndim == 5 and noise.shape[1] > 1
            and noise.is_floating_point(), 'UniPC requires floating [B,T,C,H,W] noise')
    require(torch.is_tensor(first) and tuple(first.shape) == (noise.shape[0], 1, *noise.shape[2:])
            and first.device == noise.device and first.dtype == noise.dtype,
            'UniPC first latent must match noise shape/device/dtype')
    require(bool(torch.isfinite(first).all()) and bool(torch.isfinite(noise).all()),
            'nonfinite UniPC input')
    scheduler, receipt = make_scheduler(steps, noise.device)
    receipt['model_input_dtype'] = str(noise.dtype)
    if trace is not None:
        trace.update(receipt)
    # Preserve the same already-quantized initial noise. FP32 scheduler state is
    # explicit and is not presented as an identical-node/precision Euler ablation.
    fixed_first = first.detach().float().clone()
    future = noise[:, 1:].detach().float().clone()
    with torch.no_grad():
        for timestep in scheduler.timesteps:
            current = torch.cat((fixed_first, future), dim=1).to(dtype=noise.dtype)
            model_t = timestep.to(dtype=torch.float32).expand(current.shape[:2]).clone()
            model_t[:, 0] = 0
            velocity, _ = model(current, conditional_dict=conditions, timestep=model_t,
                                replace_first_timestep_and_noise_latents=True)
            require(torch.is_tensor(velocity) and velocity.shape == current.shape
                    and bool(torch.isfinite(velocity).all()), 'invalid/nonfinite UniPC velocity')
            # Only future slots are integrated; first-frame predictions never
            # enter scheduler.model_outputs or its multistep corrector history.
            # The caller autocasts the actual Wan forward. Do not autocast the
            # scheduler's multistep einsums back to BF16 while claiming FP32.
            with torch.autocast(device_type=noise.device.type, enabled=False):
                future = scheduler.step(velocity[:, 1:].float(), timestep, future).prev_sample
            require(future.dtype == torch.float32 and bool(torch.isfinite(future).all()),
                    'nonfinite or unexpected-precision UniPC state')
    result = torch.cat((fixed_first, future), dim=1)
    require(torch.equal(result[:, :1], first.float()), 'UniPC changed the clean first latent')
    return result


def generate(runtime, flow, first, prompt, keys, noise, steps, trace):
    """Build the same actual Action conditions, then use the isolated candidate."""
    from training.models.action_adapter import build_action_context

    torch = runtime.torch
    condition = first.to(device='cuda', dtype=torch.bfloat16)
    conditions = dict(prompt_embeds=[prompt.to(device='cuda', dtype=torch.bfloat16)],
        act_context_scale=runtime.config.action_scale,
        act_context=build_action_context(keys, height=first.shape[-2] * 16,
            width=first.shape[-1] * 16, device='cuda', dtype=torch.bfloat16))
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        current = unipc_rollout(flow, first=condition,
            noise=noise.to(device='cuda', dtype=torch.bfloat16),
            conditions=conditions, steps=steps, trace=trace)
    return current.detach().cpu()
