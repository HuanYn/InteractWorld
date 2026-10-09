"""Isolated fixed-view, spatial known-region flow projection.

This is a sampling constraint, not learned door interaction or navigation. The
caller must encode a repeated *pixel video* with the real temporal VAE to obtain
``static_latents``; repeating/broadcasting one image latent is not equivalent.
The tensor API can validate layout, not certify the caller's encoding provenance.
No future/open-door image or output-pixel compositing is used here. VAE spatial
receptive fields can still change decoded pixels across the mask boundary.
"""
from collections.abc import Mapping

import torch


def _finite_float(value, name):
    if not isinstance(value, torch.Tensor) or value.dtype not in (
        torch.float16, torch.bfloat16, torch.float32, torch.float64
    ):
        raise ValueError(f'{name} must be a floating tensor')
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite')


@torch.no_grad()
def spatial_known_rollout(model, *, first, static_latents, free_mask,
                          noise, conditions, sigmas):
    """Return latent video with original Euler/clean-first conventions.

    Layout is ``[B, F, C, H, W]`` (not VAE-native ``[B, C, F, H, W]``).
    ``static_latents`` must have the entire sequence shape; no temporal broadcast
    is performed. ``free_mask`` is binary ``[1 or B, 1, 1, H, W]``: one leaves a
    location free, zero follows the fixed, forward-noised static-video path.
    All-free/all-known masks are valid mathematical controls, not event evidence.
    Inputs are not modified. The supplied complete initial noise is fixed for
    the entire solve, with no new/random per-step noise draws.
    """
    if not callable(model) or not isinstance(conditions, Mapping):
        raise ValueError('A callable model and mapping conditions are required')
    for name, value in (('noise', noise), ('first', first),
                        ('static_latents', static_latents), ('sigmas', sigmas)):
        _finite_float(value, name)
    if noise.ndim != 5 or any(n <= 0 for n in noise.shape) or noise.shape[1] < 2:
        raise ValueError('noise layout must be [B,F,C,H,W] with F >= 2')
    if first.shape != (noise.shape[0], 1, *noise.shape[2:]):
        raise ValueError('first must contain exactly the first temporal slot')
    if static_latents.shape != noise.shape:
        raise ValueError('static_latents needs the complete [B,F,C,H,W] sequence')
    if any(value.dtype != noise.dtype or value.device != noise.device
           for value in (first, static_latents, sigmas)):
        raise ValueError('first/static_latents/sigmas must match noise dtype/device')
    if not torch.equal(static_latents[:, :1], first):
        raise ValueError('static_latents first slot must match the clean first latent')
    if (not isinstance(free_mask, torch.Tensor) or free_mask.ndim != 5
            or free_mask.shape[0] not in (1, noise.shape[0])
            or tuple(free_mask.shape[1:]) != (1, 1, *noise.shape[-2:])
            or free_mask.device != noise.device
            or free_mask.dtype not in (torch.bool, noise.dtype)):
        raise ValueError('free_mask must be binary [1 or B,1,1,H,W] on the noise device')
    if not bool(((free_mask == 0) | (free_mask == 1)).all()):
        raise ValueError('free_mask must be binary and finite')
    if (sigmas.ndim != 1 or sigmas.numel() < 2
            or not bool((sigmas >= 0).all() and (sigmas <= 1).all())
            or sigmas[0].item() != 1 or sigmas[-1].item() != 0
            or not bool((sigmas[1:] < sigmas[:-1]).all())):
        raise ValueError('sigmas must strictly decrease from 1 to 0')

    mask = free_mask.bool()
    fixed_noise = noise.detach()

    def project(value, sigma):
        known = (1 - sigma) * static_latents + sigma * fixed_noise
        projected = torch.where(mask, value, known)
        projected[:, :1] = first
        return projected

    current = project(noise.clone(), sigmas[0])
    for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        timestep = (sigma * 1000).expand(current.shape[:2]).clone()
        timestep[:, 0] = 0
        velocity, _ = model(
            current, conditional_dict=conditions, timestep=timestep,
            replace_first_timestep_and_noise_latents=True,
        )
        _finite_float(velocity, 'velocity')
        if (velocity.shape != noise.shape or velocity.dtype != noise.dtype
                or velocity.device != noise.device):
            raise ValueError('velocity must match noise shape/dtype/device')
        updated = current + (next_sigma - sigma) * velocity
        if not bool(torch.isfinite(updated).all()):
            raise ValueError('Nonfinite spatial Euler update')
        current = project(updated, next_sigma)
        if not bool(torch.isfinite(current).all()):
            raise ValueError('Nonfinite spatial projection')
    return current
