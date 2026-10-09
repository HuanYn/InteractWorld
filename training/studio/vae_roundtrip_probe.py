"""Bounded VAE-only reconstruction diagnostic, not video generation or quality evaluation.

Two already saved stills, six serial encode/decode cycles each. This isolates
reconstruction loss; it does not simulate diffusion continuation or motion.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np

from training.demo.backend import validate_active_authority
from training.demo.contracts import contained, require, sha256, write_json

DATA_ROOT = Path('/DATA_DISK/InteractWorld')
SHAPE = (1, 3, 1, 480, 832)


def pixels(value):
    if value.dtype == np.uint8 and value.shape == (480, 832, 3):
        return value.transpose(2, 0, 1)[None, :, None].astype(np.float32) / 127.5 - 1
    require(value.dtype == np.float32 and value.shape == SHAPE
            and np.isfinite(value).all() and value.min() >= -1 and value.max() <= 1,
            'expected saved float32 endpoint or original RGB input')
    return value.copy()


def difference(reference, current):
    require(reference.shape == current.shape and np.isfinite(current).all(), 'invalid reconstruction')
    delta = current.astype(np.float64) - reference.astype(np.float64)
    rmse = float(np.sqrt(np.mean(delta ** 2)))
    return dict(mae_255=float(np.mean(np.abs(delta)) * 127.5), rmse_255=rmse * 127.5,
                psnr_db=None if rmse == 0 else 20 * math.log10(2 / rmse),
                exact_match=rmse == 0, max_absolute_255=float(np.abs(delta).max() * 127.5))


def run(request, directory, lease):
    from PIL import Image
    from training.demo.action_backend import VAE_SHA256
    from training.demo.relocation import ArtifactHashes

    began = time.monotonic()
    require(request.get('max_seconds') == 180 and request.get('cycles') == 6,
            'this diagnostic is fixed at six cycles / 180 seconds')
    directory = contained(directory, DATA_ROOT)
    require(not (directory / 'roundtrip.json').exists(), 'do not overwrite previous evidence')
    vae_path = contained(request['vae_path'], DATA_ROOT)
    require(vae_path.name == 'Wan2.2_VAE.pth', 'expected the existing Wan VAE')
    digest = ArtifactHashes(contained(request['verification_cache'], DATA_ROOT), protected_paths=[vae_path])
    require(digest(vae_path) == VAE_SHA256, 'VAE does not match the verified production artifact')
    source = contained(request['source_job'], DATA_ROOT)
    source_request = json.loads((source / 'request.json').read_text())
    require(sha256(source / 'initial.npy') == source_request['initial_sha256'], 'source image changed')
    segment = contained(request['source_segment'], DATA_ROOT)
    job = json.loads((segment / 'job.json').read_text())
    require(job['segment_id'] == segment.name and job['status'] == 'completed'
            and job['generation']['outcome'] == 'completed', 'endpoint source is not a completed real segment')
    cases = [('initial', source / 'initial.npy'), ('generated_endpoint', segment / 'endpoint.npy')]
    result = dict(status='running', mode='vae_roundtrip_probe', cycles=6, source_revision=request.get('source_revision'),
        vae_path=str(vae_path), vae_sha256=VAE_SHA256, cases=[], new_video_generation=False,
        diffusion_loaded=False, text_encoder_loaded=False, quality_claim='none',
        limitation='Serial still reconstruction only; not an actual motion rollout or proof of a drift cause.',
        metric_scope='float RGB reconstruction vs the saved still; pixel units 0..255; PSNR range255; exact PSNR stored null')
    def check():
        require(time.monotonic() - began < 160, 'internal diagnostic deadline reached')
        validate_active_authority(lease)

    import torch
    from utils.wan_wrapper import WanVAEWrapper
    check()
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one leased GPU required')
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    vae = None
    try:
        load_start = time.monotonic()
        vae = WanVAEWrapper(pretrained_path=str(vae_path)).requires_grad_(False).eval().to('cuda', dtype=torch.bfloat16)
        torch.cuda.synchronize()
        result['load_seconds'] = time.monotonic() - load_start
        for name, source_path in cases:
            check()
            reference = pixels(np.load(source_path, allow_pickle=False))
            current = reference.copy()
            target = directory / name
            target.mkdir(exist_ok=False)
            record = dict(name=name, source=str(source_path), source_kind='original_RGB' if name=='initial' else 'unquantized_generated_float_RGB', rounds=[])
            def save(value, cycle):
                rgb = np.clip(np.rint((value[0, :, 0].transpose(1, 2, 0) + 1) * 127.5), 0, 255).astype(np.uint8)
                Image.fromarray(rgb).save(target / f'cycle-{cycle}.png')
            save(reference, 0)
            for cycle in range(1, 7):
                check()
                previous = current
                torch.cuda.synchronize()
                tick = time.monotonic()
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    latent = vae.encode_to_latent(torch.from_numpy(current).to('cuda', dtype=torch.bfloat16))
                    require(tuple(latent.shape) == (1, 1, 48, 30, 52), 'single-frame VAE latent mismatch')
                    decoded = vae.decode_to_pixel(latent.to(dtype=torch.bfloat16), use_cache=False, return_in_cpu=True)
                    current = pixels(decoded.permute(0, 2, 1, 3, 4).float().numpy())
                torch.cuda.synchronize()
                record['rounds'].append(dict(cycle=cycle, seconds=time.monotonic()-tick,
                    vs_original=difference(reference, current), vs_previous=difference(previous, current)))
                if cycle in (1, 3, 6): save(current, cycle)
                del latent, decoded
            result['cases'].append(record)
            write_json(directory / 'roundtrip.json', result)
        result.update(status='completed', elapsed_seconds=time.monotonic()-began,
                      peak_vram_bytes=torch.cuda.max_memory_allocated())
        write_json(directory / 'roundtrip.json', result)
        return result
    finally:
        del vae
        if torch.cuda.is_initialized(): torch.cuda.empty_cache()
