"""Isolated, guarded RoPE microprofile; never modifies the model inference path."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import statistics
import sys
import time

import yaml

from training.demo.backend import validate_active_authority
from training.demo.contracts import require, write_json


WARMUP, REPEATS, STEPS = 3, 10, 20


def shape_specs(config, geometry, *, default_patch_size):
    """Derive real patch grids from the checkpoint config and frozen rollout."""
    for key in ('dim', 'num_heads', 'num_layers'):
        require(type(config.get(key)) is int and config[key] > 0, f'invalid model config {key}')
    require(config['dim'] % config['num_heads'] == 0, 'dim must divide into attention heads')
    head_dim = config['dim'] // config['num_heads']
    require(head_dim % 2 == 0, 'RoPE head dimension must be even')
    patch = config.get('patch_size', default_patch_size)
    require(isinstance(patch, (list, tuple)) and len(patch) == 3
            and all(type(value) is int and value > 0 for value in patch), 'invalid patch_size')
    require(geometry.get('width') == 832 and geometry.get('height') == 480
            and geometry.get('fps') == 16 and geometry.get('chunk_future_frames') == [96, 96, 48]
            and geometry.get('chunk_latent_frames') == [25, 25, 13],
            'expected the verified832x480 six-second-window rollout geometry')
    latent_frames = geometry['chunk_latent_frames'][0]
    specs = []
    for width, height in [(geometry['width'], geometry['height']), (512, 288)]:
        # The current Action/Wan2.2 contract has spatial VAE stride16 and
        # temporal stride4:96 future RGB frames plus condition ->25 latents.
        latent = (latent_frames, height // 16, width // 16)
        require(width % 16 == height % 16 == 0
                and all(size % factor == 0 for size, factor in zip(latent, patch)),
                'latent shape does not divide by the actual model patch_size')
        grid = [size // factor for size, factor in zip(latent, patch)]
        sequence = grid[0] * grid[1] * grid[2]
        require(sequence * config['dim'] <= 32_000_000, 'microprofile tensor size exceeds its bound')
        specs.append(dict(width=width, height=height, future_seconds=6.0,
            latent_grid=list(latent), patch_size=list(patch), patch_grid=grid,
            qk_shape=[1, sequence, config['num_heads'], head_dim],
            num_layers=config['num_layers'], inference_steps=STEPS))
    return specs


def read_config(request):
    """Read small configuration files only; never load checkpoints or state."""
    source = Path(request['source_job'])
    rollout_path = source / 'rollout.yaml'
    rollout = yaml.safe_load(rollout_path.read_text(encoding='utf-8'))
    config_path = Path(rollout['lineage']['expected_base_model_path']) / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    return config, rollout['geometry'], dict(source_job=str(source),
        rollout_config=str(rollout_path), model_config=str(config_path), actual_model_config=config,
        no_weights_loaded=True, geometry_source='frozen rollout geometry; Wan2.2 VAE spatial stride16')


def rope_apply_fp32(x, grid_sizes, freqs):
    """Experimental same rotation formula, with FP32/complex64 arithmetic.

    Frequencies are supplied from the original FP64 rope_params construction.
    Their conversion is inside this function/timed path; no new theta, positions,
    caching, approximation, or model/global monkeypatch is introduced.
    """
    import torch

    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1).to(torch.complex64)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output)


def error_metrics(reference, candidate):
    import torch

    require(reference.shape == candidate.shape, 'RoPE comparison shapes differ')
    difference = reference.double() - candidate.double()
    different = (reference.to(torch.bfloat16) != candidate.to(torch.bfloat16)).sum().item()
    return dict(elements=reference.numel(), max_absolute_error=difference.abs().max().item(),
        rms_error=difference.square().mean().sqrt().item(), bf16_different_elements=int(different),
        bf16_different_fraction=different / reference.numel())


def _timing(pair_call, check_deadline):
    import torch

    for _ in range(WARMUP):
        check_deadline()
        pair_call()
        torch.cuda.synchronize()
    events_ms, wall_seconds = [], []
    for _ in range(REPEATS):
        check_deadline()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        began = time.perf_counter()
        start.record()
        result = pair_call()
        end.record()
        end.synchronize()
        wall_seconds.append(time.perf_counter() - began)
        events_ms.append(start.elapsed_time(end))
        del result
    return dict(warmup_repetitions=WARMUP, timed_repetitions=REPEATS,
        cuda_event_qk_ms=events_ms, cuda_event_qk_mean_ms=statistics.mean(events_ms),
        cuda_event_qk_median_ms=statistics.median(events_ms), cuda_event_qk_sum_ms=sum(events_ms),
        wall_qk_seconds=wall_seconds, wall_qk_mean_seconds=statistics.mean(wall_seconds),
        event_scope='current CUDA stream interval for Q+K including final BF16 casts; may include host launch gaps',
        wall_scope='Q+K host dispatch through end-event synchronization',
        kernel_duration_sum_measured=False)


def _profile_shape(spec, rope_apply, rope_params, check_deadline):
    import torch

    check_deadline()
    generator = torch.Generator(device='cpu').manual_seed(42)
    shape = spec['qk_shape']
    q = torch.randn(shape, generator=generator, dtype=torch.float32).to(device='cuda', dtype=torch.bfloat16)
    k = torch.randn(shape, generator=generator, dtype=torch.float32).to(device='cuda', dtype=torch.bfloat16)
    grid = torch.tensor([spec['patch_grid']], dtype=torch.long, device='cpu')
    d = shape[-1]
    # Exactly WanModel's double-frequency construction, without constructing it.
    freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)),
                       rope_params(1024, 2 * (d // 6)),
                       rope_params(1024, 2 * (d // 6))], dim=1).to('cuda')
    errors, timing = {}, {}
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        for name, value in [('q', q), ('k', k)]:
            check_deadline()
            original = rope_apply(value, grid, freqs)
            candidate = rope_apply_fp32(value, grid, freqs)
            errors[name] = error_metrics(original, candidate)
            del original, candidate
        for name, method in [('fp64_original', rope_apply), ('fp32_experimental', rope_apply_fp32)]:
            def pair_call():
                return (method(q, grid, freqs).to(torch.bfloat16),
                        method(k, grid, freqs).to(torch.bfloat16))
            timing[name] = _timing(pair_call, check_deadline)
    multiplier = spec['num_layers'] * STEPS
    original_ms = timing['fp64_original']['cuda_event_qk_mean_ms']
    candidate_ms = timing['fp32_experimental']['cuda_event_qk_mean_ms']
    return dict(**spec, seed=42, input_distribution='synthetic independent Gaussian, rounded toBF16; not captured activations',
        input_dtype='bfloat16', frequency_construction='original FP64 rope_params; candidate converts rotation tocomplex64',
        errors=errors, timings=timing, isolated_speed_ratio=original_ms / candidate_ms,
        optimistic_20step_window_estimate=dict(qk_calls=multiplier,
            original_seconds=original_ms * multiplier / 1000,
            candidate_seconds=candidate_ms * multiplier / 1000,
            saved_seconds=(original_ms - candidate_ms) * multiplier / 1000,
            method='isolated mean Q+K event interval times actual layers times20steps',
            validated_end_to_end=False, caveat='optimistic extrapolation, not additive attribution or an observed rollout speedup'))


def run(request, directory, lease):
    """Guarded worker entry; internal<=60s, outer watchdog also allows setup."""
    probe_seconds = request.get('probe_seconds', 60)
    require(type(probe_seconds) is int and 0 < probe_seconds <= 60,
            'RoPE microprofile requires an internal timeout of at most60 seconds')
    began = time.monotonic()
    deadline = began + probe_seconds

    def check_deadline():
        if time.monotonic() >= deadline:
            raise TimeoutError('bounded RoPE microprofile deadline reached')

    output = Path(directory) / 'profile_output' / 'rope.json'
    require(not output.exists(), 'refusing to overwrite a previous microprofile')
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status='running', mode='rope_probe', isolated_microprofile=True,
        end_to_end_measurement=False, identifies_gpu_kernel_dominance=False,
        numerical_equivalence_claim='same formula, changed arithmetic precision; not bitwise equivalence or video quality',
        actual_process=dict(executable=sys.executable, argv=list(getattr(sys, 'orig_argv', sys.argv))),
        external_timeout_seconds=request.get('max_seconds'), internal_timeout_seconds=probe_seconds,
        internal_timeout_scope='source/config reads, imports, allocation, accuracy, warmup and timing share one deadline',
        warmup=WARMUP, repetitions=REPEATS,
        model_loaded=False, weights_loaded=False, global_monkeypatch=False, shapes=[])
    try:
        validate_active_authority(lease)
        config, geometry, source = read_config(request)
        import torch
        from wan.modules.model import WanModel, rope_apply, rope_params

        patch_default = inspect.signature(WanModel.__init__).parameters['patch_size'].default
        specs = shape_specs(config, geometry, default_patch_size=patch_default)
        result.update(source=source, default_patch_size=list(patch_default), torch_version=torch.__version__)
        check_deadline()
        require(torch.cuda.is_available(), 'CUDA is required; no CPU timing fallback')
        result['device_name'] = torch.cuda.get_device_name()
        require('RTX 3090' in result['device_name'], 'this bounded microprofile targets the approvedRTX3090')
        torch.cuda.reset_peak_memory_stats()
        for spec in specs:
            validate_active_authority(lease)
            result['shapes'].append(_profile_shape(spec, rope_apply, rope_params, check_deadline))
            check_deadline()
            write_json(output, result)
        result.update(status='completed', peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      elapsed_seconds=time.monotonic() - began)
        write_json(output, result)
        return result
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic() - began)
        write_json(output, result)
        raise
