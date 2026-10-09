"""One bounded R005 text-K/V-cache equivalence/latency probe, never a default.

Exactly baseline then cached on one resident model, same first latent/noise/WJ48.
The external Creator launcher owns the GPU lease and600-second watchdog.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import time

from training.demo.contracts import contained, require, write_json
from training.demo.latency_engine import _WanRuntime
from training.studio.overlap_probe import DATA_ROOT, FPS, HEIGHT, R005_SHA, R005_SOURCE_JOB, SEED, SETTINGS, WIDTH
from training.studio.quality_probe import DEFAULT_NEGATIVE_CACHE, QualityRuntime, load_negative

FIXED = dict(mode='text_kv_probe', max_seconds=600, source_job=str(R005_SOURCE_JOB),
    checkpoint_sha256=R005_SHA, negative_cache=DEFAULT_NEGATIVE_CACHE, residency='phase', fastpath=False,
    width=WIDTH, height=HEIGHT, fps=FPS, seed=SEED, future_frames=48, **SETTINGS)


def validate_request(request):
    require(set(request) == set(FIXED) | {'job_id', 'source_revision', 'experiment_revision'},
            'text KV probe accepts only its fixed prepared request')
    require(all(request[k] == v and type(request[k]) is type(v) for k, v in FIXED.items()),
            'text KV probe fixes original R005/WJ48/40EulerCFG5/phase/600seconds')
    require(isinstance(request['job_id'], str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', request['job_id'])
            and isinstance(request['source_revision'], str) and bool(request['source_revision'])
            and request['experiment_revision'] == 'R005-scoped-text-KV-equivalence3s-v1', 'invalid probe identity')


def difference(left, right):
    """Exact CPU tensor comparison; nonfinite results never qualify."""
    import torch
    require(torch.is_tensor(left) and torch.is_tensor(right) and left.shape == right.shape
            and left.device.type == right.device.type == 'cpu' and left.dtype == right.dtype,
            'comparison requires same-shape/dtype CPU tensors')
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    return dict(shape=list(left.shape), dtype=str(left.dtype), finite=finite,
        torch_equal=finite and torch.equal(left, right),
        max_absolute_difference=float((left.float() - right.float()).abs().max()) if finite else None)


def summarize(cases, latent, decoded=None, uint8=None):
    require([row['name'] for row in cases] == ['baseline', 'cached'], 'exactly baseline then cached required')
    require(all(row['actual_model_forward_calls'] == 80 for row in cases), 'CFG must retain80 forwards per case')
    cache = cases[1]['cache_stats']
    layers = cache['layers']
    require(type(layers) is int and layers > 0 and cache['cache_bytes'] > 0 and cache['forward_calls'] == 80,
            'no real text K/V cache usage recorded')
    for name in ('positive', 'negative'):
        branch = cache['branches'][name]
        require(branch['misses'] == layers and branch['hits'] == 39 * layers and branch['active_entries'] == layers,
                'text K/V cache did not reuse one projection pair per branch/layer across40steps')
    require(cases[1]['cache_after_close']['closed'] is True and cases[1]['cache_after_close']['cache_bytes'] == 0,
            'text cache must be released before model/VAE transfer')
    equal = (all(row.get('fixed_first_latent_unchanged', True) for row in cases)
             and all(value is not None and value['finite'] and value['torch_equal'] for value in (latent, decoded, uint8)))
    return dict(status='passed_exact_values' if equal else 'comparison_failed', candidate_qualified=equal,
        latent=latent, decoded_float=decoded, displayed_uint8=uint8,
        actual_cached_projection_pairs=sum(v['misses'] for v in cache['branches'].values()),
        projection_pair_definition='one actual cache miss computes this layer text K and V (including norm_k)',
        qualification_scope='one fixed3second window only;not quality improvement or stable speedup',
        steady_speedup_verified=False, quality_improvement_verified=False, automatic_retry=False)


def sample(runtime, first, prompt, keys, noise, *, cached):
    """Timing includes conditioning and the cache scope, excludes phase transfer."""
    from training.studio.text_kv_cache import CachedTextCFGFlow
    torch = runtime.torch
    runtime.configure(SETTINGS)
    runtime.synchronize()
    torch.cuda.reset_peak_memory_stats()
    began, cache_stats, closed = time.monotonic(), None, None
    if cached:
        original = runtime.model
        # Align the cache's precision scope with the runtime's BF16 autocast.
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            with CachedTextCFGFlow(original, 5, runtime.negative.to('cuda', dtype=torch.bfloat16)) as flow:
                runtime.model = flow
                try:
                    latent = _WanRuntime.generate(runtime, first, prompt, keys, noise, 40, fastpath=False)
                    runtime.synchronize()
                    cache_stats = copy.deepcopy(flow.stats)
                    calls = flow.forward_calls
                finally:
                    runtime.model = original
            closed = copy.deepcopy(flow.stats)
    else:
        latent = runtime.generate(first, prompt, keys, noise, 40, fastpath=False)
        calls = runtime.forward_calls
    runtime.synchronize()
    seconds = time.monotonic() - began
    require(calls == 80, 'actual text-CFG forward count changed')
    return latent, dict(name='cached' if cached else 'baseline', status='completed',
        sampling_seconds=seconds, timing_kind='cuda_synchronized_host_wall_clock',
        sampling_scope='conditioning+40stepEuler+CPU_final_latent+cache_open_close;excludes_model_transfer',
        actual_model_forward_calls=calls, peak_vram_bytes=int(torch.cuda.max_memory_allocated()),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()), cache_stats=cache_stats, cache_after_close=closed)


def run(request, directory, lease):
    import numpy as np
    import torch
    from PIL import Image
    from training.demo import action_backend as action
    from training.demo.backend import validate_active_authority
    from training.studio.worker import read_source, write_clip
    validate_request(request)
    directory = contained(directory, DATA_ROOT)
    require(directory.name == request['job_id'] and directory.is_dir()
            and not any((directory / name).exists() for name in ('inputs.json', 'progress.json', 'baseline', 'cached')),
            'new probe job directory required; preserve old results')
    began = time.monotonic()

    def check():
        require(time.monotonic() - began < 570, 'bounded text KV probe deadline reached; no retry')
        validate_active_authority(lease)

    check()
    state, config, prompt, lineage, initial, source_actions, scene = read_source(request)
    require(lineage['sha256'] == R005_SHA and lineage['step'] == 1040 and scene['seed'] == SEED
            and initial.shape == (HEIGHT, WIDTH, 3) and initial.dtype == np.uint8,
            'actual loaded R005/source image differs')
    require(Path(config.base_model_path).resolve().is_relative_to(DATA_ROOT.resolve()), 'base weights must remain on data disk')
    actions = source_actions[:48].copy()
    expected = np.zeros((48, 8), np.float32)
    expected[:, [0, 5]] = 1
    require(actions.dtype == np.float32 and np.array_equal(actions, expected), 'source first48 controls are not exact WJ')
    negative, negative_source = load_negative(DEFAULT_NEGATIVE_CACHE, config.base_model_path)
    inputs = dict(mode='text_kv_probe', source_job=request['source_job'], checkpoint=lineage['path'],
        checkpoint_sha256=lineage['sha256'], checkpoint_step=1040, scene_id=scene['scene_id'], prompt=scene['prompt'],
        source_actions=scene['action_segments'], actual_actions=[dict(frames=48, keys=['W', 'J'])],
        source_revision=request['source_revision'], experiment_revision=request['experiment_revision'],
        seed=42, settings=SETTINGS, negative_prompt=negative_source, width=WIDTH, height=HEIGHT, fps=FPS,
        future_frames=48, total_frames=49, residency='phase', fastpath=False,
        run_order=['baseline', 'cached'], first_latent_encode_count=1,
        model_resident_during_both_sampling_calls=True,
        timing_caveat='baseline runs first;cached runs second with warm kernels/allocator;one pair is not steady-state speed evidence',
        new_training=False, default_demo_changed=False, quality_improvement_verified=False)
    Image.fromarray(initial).save(directory / 'initial.png')
    np.save(directory / 'actions.npy', actions, allow_pickle=False)
    write_json(directory / 'inputs.json', inputs)
    runtime, rows, latents, decoded, displayed = None, [], [], [], []
    comparison, cold, encode_seconds, transfer_seconds = None, None, None, None
    try:
        check()
        tick = time.monotonic()
        runtime = QualityRuntime(state, config, 'phase', negative)
        runtime.synchronize()
        cold = time.monotonic() - tick
        tick = time.monotonic()
        runtime.transfer('vae')
        runtime.begin()
        pixel = torch.from_numpy(initial.copy()).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).float().div(127.5).sub(1)
        first = runtime.encode(pixel).to(torch.bfloat16)
        runtime.synchronize()
        encode_seconds = time.monotonic() - tick
        require(tuple(first.shape) == (1, 1, 48, 30, 52) and bool(torch.isfinite(first).all()), 'invalid single encoded condition')
        noise = torch.randn((1, 13, 48, 30, 52), generator=torch.Generator(device='cpu').manual_seed(42), dtype=torch.float32)
        keys = torch.from_numpy(actions).unsqueeze(0)
        inputs.update(first_latent_sha256_float32=action.tensor_sha(first), noise_sha256_float32=action.tensor_sha(noise),
            noise_policy='original first CPU FP32 13slot draw from generator seed42;same object passed to both paths')
        write_json(directory / 'inputs.json', inputs)
        tick = time.monotonic()
        runtime.transfer('model')
        runtime.synchronize()
        transfer_seconds = time.monotonic() - tick
        for cached in (False, True):
            check()
            latent, row = sample(runtime, first, prompt, keys, noise, cached=cached)
            require(tuple(latent.shape) == (1, 13, 48, 30, 52) and latent.device.type == 'cpu', 'sampler changed latent geometry/device')
            row['fixed_first_latent_unchanged'] = torch.equal(latent[:, :1], first)
            latents.append(latent)
            rows.append(row)
            (directory / row['name']).mkdir(exist_ok=False)
            write_json(directory / row['name'] / 'result.json', row)
            write_json(directory / 'progress.json', dict(status='running', phase='sampling', cases=rows))
        latent_comparison = difference(*latents)
        comparison = summarize(rows, latent_comparison)
        if latent_comparison['torch_equal']:
            check()  # Cache is closed here, before either model/VAE transfer.
            runtime.transfer('vae')
            runtime.synchronize()
            for latent, row in zip(latents, rows):
                check()
                runtime.vae.model.clear_cache()
                tick = time.monotonic()
                pixels = runtime.decode(latent)
                runtime.synchronize()
                row['decode_seconds'] = time.monotonic() - tick
                require(tuple(pixels.shape) == (1, 49, 3, HEIGHT, WIDTH) and pixels.device.type == 'cpu'
                        and bool(torch.isfinite(pixels).all()) and pixels.min() >= -1 and pixels.max() <= 1,
                        'decoder did not return finite49 CPU RGB frames')
                decoded.append(pixels)
                future = pixels[0, 1:].permute(0, 2, 3, 1).add(1).mul(127.5).round().clamp(0, 255).byte().numpy()
                rgb = np.concatenate((initial[None], future))
                displayed.append(torch.from_numpy(rgb))
                target = directory / row['name'] / 'raw.mp4'
                write_clip(target, rgb)
                row.update(raw_video=str(target), frames_written=49, initial_output='actual_submitted_RGB', raw_video_finalized=True)
                write_json(directory / row['name'] / 'result.json', row)
            comparison = summarize(rows, latent_comparison, difference(*decoded), difference(*displayed))
        write_json(directory / 'comparison.json', comparison)
    finally:
        if runtime is not None:
            runtime.close()
    result = dict(status='completed', mode='text_kv_probe', cases=rows, comparison=comparison,
        candidate_qualified=comparison['candidate_qualified'], comparison_failed=not comparison['candidate_qualified'],
        load_seconds=cold, load_seconds_scope='one CPU runtime construction and CUDA context/synchronization',
        shared_initial_encode_seconds=encode_seconds, shared_transfer_to_model_seconds=transfer_seconds,
        elapsed_seconds=time.monotonic() - began, run_order=['baseline', 'cached'],
        timing_caveat=inputs['timing_caveat'], no_long_video_test=True, new_training=False,
        default_demo_changed=False, realtime=False, quality_improvement_verified=False, automatic_retry=False)
    write_json(directory / 'progress.json', result)
    return result


def prepare(*, output, source_revision):
    directory = contained(output, DATA_ROOT)
    request = dict(FIXED, job_id=directory.name, source_revision=source_revision,
                   experiment_revision='R005-scoped-text-KV-equivalence3s-v1')
    validate_request(request)
    require(directory != DATA_ROOT.resolve() and not directory.exists(), 'new output directory required')
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'request.json', request)
    return dict(status='cpu_request_prepared', request=str(directory / 'request.json'), gpu_launched=False,
                model_loaded=False, max_seconds=600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), ensure_ascii=False))


if __name__ == '__main__':
    main()
