"""Isolated R005 one-versus-five RGB history probe; never a Studio default.

The external Creator launcher owns the lease and 900-second watchdog. This
module does not acquire a GPU, deploy code, retry a case, or promote a model.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import numpy as np

from training.demo.contracts import require, write_json
from training.demo.latency_engine import ReusableActionEngine
from training.studio.quality_probe import (
    DEFAULT_NEGATIVE_CACHE, QualityRuntime, TextCFGFlow, annotate_report, load_negative,
)

WIDTH, HEIGHT, FPS, SEED, STEPS = 832, 480, 16, 42, 40
DATA_ROOT = Path('/DATA_DISK/InteractWorld')
R005_SHA = '9bc9f57cce1095cbbc472307bc3963da2e6153e9f0f19080b542b6bf118dfaee'
R005_SOURCE_JOB = DATA_ROOT / 'creator-runtime/jobs/9be8bfe3ef4a499d9e9b1847a80af7fa'
REVERSE_PARENT_JOB = DATA_ROOT / 'creator-runtime/native97-20260928/native97-overlap5-0929'
SETTINGS = dict(steps=STEPS, cfg_scale=5, lora_strength=1, action_scale=.03,
                solver='flow_euler')


def overlap_noise(seed, height, width):
    """CPU FP32 baseline13/candidate14 noises; identical 12 future slots.

Height and width are RGB dimensions. The candidate's two unused condition
slots are zero placeholders, not extra random draws.
"""
    import torch
    require(type(seed) is int and 0 <= seed <= 2**32 - 1, 'invalid noise seed')
    require(all(type(value) is int and value > 0 and value % 16 == 0
                for value in (height, width)), 'RGB dimensions must be multiples of16')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    baseline = torch.randn((1, 13, 48, height // 16, width // 16),
                           generator=generator, dtype=torch.float32, device='cpu')
    candidate = torch.cat([torch.zeros_like(baseline[:, :2]), baseline[:, 1:]], dim=1)
    require(torch.equal(candidate[:, 2:], baseline[:, 1:]), 'future noise mapping differs')
    return baseline, candidate


def prefix_euler(model, *, first, noise, conditions, sigmas):
    """Original-dtype Euler, fixed K-latent prefix and per-frame t[:K]=0.

This generalizes only the original sampler's clean-slot count. K=1 preserves
the arithmetic of action_backend.euler_rollout. It creates no attention mask,
KV history, FP32 accumulator, extra model, or new learned parameters.
"""
    import torch
    require(torch.is_tensor(first) and torch.is_tensor(noise)
            and first.ndim == noise.ndim == 5 and first.shape[0] == noise.shape[0]
            and first.shape[2:] == noise.shape[2:] and 0 < first.shape[1] < noise.shape[1],
            'prefix/noise must be aligned [B,T,C,H,W] with nonempty future')
    require(first.is_floating_point() and noise.is_floating_point()
            and first.dtype == noise.dtype and first.device == noise.device
            and bool(torch.isfinite(first).all()) and bool(torch.isfinite(noise).all()),
            'prefix/noise must be finite and share device/dtype')
    require(torch.is_tensor(sigmas) and sigmas.ndim == 1 and len(sigmas) >= 2
            and sigmas.device == noise.device and sigmas.dtype == noise.dtype
            and bool(torch.isfinite(sigmas).all())
            and bool((sigmas[:-1] > sigmas[1:]).all()) and sigmas[-1] >= 0,
            'Euler sigmas must be finite descending values on noise device/dtype')
    count = first.shape[1]
    current = noise.clone()
    current[:, :count] = first
    for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        timestep = (sigma * 1000).expand(current.shape[:2]).clone()
        timestep[:, :count] = 0
        velocity, _ = model(current, conditional_dict=conditions, timestep=timestep,
                            replace_first_timestep_and_noise_latents=True)
        require(velocity.shape == current.shape and bool(torch.isfinite(velocity).all()),
                'invalid overlap model velocity')
        current = current + (next_sigma - sigma) * velocity
        current[:, :count] = first
        require(bool(torch.isfinite(current).all()), 'nonfinite overlap Euler state')
    return current


def overlap_actions(parent_actions, future_actions):
    """Return past four destination-frame keys +48 new keys, without mutation."""
    for label, value in [('parent', parent_actions), ('future', future_actions)]:
        require(isinstance(value, np.ndarray) and value.dtype == np.float32
                and value.ndim == 2 and value.shape[1] == 8
                and np.isfinite(value).all() and np.isin(value, [0, 1]).all(),
                f'{label} actions must be binary finite float32 [N,8]')
        require(not any(np.any((value[:, left] != 0) & (value[:, right] != 0))
                        for left, right in [(0, 2), (1, 3), (4, 6), (5, 7)]),
                f'{label} actions contain opposite keys')
    require(len(parent_actions) >= 4 and len(future_actions) == 48,
            'overlap requires four historical and48 new action frames')
    return np.concatenate([parent_actions[-4:], future_actions]).copy()


def overlap_rgb(history, pixels):
    """Render original historical endpoint +48 new decoded frames, not overlap."""
    import torch
    require(torch.is_tensor(history) and torch.is_tensor(pixels)
            and history.device.type == pixels.device.type == 'cpu'
            and history.ndim == pixels.ndim == 5 and tuple(history.shape[:3]) == (1, 3, 5)
            and tuple(pixels.shape[:3]) == (1, 53, 3)
            and history.shape[-2:] == pixels.shape[-2:], 'invalid five-frame RGB/decode shape')
    for value in (history, pixels):
        require(value.is_floating_point() and bool(torch.isfinite(value).all())
                and value.min() >= -1 and value.max() <= 1, 'RGB must be finite float in[-1,1]')
    initial = history[0, :, -1].permute(1, 2, 0).unsqueeze(0)
    future = pixels[0, 5:].permute(0, 2, 3, 1)
    return torch.cat([initial, future]).add(1).mul(127.5).round().clamp(0, 255).byte().numpy()


def load_reverse_history(request, *, lineage, scene, negative_source):
    """Read the single settled parent on the data disk; never regenerate/copy it."""
    import torch
    parent = Path(request.get('parent_job', '')).resolve()
    require(parent == REVERSE_PARENT_JOB.resolve(), 'reverse parent_job is not the pinned overlap job')

    def read(name):
        return json.loads((parent / name).read_text(encoding='utf-8'))

    original, inputs, result = read('request.json'), read('inputs.json'), read('result.json')
    settlement, lease, root = read('settlement.json'), read('lease.json'), read('root3/result.json')
    require(original.get('job_id') == lease.get('job_id') == parent.name
            and original.get('mode') == result.get('mode') == inputs.get('mode') == 'overlap_probe'
            and result.get('status') == 'completed', 'parent overlap job is not completed')
    require(settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed'
            and bool(lease.get('reservation_id'))
            and settlement.get('reservation_id') == lease['reservation_id'], 'parent overlap job is not settled')
    cases = result.get('cases', [])
    require([case.get('name') for case in cases] == ['root3', 'baseline1', 'overlap5']
            and all(case.get('status') == 'completed' for case in cases)
            and cases[0] == root, 'parent cases/root record differs')
    source = R005_SOURCE_JOB.resolve()
    require(all(Path(value).resolve() == source for value in
                (request['source_job'], original['source_job'], inputs['source_job'])),
            'reverse source_job differs from the pinned R005 input source')
    require(original.get('checkpoint_sha256') == inputs.get('checkpoint_sha256')
            == result.get('checkpoint_sha256') == request.get('checkpoint_sha256') == lineage['sha256'] == R005_SHA
            and inputs.get('checkpoint') == lineage['path']
            and inputs.get('checkpoint_step') == lineage['step'] == 1040, 'parent checkpoint binding differs')
    require(inputs.get('scene_id') == scene['scene_id'] and inputs.get('prompt') == scene['prompt']
            and inputs.get('seed') == inputs.get('source_seed') == scene['seed'] == SEED
            and Path(inputs['source_initial_npy']).resolve() == source / 'initial.npy'
            and inputs.get('negative_prompt') == negative_source, 'parent scene/seed/prompt/input binding differs')
    require(inputs.get('settings') == SETTINGS and inputs.get('width') == WIDTH
            and inputs.get('height') == HEIGHT and inputs.get('fps') == FPS
            and inputs.get('frames_per_case') == 49 and inputs.get('root_keys') == ['W', 'J']
            and inputs.get('future_keys') == ['W', 'L'], 'parent geometry/settings/actions differ')
    history_path, root_video = parent / 'history.npy', parent / 'root3/raw.mp4'
    require(Path(inputs['history_file']).resolve() == history_path
            and Path(result['history']).resolve() == history_path
            and root.get('name') == 'root3' and root.get('status') == 'completed'
            and root.get('seed') == SEED and root.get('decoded_rgb_frames') == 49
            and root.get('actual_model_forward_calls') == STEPS * 2
            and Path(root['raw_video']).resolve() == root_video
            and root_video.is_file() and root_video.stat().st_size > 0, 'parent root/history provenance differs')
    generation = root.get('generation', {})
    require(generation.get('outcome') == 'completed' and generation.get('frames_written') == 49
            and generation.get('future_frames') == 48 and generation.get('width') == WIDTH
            and generation.get('height') == HEIGHT and generation.get('fps') == FPS,
            'parent root generation frame contract differs')
    history = np.load(history_path, allow_pickle=False)
    require(history.dtype == np.float32 and history.shape == (1, 3, 5, HEIGHT, WIDTH)
            and np.isfinite(history).all() and history.min() >= -1 and history.max() <= 1,
            'parent history must be finite float32 [1,3,5,H,W]')
    endpoint = np.load(parent / 'root3/endpoint.npy', allow_pickle=False)
    require(endpoint.dtype == np.float32 and endpoint.shape == (1, 3, 1, HEIGHT, WIDTH)
            and np.array_equal(endpoint, history[:, :, -1:]), 'parent historical endpoint differs')
    actions = np.load(parent / 'root3/actions.npy', allow_pickle=False)
    expected = np.zeros((48, 8), dtype=np.float32)
    expected[:, [0, 5]] = 1
    require(actions.dtype == np.float32 and np.array_equal(actions, expected), 'parent root actions must be48 WJ')
    metadata = dict(parent_job=str(parent), root_video=str(root_video), root_reused=True,
        root_generated_this_job=False, parent_history_verified=True, parent_history_written=False,
        parent_settlement=str(parent / 'settlement.json'),
        parent_reservation_id=settlement['reservation_id'],
        parent_future_noise_sha256_float32=inputs['noise_comparison']['future_noise_sha256_float32'])
    return torch.from_numpy(history.copy()), actions.copy(), metadata


class _OverlapRuntime(QualityRuntime):
    """Keep the exact parent quality path for one-slot root/baseline calls."""

    def generate(self, first, prompt, keys, noise, steps, *, fastpath=False):
        self.seen_noise = noise.detach().cpu().clone()
        if first.shape[1] == 1:
            return super().generate(first, prompt, keys, noise, steps, fastpath=fastpath)
        import torch
        from training.longforcing_lite import euler_sigmas
        from training.models.action_adapter import build_action_context
        require(first.shape[1] == 2 and noise.shape[1] == 14 and keys.shape[1] == 52
                and steps == STEPS and fastpath is False and self.cfg_scale == 5
                and self.solver == 'flow_euler', 'unexpected five-frame overlap configuration')
        condition = first.to(device='cuda', dtype=torch.bfloat16)
        conditions = dict(prompt_embeds=[prompt.to(device='cuda', dtype=torch.bfloat16)],
            act_context_scale=self.config.action_scale,
            act_context=build_action_context(keys, height=HEIGHT, width=WIDTH,
                                            device='cuda', dtype=torch.bfloat16))
        sigmas = euler_sigmas(STEPS, shift=5.0, device=torch.device('cuda'), dtype=torch.bfloat16)
        flow = TextCFGFlow(self.model, 5, self.negative.to(device='cuda', dtype=torch.bfloat16))
        try:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                result = prefix_euler(flow, first=condition,
                    noise=noise.to(device='cuda', dtype=torch.bfloat16), conditions=conditions, sigmas=sigmas)
            return result.detach().cpu()
        finally:
            self.forward_calls += flow.forward_calls


class _HistoryEngine(ReusableActionEngine):
    """Only this experimental engine retains float RGB before uint8 encoding."""

    def _decode(self, latent, index):
        pixels = super()._decode(latent, index)
        self.history_tail = pixels[:, -5:].detach().permute(0, 2, 1, 3, 4).float().cpu().contiguous().clone()
        return pixels


def _run_overlap(engine, history, prompt, actions, noise, output):
    import torch
    from training.demo.action_backend import tensor_sha
    require(not output.exists(), 'refusing to overwrite overlap video')
    require(engine._lock.acquire(blocking=False), 'experimental engine already running')
    writer = None
    began = time.monotonic()
    engine._timings, engine._steps = [], STEPS
    engine._runs += 1
    report = dict(outcome='failed', method='experimental_five_RGB_two_clean_latents_v1',
        width=WIDTH, height=HEIGHT, fps=FPS, frames_written=0,
        historical_rgb_frames=5, clean_latents=2, total_latents=14,
        decoded_rgb_frames=53, future_frames=48, future_duration_seconds=3,
        discarded_decoded_history_frames=5, displayed_prefix='original_parent_float_RGB_endpoint',
        action_rows=52, historical_action_rows=4, action_alignment='slot1=past4;slots2:14=new48',
        local_temporal_positions=list(range(14)), attention='unchanged_noncausal;no_KV_or_reference_mask',
        sampler=dict(solver='flow_euler', steps=STEPS, shift=5, cfg_scale=5),
        solver_arithmetic='original_BF16_Euler_no_FP32_accumulation',
        residency=engine.residency, fastpath=False, model_load_count=1, model_reused=True,
        timing_kind=engine._runtime.timing_kind, load_seconds=engine.load_seconds,
        elapsed_seconds_scope='candidate_encode_sample_decode_write;excludes_shared_cold_load',
        realtime=False, world_state=False, quality_evaluation='pending_visual_comparison')
    engine.last_run_report = report
    try:
        require(not engine._closed, 'experimental engine is closed')
        engine._runtime.begin()
        writer = engine._timed('raw_writer_open', lambda: engine._writer_factory(output, WIDTH, HEIGHT, FPS))
        first = engine._encode(history.clone(), -1).to(dtype=torch.bfloat16)
        require(tuple(first.shape) == (1, 2, 48, 30, 52), 'five-frame VAE condition must have2 latents')
        report['condition_latent_sha256_float32'] = tensor_sha(first)
        latent = engine._generate(first, prompt, torch.from_numpy(actions.copy()).unsqueeze(0), noise, 0)
        require(tuple(latent.shape) == (1, 14, 48, 30, 52)
                and torch.equal(latent[:, :2], first), 'overlap sampler modified the clean prefix')
        pixels = engine._decode(latent, 0)
        frames = overlap_rgb(history, pixels)
        require(frames.shape == (49, HEIGHT, WIDTH, 3), 'overlap display frame count changed')
        engine._timed('raw_write', lambda: writer.write(frames))
        engine._timed('raw_finalize', writer.close)
        report.update(outcome='completed', frames_written=49)
        return report, engine._endpoint
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        if writer is not None:
            try:
                writer.abort()
            except Exception:
                pass
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - began
        report['stage_timings'] = list(engine._timings)
        totals = {}
        for row in engine._timings:
            totals[row['stage']] = totals.get(row['stage'], 0.0) + row['elapsed_seconds']
        report['stage_totals_seconds'] = totals
        try:
            report['peak_vram_bytes'] = engine._runtime.peak_bytes()
        finally:
            engine._lock.release()


def run(request, directory, lease):
    """Original root+WL pair, or a WJ pair from the one explicitly pinned parent."""
    import torch
    from PIL import Image
    from training.demo.action_backend import tensor_sha
    from training.demo.backend import validate_active_authority
    from training.studio.worker import read_source

    began = time.monotonic()
    mode = request.get('mode')
    reverse = mode == 'overlap_reverse_probe'
    maximum = 600 if reverse else 900
    next_case_deadline = maximum - 60
    require(mode in ('overlap_probe', 'overlap_reverse_probe') and request.get('max_seconds') == maximum,
            'overlap mode requires its fixed900/600-second external watchdog')
    require(request.get('checkpoint_sha256') == R005_SHA, 'overlap probe is pinned to original R005')
    for field, value in dict(width=WIDTH, height=HEIGHT, fps=FPS, seed=SEED, **SETTINGS).items():
        require(field not in request or request[field] == value, f'overlap probe fixes {field}={value}')
    require(request.get('fastpath', False) is False and request.get('residency', 'co_resident') == 'co_resident',
            'overlap probe fixes no-fastpath co-resident execution')
    require('action_adapter_donor_source' not in request and 'cases' not in request,
            'overlap probe does not accept hybrid weights or configurable cases')
    directory = Path(directory).resolve()
    require(directory.is_dir() and directory.is_relative_to(DATA_ROOT.resolve()) and directory != DATA_ROOT.resolve(),
            'overlap outputs must be in a new job below the project data root')
    if reverse:
        require(directory == REVERSE_PARENT_JOB.resolve().parent / 'native97-overlap5-reverse-0929',
                'reverse outputs must use the isolated pinned reverse job directory')
    owned = ['inputs.json', 'initial.png', 'history.npy', 'progress.json', 'overlap-failure.json',
             'root3', 'baseline1', 'overlap5']
    require(not any((directory / name).exists() for name in owned), 'refusing to overwrite overlap evidence')
    require(Path(request['source_job']).resolve().is_relative_to(DATA_ROOT.resolve()), 'source must remain on data disk')
    negative_path = Path(request.get('negative_cache', DEFAULT_NEGATIVE_CACHE)).resolve()
    require(negative_path.is_relative_to(DATA_ROOT.resolve()), 'negative cache must remain on data disk')
    validate_active_authority(lease)
    state, config, prompt, lineage, initial, _, scene = read_source(request)
    require(lineage['sha256'] == request['checkpoint_sha256'] == R005_SHA,
            'verified source checkpoint is not the pinned original R005')
    require(scene['seed'] == SEED and initial.shape == (HEIGHT, WIDTH, 3)
            and initial.dtype == np.uint8, 'source must preserve seed42 and original832x480 RGB')
    require(Path(config.base_model_path).resolve().is_relative_to(DATA_ROOT.resolve()), 'base weights must stay on data disk')
    negative, negative_source = load_negative(negative_path, config.base_model_path)
    history, parent_metadata = None, {}
    history_path = directory / 'history.npy'
    root_actions = np.zeros((48, 8), dtype=np.float32)
    root_actions[:, [0, 5]] = 1
    if reverse:
        history, root_actions, parent_metadata = load_reverse_history(
            request, lineage=lineage, scene=scene, negative_source=negative_source)
        history_path = Path(parent_metadata['parent_job']) / 'history.npy'
    new_actions = np.zeros((48, 8), dtype=np.float32)
    new_actions[:, [0, 5 if reverse else 7]] = 1
    historical_actions = overlap_actions(root_actions, new_actions)
    baseline_noise, candidate_noise = overlap_noise(SEED, HEIGHT, WIDTH)
    noise_record = dict(seed=SEED, baseline_shape=list(baseline_noise.shape), candidate_shape=list(candidate_noise.shape),
        source='one_CPU_FP32_13slot_draw;candidate_prepends2_fixed_placeholders_to_same12_future_slots',
        future_equal_fp32=bool(torch.equal(baseline_noise[:, 1:], candidate_noise[:, 2:])),
        future_equal_bf16=bool(torch.equal(baseline_noise[:, 1:].bfloat16(), candidate_noise[:, 2:].bfloat16())),
        future_noise_sha256_float32=tensor_sha(baseline_noise[:, 1:]),
        baseline_full_noise_sha256_float32=tensor_sha(baseline_noise),
        candidate_full_noise_sha256_float32=tensor_sha(candidate_noise))
    if reverse:
        require(noise_record['future_noise_sha256_float32'] == parent_metadata['parent_future_noise_sha256_float32'],
                'reverse future noise differs from completed WL comparison')
    inputs = dict(mode=mode, source_job=request['source_job'], scene_id=scene['scene_id'],
        prompt=scene['prompt'], seed=SEED, source_seed=scene['seed'], negative_prompt=negative_source,
        checkpoint=lineage['path'], checkpoint_sha256=lineage['sha256'], checkpoint_step=lineage['step'],
        source_revision=request.get('source_revision'), experiment_revision=request.get('experiment_revision'),
        width=WIDTH, height=HEIGHT, fps=FPS, settings=SETTINGS, residency='co_resident', fastpath=False,
        root_keys=['W', 'J'], future_keys=['W', 'J' if reverse else 'L'], frames_per_case=49,
        source_initial_npy=str(Path(request['source_job']) / 'initial.npy'),
        root_source='reused_settled_parent_root3;no_new_root' if reverse else 'new_original_RGB_generation;old_endpoint_stores_only1_frame',
        history_source='root3 decoded float32 RGB frames44:49 before uint8/MP4;independently_VAE_reencoded',
        history_file=str(history_path), historical_action_source=str(Path(parent_metadata['parent_job']) / 'root3/actions.npy') + '[-4:]' if reverse else 'root3 actions[-4:]',
        history_conditioning_trained=False, default_demo_changed=False, new_training=False,
        noise_comparison=noise_record, external_timeout_seconds=maximum, no_new_case_after_seconds=next_case_deadline,
        input_loading_seconds=time.monotonic() - began,
        limitations='two-clean-slot conditioning is unvalidated;5frames cover0.25s,not persistent world state;no quality claim',
        **parent_metadata)
    write_json(directory / 'inputs.json', inputs)
    Image.fromarray(initial).save(directory / 'initial.png')
    runtime = engine = None
    results, active_case = [], None
    try:
        validate_active_authority(lease)
        require(time.monotonic() - began < next_case_deadline, 'deadline reached before model load')
        loaded = time.monotonic()
        runtime = _OverlapRuntime(state, config, 'co_resident', negative)
        runtime.synchronize()
        cold_seconds = time.monotonic() - loaded
        engine = _HistoryEngine(state, config, residency='co_resident', fastpath=False, _runtime=runtime)
        engine.load_seconds = cold_seconds
        for name in (('baseline1', 'overlap5') if reverse else ('root3', 'baseline1', 'overlap5')):
            require(time.monotonic() - began < next_case_deadline, 'deadline reached; no next overlap case started')
            validate_active_authority(lease)
            active_case = name
            target = directory / name
            target.mkdir(exist_ok=False)
            runtime.configure(SETTINGS)
            engine.action_scale = SETTINGS['action_scale']
            engine.last_run_report = None
            write_json(directory / 'progress.json', dict(status='running', active_case=name, cases=results))
            started = time.monotonic()
            try:
                actions = root_actions if name == 'root3' else historical_actions if name == 'overlap5' else new_actions
                np.save(target / 'actions.npy', actions, allow_pickle=False)
                if name == 'overlap5':
                    report, endpoint = _run_overlap(engine, history, prompt, actions, candidate_noise, target / 'raw.mp4')
                else:
                    condition = initial if name == 'root3' else history[:, :, -1:].clone()
                    report, endpoint = engine.run_segment(initial=condition, prompt=prompt, actions=actions.copy(),
                        seed=SEED, output_path=target / 'raw.mp4', steps=STEPS)
                annotate_report(report, SETTINGS, runtime, negative_source, cold_seconds)
                require(runtime.forward_calls == STEPS * 2, 'actual CFG forwards must equal40*2 per case')
                expected_noise = candidate_noise if name == 'overlap5' else baseline_noise
                require(torch.equal(runtime.seen_noise, expected_noise), 'actual sampler input differs from fixed CPU noise')
                if name == 'root3':
                    history = engine.history_tail.clone()
                    require(tuple(history.shape) == (1, 3, 5, HEIGHT, WIDTH)
                            and torch.equal(history[:, :, -1:], endpoint), 'captured root history endpoint differs')
                    np.save(history_path, history.numpy(), allow_pickle=False)
                else:
                    require(torch.equal(runtime.seen_noise[:, 2:] if name == 'overlap5' else runtime.seen_noise[:, 1:],
                                        baseline_noise[:, 1:]), 'actual branch future noise differs')
                    require(np.array_equal(np.load(history_path, allow_pickle=False), history.numpy()),
                            'parent float history changed during branch generation')
                np.save(target / 'endpoint.npy', endpoint.numpy(), allow_pickle=False)
                entry = dict(name=name, status='completed', seed=SEED, generation=report,
                    raw_video=str(target / 'raw.mp4'), actions=str(target / 'actions.npy'),
                    initial_source=str(directory / 'initial.png') if name == 'root3' else str(history_path),
                    clean_history_rgb_frames=1 if name != 'overlap5' else 5,
                    decoded_rgb_frames=53 if name == 'overlap5' else 49,
                    actual_model_forward_calls=runtime.forward_calls, elapsed_seconds=time.monotonic() - started,
                    elapsed_seconds_scope='case_including_small_evidence_writes;excludes_shared_cold_load',
                    noise_verified_equal=True, future_noise_sha256_float32=noise_record['future_noise_sha256_float32'],
                    quality_evaluation='pending_visual_comparison', realtime=False)
                write_json(target / 'result.json', entry)
                results.append(entry)
                write_json(directory / 'progress.json', dict(status='running', active_case=None, cases=results))
                active_case = None
            except BaseException as error:
                failure = dict(name=name, status='failed', error=f'{type(error).__name__}: {error}',
                    elapsed_seconds=time.monotonic() - started, generation=copy.deepcopy(engine.last_run_report),
                    actual_model_forward_calls=runtime.forward_calls, retry=False)
                write_json(target / 'result.json', failure)
                raise
    except BaseException as error:
        failure = dict(status='failed', active_case=active_case, cases=results,
            error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic() - began,
            retry=False, completed_cases_preserved=True, quality_improvement_verified=False)
        write_json(directory / 'overlap-failure.json', failure)
        write_json(directory / 'progress.json', failure)
        raise
    finally:
        if engine is not None:
            engine.close()
        elif runtime is not None:
            runtime.close()
    result = dict(status='completed', mode=mode, cases=results, noise_comparison=noise_record,
        checkpoint_sha256=lineage['sha256'], load_seconds=cold_seconds, model_load_count=1,
        elapsed_seconds=time.monotonic() - began, elapsed_seconds_scope='source_read_through_engine_close',
        history=str(history_path), default_demo_changed=False, new_training=False,
        total_model_forward_calls=sum(case['actual_model_forward_calls'] for case in results),
        quality_evaluation='pending_visual_comparison', quality_improvement_verified=False, realtime=False,
        **parent_metadata)
    write_json(directory / 'progress.json', dict(result, active_case=None))
    return result
