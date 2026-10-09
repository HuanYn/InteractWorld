"""Bounded15s five-RGB history: R005 by default, explicit GT-prefix step20 opt-in.

No training, retained attention/VAE state, retries, or baseline regeneration.
The external Creator launcher owns the authorized GPU and1200-second watchdog.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import platform
import time

import numpy as np

from training.demo import action_backend as action
from training.demo.contracts import require, sha256, write_json
from training.studio.overlap_probe import (
    DATA_ROOT, FPS, HEIGHT, R005_SHA, R005_SOURCE_JOB, SEED, SETTINGS, STEPS, WIDTH,
    _HistoryEngine, prefix_euler,
)
from training.studio.quality_probe import (
    DEFAULT_NEGATIVE_CACHE, QualityRuntime, TextCFGFlow, load_negative,
)

FUTURE_COUNTS = (96, 96, 48)
STARTS = (0, 96, 192)
BASELINE_RECORD = (DATA_ROOT / 'creator-runtime/quality-20260928/quality-cfg5-full15'
                   / 'text_cfg5_full15/result.json')
BASELINE_NOISE_SHAS = (
    'a5464086b10298db73a58e0f014482619aa4082ed45f32edb0aa2be5c4da8bd0',
    '60fa3d219c0e1927fdf1e472a529254173c3ddb82fe2bee0edfdd1b48cbd1d03',
    '3450d4cc419681ad682517eb682abd481f41b24711e37db1bd052adb7d3222ca',
)


def validate_source_binding(request):
    """Keep original default; only accept a prepared copy of its fixed inputs."""
    candidate = request.get('gt_prefix_mix_candidate', False)
    require(type(candidate) is bool, 'gt_prefix_mix_candidate must be an explicit boolean')
    reverse = request.get('reverse_timeline', False)
    require(type(reverse) is bool, 'reverse_timeline must be an explicit boolean')
    if reverse:
        from training.studio.long_history_reverse import validate_reverse_source
        return validate_reverse_source(request)
    require('reverse_binding' not in request, 'reverse source binding requires explicit reverse_timeline')
    source = Path(request['source_job']).resolve()
    parent = R005_SOURCE_JOB.resolve()
    if not candidate:
        require(request.get('checkpoint_sha256') == R005_SHA and source == parent,
                'long-history probe must use the original fixed R005 source')
        return None
    require(source != parent and source.is_relative_to(DATA_ROOT.resolve())
            and Path(request.get('parent_source_job', '')).resolve() == parent
            and request.get('parent_checkpoint_sha256') == R005_SHA,
            'GT candidate requires a new prepared source bound to original R005')
    original = json.loads((parent / 'request.json').read_text(encoding='utf-8'))
    copied = json.loads((source / 'request.json').read_text(encoding='utf-8'))
    require(copied.get('gt_prefix_mix_candidate') is True
            and Path(copied.get('parent_source_job', '')).resolve() == parent
            and original['checkpoint_sha256'] == copied.get('parent_checkpoint_sha256') == R005_SHA
            and copied['checkpoint_sha256'] == request.get('checkpoint_sha256') != R005_SHA,
            'GT candidate parent/checkpoint binding differs')
    require(copied.get('source_rollout_sha256') == original['config_sha256'] == sha256(parent / 'rollout.yaml'),
            'original R005 rollout binding changed')
    for name, field in (('initial.npy', 'initial_sha256'), ('actions.npy', 'actions_sha256')):
        require(copied[field] == original[field] == sha256(parent / name) == sha256(source / name),
                'GT source copy differs from original: ' + name)
    for field in ('prompt', 'source_episode_id', 'seed', 'action_segments'):
        require(copied[field] == original[field], 'GT source scene binding differs: ' + field)
    require(copied.get('ground_truth_future_used') is False, 'candidate evaluation may not inject GT future')
    return dict(parent_source_job=str(parent), parent_checkpoint_sha256=R005_SHA,
        source_rollout_sha256=original['config_sha256'], initial_sha256=original['initial_sha256'],
        actions_sha256=original['actions_sha256'], original_input_copy_verified=True)


def validate_loaded_source(request, lineage):
    """Checkpoint verification remains owned by read_source/load_action_inputs."""
    if request.get('gt_prefix_mix_candidate', False):
        from training.demo.action_backend import validate_gt_prefix_mix_contract
        require(lineage['sha256'] == request['checkpoint_sha256'] != R005_SHA,
                'loaded GT checkpoint differs from its immutable request pin')
        contract = validate_gt_prefix_mix_contract(lineage['config'], lineage['initialization'], step=lineage['step'])
        require(request.get('candidate_contract') == contract, 'prepared GT candidate contract changed')
        return contract
    require(lineage['sha256'] == R005_SHA and lineage['step'] == 1040,
            'verified checkpoint differs from original R005')
    return None


def validate_action_timeline(actions, scene, *, reverse_timeline=False):
    require(type(reverse_timeline) is bool, 'reverse_timeline must be an explicit boolean')
    first, second = (7, 5) if reverse_timeline else (5, 7)
    expected = np.zeros((240, 8), dtype=np.float32)
    expected[:120, [0, first]] = 1
    expected[120:, [0, second]] = 1
    keys = ('L', 'J') if reverse_timeline else ('J', 'L')
    segments = [dict(frames=120, keys=['W', key]) for key in keys]
    require(actions.dtype == np.float32 and np.array_equal(actions, expected)
            and scene['action_segments'] == segments,
            'long-history source must preserve the exact explicitly selected forward/reverse timeline')


def long_history_noises(seed, height, width):
    """Return reference/candidate lists from the original five13-slot CPU draws."""
    import torch
    require(type(seed) is int and 0 <= seed <= 2**32 - 1, 'invalid noise seed')
    require(all(type(value) is int and value > 0 and value % 16 == 0
                for value in (height, width)), 'RGB dimensions must be positive multiples of16')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    draws = torch.stack([torch.randn((1, 13, 48, height // 16, width // 16),
        generator=generator, dtype=torch.float32, device='cpu') for _ in range(5)])
    joint = action.joint_noise_from_chunks(draws)
    references, candidates = [], []
    for index, (start, count) in enumerate(zip(STARTS, FUTURE_COUNTS)):
        future = joint[:, 1 + start // 4:1 + (start + count) // 4]
        reference = torch.cat([joint[:, :1], future], dim=1)
        candidate = reference.clone() if index == 0 else torch.cat([torch.zeros_like(joint[:, :2]), future], dim=1)
        clean_count = 1 if index == 0 else 2
        require(torch.equal(candidate[:, clean_count:], reference[:, 1:]), 'future noise partition differs')
        references.append(reference)
        candidates.append(candidate)
    return references, candidates


def long_history_actions(actions, index):
    """Future-only for window0; the exact preceding4 actions plus future otherwise."""
    require(type(index) is int and 0 <= index < 3, 'invalid long-history window index')
    require(isinstance(actions, np.ndarray) and actions.shape == (240, 8)
            and actions.dtype == np.float32 and np.isfinite(actions).all()
            and np.isin(actions, [0, 1]).all(), 'timeline must be binary finite float32 [240,8]')
    require(not any(np.any((actions[:, left] != 0) & (actions[:, right] != 0))
                    for left, right in ((0, 2), (1, 3), (4, 6), (5, 7))), 'opposite actions in timeline')
    start = STARTS[index]
    return actions[start - (4 if index else 0):start + FUTURE_COUNTS[index]].copy()


def long_history_rgb(initial, pixels, index):
    """Return97/96/48 emitted RGB frames, never repeat reconstructed history."""
    import torch
    require(type(index) is int and 0 <= index < 3, 'invalid long-history window index')
    history_frames = 1 if index == 0 else 5
    require(torch.is_tensor(pixels) and pixels.device.type == 'cpu' and pixels.ndim == 5
            and tuple(pixels.shape[:3]) == (1, FUTURE_COUNTS[index] + history_frames, 3)
            and pixels.is_floating_point() and bool(torch.isfinite(pixels).all())
            and pixels.min() >= -1 and pixels.max() <= 1, 'invalid long-history decoded RGB')
    require(isinstance(initial, np.ndarray) and initial.dtype == np.uint8
            and initial.shape == (*pixels.shape[-2:], 3), 'initial must match decoded RGB geometry')
    future = pixels[0, history_frames:].permute(0, 2, 3, 1).add(1).mul(127.5).round().clamp(0, 255).byte().numpy()
    return np.concatenate([initial[None], future]) if index == 0 else future


class _LongHistoryRuntime(QualityRuntime):
    """Original one-clean path plus explicitly bounded2-clean 26/14-slot solves."""

    def generate(self, first, prompt, keys, noise, steps, *, fastpath=False):
        self.seen_noise = noise.detach().cpu().clone()
        if first.shape[1] == 1:
            return super().generate(first, prompt, keys, noise, steps, fastpath=fastpath)
        import torch
        from training.longforcing_lite import euler_sigmas
        from training.models.action_adapter import build_action_context
        require(first.shape[1] == 2 and keys.shape[1] in (100, 52)
                and noise.shape[1] == 2 + (keys.shape[1] - 4) // 4
                and steps == STEPS and fastpath is False and self.cfg_scale == 5
                and self.solver == 'flow_euler', 'unexpected long-history runtime configuration')
        conditions = dict(prompt_embeds=[prompt.to(device='cuda', dtype=torch.bfloat16)],
            act_context_scale=self.config.action_scale,
            act_context=build_action_context(keys, height=HEIGHT, width=WIDTH,
                                            device='cuda', dtype=torch.bfloat16))
        sigmas = euler_sigmas(STEPS, shift=5.0, device=torch.device('cuda'), dtype=torch.bfloat16)
        flow = TextCFGFlow(self.model, 5, self.negative.to(device='cuda', dtype=torch.bfloat16))
        try:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                result = prefix_euler(flow, first=first.to(device='cuda', dtype=torch.bfloat16),
                    noise=noise.to(device='cuda', dtype=torch.bfloat16), conditions=conditions, sigmas=sigmas)
            return result.detach().cpu()
        finally:
            self.forward_calls += flow.forward_calls


def run(request, directory, lease):
    """One model load; phase-offloaded 96+96+48 future frames; no auto retry."""
    import torch
    from PIL import Image
    from training.demo.backend import validate_active_authority
    from training.eval.rollout15s import ffmpeg_writer_factory
    from training.studio.worker import read_source

    began = time.monotonic()
    require(request.get('mode') == 'long_history_probe' and request.get('max_seconds') == 1200,
            'long-history probe requires its explicit mode and1200-second external watchdog')
    input_binding = validate_source_binding(request)
    require(request.get('residency', 'phase') == 'phase' and request.get('fastpath', False) is False,
            'long-history probe fixes phase residency and disables fastpath')
    require('cases' not in request and 'action_adapter_donor_source' not in request,
            'long-history probe does not accept a sweep or component hybrid')
    for field, value in dict(width=WIDTH, height=HEIGHT, fps=FPS, seed=SEED, **SETTINGS).items():
        require(field not in request or request[field] == value, f'long-history probe fixes {field}={value}')
    directory = Path(directory).resolve()
    require(directory.is_dir() and directory.is_relative_to(DATA_ROOT.resolve()) and directory != DATA_ROOT.resolve(),
            'long-history outputs must stay under the project data root')
    owned = ('raw.mp4', 'raw.partial.mp4', 'actions.npy', 'initial.png', 'inputs.json', 'progress.json',
             'long-history-failure.json', 'window0', 'window1', 'window2')
    require(not any((directory / name).exists() for name in owned), 'refusing to overwrite long-history evidence')
    negative_path = Path(request.get('negative_cache', DEFAULT_NEGATIVE_CACHE)).resolve()
    require(negative_path.is_relative_to(DATA_ROOT.resolve()), 'negative cache must stay on data disk')
    validate_active_authority(lease)
    state, config, prompt, lineage, initial, actions, scene = read_source(request)
    candidate_contract = validate_loaded_source(request, lineage)
    require(scene['seed'] == SEED and initial.shape == (HEIGHT, WIDTH, 3)
            and initial.dtype == np.uint8, 'long-history source RGB/seed changed')
    require(Path(config.base_model_path).resolve().is_relative_to(DATA_ROOT.resolve()), 'base weights must stay on data disk')
    reverse_timeline = request.get('reverse_timeline', False)
    validate_action_timeline(actions, scene, reverse_timeline=reverse_timeline)
    negative, negative_source = load_negative(negative_path, config.base_model_path)
    references, noises = long_history_noises(SEED, HEIGHT, WIDTH)
    noise_records = []
    for index, (reference, noise) in enumerate(zip(references, noises)):
        clean = 1 if index == 0 else 2
        reference_sha = action.tensor_sha(reference)
        require(reference_sha == BASELINE_NOISE_SHAS[index],
                'CPU RNG noise differs from saved historical R005 baseline; do not regenerate baseline')
        noise_records.append(dict(index=index, baseline_shape=list(reference.shape), candidate_shape=list(noise.shape),
            baseline_noise_sha256_float32=reference_sha,
            noise_sha256_float32=action.tensor_sha(noise),
            future_noise_sha256_float32=action.tensor_sha(reference[:, 1:]),
            future_noise_equal_fp32=bool(torch.equal(noise[:, clean:], reference[:, 1:])),
            future_noise_equal_bf16=bool(torch.equal(noise[:, clean:].bfloat16(), reference[:, 1:].bfloat16()))))
    inputs = dict(mode='long_history_probe', source_job=request['source_job'], scene_id=scene['scene_id'],
        prompt=scene['prompt'], source_actions=scene['action_segments'], seed=SEED,
        source_initial_npy=str(Path(request['source_job']) / 'initial.npy'), negative_prompt=negative_source,
        checkpoint=lineage['path'], checkpoint_sha256=lineage['sha256'], checkpoint_step=lineage['step'],
        source_revision=request.get('source_revision'), experiment_revision=request.get('experiment_revision'),
        width=WIDTH, height=HEIGHT, fps=FPS, settings=SETTINGS, residency='phase', fastpath=False,
        future_counts=list(FUTURE_COUNTS), action_slices=[[0, 96], [96, 192], [192, 240]],
        conditioning_action_slices=[[0, 96], [92, 192], [188, 240]],
        clean_latents=[1, 2, 2], latent_frames=[25, 26, 14], decoded_rgb_frames=[97, 101, 53],
        emitted_frames=[97, 96, 48], total_frames=241, future_duration_seconds=15,
        noise_policy='original_window6_five13slot_CPU_FP32_draws_joint_noise_from_chunks;future24_24_12',
        noise_windows=noise_records,
        baseline_noise_record=str(BASELINE_RECORD), baseline_noise_sha256_verified=True,
        cpu_rng=dict(torch_version=torch.__version__, platform=platform.platform(),
            machine=platform.machine(), device='cpu', dtype='float32', seed=SEED,
            cross_environment_bitwise_reproducibility_assumed=False),
        history_policy='window0=submitted_RGB;later=own_previous_window_raw_float_RGB_tail5_independently_VAE_encoded',
        history_not_from_video=True, historical_actions='the4_source_timeline_rows_immediately_before_new_future',
        external_timeout_seconds=1200, no_new_window_after_seconds=1100,
        input_loading_seconds=time.monotonic() - began, new_training=False, causal_KV=False,
        baseline_regenerated=False, default_demo_changed=False,
        timing_comparison='phase_offload_differs_from_historical_co_resident_baseline;no_same_residency_speed_claim',
        quality_evaluation='pending_full_video_visual_comparison', quality_improvement_verified=False)
    if candidate_contract is not None:
        inputs.update(gt_prefix_mix_candidate=True, candidate_contract=candidate_contract,
            parent_checkpoint_sha256=R005_SHA, parent_source_job=str(R005_SOURCE_JOB.resolve()),
            fixed_input_binding=input_binding, checkpoint_initialization=copy.deepcopy(lineage['initialization']),
            inference_only=True, training_history='GT_prefix_only;not_generated_history_or_error_recovery',
            trained_clean_latents=[1, 2], trained_future_rgb_frames=[96, 92],
            inference_length_boundary='K2_training5+92_RGB_vs_inference5+96_RGB_26_latents')
    if reverse_timeline:
        inputs.update(reverse_timeline=True, fixed_input_binding=input_binding,
            source_actions_sha256=sha256(Path(request['source_job']) / 'actions.npy'),
            paired_forward_request=input_binding['forward_request_path'])
    write_json(directory / 'inputs.json', inputs)
    np.save(directory / 'actions.npy', actions, allow_pickle=False)
    Image.fromarray(initial).save(directory / 'initial.png')
    runtime = engine = writer = None
    rows, written, active_window = [], 0, None
    try:
        require(time.monotonic() - began < 1100, 'deadline reached before model load')
        validate_active_authority(lease)
        loaded = time.monotonic()
        runtime = _LongHistoryRuntime(state, config, 'phase', negative)
        runtime.synchronize()
        cold_seconds = time.monotonic() - loaded
        engine = _HistoryEngine(state, config, residency='phase', fastpath=False, _runtime=runtime)
        engine.load_seconds = cold_seconds
        writer = ffmpeg_writer_factory(directory / 'raw.mp4', WIDTH, HEIGHT, FPS)
        history = None
        for index, (start, count) in enumerate(zip(STARTS, FUTURE_COUNTS)):
            require(time.monotonic() - began < 1100, 'deadline reached; no next window started')
            validate_active_authority(lease)
            active_window = index
            target = directory / f'window{index}'
            target.mkdir(exist_ok=False)
            clean, history_frames = (1, 1) if index == 0 else (2, 5)
            keys = long_history_actions(actions, index)
            np.save(target / 'actions.npy', keys, allow_pickle=False)
            row = dict(index=index, status='running', action_start=start, action_stop=start + count,
                conditioning_action_start=start - (4 if index else 0), condition_rgb_frames=history_frames,
                clean_latents=clean, latent_frames=clean + count // 4, decoded_rgb_frames=count + history_frames,
                emitted_frames=0, future_frames=count, expected_emitted_frames=count + (1 if index == 0 else 0),
                condition_source=str(directory / 'initial.png') if index == 0 else str(directory / f'window{index-1}/history.npy'),
                history_path=str(target / 'history.npy'), residency='phase',
                actual_model_forward_calls=0, **{key: value for key, value in noise_records[index].items() if key != 'index'})
            write_json(directory / 'progress.json', dict(status='running', active_window=index, windows=rows,
                current_window=row, frames_written=written, raw_video_finalized=False))
            runtime.configure(SETTINGS)
            engine._timings, engine._steps = [], STEPS
            engine.action_scale = SETTINGS['action_scale']
            runtime.begin()
            started = time.monotonic()
            try:
                condition = (torch.from_numpy(initial).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).float().div(127.5).sub(1)
                             if index == 0 else history.clone())
                first = engine._encode(condition, -1 if index == 0 else index - 1).to(dtype=torch.bfloat16)
                require(tuple(first.shape) == (1, clean, 48, 30, 52), 'long-history VAE condition shape changed')
                row['first_latent_sha256_float32'] = action.tensor_sha(first)
                latent = engine._generate(first, prompt, torch.from_numpy(keys).unsqueeze(0), noises[index], index)
                require(tuple(latent.shape) == (1, clean + count // 4, 48, 30, 52)
                        and bool(torch.isfinite(latent).all()) and torch.equal(latent[:, :clean], first),
                        'long-history solve shape changed or clean prefix was modified')
                require(runtime.forward_calls == STEPS * 2, 'long-history window must make40*2 CFG forwards')
                require(torch.equal(runtime.seen_noise, noises[index])
                        and torch.equal(runtime.seen_noise[:, clean:], references[index][:, 1:]),
                        'actual long-history sampler future differs from original window6 noise')
                require(torch.equal(runtime.seen_noise[:, clean:].bfloat16(), references[index][:, 1:].bfloat16()),
                        'actual model-input BF16 future noise differs')
                row.update(actual_noise_verified=True,
                    noise_sha256_float32=action.tensor_sha(runtime.seen_noise),
                    future_noise_sha256_float32=action.tensor_sha(runtime.seen_noise[:, clean:]))
                pixels = engine._decode(latent, index)
                frames = long_history_rgb(initial, pixels, index)
                engine._timed('raw_write', lambda: writer.write(frames))
                history = engine.history_tail.clone()
                require(tuple(history.shape) == (1, 3, 5, HEIGHT, WIDTH), 'next raw float history shape changed')
                np.save(target / 'history.npy', history.numpy(), allow_pickle=False)
                written += len(frames)
                row.update(status='completed', emitted_frames=len(frames), written_frames=written)
                del condition, first, latent, pixels, frames
            except BaseException as error:
                row.update(status='failed', error=f'{type(error).__name__}: {error}', written_frames=written,
                    retry=False)
                raise
            finally:
                row.update(elapsed_seconds=time.monotonic() - started, peak_vram_bytes=runtime.peak_bytes(),
                    actual_model_forward_calls=runtime.forward_calls, stage_timings=copy.deepcopy(engine._timings),
                    elapsed_seconds_scope='window_encode_sample_decode_write_and_history_save;excludes_shared_cold_load',
                    quality_evaluation='pending_visual_comparison')
                totals = {}
                for stage in engine._timings:
                    totals[stage['stage']] = totals.get(stage['stage'], 0.0) + stage['elapsed_seconds']
                row['stage_totals_seconds'] = totals
                write_json(target / 'result.json', row)
            rows.append(row)
            write_json(directory / 'progress.json', dict(status='running', active_window=None,
                windows=rows, frames_written=written, raw_video_finalized=False))
            active_window = None
        require(written == 241, 'long-history output must contain241 frames')
        finalized = time.monotonic()
        writer.close()
        finalize_seconds = time.monotonic() - finalized
        writer = None
    except BaseException as error:
        if writer is not None:
            try:
                writer.abort()
            except Exception:
                pass
        failure = dict(status='failed', mode='long_history_probe', active_window=active_window,
            error=f'{type(error).__name__}: {error}', windows=rows, frames_written=written,
            raw_video_finalized=False, elapsed_seconds=time.monotonic() - began,
            completed_window_histories_preserved=True, retry=False, default_demo_changed=False)
        write_json(directory / 'long-history-failure.json', failure)
        write_json(directory / 'progress.json', failure)
        raise
    finally:
        if engine is not None:
            engine.close()
        elif runtime is not None:
            runtime.close()
    result = dict(status='completed', mode='long_history_probe', raw_video=str(directory / 'raw.mp4'),
        windows=rows, frames_written=written, fps=FPS, width=WIDTH, height=HEIGHT,
        future_duration_seconds=15, encoded_duration_seconds=written / FPS,
        checkpoint_sha256=lineage['sha256'], seed=SEED, residency='phase', settings=SETTINGS,
        model_load_count=1, load_seconds=cold_seconds,
        load_seconds_scope='shared_CPU_model_construction_and_CUDA_context;per_window_transfers_are_separate',
        elapsed_seconds=time.monotonic() - began, elapsed_seconds_scope='source_loading_through_engine_close',
        raw_finalize_seconds=finalize_seconds, total_model_forward_calls=sum(row['actual_model_forward_calls'] for row in rows),
        peak_vram_bytes=max(row['peak_vram_bytes'] for row in rows),
        baseline_regenerated=False, default_demo_changed=False, new_training=False, causal_KV=False,
        realtime=False, raw_video_finalized=True, timing_comparison=inputs['timing_comparison'],
        quality_evaluation='pending_full_video_visual_comparison', quality_improvement_verified=False)
    if candidate_contract is not None:
        result.update(gt_prefix_mix_candidate=True, candidate_contract=candidate_contract,
            checkpoint_step=lineage['step'], parent_checkpoint_sha256=R005_SHA,
            fixed_input_binding=input_binding, inference_only=True,
            training_history='GT_prefix_only;not_generated_history_or_error_recovery')
    if reverse_timeline:
        result.update(reverse_timeline=True, fixed_input_binding=input_binding,
            source_actions=copy.deepcopy(scene['action_segments']), source_actions_sha256=inputs['source_actions_sha256'],
            paired_forward_request=input_binding['forward_request_path'])
    write_json(directory / 'progress.json', dict(result, active_window=None))
    return result
