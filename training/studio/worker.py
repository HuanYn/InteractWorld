"""GPU worker for explicit latency trials; never bypasses an existing Creator lease."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from training.demo.backend import validate_lease, validate_active_authority
from training.demo.contracts import require, sha256, write_json


def read_source(request, *, allow_fp32_lora_candidate=False, allow_low_sigma_candidate=False):
    """Reuse the verified checkpoint contract and a frozen real job's small inputs."""
    import numpy as np
    import yaml
    from training.demo.action_backend import load_action_inputs
    from training.demo.contracts import load_initial, action_array
    source = Path(request['source_job'])
    original = json.loads((source / 'request.json').read_text(encoding='utf-8'))
    for filename, field in [('rollout.yaml', 'config_sha256'), ('initial.npy', 'initial_sha256'), ('actions.npy', 'actions_sha256')]:
        require(sha256(source / filename) == original[field], f'frozen source changed: {filename}')
    raw = yaml.safe_load((source / 'rollout.yaml').read_text(encoding='utf-8'))
    require(len(raw['scenes']) == 1, 'one scene per bounded trial')
    scene = raw['scenes'][0]
    require(scene['prompt'] == original['prompt'] and scene['source_episode_id'] == original['source_episode_id'],
            'source scene binding differs')
    require(type(allow_fp32_lora_candidate) is bool, 'FP32 source opt-in must be explicit')
    require(type(allow_low_sigma_candidate) is bool and (not allow_low_sigma_candidate or allow_fp32_lora_candidate),
            'low-sigma source requires explicit FP32 opt-in')
    state, config, prompt, lineage = (load_action_inputs(raw, scene, allow_fp32_lora_candidate=True, allow_low_sigma_candidate=True)
        if allow_low_sigma_candidate else load_action_inputs(raw, scene, allow_fp32_lora_candidate=True)
        if allow_fp32_lora_candidate else load_action_inputs(raw, scene))
    initial = load_initial(source / 'initial.npy')
    actions = np.load(source / 'actions.npy', allow_pickle=False)
    require(np.array_equal(actions, action_array(original['action_segments'])), 'source action tensor differs')
    return state, config, prompt, lineage, initial, actions, scene


def write_clip(path, frames):
    from training.eval.rollout15s import ffmpeg_writer_factory
    writer = ffmpeg_writer_factory(path, 832, 480, 16)
    try:
        writer.write(frames)
        writer.close()
    except BaseException:
        writer.abort()
        raise


def sweep(request, directory, lease):
    import numpy as np
    from PIL import Image
    from training.demo.latency_engine import ReusableActionEngine
    began = time.monotonic()
    state, config, prompt, lineage, initial, actions, scene = read_source(request)
    write_json(directory / 'inputs.json', dict(scene_id=scene['scene_id'], prompt=scene['prompt'],
        checkpoint=lineage['path'], checkpoint_sha256=lineage['sha256'], source_job=request['source_job'],
        source_revision=request.get('source_revision'), experiment_revision=request.get('experiment_revision'),
        source_actions=scene['action_segments'], input_loading_seconds=time.monotonic() - began))
    Image.fromarray(initial).save(directory / 'initial.png')
    engine = ReusableActionEngine(state, config, residency=request.get('residency', 'phase'), fastpath=request.get('fastpath', False))
    results = []
    try:
        for case in request['cases']:
            validate_active_authority(lease)
            name, steps = case['name'], case['steps']
            require(isinstance(name, str) and name.isascii() and name.replace('-', '').replace('_', '').isalnum(), 'unsafe trial name')
            target = directory / name
            target.mkdir(exist_ok=False)
            started = time.monotonic()
            chunks = []

            def emit(frames, offset):
                if offset == 0 and len(frames) == 1:
                    return  # A conditioning still is not first generated output.
                playable = np.concatenate([initial[None], frames]) if offset == 1 else frames
                output = target / f'chunk-{len(chunks):02d}.mp4'
                write_clip(output, playable)
                chunks.append(dict(path=str(output), start_frame=offset, generated_frames=len(frames),
                    available_seconds=time.monotonic() - started))
                write_json(target / 'progress.json', dict(status='running', chunks=chunks))

            try:
                report = engine.run(initial=initial, prompt=prompt, actions=actions,
                    seed=case.get('seed', scene['seed']), output_path=target / 'raw.mp4',
                    steps=steps, window6=case.get('window6', True), on_chunk=emit)
                entry = dict(name=name, status='completed', steps=steps, seed=case.get('seed', scene['seed']),
                    elapsed_seconds=time.monotonic() - started, first_generated_clip_seconds=chunks[0]['available_seconds'],
                    chunks=chunks, generation=report, raw_video=str(target / 'raw.mp4'),
                    quality_evaluation='pending_visual_comparison', realtime=False)
                write_json(target / 'result.json', entry)
                write_json(target / 'progress.json', dict(status='completed', chunks=chunks))
                results.append(entry)
                write_json(directory / 'progress.json', dict(status='running', cases=results))
            except Exception as error:
                failure = dict(name=name, status='failed', error=f'{type(error).__name__}: {error}',
                               elapsed_seconds=time.monotonic() - started)
                write_json(target / 'result.json', failure)
                raise
    finally:
        engine.close()
    return dict(status='completed', mode='sweep', residency=request.get('residency', 'phase'), cases=results,
                elapsed_seconds=time.monotonic() - began, realtime=False)


def preview(request, directory, lease):
    """Explicit resolution/step tradeoff probe, never a same-quality speed claim."""
    import numpy as np
    from PIL import Image
    from training.demo.latency_engine import ReusableActionEngine
    state, config, prompt, lineage, initial, _, scene = read_source(request)
    results = []
    for case in request['cases']:
        validate_active_authority(lease)
        width, height = case['width'], case['height']
        name = case['name']
        require(isinstance(name, str) and name.isascii() and name.replace('-', '').replace('_', '').isalnum(), 'unsafe preview name')
        target = directory / name
        target.mkdir(exist_ok=False)
        resized = np.array(Image.fromarray(initial).resize((width, height), Image.Resampling.LANCZOS))
        Image.fromarray(resized).save(target / 'initial.png')
        engine = ReusableActionEngine(state, config, residency='co_resident', fastpath=True, width=width, height=height)
        try:
            actions = np.zeros((48, 8), dtype=np.float32)
            actions[:, [0, 5]] = 1  # W + J, same first movement as the full-resolution pair.
            report, endpoint = engine.run_segment(initial=resized, prompt=prompt, actions=actions,
                seed=42, output_path=target / 'video.mp4', steps=case['steps'])
            entry = dict(name=name, status='completed', width=width, height=height, seed=42,
                action_keys=['W', 'J'], steps=case['steps'], report=report,
                elapsed_seconds=report['elapsed_seconds'], load_seconds=engine.load_seconds,
                generated_fps=48 / report['elapsed_seconds'], realtime_factor=report['elapsed_seconds'] / 3,
                quality_evaluation='requires visual inspection; resolution and sampling changed',
                realtime_verified=False, checkpoint=lineage['path'], scene_id=scene['scene_id'])
            write_json(target / 'result.json', entry)
            np.save(target / 'endpoint.npy', endpoint.numpy(), allow_pickle=False)
            results.append(entry)
            write_json(directory / 'progress.json', dict(status='running', cases=results))
        finally:
            engine.close()
    return dict(status='completed', mode='preview', cases=results, realtime_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    directory = args.request.resolve().parent
    request = json.loads(args.request.read_text(encoding='utf-8'))
    lease = json.loads((directory / 'lease.json').read_text(encoding='utf-8'))
    validate_lease(lease, max_seconds=request['max_seconds'], job_id=request['job_id'], request_sha256=sha256(args.request))
    from training import gpu_gate
    snapshot = gpu_gate.query_shared_gpu(confirmed_index=lease['gpu_index'], confirmed_uuid=lease['gpu_uuid'],
        profile=lease['profile'], memory_max_exclusive=lease['desktop_memory_max_exclusive'])
    write_json(directory / 'gpu-start.json', snapshot.as_dict())
    try:
        if request['mode'] == 'sweep':
            result = sweep(request, directory, lease)
        elif request['mode'] == 'preview':
            result = preview(request, directory, lease)
        elif request['mode'] == 'continuation_probe':
            from training.studio.continuation_probe import run
            result = run(request, directory, lease)
        elif request['mode'] == 'rope_probe':
            from training.studio.rope_probe import run
            result = run(request, directory, lease)
        elif request['mode'] == 'vae_roundtrip_probe':
            from training.studio.vae_roundtrip_probe import run
            result = run(request, directory, lease)
        elif request['mode'] in ('overlap_probe', 'overlap_reverse_probe'):
            from training.studio.overlap_probe import run
            result = run(request, directory, lease)
        elif request['mode'] == 'long_history_probe':
            from training.studio.long_history_probe import run
            result = run(request, directory, lease)
        elif request['mode'] == 'text_kv_probe':
            from training.studio.text_kv_probe import run
            result = run(request, directory, lease)
        elif request['mode'] == 'operator_profile':
            from training.studio.operator_profile import run
            result = run(request, directory, lease)
        elif request['mode'] == 'lora_inplace_probe':
            from training.studio.lora_inplace_probe import run
            result = run(request, directory, lease)
        elif request['mode'] == 'batched_cfg_probe':
            from training.studio.batched_cfg_probe import run
            result = run(request, directory, lease)
        elif request['mode'] == 'ffn_compile_probe':
            from training.studio.ffn_compile_probe import run
            result = run(request, directory, lease)
        elif request['mode'] == 'prompt_clause_probe':
            from training.studio.prompt_clause_probe import run
            result = run(request, directory, lease)
        elif request['mode'] == 'action_denoise_probe':
            from training.studio.action_denoise_probe import run
            result = run(request, directory, lease)
        elif request['mode'] == 'timestep_precision_probe':
            from training.studio.timestep_precision_probe import run
            result = run(request, directory, lease)
        elif request['mode'] == 'optimizer_precision_probe':
            from training.studio.optimizer_precision_probe import run
            result = run(request, directory, lease)
        elif request['mode'] == 'fp32_lora_eval_probe':
            from training.studio.fp32_lora_eval_probe import run
            result = run(request, directory, lease)
        elif request['mode'] == 'cfg_delta_probe':
            from training.studio.cfg_delta_probe import run
            result = run(request, directory, lease)
        elif request['mode'] == 'cfg_delta_pair':
            from training.studio.cfg_delta_pair import run
            result = run(request, directory, lease)
        elif request['mode'] == 'codec_condition_probe':
            from training.studio.codec_condition_probe import run
            result = run(request, directory, lease)
        elif request['mode'] == 'cfg_grass_timing_probe':
            from training.studio.cfg_grass_timing_probe import run
            result = run(request, directory, lease)
        elif request['mode'] == 'quality_probe':
            from training.studio.quality_probe import run
            result = run(request, directory, lease)
        elif request['mode'] == 'rebuild_cache':
            from training.studio.rebuild_subset import cache
            result = cache(request, directory, lease)
        elif request['mode'] == 'subset_train':
            from training.studio.subset_train import run
            result = run(request, directory, lease)
        elif request['mode'] == 'window97_cache':
            from training.studio.expand_window97 import cache
            result = cache(request, directory, lease)
        elif request['mode'] == 'paired_turn_cache':
            from training.studio.paired_turn_cache import run
            result = run(request, directory, lease)
        elif request['mode'] == 'paired_turn_eval':
            from training.studio.paired_turn_eval import run
            result = run(request, directory, lease)
        elif request['mode'] == 'paired_turn_train':
            from training.studio.paired_turn_train import run
            result = run(request, directory, lease)
        elif request['mode'] == 'paired_turn_b_eval':
            from training.studio.paired_turn_b_eval import run
            result = run(request, directory, lease)
        elif request['mode'] == 'paired_turn_cfg_signal':
            from training.studio.paired_turn_cfg_signal import run
            result = run(request, directory, lease)
        elif request['mode'] == 'paired_turn_update_signal':
            from training.studio.paired_turn_update_signal import run
            result = run(request, directory, lease)
        elif request['mode'] == 'master_writeback_train':
            from training.studio.master_writeback_train import run
            result = run(request, directory, lease)
        elif request['mode'] == 'master_writeback_eval':
            from training.studio.master_writeback_eval import run
            result = run(request, directory, lease)
        elif request['mode'] in ('neutral_caption_encode', 'neutral_caption_eval'):
            from training.studio.neutral_caption import run
            result = run(request, directory, lease)
        elif request['mode'] == 'reflection_pair_cache':
            from training.studio.reflection_pair_cache import run
            result = run(request, directory, lease)
        elif request['mode'] == 'reflection_pair_train':
            from training.studio.reflection_pair_train import run
            result = run(request, directory, lease)
        elif request['mode'] == 'reflection_pair_eval':
            from training.studio.reflection_pair_eval import run
            result = run(request, directory, lease)
        elif request['mode'] == 'balanced_turn_cache':
            from training.studio.balanced_turn_cache import run
            result = run(request, directory, lease)
        elif request['mode'] == 'balanced_fresh_train':
            from training.studio.balanced_fresh_train import run
            result = run(request, directory, lease)
        elif request['mode'] == 'balanced_fresh_eval':
            from training.studio.balanced_fresh_eval import run
            result = run(request, directory, lease)
        elif request['mode'] == 'balanced_lora_pair_train':
            from training.studio.balanced_lora_pair_train import run
            result = run(request, directory, lease)
        elif request['mode'] == 'balanced_lora_pair_eval':
            from training.studio.balanced_lora_pair_eval import run
            result = run(request, directory, lease)
        elif request['mode'] == 'teacher_reference_cache':
            from training.studio.teacher_reference_cache import run
            result = run(request, directory, lease)
        elif request['mode'] == 'teacher_reference_train':
            from training.studio.teacher_reference_train import run
            result = run(request, directory, lease)
        elif request['mode'] == 'teacher_reference_eval':
            from training.studio.teacher_reference_eval import run
            result = run(request, directory, lease)
        elif request['mode'] == 'expanded_teacher_train':
            from training.studio.expanded_teacher_train import run
            result = run(request, directory, lease)
        elif request['mode'] == 'expanded_teacher_eval':
            from training.studio.expanded_teacher_eval import run
            result = run(request, directory, lease)
        elif request['mode'] == 'expanded_reference_cache':
            from training.studio.expanded_reference_cache import run
            result = run(request, directory, lease)
        elif request['mode'] == 'expanded_reference_train':
            from training.studio.expanded_reference_train import run
            result = run(request, directory, lease)
        elif request['mode'] == 'expanded_reference_eval':
            from training.studio.expanded_reference_eval import run
            result = run(request, directory, lease)
        elif request['mode'] == 'action_margin_train':
            from training.studio.action_margin_train import run
            result = run(request, directory, lease)
        elif request['mode'] == 'action_margin_eval':
            from training.studio.action_margin_eval import run
            result = run(request, directory, lease)
        elif request['mode'] == 'completion_causal_train':
            from training.studio.completion_causal_train import run
            result = run(request, directory, lease)
        elif request['mode'] == 'completion_causal_eval':
            from training.studio.completion_causal_eval import run
            result = run(request, directory, lease)
        elif request['mode'] == 'completion_causal_sufficiency_train':
            from training.studio.completion_causal_sufficiency_train import run
            result = run(request, directory, lease)
        elif request['mode'] == 'completion_causal_sufficiency_validate':
            from training.studio.completion_causal_sufficiency_eval import run
            result = run(request, directory, lease)
        elif request['mode'] == 'completion_causal_sufficiency_eval':
            from training.studio.completion_causal_sufficiency_eval import run
            result = run(request, directory, lease)
        elif request['mode'] == 'window97_train':
            from training.studio.window97_train import run
            result = run(request, directory, lease)
        elif request['mode'] == 'causal_migration_train':
            from training.studio.causal_migration_probe import run
            result = run(request, directory, lease)
        elif request['mode'] == 'causal_migration_bid_train':
            from training.studio.causal_bid_probe import run
            result = run(request, directory, lease)
        elif request['mode'] == 'causal_native_eval':
            from training.studio.causal_native_eval import run
            result = run(request, directory, lease)
        elif request['mode'] == 'abot_fresh_teacher_train':
            from training.studio.abot_fresh_train import run
            result = run(request, directory, lease)
        elif request['mode'] == 'abot_fresh_eval':
            from training.studio.abot_fresh_eval import run
            result = run(request, directory, lease)
        elif request['mode'] == 'controlled_action_train':
            from training.studio.controlled_action_train import run
            result = run(request, directory, lease)
        elif request['mode'] == 'controlled_action_eval':
            from training.studio.controlled_action_eval import run
            result = run(request, directory, lease)
        elif request['mode'] == 'controlled_action_diagnostic':
            from training.studio.controlled_action_diagnostic import run_diagnostic
            result = run_diagnostic(request, directory, lease, sha256(args.request))
        elif request['mode'] == 'controlled_lora_pair_train':
            from training.studio.controlled_lora_pair_train import run
            result = run(request, directory, lease)
        elif request['mode'] == 'controlled_lora_pair_eval':
            from training.studio.controlled_lora_pair_eval import run
            result = run(request, directory, lease)
        elif request['mode'] == 'controlled_lora_pair_diagnostic':
            from training.studio.controlled_lora_pair_diagnostic import run_diagnostic
            result = run_diagnostic(request, directory, lease, sha256(args.request))
        elif request['mode'] == 'controlled_matched_window_eval':
            from training.studio.controlled_matched_window_eval import run
            result = run(request, directory, lease)
        elif request['mode'] == 'controlled_tinyfit_train':
            from training.studio.controlled_tinyfit_train import run
            result = run(request, directory, lease)
        elif request['mode'] == 'controlled_tinyfit_eval':
            from training.studio.controlled_tinyfit_eval import run
            result = run(request, directory, lease)
        elif request['mode'] == 'controlled_tinyfit_diagnostic':
            from training.studio.controlled_tinyfit_diagnostic import run
            result = run(request, directory, lease)
        elif request['mode'] == 'serve':
            from training.studio.server import serve
            result = serve(request, directory, lease)
        else:
            raise ValueError('unsupported studio mode')
        write_json(directory / 'result.json', result)
    except Exception as error:
        write_json(directory / 'failure.json', dict(status='failed', error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    main()
