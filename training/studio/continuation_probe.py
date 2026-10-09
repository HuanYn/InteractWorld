"""Bounded 512x288 continuation probe; the supervisor owns GPU authorization.

This is an explicit resolution/sampling/history tradeoff experiment, not an
automatic UI mode or a same-quality speed comparison with the original demo.
"""
from __future__ import annotations

from pathlib import Path
import statistics
import sys
import time

import numpy as np
from PIL import Image

from training.demo.backend import validate_active_authority
from training.demo.contracts import require, write_json
from training.demo.latency_engine import ReusableActionEngine
from training.eval.rollout15s import ffmpeg_writer_factory
from training.studio.worker import read_source


WIDTH, HEIGHT, FPS, STEPS = 512, 288, 16, 10
DEFAULT_COUNTS, BASE_SEED = (48, 48, 48, 48, 48), 42


class _CombinedWriter:
    """Keep the very first conditioning frame; drop only later duplicates."""

    def __init__(self, writer):
        self.writer = writer
        self.frames = 0
        self.segment_index = -1
        self.local_frames = 0
        self.future_count = 48
        self.expected_frames = 1

    def begin(self, index, count=48):
        require(index == self.segment_index + 1, 'segments must be appended in order')
        require(type(count) is int and count in (48, 96), 'segment count must be48 or96')
        require(index == 0 or self.local_frames == self.future_count + 1,
                'preceding segment callback is incomplete')
        self.segment_index, self.local_frames = index, 0
        self.future_count = count
        self.expected_frames += count

    def emit(self, frames, offset):
        require(isinstance(frames, np.ndarray) and frames.dtype == np.uint8
                and frames.ndim == 4 and frames.shape[1:] == (HEIGHT, WIDTH, 3),
                'unexpected continuation RGB shape/dtype')
        require(offset == self.local_frames and len(frames) > 0,
                'segment callback offsets must be contiguous')
        if offset == 0:
            require(len(frames) == 1, 'conditioning frame must be emitted separately')
        require(self.local_frames + len(frames) <= self.future_count + 1,
                'too many segment frames')
        self.local_frames += len(frames)
        if offset == 0 and self.segment_index > 0:
            return
        self.writer.write(frames)
        self.frames += len(frames)

    def finish_segment(self):
        require(self.local_frames == self.future_count + 1, 'segment callback is incomplete')
        require(self.frames == self.expected_frames,
                'combined frame count differs from the action timeline')


def _validate_request(request):
    require(type(request.get('max_seconds')) is int and 0 < request['max_seconds'] <= 600,
            'continuation probe requires a positive external timeout of at most600 seconds')
    for name, expected in [('width', WIDTH), ('height', HEIGHT), ('seed', BASE_SEED)]:
        value = request.get(name, expected)
        require(type(value) is int and value == expected, f'continuation probe requires {name}={expected}')
    steps = request.get('steps', STEPS)
    require(type(steps) is int and steps in (10, 20), 'continuation probe requires steps=10 or20')
    counts = request.get('future_counts', list(DEFAULT_COUNTS))
    require(isinstance(counts, list) and all(type(value) is int for value in counts)
            and counts in (list(DEFAULT_COUNTS), [96, 96, 48]),
            'continuation future_counts must be [48,48,48,48,48] or [96,96,48]')
    require(request.get('fastpath', True) is True, 'continuation probe requires fastpath=True')
    require(request.get('residency', 'co_resident') == 'co_resident',
            'continuation probe requires co_resident residency')
    return steps, counts.copy()


def run(request, directory, lease):
    """Generate real 15-second continuation (5x3s or6+6+3s) with one engine.

    The externally guarded worker must already have validated the lease and
    installed its finite watchdog. This function rechecks active authority
    before loading GPU modules and immediately before every segment.
    """
    steps, counts = _validate_request(request)
    directory = Path(directory)
    require(directory.is_dir(), 'continuation output directory must exist')
    owned = ['inputs.json', 'initial.png', 'combined.mp4', 'combined.partial.mp4',
             'continuation-progress.json', 'continuation-result.json', 'continuation-failure.json']
    for index in range(len(counts)):
        stem = f'segment-{index:02d}'
        owned.extend([stem + suffix for suffix in
                      ('.mp4', '.partial.mp4', '-actions.npy', '-endpoint.npy', '-result.json')])
    require(not any((directory / name).exists() for name in owned),
            'refusing to overwrite continuation probe evidence')
    began = time.monotonic()
    state, config, prompt, lineage, initial, actions, scene = read_source(request)
    require(isinstance(actions, np.ndarray) and actions.shape == (240, 8)
            and actions.dtype == np.float32 and np.isfinite(actions).all()
            and np.isin(actions, [0, 1]).all(), 'source actions must be frozen binary float32 [240,8]')
    resized = np.array(Image.fromarray(initial).resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS))
    Image.fromarray(resized).save(directory / 'initial.png')
    inputs = dict(scene_id=scene['scene_id'], prompt=scene['prompt'], source_job=request['source_job'],
        checkpoint=lineage['path'], checkpoint_sha256=lineage['sha256'],
        source_actions=scene['action_segments'], source_seed=scene['seed'],
        source_revision=request.get('source_revision'), experiment_revision=request.get('experiment_revision'),
        width=WIDTH, height=HEIGHT, fps=FPS, steps=steps, fastpath=True, residency='co_resident',
        future_counts=counts, segment_future_duration_seconds=[count / FPS for count in counts],
        resize=dict(method='PIL.LANCZOS', source_width=int(initial.shape[1]),
                    source_height=int(initial.shape[0]), explicit=True),
        seed_policy='independent CPU noise seed42+zero_based_segment_index',
        seeds=list(range(BASE_SEED, BASE_SEED + len(counts))),
        action_policy='exact contiguous slices of verified frozen source; boundaries are cumulative future_counts',
        action_slices=[[sum(counts[:index]), sum(counts[:index + 1])] for index in range(len(counts))],
        conditioning_policy='previous unquantized float32 RGB endpoint, independently VAE-reencoded',
        input_loading_seconds=time.monotonic() - began,
        actual_process=dict(executable=sys.executable, argv=list(sys.argv),
                            original_argv=list(getattr(sys, 'orig_argv', sys.argv)), cwd=str(Path.cwd())),
        external_timeout_seconds=request['max_seconds'], external_supervisor_required=True)
    write_json(directory / 'inputs.json', inputs)
    engine = writer = combined = None
    results = []
    current = resized
    try:
        validate_active_authority(lease)
        engine = ReusableActionEngine(state, config, residency='co_resident', fastpath=True,
                                      width=WIDTH, height=HEIGHT)
        writer = ffmpeg_writer_factory(directory / 'combined.mp4', WIDTH, HEIGHT, FPS)
        combined = _CombinedWriter(writer)
        start = 0
        for index, count in enumerate(counts):
            validate_active_authority(lease)
            stop = start + count
            selected = actions[start:stop].copy()
            stem = f'segment-{index:02d}'
            action_path = directory / f'{stem}-actions.npy'
            video_path = directory / f'{stem}.mp4'
            endpoint_path = directory / f'{stem}-endpoint.npy'
            np.save(action_path, selected, allow_pickle=False)
            combined.begin(index, count)
            started = time.monotonic()
            report, endpoint = engine.run_segment(initial=current, prompt=prompt, actions=selected,
                seed=BASE_SEED + index, output_path=video_path, steps=steps, on_chunk=combined.emit)
            wall_seconds = time.monotonic() - started
            combined.finish_segment()
            # Persist and forward exactly the engine's CPU float RGB, not the MP4 decode.
            np.save(endpoint_path, endpoint.detach().cpu().numpy(), allow_pickle=False)
            entry = dict(index=index, status='completed', video=str(video_path),
                actions=str(action_path), endpoint=str(endpoint_path), seed=BASE_SEED + index,
                action_slice_start=start, action_slice_stop_exclusive=stop,
                action_interval_seconds=[start / FPS, stop / FPS],
                initial_condition=str(directory / 'initial.png') if index == 0 else results[-1]['endpoint'],
                initial_condition_kind='resized_source_RGB' if index == 0 else 'previous_float_RGB_endpoint',
                call=dict(method='ReusableActionEngine.run_segment', steps=steps,
                          seed=BASE_SEED + index, future_action_frames=count,
                          output_path=str(video_path), callback='append_combined_skip_later_condition_frame'),
                elapsed_seconds=report['elapsed_seconds'], call_wall_seconds=wall_seconds,
                future_frames=count, future_duration_seconds=count / FPS, standalone_frames=count + 1,
                generated_fps=count / report['elapsed_seconds'],
                realtime_factor=report['elapsed_seconds'] / (count / FPS),
                combined_frames=combined.frames, generation=report)
            results.append(entry)
            write_json(directory / f'{stem}-result.json', entry)
            write_json(directory / 'continuation-progress.json', dict(status='running', segments=results,
                       combined_frames=combined.frames, combined_video_finalized=False))
            current = endpoint
            start = stop
        require(combined.frames == 241, 'complete continuation must contain241 frames')
        writer.close()
    except BaseException as error:
        if writer is not None:
            try:
                writer.abort()  # Only this combined writer's own partial is removed.
            except Exception:
                pass
        failure = dict(status='failed', error=f'{type(error).__name__}: {error}', segments=results,
            combined_frames=combined.frames if combined else 0, elapsed_seconds=time.monotonic() - began,
            completed_segments_preserved=True, generation=getattr(engine, 'last_run_report', None))
        write_json(directory / 'continuation-failure.json', failure)
        raise
    finally:
        if engine is not None:
            engine.close()
    steady = [item['elapsed_seconds'] for item in results[1:]]
    result = dict(status='completed', mode='continuation_probe', width=WIDTH, height=HEIGHT,
        fps=FPS, steps=steps, fastpath=True, residency='co_resident', resolution_reduced=True,
        future_counts=counts, segment_count=len(counts),
        segment_future_duration_seconds=[count / FPS for count in counts],
        model_load_count=1, load_seconds=engine.load_seconds, segments=results,
        segment_run_seconds=[item['elapsed_seconds'] for item in results],
        first_segment_seconds=results[0]['elapsed_seconds'], first_segment_future_duration_seconds=counts[0] / FPS,
        steady_segment_seconds=steady, steady_segment_mean_seconds=statistics.mean(steady),
        steady_segment_median_seconds=statistics.median(steady),
        steady_segment_future_duration_seconds=[count / FPS for count in counts[1:]],
        steady_generated_fps=sum(counts[1:]) / sum(steady),
        steady_realtime_factor=sum(steady) / (sum(counts[1:]) / FPS),
        steady_timing_scope='all segments after first; durations may differ; excludes one-time load, includes encode/sample/decode/write/callback',
        elapsed_seconds=time.monotonic() - began, elapsed_scope='source loading through engine close',
        combined_video=str(directory / 'combined.mp4'), combined_frames=241,
        combined_duration_seconds=241 / FPS, future_duration_seconds=15.0,
        endpoint_state='float RGB only; no retained transformer/VAE history or world state',
        realtime=False, not_realtime=True, realtime_verified=False, automatic_ui_promotion=False,
        same_quality_speed_comparison=False,
        quality_evaluation='pending visual inspection of the entire15s continuation',
        changed_axes=['resolution', 'sampling_steps', 'generation_window', 'per_segment_seed_policy'])
    write_json(directory / 'continuation-result.json', result)
    write_json(directory / 'continuation-progress.json', dict(status='completed', segments=results,
               combined_frames=241, combined_video_finalized=True))
    return result
