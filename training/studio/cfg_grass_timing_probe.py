"""Four bounded grassland runs: exact, cached, cached, exact on one model load.

This is a one-scene/one-seed reversed-order repeat, not four independent scenes
or a quality verdict. The existing Creator launcher owns authorization, the
shared GPU lease and the fixed 1500-second watchdog. No default is changed.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import re
import statistics
import time

from training.demo.contracts import contained, require, write_json
from training.demo.latency_engine import ReusableActionEngine, _WanRuntime
from training.studio.cfg_delta_probe import CFGDeltaFlow, POLICY
from training.studio.overlap_probe import DATA_ROOT, HEIGHT, R005_SHA, SETTINGS, WIDTH
from training.studio.quality_probe import (
    DEFAULT_NEGATIVE_CACHE, QualityRuntime, TextCFGFlow, annotate_report, load_negative,
)

REVISION = 'R005-grassland-CFG5-exact-cached-cached-exact-6s-v1'
SOURCE = DATA_ROOT / 'creator-runtime/quality-20260928/grassland-source'
EPISODE = '009b81104345761d84612406c8f74c91'
PROMPT_EMBEDDING_SHA = '913646dba6ea8dbb2f6bda1e17df798111a11cdf05dcf6cf7b7249f8a6396a79'
# Reuse the completed grass_text_cfg5 result, not a new weight/data hash sweep.
EXPECTED_INPUTS = dict(
    first_latent_sha256_float32='93ce16854c998b146f385d445908e116050e67a1e095798aaa25f4771fc8b915',
    noise_sha256_float32='a5464086b10298db73a58e0f014482619aa4082ed45f32edb0aa2be5c4da8bd0')
CASES = [dict(name=name, cached=cached, mode='segment6', **SETTINGS) for name, cached in
         [('exact_first', False), ('cached_first', True), ('cached_second', True), ('exact_second', False)]]
FIXED = dict(mode='cfg_grass_timing_probe', max_seconds=1500, source_job=str(SOURCE),
    checkpoint_sha256=R005_SHA, negative_cache=DEFAULT_NEGATIVE_CACHE, residency='co_resident',
    fastpath=False, width=WIDTH, height=HEIGHT, seed=42, future_frames=96, **SETTINGS, **EXPECTED_INPUTS)


def validate_request(request):
    require(type(request) is dict and set(request) == set(FIXED) |
            {'job_id', 'source_revision', 'experiment_revision'}, 'unknown grass timing fields')
    require(all(request[key] == value and type(request[key]) is type(value) for key, value in FIXED.items()),
            'grass timing fixes R005/WJ96/40EulerCFG5/1500seconds and exact-cached-cached-exact')
    require(type(request['job_id']) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', request['job_id'])
            and type(request['source_revision']) is str and bool(request['source_revision'])
            and request['experiment_revision'] == REVISION, 'invalid grass timing identity')


def selected_actions(original):
    import numpy as np
    expected = np.zeros((240, 8), np.float32)
    expected[:, 0], expected[:120, 5], expected[120:, 7] = 1, 1, 1
    require(isinstance(original, np.ndarray) and original.dtype == np.float32
            and np.array_equal(original, expected), 'grass source must preserve declared WJ120 then WL120')
    return original[:96].copy()


def check_grass_inputs(first, noise, keys):
    import torch
    from training.demo.action_backend import tensor_sha
    require(tuple(first.shape) == (1, 1, 48, 30, 52) and tuple(noise.shape) == (1, 25, 48, 30, 52)
            and tuple(keys.shape) == (1, 96, 8), 'fixed grass 97-frame latent/action geometry required')
    expected = torch.zeros_like(keys)
    expected[:, :, [0, 5]] = 1
    require(torch.equal(keys, expected), 'grass timing requires the same WJ96 actions on every case')
    actual = dict(first_latent_sha256_float32=tensor_sha(first), noise_sha256_float32=tensor_sha(noise))
    require(actual == EXPECTED_INPUTS, 'grass initial latent/noise differs from existing grass_text_cfg5')
    return dict(actual, action_sha256_float32=tensor_sha(keys))


class ExactTraceFlow(TextCFGFlow):
    """Unchanged TextCFGFlow arithmetic, with bounded step/call accounting."""
    def __init__(self, model, negative, on_trace=None):
        super().__init__(model, 5, negative)
        self.on_trace, self.trace = on_trace, []
        self.completed_steps, self.failed, self.closed = 0, False, False

    def __call__(self, *args, **kwargs):
        require(not self.closed and not self.failed and len(self.trace) < 40, 'one exact window; no retry')
        row = dict(step=len(self.trace), approximate=False, negative_recomputed=True,
                   status='attempted', model_forward_calls_before=self.forward_calls)
        self.trace.append(row)
        try:
            if self.on_trace:
                self.on_trace(copy.deepcopy(self.trace))
            result = super().__call__(*args, **kwargs)
            self.completed_steps += 1
            row['status'] = 'completed'
            return result
        except BaseException as error:
            self.failed = True
            row.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            row.update(model_forward_calls_after=self.forward_calls,
                       attempted_model_forward_calls=self.forward_calls - row['model_forward_calls_before'])
            if self.on_trace:
                self.on_trace(copy.deepcopy(self.trace))

    def close(self):
        self.model = self.negative = None
        self.closed = True


class GrassTimingRuntime(QualityRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.case_cursor, self.probe_failed = 0, False
        self.case_finished = True

    def configure(self, case):
        require(not self.probe_failed and self.case_finished and self.case_cursor < len(CASES)
                and case == CASES[self.case_cursor], 'fixed case order, no retries or partial restarts')
        super().configure(case)
        self.case_cursor += 1
        self.cached = case['cached']
        self.case_started, self.case_finished = False, False
        self.cfg_trace, self.cfg_stats, self.actual_inputs = [], {}, {}

    def generate(self, first, prompt, keys, noise, steps, *, fastpath=False):
        require(not self.probe_failed and not self.case_started and not self.case_finished
                and steps == 40 and type(steps) is int and fastpath is False
                and self.cfg_scale == 5 and self.solver == 'flow_euler', 'one fixed segment per case')
        self.case_started = True
        original, flow = self.model, None
        try:
            self.actual_inputs = check_grass_inputs(first, noise, keys)
            negative = self.negative.to(device='cuda', dtype=self.torch.bfloat16)
            callback = getattr(self, 'trace_callback', None)
            flow = (CFGDeltaFlow(original, negative, on_trace=callback) if self.cached
                    else ExactTraceFlow(original, negative, on_trace=callback))
            self.model = flow
            latent = _WanRuntime.generate(self, first, prompt, keys, noise, steps, fastpath=False)
            expected = 64 if self.cached else 80
            require(flow.completed_steps == len(flow.trace) == 40 and flow.forward_calls == expected,
                    'incomplete grass40step/64-or80forward schedule')
            if self.cached:
                require(flow.positive_calls == 40 and flow.negative_calls == 24, 'cached branch counts differ')
            self.case_finished = True
            return latent
        except BaseException:
            self.probe_failed = True
            raise
        finally:
            self.model = original
            if flow is not None:
                self.forward_calls += flow.forward_calls
                self.cfg_trace = copy.deepcopy(flow.trace)
                self.cfg_stats = dict(completed_logical_steps=flow.completed_steps,
                    actual_model_forward_calls=flow.forward_calls,
                    positive_forward_calls=(flow.positive_calls if self.cached else
                        sum(row.get('attempted_model_forward_calls', 0) >= 1 for row in flow.trace)),
                    negative_forward_calls=(flow.negative_calls if self.cached else
                        sum(row.get('attempted_model_forward_calls', 0) >= 2 for row in flow.trace)),
                    cached_steps=sum(row['approximate'] and row['status'] == 'completed' for row in flow.trace),
                    peak_delta_cache_bytes=getattr(flow, 'peak_cache_bytes', 0),
                    cross_case_reuse=False, cross_window_reuse=False)
                flow.close()
                self.cfg_stats['cache_released_before_decode'] = flow.closed and getattr(flow, 'delta', None) is None


def annotate_case(report, case, runtime, negative_source, cold):
    annotate_report(report, case, runtime, negative_source, cold)
    report.update(cfg_trace=copy.deepcopy(runtime.cfg_trace), cfg_stats=copy.deepcopy(runtime.cfg_stats),
                  actual_inputs=copy.deepcopy(runtime.actual_inputs), approximate_inference=case['cached'])
    report['sampler']['actual_model_forward_calls'] = runtime.forward_calls
    if case['cached']:
        report['cfg_delta_policy'] = copy.deepcopy(POLICY)
        report['sampler']['model_forwards_per_step'] = '2 exact / 1 approximate; see cfg_trace'
        report['mathematical_changes'].update(
            text_cfg='exact:vu+5*(vc-vu);approximate:vc_current+4*delta_previous', guidance_delta_reuse=True,
            sampler_integration='unchanged originalBF16 Euler and timestep labels')
    else:
        report['mathematical_changes']['guidance_delta_reuse'] = False
    return report


def timing_summary(results):
    require(len(results) == 4 and [row.get('name') for row in results] == [row['name'] for row in CASES]
            and all(row.get('status') == 'completed' for row in results), 'timing requires all four ordered completed cases')
    values = [row['generation']['elapsed_seconds'] for row in results]
    require(all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in values),
            'positive finite generation wall times required')
    exact, cached = [values[0], values[3]], [values[1], values[2]]
    summarize = lambda times: dict(seconds=times, mean=statistics.mean(times), median=statistics.median(times),
                                   minimum=min(times), maximum=max(times), count=2)
    return dict(order=[case['name'] for case in CASES], exact=summarize(exact), cached=summarize(cached),
        mean_latency_reduction_percent=100 * (1 - statistics.mean(cached) / statistics.mean(exact)),
        ratio_of_mean_times=statistics.mean(exact) / statistics.mean(cached),
        ordered_pair_latency_reduction_percent=[100 * (1 - values[1] / values[0]), 100 * (1 - values[2] / values[3])],
        measurement='generation.elapsed_seconds;encode/sampling/decode/writer;shared_cold_load_reported_separately',
        experimental_units='one scene, one seed, two timed runs per method; reverse-order repeat, not independent scenes',
        warmup_note='all four clips retained, no discarded GPU warmup run; first exact may include one-time warmup',
        uncertainty='n=2 descriptive timings only; no confidence interval or population-speed claim',
        quality_evaluation='pending_actual_visual_review', quality_improvement_verified=False,
        speed_improvement_verified=False, default_changed=False, realtime=False)


def run(request, directory, lease):
    import numpy as np
    from PIL import Image
    from training.demo.action_backend import tensor_sha
    from training.demo.backend import validate_active_authority
    from training.studio.worker import read_source
    validate_request(request)
    directory = contained(directory, DATA_ROOT)
    require(directory.name == request['job_id'] and directory.is_dir()
            and not any((directory / name).exists() for name in ('inputs.json', 'progress.json', *[c['name'] for c in CASES])),
            'new grass timing job required')
    began, runtime, engine, cold, results = time.monotonic(), None, None, 0., []
    def check():
        require(time.monotonic() - began < 1470, 'finite grass timing deadline; no retry')
        validate_active_authority(lease)
    check()
    state, config, prompt, lineage, initial, original_actions, scene = read_source(request)
    require(lineage['sha256'] == R005_SHA and lineage['step'] == 1040 and scene['seed'] == 42
            and scene['source_episode_id'] == EPISODE and scene['scene_id'] == 'quality-grassland'
            and initial.shape == (HEIGHT, WIDTH, 3) and initial.dtype == np.uint8,
            'existing grassland R005 source required')
    require(tensor_sha(prompt) == PROMPT_EMBEDDING_SHA, 'existing grassland static prompt embedding differs')
    actions = selected_actions(original_actions)
    negative, negative_source = load_negative(request['negative_cache'], config.base_model_path)
    write_json(directory / 'inputs.json', dict(mode=FIXED['mode'], source_job=str(SOURCE),
        source_episode_id=EPISODE, scene_id=scene['scene_id'], prompt=scene['prompt'],
        prompt_condition=lineage['prompt_condition'], negative_prompt=negative_source,
        initial_origin='existing_20260928_grassland_source_from_historical_frame0_lossy_preview;not_original_raw_RGB',
        checkpoint=lineage['path'], checkpoint_sha256=R005_SHA, checkpoint_step=1040,
        seed=42, width=WIDTH, height=HEIGHT, expected_input_digests=EXPECTED_INPUTS,
        source_actions=scene['action_segments'], actual_actions=[dict(frames=96, keys=['W', 'J'])],
        cases=CASES, expected_frames_per_video=97, expected_model_forwards=[80, 64, 64, 80],
        expected_model_forwards_total=288, model_load_count=1, residency='co_resident', fastpath=False,
        source_revision=request['source_revision'], experiment_revision=REVISION,
        comparison='one grassland scene/seed;exact-cached-cached-exact reverse-order timing repeat',
        cfg_delta_policy=POLICY, new_training=False, default_changed=False, quality_improvement_verified=False))
    Image.fromarray(initial).save(directory / 'initial.png')
    np.save(directory / 'actions.npy', actions, allow_pickle=False)
    write_json(directory / 'progress.json', dict(status='running', phase='loading', cases=[]))
    try:
        check()
        tick = time.monotonic()
        runtime = GrassTimingRuntime(state, config, 'co_resident', negative)
        runtime.synchronize()
        cold = time.monotonic() - tick
        engine = ReusableActionEngine(state, config, residency='co_resident', fastpath=False, _runtime=runtime)
        engine.load_seconds = cold
        for case in CASES:
            check()
            require(request['max_seconds'] - (time.monotonic() - began) >= 300,
                    'less than300seconds remain; do not start another case')
            target = directory / case['name']
            target.mkdir(exist_ok=False)
            runtime.configure(case)
            engine.last_run_report = None
            def trace_callback(trace):
                check()
                write_json(directory / 'progress.json', dict(status='running', phase='denoising',
                    active_case=case['name'], trace=trace, completed_cases=[row['name'] for row in results]))
            runtime.trace_callback = trace_callback
            started = time.monotonic()
            try:
                report, endpoint = engine.run_segment(initial=initial, prompt=prompt, actions=actions.copy(),
                    seed=42, output_path=target / 'raw.mp4', steps=40)
                del endpoint
                annotate_case(report, case, runtime, negative_source, cold)
                require(report['frames_written'] == 97 and all(report[key] == value for key, value in EXPECTED_INPUTS.items())
                        and runtime.forward_calls == (64 if case['cached'] else 80), 'grass output/input/call contract differs')
                entry = dict(**case, status='completed', seed=42, generation=report,
                    raw_video=str(target / 'raw.mp4'), elapsed_seconds=time.monotonic() - started,
                    elapsed_seconds_scope='case wrapper, primary timer is generation.elapsed_seconds',
                    quality_evaluation='pending_actual_visual_review', quality_improvement_verified=False,
                    speed_improvement_verified=False, automatic_retry=False)
                write_json(target / 'result.json', entry)
                results.append(entry)
            except BaseException as error:
                report = copy.deepcopy(engine.last_run_report or {})
                annotate_case(report, case, runtime, negative_source, cold)
                entry = dict(**case, status='failed', generation=report,
                    error=f'{type(error).__name__}: {error}', automatic_retry=False)
                write_json(target / 'result.json', entry)
                results.append(entry)
                raise
    except BaseException as error:
        write_json(directory / 'progress.json', dict(status='failed', cases=results,
            error=f'{type(error).__name__}: {error}', completed_cases_preserved=True, automatic_retry=False))
        raise
    finally:
        if engine is not None:
            engine.close()
        elif runtime is not None:
            runtime.close()
    result = dict(status='completed', mode=FIXED['mode'], cases=results, load_seconds=cold, model_load_count=1,
        actual_model_forward_calls=sum(row['generation']['actual_model_forward_calls'] for row in results),
        timing_summary=timing_summary(results), elapsed_seconds=time.monotonic() - began,
        default_changed=False, original_demo_changed=False, new_training=False, automatic_retry=False,
        quality_improvement_verified=False, speed_improvement_verified=False, realtime=False)
    require(result['actual_model_forward_calls'] == 288, 'four grass clips require288actual model calls')
    write_json(directory / 'progress.json', result)
    return result


def prepare(*, output, source_revision):
    directory = contained(output, DATA_ROOT)
    request = dict(copy.deepcopy(FIXED), job_id=directory.name, source_revision=source_revision, experiment_revision=REVISION)
    validate_request(request)
    require(not directory.exists(), 'new output directory required')
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'request.json', request)
    return dict(status='cpu_request_prepared', request=str(directory / 'request.json'), gpu_launched=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), ensure_ascii=False))


if __name__ == '__main__':
    main()
