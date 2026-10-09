"""Two bounded R00515s action timelines with window-local approximate CFG reuse.

No baseline generation, new checkpoint, cross-window cache, or default promotion.
The external Creator launcher owns the shared lease and1500second watchdog.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import time

from training.demo.contracts import contained, require, write_json
from training.demo.latency_engine import ReusableActionEngine, _WanRuntime
from training.studio.cfg_delta_probe import CFGDeltaFlow, POLICY
from training.studio.overlap_probe import DATA_ROOT, HEIGHT, R005_SHA, R005_SOURCE_JOB, SETTINGS, WIDTH
from training.studio.quality_probe import DEFAULT_NEGATIVE_CACHE, QualityRuntime, annotate_report, load_negative
from training.studio.timestep_precision_probe import EXPECTED_INPUTS, check_original_input_digests

REVISION = 'R005-CFG5-one-step-delta-paired15s-window6-v1'
CASES = [dict(name='cfg_delta_forward', mode='full15', reverse_timeline=False, **SETTINGS),
         dict(name='cfg_delta_reverse', mode='full15', reverse_timeline=True, **SETTINGS)]
FIXED = dict(mode='cfg_delta_pair', max_seconds=1500, source_job=str(R005_SOURCE_JOB),
    checkpoint_sha256=R005_SHA, negative_cache=DEFAULT_NEGATIVE_CACHE, residency='co_resident', fastpath=False,
    width=WIDTH, height=HEIGHT, seed=42, future_frames=240, window6=True, **SETTINGS, **EXPECTED_INPUTS)
WINDOW_FRAMES = (96, 96, 48)


def validate_request(request):
    require(type(request) is dict and set(request) == set(FIXED) |
            {'job_id', 'source_revision', 'experiment_revision'}, 'unknown CFG-delta pair fields')
    require(all(request[key] == value and type(request[key]) is type(value) for key, value in FIXED.items()),
            'CFG-delta pair fixes R005/two15s/40EulerCFG5/1500seconds')
    require(type(request['job_id']) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', request['job_id'])
            and type(request['source_revision']) is str and bool(request['source_revision'])
            and request['experiment_revision'] == REVISION, 'invalid CFG-delta pair identity')


def planned_actions(original):
    import numpy as np
    expected = np.zeros((240, 8), np.float32)
    expected[:, 0] = 1
    expected[:120, 5], expected[120:, 7] = 1, 1
    require(isinstance(original, np.ndarray) and original.dtype == np.float32
            and np.array_equal(original, expected), 'original WJ120 then WL120 required')
    return [original.copy(), original[::-1].copy()]


def action_segments(reverse):
    return [dict(frames=120, keys=['W', 'L' if reverse else 'J']),
            dict(frames=120, keys=['W', 'J' if reverse else 'L'])]


class CFGDeltaPairRuntime(QualityRuntime):
    def configure(self, case):
        require(any(case == expected for expected in CASES), 'only the two fixed pair cases are supported')
        super().configure(case)
        self.cfg_delta_windows = []
        self.pair_failed = False

    def generate(self, first, prompt, keys, noise, steps, *, fastpath=False):
        from training.demo.action_backend import tensor_sha
        index = len(self.cfg_delta_windows)
        require(not self.pair_failed and index < 3 and steps == 40 and fastpath is False
                and self.cfg_scale == 5 and self.solver == 'flow_euler', 'one fixed3window run;no retry')
        frames = WINDOW_FRAMES[index]
        require(tuple(first.shape) == (1, 1, 48, 30, 52) and tuple(noise.shape) == (1, frames // 4 + 1, 48, 30, 52)
                and tuple(keys.shape) == (1, frames, 8), 'original window6 latent/action geometry differs')
        if index == 0:
            check_original_input_digests(first, noise)
        check = getattr(self, 'authority_check', None)
        if check:
            check()
        row = dict(chunk=index, future_frames=frames, latent_frames=frames // 4 + 1,
            first_latent_sha256_float32=tensor_sha(first), noise_sha256_float32=tensor_sha(noise),
            action_sha256_float32=tensor_sha(keys), status='attempted', trace=[], stats={})
        self.cfg_delta_windows.append(row)
        def traced(trace):
            row['trace'] = trace
            callback = getattr(self, 'trace_callback', None)
            if callback:
                callback(copy.deepcopy(self.cfg_delta_windows))
        original = self.model
        flow = CFGDeltaFlow(original, self.negative.to(device='cuda', dtype=self.torch.bfloat16),
                            expected_latent_frames=frames // 4 + 1, on_trace=traced)
        self.model = flow
        try:
            latent = _WanRuntime.generate(self, first, prompt, keys, noise, steps, fastpath=False)
            require(flow.completed_steps == len(flow.trace) == 40 and flow.forward_calls == 64
                    and flow.positive_calls == 40 and flow.negative_calls == 24, 'incomplete CFG-delta window')
            row['status'] = 'completed'
            return latent
        except BaseException as error:
            self.pair_failed = True
            row.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            self.model = original
            self.forward_calls += flow.forward_calls
            row['trace'] = copy.deepcopy(flow.trace)
            row['actual_model_forward_calls'] = flow.forward_calls
            row['stats'] = dict(completed_logical_steps=flow.completed_steps,
                positive_forward_calls=flow.positive_calls, negative_forward_calls=flow.negative_calls,
                cached_steps=sum(item['approximate'] and item['status'] == 'completed' for item in flow.trace),
                peak_delta_cache_bytes=flow.peak_cache_bytes, cross_window_reuse=False)
            flow.close()
            row['stats']['cache_released_before_decode'] = flow.closed and flow.delta is None
            callback = getattr(self, 'trace_callback', None)
            if callback:
                callback(copy.deepcopy(self.cfg_delta_windows))


def annotate_pair(report, case, runtime, negative_source, cold):
    annotate_report(report, case, runtime, negative_source, cold)
    report.update(cfg_delta_policy=copy.deepcopy(POLICY), cfg_delta_windows=copy.deepcopy(runtime.cfg_delta_windows),
                  approximate_inference=True, reverse_timeline=case['reverse_timeline'])
    report['sampler'].update(model_forwards_per_step='2 exact / 1 approximate;see each window trace',
                            actual_model_forward_calls=runtime.forward_calls)
    report['mathematical_changes'].update(text_cfg='exact:vu+5*(vc-vu);approximate:vc_current+4*delta_previous',
        same_action_and_image_condition_in_both_text_branches='true when both evaluated;delta bound within one window',
        guidance_delta_reuse=True, sampler_integration='unchanged originalBF16 Euler and timestep labels')
    return report


def run(request, directory, lease):
    import numpy as np
    from PIL import Image
    from training.demo.backend import validate_active_authority
    from training.studio.worker import read_source
    validate_request(request)
    directory = contained(directory, DATA_ROOT)
    require(directory.name == request['job_id'] and directory.is_dir()
            and not any((directory / name).exists() for name in ('inputs.json', 'progress.json', *[c['name'] for c in CASES])),
            'new pair job required')
    began, runtime, engine, cold = time.monotonic(), None, None, 0.
    results = []
    def check():
        require(time.monotonic() - began < request['max_seconds'] - 30, 'finite pair deadline;no retry')
        validate_active_authority(lease)
    check()
    state, config, prompt, lineage, initial, original_actions, scene = read_source(request)
    require(lineage['sha256'] == R005_SHA and lineage['step'] == 1040 and scene['seed'] == 42
            and initial.shape == (HEIGHT, WIDTH, 3) and initial.dtype == np.uint8, 'original R005 scene required')
    actions = planned_actions(original_actions)
    negative, negative_source = load_negative(request['negative_cache'], config.base_model_path)
    inputs = dict(mode=FIXED['mode'], source_job=request['source_job'], source_actions=scene['action_segments'],
        prompt=scene['prompt'], scene_id=scene['scene_id'], prompt_condition=lineage['prompt_condition'],
        checkpoint=lineage['path'], checkpoint_sha256=R005_SHA, checkpoint_step=1040,
        negative_prompt=negative_source, cases=[dict(case, actual_actions=action_segments(case['reverse_timeline'])) for case in CASES],
        seed=42, width=WIDTH, height=HEIGHT, residency='co_resident', fastpath=False, window6=True,
        expected_frames_per_video=241, expected_model_forwards_per_video=192, expected_model_forwards_total=384,
        source_revision=request['source_revision'], experiment_revision=REVISION, cfg_delta_policy=POLICY,
        expected_first_window_digests=EXPECTED_INPUTS, approximate_inference=True, default_changed=False,
        quality_improvement_verified=False, speed_improvement_verified=False,
        comparison='same image/text/seed/noise;reverse changes action timeline only;original baselines not rerun')
    write_json(directory / 'inputs.json', inputs)
    Image.fromarray(initial).save(directory / 'initial.png')
    write_json(directory / 'progress.json', dict(status='running', phase='loading', cases=[]))
    try:
        check()
        tick = time.monotonic()
        runtime = CFGDeltaPairRuntime(state, config, 'co_resident', negative)
        runtime.synchronize()
        cold = time.monotonic() - tick
        runtime.authority_check = check
        engine = ReusableActionEngine(state, config, residency='co_resident', fastpath=False, _runtime=runtime)
        engine.load_seconds = cold
        for case, keys in zip(CASES, actions):
            check()
            require(not results or request['max_seconds'] - (time.monotonic() - began) >= 600,
                    'insufficient600second reserve for second15s case;no partial restart')
            target = directory / case['name']
            target.mkdir(exist_ok=False)
            runtime.configure(case)
            def record_trace(windows):
                write_json(directory / 'progress.json', dict(status='running', phase='denoising',
                    case=case['name'], cfg_delta_windows=windows, completed_cases=[entry['name'] for entry in results]))
            runtime.trace_callback = record_trace
            started = time.monotonic()
            try:
                report = engine.run(initial=initial, prompt=prompt, actions=keys.copy(), seed=42,
                    output_path=target / 'raw.mp4', steps=40, window6=True)
                annotate_pair(report, case, runtime, negative_source, cold)
                require(report['frames_written'] == 241 and runtime.forward_calls == 192
                        and len(runtime.cfg_delta_windows) == 3, 'pair requires241frames/192forwards/3windows')
                require(all(report['chunks'][0][key] == value for key, value in EXPECTED_INPUTS.items()),
                        'first window original input mismatch')
                if results:
                    require([row['noise_sha256_float32'] for row in report['chunks']] ==
                            [row['noise_sha256_float32'] for row in results[0]['generation']['chunks']],
                            'reverse must reuse the identical per-window noise sequence')
                entry = dict(**case, status='completed', actual_actions=action_segments(case['reverse_timeline']),
                    prompt=scene['prompt'], seed=42, generation=report, raw_video=str(target / 'raw.mp4'),
                    elapsed_seconds=time.monotonic() - started, elapsed_seconds_scope='case_generation_excludes_one_time_load',
                    quality_evaluation='pending_full_paired_visual_review', quality_improvement_verified=False,
                    speed_improvement_verified=False, automatic_retry=False)
                write_json(target / 'result.json', entry)
                results.append(entry)
            except BaseException as error:
                report = copy.deepcopy(engine.last_run_report or {})
                annotate_pair(report, case, runtime, negative_source, cold)
                entry = dict(**case, status='failed', generation=report, error=f'{type(error).__name__}: {error}', automatic_retry=False)
                write_json(target / 'result.json', entry)
                results.append(entry)
                raise
    except BaseException as error:
        write_json(directory / 'progress.json', dict(status='failed', cases=results,
            error=f'{type(error).__name__}: {error}', automatic_retry=False))
        raise
    finally:
        if engine is not None:
            engine.close()
        elif runtime is not None:
            runtime.close()
    result = dict(status='completed', mode=FIXED['mode'], cases=results, load_seconds=cold,
        actual_model_forward_calls=sum(entry['generation']['actual_model_forward_calls'] for entry in results),
        elapsed_seconds=time.monotonic() - began, default_changed=False, original_demo_changed=False,
        approximate_inference=True, quality_improvement_verified=False, speed_improvement_verified=False,
        automatic_retry=False, realtime=False)
    require(result['actual_model_forward_calls'] == 384, 'two complete clips must use384model calls')
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
