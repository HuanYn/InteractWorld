"""One fixed six-second R005 candidate: action conditioning only in early denoising.

This changes conditioning mathematics, not video-time actions or compute cost.
It is isolated from default runtime/CFG implementations; no sweeps or retries.
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
from training.studio.overlap_probe import DATA_ROOT, HEIGHT, R005_SHA, R005_SOURCE_JOB, SETTINGS, WIDTH
from training.studio.quality_probe import DEFAULT_NEGATIVE_CACHE, QualityRuntime, TextCFGFlow, annotate_report, load_negative

CASE = dict(name='early20_action', mode='segment6', **SETTINGS)
REVISION = 'R005-action-first20-denoising-steps6s-v1'
FIXED = dict(mode='action_denoise_probe', max_seconds=600, source_job=str(R005_SOURCE_JOB),
    checkpoint_sha256=R005_SHA, negative_cache=DEFAULT_NEGATIVE_CACHE, residency='co_resident', fastpath=False,
    width=WIDTH, height=HEIGHT, seed=42, future_frames=96, action_active_steps=20, **SETTINGS)
SCHEDULE = dict(kind='denoising_iteration_action_scale', index_base=0, total_steps=40,
    ranges=[dict(start=0, stop_exclusive=20, scale=.03), dict(start=20, stop_exclusive=40, scale=0.)],
    same_scale_for_positive_and_negative=True, same_original_action_context=True,
    interpretation='denoising iteration index, NOT video timeline;adapter calculation is not skipped')


def validate_request(request):
    require(set(request) == set(FIXED) | {'job_id', 'source_revision', 'experiment_revision'}, 'unknown action denoise request fields')
    require(all(request[key] == value and type(request[key]) is type(value) for key, value in FIXED.items()),
            'action denoise probe fixes originalR005/WJ96/40EulerCFG5/early20/600seconds')
    require(isinstance(request['job_id'], str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', request['job_id'])
            and isinstance(request['source_revision'], str) and bool(request['source_revision'])
            and request['experiment_revision'] == REVISION, 'invalid action denoise identity')


class EarlyActionFlow(TextCFGFlow):
    """Exactly forty guided velocity calls; each retains two serial text branches."""

    def __init__(self, model, scale, negative, *, on_trace=None):
        require(type(scale) in (int, float) and scale == 5 and negative is not None, 'fixed CFG5 and verified negative required')
        super().__init__(model, scale, negative)
        self.trace, self.completed_steps, self.failed = [], 0, False
        self.on_trace = on_trace

    def __call__(self, *args, conditional_dict, **kwargs):
        require(not self.failed, 'failed action schedule cannot be retried')
        require(len(self.trace) < 40, 'action schedule permits exactly40 logical steps')
        require(type(conditional_dict) is dict and conditional_dict.get('act_context_scale') == .03
                and conditional_dict.get('act_context') is not None, 'original action condition must retain nominal0.03')
        step = len(self.trace)
        scale = .03 if step < 20 else 0.
        conditions = dict(conditional_dict, act_context_scale=scale)  # Never mutate the caller or its actions.
        row = dict(step=step, act_context_scale=scale, status='attempted', model_forward_calls_before=self.forward_calls,
            positive_action_scale=scale, negative_action_scale=scale,
            same_original_action_context=conditions['act_context'] is conditional_dict['act_context'],
            branch_condition_contract='TextCFGFlow changes only negative prompt;both branches receive this same action scale/context')
        self.trace.append(row)
        try:
            if self.on_trace is not None:
                self.on_trace(copy.deepcopy(self.trace))  # Persist the attempt before model work.
            output = super().__call__(*args, conditional_dict=conditions, **kwargs)
            self.completed_steps += 1
            row['status'] = 'completed'
            return output
        except BaseException as error:
            self.failed = True
            row.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            row['model_forward_calls_after'] = self.forward_calls
            row['attempted_model_forward_calls'] = self.forward_calls - row['model_forward_calls_before']
            if self.on_trace is not None:
                self.on_trace(copy.deepcopy(self.trace))


class EarlyActionRuntime(QualityRuntime):
    def configure(self, case):
        require(all(case.get(key) == value and type(case.get(key)) is type(value) for key, value in SETTINGS.items()),
                'early-action runtime only supports fixed40EulerCFG5/LoRA1/action0.03')
        super().configure(case)
        self.action_denoise_trace = []
        self.action_denoise_completed_steps = 0

    def generate(self, first, prompt, keys, noise, steps, *, fastpath=False):
        require(type(steps) is int and steps == 40 and fastpath is False
                and self.cfg_scale == 5 and self.solver == 'flow_euler', 'fixed40Euler action schedule only')
        require(not self.action_denoise_trace, 'one segment only;do not retry this runtime')
        negative = self.negative.to(device='cuda', dtype=self.torch.bfloat16)
        original_model = self.model
        flow = EarlyActionFlow(original_model, 5, negative, on_trace=getattr(self, 'trace_callback', None))
        self.model = flow
        try:
            # Bypass QualityRuntime.generate's normal TextCFGFlow wrapping, but keep
            # the exact original noise/sigmas/conditioning/FP32 policy of this path.
            latent = _WanRuntime.generate(self, first, prompt, keys, noise, steps, fastpath=False)
            require(len(flow.trace) == flow.completed_steps == 40 and flow.forward_calls == 80,
                    'actual early-action schedule did not complete40guided/80model calls')
            return latent
        finally:
            self.model = original_model
            self.forward_calls += flow.forward_calls
            self.action_denoise_trace = copy.deepcopy(flow.trace)
            self.action_denoise_completed_steps = flow.completed_steps


def annotate_schedule(report, runtime, negative_source, load_seconds):
    annotate_report(report, CASE, runtime, negative_source, load_seconds)
    report.update(act_context_scale_schedule=copy.deepcopy(SCHEDULE), nominal_action_scale=.03,
        action_scale_interpretation='nominal active-phase strength only;later20denoising steps are zero',
        action_denoise_trace=copy.deepcopy(getattr(runtime, 'action_denoise_trace', [])),
        completed_logical_denoising_steps=getattr(runtime, 'action_denoise_completed_steps', 0))
    report['mathematical_changes']['action_conditioning'] = 'act_context_scale0.03 at steps0..19;0 at steps20..39;both CFG branches identical schedule'
    return report


def run(request, directory, lease):
    import numpy as np
    from PIL import Image
    from training.demo.backend import validate_active_authority
    from training.studio.worker import read_source
    validate_request(request)
    directory = contained(directory, DATA_ROOT)
    require(directory.name == request['job_id'] and directory.is_dir()
            and not any((directory / name).exists() for name in ('inputs.json', 'progress.json', CASE['name'])),
            'new action denoise job directory required')
    began, runtime, engine = time.monotonic(), None, None

    def check():
        require(time.monotonic() - began < 570, 'action denoise deadline;no retry')
        validate_active_authority(lease)

    check()
    state, config, prompt, lineage, initial, actions, scene = read_source(request)
    require(lineage['sha256'] == R005_SHA and lineage['step'] == 1040 and scene['seed'] == 42
            and initial.shape == (HEIGHT, WIDTH, 3) and initial.dtype == np.uint8, 'actual originalR005 source differs')
    require(Path(config.base_model_path).resolve().is_relative_to(DATA_ROOT.resolve()), 'weights must remain on data disk')
    selected = actions[:96].copy()
    expected = np.zeros((96, 8), np.float32)
    expected[:, [0, 5]] = 1
    require(selected.dtype == np.float32 and np.array_equal(selected, expected), 'source first96 must remain WJ')
    negative, negative_source = load_negative(request['negative_cache'], config.base_model_path)
    inputs = dict(mode='action_denoise_probe', source_job=request['source_job'], source_actions=scene['action_segments'],
        actual_actions=[dict(frames=96, keys=['W', 'J'])], prompt=scene['prompt'], scene_id=scene['scene_id'],
        prompt_condition=copy.deepcopy(lineage['prompt_condition']), checkpoint=lineage['path'], checkpoint_sha256=R005_SHA,
        checkpoint_step=1040, negative_prompt=negative_source, cases=[copy.deepcopy(CASE)], seed=42, width=WIDTH, height=HEIGHT,
        residency='co_resident', fastpath=False, source_revision=request['source_revision'], experiment_revision=REVISION,
        act_context_scale_schedule=copy.deepcopy(SCHEDULE), action_denoise_trace=[], prompt_cache_unchanged=True,
        expected_frames=97, default_changed=False, action_success_verified=False, speed_improvement_verified=False)
    write_json(directory / 'inputs.json', inputs)
    Image.fromarray(initial).save(directory / 'initial.png')
    np.save(directory / 'actions.npy', selected, allow_pickle=False)
    target = contained(directory / CASE['name'], directory)
    target.mkdir(exist_ok=False)
    write_json(target / 'inputs.json', inputs)
    write_json(directory / 'progress.json', dict(status='running', phase='loading', cases=[]))
    cold, started = 0., time.monotonic()

    def record_trace(trace):
        write_json(directory / 'progress.json', dict(status='running', phase='denoising', case=CASE['name'],
            act_context_scale_schedule=SCHEDULE, action_denoise_trace=trace))

    try:
        check()
        tick = time.monotonic()
        runtime = EarlyActionRuntime(state, config, 'co_resident', negative)
        runtime.synchronize()
        cold = time.monotonic() - tick
        runtime.configure(CASE)
        runtime.trace_callback = record_trace
        engine = ReusableActionEngine(state, config, residency='co_resident', fastpath=False, _runtime=runtime)
        engine.load_seconds = cold
        check()
        started = time.monotonic()
        report, endpoint = engine.run_segment(initial=initial, prompt=prompt, actions=selected,
            seed=42, output_path=target / 'raw.mp4', steps=40)
        del endpoint
        annotate_schedule(report, runtime, negative_source, cold)
        require(report['frames_written'] == 97 and runtime.forward_calls == 80
                and runtime.action_denoise_completed_steps == 40, 'unexpected segment or schedule completion')
        inputs['action_denoise_trace'] = copy.deepcopy(runtime.action_denoise_trace)
        write_json(directory / 'inputs.json', inputs)
        write_json(target / 'inputs.json', inputs)
        entry = dict(**CASE, status='completed', prompt=scene['prompt'], seed=42, generation=report,
            raw_video=str(target / 'raw.mp4'), elapsed_seconds=time.monotonic() - started,
            elapsed_seconds_scope='case_run_excludes_model_load', act_context_scale_schedule=copy.deepcopy(SCHEDULE),
            action_denoise_trace=copy.deepcopy(runtime.action_denoise_trace), quality_evaluation='pending_visual_comparison',
            action_success_verified=False, speed_improvement_verified=False, automatic_retry=False, realtime=False)
        write_json(target / 'result.json', entry)
    except BaseException as error:
        failed = copy.deepcopy(engine.last_run_report or {}) if engine is not None else {}
        if runtime is not None:
            annotate_schedule(failed, runtime, negative_source, cold)
            inputs['action_denoise_trace'] = copy.deepcopy(getattr(runtime, 'action_denoise_trace', []))
        write_json(directory / 'inputs.json', inputs)
        write_json(target / 'inputs.json', inputs)
        entry = dict(**CASE, status='failed', prompt=scene['prompt'], generation=failed,
            error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic() - started,
            act_context_scale_schedule=copy.deepcopy(SCHEDULE), action_denoise_trace=inputs['action_denoise_trace'], automatic_retry=False)
        write_json(target / 'result.json', entry)
        write_json(directory / 'progress.json', dict(status='failed', cases=[entry]))
        raise
    finally:
        if engine is not None:
            engine.close()
        elif runtime is not None:
            runtime.close()
    result = dict(status='completed', mode='action_denoise_probe', cases=[entry],
        act_context_scale_schedule=copy.deepcopy(SCHEDULE), action_denoise_trace=entry['action_denoise_trace'],
        actual_model_forward_calls=80, completed_logical_denoising_steps=40, load_seconds=cold,
        elapsed_seconds=time.monotonic() - began, default_changed=False, original_demo_changed=False,
        quality_evaluation='pending_visual_comparison', action_success_verified=False, speed_improvement_verified=False,
        automatic_retry=False, realtime=False)
    write_json(directory / 'progress.json', result)
    return result


def prepare(*, output, source_revision):
    directory = contained(output, DATA_ROOT)
    request = dict(copy.deepcopy(FIXED), job_id=directory.name, source_revision=source_revision, experiment_revision=REVISION)
    validate_request(request)
    require(directory != DATA_ROOT.resolve() and not directory.exists(), 'new output directory required')
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'request.json', request)
    return dict(status='cpu_request_prepared', request=str(directory / 'request.json'), max_seconds=600,
                gpu_launched=False, model_loaded=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), ensure_ascii=False))


if __name__ == '__main__':
    main()
