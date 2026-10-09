"""One approximate CFG-delta reuse candidate;not a Studio default or TeaCache.

All forty positive branches are fresh. Negative branches are evaluated at24
steps;the remaining16 reuse only the immediately preceding guidance delta.
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
from training.studio.quality_probe import DEFAULT_NEGATIVE_CACHE, QualityRuntime, annotate_report, load_negative
from training.studio.timestep_precision_probe import EXPECTED_INPUTS, check_original_input_digests

CASE = dict(name='cfg_delta_cached', mode='segment6', **SETTINGS)
REVISION = 'R005-CFG5-adjacent-delta-reuse-middle32-6s-v1'
REUSE_STEPS = tuple(range(5, 36, 2))
POLICY = dict(kind='one_step_CFG_delta_reuse', steps=40, index_base=0,
    exact_edge_steps=[0, 1, 2, 3, 36, 37, 38, 39], reuse_steps=list(REUSE_STEPS),
    positive_forwards=40, negative_forwards=24, model_forwards=64, max_cache_age_steps=1,
    exact_formula='v_negative + 5 * (v_positive - v_negative)',
    approximate_formula='v_positive_current + 4 * cached_delta_previous',
    delta_definition='v_positive - v_negative from the same previous exact step',
    delta_dtype='torch.bfloat16', cross_window_reuse=False, approximate_inference=True)
FIXED = dict(mode='cfg_delta_probe', max_seconds=600, source_job=str(R005_SOURCE_JOB),
    checkpoint_sha256=R005_SHA, negative_cache=DEFAULT_NEGATIVE_CACHE, residency='co_resident', fastpath=False,
    width=WIDTH, height=HEIGHT, seed=42, future_frames=96, **SETTINGS, **EXPECTED_INPUTS)


def validate_request(request):
    require(type(request) is dict and set(request) == set(FIXED) |
            {'job_id', 'source_revision', 'experiment_revision'}, 'unknown CFG-delta fields')
    require(all(request[key] == value and type(request[key]) is type(value) for key, value in FIXED.items()),
            'CFG-delta probe fixes originalR005/WJ96/40EulerCFG5/600seconds')
    require(type(request['job_id']) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', request['job_id'])
            and type(request['source_revision']) is str and bool(request['source_revision'])
            and request['experiment_revision'] == REVISION, 'invalid CFG-delta identity')


def _stamp(value):
    return (id(value), value.data_ptr(), tuple(value.shape), str(value.dtype), str(value.device), value._version)


class CFGDeltaFlow:
    def __init__(self, model, negative, *, on_trace=None, expected_latent_frames=25):
        import torch
        require(negative is not None and torch.is_tensor(negative) and negative.dtype == torch.bfloat16,
                'verified BF16 negative embedding required')
        require(model.training is False and not any(p.requires_grad for p in model.parameters()),
                'CFG-delta requires frozen eval model')
        require(type(expected_latent_frames) is int and expected_latent_frames in (13, 25),
                'CFG-delta supports explicit3s/6s windows only')
        self.expected_latent_frames = expected_latent_frames
        self.model, self.negative, self.on_trace = model, negative, on_trace
        self.forward_calls = self.positive_calls = self.negative_calls = self.completed_steps = 0
        self.trace, self._binding = [], None
        self.delta, self.delta_step, self.peak_cache_bytes = None, None, 0
        self.failed = self.closed = False

    def __call__(self, current, *, conditional_dict, timestep, **kwargs):
        import torch
        require(not self.closed and not self.failed and len(self.trace) < 40 and not torch.is_grad_enabled(),
                'CFG-delta scope is no_grad,one window40steps,no retry')
        require(current.ndim == 5 and current.shape[:2] == (1, self.expected_latent_frames) and current.dtype == torch.bfloat16
                and timestep.shape == (1, self.expected_latent_frames) and timestep.dtype == torch.bfloat16
                and current.device == timestep.device and timestep[0, 0].item() == 0
                and kwargs == {'replace_first_timestep_and_noise_latents': True}, 'plain original window BF16 conditions required')
        require(type(conditional_dict) is dict and set(conditional_dict) == {'prompt_embeds', 'act_context', 'act_context_scale'}
                and conditional_dict['act_context_scale'] == .03, 'fixed original action/image/text conditions required')
        positive, actions = conditional_dict['prompt_embeds'], conditional_dict['act_context']
        require(isinstance(positive, (list, tuple)) and isinstance(actions, (list, tuple))
                and len(positive) == len(actions) == 1
                and all(torch.is_tensor(value) and value.dtype == current.dtype and value.device == current.device
                        for value in [*positive, *actions, self.negative]), 'one bound positive/action/negative context required')
        binding = (tuple(current.shape), str(current.dtype), str(current.device), id(positive), id(actions),
                   _stamp(positive[0]), _stamp(actions[0]), _stamp(self.negative), conditional_dict['act_context_scale'])
        if self._binding is None:
            self._binding = binding
        require(binding == self._binding, 'CFG-delta shape/precision/device/prompt/action binding changed')
        step, reused = len(self.trace), len(self.trace) in REUSE_STEPS
        require(not reused or (self.delta is not None and self.delta_step == step - 1), 'guidance delta must be exactly one step old')
        row = dict(step=step, approximate=reused, negative_recomputed=not reused,
            cached_delta_source_step=self.delta_step if reused else step,
            cache_age_steps=1 if reused else 0, positive_action_scale=.03,
            negative_action_scale=.03 if not reused else None, action_condition_unchanged=True,
            current_positive_negative_same_action_context=True if not reused else None,
            cached_delta_condition_binding_unchanged=True, timestep=float(timestep[0, 1]),
            input_shape=list(current.shape), input_dtype=str(current.dtype), delta_dtype='torch.bfloat16',
            status='attempted', model_forward_calls_before=self.forward_calls)
        self.trace.append(row)
        try:
            if self.on_trace:
                self.on_trace(copy.deepcopy(self.trace))
            self.forward_calls += 1
            self.positive_calls += 1
            vc = self.model(current, conditional_dict=conditional_dict, timestep=timestep, **kwargs)[0]
            require(vc.shape == current.shape and vc.dtype == torch.bfloat16, 'positive velocity geometry/dtype changed')
            if reused:
                velocity = vc + 4 * self.delta
            else:
                negative_conditions = dict(conditional_dict, prompt_embeds=[self.negative])
                self.forward_calls += 1
                self.negative_calls += 1
                vu = self.model(current, conditional_dict=negative_conditions, timestep=timestep, **kwargs)[0]
                require(vu.shape == current.shape and vu.dtype == torch.bfloat16, 'negative velocity geometry/dtype changed')
                delta = vc - vu
                require(bool(torch.isfinite(delta).all()), 'nonfinite guidance delta')
                self.delta, self.delta_step = delta.detach().clone(), step
                self.peak_cache_bytes = max(self.peak_cache_bytes, self.delta.numel() * self.delta.element_size())
                velocity = vu + 5 * delta  # Preserve the original expression and BF16 order on exact steps.
            self.completed_steps += 1
            row['status'] = 'completed'
            return velocity, None
        except BaseException as error:
            self.failed = True
            self.delta = None
            row.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            row.update(model_forward_calls_after=self.forward_calls,
                       attempted_model_forward_calls=self.forward_calls - row['model_forward_calls_before'])
            if self.on_trace:
                self.on_trace(copy.deepcopy(self.trace))

    def close(self):
        self.delta = self.model = self.negative = self._binding = None
        self.closed = True


class CFGDeltaRuntime(QualityRuntime):
    def configure(self, case):
        require(all(case.get(key) == value and type(case.get(key)) is type(value) for key, value in SETTINGS.items()),
                'fixed40EulerCFG5/LoRA1/action0.03 required')
        super().configure(case)
        self.cfg_delta_trace, self.cfg_delta_stats = [], {}

    def generate(self, first, prompt, keys, noise, steps, *, fastpath=False):
        require(type(steps) is int and steps == 40 and fastpath is False and not self.cfg_delta_trace
                and self.cfg_scale == 5 and self.solver == 'flow_euler', 'one fixed CFG-delta segment;no retry')
        check_original_input_digests(first, noise)
        negative = self.negative.to(device='cuda', dtype=self.torch.bfloat16)
        original = self.model
        flow = CFGDeltaFlow(original, negative, on_trace=getattr(self, 'trace_callback', None))
        self.model = flow
        try:
            latent = _WanRuntime.generate(self, first, prompt, keys, noise, steps, fastpath=False)
            require(flow.completed_steps == len(flow.trace) == 40 and flow.forward_calls == 64
                    and flow.positive_calls == 40 and flow.negative_calls == 24,
                    'requires40logical/64actualcalls/16cached steps')
            return latent
        finally:
            self.model = original
            self.forward_calls += flow.forward_calls
            self.cfg_delta_trace = copy.deepcopy(flow.trace)
            self.cfg_delta_stats = dict(completed_logical_steps=flow.completed_steps,
                positive_forward_calls=flow.positive_calls, negative_forward_calls=flow.negative_calls,
                cached_steps=sum(row['approximate'] and row['status'] == 'completed' for row in flow.trace),
                peak_delta_cache_bytes=flow.peak_cache_bytes, cross_window_reuse=False)
            flow.close()
            self.cfg_delta_stats['cache_released_before_decode'] = flow.delta is None and flow.closed


def annotate_delta(report, runtime, negative_source, load_seconds):
    annotate_report(report, CASE, runtime, negative_source, load_seconds)
    report.update(cfg_delta_policy=copy.deepcopy(POLICY), cfg_delta_trace=copy.deepcopy(runtime.cfg_delta_trace),
                  cfg_delta_stats=copy.deepcopy(runtime.cfg_delta_stats), approximate_inference=True)
    report['sampler'].update(model_forwards_per_step='2 exact / 1 approximate;see cfg_delta_trace',
                            actual_model_forward_calls=runtime.forward_calls)
    report['mathematical_changes'].update(text_cfg='exact:vu+5*(vc-vu);approximate:vc_current+4*delta_previous',
        same_action_and_image_condition_in_both_text_branches='true when both evaluated;reused delta has identical bound conditions',
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
            and not any((directory / name).exists() for name in ('inputs.json', 'progress.json', CASE['name'])), 'new CFG-delta job required')
    began, runtime, engine, cold = time.monotonic(), None, None, 0.
    def check():
        require(time.monotonic() - began < 570, 'CFG-delta finite deadline;no retry')
        validate_active_authority(lease)
    check()
    state, config, prompt, lineage, initial, actions, scene = read_source(request)
    require(lineage['sha256'] == R005_SHA and lineage['step'] == 1040 and scene['seed'] == 42
            and initial.shape == (HEIGHT, WIDTH, 3) and initial.dtype == np.uint8, 'original R005 scene required')
    selected = actions[:96].copy()
    expected = np.zeros((96, 8), np.float32)
    expected[:, [0, 5]] = 1
    require(selected.dtype == np.float32 and np.array_equal(selected, expected), 'original WJ96 actions required')
    negative, negative_source = load_negative(request['negative_cache'], config.base_model_path)
    inputs = dict(mode=FIXED['mode'], source_job=request['source_job'], source_actions=scene['action_segments'],
        actual_actions=[dict(frames=96, keys=['W', 'J'])], prompt=scene['prompt'], scene_id=scene['scene_id'],
        prompt_condition=lineage['prompt_condition'], checkpoint=lineage['path'], checkpoint_sha256=R005_SHA,
        checkpoint_step=1040, negative_prompt=negative_source, cases=[CASE], seed=42, width=WIDTH, height=HEIGHT,
        residency='co_resident', fastpath=False, source_revision=request['source_revision'], experiment_revision=REVISION,
        cfg_delta_policy=POLICY, expected_input_digests=EXPECTED_INPUTS, approximate_inference=True,
        expected_frames=97, default_changed=False, quality_improvement_verified=False, speed_improvement_verified=False)
    write_json(directory / 'inputs.json', inputs)
    Image.fromarray(initial).save(directory / 'initial.png')
    target = directory / CASE['name']
    target.mkdir(exist_ok=False)
    write_json(directory / 'progress.json', dict(status='running', phase='loading'))
    def record_trace(trace):
        write_json(directory / 'progress.json', dict(status='running', phase='denoising', cfg_delta_trace=trace))
    started = time.monotonic()
    try:
        check()
        tick = time.monotonic()
        runtime = CFGDeltaRuntime(state, config, 'co_resident', negative)
        runtime.synchronize()
        cold = time.monotonic() - tick
        runtime.configure(CASE)
        runtime.trace_callback = record_trace
        engine = ReusableActionEngine(state, config, residency='co_resident', fastpath=False, _runtime=runtime)
        engine.load_seconds = cold
        check()
        started = time.monotonic()
        report, endpoint = engine.run_segment(initial=initial, prompt=prompt, actions=selected, seed=42,
            output_path=target / 'raw.mp4', steps=40)
        del endpoint
        annotate_delta(report, runtime, negative_source, cold)
        require(report['frames_written'] == 97 and all(report[key] == value for key, value in EXPECTED_INPUTS.items()),
                'original first/noise or97frame count differs')
        entry = dict(**CASE, status='completed', prompt=scene['prompt'], seed=42, generation=report,
            raw_video=str(target / 'raw.mp4'), elapsed_seconds=time.monotonic() - started,
            elapsed_seconds_scope='case_generation_excludes_one_time_load', quality_evaluation='pending_visual_comparison',
            quality_improvement_verified=False, speed_improvement_verified=False, automatic_retry=False)
        write_json(target / 'result.json', entry)
    except BaseException as error:
        failed = copy.deepcopy(engine.last_run_report or {}) if engine else {}
        if runtime is not None:
            annotate_delta(failed, runtime, negative_source, cold)
        entry = dict(**CASE, status='failed', generation=failed, error=f'{type(error).__name__}: {error}', automatic_retry=False)
        write_json(target / 'result.json', entry)
        write_json(directory / 'progress.json', dict(status='failed', cases=[entry]))
        raise
    finally:
        if engine is not None:
            engine.close()
        elif runtime is not None:
            runtime.close()
    result = dict(status='completed', mode=FIXED['mode'], cases=[entry], load_seconds=cold,
        elapsed_seconds=time.monotonic() - began, default_changed=False, original_demo_changed=False,
        approximate_inference=True, quality_improvement_verified=False, speed_improvement_verified=False,
        automatic_retry=False, realtime=False)
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
