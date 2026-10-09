"""One R005 six-second candidate changing only model timestep multiplication precision.

Original BF16 sigma nodes, delta-sigma and latent updates remain in the existing
Euler implementation. Only the label entering both CFG model branches is FP32.
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
from training.longforcing_lite import euler_sigmas
from training.studio.overlap_probe import DATA_ROOT, HEIGHT, R005_SHA, R005_SOURCE_JOB, SETTINGS, WIDTH
from training.studio.quality_probe import DEFAULT_NEGATIVE_CACHE, QualityRuntime, TextCFGFlow, annotate_report, load_negative

CASE = dict(name='fp32_timestep', mode='segment6', **SETTINGS)
REVISION = 'R005-FP32-time-labels-original-BF16-Euler6s-v1'
EXPECTED_INPUTS = dict(
    first_latent_sha256_float32='1f73ca78ee570b6fc1a0cadd89526faeef6332aa2e0a7666886013970ec9eb48',
    noise_sha256_float32='a5464086b10298db73a58e0f014482619aa4082ed45f32edb0aa2be5c4da8bd0')
FIXED = dict(mode='timestep_precision_probe', max_seconds=600, source_job=str(R005_SOURCE_JOB),
    checkpoint_sha256=R005_SHA, negative_cache=DEFAULT_NEGATIVE_CACHE, residency='co_resident', fastpath=False,
    width=WIDTH, height=HEIGHT, seed=42, future_frames=96, fp32_accumulation=False, **SETTINGS, **EXPECTED_INPUTS)
POLICY = dict(old='BF16(sigma_BF16 * 1000)', new='float32(sigma_BF16) * 1000', clean_first_slot=0,
    sigma_nodes_dtype='torch.bfloat16', sigma_nodes_changed=False, sigma_max_absolute_difference=0.,
    delta_sigma_dtype='torch.bfloat16', latent_update_dtype='torch.bfloat16', fp32_accumulation=False,
    integration='unchanged action_backend.euler_rollout default;not a new solver or schedule',
    same_new_timestep_for_positive_and_negative=True, action_scale_all_steps=.03)


def validate_request(request):
    require(set(request) == set(FIXED) | {'job_id', 'source_revision', 'experiment_revision'}, 'unknown timestep precision fields')
    require(all(request[key] == value and type(request[key]) is type(value) for key, value in FIXED.items()),
            'timestep probe fixes originalR005/WJ96/40EulerCFG5/BF16integration/600seconds')
    require(isinstance(request['job_id'], str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', request['job_id'])
            and isinstance(request['source_revision'], str) and bool(request['source_revision'])
            and request['experiment_revision'] == REVISION, 'invalid timestep probe identity')


def check_original_input_digests(first, noise):
    from training.demo.action_backend import tensor_sha
    actual = dict(first_latent_sha256_float32=tensor_sha(first), noise_sha256_float32=tensor_sha(noise))
    require(actual == EXPECTED_INPUTS, 'original first latent/noise differs;do not generate this candidate')
    return actual


class TimestepPrecisionFlow(TextCFGFlow):
    def __init__(self, model, scale, negative, sigmas, *, on_trace=None):
        import torch
        require(type(scale) in (int, float) and scale == 5 and negative is not None, 'fixed CFG5 and negative required')
        require(torch.is_tensor(sigmas) and sigmas.dtype == torch.bfloat16 and tuple(sigmas.shape) == (41,)
                and bool(torch.isfinite(sigmas).all()) and sigmas[0].item() == 1 and sigmas[-1].item() == 0,
                'original finite41 BF16 sigma nodes required')
        super().__init__(model, scale, negative)
        self.sigmas = sigmas.detach().clone()
        self.sigma_values = self.sigmas.float().cpu().tolist()
        self.trace, self.completed_steps, self.failed, self.on_trace = [], 0, False, on_trace

    def __call__(self, current, *, conditional_dict, timestep, **kwargs):
        import torch
        require(not self.failed and len(self.trace) < 40, 'no retry or more than40 logical timestep calls')
        require(torch.is_tensor(current) and current.dtype == torch.bfloat16 and current.shape[:2] == (1, 25)
                and tuple(timestep.shape) == (1, 25) and timestep.dtype == torch.bfloat16
                and timestep.device == current.device == self.sigmas.device, 'original6s BF16 state/time contract differs')
        require(type(conditional_dict) is dict and conditional_dict.get('act_context_scale') == .03
                and conditional_dict.get('act_context') is not None
                and kwargs == {'replace_first_timestep_and_noise_latents': True}, 'plain original action/image conditions required')
        step, sigma = len(self.trace), self.sigmas[len(self.trace)]
        expected_old = (sigma * 1000).expand(timestep.shape).clone()
        expected_old[:, 0] = 0
        require(torch.equal(timestep, expected_old), 'sampler step count or original BF16 time labels differ')
        new = (sigma.float() * 1000).expand(timestep.shape).clone()
        new[:, 0] = 0
        conditions = dict(conditional_dict)
        row = dict(step=step, sigma_bf16=float(sigma.float()), sigma_dtype=str(sigma.dtype),
            old_timestep=timestep.float().cpu().tolist(), new_timestep=new.cpu().tolist(),
            old_timestep_dtype=str(timestep.dtype), new_timestep_dtype=str(new.dtype),
            model_input_dtype=str(current.dtype), model_input_shape=list(current.shape),
            max_abs_timestep_difference=float((new - timestep.float()).abs().max()),
            sigma_max_absolute_difference=0., original_time_labels_verified=True,
            same_new_timestep_for_positive_and_negative=True,
            same_original_action_context=conditions['act_context'] is conditional_dict['act_context'],
            act_context_scale=.03, status='attempted', model_forward_calls_before=self.forward_calls)
        self.trace.append(row)
        try:
            if self.on_trace is not None:
                self.on_trace(copy.deepcopy(self.trace))
            output = super().__call__(current, conditional_dict=conditions, timestep=new, **kwargs)
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


class TimestepPrecisionRuntime(QualityRuntime):
    def configure(self, case):
        require(all(case.get(key) == value and type(case.get(key)) is type(value) for key, value in SETTINGS.items()),
                'only fixed40EulerCFG5/LoRA1/action0.03 is supported')
        super().configure(case)
        self.timestep_precision_trace, self.sigma_nodes = [], []
        self.timestep_completed_steps = 0

    def generate(self, first, prompt, keys, noise, steps, *, fastpath=False):
        require(type(steps) is int and steps == 40 and fastpath is False
                and self.cfg_scale == 5 and self.solver == 'flow_euler' and not self.timestep_precision_trace,
                'one fixed40Euler segment only;no retry')
        self.input_digests = check_original_input_digests(first, noise)
        torch = self.torch
        # These nodes only reconstruct labels. The existing generator constructs
        # its own identical nodes and retains its original BF16 integration path.
        nodes = euler_sigmas(40, shift=5., device=torch.device('cuda'), dtype=torch.bfloat16)
        negative = self.negative.to(device='cuda', dtype=torch.bfloat16)
        original_model = self.model
        flow = TimestepPrecisionFlow(original_model, 5, negative, nodes, on_trace=getattr(self, 'trace_callback', None))
        self.model = flow
        try:
            latent = _WanRuntime.generate(self, first, prompt, keys, noise, steps, fastpath=False)
            require(len(flow.trace) == flow.completed_steps == 40 and flow.forward_calls == 80
                    and latent.dtype == torch.bfloat16, '40guided/80model calls and originalBF16 output required')
            return latent
        finally:
            self.model = original_model
            self.forward_calls += flow.forward_calls
            self.timestep_precision_trace = copy.deepcopy(flow.trace)
            self.timestep_completed_steps = flow.completed_steps
            self.sigma_nodes = flow.sigma_values


def annotate_precision(report, runtime, negative_source, load_seconds):
    annotate_report(report, CASE, runtime, negative_source, load_seconds)
    trace = copy.deepcopy(getattr(runtime, 'timestep_precision_trace', []))
    report.update(timestep_precision_policy=copy.deepcopy(POLICY), timestep_precision_trace=trace,
        sigma_nodes_bf16_as_float32=copy.deepcopy(getattr(runtime, 'sigma_nodes', [])),
        completed_logical_denoising_steps=getattr(runtime, 'timestep_completed_steps', 0),
        maximum_timestep_label_difference=max((row['max_abs_timestep_difference'] for row in trace), default=0.),
        changed_timestep_steps=sum(row['max_abs_timestep_difference'] > 0 for row in trace))
    report['mathematical_changes']['timestep_label'] = 'originalBF16sigma.float()*1000;no extraBF16 multiplication rounding;firstslot0'
    report['sampler'].update(fp32_accumulation=False, timestep_label_dtype='float32', sigma_dtype='bfloat16',
                            delta_sigma_dtype='bfloat16', latent_update_dtype='bfloat16')
    return report


def run(request, directory, lease):
    import numpy as np
    from PIL import Image
    from training.demo.backend import validate_active_authority
    from training.studio.worker import read_source
    validate_request(request)
    directory = contained(directory, DATA_ROOT)
    require(directory.name == request['job_id'] and directory.is_dir()
            and not any((directory / name).exists() for name in ('inputs.json', 'progress.json', CASE['name'])), 'new timestep job required')
    began, runtime, engine = time.monotonic(), None, None

    def check():
        require(time.monotonic() - began < 570, 'timestep probe deadline;no retry')
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
    inputs = dict(mode='timestep_precision_probe', source_job=request['source_job'], source_actions=scene['action_segments'],
        actual_actions=[dict(frames=96, keys=['W', 'J'])], prompt=scene['prompt'], scene_id=scene['scene_id'],
        prompt_condition=copy.deepcopy(lineage['prompt_condition']), checkpoint=lineage['path'], checkpoint_sha256=R005_SHA,
        checkpoint_step=1040, negative_prompt=negative_source, cases=[copy.deepcopy(CASE)], seed=42, width=WIDTH, height=HEIGHT,
        residency='co_resident', fastpath=False, source_revision=request['source_revision'], experiment_revision=REVISION,
        timestep_precision_policy=copy.deepcopy(POLICY), timestep_precision_trace=[], expected_input_digests=EXPECTED_INPUTS,
        prompt_cache_unchanged=True, expected_frames=97, default_changed=False)
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
            timestep_precision_policy=POLICY, timestep_precision_trace=trace))

    try:
        check()
        tick = time.monotonic()
        runtime = TimestepPrecisionRuntime(state, config, 'co_resident', negative)
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
        annotate_precision(report, runtime, negative_source, cold)
        require(report['frames_written'] == 97 and all(report[key] == value for key, value in EXPECTED_INPUTS.items()),
                'generated input hashes or97frame count changed')
        inputs.update(timestep_precision_trace=copy.deepcopy(runtime.timestep_precision_trace),
            sigma_nodes_bf16_as_float32=runtime.sigma_nodes, verified_input_digests=runtime.input_digests)
        write_json(directory / 'inputs.json', inputs)
        write_json(target / 'inputs.json', inputs)
        entry = dict(**CASE, status='completed', prompt=scene['prompt'], seed=42, generation=report,
            raw_video=str(target / 'raw.mp4'), elapsed_seconds=time.monotonic() - started,
            elapsed_seconds_scope='case_run_excludes_model_load', timestep_precision_policy=copy.deepcopy(POLICY),
            timestep_precision_trace=copy.deepcopy(runtime.timestep_precision_trace), quality_evaluation='pending_visual_comparison',
            quality_improvement_verified=False, automatic_retry=False, realtime=False)
        write_json(target / 'result.json', entry)
    except BaseException as error:
        failed = copy.deepcopy(engine.last_run_report or {}) if engine is not None else {}
        if runtime is not None:
            annotate_precision(failed, runtime, negative_source, cold)
            inputs.update(timestep_precision_trace=copy.deepcopy(getattr(runtime, 'timestep_precision_trace', [])),
                sigma_nodes_bf16_as_float32=copy.deepcopy(getattr(runtime, 'sigma_nodes', [])),
                verified_input_digests=getattr(runtime, 'input_digests', None))
        write_json(directory / 'inputs.json', inputs)
        write_json(target / 'inputs.json', inputs)
        entry = dict(**CASE, status='failed', prompt=scene['prompt'], generation=failed,
            error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic() - started,
            timestep_precision_policy=copy.deepcopy(POLICY), timestep_precision_trace=inputs['timestep_precision_trace'], automatic_retry=False)
        write_json(target / 'result.json', entry)
        write_json(directory / 'progress.json', dict(status='failed', cases=[entry]))
        raise
    finally:
        if engine is not None:
            engine.close()
        elif runtime is not None:
            runtime.close()
    result = dict(status='completed', mode='timestep_precision_probe', cases=[entry],
        timestep_precision_policy=copy.deepcopy(POLICY), timestep_precision_trace=entry['timestep_precision_trace'],
        actual_model_forward_calls=80, completed_logical_denoising_steps=40, load_seconds=cold,
        elapsed_seconds=time.monotonic() - began, default_changed=False, original_demo_changed=False,
        quality_evaluation='pending_visual_comparison', quality_improvement_verified=False, automatic_retry=False, realtime=False)
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
