"""Fixed first-step unit-scale LoRA qualification: warm A/B, timed A/B/B/A.

Six CFG forwards/twelve model calls, no sampling trajectory, video or training.
The existing Creator launcher owns the authorized GPU and180-second deadline.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import re
from statistics import median
import time

from training.demo.contracts import contained, require, write_json
from training.studio.operator_profile import first_step_inputs
from training.studio.overlap_probe import DATA_ROOT, HEIGHT, R005_SHA, R005_SOURCE_JOB, SETTINGS, WIDTH
from training.studio.quality_probe import DEFAULT_NEGATIVE_CACHE, QualityRuntime, TextCFGFlow, load_negative
from training.studio.text_kv_probe import difference

ORDER = (('warmup', 'A'), ('warmup', 'B'), ('timed', 'A'), ('timed', 'B'), ('timed', 'B'), ('timed', 'A'))
MIN_REDUCTION = .03
FIXED = dict(mode='lora_inplace_probe', max_seconds=180, source_job=str(R005_SOURCE_JOB),
    checkpoint_sha256=R005_SHA, negative_cache=DEFAULT_NEGATIVE_CACHE, residency='phase', fastpath=False,
    width=WIDTH, height=HEIGHT, seed=42, future_frames=48, warmup_order=['A', 'B'],
    timed_order=['A', 'B', 'B', 'A'], minimum_median_reduction=MIN_REDUCTION, **SETTINGS)


def validate_request(request):
    require(set(request) == set(FIXED) | {'job_id', 'source_revision', 'experiment_revision'}, 'unknown LoRA probe request fields')
    require(all(request[k] == v and type(request[k]) is type(v) for k, v in FIXED.items()),
            'LoRA probe fixes originalR005/CFG5/BF16/ABBA/3percent/180seconds')
    require(isinstance(request['job_id'], str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', request['job_id'])
            and isinstance(request['source_revision'], str) and bool(request['source_revision'])
            and request['experiment_revision'] == 'R005-unit-scale-inplace-LoRA-firststep-ABBA-v1', 'invalid probe identity')


def gate_summary(rows):
    require([(row['phase'], row['path']) for row in rows] == list(ORDER)
            and all(row['actual_model_forward_calls'] == 2 for row in rows), 'exact sixCFG/twelve-model-forward order required')
    require(all(math.isfinite(row['seconds']) and row['seconds'] > 0 for row in rows), 'invalid synchronized timing')
    all_equal = all(row['vs_warmup_A']['finite'] and row['vs_warmup_A']['torch_equal'] for row in rows)
    scopes_valid = all(row['scope_stats']['verified'] and row['scope_after_close']['restored']
                       for row in rows if row['path'] == 'B')
    measured = {name: [row['seconds'] for row in rows if row['phase'] == 'timed' and row['path'] == name]
                for name in ('A', 'B')}
    baseline, optimized = median(measured['A']), median(measured['B'])
    timing_passed = optimized <= baseline * (1 - MIN_REDUCTION)
    return dict(eligible_for_full_window_check=all_equal and timing_passed and scopes_valid,
        all_six_velocity_fields_bitwise_equal=all_equal, median_timing_gate_passed=timing_passed,
        actual_optimized_scope_calls_verified=scopes_valid,
        timed_seconds=measured, baseline_median_seconds=baseline, optimized_median_seconds=optimized,
        observed_median_reduction=1 - optimized / baseline, required_median_reduction=MIN_REDUCTION,
        rule='every warmup/timed velocity bitwise equals warmup_A AND median(timed_B)<=0.97*median(timed_A)',
        actual_model_forward_calls=sum(row['actual_model_forward_calls'] for row in rows),
        qualification_scope='single first diffusion step only;eligibility is not full-window or stable speedup evidence',
        quality_improvement_verified=False, steady_speedup_verified=False, default_enabled=False, automatic_retry=False)


def summarize_scope(during, after):
    """Check actual300-layer/600-call counters; retain compact evidence, not repeated per-layer dumps."""
    counters = dict(calls=2, completed_calls=2, base_calls=2, lora_a_calls=2, lora_b_calls=2)
    mismatches = [name for name, counts in during['modules'].items() if counts != counters]
    preserved = after['modules'] == during['modules'] and after['total_calls'] == during['total_calls']
    restored = after['closed'] is True and after['active_modules'] == 0 and preserved
    require(restored, 'optimized scope did not restore normal model forwards; stop without retry')
    verified = (during['kind'] == 'frozen_unit_scale_lora_inplace_add_v1'
        and during['modules_patched'] == during['active_modules'] == len(during['modules']) == 300
        and during['total_calls'] == during['completed_calls'] == 600 and during['closed'] is False
        and not mismatches and all(during[key] is False for key in
            ('merged_weights', 'quantized', 'rank_changed', 'parameters_modified', 'inputs_modified')))
    return (dict({key: value for key, value in during.items() if key != 'modules'}, verified=verified,
                 per_module_expected_counters=counters, counter_mismatch_module_names=mismatches),
            dict(closed=after['closed'], active_modules=after['active_modules'], total_calls=after['total_calls'],
                 completed_calls=after['completed_calls'], all_module_counters_preserved=preserved, restored=restored))


def one_forward(runtime, flow, current, kwargs, *, optimized):
    """Include optimized scope construction/enter/exit in synchronized wall time."""
    from training.studio.lora_inplace import InplaceUnitScaleLoRA
    torch = runtime.torch
    runtime.synchronize()
    torch.cuda.reset_peak_memory_stats()
    calls_before = flow.forward_calls
    tick, during, after = time.monotonic(), None, None
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        if optimized:
            with InplaceUnitScaleLoRA(runtime.model) as scope:
                velocity, _ = flow(current, **kwargs)
                runtime.synchronize()
                during = copy.deepcopy(scope.stats)
            after = copy.deepcopy(scope.stats)
        else:
            velocity, _ = flow(current, **kwargs)
        runtime.synchronize()
    seconds = time.monotonic() - tick
    calls = flow.forward_calls - calls_before
    require(calls == 2, 'each measured CFG must call positive and negative once')
    if optimized:
        during, after = summarize_scope(during, after)
    row = dict(seconds=seconds, actual_model_forward_calls=calls,
        peak_vram_bytes=int(torch.cuda.max_memory_allocated()), peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        scope_stats=during, scope_after_close=after,
        timing_scope='CUDA-synchronized CFG forward;B includes scope construction/enter/exit;CPU output copy excluded')
    return velocity.detach().cpu(), row


def run(request, directory, lease):
    import numpy as np
    import torch
    from training.demo import action_backend as action
    from training.demo.backend import validate_active_authority
    from training.longforcing_lite import euler_sigmas
    from training.models.action_adapter import build_action_context
    from training.studio.worker import read_source
    validate_request(request)
    directory = contained(directory, DATA_ROOT)
    require(directory.name == request['job_id'] and directory.is_dir()
            and not any((directory / name).exists() for name in ('inputs.json', 'progress.json', 'comparison.json')),
            'new probe directory required; preserve old evidence')
    began, runtime, flow, rows = time.monotonic(), None, None, []

    def check():
        require(time.monotonic() - began < 155, 'LoRA qualification deadline;no retry')
        validate_active_authority(lease)

    check()
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one leased CUDA GPU required')
    state, config, prompt, lineage, initial, actions, scene = read_source(request)
    require(lineage['sha256'] == R005_SHA and lineage['step'] == 1040 and scene['seed'] == 42
            and initial.shape == (HEIGHT, WIDTH, 3) and initial.dtype == np.uint8, 'actual originalR005 input differs')
    require(Path(config.base_model_path).resolve().is_relative_to(DATA_ROOT.resolve()), 'weights must remain on data disk')
    expected = np.zeros((48, 8), np.float32)
    expected[:, [0, 5]] = 1
    require(actions.dtype == np.float32 and np.array_equal(actions[:48], expected), 'first48 source actions must remainWJ')
    negative, negative_source = load_negative(DEFAULT_NEGATIVE_CACHE, config.base_model_path)
    inputs = dict(mode='lora_inplace_probe', source_job=request['source_job'], checkpoint=lineage['path'],
        checkpoint_sha256=R005_SHA, checkpoint_step=1040, scene_id=scene['scene_id'], prompt=scene['prompt'],
        negative_prompt=negative_source, seed=42, width=WIDTH, height=HEIGHT, settings=SETTINGS,
        precision='bf16', residency='phase', actual_actions=[dict(frames=48, keys=['W', 'J'])],
        source_revision=request['source_revision'], experiment_revision=request['experiment_revision'],
        warmup_order=request['warmup_order'], timed_order=request['timed_order'], model_forward_budget=12,
        minimum_median_reduction=request['minimum_median_reduction'],
        qualification_rule='all6samefirststepvelocitiesbitwiseequal AND median(timedB)<=0.97*median(timedA)',
        input_encode_count=1, model_resident_during_all_forwards=True, trajectory_steps_executed=0,
        modification='unit-scale LoRA only;remove multiply-by1 and unnecessary output allocation in bounded scope',
        profiler_enabled=False, text_kv_cache_enabled=False, video_generated=False, new_training=False, default_changed=False)
    write_json(directory / 'inputs.json', inputs)  # Gate is persisted before any measured outcomes.
    try:
        check()
        tick = time.monotonic()
        runtime = QualityRuntime(state, config, 'phase', negative)
        runtime.configure(SETTINGS)
        runtime.synchronize()
        cold = time.monotonic() - tick
        tick = time.monotonic()
        runtime.transfer('vae')
        runtime.begin()
        pixel = torch.from_numpy(initial.copy()).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).float().div(127.5).sub(1)
        first = runtime.encode(pixel).to(torch.bfloat16)
        runtime.synchronize()
        encode_seconds = time.monotonic() - tick
        check()
        runtime.transfer('model')
        runtime.synchronize()
        noise = torch.randn((1, 13, 48, 30, 52), generator=torch.Generator(device='cpu').manual_seed(42), dtype=torch.float32)
        sigmas = euler_sigmas(40, shift=5., device=torch.device('cuda'), dtype=torch.bfloat16)
        current, timestep = first_step_inputs(first.to('cuda'), noise.to('cuda', dtype=torch.bfloat16), sigmas)
        conditions = dict(prompt_embeds=[prompt.to('cuda', dtype=torch.bfloat16)], act_context_scale=.03,
            act_context=build_action_context(torch.from_numpy(expected).unsqueeze(0), height=HEIGHT, width=WIDTH,
                                            device='cuda', dtype=torch.bfloat16))
        inputs.update(first_latent_sha256_float32=action.tensor_sha(first), noise_sha256_float32=action.tensor_sha(noise),
            actual_timestep=timestep.detach().float().cpu().tolist(), model_input_shape=list(current.shape),
            identical_input_objects_for_all_runs=True)
        write_json(directory / 'inputs.json', inputs)
        flow = TextCFGFlow(runtime.model, 5, negative.to('cuda', dtype=torch.bfloat16))
        kwargs = dict(conditional_dict=conditions, timestep=timestep, replace_first_timestep_and_noise_latents=True)
        reference = None
        for index, (phase, name) in enumerate(ORDER):
            check()
            velocity, row = one_forward(runtime, flow, current, kwargs, optimized=name == 'B')
            require(velocity.shape == current.shape, 'velocity shape differs from fixedfirststep')
            if reference is None:
                reference = velocity.clone()
            row.update(index=index, phase=phase, path=name, vs_warmup_A=difference(reference, velocity))
            rows.append(row)
            del velocity
            write_json(directory / 'progress.json', dict(status='running', runs=rows, actual_model_forward_calls=flow.forward_calls))
        require(flow.forward_calls == 12, 'qualification must stop after12 actual model forwards')
        comparison = gate_summary(rows)
        write_json(directory / 'comparison.json', comparison)
    finally:
        flow = None
        if runtime is not None:
            runtime.close()
    result = dict(status='completed', mode='lora_inplace_probe', runs=rows, comparison=comparison,
        actual_model_forward_calls=12, eligible_for_full_window_check=comparison['eligible_for_full_window_check'],
        load_seconds=cold, initial_encode_seconds=encode_seconds, elapsed_seconds=time.monotonic() - began,
        peak_vram_bytes=max(row['peak_vram_bytes'] for row in rows), float_outputs_saved=False,
        video_generated=False, new_training=False, default_changed=False, steady_speedup_verified=False,
        automatic_retry=False, limitation='One first diffusion step and only2timedCFG calls per path;no trajectory/video or quality evidence')
    write_json(directory / 'progress.json', result)
    return result


def prepare(*, output, source_revision):
    directory = contained(output, DATA_ROOT)
    request = dict(copy.deepcopy(FIXED), job_id=directory.name, source_revision=source_revision,
                   experiment_revision='R005-unit-scale-inplace-LoRA-firststep-ABBA-v1')
    validate_request(request)
    require(directory != DATA_ROOT.resolve() and not directory.exists(), 'new output directory required')
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'request.json', request)
    return dict(status='cpu_request_prepared', request=str(directory / 'request.json'), max_seconds=180,
                model_loaded=False, gpu_launched=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), ensure_ascii=False))


if __name__ == '__main__':
    main()
