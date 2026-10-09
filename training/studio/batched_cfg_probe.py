"""Fixed first-step serial/batched CFG qualification; no trajectory or video.

Warm A/B then timed A/B/B/A: six CFGs, twelve logical branches, nine model calls.
The existing Creator launcher owns GPU authorization and the180-second watchdog.
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
REVISION = 'R005-batched-text-CFG-firststep-ABBA-v1'
FIXED = dict(mode='batched_cfg_probe', max_seconds=180, source_job=str(R005_SOURCE_JOB),
    checkpoint_sha256=R005_SHA, negative_cache=DEFAULT_NEGATIVE_CACHE, residency='phase', fastpath=False,
    width=WIDTH, height=HEIGHT, seed=42, future_frames=48, warmup_order=['A', 'B'],
    timed_order=['A', 'B', 'B', 'A'], minimum_median_reduction=MIN_REDUCTION, **SETTINGS)


def validate_request(request):
    require(set(request) == set(FIXED) | {'job_id', 'source_revision', 'experiment_revision'},
            'unknown batched CFG request fields')
    require(all(request[k] == v and type(request[k]) is type(v) for k, v in FIXED.items()),
            'batched CFG fixes originalR005/CFG5/BF16/ABBA/3percent/180seconds')
    require(isinstance(request['job_id'], str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', request['job_id'])
            and isinstance(request['source_revision'], str) and bool(request['source_revision'])
            and request['experiment_revision'] == REVISION, 'invalid probe identity')


def velocity_difference(reference, candidate):
    """CPU-only scalar diagnostics; exact equality, not tolerance, is the gate."""
    metrics = difference(reference, candidate)
    rmse = rms = relative = None
    if metrics['finite']:
        ref, value = reference.float(), candidate.float()
        rmse = float((value - ref).square().mean().sqrt())
        rms = float(ref.square().mean().sqrt())
        relative = rmse / rms if rms else (0. if rmse == 0 else None)
    return dict(metrics, rmse=rmse, reference_rms=rms, relative_rmse=relative,
                relative_rmse_definition='RMSE(candidate-reference)/RMS(reference);null if nonfinite or zero reference with nonzero error')


def gate_summary(rows):
    require([(row['phase'], row['path']) for row in rows] == list(ORDER), 'fixed warmAB/timedABBA order required')
    require(all(row['actual_model_forward_calls'] == (2 if row['path'] == 'A' else 1)
                and row['logical_branch_evaluations'] == 2
                and row['model_input_shape'] == [1 if row['path'] == 'A' else 2, 13, 48, 30, 52]
                for row in rows), 'exact sixCFG/nine-model-call/twelve-branch geometry required')
    require(all(math.isfinite(row['seconds']) and row['seconds'] > 0 for row in rows), 'invalid synchronized timing')
    all_equal = all(row['vs_warmup_A']['finite'] and row['vs_warmup_A']['torch_equal'] for row in rows)
    measured = {name: [row['seconds'] for row in rows if row['phase'] == 'timed' and row['path'] == name]
                for name in ('A', 'B')}
    baseline, batched = median(measured['A']), median(measured['B'])
    timing_passed = batched <= baseline * (1 - MIN_REDUCTION)
    return dict(eligible_for_full_window_check=all_equal and timing_passed,
        all_six_velocity_fields_bitwise_equal=all_equal, median_timing_gate_passed=timing_passed,
        timed_seconds=measured, baseline_median_seconds=baseline, batched_median_seconds=batched,
        observed_median_reduction=1 - batched / baseline, required_median_reduction=MIN_REDUCTION,
        rule='every warmup/timed velocity bitwise equals warmup_A AND median(timed_B)<=0.97*median(timed_A)',
        actual_model_forward_calls=sum(row['actual_model_forward_calls'] for row in rows),
        logical_branch_evaluations=sum(row['logical_branch_evaluations'] for row in rows),
        qualification_scope='one first diffusion step only;eligibility is not full-window or stable speedup evidence',
        quality_improvement_verified=False, steady_speedup_verified=False, default_enabled=False, automatic_retry=False)


def check_batched_stats(stats):
    require(stats['forward_calls'] == stats['completed_calls'] == 1
            and stats['logical_branch_evaluations'] == 2
            and stats['last_model_input_shape'] == [2, 13, 48, 30, 52]
            and stats['last_timestep_shape'] == [2, 13]
            and stats['last_action_context_count'] == 2
            and stats['last_action_context_shared_reference'] is True
            and stats['branch_order'] == ['positive', 'negative'],
            'batched model call must retain two same-action/time branches and positive/negative order')
    return copy.deepcopy(stats)


def one_forward(runtime, current, kwargs, negative, *, batched):
    """Include flow construction and batch concatenation in synchronized wall time."""
    from training.studio.batched_cfg import BatchedTextCFGFlow
    torch = runtime.torch
    runtime.synchronize()
    torch.cuda.reset_peak_memory_stats()
    tick = time.monotonic()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        flow = BatchedTextCFGFlow(runtime.model, 5, negative) if batched else TextCFGFlow(runtime.model, 5, negative)
        velocity, _ = flow(current, **kwargs)
        runtime.synchronize()
    seconds = time.monotonic() - tick
    if batched:
        stats = check_batched_stats(flow.stats)
        calls, logical = flow.forward_calls, flow.logical_branch_evaluations
        shape = list(stats['last_model_input_shape'])
    else:
        stats = None
        calls, logical, shape = flow.forward_calls, flow.forward_calls, list(current.shape)
    require(calls == (1 if batched else 2) and logical == 2, 'unexpected real CFG forward counters')
    row = dict(seconds=seconds, actual_model_forward_calls=calls, logical_branch_evaluations=logical,
        model_input_shape=shape, batched_flow_stats=stats, peak_vram_bytes=int(torch.cuda.max_memory_allocated()),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        timing_scope='CUDA-synchronized CFG including flow construction and B batch concatenation;CPU output copy excluded')
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
    began, runtime, rows = time.monotonic(), None, []
    current = timestep = conditions = negative_cuda = first = sigmas = None

    def check():
        require(time.monotonic() - began < 155, 'batched CFG qualification deadline;no retry')
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
    inputs = dict(mode='batched_cfg_probe', source_job=request['source_job'], checkpoint=lineage['path'],
        checkpoint_sha256=R005_SHA, checkpoint_step=1040, scene_id=scene['scene_id'], prompt=scene['prompt'],
        negative_prompt=negative_source, seed=42, width=WIDTH, height=HEIGHT, settings=SETTINGS,
        precision='bf16', residency='phase', actual_actions=[dict(frames=48, keys=['W', 'J'])],
        source_revision=request['source_revision'], experiment_revision=request['experiment_revision'],
        warmup_order=request['warmup_order'], timed_order=request['timed_order'], model_forward_budget=9,
        logical_branch_budget=12, minimum_median_reduction=request['minimum_median_reduction'],
        qualification_rule='all6samefirststepvelocitiesbitwiseequal AND median(timedB)<=0.97*median(timedA)',
        input_encode_count=1, model_resident_during_all_forwards=True, trajectory_steps_executed=0,
        modification='text CFG branches batched along B only;both image/action/time conditions duplicated',
        profiler_enabled=False, text_kv_cache_enabled=False, inplace_lora_enabled=False,
        video_generated=False, new_training=False, default_changed=False)
    write_json(directory / 'inputs.json', inputs)  # Persist the gate before any outcome.
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
        negative_cuda = negative.to('cuda', dtype=torch.bfloat16)
        inputs.update(first_latent_sha256_float32=action.tensor_sha(first), noise_sha256_float32=action.tensor_sha(noise),
            actual_timestep=timestep.detach().float().cpu().tolist(), source_model_input_shape=list(current.shape),
            identical_source_input_objects_for_all_runs=True)
        write_json(directory / 'inputs.json', inputs)
        kwargs = dict(conditional_dict=conditions, timestep=timestep, replace_first_timestep_and_noise_latents=True)
        reference = None
        for index, (phase, name) in enumerate(ORDER):
            check()
            write_json(directory / 'progress.json', dict(status='running', current_run=dict(index=index, phase=phase, path=name), runs=rows))
            velocity, row = one_forward(runtime, current, kwargs, negative_cuda, batched=name == 'B')
            require(velocity.shape == current.shape, 'guided velocity shape differs from fixedfirststep')
            if reference is None:
                reference = velocity.clone()
            row.update(index=index, phase=phase, path=name, vs_warmup_A=velocity_difference(reference, velocity))
            rows.append(row)
            del velocity
            write_json(directory / 'progress.json', dict(status='running', runs=rows,
                actual_model_forward_calls=sum(row['actual_model_forward_calls'] for row in rows)))
        comparison = gate_summary(rows)
        write_json(directory / 'comparison.json', comparison)
    finally:
        current = timestep = conditions = negative_cuda = first = sigmas = None
        if 'kwargs' in locals():
            kwargs.clear()
        if runtime is not None:
            runtime.close()
    result = dict(status='completed', mode='batched_cfg_probe', runs=rows, comparison=comparison,
        actual_model_forward_calls=comparison['actual_model_forward_calls'],
        logical_branch_evaluations=comparison['logical_branch_evaluations'],
        eligible_for_full_window_check=comparison['eligible_for_full_window_check'],
        load_seconds=cold, initial_encode_seconds=encode_seconds, elapsed_seconds=time.monotonic() - began,
        peak_vram_bytes=max(row['peak_vram_bytes'] for row in rows), float_outputs_saved=False,
        video_generated=False, new_training=False, default_changed=False, steady_speedup_verified=False,
        automatic_retry=False, limitation='Single first diffusion step;two timed CFGs per path;no trajectory/video or quality evidence')
    write_json(directory / 'progress.json', result)
    return result


def prepare(*, output, source_revision):
    directory = contained(output, DATA_ROOT)
    request = dict(copy.deepcopy(FIXED), job_id=directory.name, source_revision=source_revision, experiment_revision=REVISION)
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
