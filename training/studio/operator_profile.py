"""Bounded first-step R005 operator profile, not a sampler or video generator.

One warm CFG forward then one profiled CFG forward: exactly four model calls.
Only the standalone runner installs temporary record_function module hooks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import re
import time

from training.demo.contracts import contained, require, write_json
from training.studio.overlap_probe import DATA_ROOT, HEIGHT, R005_SHA, R005_SOURCE_JOB, SETTINGS, WIDTH
from training.studio.quality_probe import DEFAULT_NEGATIVE_CACHE, QualityRuntime, TextCFGFlow, load_negative

FIXED = dict(mode='operator_profile', max_seconds=180, source_job=str(R005_SOURCE_JOB),
    checkpoint_sha256=R005_SHA, negative_cache=DEFAULT_NEGATIVE_CACHE, residency='phase', fastpath=False,
    width=WIDTH, height=HEIGHT, seed=42, future_frames=48, warmup_cfg_forwards=1, profiled_cfg_forwards=1, **SETTINGS)


def validate_request(request):
    require(set(request) == set(FIXED) | {'job_id', 'source_revision', 'experiment_revision'}, 'unknown profile request fields')
    require(all(request[k] == v and type(request[k]) is type(v) for k, v in FIXED.items()),
            'profile fixes original R005/first3s geometry/CFG5 and180seconds')
    require(isinstance(request['job_id'], str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', request['job_id'])
            and isinstance(request['source_revision'], str) and bool(request['source_revision'])
            and request['experiment_revision'] == 'R005-single-first-step-operator-profile-v1', 'invalid profile identity')


def first_step_inputs(first, noise, sigmas):
    """Exact initial-state/timestep arithmetic from action_backend.euler_rollout."""
    import torch
    require(tuple(first.shape) == (1, 1, 48, 30, 52) and tuple(noise.shape) == (1, 13, 48, 30, 52)
            and first.dtype == noise.dtype == sigmas.dtype == torch.bfloat16
            and first.device == noise.device == sigmas.device and sigmas.ndim == 1 and len(sigmas) == 41
            and bool(torch.isfinite(first).all() and torch.isfinite(noise).all() and torch.isfinite(sigmas).all()),
            'profile requires exact finite BF16 first49-RGB-frame state')
    current = noise.clone()
    current[:, :1] = first
    timestep = (sigmas[0] * 1000).expand(current.shape[:2]).clone()
    timestep[:, 0] = 0
    return current, timestep


def operator_category(name):
    name = name.lower()
    if any(key in name for key in ('scaled_dot_product', 'flash_attn', 'flash_attention', 'efficient_attention')):
        return 'attention_kernel'
    if any(key in name for key in ('aten::mm', 'aten::bmm', 'aten::addmm', 'gemm', 'matmul')):
        return 'matrix_multiply'
    if any(key in name for key in ('aten::copy', 'aten::_to_copy', 'memcpy', 'aten::contiguous', 'aten::clone')):
        return 'copy_and_layout'
    if any(key in name for key in ('aten::add', 'aten::mul', 'aten::div', 'aten::sub', 'aten::pow', 'aten::sqrt',
            'aten::rsqrt', 'aten::gelu', 'aten::silu', 'aten::tanh', 'aten::sin', 'aten::cos', 'norm', 'aten::mean', 'aten::sum')):
        return 'elementwise_and_norm'
    return 'other'


def summarize_events(events):
    """Use exclusive device attribution, never sum nested total/kernel times."""
    import torch
    rows, scopes, overhead = [], [], []
    raw_cuda_events = 0
    for event in events:
        device_type = getattr(event, 'device_type', None)
        if device_type != torch.autograd.DeviceType.CPU:
            raw_cuda_events += int(device_type == torch.autograd.DeviceType.CUDA)
            continue  # Kernel rows duplicate GPU time already attributed to CPU operators.
        device_us = getattr(event, 'self_device_time_total', None)
        if device_us is None:
            device_us = getattr(event, 'self_cuda_time_total', 0.)
        device_us = float(device_us)
        require(math.isfinite(device_us) and device_us >= 0, 'invalid profiler self device time')
        row = dict(name=str(event.key), count=int(event.count), input_shapes=event.input_shapes,
            self_device_time_us=device_us, self_cpu_time_us=float(event.self_cpu_time_total),
            cpu_time_total_us=float(event.cpu_time_total))
        # NVIDIA identifies this exact event as driver submission/scheduling,
        # not a model kernel: nvidia/station-kernel-dev-ft/README.md in
        # https://github.com/NVIDIA/dgx-spark-playbooks . Keep it, but separate.
        if row['name'].strip().casefold() == 'command buffer full':
            row['exclusion_reason'] = 'driver submission queue/scheduling event;not a model operator'
            overhead.append(row)
            continue
        if getattr(event, 'is_user_annotation', False) or row['name'].startswith('InterActWorld/'):
            if row['name'].startswith('InterActWorld/'):
                inclusive_us = getattr(event, 'device_time_total', None)
                row['inclusive_device_time_us'] = float(inclusive_us if inclusive_us is not None
                                                        else getattr(event, 'cuda_time_total', 0.))
                scopes.append(row)
            continue
        row['category'] = operator_category(row['name'])
        rows.append(row)
    total = sum(row['self_device_time_us'] for row in rows)
    overhead_device_us = sum(row['self_device_time_us'] for row in overhead)
    categories = {}
    for row in rows:
        value = categories.setdefault(row['category'], dict(self_device_time_us=0., operator_calls=0))
        value['self_device_time_us'] += row['self_device_time_us']
        value['operator_calls'] += row['count']
    for value in categories.values():
        value['fraction_of_attributed_self_device_time'] = value['self_device_time_us'] / total if total else None
    sort_key = 'self_device_time_us' if total else 'self_cpu_time_us'
    return dict(summary_version=2, time_unit='microseconds', wall_time_unit='seconds', group_by_input_shape=True,
        cuda_time_available=total > 0, raw_cuda_event_groups_excluded=raw_cuda_events,
        attribution='CPU model operator rows, exclusive self device time only; excludes raw kernels, user scopes and known driver overhead',
        fraction_denominator='sum of attributed model operator exclusive self-device times after known runtime filtering; NOT wall time or GPU utilization',
        raw_attributed_self_device_time_us_before_runtime_filter=total + overhead_device_us,
        excluded_runtime_overhead_events=overhead,
        excluded_runtime_overhead_self_device_time_us=overhead_device_us,
        excluded_runtime_overhead_self_cpu_time_us=sum(row['self_cpu_time_us'] for row in overhead),
        denominator_change='v1 included Command Buffer Full;v2 subtracts only this known driver event from model shares/ranking and preserves it separately;raw captures unchanged',
        attributed_self_device_time_us=total, categories=categories,
        top40_sorted_by=sort_key, top40=sorted(rows, key=lambda row: row[sort_key], reverse=True)[:40],
        module_scopes_inclusive_not_in_denominator=scopes,
        degradation=None if total else 'No attributed CUDA timing; CPU-only view, do not infer GPU bottleneck shares')


@contextmanager
def module_scopes(model):
    """Optional profiling-only scopes; always remove hooks, even on forward error."""
    import torch
    handles, active = [], []
    names = []
    try:
        blocks = getattr(getattr(model, 'model', None), 'blocks', ())
        for index, block in enumerate(blocks):
            for attribute in ('self_attn', 'cross_attn', 'ffn'):
                module = getattr(block, attribute, None)
                if module is None:
                    continue
                label = f'InterActWorld/block_{index:02d}/{attribute}'
                names.append(label)

                def before(_module, _args, label=label):
                    scope = torch.profiler.record_function(label)
                    scope.__enter__()
                    active.append(scope)

                def after(_module, _args, _output):
                    if active:
                        active.pop().__exit__(None, None, None)

                handles.append(module.register_forward_pre_hook(before))
                handles.append(module.register_forward_hook(after, always_call=True))
        yield names
    finally:
        for handle in handles:
            handle.remove()
        while active:
            active.pop().__exit__(None, None, None)


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
    target = directory / 'profile_output'
    require(directory.name == request['job_id'] and directory.is_dir() and not target.exists()
            and not (directory / 'inputs.json').exists(), 'new profile job required; preserve previous output')
    began, runtime, flow = time.monotonic(), None, None

    def check():
        require(time.monotonic() - began < 155, 'profile deadline reached; no repeated forward')
        validate_active_authority(lease)

    check()
    require(hasattr(torch, 'profiler') and callable(getattr(torch.profiler, 'profile', None))
            and torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one leased CUDA device and torch.profiler required')
    available = torch.profiler.supported_activities()
    activities = [torch.profiler.ProfilerActivity.CPU]
    require(activities[0] in available, 'CPU operator profiler unavailable')
    if torch.profiler.ProfilerActivity.CUDA in available:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    state, config, prompt, lineage, initial, actions, scene = read_source(request)
    require(lineage['sha256'] == R005_SHA and lineage['step'] == 1040 and scene['seed'] == 42
            and initial.dtype == np.uint8 and initial.shape == (HEIGHT, WIDTH, 3), 'actual R005/source identity differs')
    expected = np.zeros((48, 8), np.float32)
    expected[:, [0, 5]] = 1
    require(actions.dtype == np.float32 and np.array_equal(actions[:48], expected), 'original first48 WJ controls differ')
    require(Path(config.base_model_path).resolve().is_relative_to(DATA_ROOT.resolve()), 'model weights must remain on data disk')
    negative, negative_source = load_negative(DEFAULT_NEGATIVE_CACHE, config.base_model_path)
    target.mkdir(exist_ok=False)
    inputs = dict(mode='operator_profile', source_job=request['source_job'], checkpoint=lineage['path'],
        checkpoint_sha256=R005_SHA, checkpoint_step=1040, scene_id=scene['scene_id'], prompt=scene['prompt'],
        negative_prompt=negative_source, seed=42, settings=SETTINGS, precision='bf16', residency='phase',
        actual_actions=[dict(frames=48, keys=['W', 'J'])], width=WIDTH, height=HEIGHT,
        source_revision=request['source_revision'], experiment_revision=request['experiment_revision'],
        activities_requested=[str(activity) for activity in activities], torch_version=torch.__version__,
        record_shapes=True, with_stack=False, profile_memory=False, trace_exported=False,
        sample_steps_executed=0, scheduled_steps_reference_only=40, model_forward_budget=4,
        new_training=False, video_generated=False, text_kv_cache_enabled=False, default_changed=False)
    write_json(directory / 'inputs.json', inputs)
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
            model_input_shape=list(current.shape), first_sigma=float(sigmas[0]), actual_timestep=timestep.detach().float().cpu().tolist(),
            timestep_contract='exact first BF16 sigma*1000 from existing Euler;firstslot0;no manual clamp or999 substitution')
        write_json(directory / 'inputs.json', inputs)
        flow = TextCFGFlow(runtime.model, 5, negative.to('cuda', dtype=torch.bfloat16))
        kwargs = dict(conditional_dict=conditions, timestep=timestep, replace_first_timestep_and_noise_latents=True)
        check()
        runtime.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            velocity, _ = flow(current, **kwargs)
        runtime.synchronize()
        warm_seconds = time.monotonic() - tick
        warm_peak = int(torch.cuda.max_memory_allocated())
        warm_velocity = velocity.detach().cpu()
        del velocity
        require(flow.forward_calls == 2, 'warmup must call positive and negative once each')
        write_json(directory / 'progress.json', dict(status='running', stage='warmup_completed', actual_model_forward_calls=2,
            warmup_seconds=warm_seconds, warmup_peak_vram_bytes=warm_peak))
        check()
        runtime.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        with module_scopes(runtime.model) as scopes:
            with torch.profiler.profile(activities=activities, record_shapes=True, with_stack=False, profile_memory=False) as profiler:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16), torch.profiler.record_function('InterActWorld/CFG_forward'):
                    velocity, _ = flow(current, **kwargs)
                runtime.synchronize()
        profiled_seconds = time.monotonic() - tick
        peak = int(torch.cuda.max_memory_allocated())
        measured = velocity.detach().cpu()
        del velocity
        require(flow.forward_calls == 4 and measured.shape == warm_velocity.shape == current.shape,
                'profile must perform exactly two CFG/four model forwards with original velocity shape')
        finite = bool(torch.isfinite(measured).all() and torch.isfinite(warm_velocity).all())
        comparison = dict(finite=finite, torch_equal=finite and torch.equal(measured, warm_velocity),
            max_absolute_difference=float((measured.float() - warm_velocity.float()).abs().max()) if finite else None,
            scope='same first-step velocity only;not a generated-video or quality comparison')
        summary = summarize_events(profiler.key_averages(group_by_input_shape=True))
        summary.update(warmup_seconds=warm_seconds, profiled_wall_seconds=profiled_seconds,
            profiled_wall_scope='profiler enter/forward/synchronize/exit plus temporary hooks;contains observer overhead',
            actual_model_forward_calls=4, profiled_model_forward_calls=2, warmup_model_forward_calls=2,
            velocity_comparison=comparison, warmup_peak_vram_bytes=warm_peak, profiled_peak_vram_bytes=peak,
            module_scope_count=len(scopes), load_seconds=cold, initial_encode_seconds=encode_seconds,
            limitation='Single first diffusion step only;not full40step trajectory, video generation latency, quality or steady speedup.',
            instrumentation=dict(standalone_runner=True, temporary_forward_hooks=['self_attn', 'cross_attn', 'ffn'],
                hooks_removed=True, record_shapes=True, with_stack=False, profile_memory=False, large_trace_exported=False))
        write_json(target / 'operator-summary.json', summary)
        (target / 'table.txt').write_text('Operator times are microseconds; CPU operator rows only; exclusive GPU attribution.\n'
            + '\n'.join(f"{row['self_device_time_us']:12.3f} device_us  {row['self_cpu_time_us']:12.3f} self_cpu_us  "
                        f"{row['count']:6d} calls  {row['name']}  {row['input_shapes']}" for row in summary['top40']), encoding='utf-8')
    finally:
        flow = None  # Release its model reference before closing the runtime.
        if runtime is not None:
            runtime.close()
    result = dict(status='completed', mode='operator_profile', profile_output=str(target),
        operator_summary=str(target / 'operator-summary.json'), actual_model_forward_calls=4,
        warmup_seconds=warm_seconds, profiled_wall_seconds=profiled_seconds, peak_vram_bytes=max(warm_peak, peak),
        cuda_time_available=summary['cuda_time_available'], velocity_comparison=comparison,
        elapsed_seconds=time.monotonic() - began, new_training=False, video_generated=False,
        cache_enabled=False, default_changed=False, speedup_claimed=False, automatic_retry=False,
        limitation=summary['limitation'])
    write_json(directory / 'progress.json', result)
    return result


def prepare(*, output, source_revision):
    directory = contained(output, DATA_ROOT)
    request = dict(FIXED, job_id=directory.name, source_revision=source_revision,
                   experiment_revision='R005-single-first-step-operator-profile-v1')
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
