"""One real first-block FFN input, then a bounded isolated Inductor qualification.

No model replacement, sampler, video, training, eager fallback or automatic retry.
42 isolated FFN invocations plus one full-model capture containing that FFN once.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import copy
import json
import math
import os
from pathlib import Path
import re
import signal
from statistics import median
import sys
import time

from training.demo.contracts import contained, require, write_json
from training.studio.batched_cfg_probe import velocity_difference
from training.studio.operator_profile import first_step_inputs
from training.studio.overlap_probe import DATA_ROOT, HEIGHT, R005_SHA, R005_SOURCE_JOB, SETTINGS, WIDTH
from training.studio.quality_probe import QualityRuntime

ORDER = (('warmup', 'A', 1), ('warmup', 'B', 1), ('timed', 'A', 10),
         ('timed', 'B', 10), ('timed', 'B', 10), ('timed', 'A', 10))
OPTIONS = {'triton.cudagraphs': False, 'max_autotune': False}
REVISION = 'R005-first-FFN-Inductor-region-ABBA-v1'
FIXED = dict(mode='ffn_compile_probe', max_seconds=600, source_job=str(R005_SOURCE_JOB),
    checkpoint_sha256=R005_SHA, historical_baseline_revision='9344a3f', residency='phase', fastpath=False,
    width=WIDTH, height=HEIGHT, seed=42, future_frames=48, first_compile_max_seconds=300,
    target='model.blocks.0.ffn', backend='inductor', fullgraph=True, dynamic=False,
    compile_options=OPTIONS, compile_threads=1, minimum_region_median_reduction=.10,
    warmup_order=['A', 'B'], timed_order=['A', 'B', 'B', 'A'], timed_calls_per_group=10, **SETTINGS)


def validate_request(request):
    require(set(request) == set(FIXED) | {'job_id', 'source_revision', 'experiment_revision'}, 'unknown FFN probe fields')
    require(all(request[key] == value and type(request[key]) is type(value) for key, value in FIXED.items()),
            'FFN probe fixes source/geometry/compile options/ABBA/10percent/600seconds')
    require(all(type(value) is bool for value in request['compile_options'].values()), 'compile options must be booleans')
    require(isinstance(request['job_id'], str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', request['job_id'])
            and isinstance(request['source_revision'], str) and bool(request['source_revision'])
            and request['experiment_revision'] == REVISION, 'invalid probe identity')


@contextmanager
def capture_ffn_input(module):
    """A removable observation hook; preserve the tensor's actual dtype/stride."""
    capture = {'calls': 0, 'hook_removed': False}

    def save_input(_module, args):
        require(len(args) == 1 and capture['calls'] == 0, 'capture expects exactly one FFN input invocation')
        source = args[0]
        capture.update(calls=1, shape=list(source.shape), dtype=str(source.dtype), stride=list(source.stride()),
                       device=str(source.device), value=source.detach().clone())
        require(capture['value'].stride() == source.stride(), 'captured clone changed input stride')

    handle = module.register_forward_pre_hook(save_input)
    try:
        yield capture
    finally:
        handle.remove()
        capture['hook_removed'] = True


@contextmanager
def compile_deadline(seconds):
    """POSIX alarm for the first lazy compile call; outer watchdog remains600s."""
    require(sys.platform.startswith('linux') and hasattr(signal, 'setitimer'), 'compile qualification requires Linux timer')
    require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'do not replace an existing process timer')
    previous = signal.getsignal(signal.SIGALRM)

    def expired(_signum, _frame):
        raise TimeoutError('first compiled FFN call exceeded300seconds;no fallback or retry')

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def max_rss():
    import resource
    # Remote runner is Linux; ru_maxrss is a process high-water mark, not current RSS.
    return dict(max_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), unit='KiB',
                scope='Linux process high-water RSS;not current usage or compiler-child RSS')


def validate_compile_environment(directory):
    require(os.environ.get('TORCHINDUCTOR_COMPILE_THREADS') == '1', 'Inductor compile_threads must be1 before worker import')
    result = {}
    for key, child in [('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('TRITON_CACHE_DIR', 'triton')]:
        expected = contained(directory / 'compile-cache' / child, directory)
        require(os.environ.get(key) and Path(os.environ[key]).resolve() == expected,
                'compile caches must be inside this new job directory')
        result[key] = str(expected)
    return dict(result, TORCHINDUCTOR_COMPILE_THREADS='1')


def gate_summary(rows):
    require([(row['phase'], row['path'], row['isolated_region_calls']) for row in rows] == list(ORDER),
            'exact warmA1/B1 then timedABBA x10 required')
    require(all(math.isfinite(row['seconds']) and row['seconds'] > 0 for row in rows), 'invalid FFN timing')
    exact = all(row['vs_warmup_A']['finite'] and row['vs_warmup_A']['torch_equal'] for row in rows)
    timings = {name: [row['seconds'] / row['isolated_region_calls'] for row in rows
                     if row['phase'] == 'timed' and row['path'] == name] for name in ('A', 'B')}
    baseline, compiled = median(timings['A']), median(timings['B'])
    passed = compiled <= .9 * baseline
    return dict(eligible_for_full_model_qualification=exact and passed,
        all_six_sampled_outputs_bitwise_equal=exact, region_timing_gate_passed=passed,
        timed_seconds_per_call=timings, baseline_region_median_seconds=baseline,
        compiled_region_median_seconds=compiled, observed_region_median_reduction=1 - compiled / baseline,
        required_region_median_reduction=.10, isolated_region_calls=sum(row['isolated_region_calls'] for row in rows),
        model_capture_calls=1, first_ffn_capture_calls=1, first_ffn_source_level_invocations=43,
        rule='two warm outputs and four group-final outputs exact AND median(timed_B/10)<=0.9*median(timed_A/10)',
        unsampled_group_intermediate_outputs='not copied to CPU;no claim of checking all42outputs',
        count_scope='explicit Python callable invocations;compiler tracing/internal execution is not separately counted',
        full_model_speedup_verified=False, video_speedup_verified=False, default_enabled=False, automatic_retry=False)


def region_group(runtime, function, value, calls, *, first_compile=False):
    torch = runtime.torch
    runtime.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rss_before = max_rss()
    allocated_before, reserved_before = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    tick = time.monotonic()
    with compile_deadline(300) if first_compile else nullcontext():
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            output = None
            for _ in range(calls):
                output = function(value)
            runtime.synchronize()
    seconds = time.monotonic() - tick
    row = dict(seconds=seconds, seconds_per_call=seconds / calls, isolated_region_calls=calls,
        first_compile_included=first_compile, allocated_before_bytes=int(allocated_before),
        reserved_before_bytes=int(reserved_before), allocated_after_bytes=int(torch.cuda.memory_allocated()),
        reserved_after_bytes=int(torch.cuda.memory_reserved()), peak_vram_bytes=int(torch.cuda.max_memory_allocated()),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()), max_rss_before=rss_before, max_rss_after=max_rss(),
        timing_scope='CUDA-synchronized group total/calls;excludes capture clone and final CPU output copy')
    return output.detach().cpu(), row


def run(request, directory, lease):
    import numpy as np
    import torch
    from torch._dynamo import config as dynamo_config
    from torch._inductor import config as inductor_config
    from training.demo import action_backend as action
    from training.demo.backend import validate_active_authority
    from training.longforcing_lite import euler_sigmas
    from training.models.action_adapter import build_action_context
    from training.studio.worker import read_source
    validate_request(request)
    directory = contained(directory, DATA_ROOT)
    require(directory.name == request['job_id'] and directory.is_dir()
            and not any((directory / name).exists() for name in ('inputs.json', 'progress.json', 'comparison.json')),
            'new FFN probe directory required')
    caches = validate_compile_environment(directory)
    require(callable(getattr(torch, 'compile', None)) and inductor_config.compile_threads == 1,
            'existing torch.compile and single compiler thread required')
    began, runtime, compiled, ffn, value = time.monotonic(), None, None, None, None
    capture, conditions, current, timestep, first, sigmas, kwargs, outputs = None, None, None, None, None, None, None, None
    rows, rss_before = [], max_rss()

    def check():
        require(time.monotonic() - began < 555, 'FFN qualification deadline;no retry')
        validate_active_authority(lease)

    check()
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one leased CUDA GPU required')
    state, config, prompt, lineage, initial, actions, scene = read_source(request)
    require(lineage['sha256'] == R005_SHA and lineage['step'] == 1040 and scene['seed'] == 42
            and initial.shape == (HEIGHT, WIDTH, 3) and initial.dtype == np.uint8, 'originalR005 input differs')
    require(Path(config.base_model_path).resolve().is_relative_to(DATA_ROOT.resolve()), 'weights must remain on data disk')
    expected = np.zeros((48, 8), np.float32)
    expected[:, [0, 5]] = 1
    require(actions.dtype == np.float32 and np.array_equal(actions[:48], expected), 'first48 source actions must remainWJ')
    inputs = dict(mode='ffn_compile_probe', source_job=request['source_job'], checkpoint=lineage['path'],
        checkpoint_sha256=R005_SHA, checkpoint_step=1040, prompt=scene['prompt'], scene_id=scene['scene_id'],
        seed=42, width=WIDTH, height=HEIGHT, settings=SETTINGS, precision_scope='original CUDA BF16 autocast',
        source_revision=request['source_revision'], historical_baseline_revision='9344a3f',
        experiment_revision=REVISION, compile_backend='inductor', fullgraph=True, dynamic=False,
        compile_options=OPTIONS, compiler_environment=caches, suppress_errors=False,
        target='runtime.model.model.blocks[0].ffn', actual_actions=[dict(frames=48, keys=['W', 'J'])],
        fixed_gate='two warm and four last-group outputs exact;timed region median reduction>=10percent',
        region_order=[dict(phase=p, path=n, calls=c) for p, n, c in ORDER], input_encode_count=1,
        planned_model_capture_calls=1, planned_isolated_region_calls=42, first_compile_max_seconds=300,
        max_seconds=600, full_model_capture='positive text only;no CFG pair or Euler update',
        capture_tensor_saved=False, default_changed=False, video_generated=False, training=False)
    write_json(directory / 'inputs.json', inputs)
    try:
        check()
        tick = time.monotonic()
        runtime = QualityRuntime(state, config, 'phase', None)
        runtime.configure(SETTINGS)
        runtime.synchronize()
        load_seconds = time.monotonic() - tick
        tick = time.monotonic()
        runtime.transfer('vae')
        runtime.begin()
        pixels = torch.from_numpy(initial.copy()).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).float().div(127.5).sub(1)
        first = runtime.encode(pixels).to(torch.bfloat16)
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
        ffn = runtime.model.model.blocks[0].ffn
        require(not ffn.training and all(not parameter.requires_grad for parameter in ffn.parameters()), 'FFN must remain frozen/eval')
        inputs.update(first_latent_sha256_float32=action.tensor_sha(first), noise_sha256_float32=action.tensor_sha(noise),
            actual_timestep=timestep.detach().float().cpu().tolist())
        write_json(directory / 'inputs.json', inputs)
        write_json(directory / 'progress.json', dict(status='running', phase='one_positive_model_capture'))
        tick = time.monotonic()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16), capture_ffn_input(ffn) as capture:
            outputs = runtime.model(current, conditional_dict=conditions, timestep=timestep,
                                    replace_first_timestep_and_noise_latents=True)
            runtime.synchronize()
        capture_seconds = time.monotonic() - tick
        capture_peak_vram_bytes = int(torch.cuda.max_memory_allocated())
        capture_peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
        require(capture['calls'] == 1 and capture['hook_removed'] and capture['shape'] == [1, 5070, 3072],
                'real first FFN capture must be one [1,5070,3072] tensor with hook removed')
        value = capture.pop('value')
        require(value.device.type == 'cuda' and bool(torch.isfinite(value).all()), 'captured actual FFN input must be finite CUDA tensor')
        inputs.update(ffn_input=capture, model_capture_calls=1, clone_excluded_from_region_timing=True)
        write_json(directory / 'inputs.json', inputs)
        outputs = conditions = current = timestep = first = sigmas = None
        check()
        # No global reset, eager fallback or insertion of the wrapper into the model.
        with dynamo_config.patch(suppress_errors=False):
            tick = time.monotonic()
            compiled = torch.compile(ffn, backend='inductor', fullgraph=True, dynamic=False, options=dict(OPTIONS))
            wrapper_seconds = time.monotonic() - tick
            reference = None
            for index, (phase, name, calls) in enumerate(ORDER):
                check()
                write_json(directory / 'progress.json', dict(status='running', runs=rows,
                    current_group=dict(index=index, phase=phase, path=name, calls=calls)))
                output, row = region_group(runtime, ffn if name == 'A' else compiled, value, calls,
                                          first_compile=phase == 'warmup' and name == 'B')
                require(output.shape == value.shape, 'FFN output shape changed')
                if reference is None:
                    reference = output.clone()
                row.update(index=index, phase=phase, path=name, vs_warmup_A=velocity_difference(reference, output))
                rows.append(row)
                del output
                write_json(directory / 'progress.json', dict(status='running', runs=rows,
                    isolated_region_calls=sum(row['isolated_region_calls'] for row in rows), model_capture_calls=1))
        comparison = gate_summary(rows)
        write_json(directory / 'comparison.json', comparison)
        require(runtime.model.model.blocks[0].ffn is ffn and capture['hook_removed'], 'original FFN/hook state changed')
    finally:
        compiled = ffn = value = capture = conditions = current = timestep = first = sigmas = kwargs = outputs = None
        if runtime is not None:
            runtime.close()
    result = dict(status='completed', mode='ffn_compile_probe', runs=rows, comparison=comparison,
        model_capture_calls=1, isolated_region_calls=42, first_ffn_source_level_invocations=43,
        eligible_for_full_model_qualification=comparison['eligible_for_full_model_qualification'],
        load_seconds=load_seconds, encode_seconds=encode_seconds, capture_seconds=capture_seconds,
        compile_wrapper_creation_seconds=wrapper_seconds, first_compiled_call_seconds=rows[1]['seconds'],
        compile_cold_scope='first compiled call includes lazy compilation and one FFN execution;not compile-only time',
        elapsed_seconds=time.monotonic() - began, max_rss_before=rss_before, max_rss_after=max_rss(),
        capture_peak_vram_bytes=capture_peak_vram_bytes, capture_peak_reserved_bytes=capture_peak_reserved_bytes,
        peak_vram_bytes=max(capture_peak_vram_bytes, *(row['peak_vram_bytes'] for row in rows)),
        peak_scope='maximum of original input/model capture and each isolated region group',
        model_replaced=False, temporary_hook_removed=True,
        compiler_cache_preserved=True, float_outputs_saved=False, video_generated=False, default_changed=False,
        automatic_retry=False, full_model_speedup_verified=False,
        limitation='One real first-block input, two timed groups per path;only group-final outputs checked;not full-model/video speedup')
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
                gpu_launched=False, model_loaded=False, compilation_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), ensure_ascii=False))


if __name__ == '__main__':
    main()
