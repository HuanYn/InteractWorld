"""Bounded synthetic AdamW8bit update diagnostic; not training or model evaluation.

Only two copied R005 LoRA tensors are updated. No base model, real gradients,
backpropagation, dataset, checkpoint writing or optimizer-state resume is used.
"""
from __future__ import annotations

import copy
import gc
from pathlib import Path
import re
import time

from training.demo.contracts import contained, require, write_json

DATA_ROOT = Path('/DATA_DISK/InteractWorld')
REVISION = 'R005-two-LoRA-synthetic-AdamW8bit-precision-v1'
SUFFIXES = ('blocks.0.self_attn.q.lora_a.weight', 'blocks.0.self_attn.q.lora_b.weight')
SHAPES = ((16, 3072), (3072, 16))
FIXED = dict(mode='optimizer_precision_probe', max_seconds=120, seed=42, updates=20,
    optimizer='bitsandbytes.optim.AdamW8bit', lr=2e-6, betas=[.9, .95], weight_decay=.01,
    eps=1e-8, gradient_kind='fixed_flat_index_alternating_sign_bf16', gradient_magnitude=.001,
    parent_step=1040, parameter_count=2)
LIMITATIONS = dict(synthetic_gradient=True, real_gradient=False, backward_called=False,
    base_model_loaded=False, dataset_loaded=False, optimizer_state_resumed=False,
    quality_evidence=False, trained_checkpoint_saved=False, default_training_changed=False,
    scope='Mechanism diagnostic only;constant artificial gradients do not establish why real training succeeds or fails.')


def validate_request(request):
    require(type(request) is dict and set(request) == set(FIXED) |
            {'job_id', 'source_revision', 'experiment_revision', 'parent_checkpoint'}, 'unexpected optimizer probe fields')
    require(all(request[key] == value and type(request[key]) is type(value) for key, value in FIXED.items()),
            'optimizer diagnostic settings are fixed;no parameter sweep')
    require(all(type(value) is float for value in request['betas']), 'fixed float beta values required')
    require(type(request['job_id']) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', request['job_id'])
            and type(request['source_revision']) is str and bool(request['source_revision'])
            and request['experiment_revision'] == REVISION, 'invalid optimizer probe identity')
    require(type(request['parent_checkpoint']) is str and Path(request['parent_checkpoint']).is_absolute(),
            'explicit operator-provided parent checkpoint path required')
    contained(request['parent_checkpoint'], DATA_ROOT)


def select_parent_tensors(payload):
    import torch
    require(type(payload) is dict and payload.get('stage') == 'action_teacher_lora_v1'
            and payload.get('step') == 1040, 'only original R005 step1040 parent is allowed')
    state = payload.get('trainable_model')
    require(isinstance(state, dict), 'parent trainable_model is missing')
    selected = {}
    for suffix, shape in zip(SUFFIXES, SHAPES):
        matches = [name for name in state if name == suffix or name.endswith('.' + suffix)]
        require(len(matches) == 1, f'exactly one first-block query LoRA tensor required:{suffix}')
        name, value = matches[0], state[matches[0]]
        require(torch.is_tensor(value) and tuple(value.shape) == shape and value.dtype == torch.bfloat16
                and value.device.type == 'cpu' and bool(torch.isfinite(value).all()),
                'expected finite CPU BF16 R005 rank16/query3072 tensor')
        selected[name] = value.detach().clone()
    return selected


def synthetic_gradient(initial):
    """Same CPU BF16 values in both cases;FP32 gets only an exact widening cast."""
    import torch
    require(initial.device.type == 'cpu' and initial.dtype == torch.bfloat16, 'BF16 CPU initial required')
    signs = 1 - 2 * (torch.arange(initial.numel(), dtype=torch.int64) % 2)
    return (signs.float() * .001).to(torch.bfloat16).reshape(initial.shape)


def clone_case_parameters(initials, dtype, device):
    import torch
    require(dtype in (torch.bfloat16, torch.float32), 'only BF16 and FP32 parameter cases are allowed')
    return {name: torch.nn.Parameter(value.detach().to(device=device, dtype=dtype).clone())
            for name, value in initials.items()}


def change_summary(current, previous, initial):
    import torch
    require(current.shape == previous.shape == initial.shape and current.device.type == previous.device.type == initial.device.type == 'cpu'
            and bool(torch.isfinite(current).all()), 'finite CPU comparison tensors required')
    value, before, start = current.float(), previous.float(), initial.float()
    return dict(changed_from_previous=int(torch.count_nonzero(value != before)),
        changed_from_initial=int(torch.count_nonzero(value != start)),
        l2_change_from_previous=float(torch.linalg.vector_norm(value - before)),
        l2_change_from_initial=float(torch.linalg.vector_norm(value - start)),
        max_absolute_change_from_initial=float((value - start).abs().max()),
        recast_bf16_changed_from_initial=int(torch.count_nonzero(current.to(torch.bfloat16) != initial)))


def optimizer_state_summary(optimizer, parameters):
    import torch
    result = {}
    for name, parameter in parameters.items():
        entries, master_keys = {}, []
        def visit(value, path):
            if any(token in path.lower() for token in ('master', 'fp32_param', 'param_fp32', 'param32')):
                master_keys.append(path)
            if torch.is_tensor(value):
                entries[path] = dict(dtype=str(value.dtype), shape=list(value.shape), device=str(value.device))
            elif isinstance(value, dict):
                for key, item in value.items():
                    visit(item, f'{path}.{key}' if path else str(key))
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    visit(item, f'{path}[{index}]')
            else:
                entries[path] = dict(type=type(value).__name__)
        visit(optimizer.state[parameter], '')
        result[name] = dict(entries=entries, master_key_names=sorted(set(master_keys)),
            master_key_found=bool(master_keys),
            master_key_interpretation='Key-name inspection only;absence does not prove internal implementation has no transient FP32 arithmetic.')
    return result


def run(request, directory, lease):
    import torch
    from training.demo.backend import validate_active_authority
    validate_request(request)
    directory = contained(directory, DATA_ROOT)
    require(directory.is_dir() and directory.name == request['job_id']
            and not any((directory / name).exists() for name in ('inputs.json', 'progress.json', 'optimizer-summary.json')),
            'new optimizer diagnostic job directory required')
    began = time.monotonic()
    def check():
        require(time.monotonic() - began < 110, 'optimizer diagnostic soft deadline reached;no retry')
        validate_active_authority(lease)
    check()
    parent = contained(request['parent_checkpoint'], DATA_ROOT)
    require(parent.is_file(), 'existing parent checkpoint required;no download')
    # Trusted operator-owned file only. Ignore saved optimizer state completely.
    tick = time.monotonic()
    payload = torch.load(parent, map_location='cpu', weights_only=False)
    initials = select_parent_tensors(payload)
    del payload
    gc.collect()
    cpu_load_seconds = time.monotonic() - tick
    gradients = {name: synthetic_gradient(value) for name, value in initials.items()}
    inputs = dict(request=copy.deepcopy(request), limitations=copy.deepcopy(LIMITATIONS),
        parent_checkpoint=str(parent), parent_checkpoint_bytes=parent.stat().st_size,
        cpu_load_seconds=cpu_load_seconds, initial_parameters={name: dict(shape=list(value.shape),
            dtype=str(value.dtype), numel=value.numel(), minimum=float(value.float().min()),
            maximum=float(value.float().max()), l2_norm=float(torch.linalg.vector_norm(value.float())))
            for name, value in initials.items()},
        gradient=dict(formula='BF16((1-2*(flat_index%2))*0.001);same values at every step',
            bf16_then_fp32_widening=True, same_values_between_cases=True, real_gradient=False),
        case_order=['bf16_parameters', 'fp32_parameters'], optimizer_state_initialization='fresh in both cases;not a resume')
    write_json(directory / 'inputs.json', inputs)
    results = []
    optimizer, parameters = None, None
    try:
        check()
        import bitsandbytes as bnb
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one externally leased visible CUDA device required')
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        inputs.update(torch_version=torch.__version__, bitsandbytes_version=bnb.__version__)
        write_json(directory / 'inputs.json', inputs)
        for label, dtype in (('bf16_parameters', torch.bfloat16), ('fp32_parameters', torch.float32)):
            check()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            parameters = clone_case_parameters(initials, dtype, 'cuda')
            optimizer = bnb.optim.AdamW8bit(list(parameters.values()), lr=2e-6, betas=(.9, .95),
                weight_decay=.01, eps=1e-8)
            previous = {name: value.detach().cpu().clone() for name, value in parameters.items()}
            case = dict(name=label, parameter_dtype=str(dtype), status='running', steps=[])
            results.append(case)
            for step in range(1, 21):
                check()
                for name, parameter in parameters.items():
                    parameter.grad = gradients[name].to(device='cuda', dtype=dtype)
                torch.cuda.synchronize()
                tick = time.monotonic()
                optimizer.step()
                torch.cuda.synchronize()
                update_seconds = time.monotonic() - tick
                snapshots = {name: value.detach().cpu().clone() for name, value in parameters.items()}
                row = dict(step=step, synchronized_optimizer_step_seconds=update_seconds,
                    parameters={name: change_summary(value, previous[name], initials[name]) for name, value in snapshots.items()})
                case['steps'].append(row)
                previous = snapshots
                optimizer.zero_grad(set_to_none=True)
                write_json(directory / 'progress.json', dict(status='running', cases=results,
                    limitations=LIMITATIONS, elapsed_seconds=time.monotonic() - began))
            case.update(status='completed', optimizer_state=optimizer_state_summary(optimizer, parameters),
                elapsed_seconds=time.monotonic() - started,
                elapsed_seconds_scope='case allocation,updates,authority checks and CPU summaries;not a performance benchmark',
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                final=copy.deepcopy(case['steps'][-1]['parameters']))
            parameter = None
            optimizer, parameters = None, None
            gc.collect()
            torch.cuda.empty_cache()
        result = dict(status='completed', mode=FIXED['mode'], evidence='synthetic_optimizer_mechanism_only',
            cases=results, limitations=LIMITATIONS, cpu_load_seconds=cpu_load_seconds,
            elapsed_seconds=time.monotonic() - began, actual_optimizer_updates=40,
            actual_parameter_updates=80, model_forward_calls=0, backward_calls=0, seed=42,
            torch_version=torch.__version__, bitsandbytes_version=bnb.__version__,
            automatic_retry=False, quality_improvement_verified=False)
        write_json(directory / 'optimizer-summary.json', result)
        write_json(directory / 'progress.json', result)
        return result
    except BaseException as error:
        failed = dict(status='failed', error=f'{type(error).__name__}: {error}', cases=results,
            limitations=LIMITATIONS, elapsed_seconds=time.monotonic() - began, automatic_retry=False)
        write_json(directory / 'optimizer-summary.json', failed)
        write_json(directory / 'progress.json', failed)
        raise
    finally:
        parameter = None
        optimizer, parameters = None, None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
