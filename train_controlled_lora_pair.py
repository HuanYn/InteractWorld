"""Matched twenty-new-step LoRA switch experiment, never a parent strict resume.

Both arms inherit only controlled20 Adapter/gate, rebuild seed42 zero-B LoRA,
start a new optimizer, and consume absolute samples160..319. Only within-arm
step2->20 restores optimizer/RNG. No old entrypoint is patched or altered.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

import train_action_teacher as teacher
import train_controlled_action_teacher as parent
from training.config import load_config as _load_config
from training.data.action_resampled import sampling_contract as original_sampling_contract
from training.models.lora import LoRALinear, load_trainable_state_dict, trainable_state_dict
from training.runtime import OptimizerStepProfiler, append_jsonl, git_revision, seed_everything, sha256_file

STAGE = STAGE_NAME = 'controlled_action_lora_pair_native97_v1'
CHECKPOINT_SCHEMA = 'controlled_action_lora_pair_checkpoint_v1'
ARMS = ('frozen', 'trainable')
MAX_STEPS, START_SAMPLE, LORA_LR = 20, 160, 2e-6
GATE_LR, GATE_CHANNELS, RATIO_CAP = parent.GATE_LR, parent.GATE_CHANNELS, parent.RATIO_CAP
FACTORY, MANIFEST_SHA256 = parent.FACTORY, parent.MANIFEST_SHA256
_require = parent._require
base_identity = parent.base_identity
validate_base_identity = parent.validate_base_identity
_sample_parameters = parent._sample_parameters
_gate_statistics = parent._gate_statistics
_save_checkpoint = parent._save_checkpoint


def arm_for_config(config, arm=None):
    inferred = {0.0: 'frozen', LORA_LR: 'trainable'}.get(config.optimizer.lora_lr)
    _require(inferred in ARMS and (arm is None or arm == inferred), 'LoRA arm and fixed LR disagree')
    return inferred


def validate_probe_config(config, arm=None):
    arm_for_config(config, arm)
    normalized = copy.deepcopy(config)
    normalized.optimizer.lora_lr = 0.0
    parent.validate_probe_config(normalized)
    _require(config.data.num_workers == 0, 'matched pair requires deterministic single-process data reads')


def load_config(path):
    config = _load_config(path)
    validate_probe_config(config)
    return config


def validate_config(config, *, data_root, output_dir, model_dir, arm=None):
    validate_probe_config(config, arm)
    normalized = copy.deepcopy(config)
    normalized.optimizer.lora_lr = 0.0
    parent.validate_config(normalized, data_root=data_root, output_dir=output_dir, model_dir=model_dir)


def artifact_hashes(config, config_path):
    validate_probe_config(config)
    normalized = copy.deepcopy(config)
    normalized.optimizer.lora_lr = 0.0
    return parent.artifact_hashes(normalized, config_path)


def precision_contract(arm):
    _require(arm in ARMS, 'unknown LoRA arm')
    return dict(backbone_parameter_dtype='bfloat16', adapter_parameter_dtype='bfloat16',
        lora_parameter_dtype='float32', lora_trainable=arm == 'trainable',
        lora_gradient_dtype='float32' if arm == 'trainable' else None,
        gate_parameter_dtype='float32', adapter_gradient_dtype='bfloat16',
        gate_gradient_dtype='float32', forward_autocast_dtype='bfloat16', separate_master_copy=False)


def method_contract(arm):
    _require(arm in ARMS, 'unknown LoRA arm')
    result = parent.method_contract()
    result.update(method=STAGE, arm=arm, hyperparameters_source='project_fixed_equal_lora_switch_protocol_v1',
        train_lora=arm == 'trainable', frozen_lora_zero_residual=arm == 'frozen',
        lora_lr=LORA_LR if arm == 'trainable' else 0.0, lora_weight_decay=.01,
        lora_checkpoint_storage=arm == 'trainable',
        gate_initialization='inherited_controlled20_gate_weights_only',
        new_optimizer=True, parent_strict_resume=False, first_absolute_sample_index=START_SAMPLE,
        last_absolute_sample_index=START_SAMPLE + MAX_STEPS * 8 - 1,
        precision=precision_contract(arm))
    return result


def sampling_contract(training):
    policy = original_sampling_contract(replace(training, max_steps=40))
    policy.update(first_absolute_sample_index=START_SAMPLE, last_absolute_sample_index=319,
        stage_virtual_samples=160, stage_micro_batches=160,
        loader_virtual_samples=320, consumed_prefix_samples=START_SAMPLE,
        offset_policy='absolute_original_sample_index_not_reseed_or_modulo')
    return policy


def _validate_lineage(lineage):
    _require(isinstance(lineage, dict) and lineage.get('mode') == 'weights_only_new_stage'
        and lineage.get('parent_stage') == parent.STAGE and lineage.get('parent_step') == 20
        and lineage.get('parent_micro_batches_consumed') == START_SAMPLE
        and lineage.get('optimizer_restored') is False and lineage.get('rng_restored') is False
        and lineage.get('r005_inherited') is False and lineage.get('teacher64_inherited') is False,
        'controlled20 weights-only parent lineage required')
    for key in ('parent_checkpoint', 'parent_config', 'parent_job'):
        parent._absolute(lineage.get(key, ''))
    _require(Path(lineage['parent_checkpoint']).name == 'step-0000020.pt', 'parent must be controlled20 endpoint')
    _require(isinstance(lineage.get('parent_checkpoint_stat'), dict)
        and isinstance(lineage.get('parent_manifest_hashes'), dict)
        and isinstance(lineage.get('base_identity'), dict), 'parent identity metadata missing')


def initialization_contract(config, lineage, arm=None):
    arm = arm_for_config(config, arm)
    _validate_lineage(lineage)
    return dict(mode='controlled20_adapter_gate_weights_only_new_optimizer',
        base_model_path=config.model.base_model_path, source_checkpoint=lineage['parent_checkpoint'],
        adapter_initialization='controlled20_weights', gate_initialization='controlled20_FP32_weights',
        lora_initialization='FP32_kaiming_uniform_A_zero_B_seed42', lora_seed=42,
        lora_trainable=arm == 'trainable', optimizer_restored=False, rng_restored=False,
        start_step=0, parent_optimizer_steps=20, micro_batches_consumed=START_SAMPLE,
        training_seed=42, parent_strict_resume=False)


def load_parent(*, parent_checkpoint, parent_config, parent_job, config):
    """Existing trusted parent checkpoint; no full base/weight SHA sweep."""
    source_config = parent.load_config(parent_config)
    source_hashes = parent.artifact_hashes(source_config, parent_config)
    payload = torch.load(parent_checkpoint, map_location='cpu', weights_only=False)
    parent.validate_checkpoint(payload, config=source_config, hashes=source_hashes)
    _require(payload['step'] == 20, 'parent must be the fixed controlled20 endpoint')
    _require(source_config.to_dict()['model'] == config.to_dict()['model']
        and source_config.to_dict()['data'] == config.to_dict()['data'], 'parent model/data configuration changed')
    stat = Path(parent_checkpoint).stat()
    lineage = dict(mode='weights_only_new_stage', parent_stage=payload['stage'], parent_step=20,
        parent_checkpoint=str(Path(parent_checkpoint).resolve()), parent_config=str(Path(parent_config).resolve()),
        parent_job=str(Path(parent_job).resolve()), parent_micro_batches_consumed=160,
        parent_checkpoint_stat=dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns),
        parent_manifest_hashes=source_hashes, base_identity=payload['base_identity'],
        optimizer_restored=False, rng_restored=False, r005_inherited=False, teacher64_inherited=False,
        verification='existing_parent_schema_tensor_contract_manifest_receipt_and_stat_no_repeated_full_weight_hash')
    _validate_lineage(lineage)
    return payload['trainable_model'], lineage


def _group(name):
    if name.endswith('.lora_a.weight'):
        return 'lora_a'
    if name.endswith('.lora_b.weight'):
        return 'lora_b'
    return parent._group(name)


def _model_signature(model, arm):
    result = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        group = _group(name)
        dtype = torch.bfloat16 if group == 'adapter' else torch.float32
        _require(p.dtype == dtype, f'incorrect actual {group} parameter dtype')
        if group == 'gate':
            _require(p.shape == (GATE_CHANNELS,), 'gate must have3072 channels')
        result[name] = dict(shape=list(p.shape), dtype=str(dtype), group=group)
    expected = {'adapter', 'gate'} | ({'lora_a', 'lora_b'} if arm == 'trainable' else set())
    _require({v['group'] for v in result.values()} == expected, 'arm trainable groups differ')
    return result


def _build_model(config, device, arm=None):
    arm = arm_for_config(config, arm)
    validate_probe_config(config, arm)
    model, summary = parent._build_model(config, device)
    for module in model.modules():
        if type(module) is LoRALinear:
            module.lora_a.requires_grad_(arm == 'trainable')
            module.lora_b.requires_grad_(arm == 'trainable')
    _model_signature(model, arm)
    return model, replace(summary, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))


def install_parent_weights(model, state):
    parameters = dict(model.named_parameters())
    expected = {name for name, p in parameters.items() if p.requires_grad and _group(name) in ('adapter', 'gate')}
    _require(set(state) == expected, 'parent must contain only matching Adapter/gate tensors')
    with torch.no_grad():
        for name, tensor in state.items():
            p = parameters[name]
            _require(tensor.dtype == p.dtype and tensor.shape == p.shape and bool(torch.isfinite(tensor).all()),
                'parent tensor shape/precision differs: ' + name)
            p.copy_(tensor)


def _optimizer_groups(model, config, arm):
    _model_signature(model, arm)
    params = {group: [p for name, p in model.named_parameters() if p.requires_grad and _group(name) == group]
              for group in ('adapter', 'gate', 'lora_a', 'lora_b')}
    groups = [dict(params=params['adapter'], name='action_adapter', lr=1e-5, weight_decay=.01),
              dict(params=params['gate'], name='controlled_action_gate', lr=GATE_LR, weight_decay=0.)]
    if arm == 'trainable':
        groups.append(dict(params=params['lora_a'] + params['lora_b'], name='lora', lr=LORA_LR, weight_decay=.01))
    return groups


def _build_optimizer(model, config, arm=None):
    arm = arm_for_config(config, arm)
    import bitsandbytes as bnb
    return bnb.optim.AdamW8bit(_optimizer_groups(model, config, arm), betas=config.optimizer.betas)


def _gradient_precision(model, arm):
    groups = {}
    _model_signature(model, arm)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            _require(p.grad is None, 'frozen parameter unexpectedly has a gradient')
            continue
        group = _group(name)
        dtype = torch.bfloat16 if group == 'adapter' else torch.float32
        _require(p.grad is not None and p.dtype == p.grad.dtype == dtype, f'{group} actual gradient precision differs')
        sample = p.grad.detach().reshape(-1)[::max(1, math.ceil(p.numel() / parent.SAMPLE_ELEMENTS))][:parent.SAMPLE_ELEMENTS].float()
        _require(bool(torch.isfinite(sample).all()), 'nonfinite sampled gradient')
        row = groups.setdefault(group, dict(parameter_dtype=str(dtype), gradient_dtype=str(dtype),
            gradient_tensors=0, sampled_gradient_elements=0, sampled_nonzero_gradients=0, sampled_gradient_l2_squared=0.))
        row['gradient_tensors'] += 1
        row['sampled_gradient_elements'] += sample.numel()
        row['sampled_nonzero_gradients'] += int(torch.count_nonzero(sample))
        row['sampled_gradient_l2_squared'] += float(sample.double().square().sum())
    for row in groups.values():
        row['sampled_gradient_l2'] = math.sqrt(row.pop('sampled_gradient_l2_squared'))
    return dict(observed_at='after_accumulation_before_clip_and_step', groups=groups,
        gradient_value_evidence='deterministic_per_tensor_strided_sample', frozen_lora_has_gradients=False)


def _update_evidence(model, before, *, step, arm):
    after = _sample_parameters(model)
    _require(set(before) == set(after), 'parameter sample names changed')
    groups = {}
    for name, value in after.items():
        _require(bool(torch.isfinite(value).all()), 'nonfinite updated parameter sample')
        diff = value.float() - before[name].float()
        row = groups.setdefault(_group(name), dict(sampled_elements=0, changed_elements=0, update_l2=0.))
        row['sampled_elements'] += diff.numel()
        row['changed_elements'] += int(torch.count_nonzero(diff))
        row['update_l2'] += float(diff.double().square().sum())
    for row in groups.values():
        row['update_l2'] = math.sqrt(row['update_l2'])
    for group in ('adapter', 'gate'):
        _require(groups[group]['changed_elements'] > 0, f'no actual {group} update')
    if arm == 'trainable':
        _require(groups['lora_b']['changed_elements'] > 0, 'FP32 LoRA B has no actual update')
        if step >= 2:
            _require(groups['lora_a']['changed_elements'] > 0, 'step2 onward FP32 LoRA A has no actual update')
    return dict(method='deterministic_per_tensor_strided_cpu_sample_v1', groups=groups,
        maximum_elements_per_tensor=parent.SAMPLE_ELEMENTS, sampled_not_full_parameter_census=True,
        adapter_updated=True, gate_updated=True, lora_trainable=arm == 'trainable',
        lora_a_zero_first_gradient_expected=arm == 'trainable' and step == 1)


def _checkpoint(model, optimizer, *, config, hashes, identity, lineage, arm, step, metrics):
    return dict(format_version=1, checkpoint_schema=CHECKPOINT_SCHEMA, stage=STAGE, arm=arm,
        config=config.to_dict(), manifest_hashes=hashes, base_identity=identity, parent_lineage=lineage,
        initialization=initialization_contract(config, lineage, arm), method_contract=method_contract(arm),
        precision_contract=precision_contract(arm), sampling_contract=sampling_contract(config.training),
        model_signature=_model_signature(model, arm), trainable_model=trainable_state_dict(model),
        optimizer=optimizer.state_dict(), step=step, micro_batches_consumed=START_SAMPLE + step * 8,
        new_stage_micro_batches_consumed=step * 8, parent_optimizer_steps=20, metrics=metrics,
        torch_rng_state=torch.get_rng_state(), cuda_rng_state_all=torch.cuda.get_rng_state_all(),
        python_rng_state=random.getstate(), numpy_rng_state=np.random.get_state())


def validate_checkpoint(payload, *, config, hashes, arm=None):
    arm = arm_for_config(config, arm)
    validate_probe_config(config, arm)
    _require(isinstance(payload, dict) and payload.get('format_version') == 1
        and payload.get('checkpoint_schema') == CHECKPOINT_SCHEMA and payload.get('stage') == STAGE
        and payload.get('arm') == arm, 'checkpoint is not the selected new LoRA-pair arm')
    lineage = payload.get('parent_lineage')
    _validate_lineage(lineage)
    expected = dict(config=config.to_dict(), manifest_hashes=hashes,
        initialization=initialization_contract(config, lineage, arm), method_contract=method_contract(arm),
        precision_contract=precision_contract(arm), sampling_contract=sampling_contract(config.training))
    for name, value in expected.items():
        _require(payload.get(name) == value, f'strict pair checkpoint {name} changed')
    step = payload.get('step')
    _require(type(step) is int and step in (2,20)
        and payload.get('micro_batches_consumed') == START_SAMPLE + step * 8
        and payload.get('new_stage_micro_batches_consumed') == step * 8
        and payload.get('parent_optimizer_steps') == 20, 'pair absolute sample/step cursor differs')
    state, signature = payload.get('trainable_model'), payload.get('model_signature')
    _require(isinstance(state, dict) and bool(state) and isinstance(signature, dict)
        and set(state) == set(signature), 'pair state/signature tensor names differ')
    groups = set()
    for name, value in state.items():
        group = _group(name)
        groups.add(group)
        dtype = torch.bfloat16 if group == 'adapter' else torch.float32
        _require(torch.is_tensor(value) and value.dtype == dtype and bool(torch.isfinite(value).all())
            and signature[name] == dict(shape=list(value.shape), dtype=str(dtype), group=group),
            'pair tensor shape/precision differs: ' + name)
        if group == 'gate':
            _require(value.shape == (GATE_CHANNELS,), 'gate must have3072 channels')
    _require(groups == {'adapter', 'gate'} | ({'lora_a', 'lora_b'} if arm == 'trainable' else set()),
        'checkpoint trainable arm groups differ')
    _require(payload.get('base_identity') == lineage['base_identity'], 'parent base identity differs')
    return payload


def validate_resume(path, *, config, hashes, arm=None):
    arm = arm_for_config(config, arm)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    validate_checkpoint(payload, config=config, hashes=hashes, arm=arm)
    optimizer = payload.get('optimizer', {})
    groups = optimizer.get('param_groups', [])
    names = ['action_adapter', 'controlled_action_gate'] + (['lora'] if arm == 'trainable' else [])
    _require(isinstance(optimizer.get('state'), dict) and [g.get('name') for g in groups] == names,
        'resume optimizer groups differ')
    ids = []
    for group, name, lr, decay in zip(groups, names, [1e-5, GATE_LR, LORA_LR], [.01, 0., .01]):
        group_names = {'action_adapter': {'adapter'}, 'controlled_action_gate': {'gate'}, 'lora': {'lora_a', 'lora_b'}}[name]
        count = sum(s['group'] in group_names for s in payload['model_signature'].values())
        _require(group.get('lr') == lr and tuple(group.get('betas', ())) == (.9, .95)
            and group.get('weight_decay') == decay and len(group.get('params', [])) == count,
            'resume optimizer configuration/count changed')
        ids.extend(group['params'])
    _require(len(ids) == len(set(ids)) and set(ids) == set(optimizer['state'])
        and all(bool(optimizer['state'][key]) for key in ids), 'resume optimizer state incomplete')
    try:
        torch.Generator(device='cpu').set_state(payload['torch_rng_state'])
        random.Random().setstate(payload['python_rng_state'])
        np.random.RandomState().set_state(payload['numpy_rng_state'])
        cuda = payload['cuda_rng_state_all']
        _require(isinstance(cuda, list) and len(cuda) == 1 and all(torch.is_tensor(v)
            and v.dtype == torch.uint8 and v.ndim == 1 and v.numel() for v in cuda), 'single-GPU RNG required')
    except (KeyError, RuntimeError, TypeError) as exc:
        raise ValueError('resume RNG state missing/invalid') from exc
    return payload


def build_loader(config):
    # The underlying dataset receives320 absolute indices; skip160 once below.
    # A max20 loader would modulo-reset cursor160 to0 and silently repeat data.
    return teacher._import_factory(FACTORY)(config=config.data, training=replace(config.training, max_steps=40))


def launch(config, config_path, *, data_root, output_dir, model_dir, arm, parent_checkpoint,
           parent_config, parent_job, parent_lineage, stop_after_step, resume=None,
           shared_lease, shared_max_seconds, shared_request_sha256):
    validate_config(config, data_root=data_root, output_dir=output_dir, model_dir=model_dir, arm=arm)
    _validate_lineage(parent_lineage)
    _require(type(stop_after_step) is int and stop_after_step in (2, 20), 'pair endpoints are exactly2 and20')
    _require(type(shared_max_seconds) is int and 1 <= shared_max_seconds <=3600, 'finite <=3600s watchdog required')
    output = Path(output_dir).resolve()
    if resume is None:
        _require(stop_after_step == 2, 'new arm must first measure exactly two steps')
        _require(not output.exists() or not any(output.iterdir()), 'new arm output must be empty')
    else:
        _require(Path(resume).resolve().parent == output / 'checkpoints', 'strict resume belongs to this arm run')
    hashes, identity = artifact_hashes(config, config_path), base_identity(config)
    _require(identity == parent_lineage['base_identity'], 'official base changed from the controlled20 parent')
    restored = validate_resume(resume, config=config, hashes=hashes, arm=arm) if resume else None
    state = None
    if restored:
        _require(restored['step'] == 2 and stop_after_step ==20 and restored['parent_lineage'] == parent_lineage,
            'only the same arm2->20 strict continuation is allowed')
        _require(Path(resume).name == 'step-0000002.pt', 'resume filename/cursor differs')
    else:
        state, observed = load_parent(parent_checkpoint=parent_checkpoint, parent_config=parent_config,
            parent_job=parent_job, config=config)
        _require(observed == parent_lineage, 'parent weights-only identity changed after preparation')
    start = restored['step'] if restored else 0
    gpu = teacher._shared_training_gpu(shared_lease, max_seconds=shared_max_seconds,
        request_sha256=shared_request_sha256)
    _require(torch.cuda.is_available(), 'leased pair training requires CUDA')
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(42)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(stage=STAGE, arm=arm, config=config.to_dict(), manifest_hashes=hashes,
        base_identity=identity, parent_lineage=parent_lineage, method_contract=method_contract(arm),
        initialization=initialization_contract(config, parent_lineage, arm), sampling_contract=sampling_contract(config.training),
        command=sys.argv, git_revision=git_revision(Path(__file__).parent), entry_sha256=sha256_file(__file__),
        gpu_state=gpu.as_dict(), shared_guard=dict(lease=shared_lease, max_seconds=shared_max_seconds,
            request_sha256=shared_request_sha256, watchdog_owner='external_guard_worker'),
        resume=resume, resume_step=start, stop_after_step=stop_after_step, started_unix=time.time())
    if not resume:
        with (output / 'run_metadata.json').open('x', encoding='utf-8') as handle:
            json.dump(metadata, handle, indent=2)
    append_jsonl(output / 'executions.jsonl', metadata)
    profiler = OptimizerStepProfiler(device, output_dir=output / 'profile_output', gpu_uuid=shared_lease['gpu_uuid'])
    began = time.monotonic()
    try:
        with profiler.phase('model_load'):
            model, summary = _build_model(config, device, arm=arm)
        if state is not None:
            install_parent_weights(model, state)
            del state
        optimizer = _build_optimizer(model, config, arm)
        if restored:
            _require(restored['model_signature'] == _model_signature(model, arm), 'real model resume signature differs')
            load_trainable_state_dict(model, restored['trainable_model'])
            optimizer.load_state_dict(restored['optimizer'])
        loader = build_loader(config)
        _require(len(loader) == 320, 'absolute dataset must expose320 micro batches')
        _, iterator = teacher._iterator_at_micro_batch(loader, START_SAMPLE + start * 8)
        if restored:
            teacher._restore_rng_state(restored, require_complete=True)
            del restored
        model.zero_grad(set_to_none=True)
        latest = {}
        for step in range(start + 1, stop_after_step + 1):
            before = _sample_parameters(model)
            profiler.begin_step(step)
            losses = []
            for _ in range(8):
                profiler.set_phase('data_wait')
                batch = next(iterator)
                profiler.set_phase('forward')
                with torch.autograd.graph.save_on_cpu(pin_memory=True), torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = teacher._forward_loss(model, batch, config, device)
                _require(bool(torch.isfinite(loss)), 'nonfinite pair loss')
                profiler.set_phase('backward')
                (loss / 8).backward()
                losses.append(float(loss.detach()))
                del loss
            precision = _gradient_precision(model, arm)
            if arm == 'trainable':
                _require(precision['groups']['lora_b']['sampled_nonzero_gradients'] > 0,
                    'LoRA B needs an actual learning gradient, not only optimizer decay')
                if step >=2:
                    _require(precision['groups']['lora_a']['sampled_nonzero_gradients'] > 0,
                        'LoRA A needs an actual learning gradient by step2')
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            _require(bool(torch.isfinite(norm)), 'nonfinite accumulated gradient')
            precision['gradients_finite'] = True
            profiler.set_phase('optimizer_update')
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            timing = profiler.end_step(step)
            evidence = _update_evidence(model, before, step=step, arm=arm)
            del before
            memory = timing['memory']
            latest = dict(loss=sum(losses) / 8, grad_norm=float(norm), precision_summary=precision,
                update_evidence=evidence, gate_statistics=_gate_statistics(model),
                optimizer_step_seconds=timing['seconds'], step_profile=timing,
                absolute_sample_first=START_SAMPLE + (step-1) * 8, absolute_sample_last=START_SAMPLE + step * 8 - 1,
                elapsed_seconds=time.monotonic() - began, peak_vram_bytes=memory['cumulative_peak_allocated_bytes'],
                peak_reserved_vram_bytes=memory['cumulative_peak_reserved_bytes'],
                host_peak_rss_bytes=memory.get('host_peak_rss_bytes'), samples_per_second=8 / timing['seconds'])
            append_jsonl(output / 'metrics.jsonl', dict(step=step, micro_batches_consumed=START_SAMPLE + step * 8, **latest))
            print(json.dumps(dict(step=step, arm=arm, **latest)), flush=True)
            if step in (2,20):
                profiler.begin_checkpoint(step)
                target = _save_checkpoint(output, _checkpoint(model, optimizer, config=config, hashes=hashes,
                    identity=identity, lineage=parent_lineage, arm=arm, step=step, metrics=latest))
                profiler.end_checkpoint(step, checkpoint_path=str(target))
        result = dict(status='completed', stage=STAGE, arm=arm, step=stop_after_step,
            micro_batches_consumed=START_SAMPLE + stop_after_step * 8, new_stage_micro_batches_consumed=stop_after_step * 8,
            parent_optimizer_steps=20, parent_lineage=parent_lineage, resume_step=start, checkpoint_path=str(target),
            configured_max_steps=20, trainable_parameters=summary.trainable_parameters, **latest,
            qualification=dict(adapter_updated=True, gate_updated=True, lora_trainable=arm == 'trainable',
                lora_b_updated=arm == 'trainable', lora_a_updated_after_step2=arm == 'trainable',
                gradients_finite=True, peak_reserved_bytes=latest['peak_reserved_vram_bytes']),
            quality_improvement_verified=False)
        append_jsonl(output / 'execution_segments.jsonl', result)
        (output / 'execution_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
    except BaseException as exc:
        profiler.close(error=exc)
        append_jsonl(output / 'failures.jsonl', dict(status='failed', stage=STAGE, arm=arm,
            failure_type=type(exc).__name__, error=str(exc), timestamp_unix=time.time(), resume_step=start))
        raise
    finally:
        profiler.close()
