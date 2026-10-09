"""Independent TF20 -> TF160 continuation; never rewrites the running v1.

The only bridge is an explicitly recorded checkpoint-schema/config migration.
LoRA, frozen conditioning, complete native optimizer state, all RNG states and
the absolute data cursor survive it. This is not a weights-only restart, and
160 updates are an experiment length, not a claim of convergence.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_completion_causal_tf as previous

STAGE = 'completion_causal_tf60_sufficiency_v2'
CHECKPOINT_SCHEMA = 'completion_causal_tf60_sufficiency_checkpoint_v2'
ROOT = previous.ROOT.parent / 'causal-tf-sufficiency-v2'
FACTORY = 'train_completion_causal_tf_sufficiency:build_pair_dataloader'
NAMESPACE = previous.NAMESPACE
ENDPOINTS = (20, 40, 80, 160)
MAX_STEPS, ACCUMULATION, MICRO_BATCHES, HISTORY = 160, 8, 1280, previous.HISTORY
MAX_SEGMENT_STEPS, MAX_JOB_SECONDS = 16, 3600
TRAIN_PROTOCOL_GPU_SECONDS = 14 * 3600  # v1 and v2 combined, never an added grant
require = previous.require


@dataclass
class SufficiencyConfig(previous.CompletionConfig):
    def validate(self):
        validate_probe_config(self)


def validate_probe_config(config):
    require(type(config) is SufficiencyConfig, 'independent sufficiency configuration required')
    d, t = config.data, config.training
    require(d.data_factory == FACTORY and t.max_steps == MAX_STEPS
        and t.checkpoint_every == MAX_SEGMENT_STEPS and t.keep_last == 4 and t.keep_best == 0
        and Path(t.output_dir) == ROOT / 'training', 'fixed160/new output and bounded segment contract required')
    # A validation-only view checks every unchanged v1 hyperparameter. It is
    # never passed to a loader/model or serialized as the new configuration.
    view = previous.CompletionConfig(**{name:copy.deepcopy(getattr(config, name))
        for name in ('model', 'data', 'optimizer', 'training', 'error_recycling')})
    view.data.data_factory = previous.FACTORY
    view.training.max_steps = view.training.checkpoint_every = 20
    view.training.output_dir = str(previous.ROOT / previous.PROFILE_JOB / 'training')
    previous.validate_probe_config(view)


def config_from_v1(source_config):
    source_config.validate()
    result = SufficiencyConfig(**{name:copy.deepcopy(getattr(source_config, name))
        for name in ('model', 'data', 'optimizer', 'training', 'error_recycling')})
    result.data.data_factory = FACTORY
    result.training.max_steps = MAX_STEPS
    result.training.checkpoint_every = MAX_SEGMENT_STEPS
    result.training.output_dir = str(ROOT / 'training')
    result.validate()
    return result


def load_config(path):
    import yaml
    source = Path(path).read_text(encoding='utf-8')
    try:
        raw = json.loads(source)
    except json.JSONDecodeError:
        raw = yaml.safe_load(source)
    require(isinstance(raw, dict) and set(raw) == {'model', 'data', 'optimizer', 'training', 'error_recycling'},
        'sufficiency config fields differ')
    raw['data']['canonical_action_keys'] = tuple(raw['data']['canonical_action_keys'])
    raw['optimizer']['betas'] = tuple(raw['optimizer']['betas'])
    config = SufficiencyConfig(model=previous.CausalModelConfig(**raw['model']),
        data=previous.CausalDataConfig(**raw['data']), optimizer=previous.CausalOptimizerConfig(**raw['optimizer']),
        training=previous.CompletionTrainingConfig(**raw['training']),
        error_recycling=previous.ErrorRecyclingConfig(**raw['error_recycling']))
    config.validate()
    return config


def validate_config(config, *, output_dir, data_binding):
    config.validate()
    previous.validate_data_binding(data_binding)
    require(Path(output_dir) == Path(config.training.output_dir)
        and config.data.manifest_path == data_binding['subset']['path']
        and config.data.manifest_sha256 == data_binding['subset']['sha256'], 'exact subset/output binding differs')


class SufficiencyExactDataset(previous.CompletionExactDataset):
    def __init__(self, selection, *, cache_bindings):
        # Do not call the v1 constructor with its 160-sample ceiling. The
        # underlying exact-cache validator and every selected asset pin remain.
        previous.ExactCachedDataset.__init__(self, selection, cache_bindings=cache_bindings,
            num_samples=MICRO_BATCHES, seed=42)
        self.data_seed = previous._seed64(42, NAMESPACE)
        self.contract.update(namespace=NAMESPACE, data_seed=self.data_seed, virtual_samples=MICRO_BATCHES)


def build_dataset(binding):
    return SufficiencyExactDataset(previous.validate_data_binding(binding), cache_bindings=binding['cache_bindings'])


def sampling_contract(training, data_binding, *, dataset=None):
    require(training.max_steps == MAX_STEPS and training.seed == 42 and training.micro_batch_size == 1
        and training.gradient_accumulation_steps == ACCUMULATION, '160 absolute updates/1280 draws required')
    dataset = build_dataset(data_binding) if dataset is None else dataset
    return dict(namespace=NAMESPACE, dataset=copy.deepcopy(dataset.contract),
        qualified_data_binding=copy.deepcopy(data_binding), first_stage_sample_index=0,
        last_stage_sample_index=MICRO_BATCHES-1, virtual_samples=MICRO_BATCHES,
        historical_micro_batches_before_stage=HISTORY, reference_enabled=False,
        qualification_windows_used=0, sampling_restarts_at_each_segment=False,
        inherited_v1_micro_batches=160, bridge_next_absolute_sample_index=160)


def build_pair_dataloader(*, config, training, data_binding):
    require(config.data_factory == FACTORY and config.manifest_path == data_binding['subset']['path']
        and config.manifest_sha256 == data_binding['subset']['sha256'] and config.num_workers == 0,
        'new exact60 continuation factory required')
    dataset = build_dataset(data_binding)
    sampling_contract(training, data_binding, dataset=dataset)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)


def exposures(dataset, consumed):
    """Whole causal trajectory exposures, including inherited samples 0..159."""
    require(type(consumed) is int and 0 <= consumed <= MICRO_BATCHES
        and len(dataset) == MICRO_BATCHES and len(dataset.episodes) == 60, 'invalid1280 exact60 sampling cursor')
    module = previous.common.data_module()
    count = len(dataset.episodes)
    return [dict(**dict(zip(module.IDENTITY_FIELDS, module.window_key(row))), target=row['target'],
        exposures=(consumed+count-1-slot)//count) for slot, row in enumerate(dataset.episodes)]


def counters(step):
    require(type(step) is int and 20 <= step <= MAX_STEPS, 'invalid continued causal optimizer step')
    return dict(step=step, optimizer_internal_step=step, micro_batches_consumed=step*ACCUMULATION,
        new_stage_micro_batches_consumed=step*ACCUMULATION,
        historical_micro_batches_before_stage=HISTORY,
        lifetime_micro_batches_consumed=HISTORY+step*ACCUMULATION, sampling_cursor=step*ACCUMULATION,
        inherited_v1_optimizer_steps=20, v2_additional_optimizer_steps=step-20,
        v2_additional_micro_batches_consumed=(step-20)*ACCUMULATION)


def method_contract():
    result = previous.method_contract()
    result.update(method=STAGE, fresh_fp32_rank16_lora=False, new_optimizer=False,
        checkpoint_policy='every_finite_segment_and_fixed20_40_80_160_no_pruning',
        bridge_mode='explicit_v1_to_v2_schema_config_migration_with_complete_training_state',
        complete_optimizer_restored=True, complete_rng_restored=True, absolute_sampling_cursor_preserved=True,
        strict_v2_internal_resume=True, training_length_adequacy_proven=False)
    return result


def validate_rng(payload):
    require(all(k in payload for k in ('torch_rng_state', 'cuda_rng_state_all', 'python_rng_state', 'numpy_rng_state')),
        'complete Python/NumPy/Torch/CUDA RNG state required')
    cpu = payload['torch_rng_state']
    require(torch.is_tensor(cpu) and cpu.dtype == torch.uint8 and cpu.device.type == 'cpu'
        and cpu.ndim == 1, 'native CPU torch RNG required')
    torch.Generator(device='cpu').set_state(cpu)
    random.Random().setstate(payload['python_rng_state'])
    np.random.RandomState().set_state(payload['numpy_rng_state'])
    cuda = payload['cuda_rng_state_all']
    require(isinstance(cuda, list) and len(cuda) == 1 and torch.is_tensor(cuda[0])
        and cuda[0].dtype == torch.uint8 and cuda[0].device.type == 'cpu'
        and cuda[0].ndim == 1 and cuda[0].numel() > 0, 'complete single-GPU CUDA RNG required')


def bridge_source_binding():
    """Read completed v1 records and exact asset pins, without loading weights."""
    from training.studio import completion_causal_train as old
    request, execution, _ = old.completed_endpoint(20)
    paths = old.paths(20)
    stat = paths['checkpoint'].stat()
    require(execution['sampling_cursor'] == 160 and execution['optimizer_internal_step'] == 20,
        'only settled complete v1 TF20 may bridge')
    return dict(mode='v1_tf20_complete_state_schema_migration_not_weights_only',
        source_stage=previous.STAGE, source_checkpoint_schema=previous.CHECKPOINT_SCHEMA,
        checkpoint=str(paths['checkpoint']), checkpoint_stat=dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns),
        config=str(paths['config']), config_sha256=request['config_sha256'],
        endpoint=str(paths['endpoint']), records=old.record_binding(paths['endpoint']),
        source_step=20, source_sampling_cursor=160, namespace=NAMESPACE,
        parent_lineage=copy.deepcopy(execution['parent_lineage']),
        qualified_data_binding=copy.deepcopy(request['data_binding']),
        optimizer_restored=True, rng_restored=True, new_optimizer=False, weights_only=False)


def validate_bridge_binding(binding):
    require(isinstance(binding, dict) and binding.get('mode') == 'v1_tf20_complete_state_schema_migration_not_weights_only'
        and binding.get('source_stage') == previous.STAGE
        and binding.get('source_checkpoint_schema') == previous.CHECKPOINT_SCHEMA
        and binding.get('source_step') == 20 and binding.get('source_sampling_cursor') == 160
        and binding.get('namespace') == NAMESPACE and binding.get('optimizer_restored') is True
        and binding.get('rng_restored') is True and binding.get('new_optimizer') is False
        and binding.get('weights_only') is False, 'explicit complete-state TF20 bridge required')
    expected_checkpoint = previous.ROOT / previous.PROFILE_JOB / 'training/checkpoints/step-0000020.pt'
    require(Path(binding.get('checkpoint', '')) == expected_checkpoint
        and Path(binding.get('config', '')) == previous.ROOT / previous.PROFILE_JOB / 'training.yaml'
        and Path(binding.get('endpoint', '')) == previous.ROOT / 'ccf-to20-v1', 'bridge source locations differ')
    require(set(binding.get('checkpoint_stat', {})) == {'bytes', 'mtime_ns'}
        and all(type(v) is int and v > 0 for v in binding['checkpoint_stat'].values())
        and set(binding.get('records', {})) == {n+'_sha256' for n in ('request', 'result', 'settlement')},
        'complete source stat/settled record binding required')
    pins = [binding.get('config_sha256'), *binding['records'].values()]
    require(all(isinstance(v, str) and len(v) == 64 and set(v) <= set('0123456789abcdef') for v in pins),
        'source small-record SHA256 pins required')
    previous.validate_lineage(binding.get('parent_lineage'))
    require(binding['parent_lineage']['qualified_data_binding'] == binding.get('qualified_data_binding'),
        'bridge parent/data binding differs')


def artifact_hashes(config, config_path, data_binding, bridge_binding):
    validate_config(config, output_dir=config.training.output_dir, data_binding=data_binding)
    validate_bridge_binding(bridge_binding)
    require(bridge_binding['qualified_data_binding'] == data_binding, 'bridge data cannot change')
    from training import causal_tf
    from utils import controlled_causal_action
    return dict(training_config=previous.sha256_file(config_path), qualified_data_binding=copy.deepcopy(data_binding),
        entry_source=previous.sha256_file(__file__), v1_entry_source=previous.sha256_file(previous.__file__),
        causal_tf_source=previous.sha256_file(causal_tf.__file__),
        controlled_causal_source=previous.sha256_file(controlled_causal_action.__file__),
        causal_model_source=previous.sha256_file(Path(__file__).parent/'wan/modules/causal_model.py'),
        shared_parent_source=previous.sha256_file(previous.common.__file__),
        exact_dataset_source=previous.sha256_file(previous.common.data_module().__file__),
        bridge_source_binding=copy.deepcopy(bridge_binding))


def migrate_v1_payload(payload, *, source_config, source_hashes, source_sampling,
        config, hashes, data_binding, sampling, bridge_binding):
    """Pure schema migration; validates v1 first and does not modify it in place."""
    validate_bridge_binding(bridge_binding)
    previous.validate_checkpoint(payload, config=source_config, hashes=source_hashes,
        data_binding=data_binding, sampling=source_sampling)
    validate_rng(payload)
    require(payload['step'] == 20 and payload['sampling_cursor'] == 160
        and payload['parent_lineage'] == bridge_binding['parent_lineage']
        and data_binding == bridge_binding['qualified_data_binding'], 'actual v1 source state differs')
    require(config.to_dict() == config_from_v1(source_config).to_dict(), 'only allowlisted config migration permitted')
    # References deliberately survive; no dtype cast, optimizer reinitialization,
    # RNG draw or source tensor copy is permitted by this bridge.
    result = dict(payload, format_version=2, stage=STAGE, checkpoint_schema=CHECKPOINT_SCHEMA,
        config=config.to_dict(), manifest_hashes=hashes, method_contract=method_contract(),
        sampling_contract=sampling, schema_migration=copy.deepcopy(bridge_binding), **counters(20))
    validate_checkpoint(result, config=config, hashes=hashes, data_binding=data_binding,
        sampling=sampling, bridge_binding=bridge_binding)
    return result


def validate_checkpoint(payload, *, config, hashes, data_binding, sampling, bridge_binding):
    config.validate()
    validate_bridge_binding(bridge_binding)
    previous.validate_lineage(payload.get('parent_lineage'))
    expected = dict(format_version=2, stage=STAGE, checkpoint_schema=CHECKPOINT_SCHEMA,
        config=config.to_dict(), manifest_hashes=hashes, base_identity=bridge_binding['parent_lineage']['base_identity'],
        parent_lineage=bridge_binding['parent_lineage'], method_contract=method_contract(), sampling_contract=sampling,
        frozen_action_signature=previous.frozen_action_signature(), schema_migration=bridge_binding)
    require(all(payload.get(k) == v for k, v in expected.items())
        and data_binding == bridge_binding['qualified_data_binding'], 'strict v2 checkpoint contract differs')
    step = payload.get('step')
    require(type(step) is int and 20 <= step <= MAX_STEPS
        and all(payload.get(k) == v for k, v in counters(step).items()), 'optimizer/cursor/history differs')
    previous.validate_frozen_action(payload.get('frozen_action_model'))
    signature, state = payload['model_signature'], payload['trainable_model']
    previous.validate_lora_signature(signature)
    require(set(state) == set(signature), 'complete FP32 LoRA export required')
    for name, value in state.items():
        require(torch.is_tensor(value) and value.dtype == torch.float32
            and list(value.shape) == signature[name]['shape'] and bool(torch.isfinite(value).all()),
            'FP32 LoRA export invalid')
    require(payload['optimizer_mapping'] == previous.optimizer_mapping(signature, payload['optimizer'], step),
        'complete native optimizer mapping differs')
    validate_rng(payload)
    return payload


def next_segment(start, slowest):
    """Largest <=16-step finite segment, stopping exactly at the next endpoint."""
    require(type(start) is int and 20 <= start < MAX_STEPS
        and type(slowest) in (int, float) and math.isfinite(slowest) and slowest > 0,
        'actual completed cursor and positive measured slowest step required')
    endpoint = next(step for step in ENDPOINTS if step > start)
    available = min(MAX_SEGMENT_STEPS, endpoint-start)
    count = next((n for n in range(available, 0, -1)
        if math.ceil(slowest*n*1.35+180) <= MAX_JOB_SECONDS), None)
    require(count is not None, 'even one measured step exceeds3600; stop without extending')
    return dict(start_step=start, stop_after_step=start+count, new_steps=count,
        next_comparison_endpoint=endpoint, measured_slowest_step_seconds=float(slowest),
        max_seconds=math.ceil(slowest*count*1.35+180))


def validate_transition(*, output, resume, start_step, stop_after_step, measured_slowest_step_seconds,
        shared_max_seconds):
    require(Path(output) == ROOT/'training', 'independent v2 output required')
    expected = next_segment(start_step, measured_slowest_step_seconds)
    require(stop_after_step == expected['stop_after_step'] and shared_max_seconds == expected['max_seconds'],
        'use measured bounded segment, without endpoint skip or limit extension')
    source = (previous.ROOT / previous.PROFILE_JOB / 'training/checkpoints/step-0000020.pt'
        if start_step == 20 else Path(output)/f'checkpoints/step-{start_step:07d}.pt')
    require(resume is not None and Path(resume) == source, 'complete immediate predecessor required; no restart')
    return expected


def validate_budget_contract(contract, *, shared_max_seconds):
    require(isinstance(contract, dict) and set(contract) == {'scope', 'combined_v1_v2_cap_gpu_seconds',
        'combined_settled_gpu_seconds', 'remaining_planned_gpu_seconds', 'current_job_cleanup_seconds'}
        and contract['scope'] == 'completion_causal_tf_v1_and_sufficiency_v2_combined'
        and contract['combined_v1_v2_cap_gpu_seconds'] == TRAIN_PROTOCOL_GPU_SECONDS
        and contract['current_job_cleanup_seconds'] == 60, 'root combined14GPUh protocol accounting required')
    charged, remaining = contract['combined_settled_gpu_seconds'], contract['remaining_planned_gpu_seconds']
    require(type(charged) in (int, float) and math.isfinite(charged) and charged >= 0
        and type(remaining) in (int, float) and math.isfinite(remaining) and remaining >= 0
        and charged+shared_max_seconds+60+remaining <= TRAIN_PROTOCOL_GPU_SECONDS,
        'combined v1/v2 planned cost exceeds14GPUh; do not reset or borrow')


def load_resume(path, *, config, hashes, data_binding, sampling, bridge_binding, start_step):
    require(bridge_source_binding() == bridge_binding, 'settled v1 source records changed')
    if start_step == 20:
        source_config = previous.load_config(bridge_binding['config'])
        source_sampling = previous.sampling_contract(source_config.training, data_binding)
        source_hashes = previous.artifact_hashes(source_config, bridge_binding['config'], data_binding)
        payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        previous.validate_base_identity(payload, source_config)
        saved = migrate_v1_payload(payload, source_config=source_config, source_hashes=source_hashes,
            source_sampling=source_sampling, config=config, hashes=hashes, data_binding=data_binding,
            sampling=sampling, bridge_binding=bridge_binding)
    else:
        saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        validate_checkpoint(saved, config=config, hashes=hashes, data_binding=data_binding,
            sampling=sampling, bridge_binding=bridge_binding)
        previous.validate_base_identity(saved, config)
    require(saved['step'] == start_step, 'actual complete predecessor step differs')
    return saved


def _checkpoint(model, frozen, optimizer, *, config, hashes, lineage, sampling, step, metrics, bridge_binding):
    signature, state = previous.model_signature(model), optimizer.state_dict()
    return dict(format_version=2, stage=STAGE, checkpoint_schema=CHECKPOINT_SCHEMA, config=config.to_dict(),
        manifest_hashes=hashes, base_identity=lineage['base_identity'], parent_lineage=lineage,
        method_contract=method_contract(), sampling_contract=sampling, model_signature=signature,
        frozen_action_signature=previous.frozen_action_signature(), frozen_action_model=frozen,
        trainable_model=previous.trainable_state_dict(model), optimizer=state,
        optimizer_mapping=previous.optimizer_mapping(signature, state, step), schema_migration=bridge_binding,
        **counters(step), metrics=metrics, torch_rng_state=torch.get_rng_state(),
        cuda_rng_state_all=torch.cuda.get_rng_state_all(), python_rng_state=random.getstate(),
        numpy_rng_state=np.random.get_state())


def launch(config, config_path, *, lineage, data_binding, bridge_binding, start_step, stop_after_step,
        measured_slowest_step_seconds, resume, budget_contract, shared_lease, shared_max_seconds,
        shared_request_sha256):
    """Leased execution entry only; no launch/registration/queue side effects."""
    from training.demo.backend import validate_active_authority
    validate_config(config, output_dir=config.training.output_dir, data_binding=data_binding)
    previous.validate_lineage(lineage)
    validate_bridge_binding(bridge_binding)
    validate_budget_contract(budget_contract, shared_max_seconds=shared_max_seconds)
    segment = validate_transition(output=config.training.output_dir, resume=resume, start_step=start_step,
        stop_after_step=stop_after_step, measured_slowest_step_seconds=measured_slowest_step_seconds,
        shared_max_seconds=shared_max_seconds)
    require(previous.parent_binding() == lineage == bridge_binding['parent_lineage']
        and lineage['qualified_data_binding'] == data_binding, 'actual expanded80 parent/data changed')
    hashes = artifact_hashes(config, config_path, data_binding, bridge_binding)
    dataset = build_dataset(data_binding)
    sampling = sampling_contract(config.training, data_binding, dataset=dataset)
    saved = load_resume(resume, config=config, hashes=hashes, data_binding=data_binding,
        sampling=sampling, bridge_binding=bridge_binding, start_step=start_step)
    require(saved['parent_lineage'] == lineage, 'complete resume parent differs')
    gpu = previous.teacher._shared_training_gpu(shared_lease, max_seconds=shared_max_seconds,
        request_sha256=shared_request_sha256)
    require(shared_lease.get('gpu_index') in (1, 2) and torch.cuda.is_available(), 'leased Creator GPU1/2 only')
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    previous.seed_everything(42)
    output = Path(config.training.output_dir)
    if start_step == 20:
        require(not output.exists() or not any(output.iterdir()), 'new v2 output must be empty before bridge')
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(stage=STAGE, config=config.to_dict(), manifest_hashes=hashes, parent_lineage=lineage,
        sampling_contract=sampling, method_contract=method_contract(), schema_migration=bridge_binding,
        resume=resume, resume_step=start_step, stop_after_step=stop_after_step, measured_segment=segment,
        budget_contract=budget_contract, git_revision=previous.git_revision(Path(__file__).parent),
        entry_sha256=previous.sha256_file(__file__), gpu_state=gpu.as_dict(), started_unix=time.time(),
        shared_guard=dict(lease=shared_lease, max_seconds=shared_max_seconds, request_sha256=shared_request_sha256))
    if start_step == 20:
        with (output/'run_metadata.json').open('x', encoding='utf-8') as stream:
            json.dump(metadata, stream, indent=2)
    previous.append_jsonl(output/'executions.jsonl', metadata)
    profiler = previous.OptimizerStepProfiler(device, output_dir=output/'profile_output', gpu_uuid=shared_lease['gpu_uuid'])
    began, last_check = time.monotonic(), 0.

    def check():
        nonlocal last_check
        now = time.monotonic()
        require(now-began < shared_max_seconds-30, 'finite causal continuation deadline reached')
        if now-last_check >= 2:
            validate_active_authority(shared_lease)
            last_check = now

    try:
        frozen = saved['frozen_action_model']
        with profiler.phase('model_load'):
            model, summary = previous.build_model(config, device, frozen_action_model=frozen,
                trainable_model=saved['trainable_model'])
            signature = previous.model_signature(model)
            require(signature == saved['model_signature'], 'resumed LoRA signature differs')
            optimizer = previous.make_optimizer(model, saved['optimizer'], check=check)
        previous.teacher._restore_rng_state(saved, require_complete=True)
        del saved
        frozen_versions = {name:p._version for name, p in model.named_parameters() if not p.requires_grad}
        model.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        for step in range(start_step+1, stop_after_step+1):
            check()
            before = previous.sampled_lora(model)
            profiler.begin_step(step)
            loss, samples = previous.forward_backward(model, dataset, step=step, config=config,
                device=device, profiler=profiler, check=check)
            task = previous.gradient_evidence(model, step)
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            require(bool(torch.isfinite(norm)), 'nonfinite LoRA gradient norm')
            profiler.set_phase('optimizer_update')
            optimizer.step()
            after = previous.sampled_lora(model)
            require(all(bool(torch.isfinite(p).all()) for p in after.values()), 'nonfinite updated LoRA sample')
            changed = sum(int(torch.count_nonzero(after[name]-before[name])) for name in before)
            require(changed > 0, 'no representable LoRA update')
            require(all(not p.requires_grad and p.grad is None and p._version == frozen_versions[name]
                for name, p in model.named_parameters() if name in frozen_versions), 'frozen parameters modified')
            optimizer.zero_grad(set_to_none=True)
            if step == stop_after_step:
                native = optimizer.state_dict()
                previous.optimizer_mapping(signature, native, step)
                optimizer.load_state_dict(native)
                require(previous.master.equal_tree(native, optimizer.state_dict(), check), 'native full-state roundtrip differs')
            timing = profiler.end_step(step)
            memory = timing['memory']
            require(memory['cumulative_peak_reserved_bytes'] < 23*1024**3, 'continued TF reserved memory exceeds23GiB')
            latest = dict(loss=loss, loss_scope='causal_future_flow_MSE_clean_history', grad_norm=float(norm),
                optimizer_step_seconds=timing['seconds'], task_gradient_evidence=task,
                lora_changed_sampled_elements=changed, adapter_unchanged=True, gate_unchanged=True,
                native_state_restore_matched=True, micro_samples=samples, sample_namespace=NAMESPACE,
                exposures_since_causal_trajectory_start=exposures(dataset, step*ACCUMULATION), step_profile=timing,
                peak_vram_bytes=memory['cumulative_peak_allocated_bytes'],
                peak_reserved_vram_bytes=memory['cumulative_peak_reserved_bytes'], elapsed_seconds=time.monotonic()-began)
            previous.append_jsonl(output/'metrics.jsonl', dict(**counters(step), **latest))
            print(json.dumps(dict(**counters(step), **latest)), flush=True)
        profiler.begin_checkpoint(stop_after_step)
        target = previous.master.b.pair._save_checkpoint(output, _checkpoint(model, frozen, optimizer,
            config=config, hashes=hashes, lineage=lineage, sampling=sampling, step=stop_after_step,
            metrics=latest, bridge_binding=bridge_binding))
        profiler.end_checkpoint(stop_after_step, checkpoint_path=str(target))
        result = dict(status='completed', stage=STAGE, checkpoint_schema=CHECKPOINT_SCHEMA,
            **counters(stop_after_step), parent_lineage=lineage, base_identity=lineage['base_identity'],
            schema_migration=bridge_binding, model_signature=signature,
            frozen_action_signature=previous.frozen_action_signature(), sampling_contract=sampling,
            method_contract=method_contract(), resume_step=start_step, checkpoint_path=str(target),
            configured_max_steps=MAX_STEPS, trainable_parameters=summary.trainable_parameters, **latest,
            qualification=dict(lora_updated=True, gradients_finite=True, lora_B_task_gradient_nonzero=True,
                lora_A_task_gradient_nonzero=task['lora_A_task_gradient_nonzero'], adapter_unchanged=True,
                gate_unchanged=True, frozen_action_has_gradients=False, native_state_restore_matched=True,
                peak_reserved_bytes=latest['peak_reserved_vram_bytes']), reference_enabled=False,
            quality_improvement_verified=False, training_length_adequacy_proven=False, automatic_next_stage=False)
        previous.append_jsonl(output/'execution_segments.jsonl', result)
        (output/'execution_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
    except BaseException as error:
        profiler.close(error=error)
        previous.append_jsonl(output/'failures.jsonl', dict(status='failed', stage=STAGE,
            failure_type=type(error).__name__, error=str(error), resume_step=start_step, timestamp_unix=time.time()))
        raise
    finally:
        profiler.close()
