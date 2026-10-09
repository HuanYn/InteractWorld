"""Bounded, no-reference exact-window expansion from frozen balanced24 weights.

Only this new stage owns the 2/20/40/60/80 endpoints. Candidate metadata alone
never enables training: a separately pinned, explicit accepted subset is required.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext
import json
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

import train_balanced_lora_pair as balanced
import train_reference_teacher as parent
import train_master_writeback as master
from training.config import ActionTeacherConfig, load_config as _load_config
from training.models.lora import load_trainable_state_dict, trainable_state_dict
from training.runtime import OptimizerStepProfiler, append_jsonl, git_revision, seed_everything, sha256_file

STAGE = 'controlled_action_expanded_teacher_native97_v1'
CHECKPOINT_SCHEMA = 'controlled_action_expanded_teacher_checkpoint_v1'
ROOT = parent.ROOT.parent / 'teacher-expanded-data-v1'
PROFILE_JOB = 'ced-profile2-v1'
FACTORY = 'train_expanded_teacher:build_expanded_dataloader'
PREVIOUS = {2: 0, 20: 2, 40: 20, 60: 40, 80: 60}
ENDPOINTS = tuple(PREVIOUS)
MICRO_BATCHES, HISTORY = 640, 672
GATE, ADAPTER_SHAPES = master.GATE, master.ADAPTER_SHAPES
require = master.require
parent_paths, parent_binding = parent.parent_paths, parent.parent_binding
validate_lineage, load_parent = parent.validate_lineage, parent.load_parent
validate_base_identity = parent.validate_base_identity


def data_module():
    from training.data import exact_cached_dataset
    return exact_cached_dataset


def _read_pin(pin):
    require(isinstance(pin, dict) and set(pin) == {'path', 'sha256'}
            and isinstance(pin['path'], str) and Path(pin['path']).is_absolute()
            and '..' not in Path(pin['path']).parts
            and isinstance(pin['sha256'], str) and re.fullmatch('[0-9a-f]{64}', pin['sha256']),
            'absolute small JSON path and SHA256 required')
    require(sha256_file(pin['path']) == pin['sha256'], 'qualified data JSON pin changed')
    return json.loads(Path(pin['path']).read_text(encoding='utf-8'))


def validate_data_binding(binding):
    """Read small selection/review records, not tensors; never infer approval."""
    require(isinstance(binding, dict) and set(binding) == {
        'candidate_manifest', 'subset', 'review', 'cache_bindings'}, 'explicit qualified data binding required')
    _read_pin(binding['candidate_manifest'])
    subset, review = _read_pin(binding['subset']), _read_pin(binding['review'])
    candidate_sha = binding['candidate_manifest']['sha256']
    require(subset.get('schema') == 'expanded_teacher_qualified_subset_v1'
            and subset.get('candidate_manifest_sha256') == candidate_sha
            and subset.get('review_sha256') == binding['review']['sha256']
            and isinstance(subset.get('selected'), list) and subset['selected'], 'explicit reviewed subset required')
    require(review.get('schema') == 'expanded_teacher_visual_review_v1'
            and review.get('candidate_manifest_sha256') == candidate_sha
            and review.get('review_complete') is True and isinstance(review.get('windows'), list),
            'complete pinned original-frame visual review required')
    module = data_module()
    reviews = {}
    for row in review['windows']:
        key = module.window_key(row)
        require(key not in reviews and row.get('decision') in ('accept', 'reject', 'uncertain'),
                'duplicate or invalid visual review decision')
        reviews[key] = row
    for row in subset['selected']:
        decision = reviews.get(module.window_key(row), {})
        require(decision.get('decision') == 'accept'
                and isinstance(decision.get('evidence'), list) and decision['evidence']
                and all(isinstance(v, str) and v.strip() for v in decision['evidence']),
                'every selected exact window needs explicit accept and original-frame evidence')
    selection = module.ExactWindowSelection(binding['candidate_manifest']['path'], subset=subset['selected'])
    require(selection.manifest_sha256 == candidate_sha, 'candidate changed while preparing exact selection')
    require(isinstance(binding['cache_bindings'], dict)
            and set(binding['cache_bindings']) == {r['cache_origin'] for r in selection.rows},
            'exact used cache origins required')
    return selection


def build_dataset(binding):
    selection = validate_data_binding(binding)
    return data_module().ExactCachedDataset(selection, cache_bindings=binding['cache_bindings'],
                                            num_samples=MICRO_BATCHES, seed=42)


def sampling_contract(training, data_binding, *, dataset=None):
    require(training.max_steps == 80 and training.seed == 42 and training.micro_batch_size == 1
            and training.gradient_accumulation_steps == 8, 'fixed new80 x8 schedule required')
    dataset = build_dataset(data_binding) if dataset is None else dataset
    return dict(dataset=copy.deepcopy(dataset.contract), qualified_data_binding=copy.deepcopy(data_binding),
                first_stage_sample_index=0, last_stage_sample_index=639, virtual_samples=MICRO_BATCHES,
                historical_micro_batches_before_stage=HISTORY, qualification_windows_used=0,
                reference_enabled=False, sampling_restarts_at_each_segment=False)


def validate_probe_config(config):
    require(type(config) is ActionTeacherConfig, 'typed expanded configuration required')
    config.validate()
    require(config.data.data_factory == FACTORY and config.data.num_frames == 97
            and config.data.num_workers == 0 and config.data.prompt_cache_path is None
            and not config.data.low_sigma_stratification and config.data.precomputed_latents
            and config.data.precomputed_text_embeddings, 'exact native97 static-text cache required')
    require(config.model.action_scale == 1 and config.model.lora_alpha == 16
            and config.model.lora_dropout == 0 and config.model.timestep_shift == 5
            and config.model.gradient_checkpointing, 'unchanged controlled model required')
    require(config.optimizer.name == 'adamw8bit' and config.optimizer.adapter_lr == 1e-5
            and config.optimizer.lora_lr == 0 and tuple(config.optimizer.betas) == (.9, .95)
            and config.optimizer.weight_decay == .01 and config.optimizer.max_grad_norm == 1,
            'unchanged frozen-LoRA optimizer required')
    require(config.training.max_steps == 80 and config.training.checkpoint_every == 20
            and config.training.log_every == 1 and config.training.lora_parameter_dtype == 'float32'
            and Path(config.training.output_dir) == ROOT / PROFILE_JOB / 'training',
            'fixed independent80 trajectory/output required')


def load_config(path):
    config = _load_config(path)
    validate_probe_config(config)
    return config


def validate_config(config, *, output_dir, data_binding):
    from training.studio.controlled_action_train import MODEL
    validate_probe_config(config)
    validate_data_binding(data_binding)
    require(config.model.base_model_path == str(MODEL) and Path(output_dir) == Path(config.training.output_dir)
            and config.data.manifest_path == data_binding['subset']['path']
            and config.data.manifest_sha256 == data_binding['subset']['sha256'],
            'pinned base/output/explicit subset differs')


def build_expanded_dataloader(*, config, training, data_binding):
    require(config.data_factory == FACTORY and config.manifest_path == data_binding['subset']['path']
            and config.manifest_sha256 == data_binding['subset']['sha256']
            and config.num_frames == 97 and config.num_workers == 0
            and config.precomputed_latents and config.precomputed_text_embeddings, 'exact expanded factory required')
    dataset = build_dataset(data_binding)
    sampling_contract(training, data_binding, dataset=dataset)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                      pin_memory=torch.cuda.is_available(), drop_last=True)


def artifact_hashes(config, config_path, data_binding):
    validate_config(config, output_dir=config.training.output_dir, data_binding=data_binding)
    p = parent_paths()
    return dict(training_config=sha256_file(config_path),
                qualified_data_binding=copy.deepcopy(data_binding),
                entry_source=sha256_file(__file__), exact_dataset_source=sha256_file(data_module().__file__),
                parent_config=sha256_file(p['config']),
                **{'parent_' + n: sha256_file(p['endpoint'] / (n + '.json'))
                   for n in ('request', 'result', 'settlement')})


def precision_contract():
    return balanced.precision_contract('frozen')


def method_contract():
    return dict(method=STAGE, intervention='expanded_reviewed_exact_window_coverage_and_new80_steps',
                ratio_cap=.1, adapter_lr=1e-5, gate_lr=.01, lora_lr=0.,
                adapter_weight_decay=.01, gate_weight_decay=0., betas=[.9, .95], global_clip_norm=1.,
                lora_trainable=False, lora_strength=1., seed=42, max_new_steps=80,
                new_optimizer=True, parent_strict_resume=False, precision=precision_contract(),
                loss='unchanged_uniform_future_flow_MSE', negative_action_loss=False,
                reference_enabled=False, causal=False, automatic_downstream_training=False,
                data_is_unique_failure_cause_claimed=False, evaluation_endpoint=80,
                checkpoint_policy='fixed2_20_40_60_80_no_pruning_no_intermediate_selection')


def model_signature(model):
    return balanced.model_signature(model, 'frozen')


def _build_model(config, device):
    validate_probe_config(config)
    model, summary = parent.fresh.controlled._build_model(config, device)
    balanced.validate_signature(model_signature(model), 'frozen')
    return model, summary


def counters(step):
    require(type(step) is int and 0 <= step <= 80, 'invalid expanded optimizer step')
    return dict(step=step, optimizer_internal_step=step, micro_batches_consumed=step*8,
                new_stage_micro_batches_consumed=step*8, historical_micro_batches_before_stage=HISTORY,
                lifetime_micro_batches_consumed=HISTORY+step*8, sampling_cursor=step*8)


def validate_transition(*, output, stop_after_step, resume):
    require(type(stop_after_step) is int and stop_after_step in PREVIOUS, 'fixed2/20/40/60/80 endpoint required')
    start = PREVIOUS[stop_after_step]
    require((resume is None and start == 0) or (resume is not None and start > 0
            and Path(resume) == Path(output) / 'checkpoints' / f'step-{start:07d}.pt'),
            'strict immediate predecessor required; no skip/restart/warmstart')
    return start


def _checkpoint(model, masters, optimizer, *, config, hashes, lineage, sampling, step, metrics):
    state, signature = optimizer.state_dict(), model_signature(model)
    return dict(format_version=1, stage=STAGE, checkpoint_schema=CHECKPOINT_SCHEMA,
                config=config.to_dict(), manifest_hashes=hashes, base_identity=lineage['base_identity'],
                parent_lineage=lineage, method_contract=method_contract(), precision_contract=precision_contract(),
                sampling_contract=sampling, model_signature=signature, **counters(step),
                trainable_model=trainable_state_dict(model), master_parameters={n:p.detach().cpu() for n,p in masters.items()},
                master_mapping=balanced.optimizer_mapping(signature, state, arm='frozen', internal_step=step),
                optimizer=state, metrics=metrics, torch_rng_state=torch.get_rng_state(),
                cuda_rng_state_all=torch.cuda.get_rng_state_all(), python_rng_state=random.getstate(),
                numpy_rng_state=np.random.get_state())


def validate_checkpoint(payload, *, config, hashes, data_binding, sampling=None):
    validate_probe_config(config)
    lineage = payload.get('parent_lineage')
    validate_lineage(lineage)
    sampling = sampling_contract(config.training, data_binding) if sampling is None else sampling
    expected = dict(format_version=1, stage=STAGE, checkpoint_schema=CHECKPOINT_SCHEMA,
                    config=config.to_dict(), manifest_hashes=hashes, base_identity=lineage['base_identity'],
                    method_contract=method_contract(), precision_contract=precision_contract(), sampling_contract=sampling)
    require(all(payload.get(k) == v for k, v in expected.items()), 'strict expanded checkpoint contract differs')
    step = payload.get('step')
    require(type(step) is int and step in ENDPOINTS
            and all(payload.get(k) == v for k, v in counters(step).items()), 'expanded history/cursor/optimizer differs')
    state, masters, signature = payload['trainable_model'], payload['master_parameters'], payload['model_signature']
    balanced.validate_signature(signature, 'frozen')
    require(set(state) == set(signature) and set(masters) == set(ADAPTER_SHAPES), 'full Adapter/gate and master export required')
    for name, value in state.items():
        dtype = torch.bfloat16 if name in ADAPTER_SHAPES else torch.float32
        require(torch.is_tensor(value) and value.dtype == dtype and list(value.shape) == signature[name]['shape']
                and bool(torch.isfinite(value).all()), 'export geometry/dtype/nonfinite error')
        if name in ADAPTER_SHAPES:
            require(masters[name].dtype == torch.float32 and masters[name].shape == value.shape
                    and torch.equal(masters[name], value.float()), 'round-master projection mismatch')
    require(payload['master_mapping'] == balanced.optimizer_mapping(signature, payload['optimizer'],
                arm='frozen', internal_step=step), 'native Adapter/gate optimizer mapping differs')
    return payload


def validate_resume(path, *, config, hashes, data_binding, stop_after_step, sampling=None):
    start = validate_transition(output=config.training.output_dir, stop_after_step=stop_after_step, resume=path)
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    validate_checkpoint(payload, config=config, hashes=hashes, data_binding=data_binding, sampling=sampling)
    validate_base_identity(payload, config)
    require(payload['step'] == start, 'actual checkpoint is not immediate predecessor')
    torch.Generator(device='cpu').set_state(payload['torch_rng_state'])
    random.Random().setstate(payload['python_rng_state'])
    np.random.RandomState().set_state(payload['numpy_rng_state'])
    rng = payload.get('cuda_rng_state_all')
    require(isinstance(rng, list) and len(rng) == 1 and torch.is_tensor(rng[0])
            and rng[0].dtype == torch.uint8 and rng[0].ndim == 1 and rng[0].numel() > 0,
            'complete single-GPU RNG required')
    return payload


def exposures(dataset, consumed):
    require(type(consumed) is int and 0 <= consumed <= MICRO_BATCHES, 'invalid sampling cursor')
    count = len(dataset.episodes)
    return [dict(**dict(zip(data_module().IDENTITY_FIELDS, data_module().window_key(row))),
                 target=row['target'], exposures=(consumed + count - 1 - slot)//count)
            for slot, row in enumerate(dataset.episodes)]


def forward_backward(model, dataset, *, step, config, device, profiler, check):
    """Eight exact, stage-absolute draws; CPU small-model checks use this same loop."""
    losses, samples = [], []
    for offset in range(8):
        check()
        index = (step-1)*8+offset
        profiler.set_phase('data_wait')
        batch = default_collate([dataset[index]])
        require(not any(k in batch for k in ('ref_latents', 'ref_mask', 'y', 'clip_fea')),
                'expanded stage is strictly no-reference/no-extra-conditioning')
        samples.append(dict(**dataset.sampling_spec(index), actual_timestep=float(batch['timesteps'][0, 1])))
        profiler.set_phase('forward')
        saved = torch.autograd.graph.save_on_cpu(pin_memory=True) if device.type == 'cuda' else nullcontext()
        cast = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
        with saved, cast:
            loss = parent.fresh.controlled.teacher._forward_loss(model, batch, config, device)
        require(bool(torch.isfinite(loss)), 'nonfinite training loss')
        profiler.set_phase('backward')
        (loss/8).backward()
        losses.append(float(loss.detach()))
        samples[-1]['loss'] = losses[-1]
    return sum(losses)/8, samples


def launch(config, config_path, *, lineage, data_binding, stop_after_step, resume=None,
           shared_lease, shared_max_seconds, shared_request_sha256):
    from training.demo.backend import validate_active_authority
    validate_config(config, output_dir=config.training.output_dir, data_binding=data_binding)
    validate_lineage(lineage)
    start = validate_transition(output=config.training.output_dir, stop_after_step=stop_after_step, resume=resume)
    require(type(shared_max_seconds) is int and 60 <= shared_max_seconds <= (900 if start == 0 else 3600),
            'finite900/3600 training deadline required')
    output = Path(config.training.output_dir)
    require(resume is not None or not output.exists() or not any(output.iterdir()), 'fresh stage output must be empty')
    require(parent_binding() == lineage, 'actual frozen24 parent changed')
    hashes = artifact_hashes(config, config_path, data_binding)
    loader = build_expanded_dataloader(config=config.data, training=config.training, data_binding=data_binding)
    sampling = sampling_contract(config.training, data_binding, dataset=loader.dataset)
    saved = (validate_resume(resume, config=config, hashes=hashes, data_binding=data_binding,
                             stop_after_step=stop_after_step, sampling=sampling) if resume else load_parent(lineage))
    if resume:
        require(saved['parent_lineage'] == lineage, 'strict resume parent changed')
    gpu = parent.fresh.controlled.teacher._shared_training_gpu(shared_lease, max_seconds=shared_max_seconds,
                                                              request_sha256=shared_request_sha256)
    require(shared_lease.get('gpu_index') in (1, 2) and torch.cuda.is_available(), 'only leased Creator GPU1/2')
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(42)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(stage=STAGE, config=config.to_dict(), manifest_hashes=hashes, parent_lineage=lineage,
                    sampling_contract=sampling, method_contract=method_contract(), resume=resume, resume_step=start,
                    stop_after_step=stop_after_step, git_revision=git_revision(Path(__file__).parent),
                    entry_sha256=sha256_file(__file__), gpu_state=gpu.as_dict(), started_unix=time.time(),
                    shared_guard=dict(lease=shared_lease, max_seconds=shared_max_seconds, request_sha256=shared_request_sha256))
    if not resume:
        with (output/'run_metadata.json').open('x', encoding='utf-8') as stream:
            json.dump(metadata, stream, indent=2)
    append_jsonl(output/'executions.jsonl', metadata)
    profiler = OptimizerStepProfiler(device, output_dir=output/'profile_output', gpu_uuid=shared_lease['gpu_uuid'])
    began, last_check = time.monotonic(), 0.
    def check():
        nonlocal last_check
        now = time.monotonic()
        require(now-began < shared_max_seconds-30, 'finite training deadline reached')
        if now-last_check >= 2:
            validate_active_authority(shared_lease)
            last_check = now
    try:
        with profiler.phase('model_load'):
            model, summary = _build_model(config, device)
            signature = model_signature(model)
            require(signature == saved['model_signature'], 'actual parent/resume Adapter/gate signature differs')
            if resume:
                load_trainable_state_dict(model, saved['trainable_model'])
            else:
                master.b.pair.install_parent_weights(model, saved['trainable_model'])
            masters = master.make_masters(model, saved['master_parameters'] if resume else None)
            optimizer = balanced.make_optimizer(model, masters, 'frozen', saved['optimizer'] if resume else None, check=check)
        if resume:
            parent.fresh.controlled.teacher._restore_rng_state(saved, require_complete=True)
        else:
            seed_everything(42)
        del saved
        initial = {n:p for n,p in master.b.pair._sample_parameters(model).items() if n in ADAPTER_SHAPES}
        model.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        for step in range(start+1, stop_after_step+1):
            check()
            before, before_model = master.sampled_masters(masters), master.b.pair._sample_parameters(model)
            profiler.begin_step(step)
            loss, samples = forward_backward(model, loader.dataset, step=step, config=config,
                                             device=device, profiler=profiler, check=check)
            task = balanced.task_gradient_evidence(model, 'frozen', step)
            precision = master.b.pair._gradient_precision(model, 'frozen')
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            require(bool(torch.isfinite(norm)), 'nonfinite global gradient norm')
            master.copy_clipped_gradients(model, masters)
            profiler.set_phase('optimizer_update')
            optimizer.step()
            updated = master.sampled_masters(masters)
            changed = sum(int(torch.count_nonzero(updated[n]-before[n])) for n in before)
            require(changed > 0, 'no FP32 Adapter master update before round projection')
            master.writeback(model, masters, 'round')
            optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            storage = master.storage_evidence(model, masters, 'round')
            updates = balanced.update_evidence(model, before_model, 'frozen')
            endpoint_change, restored = None, bool(resume)
            if step in ENDPOINTS:
                now = master.b.pair._sample_parameters(model)
                endpoint_change = sum(int(torch.count_nonzero(now[n].float()-v.float())) for n,v in initial.items())
                require(endpoint_change > 0, 'endpoint requires cumulative actual BF16 Adapter change')
                restored = balanced.native_roundtrip(optimizer, signature, 'frozen', step, check)
            timing = profiler.end_step(step)
            memory = timing['memory']
            latest = dict(loss=loss, grad_norm=float(norm), optimizer_step_seconds=timing['seconds'],
                          task_gradient_evidence=task, precision_summary=precision, storage_evidence=storage,
                          update_evidence=dict(master_updated=True, master_changed_sampled_elements=changed,
                                               model_updates=updates, parameter_change_is_not_quality=True),
                          endpoint_adapter_update=dict(from_optimizer_step=start, to_optimizer_step=step,
                                                       changed_sampled_elements=endpoint_change),
                          native_state_restore_matched=restored, micro_samples=samples,
                          sample_namespace=loader.dataset.contract['namespace'],
                          exposures_since_stage_start=exposures(loader.dataset, step*8),
                          gate_statistics=master.b.pair._gate_statistics(model), step_profile=timing,
                          peak_vram_bytes=memory['cumulative_peak_allocated_bytes'],
                          peak_reserved_vram_bytes=memory['cumulative_peak_reserved_bytes'],
                          elapsed_seconds=time.monotonic()-began)
            append_jsonl(output/'metrics.jsonl', dict(**counters(step), **latest))
            print(json.dumps(dict(**counters(step), **latest)), flush=True)
            if step in ENDPOINTS:
                profiler.begin_checkpoint(step)
                target = master.b.pair._save_checkpoint(output, _checkpoint(model, masters, optimizer,
                    config=config, hashes=hashes, lineage=lineage, sampling=sampling, step=step, metrics=latest))
                profiler.end_checkpoint(step, checkpoint_path=str(target))
        result = dict(status='completed', stage=STAGE, checkpoint_schema=CHECKPOINT_SCHEMA, **counters(stop_after_step),
                      parent_lineage=lineage, base_identity=lineage['base_identity'], model_signature=signature,
                      sampling_contract=sampling, method_contract=method_contract(), precision_contract=precision_contract(),
                      resume_step=start, checkpoint_path=str(target), configured_max_steps=80,
                      trainable_parameters=summary.trainable_parameters, **latest,
                      qualification=dict(master_updated=True, adapter_updated=True, gate_updated=True, gradients_finite=True,
                          adapter_task_gradient_nonzero=task['adapter_task_gradient_nonzero'],
                          gate_task_gradient_nonzero=task['gate_task_gradient_nonzero'],
                          native_state_restore_matched=restored, model_master_projection_equal=True,
                          lora_trainable=False, frozen_lora_has_gradients=task['frozen_lora_has_gradients'],
                          peak_reserved_bytes=latest['peak_reserved_vram_bytes']),
                      reference_enabled=False, quality_improvement_verified=False, automatic_next_stage=False)
        append_jsonl(output/'execution_segments.jsonl', result)
        (output/'execution_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
    except BaseException as error:
        profiler.close(error=error)
        append_jsonl(output/'failures.jsonl', dict(status='failed', stage=STAGE, failure_type=type(error).__name__,
                                                error=str(error), resume_step=start, timestamp_unix=time.time()))
        raise
    finally:
        profiler.close()
