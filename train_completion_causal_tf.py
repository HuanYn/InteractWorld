"""Finite causal TF from expanded80, with its action branch kept frozen.

This is an experimental migration, not a quality-certified teacher or few-step
student. Only newly initialized FP32 LoRA is optimized. No R005/reference/DMD.
"""
from __future__ import annotations

from contextlib import nullcontext
import copy
from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

import train_action_teacher as teacher
import train_expanded_reference_teacher as common
import train_master_writeback as master
from training.causal_tf import (CausalModelConfig, CausalDataConfig, CausalOptimizerConfig,
    CausalTrainingConfig, causal_teacher_forcing_loss, construct_causal_wrapper)
from training.data.action_dataset import _seed64
from training.data.exact_cached_dataset import ExactCachedDataset
from training.error_recycling import ErrorRecyclingConfig
from training.models.action_adapter import CANONICAL_ACTION_KEYS
from training.models.lora import (TrainableSummary, inject_lora,
    load_trainable_state_dict, trainable_state_dict)
from training.runtime import OptimizerStepProfiler, append_jsonl, git_revision, seed_everything, sha256_file
from utils.controlled_action import install_controlled_action

STAGE = 'completion_causal_tf60_v1'
CHECKPOINT_SCHEMA = 'completion_causal_tf60_checkpoint_v1'
ROOT = Path('/DATA_DISK/InteractWorld/creator-runtime/abot-completion-20261007/causal-tf-v1')
PROFILE_JOB = 'ccf-profile2-v1'
FACTORY = 'train_completion_causal_tf:build_pair_dataloader'
NAMESPACE = 'completion_causal_tf60_seed42_v1'
PREVIOUS = {2: 0, 8: 2, 14: 8, 20: 14}
ENDPOINTS = tuple(PREVIOUS)
MICRO_BATCHES, HISTORY = 160, 1312
GATE, ADAPTER_SHAPES = master.GATE, master.ADAPTER_SHAPES
require = master.require
parent_paths, parent_binding = common.parent_paths, common.parent_binding
load_parent, validate_lineage = common.load_parent, common.validate_lineage
validate_data_binding, validate_base_identity = common.validate_data_binding, common.validate_base_identity
exposures = common.exposures


@dataclass
class CompletionTrainingConfig(CausalTrainingConfig):
    lora_parameter_dtype: str = 'float32'


@dataclass
class CompletionConfig:
    """Multi-cache exact-window config; does not relabel legacy R005 config."""
    model: CausalModelConfig = field(default_factory=CausalModelConfig)
    data: CausalDataConfig = field(default_factory=CausalDataConfig)
    optimizer: CausalOptimizerConfig = field(default_factory=CausalOptimizerConfig)
    training: CompletionTrainingConfig = field(default_factory=CompletionTrainingConfig)
    error_recycling: ErrorRecyclingConfig = field(default_factory=ErrorRecyclingConfig)

    def validate(self):
        validate_probe_config(self)

    def to_dict(self):
        return asdict(self)


def validate_probe_config(config):
    from training.studio.controlled_action_train import MODEL
    require(type(config) is CompletionConfig, 'independent completion configuration required')
    m, d, o, t = config.model, config.data, config.optimizer, config.training
    require(Path(m.base_model_path) == Path(MODEL) and m.model_type == 'ci2v'
        and m.timestep_shift == 5 and m.lora_rank == 16 and m.lora_alpha == 16
        and m.lora_dropout == 0 and m.action_dim == 32 and m.action_scale == 1
        and m.gradient_checkpointing and m.latent_channels == 48 and m.temporal_compression == 4
        and m.spatial_compression == m.downscale_factor_control_adapter == 16
        and m.num_frame_per_block == 3 and m.independent_first_frame
        and m.local_attn_size == -1 and m.teacher_aug_t == 0, 'fixed causal/base/action geometry required')
    require(d.data_factory == FACTORY and (d.num_frames, d.height, d.width) == (97, 480, 832)
        and d.num_workers == 0 and d.precomputed_latents and d.precomputed_text_embeddings
        and d.prompt_cache_path is None and tuple(d.canonical_action_keys) == CANONICAL_ACTION_KEYS
        and d.rgb_frames_per_action_token == 4 and d.feature_index_path == d.feature_receipt_path == '',
        'exact60 multi-cache binding required, not legacy single-cache/reference data')
    require(o.name == 'adamw8bit' and o.adapter_lr == 0 and o.lora_lr == 2e-6
        and tuple(o.betas) == (.9, .95) and o.weight_decay == .01 and o.max_grad_norm == 1,
        'only fresh FP32 LoRA optimizer is enabled')
    require(t.precision == 'bf16' and t.lora_parameter_dtype == 'float32'
        and t.micro_batch_size == 1 and t.gradient_accumulation_steps == 8 and t.seed == 42
        and t.max_steps == t.checkpoint_every == 20 and t.keep_last == 4 and t.keep_best == 0
        and t.log_every == 1 and Path(t.output_dir) == ROOT / PROFILE_JOB / 'training',
        'fixed20 trajectory, all four endpoints and independent output required')
    config.error_recycling.validate()
    require(not config.error_recycling.enabled and config.error_recycling.frames_per_block == 3,
        'this stage uses clean completed-history TF, not self-rollout/error recycling')


def make_config(data_binding, lineage):
    validate_lineage(lineage)
    validate_data_binding(data_binding)
    require(lineage['qualified_data_binding'] == data_binding, 'expanded80 exact60 binding must be preserved')
    old = common.expanded.load_config(parent_paths()['config'])
    model = CausalModelConfig(**{f.name:getattr(old.model, f.name) for f in fields(CausalModelConfig)
                               if hasattr(old.model, f.name)})
    config = CompletionConfig(model=model, data=CausalDataConfig(
        manifest_path=data_binding['subset']['path'], manifest_sha256=data_binding['subset']['sha256'],
        feature_index_path='', feature_receipt_path='', data_factory=FACTORY, num_frames=97, num_workers=0),
        optimizer=CausalOptimizerConfig(adapter_lr=0., lora_lr=2e-6),
        training=CompletionTrainingConfig(max_steps=20, checkpoint_every=20, keep_last=4, keep_best=0,
            output_dir=str(ROOT / PROFILE_JOB / 'training')))
    config.validate()
    return config


def load_config(path):
    import yaml
    source = Path(path).read_text(encoding='utf-8')
    try:
        raw = json.loads(source)
    except json.JSONDecodeError:
        raw = yaml.safe_load(source)
    require(isinstance(raw, dict) and set(raw) == {'model','data','optimizer','training','error_recycling'},
            'completion config fields differ')
    raw['data']['canonical_action_keys'] = tuple(raw['data']['canonical_action_keys'])
    raw['optimizer']['betas'] = tuple(raw['optimizer']['betas'])
    config = CompletionConfig(model=CausalModelConfig(**raw['model']), data=CausalDataConfig(**raw['data']),
        optimizer=CausalOptimizerConfig(**raw['optimizer']), training=CompletionTrainingConfig(**raw['training']),
        error_recycling=ErrorRecyclingConfig(**raw['error_recycling']))
    config.validate()
    return config


def validate_config(config, *, output_dir, data_binding):
    config.validate()
    validate_data_binding(data_binding)
    require(Path(output_dir) == Path(config.training.output_dir)
        and config.data.manifest_path == data_binding['subset']['path']
        and config.data.manifest_sha256 == data_binding['subset']['sha256'], 'exact subset/output binding differs')


class CompletionExactDataset(ExactCachedDataset):
    def __init__(self, selection, *, cache_bindings):
        super().__init__(selection, cache_bindings=cache_bindings, num_samples=MICRO_BATCHES, seed=42)
        self.data_seed = _seed64(42, NAMESPACE)
        self.contract.update(namespace=NAMESPACE, data_seed=self.data_seed, virtual_samples=MICRO_BATCHES)

    def sampling_spec(self, index):
        row = super().sampling_spec(index)
        return dict(row, index_namespace=NAMESPACE, new_stage_sample_index=row['absolute_sample_index'])


def build_dataset(binding):
    return CompletionExactDataset(validate_data_binding(binding), cache_bindings=binding['cache_bindings'])


def sampling_contract(training, data_binding, *, dataset=None):
    require(training.max_steps == 20 and training.seed == 42 and training.micro_batch_size == 1
        and training.gradient_accumulation_steps == 8, '20 steps/160 stage-absolute draws required')
    dataset = build_dataset(data_binding) if dataset is None else dataset
    return dict(namespace=NAMESPACE, dataset=copy.deepcopy(dataset.contract),
        qualified_data_binding=copy.deepcopy(data_binding), first_stage_sample_index=0,
        last_stage_sample_index=159, virtual_samples=160, historical_micro_batches_before_stage=HISTORY,
        reference_enabled=False, qualification_windows_used=0, sampling_restarts_at_each_segment=False)


def build_pair_dataloader(*, config, training, data_binding):
    require(config.data_factory == FACTORY and config.manifest_path == data_binding['subset']['path']
        and config.manifest_sha256 == data_binding['subset']['sha256'] and config.num_workers == 0,
        'new exact60 factory required')
    dataset = build_dataset(data_binding)
    sampling_contract(training, data_binding, dataset=dataset)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)


def method_contract():
    return dict(method=STAGE, objective='causal_completed_clean_blocks_flow_matching', causal=True,
        teacher='expanded80_experimental_not_quality_certified', reference_enabled=False, train_windows=60,
        action_adapter_frozen=True, fp32_gate_frozen=True, controlled_ratio_cap=.1,
        wan_backbone_frozen=True, fresh_fp32_rank16_lora=True, lora_lr=2e-6, action_scale=1.,
        latent_frames=25, independent_first_frame=True, future_latent_frames_per_block=3,
        completed_history='ground_truth_clean_latents_at_t0', inference_history='generated_latents_at_t0',
        exposure_gap_closed=False, packed_teacher_forcing=True, bidirectional_auxiliary=False,
        consistency_distillation=False, distribution_matching_distillation=False, new_optimizer=True,
        parent_strict_resume=False, r005_inherited=False, saved_tensors_on_cpu=True,
        checkpoint_policy='fixed2_8_14_20_no_pruning', quality_validated=False, realtime_validated=False)


def artifact_hashes(config, config_path, data_binding):
    from training import causal_tf
    from utils import controlled_causal_action
    validate_config(config, output_dir=config.training.output_dir, data_binding=data_binding)
    paths = parent_paths()
    return dict(training_config=sha256_file(config_path), qualified_data_binding=copy.deepcopy(data_binding),
        entry_source=sha256_file(__file__), causal_tf_source=sha256_file(causal_tf.__file__),
        controlled_causal_source=sha256_file(controlled_causal_action.__file__),
        causal_model_source=sha256_file(Path(__file__).parent/'wan/modules/causal_model.py'),
        shared_parent_source=sha256_file(common.__file__), exact_dataset_source=sha256_file(common.data_module().__file__),
        parent_config=sha256_file(paths['config']),
        **{'parent_'+name:sha256_file(paths['endpoint']/(name+'.json')) for name in ('request','result','settlement')})


def _lora(name):
    return name.endswith(('.lora_a.weight', '.lora_b.weight'))


def model_signature(model):
    result = {n:dict(shape=list(p.shape), dtype=str(p.dtype),
                    group='lora_a' if n.endswith('.lora_a.weight') else 'lora_b')
              for n,p in model.named_parameters() if p.requires_grad}
    require(bool(result) and all(_lora(n) for n in result), 'only LoRA may be trainable')
    validate_lora_signature(result)
    return result


def validate_lora_signature(signature):
    require(isinstance(signature, dict) and bool(signature), 'full LoRA signature required')
    modules = {}
    for n,s in signature.items():
        group = s.get('group'); suffix = '.'+str(group)+'.weight'; shape = s.get('shape', [])
        require(group in ('lora_a','lora_b') and n.endswith(suffix) and s.get('dtype') == 'torch.float32'
            and len(shape) == 2 and all(type(x) is int and x > 0 for x in shape)
            and shape[0 if group == 'lora_a' else 1] == 16, 'FP32 rank16 A/B geometry required')
        modules.setdefault(n[:-len(suffix)], set()).add(group)
    require(all(groups == {'lora_a','lora_b'} for groups in modules.values()), 'incomplete LoRA pair')


def frozen_action_signature():
    return {**{n:dict(shape=list(s), dtype='torch.bfloat16') for n,s in ADAPTER_SHAPES.items()},
            GATE:dict(shape=[3072], dtype='torch.float32')}


def validate_frozen_action(state):
    signature = frozen_action_signature()
    require(isinstance(state, dict) and set(state) == set(signature), 'complete frozen Adapter/gate required')
    for n,s in signature.items():
        p = state[n]
        require(torch.is_tensor(p) and list(p.shape) == s['shape'] and str(p.dtype) == s['dtype']
            and bool(torch.isfinite(p).all()), 'frozen Adapter/gate shape, dtype or values differ')


def build_model(config, device, *, frozen_action_model, trainable_model=None, wrapper_factory=None):
    """Reusable train/eval reconstruction; .to(device) preserves FP32 LoRA/gate.

Never call .to(dtype=BF16) afterwards: LoRA is deliberately created after the
base cast, so fresh A is initialized directly in FP32 (B is zero).
"""
    config.validate()
    validate_frozen_action(frozen_action_model)
    if wrapper_factory is None:
        from utils.wan_wrapper import WanDiffusionWrapper
        wrapper_factory = WanDiffusionWrapper
    model = construct_causal_wrapper(config, wrapper_factory)
    install_controlled_action(model.model, ratio_cap=.1, eps=1e-6)
    model.to(dtype=torch.bfloat16)
    model.requires_grad_(False)
    replaced = inject_lora(model.model, rank=16, alpha=16., dropout=0.)
    named = dict(model.named_parameters())
    with torch.no_grad():
        for n,value in frozen_action_model.items():
            require(n in named and named[n].shape == value.shape and named[n].dtype == value.dtype,
                'actual causal action parameters differ from expanded80')
            named[n].copy_(value)
    signature = model_signature(model)
    if trainable_model is not None:
        require(set(trainable_model) == set(signature), 'full causal LoRA state required')
        for n,p in trainable_model.items():
            require(p.dtype == torch.float32 and list(p.shape) == signature[n]['shape'], 'FP32 LoRA state differs')
        load_trainable_state_dict(model, trainable_model)
    model.to(device=device)
    if config.model.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    model.train()
    model_signature(model)
    require(all(not p.requires_grad and p.grad is None for n,p in model.named_parameters() if n in frozen_action_model),
        'frozen action/gate acquired gradients')
    return model, TrainableSummary(replaced, sum(p.numel() for p in model.parameters() if p.requires_grad),
                                  sum(p.numel() for p in model.parameters()))


def optimizer_mapping(signature, state, step):
    validate_lora_signature(signature)
    names = sorted(signature)
    groups = state.get('param_groups', [])
    require(len(groups) == 1, 'only one LoRA optimizer group allowed')
    group = groups[0]
    require(group.get('name') == 'lora' and group.get('lr') == 2e-6 and group.get('weight_decay') == .01
        and tuple(group.get('betas', ())) == (.9,.95) and group.get('params') == list(range(len(names))),
        'native LoRA optimizer order/hyperparameters differ')
    require(set(state.get('state', {})) == set(group['params']), 'complete native optimizer state required')
    for n,i in zip(names, group['params']):
        slot = state['state'][i]
        require(slot.get('step') == step and isinstance(slot.get('__bnb_optimizer_quant_state__'), dict),
            'native BNB state/internal step required')
        quant = slot['__bnb_optimizer_quant_state__']
        for key in ('state1','state2'):
            value = quant.get(key)
            require(torch.is_tensor(value) and list(value.shape) == signature[n]['shape'], 'native moment shape differs')
    return [dict(name=n, optimizer_id=i, dtype='torch.float32', shape=signature[n]['shape'])
            for i,n in enumerate(names)]


def make_optimizer(model, saved=None, *, check=lambda:None):
    import bitsandbytes as bnb
    signature = model_signature(model)
    named = dict(model.named_parameters())
    optimizer = bnb.optim.AdamW8bit([dict(params=[named[n] for n in sorted(signature)],
        name='lora', lr=2e-6, weight_decay=.01)], betas=(.9,.95))
    if saved is not None:
        optimizer.load_state_dict(saved)
        require(master.equal_tree(saved, optimizer.state_dict(), check), 'complete native optimizer restore differs')
    else:
        require(optimizer.state_dict()['state'] == {}, 'new optimizer must be empty')
    return optimizer


def counters(step):
    require(type(step) is int and 0 <= step <= 20, 'invalid causal optimizer step')
    return dict(step=step, optimizer_internal_step=step, micro_batches_consumed=step*8,
        new_stage_micro_batches_consumed=step*8, historical_micro_batches_before_stage=HISTORY,
        lifetime_micro_batches_consumed=HISTORY+step*8, sampling_cursor=step*8)


def validate_transition(*, output, stop_after_step, resume):
    require(type(stop_after_step) is int and stop_after_step in PREVIOUS, 'fixed2/8/14/20 endpoint required')
    start = PREVIOUS[stop_after_step]
    require((resume is None and start == 0) or (resume is not None and start > 0
        and Path(resume) == Path(output)/'checkpoints'/f'step-{start:07d}.pt'),
        'strict immediate predecessor required, not warmstart/restart/skip')
    return start


def _checkpoint(model, frozen, optimizer, *, config, hashes, lineage, sampling, step, metrics):
    signature, state = model_signature(model), optimizer.state_dict()
    return dict(format_version=1, stage=STAGE, checkpoint_schema=CHECKPOINT_SCHEMA, config=config.to_dict(),
        manifest_hashes=hashes, base_identity=lineage['base_identity'], parent_lineage=lineage,
        method_contract=method_contract(), sampling_contract=sampling, model_signature=signature,
        frozen_action_signature=frozen_action_signature(), frozen_action_model=frozen,
        trainable_model=trainable_state_dict(model), optimizer=state,
        optimizer_mapping=optimizer_mapping(signature, state, step), **counters(step), metrics=metrics,
        torch_rng_state=torch.get_rng_state(), cuda_rng_state_all=torch.cuda.get_rng_state_all(),
        python_rng_state=random.getstate(), numpy_rng_state=np.random.get_state())


def validate_checkpoint(payload, *, config, hashes, data_binding, sampling=None):
    config.validate()
    lineage = payload.get('parent_lineage')
    validate_lineage(lineage)
    require(lineage['qualified_data_binding'] == data_binding, 'same expanded80 exact60 data binding required')
    sampling = sampling_contract(config.training, data_binding) if sampling is None else sampling
    expected = dict(format_version=1, stage=STAGE, checkpoint_schema=CHECKPOINT_SCHEMA, config=config.to_dict(),
        manifest_hashes=hashes, base_identity=lineage['base_identity'], method_contract=method_contract(),
        sampling_contract=sampling, frozen_action_signature=frozen_action_signature())
    require(all(payload.get(k) == v for k,v in expected.items()), 'strict completion checkpoint contract differs')
    step = payload.get('step')
    require(type(step) is int and step in ENDPOINTS
        and all(payload.get(k) == v for k,v in counters(step).items()), 'optimizer/data cursor differs')
    validate_frozen_action(payload.get('frozen_action_model'))
    state, signature = payload['trainable_model'], payload['model_signature']
    validate_lora_signature(signature)
    require(set(state) == set(signature), 'complete FP32 LoRA export required')
    for n,value in state.items():
        require(torch.is_tensor(value) and value.dtype == torch.float32 and list(value.shape) == signature[n]['shape']
            and bool(torch.isfinite(value).all()), 'FP32 LoRA export invalid')
    require(payload['optimizer_mapping'] == optimizer_mapping(signature, payload['optimizer'], step),
        'complete native optimizer mapping differs')
    return payload


def validate_resume(path, *, config, hashes, data_binding, stop_after_step, sampling=None):
    start = validate_transition(output=config.training.output_dir, stop_after_step=stop_after_step, resume=path)
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    validate_checkpoint(payload, config=config, hashes=hashes, data_binding=data_binding, sampling=sampling)
    validate_base_identity(payload, config)
    require(payload['step'] == start, 'actual endpoint not the immediate predecessor')
    torch.Generator(device='cpu').set_state(payload['torch_rng_state'])
    random.Random().setstate(payload['python_rng_state'])
    np.random.RandomState().set_state(payload['numpy_rng_state'])
    rng = payload.get('cuda_rng_state_all')
    require(isinstance(rng, list) and len(rng) == 1 and torch.is_tensor(rng[0])
        and rng[0].dtype == torch.uint8 and rng[0].ndim == 1 and rng[0].numel() > 0, 'complete single-GPU RNG required')
    return payload


def forward_backward(model, dataset, *, step, config, device, profiler, check):
    losses, samples = [], []
    for offset in range(8):
        check()
        index = (step-1)*8+offset
        profiler.set_phase('data_wait')
        batch = default_collate([dataset[index]])
        require(not any(k in batch for k in ('y','clip_fea','ref_latents','ref_mask')), 'no reference/extra conditions permitted')
        sample = dict(**dataset.sampling_spec(index), actual_timestep=float(batch['timesteps'][0,1]))
        profiler.set_phase('forward_backward')
        saved = torch.autograd.graph.save_on_cpu(pin_memory=True) if device.type == 'cuda' else nullcontext()
        amp = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
        with saved, amp:
            loss = causal_teacher_forcing_loss(model, batch, config, device)
        require(bool(torch.isfinite(loss)), 'nonfinite causal TF loss')
        (loss/8).backward()
        losses.append(float(loss.detach()))
        samples.append(dict(**sample, loss=losses[-1], reference_enabled=False))
        del loss, batch
        check()
    return sum(losses)/8, samples


def sampled_lora(model):
    return {n:p.detach().reshape(-1)[::max(1,p.numel()//4096)].float().cpu().clone()
            for n,p in model.named_parameters() if p.requires_grad}


def gradient_evidence(model, step):
    precision = teacher._lora_gradient_precision(model)
    maxima = {'lora_a':0., 'lora_b':0.}
    for n,p in model.named_parameters():
        if not p.requires_grad:
            require(p.grad is None, 'frozen backbone/action/gate acquired a gradient')
            continue
        require(bool(torch.isfinite(p.grad).all()), 'nonfinite LoRA gradient')
        group = 'lora_a' if n.endswith('.lora_a.weight') else 'lora_b'
        maxima[group] = max(maxima[group], float(p.grad.detach().abs().max()))
    require(maxima['lora_b'] > 0 and (step != 2 or maxima['lora_a'] > 0), 'real B task gradients and step2 A gradient required')
    return dict(precision=precision, full_tensor_max_abs=maxima, gradients_finite=True,
        lora_A_task_gradient_nonzero=maxima['lora_a'] > 0, lora_B_task_gradient_nonzero=True,
        frozen_action_has_gradients=False, frozen_backbone_has_gradients=False, quality_evidence=False)


def launch(config, config_path, *, lineage, data_binding, stop_after_step, resume=None,
           shared_lease, shared_max_seconds, shared_request_sha256):
    from training.demo.backend import validate_active_authority
    validate_config(config, output_dir=config.training.output_dir, data_binding=data_binding)
    validate_lineage(lineage)
    start = validate_transition(output=config.training.output_dir, stop_after_step=stop_after_step, resume=resume)
    require(type(shared_max_seconds) is int and 60 <= shared_max_seconds <= (1800 if start == 0 else 3600),
        'finite1800/3600 job limit required')
    output = Path(config.training.output_dir)
    require(resume is not None or not output.exists() or not any(output.iterdir()), 'fresh output must be empty')
    require(parent_binding() == lineage and lineage['qualified_data_binding'] == data_binding, 'actual expanded80 parent/data changed')
    hashes = artifact_hashes(config, config_path, data_binding)
    dataset = build_dataset(data_binding)
    sampling = sampling_contract(config.training, data_binding, dataset=dataset)
    saved = (validate_resume(resume, config=config, hashes=hashes, data_binding=data_binding,
        stop_after_step=stop_after_step, sampling=sampling) if resume else load_parent(lineage))
    frozen = saved['frozen_action_model'] if resume else saved['trainable_model']
    if resume:
        require(saved['parent_lineage'] == lineage, 'strict resume parent differs')
    gpu = teacher._shared_training_gpu(shared_lease, max_seconds=shared_max_seconds, request_sha256=shared_request_sha256)
    require(shared_lease.get('gpu_index') in (1,2) and torch.cuda.is_available(), 'leased Creator GPU1/2 only')
    device = torch.device('cuda',0)
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
        with (output/'run_metadata.json').open('x',encoding='utf-8') as stream:
            json.dump(metadata,stream,indent=2)
    append_jsonl(output/'executions.jsonl',metadata)
    profiler = OptimizerStepProfiler(device, output_dir=output/'profile_output', gpu_uuid=shared_lease['gpu_uuid'])
    began, last_check = time.monotonic(), 0.
    def check():
        nonlocal last_check
        now = time.monotonic()
        require(now-began < shared_max_seconds-30, 'finite causal training deadline reached')
        if now-last_check >= 2:
            validate_active_authority(shared_lease)
            last_check = now
    try:
        with profiler.phase('model_load'):
            model,summary = build_model(config,device,frozen_action_model=frozen,
                trainable_model=saved['trainable_model'] if resume else None)
            signature = model_signature(model)
            if resume:
                require(signature == saved['model_signature'], 'actual resumed LoRA signature differs')
            optimizer = make_optimizer(model, saved['optimizer'] if resume else None, check=check)
        if resume:
            teacher._restore_rng_state(saved,require_complete=True)
        else:
            seed_everything(42)
        del saved
        frozen_versions = {n:p._version for n,p in model.named_parameters() if not p.requires_grad}
        model.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        for step in range(start+1,stop_after_step+1):
            check()
            before = sampled_lora(model)
            profiler.begin_step(step)
            loss,samples = forward_backward(model,dataset,step=step,config=config,device=device,profiler=profiler,check=check)
            task = gradient_evidence(model,step)
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            require(bool(torch.isfinite(norm)), 'nonfinite LoRA gradient norm')
            profiler.set_phase('optimizer_update')
            optimizer.step()
            after = sampled_lora(model)
            require(all(bool(torch.isfinite(p).all()) for p in after.values()), 'nonfinite updated LoRA sample')
            changed = sum(int(torch.count_nonzero(after[n]-before[n])) for n in before)
            require(changed > 0, 'no representable LoRA update')
            require(all(not p.requires_grad and p.grad is None and p._version == frozen_versions[n]
                for n,p in model.named_parameters() if n in frozen_versions), 'frozen backbone/action/gate modified')
            optimizer.zero_grad(set_to_none=True)
            restored = bool(resume)
            if step in ENDPOINTS:
                native = optimizer.state_dict()
                optimizer_mapping(signature,native,step)
                optimizer.load_state_dict(native)
                require(master.equal_tree(native,optimizer.state_dict(),check), 'native full-state roundtrip differs')
                restored = True
            timing = profiler.end_step(step)
            memory = timing['memory']
            require(memory['cumulative_peak_reserved_bytes'] < 23*1024**3, 'causal profile reserved memory exceeds23GiB')
            latest = dict(loss=loss, loss_scope='causal_future_flow_MSE_clean_history', grad_norm=float(norm),
                optimizer_step_seconds=timing['seconds'], task_gradient_evidence=task,
                lora_changed_sampled_elements=changed, adapter_unchanged=True, gate_unchanged=True,
                native_state_restore_matched=restored, micro_samples=samples, sample_namespace=NAMESPACE,
                exposures_since_stage_start=exposures(dataset,step*8), step_profile=timing,
                peak_vram_bytes=memory['cumulative_peak_allocated_bytes'],
                peak_reserved_vram_bytes=memory['cumulative_peak_reserved_bytes'], elapsed_seconds=time.monotonic()-began)
            append_jsonl(output/'metrics.jsonl',dict(**counters(step),**latest))
            print(json.dumps(dict(**counters(step),**latest)),flush=True)
            if step in ENDPOINTS:
                profiler.begin_checkpoint(step)
                target = master.b.pair._save_checkpoint(output,_checkpoint(model,frozen,optimizer,
                    config=config,hashes=hashes,lineage=lineage,sampling=sampling,step=step,metrics=latest))
                profiler.end_checkpoint(step,checkpoint_path=str(target))
        result = dict(status='completed',stage=STAGE,checkpoint_schema=CHECKPOINT_SCHEMA,**counters(stop_after_step),
            parent_lineage=lineage,base_identity=lineage['base_identity'],model_signature=signature,
            frozen_action_signature=frozen_action_signature(),sampling_contract=sampling,method_contract=method_contract(),
            resume_step=start,checkpoint_path=str(target),configured_max_steps=20,trainable_parameters=summary.trainable_parameters,**latest,
            qualification=dict(lora_updated=True,gradients_finite=True,lora_B_task_gradient_nonzero=True,
                lora_A_task_gradient_nonzero=task['lora_A_task_gradient_nonzero'],adapter_unchanged=True,
                gate_unchanged=True,frozen_action_has_gradients=False,native_state_restore_matched=restored,
                peak_reserved_bytes=latest['peak_reserved_vram_bytes']),
            reference_enabled=False,quality_improvement_verified=False,automatic_next_stage=False)
        append_jsonl(output/'execution_segments.jsonl',result)
        (output/'execution_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result
    except BaseException as error:
        profiler.close(error=error)
        append_jsonl(output/'failures.jsonl',dict(status='failed',stage=STAGE,failure_type=type(error).__name__,
            error=str(error),resume_step=start,timestamp_unix=time.time()))
        raise
    finally:
        profiler.close()
