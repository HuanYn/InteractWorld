"""Fixed paired reference/no-reference20 pilot from frozen balanced24 weights.

New schema/optimizer/sampling namespace; old checkpoints and protocols unchanged.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import math
from operator import index as integer_index
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

import train_balanced_fresh_teacher as fresh
import train_master_writeback as master
from training.config import ActionTeacherConfig, load_config as _load_config
from training.data.action_dataset import _seed64
from training.models.lora import LoRALinear, load_trainable_state_dict, trainable_state_dict
from training.runtime import OptimizerStepProfiler, append_jsonl, git_revision, seed_everything, sha256_file
import train_balanced_lora_pair as balanced
from training.studio.teacher_reference_train import profile_job

STAGE = 'controlled_action_reference_pilot_native97_v1'
CHECKPOINT_SCHEMA = 'controlled_action_reference_pilot_checkpoint_v1'
ROOT = fresh.ROOT.parent/'teacher-reference-pilot-v1'
DATA = fresh.DATA
FACTORY = 'train_reference_teacher:build_pair_dataloader'
NAMESPACE = 'controlled_reference_pilot_seed42_v1'
ARMS = ('no_reference','reference')
ENDPOINTS, MICRO_BATCHES, HISTORY = (2,20), 160, 672
GATE, ADAPTER_SHAPES = master.GATE, master.ADAPTER_SHAPES
require = master.require
validate_base_identity = fresh.validate_base_identity
validate_pool, pool_contract = fresh.validate_pool, fresh.pool_contract


def arm_for_config(config, arm=None):
    name = Path(config.training.output_dir).parent.name
    matches = [a for a in ARMS if name == profile_job(a)]
    require(len(matches) == 1 and (arm is None or arm == matches[0]), 'fixed independent reference arm output required')
    return matches[0]


def model_config(config):
    return copy.deepcopy(config)


def validate_probe_config(config, arm=None):
    arm = arm_for_config(config,arm)
    require(type(config) is ActionTeacherConfig, 'typed pair configuration required'); config.validate()
    require(config.data.data_factory == FACTORY and config.data.manifest_path == str(DATA/'manifests/train.jsonl')
        and isinstance(config.data.manifest_sha256,str) and len(config.data.manifest_sha256) == 64,
        'actual same six-window manifest with independent pair sampler required')
    require(config.data.num_frames == 97 and config.data.num_workers == 0 and config.data.prompt_cache_path is None
        and not config.data.low_sigma_stratification and config.data.precomputed_latents
        and config.data.precomputed_text_embeddings, 'same native97/static text cache required')
    require(config.model.action_scale == 1 and config.model.lora_alpha == 16 and config.model.lora_dropout == 0
        and config.model.timestep_shift == 5 and config.model.gradient_checkpointing, 'fixed model hyperparameters required')
    require(config.optimizer.name == 'adamw8bit' and config.optimizer.adapter_lr == 1e-5
        and config.optimizer.lora_lr == 0.
        and tuple(config.optimizer.betas) == (.9,.95) and config.optimizer.weight_decay == .01
        and config.optimizer.max_grad_norm == 1, 'fixed frozen-LoRA optimizer required')
    require(config.training.max_steps == 20 and config.training.checkpoint_every == 20
        and config.training.log_every == 1 and config.training.lora_parameter_dtype == 'float32'
        and Path(config.training.output_dir) == ROOT/profile_job(arm)/'training', 'fixed new20 trajectory required')


def load_config(path):
    config = _load_config(path); validate_probe_config(config); return config


def validate_config(config, *, output_dir, arm):
    from training.studio.controlled_action_train import MODEL
    validate_probe_config(config,arm)
    require(config.model.base_model_path == str(MODEL) and Path(output_dir) == Path(config.training.output_dir),
        'same pinned official model/new arm output required')
    require(config.data.manifest_sha256 == pool_contract()['binding']['binding']['manifest_sha256'], 'actual train cache differs')


def parent_paths():
    from training.studio.balanced_lora_pair_train import paths
    return paths('frozen')


def parent_binding():
    from training.studio import balanced_lora_pair_train as source
    request, ex, _ = source.completed_endpoint('frozen',24)
    p = parent_paths(); stat = p['checkpoint'].stat()
    require(ex['lifetime_micro_batches_consumed'] == HISTORY, 'real frozen LoRA24 parent required')
    return dict(mode='frozen_balanced24_weights_only_new_optimizer_new_namespace',
        parent_stage=balanced.STAGE,parent_checkpoint_schema=balanced.CHECKPOINT_SCHEMA,
        parent_arm='frozen',parent_step=24,parent_optimizer_internal_step=24,
        parent_checkpoint=str(p['checkpoint']),parent_checkpoint_stat=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns),
        parent_config=str(p['config']),parent_config_sha256=request['config_sha256'],parent_job=str(p['endpoint']),
        parent_records={n:sha256_file(p['endpoint']/(n+'.json')) for n in ('request','result','settlement')},
        base_identity=ex['base_identity'],ancestor_lineage=ex['parent_lineage'],
        historical_micro_batches_before_stage=HISTORY,optimizer_restored=False,rng_restored=False,
        new_optimizer=True,parent_strict_resume=False,r005_inherited=False,parent_teacher_qualified=False)


def validate_lineage(lineage):
    p = parent_paths()
    expected = dict(mode='frozen_balanced24_weights_only_new_optimizer_new_namespace',
        parent_stage=balanced.STAGE,parent_checkpoint_schema=balanced.CHECKPOINT_SCHEMA,
        parent_arm='frozen',parent_step=24,parent_optimizer_internal_step=24,
        parent_checkpoint=str(p['checkpoint']),parent_config=str(p['config']),parent_job=str(p['endpoint']),
        historical_micro_batches_before_stage=HISTORY,optimizer_restored=False,rng_restored=False,
        new_optimizer=True,parent_strict_resume=False,r005_inherited=False,parent_teacher_qualified=False)
    require(isinstance(lineage,dict) and all(lineage.get(k) == v for k,v in expected.items()),
        'genuine frozen balanced24 weights-only lineage required')
    require(set(lineage.get('parent_records',{})) == {'request','result','settlement'}
        and set(lineage.get('parent_checkpoint_stat',{})) == {'bytes','mtime_ns'}
        and isinstance(lineage.get('base_identity'),dict) and isinstance(lineage.get('ancestor_lineage'),dict),
        'small actual parent binding/base missing')


def load_parent(lineage):
    require(parent_binding() == lineage, 'settled frozen LoRA24 parent changed')
    p = parent_paths(); config = balanced.load_config(p['config'])
    payload = torch.load(p['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    balanced.validate_checkpoint(payload,config=config,hashes=balanced.artifact_hashes(config,p['config']),arm='frozen')
    balanced.validate_base_identity(payload,config)
    require(payload['step'] == 24 and payload['lifetime_micro_batches_consumed'] == HISTORY
        and payload['base_identity'] == lineage['base_identity']
        and payload['parent_lineage'] == lineage['ancestor_lineage'], 'actual frozen24 parent differs')
    stat = p['checkpoint'].stat()
    require(dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns) == lineage['parent_checkpoint_stat'],
        'parent changed while loading')
    return dict(trainable_model=payload['trainable_model'],model_signature=payload['model_signature'])


def sampling_contract(training, pool):
    validate_pool(pool)
    require(training.max_steps == 20 and training.seed == 42 and training.micro_batch_size == 1
        and training.gradient_accumulation_steps == 8, 'fixed20x8 same six-window schedule required')
    views = [r['view_id'] for r in fresh.ordered_windows(pool)]
    return dict(namespace=NAMESPACE,training_seed=42,data_seed=_seed64(42,NAMESPACE),virtual_samples=160,
        first_stage_sample_index=0,last_stage_sample_index=159,view_schedule=views,target_schedule=['J','L']*3,
        exposures_per_view={v:(160+5-i)//6 for i,v in enumerate(views)},historical_micro_batches_before_stage=HISTORY,
        diffusion_seed_policy='seed64(seed64(42,namespace),diffusion,new_stage_index)',
        complete_six_window_cycles=26,remaining_views=views[:4],qualification_windows_used=0,pool=pool)


def reference_binding(purpose='train'):
    from training.studio.teacher_reference_cache import validate_completed_cache
    result = validate_completed_cache(purpose)
    return result['indexes'][purpose]


def reference_store(purpose='train'):
    from training.reference_dataset import ReferenceFeatureStore
    binding = reference_binding(purpose)
    return ReferenceFeatureStore(binding['path'],purpose=purpose,expected_index_sha256=binding['sha256'])


class PairDataset(fresh.BalancedDataset):
    """Identical six-window noise draws; only the reference arm supplies refs."""
    def __init__(self, *, pool, arm, store=None):
        require(arm in ARMS, 'explicit reference arm required')
        super().__init__(pool=pool)
        self.arm, self.store = arm, reference_store() if store is None else store
        self.num_samples, self.data_seed = MICRO_BATCHES, _seed64(42,NAMESPACE)

    def sampling_spec(self, index):
        require(not isinstance(index,bool), 'sample index cannot be bool'); index = integer_index(index)
        if not 0 <= index < MICRO_BATCHES: raise IndexError(index)
        slot = index % 6; row = self.episodes[slot]
        return dict(absolute_sample_index=index,new_stage_sample_index=index,index_namespace=NAMESPACE,
            episode_slot=slot,episode_id=row['episode_id'],window_index=0,source_start=row['source_start'],
            view_id=row['view_id'],source_target=row['target'],purpose='train',
            diffusion_seed=_seed64(self.data_seed,'diffusion',index))

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        spec = self.sampling_spec(index)
        refs = self.store.for_window(spec['episode_id'],spec['source_start'],batched=False)
        require(set(refs) == {'ref_latents','ref_mask'} and refs['ref_mask'].dtype == torch.bool
            and tuple(refs['ref_mask'].shape) == (1,) and tuple(refs['ref_latents'].shape) == (1,48,1,16,16)
            and refs['ref_latents'].dtype == torch.bfloat16, 'fixed one-slot BF16 reference required')
        if self.arm == 'reference': batch.update(refs)
        return batch


def build_pair_dataloader(*, config, training, pool=None, arm=None):
    pool = pool_contract() if pool is None else pool; sampling_contract(training,pool)
    require(config.manifest_path == pool['binding']['binding']['manifest'] and config.precomputed_latents
        and config.precomputed_text_embeddings and config.num_frames == 97 and config.num_workers == 0,
        'same exact read-only native97 train cache required')
    arm = arm_for_config(type('Configuration',(),{'training':training})(),arm)
    return DataLoader(PairDataset(pool=pool,arm=arm),batch_size=1,shuffle=False,num_workers=0,
        pin_memory=torch.cuda.is_available(),drop_last=True)


def artifact_hashes(config, config_path):
    validate_probe_config(config)
    binding = fresh.cache_module().validate_completed_cache('train'); info = binding['binding']; p = parent_paths()
    require(config.data.manifest_path == info['manifest'] and config.data.manifest_sha256 == info['manifest_sha256'], 'same train6 manifest required')
    refs = reference_binding()
    return dict(reference_index=refs['sha256'],reference_receipt=refs['receipt_sha256'],
        reference_manifest=refs['manifest_sha256'],dataset_manifest=info['manifest_sha256'],feature_index=info['sha256'],feature_receipt=info['receipt_sha256'],
        cache_result=binding['cache_result_sha256'],training_config=sha256_file(config_path),parent_config=sha256_file(p['config']),
        **{'parent_'+n:sha256_file(p['endpoint']/(n+'.json')) for n in ('request','result','settlement')})


def precision_contract(arm):
    require(arm in ARMS, 'fixed arm required')
    return balanced.precision_contract('frozen')


def method_contract(arm):
    require(arm in ARMS, 'fixed no_reference/reference arm required')
    return dict(method=STAGE,arm=arm,intervention='raw_initial_character_reference_conditioning_only',
        ratio_cap=.1,adapter_lr=1e-5,gate_lr=.01,lora_lr=0.,adapter_weight_decay=.01,gate_weight_decay=0.,
        betas=[.9,.95],global_clip_norm=1.,global_clip_scope='all_trainable_forward_parameters',
        lora_trainable=False,lora_strength=1.,seed=42,max_new_steps=20,new_optimizer=True,parent_strict_resume=False,
        precision=precision_contract(arm),loss='unchanged_uniform_future_flow_MSE',negative_action_loss=False,
        causal=False,reference_memory_implemented=True,reference_enabled=arm == 'reference',
        reference_source='original_raw_initial_frame_only',reference_shape=[1,48,1,16,16],
        invalid_reference='zero_slot_mask_false_never_valid_black_image',future_reference=False,
        automatic_downstream_training=False,checkpoint_policy='fixed2_20_no_pruning_no_intermediate_selection')


def model_signature(model, arm):
    require(arm in ARMS, 'fixed arm required')
    return balanced.model_signature(model,'frozen')


def _build_model(config, device, arm=None):
    arm = arm_for_config(config,arm); validate_probe_config(config,arm)
    model,summary = fresh.controlled._build_model(config,device)
    for module in model.modules():
        if type(module) is LoRALinear:
            module.lora_a.requires_grad_(False); module.lora_b.requires_grad_(False)
            require(module.lora_a.weight.dtype == module.lora_b.weight.dtype == torch.float32, 'LoRA must stay FP32')
    model_signature(model,arm)
    return model,replace(summary,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))


def lora_names(signature):
    return balanced.lora_names(signature)


def validate_signature(signature, arm):
    require(arm in ARMS, 'fixed arm required')
    return balanced.validate_signature(signature,'frozen')


def optimizer_mapping(signature, state, *, arm, internal_step):
    require(arm in ARMS, 'fixed arm required')
    return balanced.optimizer_mapping(signature,state,arm='frozen',internal_step=internal_step)


def make_optimizer(model, masters, arm, saved=None, *, check=lambda:None):
    require(arm in ARMS, 'fixed arm required')
    return balanced.make_optimizer(model,masters,'frozen',saved,check=check)


def native_roundtrip(optimizer, signature, arm, step, check=lambda:None):
    require(arm in ARMS, 'fixed arm required')
    return balanced.native_roundtrip(optimizer,signature,'frozen',step,check)


def task_gradient_evidence(model, arm, step):
    require(arm in ARMS, 'fixed arm required')
    return balanced.task_gradient_evidence(model,'frozen',step)


def update_evidence(model, before, arm):
    require(arm in ARMS, 'fixed arm required')
    return balanced.update_evidence(model,before,'frozen')


def _checkpoint(model, masters, optimizer, *, config, hashes, arm, lineage, pool, step, metrics):
    state = optimizer.state_dict(); signature = model_signature(model,arm)
    return dict(format_version=1,stage=STAGE,checkpoint_schema=CHECKPOINT_SCHEMA,arm=arm,
        config=config.to_dict(),manifest_hashes=hashes,base_identity=lineage['base_identity'],parent_lineage=lineage,
        method_contract=method_contract(arm),precision_contract=precision_contract(arm),
        sampling_contract=sampling_contract(config.training,pool),model_signature=signature,
        trainable_model=trainable_state_dict(model),master_parameters={n:p.detach().cpu() for n,p in masters.items()},
        master_mapping=optimizer_mapping(signature,state,arm=arm,internal_step=step),optimizer=state,
        step=step,optimizer_internal_step=step,micro_batches_consumed=step*8,new_stage_micro_batches_consumed=step*8,
        historical_micro_batches_before_stage=672,lifetime_micro_batches_consumed=672+step*8,sampling_cursor=step*8,
        metrics=metrics,torch_rng_state=torch.get_rng_state(),cuda_rng_state_all=torch.cuda.get_rng_state_all(),
        python_rng_state=random.getstate(),numpy_rng_state=np.random.get_state())


def validate_checkpoint(payload, *, config, hashes, arm=None):
    arm = arm_for_config(config,arm); validate_probe_config(config,arm)
    require(payload.get('format_version') == 1 and payload.get('stage') == STAGE
        and payload.get('checkpoint_schema') == CHECKPOINT_SCHEMA and payload.get('arm') == arm, 'new pair stage/schema/arm required')
    lineage = payload.get('parent_lineage'); validate_lineage(lineage)
    pool = payload.get('sampling_contract',{}).get('pool'); validate_pool(pool)
    for key,value in dict(config=config.to_dict(),manifest_hashes=hashes,base_identity=lineage['base_identity'],
        method_contract=method_contract(arm),precision_contract=precision_contract(arm),
        sampling_contract=sampling_contract(config.training,pool)).items():
        require(payload.get(key) == value, 'strict pair contract changed: '+key)
    step = payload.get('step')
    require(type(step) is int and step in ENDPOINTS and payload.get('optimizer_internal_step') == step
        and payload.get('micro_batches_consumed') == payload.get('new_stage_micro_batches_consumed') == step*8
        and payload.get('historical_micro_batches_before_stage') == 672 and payload.get('lifetime_micro_batches_consumed') == 672+step*8
        and payload.get('sampling_cursor') == step*8, 'new optimizer/history/sampling cursor differs')
    state,masters,signature = payload['trainable_model'],payload['master_parameters'],payload['model_signature']
    validate_signature(signature,arm)
    require(set(state) == set(signature) and set(masters) == set(ADAPTER_SHAPES), 'full exported arm weights and six Adapter masters required')
    for name,value in state.items():
        s = signature[name]; dtype = torch.bfloat16 if name in ADAPTER_SHAPES else torch.float32
        require(torch.is_tensor(value) and value.dtype == dtype and list(value.shape) == s['shape']
            and bool(torch.isfinite(value).all()), 'full export geometry/dtype/finite value differs')
        if name in ADAPTER_SHAPES:
            require(masters[name].dtype == torch.float32 and masters[name].shape == value.shape
                and torch.equal(masters[name],value.float()), 'only Adapter masters round through BF16')
    require(payload['master_mapping'] == optimizer_mapping(signature,payload['optimizer'],arm=arm,internal_step=step),
        'all Adapter/gate optimizer states and mapping must match')
    return payload


def validate_resume(path, *, config, hashes, arm):
    payload = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    validate_checkpoint(payload,config=config,hashes=hashes,arm=arm); validate_base_identity(payload,config)
    torch.Generator(device='cpu').set_state(payload['torch_rng_state'])
    random.Random().setstate(payload['python_rng_state']); np.random.RandomState().set_state(payload['numpy_rng_state'])
    rng = payload.get('cuda_rng_state_all')
    require(isinstance(rng,list) and len(rng) == 1 and rng[0].dtype == torch.uint8
        and rng[0].ndim == 1 and rng[0].numel() > 0, 'complete single-GPU RNG state required')
    return payload


def launch(config, config_path, *, arm, lineage, pool, stop_after_step, resume=None,
           shared_lease, shared_max_seconds, shared_request_sha256):
    from training.demo.backend import validate_active_authority
    validate_config(config,output_dir=config.training.output_dir,arm=arm); validate_lineage(lineage); validate_pool(pool)
    require(stop_after_step in ENDPOINTS and type(shared_max_seconds) is int
        and 60 <= shared_max_seconds <= (900 if stop_after_step == 2 else 3600), 'finite fixed2/20 job required')
    output = Path(config.training.output_dir)
    if resume is None:
        require(stop_after_step == 2 and (not output.exists() or not any(output.iterdir())), 'new arm profile only')
    else:
        require(stop_after_step == 20 and Path(resume) == output/'checkpoints/step-0000002.pt', 'strict same-arm2->20 only')
    require(parent_binding() == lineage and pool_contract() == pool, 'actual parent or six-window binding changed')
    hashes = artifact_hashes(config,config_path)
    saved = validate_resume(resume,config=config,hashes=hashes,arm=arm) if resume else load_parent(lineage)
    if resume: require(saved['step'] == 2 and saved['parent_lineage'] == lineage, 'wrong strict arm resume lineage')
    start = 2 if resume else 0
    gpu = fresh.controlled.teacher._shared_training_gpu(shared_lease,max_seconds=shared_max_seconds,request_sha256=shared_request_sha256)
    require(shared_lease.get('gpu_index') in (1,2) and torch.cuda.is_available(), 'only leased Creator GPU1/2')
    device = torch.device('cuda',0); torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    seed_everything(42); output.mkdir(parents=True,exist_ok=True)
    metadata = dict(stage=STAGE,arm=arm,config=config.to_dict(),manifest_hashes=hashes,parent_lineage=lineage,
        sampling_contract=sampling_contract(config.training,pool),method_contract=method_contract(arm),resume=resume,
        resume_step=start,stop_after_step=stop_after_step,git_revision=git_revision(Path(__file__).parent),
        entry_sha256=sha256_file(__file__),gpu_state=gpu.as_dict(),started_unix=time.time(),
        shared_guard=dict(lease=shared_lease,max_seconds=shared_max_seconds,request_sha256=shared_request_sha256))
    if not resume:
        with (output/'run_metadata.json').open('x',encoding='utf-8') as stream: json.dump(metadata,stream,indent=2)
    append_jsonl(output/'executions.jsonl',metadata)
    profiler = OptimizerStepProfiler(device,output_dir=output/'profile_output',gpu_uuid=shared_lease['gpu_uuid'])
    began,last_check = time.monotonic(),0.
    def check():
        nonlocal last_check
        now = time.monotonic(); require(now-began < shared_max_seconds-30, 'finite training deadline reached')
        if now-last_check >= 2: validate_active_authority(shared_lease); last_check = now
    try:
        with profiler.phase('model_load'):
            model,summary = _build_model(config,device,arm); signature = model_signature(model,arm)
            validate_signature(signature,arm)
            if resume:
                require(signature == saved['model_signature'], 'actual full target-module signature differs')
                load_trainable_state_dict(model,saved['trainable_model'])
            else:
                require({n:s for n,s in signature.items() if s['group'] in ('adapter','gate')} == saved['model_signature'],
                    'actual parent Adapter/gate signature differs')
                master.b.pair.install_parent_weights(model,saved['trainable_model'])
            masters = master.make_masters(model,saved['master_parameters'] if resume else None)
            optimizer = make_optimizer(model,masters,arm,saved['optimizer'] if resume else None,check=check)
        loader = build_pair_dataloader(config=config.data,training=config.training,pool=pool,arm=arm)
        profile_lora_A_nonzero = None  # Both pilot arms freeze every LoRA tensor.
        if resume: fresh.controlled.teacher._restore_rng_state(saved,require_complete=True)
        else: seed_everything(42)
        del saved
        segment_adapter_before = {n:p for n,p in master.b.pair._sample_parameters(model).items() if n in ADAPTER_SHAPES}
        model.zero_grad(set_to_none=True); optimizer.zero_grad(set_to_none=True)
        latest,roundtrip_ok = {},bool(resume)
        for step in range(start+1,stop_after_step+1):
            check(); before = master.sampled_masters(masters); before_model = master.b.pair._sample_parameters(model)
            profiler.begin_step(step); losses,samples = [],[]
            for offset in range(8):
                check(); index = (step-1)*8+offset; profiler.set_phase('data_wait')
                batch = default_collate([loader.dataset[index]])
                samples.append(dict(**loader.dataset.sampling_spec(index),actual_timestep=float(batch['timesteps'][0,1])))
                samples[-1]['reference_enabled'] = arm == 'reference'
                samples[-1]['reference_valid_slots'] = int(batch['ref_mask'].sum()) if 'ref_mask' in batch else 0
                profiler.set_phase('forward')
                with torch.autograd.graph.save_on_cpu(pin_memory=True), torch.autocast('cuda',dtype=torch.bfloat16):
                    loss = fresh.controlled.teacher._forward_loss(model,batch,config,device)
                require(bool(torch.isfinite(loss)), 'nonfinite loss')
                profiler.set_phase('backward'); (loss/8).backward(); losses.append(float(loss.detach()))
                samples[-1]['loss'] = losses[-1]; del loss,batch
            task_gradient = task_gradient_evidence(model,arm,step)
            precision = master.b.pair._gradient_precision(model,'frozen')
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            require(bool(torch.isfinite(norm)), 'nonfinite all-trainable global gradient norm')
            master.copy_clipped_gradients(model,masters); precision['gradients_finite'] = True
            profiler.set_phase('optimizer_update'); optimizer.step()
            unprojected = master.sampled_masters(masters)
            changed = sum(int(torch.count_nonzero(unprojected[n]-before[n])) for n in before)
            require(changed > 0, 'no sampled FP32 Adapter master update before projection'); del unprojected
            master.writeback(model,masters,'round'); optimizer.zero_grad(set_to_none=True); model.zero_grad(set_to_none=True)
            storage = master.storage_evidence(model,masters,'round'); after = master.sampled_masters(masters)
            projected = sum(int(torch.count_nonzero(after[n]-before[n])) for n in before)
            updates = update_evidence(model,before_model,arm); del before,after,before_model
            endpoint_adapter_changed = None
            if step in ENDPOINTS:
                samples_now = master.b.pair._sample_parameters(model)
                endpoint_adapter_changed = sum(int(torch.count_nonzero(samples_now[n].float()-value.float()))
                    for n,value in segment_adapter_before.items())
                require(endpoint_adapter_changed > 0, 'fixed endpoint requires cumulative actual BF16 Adapter change')
                del samples_now
                roundtrip_ok = native_roundtrip(optimizer,signature,arm,step,check)
            timing = profiler.end_step(step); memory = timing['memory']
            latest = dict(loss=sum(losses)/8,grad_norm=float(norm),precision_summary=precision,task_gradient_evidence=task_gradient,
                update_evidence=dict(master_updated=True,master_changed_sampled_elements=changed,
                    master_measurement='after_optimizer_before_BF16_projection',projected_master_changed_sampled_elements=projected,
                    model_updates=updates,sampled_not_full_census=True,parent_momentum_inherited=False,
                    includes_weight_decay_and_stage_momentum=True,parameter_change_is_not_quality=True),
                storage_evidence=storage,native_state_restore_matched=roundtrip_ok,
                endpoint_adapter_update=dict(applicable=step in ENDPOINTS,from_optimizer_step=start,
                    to_optimizer_step=step,changed_sampled_elements=endpoint_adapter_changed,
                    per_step_BF16_change_required=False,master_update_required_each_step=True,
                    sampled_not_full_census=True,weight_decay_and_stage_momentum_also_contribute=True),
                gate_statistics=master.b.pair._gate_statistics(model),optimizer_step_seconds=timing['seconds'],step_profile=timing,
                micro_samples=samples,sample_namespace=NAMESPACE,new_stage_sample_first=(step-1)*8,new_stage_sample_last=step*8-1,
                exposures_since_stage_start={r['view_id']:(step*8+5-i)//6 for i,r in enumerate(loader.dataset.episodes)},
                peak_vram_bytes=memory['cumulative_peak_allocated_bytes'],peak_reserved_vram_bytes=memory['cumulative_peak_reserved_bytes'],
                elapsed_seconds=time.monotonic()-began)
            append_jsonl(output/'metrics.jsonl',dict(step=step,optimizer_internal_step=step,**latest))
            print(json.dumps(dict(step=step,arm=arm,**latest)),flush=True)
            if step in ENDPOINTS:
                profiler.begin_checkpoint(step)
                target = master.b.pair._save_checkpoint(output,_checkpoint(model,masters,optimizer,config=config,hashes=hashes,
                    arm=arm,lineage=lineage,pool=pool,step=step,metrics=latest))
                profiler.end_checkpoint(step,checkpoint_path=str(target))
        result = dict(status='completed',stage=STAGE,checkpoint_schema=CHECKPOINT_SCHEMA,arm=arm,step=stop_after_step,
            optimizer_internal_step=stop_after_step,micro_batches_consumed=stop_after_step*8,new_stage_micro_batches_consumed=stop_after_step*8,
            historical_micro_batches_before_stage=672,lifetime_micro_batches_consumed=672+stop_after_step*8,sampling_cursor=stop_after_step*8,
            parent_lineage=lineage,base_identity=lineage['base_identity'],model_signature=signature,
            sampling_contract=sampling_contract(config.training,pool),method_contract=method_contract(arm),precision_contract=precision_contract(arm),
            resume_step=start,checkpoint_path=str(target),configured_max_steps=20,trainable_parameters=summary.trainable_parameters,**latest,
            qualification=dict(master_updated=True,adapter_updated=True,gate_updated=True,gradients_finite=True,
                adapter_task_gradient_nonzero=True,gate_task_gradient_nonzero=True,native_state_restore_matched=roundtrip_ok,
                model_master_projection_equal=True,lora_trainable=False,frozen_lora_has_gradients=False,
                lora_B_updated=updates['lora_B_updated'],
                step2_lora_A_task_gradient_nonzero=profile_lora_A_nonzero,
                peak_reserved_bytes=latest['peak_reserved_vram_bytes']),quality_improvement_verified=False)
        append_jsonl(output/'execution_segments.jsonl',result)
        (output/'execution_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result
    except BaseException as error:
        profiler.close(error=error)
        append_jsonl(output/'failures.jsonl',dict(status='failed',stage=STAGE,arm=arm,failure_type=type(error).__name__,
            error=str(error),resume_step=start,timestamp_unix=time.time()))
        raise
    finally:
        profiler.close()
