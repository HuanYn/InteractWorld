"""Fresh controlled action teacher on six reviewed windows; one fixed60 trajectory.

Only numerical/model-building helpers are reused. This stage has independent
data, initialization and checkpoint contracts, not a relabelled old teacher.
"""
from __future__ import annotations

import copy
import json
import math
from operator import index as integer_index
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

import train_controlled_action_teacher as controlled
import train_master_writeback as master
from training.config import ActionTeacherConfig, load_config as _load_config
from training.data.action_dataset import _seed64, WINDOW97_CACHE_KIND, WINDOW97_ENCODING_POLICY
from training.data.action_resampled import ResampledActionDataset
from training.models.lora import load_trainable_state_dict, trainable_state_dict
from training.runtime import OptimizerStepProfiler, append_jsonl, git_revision, seed_everything, sha256_file

STAGE = 'controlled_action_balanced_fresh60_native97_v1'
CHECKPOINT_SCHEMA = 'controlled_action_balanced_fresh60_checkpoint_v1'
FACTORY = 'train_balanced_fresh_teacher:build_balanced_dataloader'
ROOT = master.ROOT.parent / 'teacher-balanced-fresh60-v1'
DATA = master.ROOT.parent / 'teacher-balanced-cache-v1/data'
NAMESPACE = 'controlled_balanced_six_windows_fresh_seed42_v1'
POOL_SCHEMA = 'controlled_balanced_six_original_windows_v1'
ENDPOINTS, MICRO_BATCHES = (2, 20, 40, 60), 480
# Identity order was frozen before model outputs; no loss-based sample selection.
IDENTITIES = ('0306d1c583a017d95f5a3baadbe2c0da', '010f014c51157abf3525c46014d10db0',
              '006047b2418cdd9de904dc4c8e2d6c55', '036450c7b62e47bc0dea6a00481b181a',
              '029610cd22fa5391818880cb8ecb7280', '017ebc69dde12426310d8c8ea9f0e3a6')
GATE, ADAPTER_SHAPES = master.GATE, master.ADAPTER_SHAPES
require = master.require
validate_base_identity = controlled.validate_base_identity


def cache_module():
    from training.studio import balanced_turn_cache
    return balanced_turn_cache


def model_config(config):
    """Actual60-step configuration; no old-stage compatibility projection."""
    return copy.deepcopy(config)


def validate_probe_config(config):
    require(type(config) is ActionTeacherConfig, 'typed fresh teacher configuration required')
    config.validate()
    require(config.data.data_factory == FACTORY and config.data.manifest_path == str(DATA/'manifests/train.jsonl')
        and isinstance(config.data.manifest_sha256, str) and len(config.data.manifest_sha256) == 64,
        'independent six-window train manifest required')
    require(config.data.num_frames == 97 and config.data.num_workers == 0
        and config.data.prompt_cache_path is None and not config.data.low_sigma_stratification
        and config.data.precomputed_latents and config.data.precomputed_text_embeddings, 'native97 inline static cache required')
    require(config.model.action_scale == 1 and config.model.lora_alpha == 16
        and config.model.lora_dropout == 0 and config.model.timestep_shift == 5
        and config.model.gradient_checkpointing, 'fixed controlled model parameters required')
    require(config.optimizer.name == 'adamw8bit' and config.optimizer.adapter_lr == 1e-5
        and config.optimizer.lora_lr == 0 and tuple(config.optimizer.betas) == (.9,.95)
        and config.optimizer.weight_decay == .01 and config.optimizer.max_grad_norm == 1,
        'fixed optimizer parameters required')
    require(config.training.lora_parameter_dtype == 'float32' and config.training.max_steps == 60
        and config.training.checkpoint_every == 20 and config.training.log_every == 1
        and Path(config.training.output_dir) == ROOT/'cbf-profile2-v1/training', 'fixed fresh60 output/trajectory required')


def load_config(path):
    config = _load_config(path); validate_probe_config(config); return config


def validate_config(config, *, output_dir):
    from training.studio.controlled_action_train import MODEL
    validate_probe_config(config)
    require(Path(output_dir) == Path(config.training.output_dir)
        and config.model.base_model_path == str(MODEL), 'fixed official base/new output required')
    pool = pool_contract()
    require(config.data.manifest_sha256 == pool['binding']['binding']['manifest_sha256'], 'completed train manifest mismatch')


def initialization_contract(config):
    result = controlled.initialization_contract(config)
    result.update(new_optimizer=True, historical_micro_batches_before_stage=0,
        r005_inherited=False, causal_weights_inherited=False, old_teacher_inherited=False,
        foundation_pretrained=True, foundation_random_pretraining=False,
        adapter_nonzero_before_zero_gate=True, start_step=0)
    return result


def pool_contract():
    binding = cache_module().validate_completed_cache('train')
    result = dict(schema=POOL_SCHEMA, binding=binding)
    validate_pool(result); return result


def ordered_windows(pool):
    by_id = {r['episode_id']: r for r in pool['binding']['windows']}
    return [by_id[identity] for identity in IDENTITIES]


def validate_pool(pool):
    require(isinstance(pool, dict) and pool.get('schema') == POOL_SCHEMA, 'independent fresh six-window pool required')
    selected = pool.get('binding', {}); binding = selected.get('binding', {})
    require(selected.get('schema') == 'balanced_native97_separated_cache_v1'
        and selected.get('purpose') == 'train' and selected.get('qualification_never_training') is True
        and binding.get('manifest') == str(DATA/'manifests/train.jsonl')
        and binding.get('path') == str(DATA/'features/train.features.jsonl')
        and binding.get('episodes') == binding.get('windows') == 6, 'train6-only cache binding required')
    rows = selected.get('windows', [])
    require(len(rows) == 6 and {r.get('episode_id') for r in rows} == set(IDENTITIES), 'only the six prespecified episodes may train')
    for i, row in enumerate(ordered_windows(pool)):
        require(row.get('purpose') == 'train' and row.get('candidate_role') == 'train_candidate'
            and row.get('target') == ('J' if i % 2 == 0 else 'L') and row.get('window_index') == 0
            and type(row.get('source_start')) is int and row['source_start'] >= 0
            and row.get('view_id') == row['episode_id']+'-s'+str(row['source_start'])
            and row.get('source_kind') == 'original_dataset_rgb97', 'fixed original direction/role/window required')
        actions = row.get('future_actions_96x8', [])
        require(len(actions) == 96 and all(len(r) == 8 and all(v in (0,1) for v in r) for r in actions),
            '96 complete canonical8-key action rows required')
        require(Path(row['shard_path']).is_relative_to(DATA/'features')
            and set(row.get('shard_stat', {})) == {'bytes','mtime_ns'}, 'new cache shard/stat required')


def sampling_contract(training, pool):
    validate_pool(pool)
    require(training.seed == 42 and training.max_steps == 60
        and training.micro_batch_size == 1 and training.gradient_accumulation_steps == 8, 'fixed60x8 required')
    views = [r['view_id'] for r in ordered_windows(pool)]
    return dict(namespace=NAMESPACE, training_seed=42, data_seed=_seed64(42,NAMESPACE),
        virtual_samples=480, first_stage_sample_index=0, last_stage_sample_index=479,
        view_schedule=views, target_schedule=['J','L']*3, exposures_per_view={v:80 for v in views},
        historical_micro_batches_before_stage=0, diffusion_seed_policy='seed64(seed64(42,namespace),diffusion,new_stage_index)',
        total_epochs_over_fixed_six_windows=80, independent_episodes=6, qualification_windows_used=0, pool=pool)


class BalancedDataset(ResampledActionDataset):
    """Only noise/flow target arithmetic is inherited; selection is new and fixed."""
    def __init__(self, *, pool):
        validate_pool(pool); self.pool, self.num_samples = pool, MICRO_BATCHES
        self.data_seed, self.timestep_shift = _seed64(42,NAMESPACE), 5.
        self.episodes = ordered_windows(pool); self._shards, self._verified = {}, set()

    def sampling_spec(self, index):
        require(not isinstance(index,bool), 'sample index cannot be bool'); index = integer_index(index)
        if not 0 <= index < MICRO_BATCHES: raise IndexError(index)
        slot = index % 6; row = self.episodes[slot]
        return dict(absolute_sample_index=index, new_stage_sample_index=index, index_namespace=NAMESPACE,
            episode_slot=slot, episode_id=row['episode_id'], window_index=0, source_start=row['source_start'],
            view_id=row['view_id'], source_target=row['target'], purpose='train',
            diffusion_seed=_seed64(self.data_seed,'diffusion',index))

    def _cached_window(self, row, window_index):
        from training.studio.causal_native_eval import tensor_sha
        require(window_index == 0, 'each train episode has exactly one chosen window')
        path = Path(row['shard_path']); stat = path.stat()
        require(dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns) == row['shard_stat'], 'cached shard changed')
        if str(path) not in self._shards:
            self._shards[str(path)] = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        payload = self._shards[str(path)]
        require(payload.get('schema_version') == 1 and payload.get('kind') == WINDOW97_CACHE_KIND
            and payload.get('episode_id') == row['episode_id']
            and payload.get('encoding_policy') == WINDOW97_ENCODING_POLICY
            and payload.get('prompt_policy') == 'scene_static_only_v1'
            and torch.is_tensor(payload.get('source_starts'))
            and payload['source_starts'].tolist() == [row['source_start']], 'genuine matching native97 cache required')
        require(tuple(payload['clean_latents'].shape) == (1,25,48,30,52)
            and tuple(payload['actions'].shape) == (1,96,8), 'exact one native97 training window required')
        latent = payload['clean_latents'][0].float(); actions = payload['actions'][0].float()
        prompt = payload['prompt_embeds'].float()
        require(prompt.ndim == 2 and 1 <= prompt.shape[0] <= 512 and prompt.shape[1] == 4096, 'static text shape changed')
        if row['view_id'] not in self._verified:
            require(all(bool(torch.isfinite(t).all()) for t in (latent,actions,prompt))
                and torch.equal(actions,torch.tensor(row['future_actions_96x8'],dtype=torch.float32))
                and tensor_sha(latent[:1][None]) == row['first_latent_sha256_float32']
                and tensor_sha(prompt) == row['prompt_sha256_float32'], 'actual training tensors differ from completed cache')
            self._verified.add(row['view_id'])
        return dict(clean_latents=latent,actions=actions,prompt_embeds=prompt)


def build_balanced_dataloader(*, config, training, pool=None):
    pool = pool_contract() if pool is None else pool
    require(config.manifest_path == pool['binding']['binding']['manifest'] and config.precomputed_latents
        and config.precomputed_text_embeddings and config.num_frames == 97 and config.num_workers == 0,
        'no raw RGB/VAE/text encoder in training loader')
    sampling_contract(training,pool)
    return DataLoader(BalancedDataset(pool=pool),batch_size=1,shuffle=False,num_workers=0,
        pin_memory=torch.cuda.is_available(),drop_last=True)


def artifact_hashes(config, config_path):
    validate_probe_config(config); selected = cache_module().validate_completed_cache('train'); info = selected['binding']
    require(config.data.manifest_path == info['manifest'] and config.data.manifest_sha256 == info['manifest_sha256'],
        'actual completed six-window manifest changed')
    return dict(dataset_manifest=info['manifest_sha256'], feature_index=info['sha256'],
        feature_receipt=info['receipt_sha256'], cache_result=selected['cache_result_sha256'],
        training_config=sha256_file(config_path))


def precision_contract():
    result = master.precision_contract()
    result.update(optimizer='native_AdamW8bit_new_empty_state_then_strict_restore',
        writeback='per_step_Adapter_master_BF16_to_FP32_projection;gate_moments_not_projected')
    return result


def method_contract():
    return dict(method=STAGE, ratio_cap=.1, adapter_lr=1e-5, gate_lr=.01, adapter_weight_decay=.01,
        gate_weight_decay=0., betas=[.9,.95], global_clip_norm=1., lora_trainable=False,
        seed=42, max_new_steps=60, precision=precision_contract(), new_optimizer=True,
        fresh_from_official_base=True, loss='unchanged_uniform_future_flow_MSE',
        negative_action_loss=False, causal=False, reference_memory_implemented=False,
        data_only_ablation=False, automatic_downstream_training=False,
        checkpoint_policy='fixed2_20_40_60_no_pruning_no_intermediate_video_or_selection')


def _build_model(config, device):
    validate_probe_config(config)
    # The controlled constructor does not invoke the old20-step data validator.
    return controlled._build_model(config,device)


def make_optimizer(model, masters, saved=None, *, check=lambda:None):
    if saved is not None: return master.make_optimizer(model,masters,saved,check=check)
    import bitsandbytes as bnb
    groups = [dict(params=list(masters.values()),name='action_adapter',lr=1e-5,weight_decay=.01),
              dict(params=[dict(model.named_parameters())[GATE]],name='controlled_action_gate',lr=.01,weight_decay=0.)]
    optimizer = bnb.optim.AdamW8bit(groups,betas=(.9,.95))
    require(optimizer.state_dict()['state'] == {}, 'fresh optimizer cannot inherit moments')
    return optimizer


def native_roundtrip(optimizer, signature, step, check=lambda:None):
    state = optimizer.state_dict(); master.optimizer_mapping(signature,state,internal_step=step)
    optimizer.load_state_dict(state)
    require(master.equal_tree(state,optimizer.state_dict(),check), 'native state restore mismatch')
    return True


def task_gradient_evidence(model, step):
    """Pre-optimizer task gradients, not decay/momentum or parameter-change proxies."""
    named = dict(model.named_parameters()); maximum = 0.
    for name in ADAPTER_SHAPES:
        grad = named[name].grad
        require(grad is not None and grad.dtype == torch.bfloat16 and bool(torch.isfinite(grad).all()),
            'complete finite BF16 Adapter task gradient required')
        maximum = max(maximum,float(grad.detach().abs().max()))
    gate = named[GATE].grad
    require(gate is not None and gate.dtype == torch.float32 and bool(torch.isfinite(gate).all()), 'finite gate task gradient required')
    require(step != 1 or maximum == 0., 'initial zero gate must block Adapter task gradient at step1')
    require(step != 2 or maximum > 0., 'step2 must have actual nonzero Adapter task gradient, not decay-only update')
    return dict(measured_before_optimizer=True, full_adapter_tensor_max_abs=maximum,
        adapter_task_gradient_nonzero=maximum > 0., gate_task_gradient_max_abs=float(gate.detach().abs().max()),
        expected_zero_adapter_gradient_at_first_step=step == 1, weight_decay_excluded=True,
        optimizer_momentum_excluded=True, quality_evidence=False)


def _checkpoint(model, masters, optimizer, *, config, hashes, base, pool, step, metrics):
    state = optimizer.state_dict(); signature = master.b.pair._model_signature(model,'frozen')
    return dict(format_version=1,stage=STAGE,checkpoint_schema=CHECKPOINT_SCHEMA,
        config=config.to_dict(),manifest_hashes=hashes,base_identity=base,
        initialization_contract=initialization_contract(config),method_contract=method_contract(),
        precision_contract=precision_contract(),sampling_contract=sampling_contract(config.training,pool),
        model_signature=signature,trainable_model=trainable_state_dict(model),
        master_parameters={n:p.detach().cpu() for n,p in masters.items()},
        master_mapping=master.optimizer_mapping(signature,state,internal_step=step),optimizer=state,
        step=step,optimizer_internal_step=step,micro_batches_consumed=step*8,new_stage_micro_batches_consumed=step*8,
        historical_micro_batches_before_stage=0,lifetime_micro_batches_consumed=step*8,sampling_cursor=step*8,
        metrics=metrics,torch_rng_state=torch.get_rng_state(),cuda_rng_state_all=torch.cuda.get_rng_state_all(),
        python_rng_state=random.getstate(),numpy_rng_state=np.random.get_state())


def validate_checkpoint(payload, *, config, hashes):
    validate_probe_config(config)
    require(payload.get('format_version') == 1 and payload.get('stage') == STAGE
        and payload.get('checkpoint_schema') == CHECKPOINT_SCHEMA, 'independent fresh60 checkpoint schema required')
    pool = payload.get('sampling_contract',{}).get('pool'); validate_pool(pool)
    for key,value in dict(config=config.to_dict(),manifest_hashes=hashes,
        initialization_contract=initialization_contract(config),method_contract=method_contract(),
        precision_contract=precision_contract(),sampling_contract=sampling_contract(config.training,pool)).items():
        require(payload.get(key) == value, 'strict fresh60 contract changed: '+key)
    step = payload.get('step')
    require(type(step) is int and step in ENDPOINTS and payload.get('optimizer_internal_step') == step
        and payload.get('micro_batches_consumed') == payload.get('new_stage_micro_batches_consumed') == step*8
        and payload.get('historical_micro_batches_before_stage') == 0
        and payload.get('lifetime_micro_batches_consumed') == payload.get('sampling_cursor') == step*8,
        'fresh optimizer and zero-history sampling cursor mismatch')
    state,masters,signature = payload['trainable_model'],payload['master_parameters'],payload['model_signature']
    require(set(state) == set(signature) == set(ADAPTER_SHAPES)|{GATE}
        and set(masters) == set(ADAPTER_SHAPES), 'complete BF16 export and six FP32 masters required')
    for name,value in state.items():
        shape = ADAPTER_SHAPES[name] if name != GATE else (3072,)
        require(torch.is_tensor(value) and value.dtype == (torch.bfloat16 if name != GATE else torch.float32)
            and tuple(value.shape) == shape and bool(torch.isfinite(value).all()), 'invalid export geometry/precision')
        if name != GATE:
            m = masters[name]
            require(m.dtype == torch.float32 and m.shape == value.shape and torch.equal(m,value.float()),
                'round-only saved masters must exactly equal BF16 export in FP32')
    require(payload['master_mapping'] == master.optimizer_mapping(signature,payload['optimizer'],internal_step=step),
        'native optimizer step/name mapping changed')
    return payload


def validate_resume(path, *, config, hashes):
    payload = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    validate_checkpoint(payload,config=config,hashes=hashes); validate_base_identity(payload,config)
    torch.Generator(device='cpu').set_state(payload['torch_rng_state'])
    random.Random().setstate(payload['python_rng_state']); np.random.RandomState().set_state(payload['numpy_rng_state'])
    rng = payload.get('cuda_rng_state_all')
    require(isinstance(rng,list) and len(rng) == 1 and rng[0].dtype == torch.uint8
        and rng[0].ndim == 1 and rng[0].numel() > 0, 'complete single-GPU RNG state required')
    return payload


def launch(config, config_path, *, pool, stop_after_step, resume=None,
           shared_lease, shared_max_seconds, shared_request_sha256):
    from training.demo.backend import validate_active_authority
    validate_config(config,output_dir=config.training.output_dir); validate_pool(pool)
    require(stop_after_step in ENDPOINTS and type(shared_max_seconds) is int
        and 60 <= shared_max_seconds <= (900 if stop_after_step == 2 else 3000), 'finite fixed endpoint required')
    output = Path(config.training.output_dir); expected_start = {2:0,20:2,40:20,60:40}[stop_after_step]
    if resume is None:
        require(expected_start == 0 and (not output.exists() or not any(output.iterdir())), 'fresh profile only')
    else:
        require(expected_start > 0 and Path(resume) == output/f'checkpoints/step-{expected_start:07d}.pt', 'strict immediate predecessor only')
    require(pool_contract() == pool, 'actual completed training cache changed')
    hashes = artifact_hashes(config,config_path); base = controlled.base_identity(config)
    saved = validate_resume(resume,config=config,hashes=hashes) if resume else None
    if saved is not None: require(saved['step'] == expected_start, 'wrong strict resume step')
    gpu = controlled.teacher._shared_training_gpu(shared_lease,max_seconds=shared_max_seconds,request_sha256=shared_request_sha256)
    require(shared_lease.get('gpu_index') in (1,2) and torch.cuda.is_available(), 'only leased Creator GPU1/2')
    device = torch.device('cuda',0); torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    seed_everything(42); output.mkdir(parents=True,exist_ok=True)
    metadata = dict(stage=STAGE,config=config.to_dict(),manifest_hashes=hashes,base_identity=base,
        initialization_contract=initialization_contract(config),method_contract=method_contract(),
        sampling_contract=sampling_contract(config.training,pool),resume=resume,resume_step=expected_start,
        stop_after_step=stop_after_step,git_revision=git_revision(Path(__file__).parent),entry_sha256=sha256_file(__file__),
        gpu_state=gpu.as_dict(),started_unix=time.time(),shared_guard=dict(lease=shared_lease,
        max_seconds=shared_max_seconds,request_sha256=shared_request_sha256))
    if not resume:
        with (output/'run_metadata.json').open('x',encoding='utf-8') as stream: json.dump(metadata,stream,indent=2)
    append_jsonl(output/'executions.jsonl',metadata)
    profiler = OptimizerStepProfiler(device,output_dir=output/'profile_output',gpu_uuid=shared_lease['gpu_uuid'])
    began, last_check = time.monotonic(), 0.
    def check():
        nonlocal last_check
        now = time.monotonic(); require(now-began < shared_max_seconds-30, 'finite training deadline reached')
        if now-last_check >= 2: validate_active_authority(shared_lease); last_check = now
    try:
        with profiler.phase('model_load'):
            model,summary = _build_model(config,device)
            signature = master.b.pair._model_signature(model,'frozen')
            if saved is not None:
                load_trainable_state_dict(model,saved['trainable_model'])
                require(signature == saved['model_signature'], 'resume forward signature differs')
            else:
                named = dict(model.named_parameters())
                require(not bool(torch.count_nonzero(named[GATE])), 'fresh gate must be exactly zero')
                require(all(bool(torch.count_nonzero(named[n])) for n in ADAPTER_SHAPES if n.endswith('weight')),
                    'fresh random Adapter weights must not also be zero')
            masters = master.make_masters(model,saved['master_parameters'] if saved is not None else None)
            optimizer = make_optimizer(model,masters,saved['optimizer'] if saved is not None else None,check=check)
        loader = build_balanced_dataloader(config=config.data,training=config.training,pool=pool)
        if saved is not None: controlled.teacher._restore_rng_state(saved,require_complete=True)
        else: seed_everything(42)
        del saved
        model.zero_grad(set_to_none=True); optimizer.zero_grad(set_to_none=True)
        latest, roundtrip_ok = {}, bool(resume)
        for step in range(expected_start+1,stop_after_step+1):
            check(); before = master.sampled_masters(masters); before_model = master.b.pair._sample_parameters(model)
            profiler.begin_step(step); losses,samples = [],[]
            for offset in range(8):
                check(); index = (step-1)*8+offset; profiler.set_phase('data_wait')
                batch = default_collate([loader.dataset[index]])
                samples.append(dict(**loader.dataset.sampling_spec(index),actual_timestep=float(batch['timesteps'][0,1])))
                profiler.set_phase('forward')
                with torch.autograd.graph.save_on_cpu(pin_memory=True), torch.autocast('cuda',dtype=torch.bfloat16):
                    loss = controlled.teacher._forward_loss(model,batch,config,device)
                require(bool(torch.isfinite(loss)), 'nonfinite loss')
                profiler.set_phase('backward'); (loss/8).backward(); losses.append(float(loss.detach()))
                samples[-1]['loss'] = losses[-1]; del loss,batch
            task_gradient = task_gradient_evidence(model,step)
            precision = master.b.pair._gradient_precision(model,'frozen')
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            require(bool(torch.isfinite(norm)), 'nonfinite global gradient norm')
            master.copy_clipped_gradients(model,masters); precision['gradients_finite'] = True
            profiler.set_phase('optimizer_update'); optimizer.step()
            unprojected = master.sampled_masters(masters)
            changed = sum(int(torch.count_nonzero(unprojected[n]-before[n])) for n in before)
            require(changed > 0, 'no sampled FP32 master update before projection'); del unprojected
            master.writeback(model,masters,'round'); optimizer.zero_grad(set_to_none=True); model.zero_grad(set_to_none=True)
            storage = master.storage_evidence(model,masters,'round'); after = master.sampled_masters(masters)
            projected_changed = sum(int(torch.count_nonzero(after[n]-before[n])) for n in before)
            updates = master.model_update_evidence(model,before_model); del before,after,before_model
            if step in ENDPOINTS: roundtrip_ok = native_roundtrip(optimizer,signature,step,check)
            timing = profiler.end_step(step); memory = timing['memory']
            latest = dict(loss=sum(losses)/8,grad_norm=float(norm),precision_summary=precision,task_gradient_evidence=task_gradient,
                update_evidence=dict(master_updated=True,master_changed_sampled_elements=changed,
                    master_measurement='after_optimizer_before_BF16_projection',projected_master_changed_sampled_elements=projected_changed,
                    sampled_not_full_census=True,model_updates=updates,includes_weight_decay_and_within_stage_momentum=True,
                    parent_momentum_inherited=False,parameter_change_is_not_quality=True,master_residual_sampled_rms=0.),
                storage_evidence=storage,native_state_restore_matched=roundtrip_ok,
                gate_statistics=master.b.pair._gate_statistics(model),optimizer_step_seconds=timing['seconds'],
                step_profile=timing,micro_samples=samples,sample_namespace=NAMESPACE,
                new_stage_sample_first=(step-1)*8,new_stage_sample_last=step*8-1,
                exposures_since_stage_start={r['view_id']:(step*8+5-i)//6 for i,r in enumerate(loader.dataset.episodes)},
                peak_vram_bytes=memory['cumulative_peak_allocated_bytes'],
                peak_reserved_vram_bytes=memory['cumulative_peak_reserved_bytes'],elapsed_seconds=time.monotonic()-began)
            append_jsonl(output/'metrics.jsonl',dict(step=step,optimizer_internal_step=step,**latest))
            print(json.dumps(dict(step=step,**latest)),flush=True)
            if step in ENDPOINTS:
                profiler.begin_checkpoint(step)
                target = master.b.pair._save_checkpoint(output,_checkpoint(model,masters,optimizer,config=config,hashes=hashes,
                    base=base,pool=pool,step=step,metrics=latest))
                profiler.end_checkpoint(step,checkpoint_path=str(target))
        result = dict(status='completed',stage=STAGE,checkpoint_schema=CHECKPOINT_SCHEMA,step=stop_after_step,
            optimizer_internal_step=stop_after_step,micro_batches_consumed=stop_after_step*8,new_stage_micro_batches_consumed=stop_after_step*8,
            historical_micro_batches_before_stage=0,lifetime_micro_batches_consumed=stop_after_step*8,sampling_cursor=stop_after_step*8,
            initialization_contract=initialization_contract(config),base_identity=base,model_signature=signature,
            sampling_contract=sampling_contract(config.training,pool),method_contract=method_contract(),precision_contract=precision_contract(),
            resume_step=expected_start,checkpoint_path=str(target),configured_max_steps=60,trainable_parameters=summary.trainable_parameters,**latest,
            qualification=dict(master_updated=True,adapter_updated=updates['adapter_forward_updated'],gate_updated=True,
                gradients_finite=True,adapter_task_gradient_nonzero=task_gradient['adapter_task_gradient_nonzero'],
                native_state_restore_matched=roundtrip_ok,model_master_projection_equal=True,lora_trainable=False,
                peak_reserved_bytes=latest['peak_reserved_vram_bytes']),quality_improvement_verified=False)
        append_jsonl(output/'execution_segments.jsonl',result)
        (output/'execution_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result
    except BaseException as error:
        profiler.close(error=error)
        append_jsonl(output/'failures.jsonl',dict(status='failed',stage=STAGE,failure_type=type(error).__name__,
            error=str(error),resume_step=expected_start,timestamp_unix=time.time()))
        raise
    finally:
        profiler.close()
