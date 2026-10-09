"""Fixed RGB-reflection supervision pair; same round writeback, fresh optimizers.

This stage owns its transformed-cache contract. Old checkpoints are validated
with their real configuration/schema, not relabelled as reflection checkpoints.
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

import train_master_writeback as master
from training.config import load_config as _load_config
from training.data.action_dataset import _seed64, WINDOW97_CACHE_KIND, WINDOW97_ENCODING_POLICY
from training.data.action_resampled import ResampledActionDataset
from training.models.lora import load_trainable_state_dict, trainable_state_dict
from training.runtime import OptimizerStepProfiler, append_jsonl, git_revision, seed_everything, sha256_file
from training.studio import master_writeback_eval as source

STAGE = 'controlled_action_reflection_pair_native97_v1'
CHECKPOINT_SCHEMA = 'controlled_action_reflection_pair_checkpoint_v1'
FACTORY = 'train_reflection_pair:build_reflection_dataloader'
ROOT = master.ROOT.parent / 'teacher-reflection-pair-v1'
DATA = ROOT / 'data'
ARMS = ('original', 'reflect')
VIEWS = ('J_original', 'L_original', 'J_reflect', 'L_reflect')
NAMESPACE = 'controlled_reflection_four_views_seed42_v1'
POOL_SCHEMA = 'controlled_reflection_four_views_pool_v1'
MICRO_BATCHES = 160
GATE, ADAPTER_SHAPES = master.GATE, master.ADAPTER_SHAPES
require = master.require
validate_base_identity = master.validate_base_identity


def cache_module():
    from training.studio import reflection_pair_cache
    return reflection_pair_cache


def arm_for_config(config, arm=None):
    name = Path(config.training.output_dir).parent.name
    matches = [a for a in ARMS if name == f'crf-{a}-profile2-v1']
    require(len(matches) == 1 and (arm is None or arm == matches[0]), 'fixed reflection arm output required')
    return matches[0]


def model_config(config):
    # Only a constructor compatibility view; never a transformed-data validator.
    return master.b.model_config(config)


def validate_probe_config(config, arm=None):
    arm_for_config(config, arm)
    require(config.data.data_factory == FACTORY, 'explicit reflection sampler required')
    require(config.data.manifest_path == str(DATA/'manifests/views.jsonl')
        and isinstance(config.data.manifest_sha256, str) and len(config.data.manifest_sha256) == 64,
        'independent four-view manifest required')
    master.b.pair.validate_probe_config(model_config(config), arm='frozen')


def load_config(path):
    value = _load_config(path)
    validate_probe_config(value)
    return value


def validate_config(config, *, output_dir, arm):
    validate_probe_config(config, arm)
    expected = ROOT/f'crf-{arm}-profile2-v1/training'
    require(Path(output_dir).resolve() == Path(config.training.output_dir).resolve() == expected.resolve(),
        'fixed independent arm output required')
    from training.studio.controlled_action_train import MODEL
    require(config.model.base_model_path == str(MODEL), 'same pinned official base required')
    binding = cache_module().validate_completed_cache()
    require(config.data.manifest_path == binding['manifest']
        and config.data.manifest_sha256 == binding['manifest_sha256'], 'actual completed transformed cache required')


def parent_binding():
    request, result = source.completed_endpoint('round')
    p = source.paths('round'); execution = result['execution']
    history = execution['lifetime_micro_batches_consumed']
    require(type(history) is int and history > 0, 'actual parent exposure history required')
    return dict(mode='weights_only_new_optimizer_new_namespace', parent_stage=master.STAGE,
        parent_checkpoint_schema=master.CHECKPOINT_SCHEMA, parent_arm='round', parent_step=20,
        parent_optimizer_internal_step=execution['optimizer_internal_step'],
        parent_checkpoint=str(p['checkpoint']), parent_checkpoint_stat=source.stat_identity(p['checkpoint']),
        parent_config=str(p['config']), parent_config_sha256=request['config_sha256'], parent_job=str(p['endpoint']),
        parent_records={n:sha256_file(p['endpoint']/(n+'.json')) for n in ('request','result','settlement')},
        historical_micro_batches_before_stage=history, base_identity=execution['parent_lineage']['base_identity'],
        ancestor_lineage=execution['parent_lineage'], optimizer_restored=False, rng_restored=False,
        new_optimizer=True, parent_strict_resume=False, r005_inherited=False, teacher64_inherited=False)


def validate_lineage(lineage):
    p = source.paths('round')
    expected = dict(mode='weights_only_new_optimizer_new_namespace', parent_stage=master.STAGE,
        parent_checkpoint_schema=master.CHECKPOINT_SCHEMA, parent_arm='round', parent_step=20,
        parent_optimizer_internal_step=40, parent_checkpoint=str(p['checkpoint']),
        parent_config=str(p['config']), parent_job=str(p['endpoint']), optimizer_restored=False,
        rng_restored=False, new_optimizer=True, parent_strict_resume=False,
        r005_inherited=False, teacher64_inherited=False)
    require(isinstance(lineage, dict) and all(lineage.get(k) == v for k,v in expected.items()),
        'genuine completed round20 BF16 export weights-only parent required')
    require(type(lineage.get('historical_micro_batches_before_stage')) is int
        and lineage['historical_micro_batches_before_stage'] > 0
        and isinstance(lineage.get('base_identity'), dict), 'parent history/base identity missing')
    require(set(lineage.get('parent_records', {})) == {'request','result','settlement'}
        and set(lineage.get('parent_checkpoint_stat', {})) == {'bytes','mtime_ns'}, 'parent metadata binding missing')


def load_parent(lineage):
    require(parent_binding() == lineage, 'completed round20 source changed')
    p = source.paths('round'); config = master.load_config(p['config'])
    hashes = master.artifact_hashes(config,p['config'])
    payload = torch.load(p['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    master.validate_checkpoint(payload,config=config,hashes=hashes,arm='round')
    master.validate_base_identity(payload,config)
    _, result = source.completed_endpoint('round')
    for key in ('parent_lineage','method_contract','sampling_contract','precision_contract'):
        require(payload[key] == result['execution'][key], 'actual parent endpoint differs: '+key)
    require(payload['step'] == 20 and payload['arm'] == 'round'
        and payload['lifetime_micro_batches_consumed'] == lineage['historical_micro_batches_before_stage'],
        'parent exposure history differs from actual checkpoint')
    require(source.stat_identity(p['checkpoint']) == lineage['parent_checkpoint_stat'], 'parent changed during load')
    # No parent optimizer, master or RNG is returned to the new stage.
    return dict(trainable_model=payload['trainable_model'], model_signature=payload['model_signature'])


def reflected_actions(actions):
    return [[row[i] for i in (0,3,2,1,4,7,6,5)] for row in actions]


def pool_contract():
    binding = cache_module().validate_completed_cache()
    result = dict(schema=POOL_SCHEMA, binding=binding, source_original_pool=master.b.pool_contract())
    validate_pool(result)
    return result


def validate_pool(pool):
    require(isinstance(pool, dict) and pool.get('schema') == POOL_SCHEMA, 'new reflection pool required')
    binding = pool.get('binding', {})
    require(binding.get('schema') == 'reflection_four_views_cache_v1'
        and binding.get('manifest') == str(DATA/'manifests/views.jsonl')
        and binding.get('cache_index') == str(DATA/'features/views.index.jsonl')
        and set(binding.get('views', {})) == set(VIEWS), 'exact explicit four-view cache required')
    original = pool.get('source_original_pool'); master.b._validate_pool(original)
    for index, target in enumerate(('J','L')):
        source_row = original['windows'][index]
        for reflected in (False,True):
            name = target+('_reflect' if reflected else '_original'); row = binding['views'][name]
            actions = reflected_actions(source_row['original_actions']) if reflected else source_row['original_actions']
            require(row.get('view_id') == name and row.get('episode_id') == source_row['episode_id']
                and row.get('window_index') == index and row.get('source_start') == (0 if target == 'J' else 210)
                and row.get('transform') == ('horizontal_reflection_v1' if reflected else 'identity')
                and row.get('future_actions_96x8') == actions
                and row.get('prompt_sha256_float32') == source_row['prompt_sha256_float32'],
                'fixed source0/210 and full A/D+J/L label reflection required')
            if not reflected:
                for key in ('shard_path','shard_stat','first_latent_sha256_float32'):
                    require(row.get(key) == source_row.get(key), 'original view must reuse actual source: '+key)
            else:
                require(row.get('shard_path') != source_row['shard_path'], 'mirror may not reuse old latent shard')
                require(Path(row['shard_path']).is_relative_to(DATA), 'mirror shard outside new data directory')


def sampling_contract(training, pool, arm, lineage):
    validate_pool(pool); validate_lineage(lineage)
    require(arm in ARMS and training.seed == 42 and training.max_steps == 20
        and training.micro_batch_size == 1 and training.gradient_accumulation_steps == 8, 'fixed20x8 arms required')
    schedule = list(VIEWS if arm == 'reflect' else VIEWS[:2]*2)
    return dict(namespace=NAMESPACE, training_seed=42,data_seed=_seed64(42,NAMESPACE), arm=arm,
        virtual_samples=160, first_stage_sample_index=0,last_stage_sample_index=159,
        view_schedule=schedule, exposures_per_view={v:40*schedule.count(v) for v in VIEWS},
        historical_micro_batches_before_stage=lineage['historical_micro_batches_before_stage'],
        original_absolute_dataset_cursor_continued=False,
        diffusion_seed_policy='seed64(seed64(42,namespace),diffusion,new_stage_index)',
        noise_and_timestep='identical_indexed_random_draw_across_arms;noise_not_reflected', pool=pool)


class ReflectionDataset(ResampledActionDataset):
    """Explicit mixed-kind cache loader; inherits only the unchanged noise/flow draw."""
    def __init__(self, *, pool, arm):
        validate_pool(pool); require(arm in ARMS, 'fixed arm required')
        self.pool,self.arm,self.num_samples = pool,arm,MICRO_BATCHES
        self.data_seed,self.timestep_shift = _seed64(42,NAMESPACE),5.
        self.episodes = [pool['binding']['views'][v] for v in VIEWS]
        self._shards,self._verified = {},set()

    def sampling_spec(self,index):
        require(not isinstance(index,bool),'sample index cannot be bool'); index=integer_index(index)
        if not 0 <= index < MICRO_BATCHES: raise IndexError(index)
        slot=index%4 if self.arm == 'reflect' else index%2
        row=self.episodes[slot]
        return dict(absolute_sample_index=index,new_stage_sample_index=index,index_namespace=NAMESPACE,
            episode_slot=slot,episode_id=row['episode_id'],window_index=row['window_index'],
            source_start=row['source_start'],view_id=row['view_id'],transform=row['transform'],
            source_target='J' if row['source_start']==0 else 'L',
            diffusion_seed=_seed64(self.data_seed,'diffusion',index))

    def _cached_window(self,row,window_index):
        from training.studio.causal_native_eval import tensor_sha
        require(window_index == row['window_index'], 'wrong local cache window')
        path=Path(row['shard_path']); st=path.stat()
        require(dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)==row['shard_stat'],'cached shard stat changed')
        if str(path) not in self._shards:
            self._shards[str(path)]=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        payload=self._shards[str(path)]
        reflected=row['transform']=='horizontal_reflection_v1'
        require(payload.get('schema_version')==1 and payload.get('episode_id')==row['episode_id']
            and payload.get('kind')==('reflection_native97_tensor_shard_v1' if reflected else WINDOW97_CACHE_KIND)
            and torch.is_tensor(payload.get('source_starts'))
            and payload['source_starts'].tolist()==[0,210], 'new/old shard identity must be explicit')
        if not reflected:
            require(payload.get('encoding_policy')==WINDOW97_ENCODING_POLICY
                and payload.get('prompt_policy')=='scene_static_only_v1','original encoding policy changed')
        else:
            require(payload.get('transform')=='horizontal_reflection_v1'
                and payload.get('key_permutation')==[0,3,2,1,4,7,6,5], 'explicit new reflection tensor contract changed')
        latent=payload['clean_latents'][window_index].float()
        actions=payload['actions'][window_index].float(); prompt=payload['prompt_embeds'].float()
        require(tuple(latent.shape)==(25,48,30,52) and tuple(actions.shape)==(96,8)
            and prompt.ndim==2 and 1<=prompt.shape[0]<=512 and prompt.shape[1]==4096,
            'native97 latent/action/text shapes required')
        if row['view_id'] not in self._verified:
            require(all(bool(torch.isfinite(t).all()) for t in (latent,actions,prompt))
                and torch.equal(actions,torch.tensor(row['future_actions_96x8'],dtype=torch.float32))
                and tensor_sha(latent[:1][None])==row['first_latent_sha256_float32']
                and tensor_sha(prompt)==row['prompt_sha256_float32'], 'actual cached tensors/labels differ')
            self._verified.add(row['view_id'])
        return dict(clean_latents=latent,actions=actions,prompt_embeds=prompt)


def build_reflection_dataloader(*,config,training,pool=None,arm=None):
    pool=pool_contract() if pool is None else pool
    if arm is None:
        matches=[a for a in ARMS if Path(training.output_dir).parent.name==f'crf-{a}-profile2-v1']
        require(len(matches)==1,'explicit sampler arm required'); arm=matches[0]
    require(config.manifest_path==pool['binding']['manifest'] and config.precomputed_latents
        and config.precomputed_text_embeddings and config.num_frames==97 and config.num_workers==0,
        'no raw RGB/VAE/text encoder in training loader')
    return DataLoader(ReflectionDataset(pool=pool,arm=arm),batch_size=1,shuffle=False,num_workers=0,
        pin_memory=torch.cuda.is_available(),drop_last=True)


def artifact_hashes(config,config_path):
    validate_probe_config(config)
    binding=cache_module().validate_completed_cache(); p=source.paths('round')
    require(config.data.manifest_path==binding['manifest']
        and config.data.manifest_sha256==binding['manifest_sha256'],'new manifest mismatch')
    return dict(dataset_manifest=binding['manifest_sha256'], feature_index=binding['cache_index_sha256'],
        feature_receipt=binding['receipt_sha256'],training_config=sha256_file(config_path),
        parent_config=sha256_file(p['config']),
        **{'parent_'+n:sha256_file(p['endpoint']/(n+'.json')) for n in ('request','result','settlement')},
        visual_gate=binding['visual_gate']['review_sha256'])


def precision_contract():
    result=master.precision_contract()
    result.update(optimizer='native_AdamW8bit_new_empty_state_then_strict_restore',
        writeback='per_step_Adapter_master_BF16_to_FP32_projection;gate_moments_not_projected')
    return result


def method_contract(arm):
    require(arm in ARMS,'fixed original/reflect arm required')
    return dict(method=STAGE,arm=arm,intervention='training_view_pool_only',ratio_cap=.1,
        adapter_lr=1e-5,gate_lr=.01,adapter_weight_decay=.01,gate_weight_decay=0.,betas=[.9,.95],
        global_clip_norm=1.,lora_trainable=False,seed=42,max_new_steps=20,
        precision=precision_contract(),new_optimizer=True,parent_strict_resume=False,
        loss='unchanged_uniform_future_flow_MSE',fresh_from_official_base=False,
        negative_action_loss=False,true_same_state_counterfactual=False,generalization_experiment=False,
        reference_memory_implemented=False,automatic_downstream_training=False)


def _build_model(config,device,arm=None):
    validate_probe_config(config,arm)
    return master.b.pair._build_model(model_config(config),device,arm='frozen')


def make_optimizer(model,masters,saved=None,*,check=lambda:None):
    if saved is not None:
        return master.make_optimizer(model,masters,saved,check=check)
    import bitsandbytes as bnb
    groups=[dict(params=list(masters.values()),name='action_adapter',lr=1e-5,weight_decay=.01),
        dict(params=[dict(model.named_parameters())[GATE]],name='controlled_action_gate',lr=.01,weight_decay=0.)]
    optimizer=bnb.optim.AdamW8bit(groups,betas=(.9,.95))
    require(optimizer.state_dict()['state']=={},'new optimizer must not inherit moments')
    return optimizer


def native_roundtrip(optimizer,signature,step,check=lambda:None):
    state=optimizer.state_dict()
    master.optimizer_mapping(signature,state,internal_step=step)
    optimizer.load_state_dict(state)
    require(master.equal_tree(state,optimizer.state_dict(),check),'native_state_restore_not_matched')
    return True


def _checkpoint(model,masters,optimizer,*,config,hashes,arm,lineage,pool,step,metrics):
    state=optimizer.state_dict(); signature=master.b.pair._model_signature(model,'frozen')
    history=lineage['historical_micro_batches_before_stage']
    return dict(format_version=1,stage=STAGE,checkpoint_schema=CHECKPOINT_SCHEMA,arm=arm,
        config=config.to_dict(),manifest_hashes=hashes,base_identity=lineage['base_identity'],parent_lineage=lineage,
        method_contract=method_contract(arm),precision_contract=precision_contract(),
        sampling_contract=sampling_contract(config.training,pool,arm,lineage),model_signature=signature,
        trainable_model=trainable_state_dict(model),master_parameters={n:p.detach().cpu() for n,p in masters.items()},
        master_mapping=master.optimizer_mapping(signature,state,internal_step=step),optimizer=state,
        step=step,optimizer_internal_step=step,micro_batches_consumed=step*8,new_stage_micro_batches_consumed=step*8,
        historical_micro_batches_before_stage=history,lifetime_micro_batches_consumed=history+step*8,sampling_cursor=step*8,
        metrics=metrics,torch_rng_state=torch.get_rng_state(),cuda_rng_state_all=torch.cuda.get_rng_state_all(),
        python_rng_state=random.getstate(),numpy_rng_state=np.random.get_state())


def validate_checkpoint(payload,*,config,hashes,arm=None):
    arm=arm_for_config(config,arm); validate_probe_config(config,arm)
    require(payload.get('format_version')==1 and payload.get('stage')==STAGE
        and payload.get('checkpoint_schema')==CHECKPOINT_SCHEMA and payload.get('arm')==arm,'new stage/arm required')
    lineage=payload['parent_lineage']; validate_lineage(lineage)
    pool=payload.get('sampling_contract',{}).get('pool'); validate_pool(pool)
    for key,value in dict(config=config.to_dict(),manifest_hashes=hashes,base_identity=lineage['base_identity'],
        method_contract=method_contract(arm),precision_contract=precision_contract(),
        sampling_contract=sampling_contract(config.training,pool,arm,lineage)).items():
        require(payload.get(key)==value,'strict reflection contract changed: '+key)
    step=payload.get('step'); history=lineage['historical_micro_batches_before_stage']
    require(type(step)is int and step in (2,20) and payload.get('optimizer_internal_step')==step
        and payload.get('micro_batches_consumed')==payload.get('new_stage_micro_batches_consumed')==step*8
        and payload.get('historical_micro_batches_before_stage')==history
        and payload.get('lifetime_micro_batches_consumed')==history+step*8
        and payload.get('sampling_cursor')==step*8,'new optimizer and namespace cursor/history mismatch')
    state,masters,signature=payload['trainable_model'],payload['master_parameters'],payload['model_signature']
    require(set(state)==set(signature)==set(ADAPTER_SHAPES)|{GATE}
        and set(masters)==set(ADAPTER_SHAPES),'complete BF16 export and six masters required')
    for name,value in state.items():
        shape=ADAPTER_SHAPES[name] if name!=GATE else (3072,)
        require(torch.is_tensor(value) and value.dtype==(torch.bfloat16 if name!=GATE else torch.float32)
            and tuple(value.shape)==shape and bool(torch.isfinite(value).all()),'invalid export geometry/precision')
        if name!=GATE:
            m=masters[name]
            require(m.dtype==torch.float32 and m.shape==value.shape and torch.equal(m,value.float()),
                'round-only checkpoint must preserve exact projected masters')
    require(payload['master_mapping']==master.optimizer_mapping(signature,payload['optimizer'],internal_step=step),
        'native optimizer step/name mapping changed')
    return payload


def validate_resume(path,*,config,hashes,arm):
    value=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    validate_checkpoint(value,config=config,hashes=hashes,arm=arm)
    torch.Generator(device='cpu').set_state(value['torch_rng_state'])
    random.Random().setstate(value['python_rng_state']); np.random.RandomState().set_state(value['numpy_rng_state'])
    rng=value.get('cuda_rng_state_all')
    require(isinstance(rng,list) and len(rng)==1 and rng[0].dtype==torch.uint8
        and rng[0].ndim==1 and rng[0].numel()>0,'complete single-GPU RNG required')
    return value


def launch(config,config_path,*,arm,lineage,pool,stop_after_step,resume=None,
           shared_lease,shared_max_seconds,shared_request_sha256):
    from training.demo.backend import validate_active_authority
    validate_config(config,output_dir=config.training.output_dir,arm=arm)
    validate_lineage(lineage); validate_pool(pool)
    require(stop_after_step in (2,20) and 60<=shared_max_seconds<=(900 if stop_after_step==2 else 3600),
        'finite2/20 job required')
    output=Path(config.training.output_dir)
    if resume is None:
        require(stop_after_step==2 and (not output.exists() or not any(output.iterdir())),'new profile only')
    else:
        require(Path(resume)==output/'checkpoints/step-0000002.pt' and stop_after_step==20,'strict same-arm2->20 only')
    require(parent_binding()==lineage and pool_contract()==pool,'actual parent/cache/visual binding changed')
    hashes=artifact_hashes(config,config_path)
    saved=validate_resume(resume,config=config,hashes=hashes,arm=arm) if resume else load_parent(lineage)
    if resume: require(saved['parent_lineage']==lineage and saved['step']==2,'wrong same-arm step2 resume lineage')
    start=saved['step'] if resume else 0
    gpu=master.b.teacher._shared_training_gpu(shared_lease,max_seconds=shared_max_seconds,
        request_sha256=shared_request_sha256)
    require(shared_lease.get('gpu_index') in (1,2) and torch.cuda.is_available(),'only leased Creator GPU1/2')
    device=torch.device('cuda',0); torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    seed_everything(42); output.mkdir(parents=True,exist_ok=True)
    metadata=dict(stage=STAGE,arm=arm,config=config.to_dict(),manifest_hashes=hashes,parent_lineage=lineage,
        method_contract=method_contract(arm),sampling_contract=sampling_contract(config.training,pool,arm,lineage),
        resume=resume,resume_step=start,stop_after_step=stop_after_step,git_revision=git_revision(Path(__file__).parent),
        entry_sha256=sha256_file(__file__),gpu_state=gpu.as_dict(),started_unix=time.time(),
        shared_guard=dict(lease=shared_lease,max_seconds=shared_max_seconds,request_sha256=shared_request_sha256))
    if not resume:
        with (output/'run_metadata.json').open('x',encoding='utf-8') as f: json.dump(metadata,f,indent=2)
    append_jsonl(output/'executions.jsonl',metadata)
    profiler=OptimizerStepProfiler(device,output_dir=output/'profile_output',gpu_uuid=shared_lease['gpu_uuid'])
    began=time.monotonic(); last_check=0.
    def check():
        nonlocal last_check
        now=time.monotonic(); require(now-began<shared_max_seconds-30,'finite training deadline reached')
        if now-last_check>=2: validate_active_authority(shared_lease); last_check=now
    try:
        with profiler.phase('model_load'):
            model,summary=_build_model(config,device,arm)
            load_trainable_state_dict(model,saved['trainable_model'])
            signature=master.b.pair._model_signature(model,'frozen')
            require(signature==saved['model_signature'],'forward signature differs')
            masters=master.make_masters(model,saved['master_parameters'] if resume else None)
            optimizer=make_optimizer(model,masters,saved['optimizer'] if resume else None,check=check)
        loader=build_reflection_dataloader(config=config.data,training=config.training,pool=pool,arm=arm)
        if resume: master.b.teacher._restore_rng_state(saved,require_complete=True)
        else: seed_everything(42)
        del saved
        model.zero_grad(set_to_none=True); optimizer.zero_grad(set_to_none=True)
        latest={}; roundtrip_ok=bool(resume)
        for step in range(start+1,stop_after_step+1):
            check(); before=master.sampled_masters(masters); before_model=master.b.pair._sample_parameters(model)
            profiler.begin_step(step); losses,samples=[],[]
            for offset in range(8):
                check(); index=(step-1)*8+offset; profiler.set_phase('data_wait')
                batch=default_collate([loader.dataset[index]])
                samples.append(dict(**loader.dataset.sampling_spec(index),actual_timestep=float(batch['timesteps'][0,1])))
                profiler.set_phase('forward')
                with torch.autograd.graph.save_on_cpu(pin_memory=True),torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=master.b.teacher._forward_loss(model,batch,config,device)
                require(bool(torch.isfinite(loss)),'nonfinite loss')
                profiler.set_phase('backward'); (loss/8).backward(); losses.append(float(loss.detach()))
                samples[-1]['loss']=losses[-1]; del loss,batch
            precision=master.b.pair._gradient_precision(model,'frozen')
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            require(bool(torch.isfinite(norm)),'nonfinite global gradient norm')
            master.copy_clipped_gradients(model,masters); precision['gradients_finite']=True
            profiler.set_phase('optimizer_update'); optimizer.step()
            unprojected=master.sampled_masters(masters)
            changed=sum(int(torch.count_nonzero(unprojected[n]-before[n])) for n in before)
            require(changed>0,'no actual sampled FP32 master update before projection')
            del unprojected
            master.writeback(model,masters,'round')
            optimizer.zero_grad(set_to_none=True); model.zero_grad(set_to_none=True)
            storage=master.storage_evidence(model,masters,'round')
            after=master.sampled_masters(masters)
            projected_changed=sum(int(torch.count_nonzero(after[n]-before[n])) for n in before)
            updates=master.model_update_evidence(model,before_model)
            del before,after,before_model
            if step in (2,20): roundtrip_ok=native_roundtrip(optimizer,signature,step,check)
            timing=profiler.end_step(step); memory=timing['memory']
            schedule=VIEWS if arm=='reflect' else VIEWS[:2]*2
            latest=dict(loss=sum(losses)/8,grad_norm=float(norm),precision_summary=precision,
                update_evidence=dict(master_updated=True,master_changed_sampled_elements=changed,
                    master_measurement='after_optimizer_before_BF16_projection',
                    projected_master_changed_sampled_elements=projected_changed,
                    sampled_not_full_census=True,model_updates=updates,
                    includes_weight_decay_and_within_stage_momentum=True,parent_momentum_inherited=False,
                    parameter_change_is_not_quality=True,master_residual_sampled_rms=0.),
                storage_evidence=storage,native_state_restore_matched=roundtrip_ok,
                gate_statistics=master.b.pair._gate_statistics(model),optimizer_step_seconds=timing['seconds'],
                step_profile=timing,micro_samples=samples,sample_namespace=NAMESPACE,
                new_stage_sample_first=(step-1)*8,new_stage_sample_last=step*8-1,
                exposures_since_stage_start={v:step*2*schedule.count(v) for v in VIEWS},
                peak_vram_bytes=memory['cumulative_peak_allocated_bytes'],
                peak_reserved_vram_bytes=memory['cumulative_peak_reserved_bytes'],elapsed_seconds=time.monotonic()-began)
            append_jsonl(output/'metrics.jsonl',dict(step=step,optimizer_internal_step=step,**latest))
            print(json.dumps(dict(step=step,arm=arm,**latest)),flush=True)
            if step in (2,20):
                profiler.begin_checkpoint(step)
                target=master.b.pair._save_checkpoint(output,_checkpoint(model,masters,optimizer,config=config,hashes=hashes,
                    arm=arm,lineage=lineage,pool=pool,step=step,metrics=latest))
                profiler.end_checkpoint(step,checkpoint_path=str(target))
        history=lineage['historical_micro_batches_before_stage']
        result=dict(status='completed',stage=STAGE,checkpoint_schema=CHECKPOINT_SCHEMA,arm=arm,step=stop_after_step,
            optimizer_internal_step=stop_after_step,micro_batches_consumed=stop_after_step*8,
            new_stage_micro_batches_consumed=stop_after_step*8,historical_micro_batches_before_stage=history,
            lifetime_micro_batches_consumed=history+stop_after_step*8,sampling_cursor=stop_after_step*8,
            parent_lineage=lineage,sampling_contract=sampling_contract(config.training,pool,arm,lineage),
            method_contract=method_contract(arm),precision_contract=precision_contract(),resume_step=start,
            checkpoint_path=str(target),configured_max_steps=20,trainable_parameters=summary.trainable_parameters,**latest,
            qualification=dict(master_updated=True,adapter_updated=updates['adapter_forward_updated'],gate_updated=True,
                gradients_finite=True,native_state_restore_matched=roundtrip_ok,model_master_projection_equal=True,
                lora_trainable=False,peak_reserved_bytes=latest['peak_reserved_vram_bytes']),quality_improvement_verified=False)
        append_jsonl(output/'execution_segments.jsonl',result)
        (output/'execution_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result
    except BaseException as exc:
        profiler.close(error=exc)
        append_jsonl(output/'failures.jsonl',dict(status='failed',stage=STAGE,arm=arm,
            failure_type=type(exc).__name__,error=str(exc),resume_step=start,timestamp_unix=time.time()))
        raise
    finally:
        profiler.close()
