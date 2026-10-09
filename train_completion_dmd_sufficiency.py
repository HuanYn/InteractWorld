"""Bounded short/long genuine self-rollout DMD, independent of active TF/ODE.

The prepared request binds a real completed/settled ODE80. No teacher endpoint
MSE, GT future, official student or toy model is substituted for DMD. Two arms
use identical deterministic draws/new optimizers; only score-window length
differs. Native resource/gradient gates run during the first real profile.
"""
from __future__ import annotations

from contextlib import nullcontext
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

import numpy as np
import torch

import train_completion_ode_sufficiency as ode
from training import completion_dmd_core_v2 as core
from training import completion_dmd_native_v2 as native

previous=ode.previous
require=core.require
STAGE='completion_longforcing_selfrollout_dmd_v2'
CHECKPOINT_SCHEMA='completion_longforcing_selfrollout_dmd_checkpoint_v2'
ROOT=ode.ROOT.parent/'longforcing-sufficiency-v2'
ENDPOINTS=(2,8,16,24,32)
PREVIOUS={2:0,8:2,16:8,24:16,32:24}
MODES=tuple('completion_dmd_v2_'+kind for kind in ('profile','train','validate','eval'))
CAP_GPU_SECONDS=16*3600


def read(path):return ode.read(path)
def sha(path):return previous.sha256_file(path)


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def object_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {name:sha(path) for name,path in dict(entry=__file__,core=core.__file__,native=native.__file__,
        ode_entry=ode.__file__,ode_core=ode.core().__file__,tf_entry=previous.__file__,
        parallel_ode_jobs=Path(__file__).parent/'training/completion_ode_jobs_v2.py',
        causal_model=Path(__file__).parent/'wan/modules/causal_model.py',
        wan_wrapper=Path(__file__).parent/'utils/wan_wrapper.py').items()}


def _manifest(spec):
    from training.completion_ode_jobs_v2 import parallel_manifest
    manifest=parallel_manifest(spec.get('parallel_budget_binding'))
    rows=[r for r in manifest['jobs'] if r['job_id']==spec.get('job_id')]
    require(len(rows)==1 and all(spec.get(k)==v for k,v in rows[0].items()),'DMD spec differs from complete immutable manifest')
    require(manifest['groups']['dmd']['accounting_root']==str(ROOT),'same shared16h DMD root required')
    from training import completion_parallel_v4 as v4
    if v4.is_v4(spec.get('parallel_budget_binding')):
        logical=[r for r in manifest['logical_jobs'] if r['group']=='dmd']
        fixed=build_manifest_jobs(spec['arguments'].get('parent_binding_from_job_id'),gpu_indices=(1,2))
        require(logical==fixed,'original complete DMD scientific contract changed')
        return manifest
    owned=[r for r in manifest['jobs'] if r['group']=='dmd']
    profiles={r['arguments'].get('arm'):r['gpu_index'] for r in owned if r['mode']=='completion_dmd_v2_profile'}
    require(set(profiles)=={'short','long'},'both equal DMD arms require their actual independent profile')
    fixed=build_manifest_jobs(spec['arguments'].get('parent_binding_from_job_id'),
        gpu_indices=(profiles['short'],profiles['long']))
    require({r['job_id']:r for r in owned}=={r['job_id']:r for r in fixed},
        'complete DMD2/8/16/24/32 trajectory and all formal numeric/video jobs cannot be omitted or changed')
    return manifest


def _settled(directory,*,ledger=None):
    directory=Path(directory)
    request,result,settlement=(read(directory/(n+'.json')) for n in ('request','result','settlement'))
    require(result.get('status')=='completed' and settlement.get('status')=='settled'
        and settlement.get('outcome')=='completed','actual successful settled predecessor required')
    if ledger is None:
        from training.studio.completion_causal_sufficiency_train import ledger_snapshot
        ledger,_=ledger_snapshot()
    reservation=ledger['reservations'].get(settlement.get('reservation_id'),{})
    context=reservation.get('context',{})
    require(reservation.get('status')=='settled' and context.get('job_id')==request['job_id']
        and context.get('job_directory')==str(directory) and context.get('request_sha256')==sha(directory/'request.json')
        and reservation.get('settlement')==settlement and reservation.get('charged_seconds')==settlement.get('charged_gpu_seconds'),
        'predecessor is not the exact original real-ledger settlement')
    return request,result.get('execution',result),settlement


def ode_parent_binding(job_directory,step=80,*,ledger=None):
    require(step==80 and Path(job_directory).is_relative_to(ode.ROOT),'only actual formal ODE80 default parent required')
    request,result,_=_settled(job_directory,ledger=ledger)
    checkpoint=ode.ROOT/'training/checkpoints/step-0000080.pt'
    require(result.get('stage')==ode.STAGE and result.get('checkpoint_schema')==ode.CHECKPOINT_SCHEMA
        and result.get('step')==80 and result.get('sampling_cursor')==640
        and result.get('checkpoint_path')==str(checkpoint),'not a genuine completed formalODE80 parent')
    config=Path(request['config_path'])
    require(config.is_relative_to(ode.ROOT) and sha(config)==request['config_sha256'],'actual ODE configuration pin changed')
    return dict(schema='completion_dmd_actual_ode_parent_v2',step=80,stage=ode.STAGE,
        checkpoint_schema=ode.CHECKPOINT_SCHEMA,job_directory=str(job_directory),checkpoint=str(checkpoint),
        checkpoint_stat=ode.stat_identity(checkpoint),config=str(config),config_sha256=sha(config),
        records={n+'_sha256':sha(Path(job_directory)/(n+'.json')) for n in ('request','result','settlement')},
        frozen_real_teacher='self_trained_expanded80_bidirectional',new_generator_optimizer=True,
        new_fake_optimizer=True,fake_initialization='same_full_ODE80_FP32_LoRA',
        historical_micro_batches_before_stage=result['lifetime_micro_batches_consumed'],
        parent_strict_resume=False,quality_certified=False)


def resolve_parent(spec,manifest,*,ledger=None):
    args=spec['arguments'];identity=args.get('parent_binding_from_job_id')
    require(isinstance(identity,str) and identity in spec['depends_on']
        and args.get('expected_parent_step')==80 and args.get('expected_parent_schema')==ode.CHECKPOINT_SCHEMA
        and args.get('expected_parent_root')==str(ode.ROOT),
        'immutable manifest must name actual future ODE80 dependency/schema/root')
    rows=[r for r in manifest['jobs'] if r['job_id']==identity]
    require(len(rows)==1 and rows[0]['group']=='ode','parent must be the owned formalODE producing job')
    return ode_parent_binding(rows[0]['directory'],ledger=ledger)


def _budget(spec,*,lease=None):
    from training.completion_ode_jobs_v2 import reconcile_parallel_budget
    from training.studio.completion_causal_sufficiency_train import ledger_snapshot
    ledger,_=ledger_snapshot()
    if lease is None:
        result=_budget_before_reserve(spec,ledger)
    else:
        result=reconcile_parallel_budget(spec['parallel_budget_binding'],ledger=ledger,lease=lease,
            max_seconds=spec['max_seconds'])
    require(result['group_paid_gpu_seconds']['dmd']+result['group_active_gpu_seconds']['dmd']
        +result['group_pending_gpu_seconds']['dmd']<=CAP_GPU_SECONDS,'shared two-arm DMD16h exhausted')
    return result,ledger


def _budget_before_reserve(spec,ledger):
    """Read all original rows even at0active; pending current ceiling once."""
    from training.completion_ode_jobs_v2 import parallel_manifest,GROUP_CAPS
    from training import completion_parallel_v4 as v4
    if v4.is_v4(spec.get('parallel_budget_binding')):return v4.reconcile(spec['parallel_budget_binding'],ledger=ledger)
    manifest=parallel_manifest(spec['parallel_budget_binding'])
    require(ledger.get('version')==1 and isinstance(ledger.get('reservations'),dict)
        and type(ledger.get('storage_bytes_upper')) in (int,float)
        and 0<=ledger['storage_bytes_upper']<=500000000000,'actual original bounded ledger required')
    roots={g:Path(r['accounting_root']) for g,r in manifest['groups'].items()}
    jobs={r['job_id']:r for r in manifest['jobs']}
    paid={g:0. for g in roots};active={g:0. for g in roots};seen=set();cards=[];uuids=[]
    global_paid=global_active=0.
    for rid,row in ledger['reservations'].items():
        context=row['context'];identity=context['job_id'];require(identity not in seen,'duplicate global ledger job')
        seen.add(identity);directory=Path(context['job_directory'])
        groups=[g for g,r in roots.items() if directory==r or r in directory.parents]
        require(len(groups)<=1,'ambiguous stage accounting');group=groups[0] if groups else None
        if row['status']=='settled':
            amount=row['charged_seconds']
            require(type(amount) in (int,float) and math.isfinite(amount) and amount>=0,'invalid historical fee')
            global_paid+=amount
            if group:paid[group]+=amount
            if identity in jobs:require(row['settlement']['outcome']=='completed','failed identity cannot be retried')
        else:
            require(row['status']=='reserved' and identity in jobs,'foreign active Creator work blocks preparation')
            job=jobs[identity];actual=read(directory/'request.json');original=read(directory/'lease.json')
            require(row['lease']==original and original['reservation_id']==rid
                and actual['parallel_budget_binding']==spec['parallel_budget_binding']
                and actual['contract_max_seconds']==job['max_seconds']
                and type(actual['max_seconds']) is int and 0<actual['max_seconds']<=job['max_seconds']
                and actual['max_seconds']==context['max_seconds']
                and context['request_sha256']==sha(directory/'request.json')
                and row['reserved_seconds']==actual['max_seconds']+60
                and original['gpu_index']==job['gpu_index'] and group==job['group'],
                'actual owned lease/card/request differs')
            global_active+=row['reserved_seconds'];active[group]+=row['reserved_seconds']
            cards.append(original['gpu_index']);uuids.append(original['gpu_uuid'])
    require(len(cards)<=2 and len(set(cards))==len(cards) and len(set(uuids))==len(uuids),
        'at most two distinct owned physicalGPU leases allowed')
    pending=copy.deepcopy(manifest['future_envelopes_gpu_seconds'])
    for identity,row in jobs.items():
        if identity not in seen:pending[row['group']]+=row['max_seconds']+60
    require(global_paid+global_active+sum(pending.values())<=360000,'real history/active/full future exceeds100h')
    for group,cap in GROUP_CAPS.items():
        require(paid[group]+active[group]+pending[group]<=cap,'real full '+group+' budget exceeds subcap')
    return dict(global_paid_gpu_seconds=global_paid,all_active_reserved_gpu_seconds=global_active,
        group_paid_gpu_seconds=paid,group_active_gpu_seconds=active,group_pending_gpu_seconds=pending,
        remaining_worst_gpu_seconds=sum(pending.values()),global_upper_gpu_seconds=global_paid+global_active+sum(pending.values()))


def arm_path(arm):
    core.DMDConfig(arm).validate();return ROOT/arm


def checkpoint_path(arm,step):
    require(step in ENDPOINTS,'fixed common DMD2/8/16/24/32 endpoint required')
    return arm_path(arm)/'training/checkpoints'/f'step-{step:07d}.pt'


def counters(step,parent):
    require(type(step) is int and 0<=step<=32,'DMD generator cursor0..32 required')
    return dict(step=step,generator_optimizer_internal_step=step,fake_optimizer_internal_step=step+2,
        generator_updates=step,fake_updates=step+2,fake_warmup_updates=2,
        rollout_samples=step,micro_batches_consumed=step,sampling_cursor=step,
        historical_micro_batches_before_stage=parent['historical_micro_batches_before_stage'],
        lifetime_micro_batches_consumed=parent['historical_micro_batches_before_stage']+step)


def sampling_spec(step,window_count=60):
    require(type(step) is int and 1<=step<=32 and window_count==60,'fixed60-window own-rollout schedule required')
    # Same original JL-interleaved slot and all stream seeds for both arms.
    index=step-1
    return dict(update=step,window_slot=index%60,namespace=core.NAMESPACE,
        rollout_seed=core.stream_seed(42,index,'own-rollout'),score_seed=42,score_update=index,
        fake_score_update=step+1,arm_independent_sampling=True,
        rollout_blocks=20,actions='original96_rows_periodic_repeat_crop240')


def build_manifest_jobs(ode80_job_id,*,gpu_indices=(1,2)):
    """Complete28-job two-arm template:55,680GPU seconds at all ceilings.

    Step24 is an explicit technical split, not a new selected formal length.
    No future endpoint hash is invented; prepare derives it after settlement.
    Actual measured training timeout must fit these fixed ceilings.
    """
    require(isinstance(ode80_job_id,str) and ode80_job_id and tuple(sorted(gpu_indices))==(1,2),
        'actual futureODE80 producer identity and the two authorized cards required')
    jobs=[]
    profile_ids={arm:f'd30-dmd-{arm}-profile2-v1' for arm in ('short','long')}
    common=dict(parent_binding_from_job_id=ode80_job_id,expected_parent_step=80,
        expected_parent_schema=ode.CHECKPOINT_SCHEMA,expected_parent_root=str(ode.ROOT))
    for arm,gpu in zip(('short','long'),gpu_indices):
        previous_job=None
        for stop in ENDPOINTS:
            start=PREVIOUS[stop];kind='profile' if stop==2 else 'train'
            identity=profile_ids[arm] if stop==2 else f'd30-dmd-{arm}-to{stop}-v1'
            deps=[ode80_job_id]+([] if previous_job is None else [previous_job])
            if stop==8:deps+=list(profile_ids.values())
            jobs.append(dict(job_id=identity,mode='completion_dmd_v2_'+kind,group='dmd',gpu_index=gpu,
                max_seconds=3600,cleanup_seconds=60,directory=str(arm_path(arm)/identity),
                depends_on=sorted(set(deps)),arguments=dict(common,arm=arm,start_step=start,stop_after_step=stop)))
            previous_job=identity
            if stop not in (8,16,32):continue
            val=f'd30-dmd-{arm}-validate{stop}-v1'
            jobs.append(dict(job_id=val,mode='completion_dmd_v2_validate',group='dmd',gpu_index=gpu,
                max_seconds=600,cleanup_seconds=60,directory=str(arm_path(arm)/val),
                depends_on=[ode80_job_id,identity],arguments=dict(common,arm=arm,endpoint_step=stop)))
            for case in ('pair-j','holdout-l'):
                eval_id=f'd30-dmd-{arm}-case-{case}-{stop}-v1'
                jobs.append(dict(job_id=eval_id,mode='completion_dmd_v2_eval',group='dmd',gpu_index=gpu,
                    max_seconds=1200,cleanup_seconds=60,directory=str(arm_path(arm)/eval_id),
                    depends_on=[ode80_job_id,identity],arguments=dict(common,arm=arm,endpoint_step=stop,case_id=case)))
    require(len(jobs)==28 and sum(r['max_seconds']+60 for r in jobs)==55680,
        'complete paired DMD template/subcap changed')
    return jobs


def _actual_timeout(spec,manifest,ledger):
    args=spec['arguments'];start,stop=args['start_step'],args['stop_after_step']
    if start==0:
        return spec['max_seconds'],dict(schema='creator_measured_timeout_v1',kind='fixed',max_seconds=spec['max_seconds'])
    require(stop in PREVIOUS and PREVIOUS[stop]==start,'strictDMD2→8→16→32 transition required')
    prior=[];durations=[]
    for row in manifest['jobs']:
        if row['group']=='dmd' and row['arguments'].get('arm')==args['arm'] and row['mode'] in MODES[:2]:
            # The shared immutable-at-prepare metrics file contains the full
            # trajectory through the latest predecessor. Earlier producers
            # cannot pin this now-grown file as if it ended at their endpoint.
            if row['arguments'].get('stop_after_step',1000)==start:
                rq,execution,_=_settled(row['directory'],ledger=ledger)
                require(execution.get('stage')==STAGE and execution.get('step')==row['arguments']['stop_after_step'],
                    'real DMD measured predecessor endpoint required')
                profile=execution.get('actual_profile',{})
                require(profile.get('native_kv_parity',{}).get('status')=='passed'
                    and profile.get('native_replay_causality',{}).get('status')=='passed'
                    and execution.get('native_optimizer_restore_matched') is True
                    and 0<execution.get('peak_reserved_bytes',0)<23*1024**3
                    and execution.get('generator_gradient',{}).get('gradient_norm',0)>0,
                    'actual full DMD context/resource/task-gradient/native-state profile required')
                metrics=arm_path(args['arm'])/'training/metrics.jsonl'
                prior.append(dict(job_id=row['job_id'],**{n:dict(path=str(Path(row['directory'])/(n+'.json')),
                    sha256=sha(Path(row['directory'])/(n+'.json'))) for n in ('request','result','settlement')},
                    metrics=dict(path=str(metrics),sha256=sha(metrics))))
    require(prior,'actual settled DMD profile/timing history required')
    metrics=arm_path(args['arm'])/'training/metrics.jsonl'
    for line in metrics.read_text(encoding='utf-8').splitlines():
        item=json.loads(line)
        if item['step']<=start:durations.append((item['step'],item['optimizer_step_seconds']))
    require([step for step,_ in durations]==list(range(1,start+1))
        and all(type(t) in (int,float) and math.isfinite(t) and t>0 for _,t in durations),
        'complete actual per-step DMD timing history required')
    slowest=max(t for _,t in durations);actual=math.ceil(slowest*(stop-start)*1.35+180)
    require(actual<=spec['max_seconds'],'measured DMD segment does not fit reviewed fixed ceiling; stop without extending')
    return actual,dict(schema='creator_measured_timeout_v1',kind='measured_steps',max_seconds=actual,additional_steps=stop-start,
        multiplier=1.35,startup_allowance_seconds=180,slowest_step_seconds=slowest,profiles=prior)


def _memory_plan(parent):
    config=ode.load_config(parent['config'])
    source=Path(config.model.base_model_path)/'config.json'
    actual=read(source);actual.setdefault('patch_size',[1,2,2])
    value=native.memory_estimate(actual)
    value.update(model_config_path=str(source),model_config_sha256=sha(source),
        base_weights_residency='one_frozen_BF16_Wan_plus_one_FP32_LoRA_bank',
        native_optimizer_residency='one_role_only;both_complete_states_parked_CPU',
        autograd_saved_tensors='CPU_offload_and_checkpoint_full122latent_finalblock_replay',
        GPU_profile_required=True,known_dense_fullKV_not_allocated=True)
    require(value['gpu_layer_kv_scratch_bytes']+value['gpu_layer_cross_scratch_bytes']<1024**3,
        'known KV scratch exceeds bounded1GiB before any reserve')
    return value


def prepare_job(spec,directory):
    """New request, CPU-only. Derive future parent hashes only after settlement."""
    directory=Path(directory);args=spec.get('arguments',{});arm=args.get('arm')
    core.DMDConfig(arm).validate()
    require(directory==arm_path(arm)/spec.get('job_id','') and directory.is_dir()
        and directory.name==spec['job_id'] and not (directory/'request.json').exists()
        and spec.get('mode') in MODES and spec.get('group')=='dmd' and spec.get('gpu_index') in spec.get('parallel_budget_binding',{}).get('allowed_gpu_indices',[1,2])
        and spec.get('cleanup_seconds')==60 and type(spec.get('max_seconds')) is int
        and 60<=spec['max_seconds']<=3600 and not torch.cuda.is_initialized(),
        'fresh owned finite CPU-only DMD request required')
    manifest=_manifest(spec);budget,ledger=_budget(spec);parent=resolve_parent(spec,manifest,ledger=ledger)
    kind=spec['mode'].removeprefix('completion_dmd_v2_')
    request=copy.deepcopy(spec)
    request.update(schema='completion_dmd_deadline_job_v2',contract_max_seconds=spec['max_seconds'],
        actual_parent_binding=parent,source_hashes=source_hashes(),method_contract=core.DMDConfig(arm).record(),
        memory_estimate=_memory_plan(parent),budget_at_prepare=budget,
        derived_timeout=dict(schema='creator_measured_timeout_v1',kind='fixed',max_seconds=spec['max_seconds']))
    if kind in ('profile','train'):
        start,stop=args.get('start_step'),args.get('stop_after_step')
        require(stop in PREVIOUS and PREVIOUS[stop]==start and ((kind=='profile')==(start==0)),
            'profile2 then same-trajectory common8/16/32 required')
        request['max_seconds'],request['derived_timeout']=_actual_timeout(spec,manifest,ledger)
        if start:
            require(checkpoint_path(arm,start).is_file(),'actual predecessor checkpoint required before reservation')
            request['resume_checkpoint_stat']=ode.stat_identity(checkpoint_path(arm,start))
    else:
        endpoint=args.get('endpoint_step')
        require(endpoint in (8,16,32),'formal DMD numeric/video checkpoints8/16/32 required')
        producers=[r for r in manifest['jobs'] if r['group']=='dmd' and r['arguments'].get('arm')==arm
            and r['arguments'].get('stop_after_step')==endpoint and r['mode'] in MODES[:2]]
        require(len(producers)==1 and producers[0]['job_id'] in spec['depends_on'],
            'validation/video must depend on exact same-arm actual endpoint')
        _settled(producers[0]['directory'],ledger=ledger)
        request['endpoint_producer']=dict(job_id=producers[0]['job_id'],directory=producers[0]['directory'],
            records={n:dict(path=str(Path(producers[0]['directory'])/(n+'.json')),
                sha256=sha(Path(producers[0]['directory'])/(n+'.json'))) for n in ('request','result','settlement')})
        request['endpoint_checkpoint_stat']=ode.stat_identity(checkpoint_path(arm,endpoint))
        if kind in ('eval','validate'):
            from training.studio import balanced_lora_pair_eval as baseline
            case=args.get('case_id') if kind=='eval' else 'pair-j'
            require(case in ('pair-j','holdout-l'),'fixed two repeated development pairs required')
            request['baseline_binding']=baseline.baseline_records(case)[2]
    require(not torch.cuda.is_initialized(),'DMD CPU preparation initialized CUDA')
    write_new(directory/'request.json',request)
    return directory/'request.json'


def _load_parent(parent):
    require(ode_parent_binding(parent['job_directory'])==parent,'actual formalODE80 parent changed')
    request=read(Path(parent['job_directory'])/'request.json');args=request['resolved_arguments']
    require(isinstance(args,dict) and isinstance(args.get('cache_binding'),dict)
        and isinstance(args.get('parent'),dict),'actual resolved ODE cache/parent bindings required')
    config=ode.load_config(parent['config'])
    dataset=ode.ODETrajectoryDataset(args['cache_binding'],args['parent'])
    state=torch.load(parent['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    ode.validate_checkpoint(state,config=config,hashes=ode.artifact_hashes(config,parent['config'],args['parent'],args['cache_binding']),
        parent=args['parent'],sampling=ode.sampling_contract(dataset,args['parent']))
    require(state['step']==80,'DMD cannot start from incomplete/pilotODE')
    # This generator inherits the actual expanded80 action parameters. Real
    # zero-B LoRA bypass is functionally the original teacher, not ODE weights.
    expanded=previous.load_parent(state['parent_lineage']['ancestor_lineage'])
    expected=expanded.get('trainable_model',expanded.get('frozen_action_model'))
    require(isinstance(expected,dict) and all(n in expected and torch.equal(v,expected[n])
        for n,v in state['frozen_action_model'].items()),'actual frozen real expanded80 action branch differs')
    return config,state


def _first_only(dataset,slot):
    """mmap exact cached first latent; never slice/copy GT future into training."""
    row=dataset.episodes[slot];dataset._check_stat(row)
    payload=torch.load(row['_path'],map_location='cpu',weights_only=True,mmap=True)
    offset=row['_offset']
    require(payload.get('schema_version')==1 and payload.get('episode_id')==row['episode_id']
        and int(payload['source_starts'][offset])==row['source_start']
        and 'y' not in payload and 'clip_fea' not in payload,'exact old-cache identity/offset/conditioning changed')
    first=payload['clean_latents'][offset,:1].detach().clone().bfloat16().unsqueeze(0)
    actions=payload['actions'][offset].detach().clone().float().unsqueeze(0)
    positive=payload['prompt_embeds'].detach().clone().bfloat16()
    require(torch.equal(actions[0],torch.tensor(row['future_actions_96x8'],dtype=torch.float32)),
        'actual recorded selected actions changed')
    finite=(bool(torch.isfinite(t).all()) for t in (first,actions,positive))
    require(all(finite) and first.shape==(1,1,48,30,52) and positive.ndim==2 and positive.shape[1]==4096,
        'real first/action/static text invalid')
    dataset._check_stat(row)
    return first,positive,core.repeat_actions(actions),dict(
        window_slot=slot,**{k:row[k] for k in ('episode_id','source_start','window_index','cache_origin')},
        future_clean_latents_read=False,future_clean_latents_supplied=False)


def _save_checkpoint(arm,step,*,banks,frozen,signature,parent,config,metrics,profile):
    target=checkpoint_path(arm,step);target.parent.mkdir(parents=True,exist_ok=True)
    require(not target.exists(),'never overwrite successful or failed DMD checkpoint identity')
    value=dict(format_version=2,stage=STAGE,checkpoint_schema=CHECKPOINT_SCHEMA,arm=arm,
        parent_binding=parent,config=config.to_dict(),method_contract=core.DMDConfig(arm).record(),
        source_hashes=source_hashes(),model_signature=signature,frozen_action_model=core.cpu_tree(frozen),
        bank_state=banks.state_dict(),**counters(step,parent),metrics=metrics,actual_profile=profile,
        torch_rng_state=torch.get_rng_state(),cuda_rng_state_all=torch.cuda.get_rng_state_all(),
        python_rng_state=random.getstate(),numpy_rng_state=np.random.get_state())
    temp=target.with_suffix('.partial.pt');require(not temp.exists(),'partial checkpoint preserved; no overwrite')
    torch.save(value,temp);temp.replace(target);return target


def _validate_checkpoint(state,*,arm,parent,config,step,banks,expected_frozen,expected_signature):
    expected=dict(stage=STAGE,checkpoint_schema=CHECKPOINT_SCHEMA,arm=arm,parent_binding=parent,
        config=config.to_dict(),method_contract=core.DMDConfig(arm).record(),source_hashes=source_hashes())
    require(all(state.get(k)==v for k,v in expected.items()) and all(state.get(k)==v for k,v in counters(step,parent).items()),
        'strict same-arm DMD model/optimizer/RNG/cursor/source contract differs')
    previous.validate_frozen_action(state['frozen_action_model'])
    require(previous.master.equal_tree(state['frozen_action_model'],expected_frozen,lambda:None)
        and state['model_signature']==expected_signature,'strict DMD frozen action/LoRA mapping differs from actual parent model')
    banks.load_state_dict(state['bank_state'])
    for role,native_step in (('generator',step),('fake',step+2)):
        previous.optimizer_mapping(state['model_signature'],banks.optimizer_states[role],native_step)
    ode.tf.validate_rng(state)
    profile=state.get('actual_profile',{})
    require(profile.get('native_kv_parity',{}).get('status')=='passed'
        and profile.get('native_replay_causality',{}).get('status')=='passed',
        'actual native context and dense/offload gates required, not CPUfixtures')


def _clear_optimizer(optimizer):
    del optimizer;gc.collect();torch.cuda.empty_cache()


def _train(request,directory,device,check):
    args=request['arguments'];arm=args['arm'];contract=core.DMDConfig(arm)
    start,stop=args['start_step'],args['stop_after_step'];parent=request['actual_parent_binding']
    config,initial=_load_parent(parent)
    model,summary=previous.build_model(config,device,frozen_action_model=initial['frozen_action_model'],
        trainable_model=initial['trainable_model'])
    signature=previous.model_signature(model);frozen=core.cpu_tree(initial['frozen_action_model'])
    banks=core.StateBanks(model,initial['trainable_model']);data=ode.tf.build_dataset(initial['parent_lineage']['qualified_data_binding'])
    require(len(data.episodes)==60,'DMD same sixty reviewed windows required')
    from training.studio.quality_probe import load_negative,DEFAULT_NEGATIVE_CACHE
    negative,_=load_negative(DEFAULT_NEGATIVE_CACHE,config.model.base_model_path);negative=negative.to(device).bfloat16()
    del initial
    callbacks=native.WanCallbacks(model,config,banks,check=check)
    profile={};latest={};warmup_completed=start>0
    if start:
        path=checkpoint_path(arm,start)
        require(ode.stat_identity(path)==request['resume_checkpoint_stat'],'DMD predecessor changed')
        state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        _validate_checkpoint(state,arm=arm,parent=parent,config=config,step=start,banks=banks,
            expected_frozen=frozen,expected_signature=signature)
        profile=state['actual_profile'];previous.teacher._restore_rng_state(state,require_complete=True)
        del state
    else:previous.seed_everything(42)
    model.eval() # dropout0; gradients remain enabled only in explicit phases.
    frozen_versions={n:p._version for n,p in model.named_parameters() if n not in banks.names}
    output=arm_path(arm)/'training';output.mkdir(parents=True,exist_ok=True)
    total_rollout_forwards=0
    for step in range(start+1,stop+1):
        tick=time.monotonic();check();draw=sampling_spec(step)
        first,positive,actions,source=_first_only(data,draw['window_slot'])
        first,positive,actions=(x.to(device) for x in (first,positive,actions))
        banks.activate('generator')
        if start==0 and step==1:
            profile['native_kv_parity']=native.probe_kv_offload_parity(model,config,first,positive,actions,check=check)
        backend=native.PositiveHistory(model,config,capacity=61,check=check,cpu_layer_offload=True)
        try:
            sampled=core.own_rollout(backend,first=first,actions=actions,positive=positive,
                seed=draw['rollout_seed'],config=contract)
            video=sampled['latents'].detach();total_rollout_forwards+=sampled['total_forwards']
            transfers=dict(bytes=backend.host_transfer_bytes,seconds=backend.host_transfer_seconds)
        finally:backend.close()
        if start==0 and step==1:
            banks.activate('generator')
            profile['native_replay_causality']=native.probe_native_replay_causality(model,config,video,actions,positive,check=check)
        fake_updates=[0,1,2] if not warmup_completed else [draw['fake_score_update']]
        fake_records=[]
        for fake_update in fake_updates:
            optimizer=native.optimizer_phase(banks,'fake',make_optimizer=previous.make_optimizer,check=check)
            before=previous.sampled_lora(model);banks.begin_graph('fake')
            saved=torch.autograd.graph.save_on_cpu(pin_memory=True)
            with saved,torch.autocast('cuda',dtype=torch.bfloat16):
                loss,fake_record=core.fake_flow_loss(callbacks.score,video=video,actions=actions,positive=positive,
                    config=contract,seed=42,update=fake_update)
            loss.backward();task=native.gradient_statistics(model,banks.names)
            norm=torch.nn.utils.clip_grad_norm_([dict(model.named_parameters())[n] for n in banks.names],1.)
            require(bool(torch.isfinite(norm)),'fake gradient clip is nonfinite')
            optimizer.step();banks.end_graph('fake')
            after=previous.sampled_lora(model)
            require(any(not torch.equal(before[n],after[n]) for n in before),'fake optimizer did not actually update LoRA')
            previous.optimizer_mapping(signature,optimizer.state_dict(),fake_update+1)
            banks.capture('fake',optimizer);fake_records.append(dict(loss=float(loss.detach()),fake_update=fake_update,**task,**fake_record))
            fake_native=copy.deepcopy(banks.optimizer_states['fake']);optimizer.load_state_dict(fake_native)
            require(previous.master.equal_tree(fake_native,optimizer.state_dict(),check),'native fake optimizer roundtrip differs')
            optimizer.zero_grad(set_to_none=True);del optimizer,loss;gc.collect();torch.cuda.empty_cache()
        warmup_completed=True
        gradient,score_record=core.distribution_gradient(callbacks.score,video=video,actions=actions,
            positive=positive,negative=negative,config=contract,seed=42,update=step-1)
        optimizer=native.optimizer_phase(banks,'generator',make_optimizer=previous.make_optimizer,check=check)
        before=previous.sampled_lora(model);banks.begin_graph('generator')
        with torch.autograd.graph.save_on_cpu(pin_memory=True),torch.autocast('cuda',dtype=torch.bfloat16):
            replay,replay_record=core.replay_last_block(callbacks.replay,video=video,actions=actions,
                positive=positive,seed=draw['rollout_seed'],config=contract)
            relative=float((replay.detach().float()-video[:,-3:].float()).square().mean().sqrt()
                /video[:,-3:].float().square().mean().sqrt().clamp_min(1e-6))
            require(relative<=.03,'packed final-block replay differs from actual KV rollout by more than3% RMS')
            loss=core.generator_surrogate(replay,gradient,config=contract)
        loss.backward();task=native.gradient_statistics(model,banks.names)
        norm=torch.nn.utils.clip_grad_norm_([dict(model.named_parameters())[n] for n in banks.names],1.)
        require(bool(torch.isfinite(norm)),'generator gradient clip is nonfinite')
        optimizer.step();banks.end_graph('generator')
        after=previous.sampled_lora(model)
        require(any(not torch.equal(before[n],after[n]) for n in before),'generator optimizer did not actually update LoRA')
        previous.optimizer_mapping(signature,optimizer.state_dict(),step)
        banks.capture('generator',optimizer);optimizer.zero_grad(set_to_none=True)
        native_state=copy.deepcopy(banks.optimizer_states['generator'])
        optimizer.load_state_dict(native_state)
        require(previous.master.equal_tree(native_state,optimizer.state_dict(),check),'native generator optimizer roundtrip differs')
        del optimizer;gc.collect();torch.cuda.empty_cache()
        require(all(not p.requires_grad and p.grad is None and p._version==frozen_versions[n]
            for n,p in model.named_parameters() if n in frozen_versions),'shared frozen Wan/Adapter/gate changed')
        peak=torch.cuda.max_memory_reserved(device);require(peak<23*1024**3,'actual DMD full replay exceeds23GiB')
        latest=dict(optimizer_step_seconds=time.monotonic()-tick,generator_loss=float(loss.detach()),
            generator_gradient=task,gradient_rms=float(gradient.square().mean().sqrt()),fake_training=fake_records,
            sampling=draw,source_input=source,rollout_forwards=sampled['total_forwards'],
            total_rollout_forwards_this_job=total_rollout_forwards,score=score_record,replay=replay_record,
            replay_vs_KV_relative_RMS=relative,CPU_KV_transfer=transfers,peak_reserved_bytes=peak,
            actual_callback_counts=copy.deepcopy(callbacks.counts),native_optimizer_restore_matched=True,
            fake_native_optimizer_restore_matched=True,fake_warmup_reuses_first_own_trajectory=True,
            shared_backbone_frozen=True,fake_teacher_unchanged_by_generator=True,quality_verified=False)
        previous.append_jsonl(output/'metrics.jsonl',dict(**counters(step,parent),**latest))
        print(json.dumps(dict(step=step,arm=arm,optimizer_step_seconds=latest['optimizer_step_seconds'],
            peak_reserved_bytes=peak,generator_loss=latest['generator_loss'])),flush=True)
        del loss,replay,gradient,video,sampled
        check()
    path=_save_checkpoint(arm,stop,banks=banks,frozen=frozen,signature=signature,parent=parent,
        config=config,metrics=latest,profile=profile)
    model.to('cpu');del model;gc.collect();torch.cuda.empty_cache()
    return dict(stage=STAGE,checkpoint_schema=CHECKPOINT_SCHEMA,arm=arm,**counters(stop,parent),
        checkpoint_path=str(path),method_contract=contract.record(),actual_profile=profile,**latest,
        pipeline_stage_executed=True,quality_verified=False,realtime_verified=False,
        training_length_adequacy_proven=False)


def _endpoint_model(request,device,check):
    args=request['arguments'];arm=args['arm'];step=args['endpoint_step'];parent=request['actual_parent_binding']
    path=checkpoint_path(arm,step)
    require(ode.stat_identity(path)==request['endpoint_checkpoint_stat'],'same-arm actualDMD endpoint changed')
    config,initial=_load_parent(parent);check()
    model,_=previous.build_model(config,device,frozen_action_model=initial['frozen_action_model'],trainable_model=initial['trainable_model'])
    banks=core.StateBanks(model,initial['trainable_model'])
    state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    _validate_checkpoint(state,arm=arm,parent=parent,config=config,step=step,banks=banks,
        expected_frozen=initial['frozen_action_model'],expected_signature=previous.model_signature(model))
    banks.activate('generator');model.eval();del initial,state
    return model,banks,config


def _generation_inputs(request,condition,case=None):
    from training.completion_ode_jobs_v2 import _generation_inputs as original
    return original(case or request['arguments'].get('case_id','pair-j'),request['baseline_binding'],condition)


@torch.no_grad()
def _validate(request,directory,device,check):
    """Fixed self-rollout score diagnostics; no parameter or GT update.

    DMD score norm/critic MSE are numeric diagnostics, not image-quality scores
    or proof of successful distribution matching. Same development pair-j,
    seed42 and own20-block rollout for both arms/all formal lengths.
    """
    model,banks,config=_endpoint_model(request,device,check)
    first,positive,original,_,negative,lineage=_generation_inputs(request,'original','pair-j')
    first=first.to(device).bfloat16();positive=positive.to(device).bfloat16();negative=negative.to(device).bfloat16()
    actions=core.repeat_actions(torch.as_tensor(original).float().unsqueeze(0)).to(device)
    backend=native.PositiveHistory(model,config,capacity=61,check=check,cpu_layer_offload=True)
    try:
        sampled=core.own_rollout(backend,first=first,actions=actions,positive=positive,seed=42,
            config=core.DMDConfig(request['arguments']['arm']))
        transfers=dict(bytes=backend.host_transfer_bytes,seconds=backend.host_transfer_seconds)
    finally:backend.close()
    callbacks=native.WanCallbacks(model,config,banks,check=check)
    gradient,scores=core.distribution_gradient(callbacks.score,video=sampled['latents'],actions=actions,
        positive=positive,negative=negative,config=core.DMDConfig(request['arguments']['arm']),seed=170071,update=0)
    banks.activate('fake')
    def inference_score(**kwargs):return callbacks.score(**dict(kwargs,gradient_enabled=False))
    fake_loss,fake_record=core.fake_flow_loss(inference_score,video=sampled['latents'],actions=actions,
        positive=positive,config=core.DMDConfig(request['arguments']['arm']),seed=170071,update=0)
    result=dict(stage=STAGE,arm=request['arguments']['arm'],endpoint_step=request['arguments']['endpoint_step'],
        objective='fixed_self_rollout_DMD_score_diagnostics_not_visual_quality',case_id='pair-j',seed=42,
        generated_future_frames=240,rollout_forwards=101,score=scores,fake_record=fake_record,
        normalized_DMD_gradient_RMS=float(gradient.square().mean().sqrt()),future_fake_flow_MSE=float(fake_loss),
        CPU_KV_transfer=transfers,input_lineage=lineage,no_parameter_updates=True,
        future_clean_latents_read=False,future_clean_latents_supplied=False,
        repeated_development_qualification=True,independent_final_test=False,quality_verified=False)
    model.to('cpu');del model,banks,callbacks,sampled,gradient;gc.collect();torch.cuda.empty_cache()
    return result


@torch.no_grad()
def _evaluate(request,directory,device,check):
    """Two true four-round CFG1 causal videos, z0-only original/JL inputs."""
    from PIL import Image
    from training.studio import completion_causal_eval as engine
    from training.eval.rollout15s import ffmpeg_writer_factory
    from utils.wan_wrapper import WanVAEWrapper
    model,banks,config=_endpoint_model(request,device,check)
    model.requires_grad_(False).eval();generated={};inputs={};rows=[]
    for condition in ('original','jl_swap'):
        check();values=inputs[condition]=_generation_inputs(request,condition)
        first,positive,actions,initial,negative,lineage=values
        target=directory/condition;target.mkdir(exist_ok=False)
        Image.fromarray(initial).save(target/'initial.png');np.save(target/'actions.npy',actions,allow_pickle=False)
        write_new(target/'inputs.json',dict(condition=condition,source_input_lineage=lineage,
            actual_actions=actions.tolist(),seed=42,future_clean_latents_read=False,future_clean_latents_supplied=False))
        first=first.to(device).bfloat16();positive=positive.to(device).bfloat16()
        actual_actions=torch.as_tensor(actions).float().unsqueeze(0).to(device)
        backend=native.PositiveHistory(model,config,capacity=25,check=check,cpu_layer_offload=True)
        tick=time.monotonic()
        try:
            # Same ODE inference noise namespace, four rounds and externalCFG1.
            sample=ode.core().sample_causal_clip(backend,first=first,actions=actual_actions,
                positive_conditions=positive,seed=42,blocks=8)
            latents=sample.pop('latents').cpu()
            generation=dict(sample,actual_model_forward_calls=41,generation_seconds=time.monotonic()-tick,
                cpu_layer_KV_offload=True,CPU_KV_transfer_bytes=backend.host_transfer_bytes,
                CPU_KV_transfer_seconds=backend.host_transfer_seconds,slow_resource_fallback_not_acceleration=True,
                no_future_ground_truth=True,streaming_latency_measured=False)
            generated[condition]=(latents,generation);write_new(target/'latent-generation.json',generation)
        finally:backend.close()
    model.to('cpu');del model,banks;gc.collect();torch.cuda.empty_cache();check()
    vae=WanVAEWrapper(vae_type='Wan2.2_VAE',pretrained_path=str(Path(config.model.base_model_path)/'Wan2.2_VAE.pth'))
    vae.requires_grad_(False).eval().to(device=device,dtype=torch.bfloat16)
    for condition in ('original','jl_swap'):
        check();latents,generation=generated.pop(condition);target=directory/condition
        writer=ffmpeg_writer_factory(target/'raw.mp4',832,480,16)
        decoded=engine.decode_clip(vae,latents,inputs[condition][3],device=device,check=check,writer=writer)
        row=dict(status='completed',arm=request['arguments']['arm'],endpoint_step=request['arguments']['endpoint_step'],
            case_id=request['arguments']['case_id'],condition=condition,source_input_lineage=inputs[condition][5],
            checkpoint_lineage=request['actual_parent_binding'],generation=generation,**decoded,
            actual_model_forward_calls=41,raw_video=str(target/'raw.mp4'),future_clean_latents_read=False,
            future_clean_latents_supplied=False,quality_evaluation='awaiting_actual_visual_review',
            action_correctness_verified=False,quality_verified=False,realtime_verified=False)
        write_new(target/'result.json',row);rows.append(row)
    vae.to('cpu');del vae;gc.collect();torch.cuda.empty_cache()
    return dict(stage=STAGE,arm=request['arguments']['arm'],endpoint_step=request['arguments']['endpoint_step'],
        cases=rows,actual_model_forward_calls=82,phase_order='two_generated_then_release_model_then_decode',
        frames_per_video=97,fps=16,student_sampler='four_endpoint_renoise_CFG1',
        quality_evaluation='awaiting_actual_visual_review',quality_verified=False,realtime_verified=False)


def run_job(request,directory,lease):
    from training.demo.backend import validate_lease,validate_active_authority
    directory=Path(directory)
    require(read(directory/'request.json')==request and request.get('schema')=='completion_dmd_deadline_job_v2'
        and request['source_hashes']==source_hashes(),'DMD request/source changed after CPU preparation')
    require(lease.get('gpu_index')==request['gpu_index'] in request['parallel_budget_binding'].get('allowed_gpu_indices',[1,2]),'only own actual Creator card allowed')
    validate_lease(lease,max_seconds=request['max_seconds'],job_id=request['job_id'],request_sha256=sha(directory/'request.json'))
    spec={k:request[k] for k in ('job_id','mode','group','gpu_index','cleanup_seconds','directory','depends_on','arguments','parallel_budget_binding')}
    spec['max_seconds']=request['contract_max_seconds']
    manifest=_manifest(spec);accounting,ledger=_budget(request,lease=lease)
    require(resolve_parent(spec,manifest,ledger=ledger)==request['actual_parent_binding'],'actual settledODE80 source changed')
    kind=request['mode'].removeprefix('completion_dmd_v2_')
    if kind in ('profile','train'):
        maximum,derived=_actual_timeout(spec,manifest,ledger)
        require(maximum==request['max_seconds'] and derived==request['derived_timeout'],
            'actual settled timing/native gates changed after CPU preparation')
    else:
        producer=request['endpoint_producer']
        require(producer['job_id'] in spec['depends_on'],'exact endpoint producer dependency changed')
        _settled(producer['directory'],ledger=ledger)
        require(all(sha(pin['path'])==pin['sha256'] for pin in producer['records'].values()),
            'settled endpoint producer records changed after CPU preparation')
    require(os.environ.get('HF_HUB_OFFLINE')==os.environ.get('TRANSFORMERS_OFFLINE')=='1' and torch.cuda.is_available(),
        'real offline GPU worker required, CPUfixture is not a substitute')
    began=time.monotonic()
    def check():
        require(time.monotonic()-began<request['max_seconds']-30,'finite owned DMD deadline reached')
        validate_active_authority(lease)
    device=torch.device('cuda',0);torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    if kind in ('profile','train'):result=_train(request,directory,device,check)
    elif kind=='validate':result=_validate(request,directory,device,check)
    else:result=_evaluate(request,directory,device,check)
    return dict(status='completed',execution=result,accounting_at_start=accounting,
        elapsed_seconds=time.monotonic()-began,quality_verified=False,realtime_verified=False)
