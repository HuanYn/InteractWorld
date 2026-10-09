"""New CPU preparation / leased execution API for the deadline coordinator.

No production registry or old TF file is changed. Preparation performs all known
CPU dependencies before reserve. Every execution reconciles the original ledger,
including the other explicitly owned card; it never ignores a foreign active job.
"""
from __future__ import annotations

import copy
import gc
import json
import math
import os
from pathlib import Path
import re
import time

import torch
from torch.utils.data import default_collate

import train_completion_ode_sufficiency as entry
from training import completion_ode_cache_v2 as cache

require = entry.require
MODES = tuple('completion_ode_v2_'+kind for kind in ('probe','cache','merge','train','validate','eval'))
COORDINATOR_ROOT = entry.ROOT.parent/'deadline30h-v1'
GROUP_CAPS = dict(ode=57600, dmd=57600, runtime=21600, evaluation=14400)


def parallel_manifest(binding):
    from training import completion_parallel_v4 as v4
    if v4.is_v4(binding):return v4.manifest(binding)
    fields = {'schema','manifest','owned_job_ids','ledger','allowed_gpu_indices','max_owned_active'}
    require(isinstance(binding, dict) and set(binding) == fields
        and binding['schema'] == 'creator_deadline30h_parallel_binding_v1'
        and binding['allowed_gpu_indices'] == [1,2] and binding['max_owned_active'] == 2,
        'explicit immutable two-card ownership binding required')
    require(Path(binding['manifest']['path']) == COORDINATOR_ROOT/'manifest.json', 'fixed deadline manifest only')
    manifest = entry.small_pin(binding['manifest'])
    from training.studio.controlled_action_train import RUNTIME
    require(manifest.get('schema') == 'creator_deadline30h_parallel_manifest_v1'
        and manifest.get('fixed_plan_complete') is True and manifest.get('coordinator_root') == str(COORDINATOR_ROOT)
        and manifest.get('ledger') == binding['ledger'] == str(RUNTIME/'budget.json')
        and manifest.get('allowed_gpu_indices') == [1,2] and manifest.get('max_owned_active') == 2
        and manifest.get('global_cap_gpu_seconds') == 360000 and manifest.get('server_cap_bytes') == 500000000000
        and manifest.get('handoff_buffer_seconds') == 3600,
        'unchanged100h/500GB/original ledger and fixed complete plan required')
    jobs = manifest.get('jobs'); groups = manifest.get('groups'); envelopes = manifest.get('future_envelopes_gpu_seconds')
    require(isinstance(jobs,list) and jobs and isinstance(groups,dict) and set(groups) == set(GROUP_CAPS)
        and isinstance(envelopes,dict) and set(envelopes) == set(GROUP_CAPS)
        and sorted(row['job_id'] for row in jobs) == binding['owned_job_ids']
        and len(set(binding['owned_job_ids'])) == len(jobs), 'all owned job identities and stage envelopes required')
    for group,cap in GROUP_CAPS.items():
        require(groups[group].get('cap_gpu_seconds') == cap and type(envelopes[group]) is int
            and 0 <= envelopes[group] <= cap, 'original independent subcaps cannot change')
    require(Path(groups['ode']['accounting_root']) == entry.ROOT
        and envelopes['runtime'] == 21600 and envelopes['evaluation'] == 14400,
        'full unimplemented runtime/supplementary evaluations remain funded')
    roots = {g:Path(row['accounting_root']) for g,row in groups.items()}
    require(all(roots[a] != roots[b] and roots[a] not in roots[b].parents and roots[b] not in roots[a].parents
        for i,a in enumerate(roots) for b in list(roots)[i+1:]), 'nonoverlapping accounting roots required')
    for row in jobs:
        require(row.get('group') in groups and row.get('gpu_index') in (1,2)
            and type(row.get('max_seconds')) is int and 1 <= row['max_seconds'] <= 3600
            and row.get('cleanup_seconds') == 60 and isinstance(row.get('arguments'),dict)
            and Path(row['directory']).is_relative_to(roots[row['group']])
            and Path(row['directory']).name == row['job_id'], 'bounded owned job/card/directory required')
    return manifest


def reconcile_parallel_budget(binding, *, ledger, lease, max_seconds, settled_stage_gpu_seconds=None):
    """All real paid + all real active + complete pending ceilings, exactly once."""
    from training import completion_parallel_v4 as v4
    if v4.is_v4(binding):return v4.reconcile(binding,ledger=ledger,lease=lease,max_seconds=max_seconds,
        settled_stage_gpu_seconds=settled_stage_gpu_seconds)
    manifest = parallel_manifest(binding)
    require(ledger.get('version') == 1 and isinstance(ledger.get('reservations'),dict)
        and 0 <= ledger.get('storage_bytes_upper', math.inf) <= 500000000000, 'actual original ledger required')
    jobs = {row['job_id']:row for row in manifest['jobs']}
    roots = {g:Path(row['accounting_root']).resolve() for g,row in manifest['groups'].items()}
    def group_of(directory):
        directory = Path(directory).resolve()
        matches = [g for g,r in roots.items() if directory == r or r in directory.parents]
        require(len(matches) <= 1, 'ambiguous actual stage accounting')
        return matches[0] if matches else None
    paid, reserved = 0., 0.
    group_paid, group_reserved = {g:0. for g in roots}, {g:0. for g in roots}
    seen, active = {}, []
    for rid,record in ledger['reservations'].items():
        context = record.get('context', {}); identity = context.get('job_id')
        require(isinstance(identity,str) and identity not in seen and record.get('status') in ('reserved','settled'),
            'invalid/duplicate real global job record')
        seen[identity] = record; group = group_of(context['job_directory'])
        if record['status'] == 'settled':
            amount = record.get('charged_seconds')
            require(type(amount) in (int,float) and math.isfinite(amount) and amount >= 0, 'invalid historical fee')
            paid += amount
            if group: group_paid[group] += amount
            if identity in jobs:
                require(record.get('settlement',{}).get('outcome') == 'completed', 'failed identity is terminal, not retryable')
        else:
            require(identity in jobs and record.get('lease',{}).get('reservation_id') == rid,
                'foreign/unowned active Creator reservation blocks launch')
            row = jobs[identity]; actual_lease = record['lease']
            path = Path(row['directory'])
            actual_request = entry.read(path/'request.json')
            actual_max = actual_request.get('max_seconds')
            require(group == row['group'] and Path(context['job_directory']).resolve() == path.resolve()
                and actual_request.get('parallel_budget_binding') == binding
                and actual_request.get('contract_max_seconds', actual_max) == row['max_seconds']
                and type(actual_max) is int and 1 <= actual_max <= row['max_seconds']
                and context.get('max_seconds') == actual_max and record.get('reserved_seconds') == actual_max+60
                and context.get('request_sha256') == entry.previous.sha256_file(path/'request.json')
                and actual_lease.get('gpu_index') == row['gpu_index']
                and actual_lease == entry.read(path/'lease.json'), 'actual owned lease/request/card binding differs')
            active.append((rid,record)); reserved += actual_max+60; group_reserved[group] += actual_max+60
    require(1 <= len(active) <= 2 and len({r['lease']['gpu_index'] for _,r in active}) == len(active)
        and len({r['lease']['gpu_uuid'] for _,r in active}) == len(active), 'at most2 distinct owned physical cards required')
    require(any(rid == lease.get('reservation_id') and record['lease'] == lease
        and record['reserved_seconds'] == max_seconds+60 for rid,record in active), 'current real lease missing')
    pending = {g:manifest['future_envelopes_gpu_seconds'][g] for g in roots}
    for identity,row in jobs.items():
        if identity not in seen: pending[row['group']] += row['max_seconds']+60
    for group,cap in GROUP_CAPS.items():
        require(group_paid[group]+group_reserved[group]+pending[group] <= cap, 'full remaining '+group+' exceeds original subcap')
    require(paid+reserved+sum(pending.values()) <= 360000, 'actual history/all active/full remaining exceeds100GPUh')
    if settled_stage_gpu_seconds is not None:
        require(group_paid['ode'] == settled_stage_gpu_seconds, 'ODE settled fees were reset or omitted')
    return dict(global_paid_gpu_seconds=paid, all_active_reserved_gpu_seconds=reserved,
        remaining_worst_gpu_seconds=sum(pending.values()), group_paid_gpu_seconds=group_paid,
        group_active_gpu_seconds=group_reserved, group_pending_gpu_seconds=pending,
        global_upper_gpu_seconds=paid+reserved+sum(pending.values()))


def _config(parent, binding):
    path = entry.ROOT/'config.yaml'
    config = entry.config_from_parent(entry.tf.load_config(parent['parent_config']), binding)
    if path.exists(): require(entry.load_config(path).to_dict() == config.to_dict(), 'immutableODE config changed')
    else: cache.write_new(path, config.to_dict())  # JSON is valid YAML; parser preserves the typed contract.
    return config, path


def _endpoint(arguments, *, parent, binding):
    step = arguments.get('endpoint_step')
    require(step in (0,20,40,80), 'fixed untrained/20/40/80 endpoint only')
    if step == 0:
        require(arguments.get('endpoint_profile') is None, 'untrained control cannot pretend a trained endpoint')
        return None
    result = entry.actual_profile(arguments.get('endpoint_profile'), kind='trajectory')
    path = entry.ROOT/'training/checkpoints'/f'step-{step:07d}.pt'
    require(result.get('stage') == entry.STAGE and result.get('step') == step
        and result.get('checkpoint_path') == str(path) and result.get('sampling_cursor') == step*8,
        'real completed/settled requestedODE endpoint required')
    config, config_path = _config(parent,binding)
    dataset = entry.ODETrajectoryDataset(binding,parent)
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    entry.validate_checkpoint(payload, config=config, hashes=entry.artifact_hashes(config,config_path,parent,binding),
        parent=parent, sampling=entry.sampling_contract(dataset,parent))
    return dict(path=str(path), stat=entry.stat_identity(path), source_records=arguments['endpoint_profile'])


def _generation_inputs(case_id, record_binding, condition):
    """Only old audited z0-only inference loader, never SourcePool/probeGT."""
    from training.studio import balanced_lora_pair_eval as baseline
    original,rows,actual = baseline.baseline_records(case_id,record_binding)
    require(actual == record_binding, 'original same-input baseline records changed')
    values = baseline.load_inputs(original,condition)
    require(values[5] == rows[condition]['lineage'] and values[5]['future_clean_latents_read'] is False
        and values[5]['future_clean_latents_supplied'] is False, 'generator may only receivez0 and own history')
    return values


def _ancestors(manifest, job_id):
    jobs={row['job_id']:row for row in manifest['jobs']}
    found=set(); pending=list(jobs[job_id]['depends_on'])
    while pending:
        identity=pending.pop()
        require(identity in jobs and identity != job_id, 'known acyclic producer dependency required')
        if identity not in found: found.add(identity); pending.extend(jobs[identity]['depends_on'])
    return found


def _producer(manifest, current, identity, *, kinds):
    """Resolve only declared transitive producers AFTER their real settlement."""
    require(identity in _ancestors(manifest,current), 'actual dependency must be a fixed manifest predecessor')
    producer=next(row for row in manifest['jobs'] if row['job_id'] == identity)
    require(producer['mode'] in ['completion_ode_v2_'+kind for kind in kinds], 'wrong declaredODE producer kind')
    directory=Path(producer['directory'])
    binding={key:cache.pin(directory/(key+'.json')) for key in ('request','result','settlement')}
    result=entry.actual_profile(binding,kind='trajectory')
    request=entry.small_pin(binding['request'])
    require(request.get('job_id') == identity and request.get('mode') == producer['mode']
        and result.get('stage') == entry.STAGE, 'actual fixed producer identity/stage differs')
    if producer['mode'] in ('completion_ode_v2_probe','completion_ode_v2_cache'):
        require(result.get('slots') == producer['arguments'].get('slots'), 'actual producer did not complete its declared slots')
    if producer['mode'] == 'completion_ode_v2_train':
        require(result.get('step') == producer['arguments'].get('stop_after_step'), 'actual strict predecessor step differs')
    return binding,result


def _comparison_identities(row):
    if row['mode']=='completion_service_v3_endpoint':
        from training.completion_parallel_v4 import comparison_identities
        return comparison_identities(row)
    mode,args=row['mode'],row['arguments']; step=args.get('endpoint_step')
    if mode == 'completion_ode_v2_validate':
        require(step in entry.ENDPOINTS, 'fixed three numeric endpoints only')
        return 'numeric_validation',[f'ode{step}-numeric-probe4-level4']
    require(mode == 'completion_ode_v2_eval' and step in (0,*entry.ENDPOINTS)
        and args.get('case_id') in ('pair-j','holdout-l'), 'fixed completeODE comparisons required')
    prefix='untrained' if step == 0 else f'ode{step}'
    return ('untrained_control' if step == 0 else 'video_evaluation'),[
        prefix+'-'+args['case_id']+'-'+condition for condition in ('original','jl_swap')]


def _budget_from_manifest(spec,manifest,arguments,actual_seconds,directory):
    """Derive real-history/current/full-future training contract, not future pins."""
    from training.studio.completion_causal_sufficiency_train import ledger_snapshot
    ledger,ledger_sha=ledger_snapshot(); records={r['context']['job_id']:r for r in ledger['reservations'].values()}
    settled=sum(r['charged_seconds'] for r in ledger['reservations'].values()
        if r['status'] == 'settled' and Path(r['context']['job_directory']).is_relative_to(entry.ROOT))
    rows,completed=[],[]
    for row in manifest['jobs']:
        if row['group'] != 'ode' or row['job_id'] == spec['job_id']: continue
        record=records.get(row['job_id']); kind=row['mode'].removeprefix('completion_ode_v2_')
        if kind in ('probe','cache','merge'):
            require(record is not None and record['status'] == 'settled'
                and record['settlement']['outcome'] == 'completed', 'all fixed cache producers must finish beforeODE training')
            continue
        if kind == 'train':
            if record is not None and record['status'] == 'settled': continue
            start,stop=row['arguments'].get('start_step'),row['arguments'].get('stop_after_step')
            require(type(start) is int and start >= arguments['stop_after_step'], 'unresolved earlier strict training job')
            rows.append(dict(kind='training',start_step=start,stop_after_step=stop,
                max_seconds=row['max_seconds'],cleanup_seconds=60))
        else:
            comparison_kind,identities=_comparison_identities(row)
            if record is not None and record['status'] == 'settled':
                require(record['settlement']['outcome'] == 'completed', 'failed fixed comparison is terminal')
                binding={key:cache.pin(Path(row['directory'])/(key+'.json')) for key in ('request','result','settlement')}
                completed.append(dict(kind=comparison_kind,comparison_identities=identities,records=binding))
            else:
                rows.append(dict(kind=comparison_kind,comparison_identities=identities,
                    max_seconds=row['max_seconds'],cleanup_seconds=60))
    schedule=dict(schema='completion_ode_complete_remaining_schedule_v2',root=str(entry.ROOT),final_optimizer_step=80,
        fixed_training_comparison_endpoints=[20,40,80],fixed_cache_trajectory_count=64,fixed_probe_count=4,
        current_start_step=arguments['start_step'],current_stop_step=arguments['stop_after_step'],
        remaining_jobs=rows,completed_evaluations=completed)
    schedule_pin=cache.write_new(directory/'remaining-schedule.json',schedule)
    value=dict(scope='completion_ode_cache_train_validation_eval_combined',stage_cap_gpu_seconds=57600,
        settled_gpu_seconds=settled,remaining_planned_gpu_seconds=sum(row['max_seconds']+60 for row in rows),
        current_job_cleanup_seconds=60,ledger_sha256=ledger_sha,remaining_schedule=schedule_pin,
        parallel_budget_binding=spec['parallel_budget_binding'])
    entry.validate_budget_contract(value,actual_seconds,current_start_step=arguments['start_step'],
        current_stop_step=arguments['stop_after_step'])
    return value


def _resolve_arguments(spec,manifest,directory):
    """Manifest arguments stay immutable; actual late-bound pins live separately."""
    arguments=copy.deepcopy(spec['arguments']); current=spec['job_id']
    mappings=dict(first_trajectory_profile_from_job_id=('first_trajectory_profile',('probe',)),
        profile_binding_from_job_id=('profile_binding',('train',)),
        endpoint_profile_from_job_id=('endpoint_profile',('train',)))
    for name,(target,kinds) in mappings.items():
        if name in arguments:
            require(target not in arguments, 'do not combine real and deferred producer bindings')
            arguments[target]=_producer(manifest,current,arguments.pop(name),kinds=kinds)[0]
    name='cache_job_profiles_from_job_ids'
    if name in arguments:
        require('cache_job_profiles' not in arguments, 'ambiguous cache producer bindings')
        identities=arguments.pop(name)
        require(isinstance(identities,list) and len(set(identities)) == len(identities), 'unique cache producer IDs required')
        arguments['cache_job_profiles']=[_producer(manifest,current,i,kinds=('cache',))[0] for i in identities]
    if 'cache_binding_path' in arguments:
        path=Path(arguments.pop('cache_binding_path'))
        require(path == entry.ROOT/'cache/binding.json' and 'cache_binding' not in arguments, 'fixed derived cache binding only')
        if not path.exists():
            merge=arguments.pop('cache_merge_from_jobs',None)
            require(isinstance(merge,dict) and set(merge) == {'first_trajectory_job_id','cache_job_ids'},
                'first train CPU preparation must explicitly declare all64 settled cache producers')
            first=_producer(manifest,current,merge['first_trajectory_job_id'],kinds=('probe',))[0]
            others=[_producer(manifest,current,i,kinds=('cache',))[0] for i in merge['cache_job_ids']]
            cache.merge_cache(arguments['cache_manifest'],first_trajectory_profile=first,cache_job_profiles=others)
        arguments.pop('cache_merge_from_jobs',None)
        arguments['cache_binding']=entry.read(path)
    if 'measured_profiles_from_job_ids' in arguments:
        identities=arguments.pop('measured_profiles_from_job_ids')
        require(isinstance(identities,list) and len(identities) == 1, 'latest strict predecessor metrics include all earlier timings')
        binding,execution=_producer(manifest,current,identities[0],kinds=('train',))
        require(execution['step'] == arguments.get('start_step'), 'actual timing endpoint must equal strict start')
        metrics_path=entry.ROOT/'training/metrics.jsonl'
        metrics=[json.loads(line) for line in metrics_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        require([row.get('step') for row in metrics] == list(range(1,execution['step']+1))
            and all(type(row.get('optimizer_step_seconds')) in (int,float)
                and math.isfinite(row['optimizer_step_seconds']) and row['optimizer_step_seconds'] > 0 for row in metrics),
            'actual contiguous strictODE timings required')
        arguments['measured_slowest_step_seconds']=max(row['optimizer_step_seconds'] for row in metrics)
        arguments['measured_profiles']=[dict(job_id=identities[0],**binding,metrics=cache.pin(metrics_path))]
    return arguments


def prepare_job(spec, directory):
    """Pure CPU, directory already exists from the queue; no lease or model."""
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink() and directory.is_relative_to(entry.ROOT)
        and directory.name == spec.get('job_id') and not (directory/'request.json').exists()
        and re.fullmatch(r'd30-ode-(probe|cache|merge|train|validate|eval)-[A-Za-z0-9_-]+-v1', spec.get('job_id',''))
        and spec.get('mode') in MODES and not torch.cuda.is_initialized(), 'new exactCPU preparedODE job only')
    manifest = parallel_manifest(spec.get('parallel_budget_binding'))
    wanted = next((row for row in manifest['jobs'] if row['job_id'] == spec['job_id']),None)
    require(wanted is not None and all(spec.get(k) == wanted[k] for k in wanted), 'spec differs from immutable owned manifest')
    arguments = _resolve_arguments(spec,manifest,directory); parent = arguments.get('parent')
    require(parent == entry.parent_binding(parent['parent_step']), 'actual completed formalTF parent binding required')
    request = copy.deepcopy(spec)
    request.update(schema='completion_ode_deadline_job_v2', contract_max_seconds=spec['max_seconds'],
        derived_timeout=dict(schema='creator_measured_timeout_v1',kind='fixed',max_seconds=spec['max_seconds']),
        source_hashes=cache.source_hashes())
    kind = spec['mode'].removeprefix('completion_ode_v2_')
    if kind in ('probe','cache','merge'):
        manifest_pin = arguments.get('cache_manifest') or cache.prepare_manifest(parent)
        actual = entry.small_pin(manifest_pin); cache.validate_manifest(actual,parent=parent)
        request['cache_manifest'] = manifest_pin
        if kind == 'probe':
            require(arguments.get('slots') == [0], 'retained first actual packed trajectory profile is exactly slot0')
            cache.original_sources(actual,[0])
        elif kind == 'cache':
            profile = entry.actual_profile(arguments.get('first_trajectory_profile'),kind='trajectory')
            require(profile.get('slots') == [0] and profile.get('causal_gradient_probe_passed') is True
                and profile.get('actual_token_mask_audited') is True and 0 < profile.get('peak_reserved_bytes',0) < 23*1024**3,
                'actual causal/peak/time first trajectory gate required before caching the rest')
            slots = arguments.get('slots')
            require(isinstance(slots,list) and slots and 0 not in slots, 'successful first trajectory must be reused')
            cost = profile['trajectories'][0]['trajectory_seconds']*len(slots)*1.35 + profile['model_load_seconds']*1.35 + 180
            require(cost <= spec['max_seconds'], 'measured cache shard does not fit its fixed ceiling')
            cache.original_sources(actual,slots)
        else:
            # CPU merge is completed before reserve; a coordinator may directly
            # call merge_cache instead and omit this bookkeeping-only GPU job.
            request['merged_cache_binding'] = cache.merge_cache(manifest_pin,
                first_trajectory_profile=arguments.get('first_trajectory_profile'),
                cache_job_profiles=arguments.get('cache_job_profiles'))
    else:
        binding = arguments.get('cache_binding')
        entry.validate_cache_binding(binding,teacher_binding=parent)
        entry.validate_probe_supervision(binding,parent)
        config,config_path = _config(parent,binding)
        request.update(config_path=str(config_path), config_sha256=entry.previous.sha256_file(config_path))
        if kind == 'train':
            start,stop = arguments.get('start_step'),arguments.get('stop_after_step')
            if start == 0:
                require(stop == 2 and spec['max_seconds'] == 1800 and arguments.get('resume') is None,
                    'first realODE2/newoptimizer resource profile required')
                entry.load_parent(parent)  # Full checkpoint/schema/assets known-good before reserving.
            else:
                profile = entry.actual_profile(arguments.get('profile_binding'),kind='training')
                require(profile.get('native_state_restore_matched') is True and profile.get('lora_updated') is True,
                    'actualODE2 full-state/resource/update profile required')
                segment = entry.next_segment(start,arguments.get('measured_slowest_step_seconds'))
                require(stop == segment['stop_after_step'] and segment['max_seconds'] <= spec['max_seconds'],
                    'measured exact strictODE segment required')
                request['max_seconds'] = segment['max_seconds']
                profiles = arguments.get('measured_profiles')
                require(isinstance(profiles,list) and profiles, 'actual settled metrics pins required for derived timeout')
                request['derived_timeout'] = dict(schema='creator_measured_timeout_v1',kind='measured_steps',
                    additional_steps=stop-start,multiplier=1.35,startup_allowance_seconds=180,
                    slowest_step_seconds=arguments['measured_slowest_step_seconds'],profiles=profiles)
            if arguments.pop('budget_from_manifest',False):
                require('budget_contract' not in arguments, 'one real-history resource contract only')
                arguments['budget_contract']=_budget_from_manifest(spec,manifest,arguments,request['max_seconds'],directory)
            entry.validate_budget_contract(arguments.get('budget_contract'),request['max_seconds'],
                current_start_step=start,current_stop_step=stop)
            require(arguments['budget_contract'].get('parallel_budget_binding') == spec['parallel_budget_binding'],
                'training must reconcile the identical owned two-card manifest')
        else:
            request['endpoint_binding'] = _endpoint(arguments,parent=parent,binding=binding)
            if kind == 'validate': require(arguments['endpoint_step'] in entry.ENDPOINTS, 'allthree trained numeric endpoints required')
            else:
                require(arguments.get('case_id') in ('pair-j','holdout-l'), 'fixed two original/JLswap evaluation cases only')
                from training.studio import balanced_lora_pair_eval as baseline
                records = baseline.baseline_records(arguments['case_id'])[2]
                request['baseline_binding'] = records
                values = [_generation_inputs(arguments['case_id'],records,c) for c in ('original','jl_swap')]
                baseline.validate_pair_inputs(*values)
    request['resolved_arguments']=arguments
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA')
    cache.write_new(directory/'request.json',request)
    return directory/'request.json'


def _load_model(request,device,check):
    arguments = request['resolved_arguments']; parent,binding = arguments['parent'],arguments['cache_binding']
    config = entry.load_config(request['config_path'])
    require(entry.previous.sha256_file(request['config_path']) == request['config_sha256'], 'actual config changed')
    endpoint = request.get('endpoint_binding')
    if endpoint is None: state = entry.load_parent(parent)
    else:
        require(entry.stat_identity(endpoint['path']) == endpoint['stat'], 'actualODE endpoint changed')
        _endpoint(arguments,parent=parent,binding=binding)
        state = torch.load(endpoint['path'],map_location='cpu',weights_only=False,mmap=True)
    check(); model,summary = entry.previous.build_model(config,device,frozen_action_model=state['frozen_action_model'],
        trainable_model=state['trainable_model'])
    require(entry.previous.model_signature(model) == state['model_signature'], 'exactFP32 LoRA identity differs')
    model.requires_grad_(False).eval()
    return model,config,summary


def _numeric(request,device,check):
    dataset = entry.ODETrajectoryDataset(request['resolved_arguments']['cache_binding'],request['resolved_arguments']['parent'],purpose='probe')
    started = time.monotonic(); model,config,_ = _load_model(request,device,check)
    load_seconds = time.monotonic()-started; rows=[]
    try:
        for index in range(16):
            check(); batch=default_collate([dataset[index]])
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                loss=entry.endpoint_loss(model,batch,config,device)
            require(bool(torch.isfinite(loss)), 'nonfinite true teacher-endpoint regression metric')
            rows.append(dict(**dataset.sampling_spec(index),endpoint_MSE=float(loss)))
        return dict(stage=entry.STAGE, numeric_rows=rows, actual_model_forward_calls=16,
            model_load_seconds=load_seconds, target='cached_teacher_endpoint_not_GT', no_parameter_update=True,
            teacher_consistency_not_visual_quality=True, comparison_identities=[
                f"ode{request['resolved_arguments']['endpoint_step']}-numeric-probe4-level4"])
    finally: del model


def _videos(request,directory,device,check):
    from training.completion_ode_wan_v2 import WanSingleBranchKV
    from training.studio.completion_causal_eval import decode_clip
    from training.eval.rollout15s import ffmpeg_writer_factory
    from utils.wan_wrapper import WanVAEWrapper
    arguments=request['resolved_arguments']; case=arguments['case_id']; step=arguments['endpoint_step']
    values={c:_generation_inputs(case,request['baseline_binding'],c) for c in ('original','jl_swap')}
    began=time.monotonic(); model,config,_=_load_model(request,device,check)
    load_seconds=time.monotonic()-began; generated={}; rows=[]
    try:
        for condition,inputs in values.items():
            first,text,actions,initial,_,lineage=inputs
            check(); tick=time.monotonic(); backend=WanSingleBranchKV(model,config,capacity=25,check=check)
            try:
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    result=entry.core().sample_causal_clip(backend,
                        first=first.to(device=device,dtype=torch.bfloat16),
                        actions=torch.from_numpy(actions.copy())[None].to(device=device,dtype=torch.bfloat16),
                        positive_conditions=text[None].to(device=device,dtype=torch.bfloat16),seed=42,blocks=8)
                torch.cuda.synchronize(device)
                require(result['denoising_forwards'] == 32 and result['commit_forwards'] == 9
                    and result['total_forwards'] == 41, 'real positive-only32denoise+9commit required')
                generated[condition]=(result['latents'].detach().cpu(),dict(
                    generation_seconds=time.monotonic()-tick,denoising_forwards=32,commit_forwards=9,
                    actual_model_forwards=41,commits=copy.deepcopy(backend.commits)))
            finally: backend.close()
        peak=torch.cuda.max_memory_reserved(device)
        require(peak < 23*1024**3, 'actualKV generation reservedmemory exceeds23GiB')
        model.to('cpu'); del model; model=None; gc.collect(); torch.cuda.empty_cache()
        check(); tick=time.monotonic()
        vae=WanVAEWrapper(vae_type='Wan2.2_VAE',pretrained_path=str(Path(config.model.base_model_path)/'Wan2.2_VAE.pth'))
        vae.requires_grad_(False).eval().to(device=device,dtype=torch.bfloat16)
        vae_load_seconds=time.monotonic()-tick
        try:
            for condition in ('original','jl_swap'):
                check(); target=directory/condition; target.mkdir(exist_ok=False)
                latent,generation=generated.pop(condition)
                writer=ffmpeg_writer_factory(target/'raw.mp4',832,480,16)
                decoded=decode_clip(vae,latent,values[condition][3],device=device,check=check,writer=writer)
                rows.append(dict(condition=condition,case_id=case,raw_video=str(target/'raw.mp4'),
                    source_input_lineage=values[condition][5],generation=generation,**decoded))
        finally: vae.to('cpu'); del vae
        prefix='untrained' if step == 0 else f'ode{step}'
        return dict(stage=entry.STAGE,cases=rows,model_load_seconds=load_seconds,vae_load_seconds=vae_load_seconds,
            actual_model_forward_calls=82,peak_reserved_bytes=max(peak,torch.cuda.max_memory_reserved(device)),
            comparison_identities=[prefix+'-'+case+'-'+c for c in ('original','jl_swap')],
            future_clean_latents_read=False,future_clean_latents_supplied=False,
            phase_order='generate_two_release_model_decode_two',streaming_latency_measured=False,
            quality_evaluation='awaiting_actual_visual_review',realtime_verified=False)
    finally:
        generated.clear()
        if model is not None: model.to('cpu')


def run_job(request,directory,lease):
    """Run real work only after the unchanged guard lease/check and CPU gate."""
    from training.demo.backend import validate_lease,validate_active_authority
    from training.studio.completion_causal_sufficiency_train import ledger_snapshot
    directory=Path(directory)
    require(entry.read(directory/'request.json') == request and request['source_hashes'] == cache.source_hashes(),
        'actual prepared request/source changed')
    allowed=request['parallel_budget_binding'].get('allowed_gpu_indices',[1,2])
    require(request['mode'] in MODES and lease.get('gpu_index') in allowed, 'owned actual Creator card only')
    request_sha=entry.previous.sha256_file(directory/'request.json')
    validate_lease(lease,max_seconds=request['max_seconds'],job_id=request['job_id'],request_sha256=request_sha)
    ledger,ledger_sha=ledger_snapshot()
    accounting=reconcile_parallel_budget(request['parallel_budget_binding'],ledger=ledger,lease=lease,
        max_seconds=request['max_seconds'])
    require(os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'existing offline assets only')
    began=time.monotonic()
    def check():
        require(time.monotonic()-began < request['max_seconds']-15, 'finite ownedODE job deadline reached')
        validate_active_authority(lease)
    kind=request['mode'].removeprefix('completion_ode_v2_'); arguments=request['resolved_arguments']
    if kind == 'merge':
        entry.validate_cache_binding(request['merged_cache_binding'],teacher_binding=arguments['parent'])
        result=dict(stage=entry.STAGE,cache_binding=request['merged_cache_binding'],CPU_merge_completed_before_reserve=True,
            model_loaded=False,cuda_initialized=False,GPU_computation_performed=False)
    else:
        require(torch.cuda.is_available(), 'actualCUDA worker required, CPUfixture not a substitute')
        device=torch.device('cuda',0)
        # entry.launch must perform its free-GPU/process check before this
        # process creates a CUDA context. Other worker kinds were checked by
        # the queue immediately before entering run_job.
        if kind != 'train':
            torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        if kind in ('probe','cache'):
            result=cache.run_cache_shard(request['cache_manifest'],slots=arguments['slots'],device=device,
                check=check,first_probe=kind == 'probe')
        elif kind == 'train':
            budget=copy.deepcopy(arguments['budget_contract'])
            budget.update(ledger_sha256=ledger_sha,settled_gpu_seconds=accounting['group_paid_gpu_seconds']['ode'])
            result=entry.launch(entry.load_config(request['config_path']),request['config_path'],
                parent=arguments['parent'],cache_binding=arguments['cache_binding'],
                start_step=arguments['start_step'],stop_after_step=arguments['stop_after_step'],resume=arguments.get('resume'),
                measured_slowest_step_seconds=arguments.get('measured_slowest_step_seconds'),
                budget_contract=budget,shared_lease=lease,shared_max_seconds=request['max_seconds'],
                shared_request_sha256=request_sha,profile_binding=arguments.get('profile_binding'))
        elif kind == 'validate': result=_numeric(request,device,check)
        else: result=_videos(request,directory,device,check)
        peak=torch.cuda.max_memory_reserved(device)
        require(peak < 23*1024**3, 'actualODE worker reservedmemory exceeded23GiB')
        result['peak_reserved_bytes']=peak
    return dict(status='completed',execution=result,accounting_at_start=accounting,
        elapsed_seconds=time.monotonic()-began,quality_verified=False,realtime_verified=False)


def build_manifest_jobs(parent, *, cache_manifest, first_probe_ceiling=3300, cache_batch_ceiling=2200,
                        train_segment_ceiling=2400, numeric_ceiling=900, video_ceiling=1200):
    """Executable full fixed candidate, NOT measured speed or launch approval.

    Eight cache batches cover slots1..63, plus the retained slot0 probe. A real
    slow profile can reject this candidate; it cannot increase these ceilings,
    silently shrink64/80, or remove comparisons. Root may explicitly replan
    finer batches under the same16h before another reservation.
    """
    entry.validate_parent_binding(parent)
    require(all(type(v) is int and 60 <= v <= 3600 for v in
        (first_probe_ceiling,cache_batch_ceiling,train_segment_ceiling,numeric_ceiling,video_ceiling)),
        'explicit bounded candidate ceilings required')
    require(isinstance(cache_manifest,dict) and set(cache_manifest) == {'path','sha256'}
        and Path(cache_manifest['path']) == entry.ROOT/'cache/manifest.json' and entry.digest(cache_manifest['sha256']),
        'actual CPU-prepared immutable64 cache manifest pin required')
    jobs=[]
    def add(kind,suffix,*,gpu,seconds,dependencies,arguments):
        identity=f'd30-ode-{kind}-{suffix}-v1'
        jobs.append(dict(job_id=identity,mode='completion_ode_v2_'+kind,group='ode',gpu_index=gpu,
            max_seconds=seconds,cleanup_seconds=60,directory=str(entry.ROOT/identity),
            depends_on=list(dependencies),arguments=dict(parent=copy.deepcopy(parent),**arguments)))
        return identity
    probe=add('probe','slot000',gpu=1,seconds=first_probe_ceiling,dependencies=[],
        arguments=dict(cache_manifest=cache_manifest,slots=[0]))
    batches=[]
    for batch,start in enumerate(range(1,64,8)):
        batches.append(add('cache',f'batch{batch:02d}',gpu=1+batch%2,seconds=cache_batch_ceiling,
            dependencies=[probe],arguments=dict(cache_manifest=cache_manifest,slots=list(range(start,min(start+8,64))),
                first_trajectory_profile_from_job_id=probe)))
    all_cache=[probe,*batches]
    common=dict(cache_manifest=cache_manifest,cache_binding_path=str(entry.ROOT/'cache/binding.json'),
        cache_merge_from_jobs=dict(first_trajectory_job_id=probe,cache_job_ids=batches))
    # The first ready prepare performs the pureCPU merge; no blank GPU merge job.
    for case in ('pair-j','holdout-l'):
        add('eval','untrained-'+case,gpu=1,seconds=video_ceiling,dependencies=all_cache,
            arguments=dict(**copy.deepcopy(common),endpoint_step=0,case_id=case))
    profile=add('train','profile2',gpu=2,seconds=1800,dependencies=all_cache,
        arguments=dict(**copy.deepcopy(common),start_step=0,stop_after_step=2,resume=None,budget_from_manifest=True))
    previous=profile; start=2
    for stop in (18,20,36,40,56,72,80):
        identity=add('train',f'to{stop:03d}',gpu=2,seconds=train_segment_ceiling,dependencies=[previous],
            arguments=dict(**copy.deepcopy(common),start_step=start,stop_after_step=stop,
                resume=str(entry.ROOT/'training/checkpoints'/f'step-{start:07d}.pt'),
                profile_binding_from_job_id=profile,measured_profiles_from_job_ids=[previous],budget_from_manifest=True))
        if stop in entry.ENDPOINTS:
            add('validate',f'step{stop:03d}',gpu=1,seconds=numeric_ceiling,dependencies=[identity],
                arguments=dict(**copy.deepcopy(common),endpoint_step=stop,endpoint_profile_from_job_id=identity))
            for case in ('pair-j','holdout-l'):
                add('eval',f'step{stop:03d}-'+case,gpu=1,seconds=video_ceiling,dependencies=[identity],
                    arguments=dict(**copy.deepcopy(common),endpoint_step=stop,case_id=case,
                        endpoint_profile_from_job_id=identity))
        previous,start=identity,stop
    full_worst=sum(row['max_seconds']+60 for row in jobs)
    require(full_worst <= entry.CAP_GPU_SECONDS, 'entire64/80/three numeric/eight pairedvideo candidate exceeds16h')
    return dict(jobs=jobs,full_candidate_worst_gpu_seconds=full_worst,
        cache_slots=list(range(64)),training_segments=[[0,2],[2,18],[18,20],[20,36],[36,40],[40,56],[56,72],[72,80]],
        operator_candidate_not_measured_speed=True,actual_profile_required_before_remaining_cache=True,
        actual_two_step_profile_required_before_remaining_training=True,
        cpu_merge_no_gpu_job=True,source_pins=source_pins())


def source_pins():
    from training import completion_ode_probe_data_v2 as data,completion_ode_wan_v2 as wan
    from training.studio import completion_causal_sufficiency_eval as supervised,balanced_lora_pair_eval as baseline
    from training import causal_tf
    paths=[__file__,entry.__file__,entry.core().__file__,data.__file__,wan.__file__,cache.__file__,
        entry.tf.__file__,entry.previous.__file__,supervised.__file__,baseline.__file__,causal_tf.__file__,
        str(Path(__file__).parents[1]/'wan/modules/causal_model.py'),
        str(Path(__file__).parents[1]/'utils/wan_wrapper.py')]
    return {str(Path(path).resolve()):entry.previous.sha256_file(path) for path in paths}
