"""Explicit three-card routing and fee bridge; original guard is not replaced.

Scientific rows and all outputs remain fixed. This module never reserves cards.
Fee bounds include setup60 + owned-process termination15 + three serialized
settlements(30+6metadata each) + original charge30 + polling1 <=216s.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

SCHEMA='creator_deadline30h_parallel_manifest_v4'
BINDING='creator_deadline30h_parallel_binding_v4'
REVISION='creator-completion-deadline-v4'
FEE_OVERHEAD=216
CAPS=dict(ode=57600,dmd=57600,runtime=21600,evaluation=14400)
FAILURE_CONTINGENCY=90
CARDS=[0,1,2]
MODE='completion_service_v3_endpoint'


def require(ok,message):
    if not ok:raise ValueError(message)


def is_v4(binding):return isinstance(binding,dict) and binding.get('schema')==BINDING


def read(path):
    from train_completion_ode_sufficiency import read as original_read
    return original_read(path)


def manifest(binding):
    from train_completion_ode_sufficiency import small_pin,ROOT
    require(set(binding)=={'schema','manifest','owned_job_ids','ledger','allowed_gpu_indices','max_owned_active'}
        and is_v4(binding) and binding['allowed_gpu_indices']==CARDS and binding['max_owned_active']==3,
        'explicit reviewed v4 three-card ownership required')
    root=ROOT.parent/'deadline30h-v4'
    require(Path(binding['manifest']['path'])==root/'manifest.json','fixed new v4 manifest required')
    value=small_pin(binding['manifest'])
    require(value.get('schema')==SCHEMA and value.get('coordinator_root')==str(root)
        and value.get('fixed_plan_complete') is True and value.get('allowed_gpu_indices')==CARDS
        and value.get('max_owned_active')==3 and value.get('global_cap_gpu_seconds')==360000
        and value.get('server_cap_bytes')==500000000000 and value.get('ledger')==binding['ledger']
        and value.get('fee_overhead_seconds')==FEE_OVERHEAD,'fixed original budget and full fee overhead required')
    require(value.get('failure_contingency_seconds_per_group')==FAILURE_CONTINGENCY,
        'multiple owned-process termination contingency must remain funded')
    require(sorted(j['job_id'] for j in value['jobs'])==binding['owned_job_ids']
        and len(set(binding['owned_job_ids']))==len(value['jobs']),'all actual physical IDs required')
    for key,cap in CAPS.items():
        require(value['groups'][key]['cap_gpu_seconds']==cap,'original independent subcap changed')
    require(value['future_envelopes_gpu_seconds']==dict(ode=0,dmd=0,runtime=0,evaluation=0),
        'concrete full runtime/evaluation jobs replace unknown envelopes, not omitted work')
    for row in value['jobs']:
        require(row['gpu_index'] in CARDS and row.get('source_revision')==REVISION
            and type(row['max_seconds']) is int and 0<row['max_seconds']<=3600
            and row['cleanup_seconds']==60 and Path(row['directory']).name==row['job_id']
            and Path(row['directory']).is_relative_to(Path(value['groups'][row['group']]['accounting_root'])),
            'bounded exact physical routing required')
    expected=complete_jobs(value['logical_jobs'])
    require(expected==value['jobs'],'complete scientific logical rows/physical routing changed')
    return value


def route_jobs(logical):
    """Full original56 rows ->37 batches; no endpoint/action/seed removed."""
    rows=copy.deepcopy(logical)
    require(len(rows)==56 and len({r['job_id'] for r in rows})==56,'full56 logical contract required')
    groups={};replace={};batches={}
    for row in rows:
        mode,args=row['mode'],row['arguments']
        if mode in ('completion_ode_v2_validate','completion_ode_v2_eval'):
            key=('ode',None,args.get('endpoint_step'))
        elif mode in ('completion_dmd_v2_validate','completion_dmd_v2_eval'):
            key=('dmd',args.get('arm'),args.get('endpoint_step'))
        else:continue
        groups.setdefault(key,[]).append(row)
    require(set(groups)=={('ode',None,s) for s in (0,20,40,80)}
        |{('dmd',a,s) for a in ('short','long') for s in (8,16,32)},'all comparison endpoints required')
    for (stage,arm,step),parts in groups.items():
        videos=[p for p in parts if p['mode'].endswith('_eval')]
        numeric=[p for p in parts if p['mode'].endswith('_validate')]
        require(len(videos)==2 and {p['arguments']['case_id'] for p in videos}=={'pair-j','holdout-l'}
            and len(numeric)==(0 if step==0 else 1),'exact numeric+four clips required')
        first=copy.deepcopy((numeric or videos)[0]);args=first['arguments'];args.pop('case_id',None)
        args.update(batch_stage=stage,cases=['pair-j','holdout-l'],conditions=['original','jl_swap'])
        identity=f'd30-service-v4-{stage}-{arm or "single"}-step{step:03d}'
        first.update(job_id=identity,mode=MODE,gpu_index=0,
            max_seconds=sum(p['max_seconds'] for p in parts),
            directory=str(Path(first['directory']).parent/identity),
            depends_on=sorted(set(d for p in parts for d in p['depends_on'])))
        require(first['max_seconds']==(2400 if step==0 else 3300 if stage=='ode' else 3000),
            'summed original work ceilings required')
        batches[identity]=first
        for part in parts:replace[part['job_id']]=identity
    physical=[];seen=set()
    for row in rows:
        identity=replace.get(row['job_id'],row['job_id'])
        if identity in seen:continue
        seen.add(identity);row=batches.get(identity,row)
        row['depends_on']=sorted({replace.get(d,d) for d in row['depends_on']})
        if row['group']=='dmd' and row['mode']!=MODE:row['gpu_index']=1 if row['arguments']['arm']=='short' else 2
        row['source_revision']=REVISION;physical.append(row)
    require(len(physical)==37,'complete37 physical jobs required')
    for group,cap in CAPS.items():
        if group in ('ode','dmd'):
            require(sum(r['max_seconds']+FEE_OVERHEAD for r in physical if r['group']==group)<=cap,
                'complete original stage including actual settlement overhead exceeds subcap')
    return physical


def comparison_identities(row):
    args=row['arguments'];step=args['endpoint_step'];prefix='untrained' if step==0 else f'ode{step}'
    require(row['mode']==MODE and args['batch_stage']=='ode','actual ODE batch required')
    identities=[f'{prefix}-{case}-{condition}' for case in ('pair-j','holdout-l') for condition in ('original','jl_swap')]
    if step:identities.insert(0,f'ode{step}-numeric-probe4-level4')
    return ('untrained_control' if step==0 else 'video_evaluation'),identities


def complete_jobs(logical):
    from training import completion_runtime_jobs_v4 as runtime,completion_evaluation_jobs_v4 as evaluation
    main=route_jobs(logical)
    ode80=next(r for r in main if r['mode']=='completion_ode_v2_train' and r['arguments']['stop_after_step']==80)
    producers={'ode80':ode80['job_id']}
    for arm in ('short','long'):
        producers[arm+'32']=next(r['job_id'] for r in main if r['mode']=='completion_dmd_v2_train'
            and r['arguments']['arm']==arm and r['arguments']['stop_after_step']==32)
    tail=runtime.build_tail_jobs(ode80,gpu_index=0)+evaluation.jobs(producers,gpu_indices=(0,1,2))
    jobs=main+tail
    def priority(row):
        if row['mode'].startswith(('completion_ode_v2_','completion_dmd_v2_')):return 0
        if row['group']=='runtime':return 1
        if row['mode']==MODE:return 2
        return 3
    jobs=sorted(jobs,key=priority)  # fixed stable order shared by live and CPU event machine
    require(len(jobs)==47 and len({r['job_id'] for r in jobs})==47,'full37+4+6 physical contract required')
    for group,cap in CAPS.items():
        require(sum(r['max_seconds']+FEE_OVERHEAD for r in jobs if r['group']==group)+FAILURE_CONTINGENCY<=cap,
            'complete '+group+' including settlement/failure fees exceeds original cap')
    return jobs


def reconcile(binding,*,ledger,lease=None,max_seconds=None,settled_stage_gpu_seconds=None,now=None):
    import time
    now=time.time() if now is None else now
    plan=manifest(binding);jobs={r['job_id']:r for r in plan['jobs']}
    roots={g:Path(r['accounting_root']) for g,r in plan['groups'].items()}
    require(ledger.get('version')==1 and 0<=ledger.get('storage_bytes_upper',math.inf)<=500000000000,
        'actual original bounded ledger required')
    paid={g:0. for g in CAPS};active={g:0. for g in CAPS};global_paid=global_active=0.;seen=set();held=[]
    for rid,row in ledger['reservations'].items():
        context=row['context'];identity=context['job_id'];require(identity not in seen,'duplicate actual job')
        seen.add(identity);directory=Path(context['job_directory'])
        matches=[g for g,p in roots.items() if directory==p or p in directory.parents]
        require(len(matches)<=1,'ambiguous actual accounting root');group=matches[0] if matches else None
        if row['status']=='settled':
            amount=row['charged_seconds'];require(type(amount) in (int,float) and math.isfinite(amount) and amount>=0,'invalid actual fee')
            global_paid+=amount
            if group:paid[group]+=amount
            if identity in jobs:require(row['settlement']['outcome']=='completed','failed identity terminal')
        else:
            require(row['status']=='reserved' and identity in jobs,'foreign active reservation')
            job=jobs[identity];rq=read(directory/'request.json');actual=row['lease']
            require(actual==read(directory/'lease.json') and actual['reservation_id']==rid
                and actual['gpu_index']==job['gpu_index'] and rq['parallel_budget_binding']==binding
                and rq['contract_max_seconds']==job['max_seconds']
                and directory==Path(job['directory']) and Path(context['job_directory'])==Path(job['directory'])
                and type(rq['max_seconds']) is int and 0<rq['max_seconds']<=job['max_seconds']
                and context['request_sha256']==hashlib.sha256((directory/'request.json').read_bytes()).hexdigest()
                and context['max_seconds']==rq['max_seconds'] and row['reserved_seconds']==rq['max_seconds']+60,
                'actual owned request/lease/card differs')
            # Include finite serialized service/cleanup debt, not only nominal guard reservation.
            amount=max(rq['max_seconds']+FEE_OVERHEAD,now-row['created_unix']+120)
            global_active+=amount;active[group]+=amount;held.append((rid,row))
    require(len(held)<=3 and len({r['lease']['gpu_index'] for _,r in held})==len(held)
        and len({r['lease']['gpu_uuid'] for _,r in held})==len(held),'max3 distinct owned cards')
    if lease is not None:
        require(any(r['lease']==lease and r['context']['max_seconds']==max_seconds for _,r in held),'current actual lease absent')
    pending=copy.deepcopy(plan['future_envelopes_gpu_seconds'])
    for group in CAPS:pending[group]+=FAILURE_CONTINGENCY
    for identity,row in jobs.items():
        if identity not in seen:pending[row['group']]+=row['max_seconds']+FEE_OVERHEAD
    for group,cap in CAPS.items():require(paid[group]+active[group]+pending[group]<=cap,'complete '+group+' fee upper exceeds original subcap')
    require(global_paid+global_active+sum(pending.values())<=360000,'all history/active/future exceeds100h')
    if settled_stage_gpu_seconds is not None:require(paid['ode']==settled_stage_gpu_seconds,'old ODE fees omitted')
    return dict(global_paid_gpu_seconds=global_paid,all_active_reserved_gpu_seconds=global_active,
        group_paid_gpu_seconds=paid,group_active_gpu_seconds=active,group_pending_gpu_seconds=pending,
        remaining_worst_gpu_seconds=sum(pending.values()),global_upper_gpu_seconds=global_paid+global_active+sum(pending.values()))
