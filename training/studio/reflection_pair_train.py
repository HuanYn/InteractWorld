"""Guarded new-optimizer reflection pair with two measured profiles and strict resume."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.studio.controlled_action_train import MODEL, RUNTIME

ROOT=RUNTIME/'abot-retrain-20261003/teacher-reflection-pair-v1'
MODE='reflection_pair_train'
REVISION='rgb-reflection-supervision-fresh-optimizer-round-writeback-v1'
SOURCE_REVISION='creator-reflection-pair-v1'
ARMS=('original','reflect')


def job(arm,step):
    require(arm in ARMS and type(step)is int and step in (2,20),'fixed arm and step required')
    return f'crf-{arm}-'+('profile2-v1' if step==2 else 'to20-v1')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_request(request,directory):
    import train_reflection_pair as entry
    directory=Path(directory); arm,step=request.get('arm'),request.get('stop_after_step')
    require(request.get('mode')==MODE and request.get('experiment_revision')==REVISION
        and request.get('source_revision')==SOURCE_REVISION,'fixed reflection revision required')
    require(request.get('job_id')==directory.name==job(arm,step),'unique reflection arm job changed')
    maximum=request.get('max_seconds')
    require(type(maximum)is int and 60<=maximum<=(900 if step==2 else 3600)
        and request.get('cleanup_seconds')==60,'finite profile/continuation watchdog required')
    original=ROOT/job(arm,2)
    require(request.get('config')==str(original/'training.yaml')
        and request.get('output_dir')==str(original/'training')
        and request.get('data_root')==str(entry.DATA) and request.get('model_dir')==str(MODEL)
        and request.get('saved_tensors_on_cpu') is True,'fixed new cache/model/output required')
    require(request.get('resume')==(None if step==2 else str(original/'training/checkpoints/step-0000002.pt')),
        'same-arm profile2 then strict2->20 only')
    require(request.get('method_contract')==entry.method_contract(arm)
        and request.get('protocol_cap_gpu_seconds')==21600
        and request.get('protocol_incremental_bytes')==16_000_000_000,'fixed intervention/bounds required')
    require(not any(k in request for k in ('initialize_from','warm_start_from','action_scale','ratio_cap',
        'adapter_lr','gate_lr','lora_lr','data_seed')),'no training override permitted')
    entry.validate_lineage(request.get('parent_lineage')); entry.validate_pool(request.get('pool'))


def prepare(*,output,source_revision,arm,max_seconds=900):
    import yaml
    import train_reflection_pair as entry
    output=contained(output,ROOT)
    require(output==ROOT/job(arm,2) and not output.exists(),'new fixed profile identity required')
    lineage=entry.parent_binding(); pool=entry.pool_contract()
    config=entry.master.load_config(entry.source.paths('round')['config'])
    config.data.data_factory=entry.FACTORY
    config.data.manifest_path=pool['binding']['manifest']; config.data.manifest_sha256=pool['binding']['manifest_sha256']
    config.training.output_dir=str(output/'training')
    entry.validate_config(config,output_dir=output/'training',arm=arm)
    request=dict(mode=MODE,experiment_revision=REVISION,source_revision=source_revision,
        job_id=output.name,arm=arm,stop_after_step=2,max_seconds=max_seconds,cleanup_seconds=60,
        config=str(output/'training.yaml'),output_dir=str(output/'training'),data_root=str(entry.DATA),
        model_dir=str(MODEL),saved_tensors_on_cpu=True,resume=None,parent_lineage=lineage,pool=pool,
        method_contract=entry.method_contract(arm),protocol_cap_gpu_seconds=21600,protocol_incremental_bytes=16_000_000_000)
    validate_request(request,output)
    output.mkdir(parents=True,exist_ok=False)
    path=output/'training.yaml'; path.write_text(yaml.safe_dump(config.to_dict(),sort_keys=False),encoding='utf-8')
    request['config_sha256']=sha256(path); write_json(output/'request.json',request)
    write_json(output/'preparation.json',dict(status='cpu_prepared',gpu_launched=False,weights_loaded=False,
        stage=entry.STAGE,parent_lineage=lineage,sampling_contract=entry.sampling_contract(config.training,pool,arm,lineage),
        parent_validation='actual_master_round20_schema_config_and_BF16_export_in_guarded_run',
        optimizer_restored_from_parent=False,quality_improvement_verified=False))
    return request


def completed_profile(arm):
    import train_reflection_pair as entry
    directory=ROOT/job(arm,2)
    request,result,settlement=(read(directory/(n+'.json')) for n in ('request','result','settlement'))
    validate_request(request,directory)
    require(result.get('status')=='completed' and result.get('mode')==MODE
        and settlement.get('status')=='settled' and settlement.get('outcome')=='completed',
        'both actual profiles must be completed/settled')
    execution=result['execution']; q=execution.get('qualification',{})
    history=request['parent_lineage']['historical_micro_batches_before_stage']
    require(execution.get('stage')==entry.STAGE and execution.get('arm')==arm and execution.get('step')==2
        and execution.get('optimizer_internal_step')==2 and execution.get('sampling_cursor')==16
        and execution.get('micro_batches_consumed')==execution.get('new_stage_micro_batches_consumed')==16
        and execution.get('historical_micro_batches_before_stage')==history
        and execution.get('lifetime_micro_batches_consumed')==history+16
        and execution.get('parent_lineage')==request['parent_lineage'],'actual new optimizer/namespace history differs')
    require(all(q.get(k)is True for k in ('master_updated','gate_updated','gradients_finite',
        'native_state_restore_matched','model_master_projection_equal')) and q.get('lora_trainable') is False
        and 0<q.get('peak_reserved_bytes',0)<23*1024**3,'actual memory/update/roundtrip profile gate failed')
    require(sha256(request['config'])==request['config_sha256'],'profile config changed')
    rows=[json.loads(line) for line in (Path(request['output_dir'])/'metrics.jsonl').read_text().splitlines() if line.strip()]
    require([r.get('step') for r in rows] in ([1,2],list(range(1,21))),'only contiguous profile2 or completed20 trace')
    if len(rows)==20:
        continuation=ROOT/job(arm,20)
        follow,final,settled=(read(continuation/(n+'.json')) for n in ('request','result','settlement'))
        validate_request(follow,continuation)
        require(final.get('status')=='completed' and final.get('mode')==MODE
            and final['execution'].get('arm')==arm and final['execution'].get('step')==20
            and final['execution'].get('resume_step')==2
            and final['execution'].get('checkpoint_path')==str(directory/'training/checkpoints/step-0000020.pt')
            and final['execution'].get('parent_lineage')==request['parent_lineage']
            and follow.get('config_sha256')==request['config_sha256']
            and settled.get('status')=='settled' and settled.get('outcome')=='completed',
            'extended metrics require actual settled same-arm continuation')
    times=[r.get('optimizer_step_seconds') for r in rows[:2]]
    require(all(type(t)in (int,float) and math.isfinite(t) and t>0 for t in times),'real finite step times required')
    return request,execution,math.ceil(max(times)*18*1.35+180)


def both_profiles():
    rows={a:completed_profile(a) for a in ARMS}
    require(rows['original'][0]['parent_lineage']==rows['reflect'][0]['parent_lineage']
        and rows['original'][0]['pool']==rows['reflect'][0]['pool'],'same actual parent and four-view binding required')
    return rows


def resume_request(*,original,output,stop_after_step=20,max_seconds):
    import train_reflection_pair as entry
    original,output=contained(original,ROOT),contained(output,ROOT)
    prior=read(original/'request.json'); arm=prior.get('arm')
    require(original==ROOT/job(arm,2) and output==ROOT/job(arm,20) and not output.exists()
        and stop_after_step==20,'new fixed same-arm continuation only')
    profiles=both_profiles(); previous,execution,measured=profiles[arm]
    require(type(max_seconds)is int and max_seconds==measured<=3600,
        'exact measured slowest own two-step formula within3600 required')
    require(entry.parent_binding()==previous['parent_lineage'] and entry.pool_contract()==previous['pool'],
        'parent/cache/actual visual gate changed')
    checkpoint=original/'training/checkpoints/step-0000002.pt'
    require(execution.get('checkpoint_path')==str(checkpoint) and checkpoint.is_file(),'actual step2 checkpoint missing')
    request=dict(previous,job_id=output.name,resume=str(checkpoint),stop_after_step=20,max_seconds=max_seconds,
        predecessor_job=str(original),both_profiles_qualified=True)
    validate_request(request,output); output.mkdir(parents=True,exist_ok=False)
    write_json(output/'request.json',request); return request


def run(request,directory,lease):
    import train_reflection_pair as entry
    from training.demo.backend import validate_active_authority
    began=time.monotonic(); directory=contained(directory,ROOT)
    validate_request(request,directory); validate_active_authority(lease)
    require(lease.get('gpu_index') in (1,2),'reflection cannot borrow expired GPU0 grant')
    require(sha256(request['config'])==request['config_sha256'],'config changed')
    if request['stop_after_step']==20:
        profiles=both_profiles()
        require(request['max_seconds']==profiles[request['arm']][2]<=3600,'measured continuation bound changed')
    config=entry.load_config(request['config'])
    execution=entry.launch(config,request['config'],arm=request['arm'],lineage=request['parent_lineage'],pool=request['pool'],
        stop_after_step=request['stop_after_step'],resume=request.get('resume'),shared_lease=lease,
        shared_max_seconds=request['max_seconds'],shared_request_sha256=sha256(directory/'request.json'))
    require(execution['stage']==entry.STAGE and execution['step']==request['stop_after_step'],'wrong endpoint')
    return dict(status='completed',mode=MODE,arm=request['arm'],execution=execution,elapsed_seconds=time.monotonic()-began,
        quality_improvement_verified=False,generalization_verified=False,default_model_changed=False,automatic_next_stage=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__); commands=parser.add_subparsers(dest='operation',required=True)
    first=commands.add_parser('prepare'); first.add_argument('--output',required=True)
    first.add_argument('--source-revision',required=True); first.add_argument('--arm',choices=ARMS,required=True)
    first.add_argument('--max-seconds',type=int,default=900)
    follow=commands.add_parser('resume'); follow.add_argument('--original',required=True)
    follow.add_argument('--output',required=True); follow.add_argument('--max-seconds',type=int,required=True)
    follow.add_argument('--stop-after-step',type=int,default=20)
    values=vars(parser.parse_args()); operation=values.pop('operation')
    print(json.dumps(prepare(**values) if operation=='prepare' else resume_request(**values)))


if __name__=='__main__': main()
