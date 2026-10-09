"""Guarded fresh60 training requests; fixed2->20->40->60, no automatic launch."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.studio.controlled_action_train import MODEL, RUNTIME

ROOT = RUNTIME/'abot-retrain-20261003/teacher-balanced-fresh60-v1'
MODE = 'balanced_fresh_train'
REVISION = 'six-reviewed-windows-fresh-zero-gate-fixed60-round-writeback-v1'
SOURCE_REVISION = 'creator-balanced-fresh60-v1'
PREVIOUS = {2:0,20:2,40:20,60:40}


def job(step):
    require(type(step) is int and step in PREVIOUS, 'fixed2/20/40/60 endpoint only')
    return 'cbf-'+('profile2-v1' if step == 2 else f'to{step}-v1')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_request(request, directory):
    import train_balanced_fresh_teacher as entry
    directory = Path(directory); step = request.get('stop_after_step')
    require(request.get('mode') == MODE and request.get('experiment_revision') == REVISION
        and request.get('source_revision') == SOURCE_REVISION, 'fixed fresh60 revision required')
    require(request.get('job_id') == directory.name == job(step) and directory == ROOT/job(step), 'unique fresh60 job required')
    maximum = request.get('max_seconds')
    require(type(maximum) is int and 60 <= maximum <= (900 if step == 2 else 3000)
        and request.get('cleanup_seconds') == 60, 'fixed finite900/3000+60 watchdog required')
    original = ROOT/job(2)
    require(request.get('config') == str(original/'training.yaml')
        and request.get('output_dir') == str(original/'training')
        and request.get('data_root') == str(entry.DATA) and request.get('model_dir') == str(MODEL)
        and request.get('saved_tensors_on_cpu') is True, 'fixed train6 cache/base/output required')
    require(request.get('resume') == (None if step == 2 else str(original/f'training/checkpoints/step-{PREVIOUS[step]:07d}.pt')),
        'only immediate predecessor strict restore permitted')
    require(request.get('method_contract') == entry.method_contract()
        and request.get('protocol_cap_gpu_seconds') == 21600
        and request.get('protocol_incremental_bytes') == 16_000_000_000, 'fixed method and cumulative stage bounds required')
    require(not any(k in request for k in ('initialize_from','warm_start_from','parent_checkpoint','action_scale',
        'ratio_cap','adapter_lr','gate_lr','lora_lr','data_seed','arm')), 'fresh candidate cannot inherit or override a trained arm')
    expected = entry.ActionTeacherConfig(); expected.model.base_model_path = str(MODEL)
    require(request.get('initialization_contract') == entry.initialization_contract(expected)
        and isinstance(request.get('base_identity'),dict), 'explicit official-base fresh initialization required')
    entry.validate_pool(request.get('pool'))


def prepare(*, output, source_revision, max_seconds=900):
    import yaml
    import train_balanced_fresh_teacher as entry
    output = contained(output,ROOT)
    require(output == ROOT/job(2) and not output.exists(), 'new fixed profile identity required')
    pool = entry.pool_contract(); info = pool['binding']['binding']
    config = entry.ActionTeacherConfig(); config.model.base_model_path = str(MODEL)
    config.data.data_factory = entry.FACTORY; config.data.manifest_path = info['manifest']
    config.data.manifest_sha256 = info['manifest_sha256']; config.data.num_frames = 97; config.data.num_workers = 0
    config.optimizer.adapter_lr = 1e-5; config.optimizer.lora_lr = 0.
    config.training.max_steps = 60; config.training.checkpoint_every = 20
    config.training.lora_parameter_dtype = 'float32'; config.training.output_dir = str(output/'training')
    entry.validate_config(config,output_dir=output/'training')
    base = entry.controlled.base_identity(config)
    request = dict(mode=MODE,experiment_revision=REVISION,source_revision=source_revision,job_id=output.name,
        stop_after_step=2,max_seconds=max_seconds,cleanup_seconds=60,config=str(output/'training.yaml'),
        output_dir=str(output/'training'),data_root=str(entry.DATA),model_dir=str(MODEL),saved_tensors_on_cpu=True,
        resume=None,initialization_contract=entry.initialization_contract(config),base_identity=base,pool=pool,
        method_contract=entry.method_contract(),protocol_cap_gpu_seconds=21600,protocol_incremental_bytes=16_000_000_000)
    validate_request(request,output); output.mkdir(parents=True,exist_ok=False)
    path = output/'training.yaml'; path.write_text(yaml.safe_dump(config.to_dict(),sort_keys=False),encoding='utf-8')
    request['config_sha256'] = sha256(path); write_json(output/'request.json',request)
    write_json(output/'preparation.json',dict(status='cpu_prepared',gpu_launched=False,model_weights_loaded=False,
        stage=entry.STAGE,initialization_contract=request['initialization_contract'],base_identity=base,
        sampling_contract=entry.sampling_contract(config.training,pool),official_base_check='headers/stat only',
        training_started=False,quality_improvement_verified=False))
    return request


def completed_endpoint(step=60):
    """Small settled records only; real tensor/state validation belongs to the entry."""
    import train_balanced_fresh_teacher as entry
    directory = ROOT/job(step)
    request,result,settlement = (read(directory/(n+'.json')) for n in ('request','result','settlement'))
    validate_request(request,directory)
    require(result.get('status') == 'completed' and result.get('mode') == MODE
        and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed', 'actual completed/settled endpoint required')
    ex = result['execution']; q = ex.get('qualification',{})
    require(ex.get('stage') == entry.STAGE and ex.get('checkpoint_schema') == entry.CHECKPOINT_SCHEMA
        and ex.get('step') == ex.get('optimizer_internal_step') == step
        and ex.get('micro_batches_consumed') == ex.get('new_stage_micro_batches_consumed') == step*8
        and ex.get('lifetime_micro_batches_consumed') == ex.get('sampling_cursor') == step*8
        and ex.get('historical_micro_batches_before_stage') == 0 and ex.get('configured_max_steps') == 60
        and ex.get('resume_step') == PREVIOUS[step], 'fresh60 counts or strict predecessor changed')
    require(ex.get('initialization_contract') == request['initialization_contract']
        and ex.get('base_identity') == request['base_identity']
        and ex.get('sampling_contract',{}).get('pool') == request['pool']
        and ex.get('method_contract') == entry.method_contract()
        and ex.get('precision_contract') == entry.precision_contract(), 'actual fresh endpoint contract mismatch')
    require(ex.get('checkpoint_path') == str(ROOT/job(2)/f'training/checkpoints/step-{step:07d}.pt')
        and sha256(request['config']) == request['config_sha256'], 'same configured trajectory checkpoint required')
    require(all(q.get(k) is True for k in ('master_updated','gate_updated','gradients_finite',
        'native_state_restore_matched','model_master_projection_equal')) and q.get('lora_trainable') is False
        and 0 < q.get('peak_reserved_bytes',0) < 23*1024**3, 'actual finite update/memory/state qualification failed')
    if step == 2:
        require(q.get('adapter_task_gradient_nonzero') is True, 'second step requires actual task gradient, not decay-only change')
    return request,ex,settlement


def completed_profile():
    request,execution,_ = completed_endpoint(2)
    path = Path(request['output_dir'])/'metrics.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    require(len(rows) in PREVIOUS and [r.get('step') for r in rows] == list(range(1,len(rows)+1)),
        'only a contiguous completed2/20/40/60 trace may supply the original profile')
    # Appended metrics cannot silently stand for a completed or resumable segment.
    for step in (20,40,60):
        if len(rows) >= step:
            following,_,_ = completed_endpoint(step)
            require(following['config_sha256'] == request['config_sha256']
                and following['pool'] == request['pool'], 'extended trace must belong to this exact trajectory')
    task = rows[1].get('task_gradient_evidence',{})
    require(task.get('measured_before_optimizer') is True and task.get('adapter_task_gradient_nonzero') is True
        and task.get('full_adapter_tensor_max_abs',0) > 0 and task.get('weight_decay_excluded') is True
        and task.get('optimizer_momentum_excluded') is True, 'actual step2 pre-optimizer gradient evidence missing')
    times = [r.get('optimizer_step_seconds') for r in rows[:2]]
    require(all(type(t) in (int,float) and math.isfinite(t) and t > 0 for t in times), 'two actual finite optimizer-step times required')
    return request,execution,max(times)


def measured_seconds(step, slowest):
    require(step in (20,40,60) and type(slowest) in (int,float) and math.isfinite(slowest) and slowest > 0,
        'fixed continuation with measured positive step speed required')
    value = math.ceil(slowest*(step-PREVIOUS[step])*1.35+180)
    require(value <= 3000, 'measured segment exceeds3000 seconds; no automatic time extension')
    return value


def resume_request(*, original, output, stop_after_step=20, max_seconds):
    import train_balanced_fresh_teacher as entry
    original,output = contained(original,ROOT),contained(output,ROOT)
    require(stop_after_step in (20,40,60) and original == ROOT/job(2)
        and output == ROOT/job(stop_after_step) and not output.exists(), 'new fixed continuation identity required')
    profile,_,slowest = completed_profile()
    require(type(max_seconds) is int and max_seconds == measured_seconds(stop_after_step,slowest), 'exact measured finite segment bound required')
    predecessor = PREVIOUS[stop_after_step]; prior,ex,_ = completed_endpoint(predecessor)
    require(entry.pool_contract() == profile['pool'] and prior['config_sha256'] == profile['config_sha256'], 'actual cache/config changed')
    checkpoint = original/f'training/checkpoints/step-{predecessor:07d}.pt'
    require(ex['checkpoint_path'] == str(checkpoint) and checkpoint.is_file(), 'actual predecessor checkpoint missing')
    request = dict(profile,job_id=output.name,resume=str(checkpoint),stop_after_step=stop_after_step,max_seconds=max_seconds,
        predecessor_job=str(ROOT/job(predecessor)),profile_qualified=True)
    validate_request(request,output); output.mkdir(parents=True,exist_ok=False)
    write_json(output/'request.json',request); return request


def run(request, directory, lease):
    import train_balanced_fresh_teacher as entry
    from training.demo.backend import validate_active_authority
    began = time.monotonic(); directory = contained(directory,ROOT)
    validate_request(request,directory); validate_active_authority(lease)
    require(lease.get('gpu_index') in (1,2), 'fresh60 cannot borrow expired GPU0 scope')
    require(sha256(request['config']) == request['config_sha256'], 'training config changed')
    step = request['stop_after_step']
    if step != 2:
        _,_,slowest = completed_profile()
        require(request['max_seconds'] == measured_seconds(step,slowest), 'measured continuation bound changed')
        prior,ex,_ = completed_endpoint(PREVIOUS[step])
        require(prior['config_sha256'] == request['config_sha256'] and ex['checkpoint_path'] == request['resume'], 'strict predecessor changed')
    config = entry.load_config(request['config'])
    require(request['initialization_contract'] == entry.initialization_contract(config), 'initialization changed')
    entry.validate_base_identity(request,config)
    execution = entry.launch(config,request['config'],pool=request['pool'],stop_after_step=step,
        resume=request.get('resume'),shared_lease=lease,shared_max_seconds=request['max_seconds'],
        shared_request_sha256=sha256(directory/'request.json'))
    require(execution['stage'] == entry.STAGE and execution['step'] == step, 'wrong training endpoint')
    return dict(status='completed',mode=MODE,execution=execution,elapsed_seconds=time.monotonic()-began,
        quality_improvement_verified=False,generalization_verified=False,default_model_changed=False,automatic_next_stage=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='operation',required=True)
    first = commands.add_parser('prepare'); first.add_argument('--output',required=True)
    first.add_argument('--source-revision',required=True); first.add_argument('--max-seconds',type=int,default=900)
    follow = commands.add_parser('resume'); follow.add_argument('--original',required=True); follow.add_argument('--output',required=True)
    follow.add_argument('--max-seconds',type=int,required=True); follow.add_argument('--stop-after-step',type=int,required=True)
    values = vars(parser.parse_args()); operation = values.pop('operation')
    print(json.dumps(prepare(**values) if operation == 'prepare' else resume_request(**values)))


if __name__ == '__main__': main()
