"""CPU preparation and guarded two-arm2->24 execution; never launches a job."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.studio.controlled_action_train import MODEL, RUNTIME

ROOT = RUNTIME/'abot-retrain-20261003/teacher-balanced-lora-pair-v1'
MODE = 'balanced_lora_pair_train'
REVISION = 'fresh60-weights-only-six-window-lora-switch-new24-v1'
SOURCE_REVISION = 'creator-balanced-lora-pair-v1'
ARMS = ('frozen','trainable')


def profile_job(arm):
    """One explicit I/O recovery; the failed frozen-v1 identity stays retired."""
    require(arm in ARMS, 'fixed frozen/trainable arm required')
    return {'frozen':'cbl-frozen-profile2-v2', 'trainable':'cbl-trainable-profile2-v1'}[arm]


def job(arm, step):
    require(arm in ARMS and type(step) is int and step in (2,24), 'fixed arm and2/24 endpoint required')
    return profile_job(arm) if step == 2 else f'cbl-{arm}-to24-v1'


def paths(arm):
    original = ROOT/job(arm,2)
    return dict(checkpoint=original/'training/checkpoints/step-0000024.pt',
        config=original/'training.yaml',endpoint=ROOT/job(arm,24),profile=original)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_request(request, directory):
    import train_balanced_lora_pair as entry
    directory = Path(directory); arm,step = request.get('arm'),request.get('stop_after_step')
    require(request.get('mode') == MODE and request.get('experiment_revision') == REVISION
        and request.get('source_revision') == SOURCE_REVISION, 'fixed new LoRA pair revision required')
    require(request.get('job_id') == directory.name == job(arm,step) and directory == ROOT/job(arm,step),
        'unique exact arm job required')
    maximum = request.get('max_seconds')
    require(type(maximum) is int and 60 <= maximum <= (900 if step == 2 else 3600)
        and request.get('cleanup_seconds') == 60, 'fixed finite900/3600+60 watchdog required')
    original = ROOT/job(arm,2)
    require(request.get('config') == str(original/'training.yaml')
        and request.get('output_dir') == str(original/'training')
        and request.get('data_root') == str(entry.DATA) and request.get('model_dir') == str(MODEL)
        and request.get('saved_tensors_on_cpu') is True, 'fixed existing train6/base/new arm directory required')
    require(request.get('resume') == (None if step == 2 else str(original/'training/checkpoints/step-0000002.pt')),
        'only actual same-arm step2 strict resume permitted')
    require(request.get('method_contract') == entry.method_contract(arm)
        and request.get('protocol_cap_gpu_seconds') == 21600
        and request.get('protocol_incremental_bytes') == 20_000_000_000, 'fixed method and cumulative stage bounds required')
    require(not any(k in request for k in ('initialize_from','warm_start_from','parent_checkpoint','action_scale',
        'ratio_cap','adapter_lr','gate_lr','lora_lr','data_seed')), 'no implicit warmstart or hyperparameter override')
    entry.validate_lineage(request.get('parent_lineage')); entry.validate_pool(request.get('pool'))
    require(request.get('base_identity') == request['parent_lineage']['base_identity'], 'same official parent base required')


def prepare(*, output, arm, source_revision, max_seconds=900):
    import yaml
    import train_balanced_lora_pair as entry
    output = contained(output,ROOT)
    require(output == ROOT/job(arm,2) and not output.exists(), 'new fixed profile identity required')
    lineage,pool = entry.parent_binding(),entry.pool_contract()
    # Read the real parent config, but do not load weights during CPU preparation.
    config = entry.fresh.load_config(entry.parent_paths()['config'])
    config.data.data_factory = entry.FACTORY
    config.optimizer.lora_lr = 2e-6 if arm == 'trainable' else 0.
    config.training.max_steps = 24; config.training.output_dir = str(output/'training')
    entry.validate_config(config,output_dir=output/'training',arm=arm)
    request = dict(mode=MODE,experiment_revision=REVISION,source_revision=source_revision,job_id=output.name,arm=arm,
        stop_after_step=2,max_seconds=max_seconds,cleanup_seconds=60,config=str(output/'training.yaml'),
        output_dir=str(output/'training'),data_root=str(entry.DATA),model_dir=str(MODEL),saved_tensors_on_cpu=True,
        resume=None,parent_lineage=lineage,base_identity=lineage['base_identity'],pool=pool,
        method_contract=entry.method_contract(arm),protocol_cap_gpu_seconds=21600,protocol_incremental_bytes=20_000_000_000)
    validate_request(request,output); output.mkdir(parents=True,exist_ok=False)
    path = output/'training.yaml'; path.write_text(yaml.safe_dump(config.to_dict(),sort_keys=False),encoding='utf-8')
    request['config_sha256'] = sha256(path); write_json(output/'request.json',request)
    write_json(output/'preparation.json',dict(status='cpu_prepared',gpu_launched=False,model_weights_loaded=False,
        stage=entry.STAGE,arm=arm,parent_lineage=lineage,base_identity=lineage['base_identity'],
        sampling_contract=entry.sampling_contract(config.training,pool),training_started=False,
        parent_teacher_qualified=False,quality_improvement_verified=False))
    return request


def completed_endpoint(arm, step=24):
    """Read only actual settled small records. No checkpoint load or quality claim."""
    import train_balanced_lora_pair as entry
    directory = ROOT/job(arm,step)
    request,result,settlement = (read(directory/(n+'.json')) for n in ('request','result','settlement'))
    validate_request(request,directory)
    require(result.get('status') == 'completed' and result.get('mode') == MODE and result.get('arm') == arm
        and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed', 'actual completed/settled arm required')
    ex = result['execution']; q = ex.get('qualification',{})
    require(ex.get('stage') == entry.STAGE and ex.get('checkpoint_schema') == entry.CHECKPOINT_SCHEMA
        and ex.get('arm') == arm and ex.get('step') == ex.get('optimizer_internal_step') == step
        and ex.get('micro_batches_consumed') == ex.get('new_stage_micro_batches_consumed') == step*8
        and ex.get('sampling_cursor') == step*8 and ex.get('historical_micro_batches_before_stage') == 480
        and ex.get('lifetime_micro_batches_consumed') == 480+step*8 and ex.get('configured_max_steps') == 24
        and ex.get('resume_step') == (0 if step == 2 else 2), 'new optimizer/history/strict same-arm counters differ')
    require(ex.get('parent_lineage') == request['parent_lineage'] and ex.get('base_identity') == request['base_identity']
        and ex.get('sampling_contract',{}).get('pool') == request['pool']
        and ex.get('method_contract') == entry.method_contract(arm)
        and ex.get('precision_contract') == entry.precision_contract(arm), 'actual new arm contract mismatch')
    require(ex.get('checkpoint_path') == str(ROOT/job(arm,2)/f'training/checkpoints/step-{step:07d}.pt')
        and sha256(request['config']) == request['config_sha256'], 'exact configured trajectory checkpoint required')
    require(all(q.get(k) is True for k in ('master_updated','adapter_updated','gate_updated','gradients_finite',
        'adapter_task_gradient_nonzero','gate_task_gradient_nonzero','native_state_restore_matched','model_master_projection_equal'))
        and q.get('lora_trainable') is (arm == 'trainable') and q.get('frozen_lora_has_gradients') is False
        and 0 < q.get('peak_reserved_bytes',0) < 23*1024**3, 'actual gradient/update/native state/resource gate failed')
    if arm == 'trainable':
        require(q.get('lora_B_updated') is True and q.get('step2_lora_A_task_gradient_nonzero') is True,
            'actual nonzero B update and second-step A task gradient required')
    else:
        require(q.get('lora_B_updated') is None and q.get('step2_lora_A_task_gradient_nonzero') is None,
            'frozen arm cannot claim optimized LoRA')
    return request,ex,settlement


def completed_profile(arm):
    request,execution,_ = completed_endpoint(arm,2)
    path = Path(request['output_dir'])/'metrics.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    require(len(rows) in (2,24) and [r.get('step') for r in rows] == list(range(1,len(rows)+1)),
        'only exact contiguous completed2/24 trace can supply original profile')
    if len(rows) == 24:
        following,_,_ = completed_endpoint(arm,24)
        require(all(following[k] == request[k] for k in ('config','config_sha256','output_dir','pool','parent_lineage'))
            and following['resume'] == str(Path(request['output_dir'])/'checkpoints/step-0000002.pt'),
            'extended trace must have actual settled same-arm continuation')
    task = rows[1].get('task_gradient_evidence',{}); maxima = task.get('full_tensor_max_abs',{})
    require(all(task.get(k) is True for k in ('measured_before_optimizer','adapter_task_gradient_nonzero',
        'gate_task_gradient_nonzero','weight_decay_excluded','optimizer_momentum_excluded'))
        and maxima.get('adapter',0) > 0 and maxima.get('gate',0) > 0
        and task.get('frozen_lora_has_gradients') is False, 'actual step2 task gradients required, not decay-only changes')
    if arm == 'trainable':
        require(task.get('lora_A_task_gradient_nonzero') is True and task.get('lora_B_task_gradient_nonzero') is True
            and maxima.get('lora_a',0) > 0 and maxima.get('lora_b',0) > 0, 'step2 actual FP32 LoRA A/B task gradients required')
    times = [r.get('optimizer_step_seconds') for r in rows[:2]]
    require(all(type(t) in (int,float) and math.isfinite(t) and t > 0 for t in times), 'actual finite two-step measurements required')
    return request,execution,max(times)


def both_profiles():
    profiles = {arm:completed_profile(arm) for arm in ARMS}
    a,b = (profiles[arm][0] for arm in ARMS)
    require(a['parent_lineage'] == b['parent_lineage'] and a['pool'] == b['pool'], 'common real parent and same train6 pool required')
    require(profiles['frozen'][1]['sampling_contract'] == profiles['trainable'][1]['sampling_contract'],
        'shared sampling/noise namespace differs')
    return profiles


def measured_seconds(slowest):
    require(type(slowest) in (int,float) and math.isfinite(slowest) and slowest > 0, 'finite actual positive profile timing required')
    value = math.ceil(slowest*22*1.35+180)
    require(value <= 3600, 'measured22-step continuation exceeds3600; do not extend or change configuration')
    return value


def resume_request(*, original, output, stop_after_step=24, max_seconds):
    import train_balanced_lora_pair as entry
    original,output = contained(original,ROOT),contained(output,ROOT)
    arm = next((a for a in ARMS if original == ROOT/job(a,2)),None)
    require(arm in ARMS and stop_after_step == 24 and output == ROOT/job(arm,24)
        and not output.exists(), 'new same-arm fixed continuation identity required')
    profiles = both_profiles(); profile,ex,slowest = profiles[arm]
    require(type(max_seconds) is int and max_seconds == measured_seconds(slowest), 'own measured finite continuation bound required')
    require(entry.pool_contract() == profile['pool'] and entry.parent_binding() == profile['parent_lineage'], 'actual common parent/cache changed')
    checkpoint = original/'training/checkpoints/step-0000002.pt'
    require(ex['checkpoint_path'] == str(checkpoint) and checkpoint.is_file(), 'actual same-arm predecessor checkpoint missing')
    request = dict(profile,job_id=output.name,resume=str(checkpoint),stop_after_step=24,max_seconds=max_seconds,
        predecessor_job=str(original),both_profiles_qualified=True)
    validate_request(request,output); output.mkdir(parents=True,exist_ok=False)
    write_json(output/'request.json',request); return request


def run(request, directory, lease):
    import train_balanced_lora_pair as entry
    from training.demo.backend import validate_active_authority
    began = time.monotonic(); directory = contained(directory,ROOT)
    validate_request(request,directory); validate_active_authority(lease)
    require(lease.get('gpu_index') in (1,2), 'new pair may use only original Creator GPU1/2')
    require(sha256(request['config']) == request['config_sha256'], 'training config changed')
    arm,step = request['arm'],request['stop_after_step']
    if step == 24:
        profiles = both_profiles(); prior,ex,slowest = profiles[arm]
        require(request['max_seconds'] == measured_seconds(slowest) and prior['config_sha256'] == request['config_sha256']
            and ex['checkpoint_path'] == request['resume'], 'actual both-profile gate or strict resume changed')
    config = entry.load_config(request['config']); entry.validate_base_identity(request,config)
    execution = entry.launch(config,request['config'],arm=arm,lineage=request['parent_lineage'],pool=request['pool'],
        stop_after_step=step,resume=request.get('resume'),shared_lease=lease,shared_max_seconds=request['max_seconds'],
        shared_request_sha256=sha256(directory/'request.json'))
    require(execution['stage'] == entry.STAGE and execution['arm'] == arm and execution['step'] == step, 'wrong training endpoint')
    return dict(status='completed',mode=MODE,arm=arm,execution=execution,elapsed_seconds=time.monotonic()-began,
        quality_improvement_verified=False,generalization_verified=False,default_model_changed=False,automatic_next_stage=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='operation',required=True)
    first = commands.add_parser('prepare'); first.add_argument('--output',required=True); first.add_argument('--arm',choices=ARMS,required=True)
    first.add_argument('--source-revision',required=True); first.add_argument('--max-seconds',type=int,default=900)
    follow = commands.add_parser('resume'); follow.add_argument('--original',required=True); follow.add_argument('--output',required=True)
    follow.add_argument('--max-seconds',type=int,required=True); follow.add_argument('--stop-after-step',type=int,default=24)
    values = vars(parser.parse_args()); operation = values.pop('operation')
    print(json.dumps(prepare(**values) if operation == 'prepare' else resume_request(**values)))


if __name__ == '__main__': main()
