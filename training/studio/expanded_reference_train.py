"""CPU preparation and guarded two-arm2->20 execution; never launches a job."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.studio.controlled_action_train import MODEL, RUNTIME

ROOT = RUNTIME/'abot-retrain-20261003/teacher-expanded-reference-v1'
MODE = 'expanded_reference_train'
REVISION = 'expanded80-weights-only-same60-reference-pair-new20-v1'
SOURCE_REVISION = 'creator-expanded-reference-v1'
ARMS = ('no_reference','reference')
CAP, BYTES = 21600, 16_000_000_000
DATA_SOURCE = RUNTIME / 'abot-retrain-20261003/teacher-expanded-data-v1'


def profile_job(arm):
    require(arm in ARMS, 'fixed no_reference/reference arm required')
    return f'cer-{arm}-profile2-v1'


def job(arm, step):
    require(arm in ARMS and type(step) is int and step in (2,20), 'fixed arm and2/20 endpoint required')
    return profile_job(arm) if step == 2 else f'cer-{arm}-to20-v1'


def paths(arm, step=20):
    original = ROOT/job(arm,2)
    return dict(checkpoint=original/f'training/checkpoints/step-{step:07d}.pt',
        config=original/'training.yaml',endpoint=ROOT/job(arm,step),profile=original)


def entry_module():
    import train_expanded_reference_teacher
    return train_expanded_reference_teacher


def record_binding(directory):
    return {n+'_sha256':sha256(Path(directory)/(n+'.json')) for n in ('request','result','settlement')}


def read(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size < 16_000_000, 'missing/oversized small record')
    return json.loads(path.read_text(encoding='utf-8'))


def validate_data_locations(binding):
    from training.studio.expanded_teacher_train import validate_data_locations as validate_original
    validate_original(binding)
    require(binding['subset']['path'] == str(DATA_SOURCE/'qualified-subset.json')
            and binding['subset']['sha256'] == 'd0db64665a88fce15f7471158825b43c882ec07b7b100e6533b4c9ab3a41678a'
            and binding['review']['path'] == str(DATA_SOURCE/'visual-review.json')
            and binding['review']['sha256'] == '56cb40775edcde58b34965fe9175c01feb4c3aa9bb72b598b2337fe988b8e4f2',
            'same expanded80 qualified60/review pins required, no reference-based reselection')


def validate_reference_binding(binding):
    require(isinstance(binding,dict) and set(binding) == {'path','sha256','receipt','receipt_sha256',
        'manifest','manifest_sha256','windows'} and binding['windows'] == 60,
        'new completed60 reference index binding required, not old train6')
    require(binding['path'] == str(ROOT/'data/reference/train.references.jsonl')
        and binding['receipt'] == binding['path']+'.receipt.json'
        and binding['manifest'] == str(ROOT/'data/manifests/train.reference.jsonl'), 'new stage reference locations differ')


def validate_request(request, directory):
    import train_expanded_reference_teacher as entry
    directory = Path(directory); arm,step = request.get('arm'),request.get('stop_after_step')
    require(request.get('mode') == MODE and request.get('experiment_revision') == REVISION
        and request.get('source_revision') == SOURCE_REVISION, 'fixed new reference pair revision required')
    require(request.get('job_id') == directory.name == job(arm,step) and directory == ROOT/job(arm,step),
        'unique exact arm job required')
    maximum = request.get('max_seconds')
    require(type(maximum) is int and 60 <= maximum <= (900 if step == 2 else 3600)
        and request.get('cleanup_seconds') == 60, 'fixed finite900/3600+60 watchdog required')
    original = ROOT/job(arm,2)
    require(request.get('config') == str(original/'training.yaml')
        and request.get('output_dir') == str(original/'training')
        and request.get('model_dir') == str(MODEL)
        and request.get('saved_tensors_on_cpu') is True, 'fixed exact60/base/new arm directory required')
    require(request.get('resume') == (None if step == 2 else str(original/'training/checkpoints/step-0000002.pt')),
        'only actual same-arm step2 strict resume permitted')
    require(request.get('method_contract') == entry.method_contract(arm)
        and request.get('protocol_cap_gpu_seconds') == CAP
        and request.get('protocol_incremental_bytes') == BYTES, 'fixed method and cumulative stage bounds required')
    allowed = {'mode','experiment_revision','source_revision','job_id','arm','stop_after_step','max_seconds',
        'cleanup_seconds','config','output_dir','model_dir','saved_tensors_on_cpu','resume','parent_lineage',
        'base_identity','data_binding','reference_binding','method_contract','protocol_cap_gpu_seconds',
        'protocol_incremental_bytes','config_sha256'}
    if step == 20:
        allowed |= {'predecessor_job','predecessor_binding','both_profiles_qualified','measured_slowest_step_seconds'}
        require(request.get('predecessor_job') == str(original) and request.get('both_profiles_qualified') is True
            and maximum == measured_seconds(request.get('measured_slowest_step_seconds')),
            'both settled profiles and own measured bound required')
    require(set(request) <= allowed and allowed-{'config_sha256'} <= set(request), 'unexpected override/missing field')
    require(step != 2 or maximum == 900, 'fixed actual2-step profile cap900')
    entry.validate_lineage(request.get('parent_lineage'))
    validate_data_locations(request['data_binding'])
    entry.validate_data_binding(request['data_binding'])
    validate_reference_binding(request['reference_binding'])
    require(request['parent_lineage']['qualified_data_binding'] == request['data_binding'], 'same parent exact60 required')
    require(request.get('base_identity') == request['parent_lineage']['base_identity'], 'same official parent base required')


def prepare(*, output, arm, source_revision=SOURCE_REVISION, max_seconds=900, data_binding=None):
    import yaml
    import train_expanded_reference_teacher as entry
    output = contained(output,ROOT)
    require(output == ROOT/job(arm,2) and not output.exists(), 'new fixed profile identity required')
    data_binding = read(DATA_SOURCE/'training-data-binding.json') if data_binding is None else data_binding
    validate_data_locations(data_binding)
    lineage = entry.parent_binding()
    # Read the real parent config, but do not load weights during CPU preparation.
    config = entry.expanded.load_config(entry.parent_paths()['config'])
    config.data.data_factory = entry.FACTORY
    config.optimizer.lora_lr = 0.
    config.training.max_steps = 20; config.training.output_dir = str(output/'training')
    entry.validate_config(config,output_dir=output/'training',arm=arm,data_binding=data_binding)
    references = entry.reference_binding()
    sampling = entry.sampling_contract(config.training,data_binding,references,arm=arm)
    request = dict(mode=MODE,experiment_revision=REVISION,source_revision=source_revision,job_id=output.name,arm=arm,
        stop_after_step=2,max_seconds=max_seconds,cleanup_seconds=60,config=str(output/'training.yaml'),
        output_dir=str(output/'training'),model_dir=str(MODEL),saved_tensors_on_cpu=True,
        resume=None,parent_lineage=lineage,base_identity=lineage['base_identity'],data_binding=data_binding,
        reference_binding=references,
        method_contract=entry.method_contract(arm),protocol_cap_gpu_seconds=CAP,protocol_incremental_bytes=BYTES)
    validate_request(request,output); output.mkdir(parents=True,exist_ok=False)
    path = output/'training.yaml'; path.write_text(yaml.safe_dump(config.to_dict(),sort_keys=False),encoding='utf-8')
    request['config_sha256'] = sha256(path); write_json(output/'request.json',request)
    write_json(output/'preparation.json',dict(status='cpu_prepared',gpu_launched=False,model_weights_loaded=False,
        stage=entry.STAGE,arm=arm,parent_lineage=lineage,base_identity=lineage['base_identity'],
        sampling_contract=sampling,training_started=False,
        parent_teacher_qualified=False,quality_improvement_verified=False))
    return request


def completed_endpoint(arm, step=20):
    """Read only actual settled small records. No checkpoint load or quality claim."""
    import train_expanded_reference_teacher as entry
    directory = ROOT/job(arm,step)
    request,result,settlement = (read(directory/(n+'.json')) for n in ('request','result','settlement'))
    validate_request(request,directory)
    require(result.get('status') == 'completed' and result.get('mode') == MODE and result.get('arm') == arm
        and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed', 'actual completed/settled arm required')
    ex = result['execution']; q = ex.get('qualification',{})
    require(ex.get('stage') == entry.STAGE and ex.get('checkpoint_schema') == entry.CHECKPOINT_SCHEMA
        and ex.get('arm') == arm and ex.get('step') == ex.get('optimizer_internal_step') == step
        and ex.get('micro_batches_consumed') == ex.get('new_stage_micro_batches_consumed') == step*8
        and ex.get('sampling_cursor') == step*8 and ex.get('historical_micro_batches_before_stage') == 1312
        and ex.get('lifetime_micro_batches_consumed') == 1312+step*8 and ex.get('configured_max_steps') == 20
        and ex.get('resume_step') == (0 if step == 2 else 2), 'new optimizer/history/strict same-arm counters differ')
    require(ex.get('parent_lineage') == request['parent_lineage'] and ex.get('base_identity') == request['base_identity']
        and ex.get('sampling_contract',{}).get('qualified_data_binding') == request['data_binding']
        and ex.get('sampling_contract',{}).get('reference_binding') == request['reference_binding']
        and ex.get('sampling_contract',{}).get('namespace') == entry.NAMESPACE
        and ex.get('reference_enabled') is (arm == 'reference')
        and ex.get('method_contract') == entry.method_contract(arm)
        and ex.get('precision_contract') == entry.precision_contract(arm), 'actual new arm contract mismatch')
    require(ex.get('checkpoint_path') == str(ROOT/job(arm,2)/f'training/checkpoints/step-{step:07d}.pt')
        and Path(ex['checkpoint_path']).is_file()
        and sha256(request['config']) == request['config_sha256'], 'exact configured trajectory checkpoint required')
    require(all(q.get(k) is True for k in ('master_updated','adapter_updated','gate_updated','gradients_finite',
        'adapter_task_gradient_nonzero','gate_task_gradient_nonzero','native_state_restore_matched','model_master_projection_equal'))
        and q.get('lora_trainable') is (False) and q.get('frozen_lora_has_gradients') is False
        and 0 < q.get('peak_reserved_bytes',0) < 23*1024**3, 'actual gradient/update/native state/resource gate failed')
    require(q.get('lora_B_updated') is None and q.get('step2_lora_A_task_gradient_nonzero') is None,
        'both arms freeze LoRA and cannot claim optimized LoRA')
    return request,ex,settlement


def completed_profile(arm):
    request,execution,_ = completed_endpoint(arm,2)
    path = Path(request['output_dir'])/'metrics.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    require(len(rows) in (2,20) and [r.get('step') for r in rows] == list(range(1,len(rows)+1)),
        'only exact contiguous completed2/20 trace can supply original profile')
    if len(rows) == 20:
        following,_,_ = completed_endpoint(arm,20)
        require(all(following[k] == request[k] for k in ('config','config_sha256','output_dir','data_binding','reference_binding','parent_lineage'))
            and following['resume'] == str(Path(request['output_dir'])/'checkpoints/step-0000002.pt'),
            'extended trace must have actual settled same-arm continuation')
    task = rows[1].get('task_gradient_evidence',{}); maxima = task.get('full_tensor_max_abs',{})
    require(all(task.get(k) is True for k in ('measured_before_optimizer','adapter_task_gradient_nonzero',
        'gate_task_gradient_nonzero','weight_decay_excluded','optimizer_momentum_excluded'))
        and maxima.get('adapter',0) > 0 and maxima.get('gate',0) > 0
        and task.get('frozen_lora_has_gradients') is False, 'actual step2 task gradients required, not decay-only changes')
    times = [r.get('optimizer_step_seconds') for r in rows[:2]]
    require(all(type(t) in (int,float) and math.isfinite(t) and t > 0 for t in times), 'actual finite two-step measurements required')
    return request,execution,max(times)


def both_profiles():
    profiles = {arm:completed_profile(arm) for arm in ARMS}
    a,b = (profiles[arm][0] for arm in ARMS)
    require(all(a[key] == b[key] for key in ('parent_lineage','data_binding','reference_binding')),
            'same expanded80 parent/train60/reference cache required')
    policies = [dict(profiles[arm][1]['sampling_contract']) for arm in ARMS]
    for policy in policies: policy.pop('reference_enabled',None)
    require(policies[0] == policies[1], 'shared sampling/noise namespace differs')
    paired = []
    for arm in ARMS:
        rows = [json.loads(line) for line in (Path(profiles[arm][0]['output_dir'])/'metrics.jsonl').read_text().splitlines() if line.strip()]
        samples = [s for row in rows[:2] for s in row['micro_samples']]
        require(len(samples) == 16 and [s.get('absolute_sample_index') for s in samples] == list(range(16)),
                'actual profile must record16 stage-absolute micro draws')
        require(all(s.get('reference_enabled') is (arm == 'reference')
                    and s.get('reference_valid_slots') in (0,1) for s in samples), 'actual reference forwarding trace missing')
        require(arm != 'reference' or any(s['reference_valid_slots'] == 1 for s in samples),
                'reference profile must actually use a valid slot')
        paired.append([{k:v for k,v in s.items() if k not in ('loss','reference_enabled','reference_valid_slots')}
                       for s in samples])
    require(paired[0] == paired[1], 'actual profile sample/noise/timestep draws differ')
    return profiles


def timing_through(arm, step=2):
    require(step == 2, 'only measured2 profile may set the single continuation')
    return completed_profile(arm)[2]


def measured_seconds(slowest):
    require(type(slowest) in (int,float) and math.isfinite(slowest) and slowest > 0, 'finite actual positive profile timing required')
    value = math.ceil(slowest*18*1.35+180)
    require(value <= 3600, 'measured18-step continuation exceeds3600; do not extend or change configuration')
    return value


def resume_request(*, original, output, stop_after_step=20, max_seconds):
    import train_expanded_reference_teacher as entry
    original,output = contained(original,ROOT),contained(output,ROOT)
    arm = next((a for a in ARMS if original == ROOT/job(a,2)),None)
    require(arm in ARMS and stop_after_step == 20 and output == ROOT/job(arm,20)
        and not output.exists(), 'new same-arm fixed continuation identity required')
    profiles = both_profiles(); profile,ex,slowest = profiles[arm]
    require(type(max_seconds) is int and max_seconds == measured_seconds(slowest), 'own measured finite continuation bound required')
    require(entry.parent_binding() == profile['parent_lineage']
        and entry.reference_binding() == profile['reference_binding'], 'actual common parent/cache changed')
    checkpoint = original/'training/checkpoints/step-0000002.pt'
    require(ex['checkpoint_path'] == str(checkpoint) and checkpoint.is_file(), 'actual same-arm predecessor checkpoint missing')
    request = dict(profile,job_id=output.name,resume=str(checkpoint),stop_after_step=20,max_seconds=max_seconds,
        predecessor_job=str(original),predecessor_binding=record_binding(original),both_profiles_qualified=True,
        measured_slowest_step_seconds=slowest)
    validate_request(request,output); output.mkdir(parents=True,exist_ok=False)
    write_json(output/'request.json',request); return request


def run(request, directory, lease):
    import train_expanded_reference_teacher as entry
    from training.demo.backend import validate_active_authority, validate_lease
    began = time.monotonic(); directory = contained(directory,ROOT)
    validate_request(request,directory); validate_active_authority(lease)
    require(lease.get('gpu_index') == 2, 'new fixed pair uses original Creator GPU2')
    validate_lease(lease,max_seconds=request['max_seconds'],job_id=request['job_id'],
                   request_sha256=sha256(directory/'request.json'))
    require(sha256(request['config']) == request['config_sha256']
        and entry.reference_binding() == request['reference_binding'], 'training config/reference cache changed')
    arm,step = request['arm'],request['stop_after_step']
    if step == 20:
        profiles = both_profiles(); prior,ex,slowest = profiles[arm]
        require(request['max_seconds'] == measured_seconds(slowest) and prior['config_sha256'] == request['config_sha256']
            and ex['checkpoint_path'] == request['resume']
            and record_binding(ROOT/job(arm,2)) == request['predecessor_binding'], 'actual both-profile gate or strict resume changed')
    config = entry.load_config(request['config']); entry.validate_base_identity(request,config)
    execution = entry.launch(config,request['config'],arm=arm,lineage=request['parent_lineage'],
        data_binding=request['data_binding'],reference_binding=request['reference_binding'],
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
    follow.add_argument('--max-seconds',type=int,required=True); follow.add_argument('--stop-after-step',type=int,default=20)
    values = vars(parser.parse_args()); operation = values.pop('operation')
    print(json.dumps(prepare(**values) if operation == 'prepare' else resume_request(**values)))


if __name__ == '__main__': main()

