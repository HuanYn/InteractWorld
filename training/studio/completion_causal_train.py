"""Bounded expanded80 -> causal TF migration, independent of closed R005 jobs."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.studio.controlled_action_train import MODEL, RUNTIME

ROOT = RUNTIME / 'abot-completion-20261007/causal-tf-v1'
MODE = 'completion_causal_train'
SOURCE_REVISION = 'creator-abot-completion-tf-v1'
REVISION = 'expanded80-frozen-action-fp32-lora-causal-tf20-v1'
PREVIOUS = {2: 0, 8: 2, 14: 8, 20: 14}
CAP, BYTES = 21600, 8_000_000_000
DATA_SOURCE = RUNTIME / 'abot-retrain-20261003/teacher-expanded-data-v1/training-data-binding.json'


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size < 16_000_000,
            'missing/unsafe small record: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def job(step):
    require(type(step) is int and step in PREVIOUS, 'fixed 2/8/14/20 endpoint required')
    return 'ccf-profile2-v1' if step == 2 else f'ccf-to{step}-v1'


def paths(step=20):
    profile = ROOT / job(2)
    return dict(profile=profile, config=profile / 'training.yaml',
        checkpoint=profile / f'training/checkpoints/step-{step:07d}.pt', endpoint=ROOT / job(step))


def measured_seconds(slowest, new_steps=6):
    require(type(slowest) in (int, float) and math.isfinite(slowest) and slowest > 0
            and type(new_steps) is int and new_steps == 6, 'positive actual timing/fixed6 steps required')
    result = math.ceil(slowest * new_steps * 1.35 + 180)
    require(result <= 3600, 'measured continuation exceeds finite job cap; stop without extending')
    return result


def validate_request(request, directory):
    import train_completion_causal_tf as entry
    from training.studio.expanded_reference_train import validate_data_locations
    step = request.get('stop_after_step')
    p = paths(step)
    expected = dict(mode=MODE, experiment_revision=REVISION, source_revision=SOURCE_REVISION,
        job_id=job(step), config=str(p['config']), output_dir=str(p['profile'] / 'training'),
        model_dir=str(MODEL), saved_tensors_on_cpu=True, cleanup_seconds=60, stop_after_step=step,
        resume=None if step == 2 else str(paths(PREVIOUS[step])['checkpoint']),
        protocol_cap_gpu_seconds=CAP, protocol_incremental_bytes=BYTES,
        method_contract=entry.method_contract(), parent_teacher_qualified=False)
    require(Path(directory) == ROOT / job(step) and all(request.get(k) == v for k,v in expected.items()),
            'fixed causal trajectory/config/output contract differs')
    extra = {'max_seconds','config_sha256','parent_lineage','base_identity','data_binding'}
    if step != 2:
        extra |= {'predecessor_job','predecessor_binding','measured_slowest_step_seconds'}
    require(set(request) <= set(expected) | extra and set(expected) | (extra-{'config_sha256'}) <= set(request),
            'unexpected/missing causal request fields')
    maximum = request['max_seconds']
    require(type(maximum) is int and 60 <= maximum <= 3600 and (step != 2 or maximum == 1800),
            'finite profile1800/continuation<=3600 required')
    entry.validate_lineage(request['parent_lineage'])
    validate_data_locations(request['data_binding'])
    entry.validate_data_binding(request['data_binding'])
    require(request['base_identity'] == request['parent_lineage']['base_identity']
            and request['data_binding'] == request['parent_lineage']['qualified_data_binding'],
            'expanded80 original base/exact60 binding changed')
    if step != 2:
        require(request['predecessor_job'] == str(ROOT / job(PREVIOUS[step]))
                and maximum == measured_seconds(request['measured_slowest_step_seconds']),
                'strict predecessor/measured bound required')


def prepare(*, output, source_revision=SOURCE_REVISION, max_seconds=1800):
    import yaml
    import train_completion_causal_tf as entry
    output = contained(output, ROOT)
    require(output == paths(2)['profile'] and not output.exists() and source_revision == SOURCE_REVISION,
            'new fixed causal profile identity required')
    binding, lineage = read(DATA_SOURCE), entry.parent_binding()
    config = entry.make_config(binding, lineage)
    entry.validate_config(config, output_dir=output / 'training', data_binding=binding)
    sampling = entry.sampling_contract(config.training, binding)
    request = dict(mode=MODE, experiment_revision=REVISION, source_revision=source_revision,
        job_id=output.name, config=str(paths(2)['config']), output_dir=str(output / 'training'),
        model_dir=str(MODEL), saved_tensors_on_cpu=True, cleanup_seconds=60, stop_after_step=2,
        resume=None, max_seconds=max_seconds, protocol_cap_gpu_seconds=CAP, protocol_incremental_bytes=BYTES,
        method_contract=entry.method_contract(), parent_teacher_qualified=False,
        parent_lineage=lineage, base_identity=lineage['base_identity'], data_binding=binding)
    validate_request(request, output)
    output.mkdir(parents=True, exist_ok=False)
    paths(2)['config'].write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding='utf-8')
    request['config_sha256'] = sha256(paths(2)['config'])
    write_json(output / 'request.json', request)
    write_json(output / 'preparation.json', dict(status='cpu_prepared', sampling_contract=sampling,
        parent_lineage=lineage, model_weights_loaded=False, gpu_launched=False,
        experimental_unqualified_teacher=True, quality_improvement_verified=False, realtime_verified=False))
    return request


def record_binding(directory):
    return {name + '_sha256':sha256(Path(directory)/(name+'.json')) for name in ('request','result','settlement')}


def completed_endpoint(step=20):
    import train_completion_causal_tf as entry
    p = paths(step)
    request, result, settlement = (read(p['endpoint']/(n+'.json')) for n in ('request','result','settlement'))
    validate_request(request, p['endpoint'])
    require(result.get('status') == 'completed' and result.get('mode') == MODE
            and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed',
            'completed/settled causal endpoint required')
    ex, q = result['execution'], result['execution'].get('qualification', {})
    require(ex.get('stage') == entry.STAGE and ex.get('checkpoint_schema') == entry.CHECKPOINT_SCHEMA
            and ex.get('step') == step and ex.get('resume_step') == PREVIOUS[step], 'wrong causal endpoint')
    require(ex.get('parent_lineage') == request['parent_lineage']
            and ex.get('base_identity') == request['base_identity']
            and ex.get('method_contract') == entry.method_contract(), 'actual causal lineage/method differs')
    require(all(ex.get(k) == v for k,v in entry.counters(step).items()), 'actual stage cursor/history differs')
    require(ex.get('sampling_contract', {}).get('qualified_data_binding') == request['data_binding']
            and ex.get('sampling_contract', {}).get('namespace') == entry.NAMESPACE, 'causal data/noise namespace differs')
    require(ex.get('checkpoint_path') == str(p['checkpoint']) and p['checkpoint'].is_file()
            and sha256(p['config']) == request['config_sha256'], 'checkpoint/config missing or changed')
    require(all(q.get(k) is True for k in ('lora_updated','gradients_finite','lora_B_task_gradient_nonzero',
                'lora_A_task_gradient_nonzero','adapter_unchanged','gate_unchanged','native_state_restore_matched'))
            and q.get('frozen_action_has_gradients') is False
            and 0 < q.get('peak_reserved_bytes',0) < 23*1024**3, 'actual causal update/resource gate failed')
    return request, ex, settlement


def timing_through(step):
    request, _, _ = completed_endpoint(step)
    for previous in PREVIOUS:
        if previous < step:
            prior, _, _ = completed_endpoint(previous)
            require(all(prior[k] == request[k] for k in ('config_sha256','parent_lineage','data_binding')),
                    'same-trajectory history differs')
    rows = [json.loads(line) for line in (Path(request['output_dir'])/'metrics.jsonl').read_text().splitlines() if line.strip()]
    require(len(rows) == step and [r.get('step') for r in rows] == list(range(1,step+1)),
            'exact contiguous measured steps required')
    times = [r.get('optimizer_step_seconds') for r in rows]
    require(all(type(t) in (int,float) and math.isfinite(t) and t>0 for t in times), 'invalid measured step time')
    return max(times)


def resume_request(*, output, stop_after_step, max_seconds):
    step, output = stop_after_step, contained(output, ROOT)
    require(step in (8,14,20) and output == ROOT/job(step) and not output.exists(), 'new continuation identity required')
    previous = PREVIOUS[step]
    prior, _, _ = completed_endpoint(previous)
    slowest = timing_through(previous)
    require(max_seconds == measured_seconds(slowest), 'use all measured completed step timings')
    request = dict(prior, job_id=output.name, resume=str(paths(previous)['checkpoint']), stop_after_step=step,
        max_seconds=max_seconds, predecessor_job=str(ROOT/job(previous)),
        predecessor_binding=record_binding(ROOT/job(previous)), measured_slowest_step_seconds=slowest)
    validate_request(request, output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'request.json', request)
    return request


def run(request, directory, lease):
    import train_completion_causal_tf as entry
    from training.demo.backend import validate_active_authority, validate_lease
    began = time.monotonic()
    directory = contained(directory, ROOT)
    validate_request(request, directory)
    validate_active_authority(lease)
    require(lease.get('gpu_index') == 2, 'this finite chain is fixed to original Creator GPU2')
    validate_lease(lease,max_seconds=request['max_seconds'],job_id=request['job_id'],request_sha256=sha256(directory/'request.json'))
    require(sha256(request['config']) == request['config_sha256'], 'causal config changed')
    step = request['stop_after_step']
    if step != 2:
        previous = PREVIOUS[step]
        require(record_binding(ROOT/job(previous)) == request['predecessor_binding']
                and timing_through(previous) == request['measured_slowest_step_seconds'], 'settled predecessor changed')
    config = entry.load_config(request['config'])
    entry.validate_base_identity(request, config)
    execution = entry.launch(config, request['config'], lineage=request['parent_lineage'], data_binding=request['data_binding'],
        stop_after_step=step,resume=request['resume'],shared_lease=lease,shared_max_seconds=request['max_seconds'],
        shared_request_sha256=sha256(directory/'request.json'))
    require(execution['stage'] == entry.STAGE and execution['step'] == step, 'wrong causal endpoint')
    return dict(status='completed',mode=MODE,execution=execution,elapsed_seconds=time.monotonic()-began,
        parent_teacher_qualified=False,quality_improvement_verified=False,realtime_verified=False,default_model_changed=False)
