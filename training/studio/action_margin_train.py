"""CPU preparation and guarded action-margin 2->8->14->20 trajectory."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.studio.controlled_action_train import MODEL, RUNTIME

ROOT = RUNTIME / 'abot-retrain-20261003/teacher-action-margin-v1'
MODE = 'action_margin_train'
REVISION = 'expanded80-same160-draws-action-margin-new20-v1'
SOURCE_REVISION = 'creator-action-margin-v1'
PREVIOUS = {2: 0, 8: 2, 14: 8, 20: 14}
CAP, BYTES = 18000, 12_000_000_000
DATA_SOURCE = RUNTIME / 'abot-retrain-20261003/teacher-expanded-data-v1'


def request_cap(step):
    require(type(step) is int and step in PREVIOUS, 'fixed expanded endpoint required')
    return CAP


def job(step):
    require(type(step) is int and step in PREVIOUS, 'fixed 2/8/14/20 endpoint required')
    return 'cam-profile2-v1' if step == 2 else f'cam-to{step}-v1'


def paths(step=20):
    profile = ROOT / job(2)
    return dict(profile=profile, config=profile / 'training.yaml',
                checkpoint=profile / f'training/checkpoints/step-{step:07d}.pt', endpoint=ROOT / job(step))


def read(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size < 16_000_000, 'missing/oversized small record')
    return json.loads(path.read_text(encoding='utf-8'))


def measured_seconds(slowest, new_steps):
    require(type(slowest) in (int, float) and math.isfinite(slowest) and slowest > 0
            and type(new_steps) is int and new_steps == 6, 'actual positive timing and fixed step increment required')
    result = math.ceil(slowest * new_steps * 1.35 + 180)
    require(result <= 3600, 'measured segment exceeds 3600; stop without expanding the cap')
    return result


def validate_data_locations(binding):
    from training.studio.expanded_reference_train import validate_data_locations as original
    original(binding)


def validate_request(request, directory):
    import train_action_margin_teacher as entry
    step = request.get('stop_after_step')
    p = paths(step)
    expected = dict(mode=MODE, experiment_revision=REVISION, source_revision=SOURCE_REVISION,
        job_id=job(step), config=str(p['config']), output_dir=str(p['profile'] / 'training'),
        model_dir=str(MODEL), saved_tensors_on_cpu=True, cleanup_seconds=60,
        resume=None if step == 2 else str(paths(PREVIOUS[step])['checkpoint']),
        protocol_cap_gpu_seconds=request_cap(step), protocol_incremental_bytes=BYTES,
        reference_enabled=False, method_contract=entry.method_contract(), stop_after_step=step)
    require(Path(directory) == ROOT / job(step) and all(request.get(k) == v for k, v in expected.items()),
            'fixed expanded job/config/trajectory contract changed')
    allowed = set(expected) | {'max_seconds', 'config_sha256', 'parent_lineage', 'base_identity', 'data_binding'}
    if step != 2:
        allowed |= {'predecessor_job', 'predecessor_binding', 'measured_slowest_step_seconds'}
    require(set(request) <= allowed and allowed - {'config_sha256'} <= set(request), 'unexpected override/missing request field')
    maximum = request['max_seconds']
    require(type(maximum) is int and 60 <= maximum <= (900 if step == 2 else 3600), 'finite segment cap required')
    require(step != 2 or maximum == 900, 'profile is fixed at 900 seconds')
    entry.validate_lineage(request['parent_lineage'])
    validate_data_locations(request['data_binding'])
    entry.validate_data_binding(request['data_binding'])  # No review/no explicit accept => reject before prepare.
    require(request['base_identity'] == request['parent_lineage']['base_identity'], 'same official base required')
    if step != 2:
        require(request['predecessor_job'] == str(ROOT / job(PREVIOUS[step]))
                and maximum == measured_seconds(request['measured_slowest_step_seconds'], step - PREVIOUS[step]),
                'exact predecessor and measured segment bound required')


def prepare(*, output, data_binding=None, source_revision=SOURCE_REVISION, max_seconds=900):
    import yaml
    import train_action_margin_teacher as entry
    output = contained(output, ROOT)
    require(source_revision == SOURCE_REVISION and output == paths(2)['profile'] and not output.exists(),
            'new fixed profile identity required')
    data_binding = read(DATA_SOURCE / "training-data-binding.json") if data_binding is None else data_binding
    validate_data_locations(data_binding)
    entry.validate_data_binding(data_binding)
    lineage = entry.parent_binding()
    config = entry.common.expanded.load_config(entry.parent_paths()['config'])
    config.data.data_factory = entry.FACTORY
    config.data.manifest_path, config.data.manifest_sha256 = data_binding['subset']['path'], data_binding['subset']['sha256']
    config.optimizer.lora_lr = 0.
    config.training.max_steps = 20
    config.training.output_dir = str(output / 'training')
    entry.validate_config(config, output_dir=output / 'training', data_binding=data_binding)
    sampling = entry.sampling_contract(config.training, data_binding)
    request = dict(mode=MODE, experiment_revision=REVISION, source_revision=source_revision, job_id=output.name,
        stop_after_step=2, max_seconds=max_seconds, cleanup_seconds=60, config=str(output / 'training.yaml'),
        output_dir=str(output / 'training'), model_dir=str(MODEL), saved_tensors_on_cpu=True,
        resume=None, parent_lineage=lineage, base_identity=lineage['base_identity'], data_binding=data_binding,
        method_contract=entry.method_contract(), reference_enabled=False,
        protocol_cap_gpu_seconds=CAP, protocol_incremental_bytes=BYTES)
    validate_request(request, output)
    output.mkdir(parents=True, exist_ok=False)
    paths(2)['config'].write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding='utf-8')
    request['config_sha256'] = sha256(paths(2)['config'])
    write_json(output / 'request.json', request)
    write_json(output / 'preparation.json', dict(status='cpu_prepared', model_weights_loaded=False,
        gpu_launched=False, training_started=False, sampling_contract=sampling, parent_lineage=lineage,
        parent_teacher_qualified=False, quality_improvement_verified=False, automatic_next_stage=False))
    return request


def record_binding(directory):
    return {name + '_sha256': sha256(Path(directory) / (name + '.json')) for name in ('request', 'result', 'settlement')}


def completed_endpoint(step=20):
    import train_action_margin_teacher as entry
    p = paths(step)
    request, result, settlement = (read(p['endpoint'] / (n + '.json')) for n in ('request', 'result', 'settlement'))
    validate_request(request, p['endpoint'])
    require(result.get('status') == 'completed' and result.get('mode') == MODE
            and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed',
            'actual completed/settled new trajectory endpoint required')
    ex, q = result['execution'], result['execution'].get('qualification', {})
    require(ex.get('stage') == entry.STAGE and ex.get('checkpoint_schema') == entry.CHECKPOINT_SCHEMA
            and ex.get('step') == ex.get('optimizer_internal_step') == step
            and ex.get('micro_batches_consumed') == ex.get('new_stage_micro_batches_consumed')
            == ex.get('sampling_cursor') == step * 8
            and ex.get('historical_micro_batches_before_stage') == 1312
            and ex.get('lifetime_micro_batches_consumed') == 1312 + step * 8
            and ex.get('configured_max_steps') == 20 and ex.get('resume_step') == PREVIOUS[step],
            'new optimizer, absolute cursor or strict predecessor counters differ')
    require(ex.get('parent_lineage') == request['parent_lineage'] and ex.get('base_identity') == request['base_identity']
            and ex.get('sampling_contract', {}).get('qualified_data_binding') == request['data_binding']
            and ex.get('method_contract') == entry.method_contract()
            and ex.get('precision_contract') == entry.precision_contract()
            and ex.get('arm') == entry.ARM
            and ex.get('sampling_contract', {}).get('namespace') == entry.NAMESPACE, 'actual new-stage contract differs')
    require(ex.get('checkpoint_path') == str(p['checkpoint']) and p['checkpoint'].is_file()
            and sha256(p['config']) == request['config_sha256'], 'actual configured endpoint checkpoint required')
    require(all(q.get(k) is True for k in ('master_updated', 'adapter_updated', 'gate_updated', 'gradients_finite',
        'adapter_task_gradient_nonzero', 'gate_task_gradient_nonzero', 'native_state_restore_matched',
        'model_master_projection_equal')) and q.get('lora_trainable') is False
        and q.get('frozen_lora_has_gradients') is False and 0 < q.get('peak_reserved_bytes', 0) < 23 * 1024**3,
        'actual action-margin gradient/update/native-state/resource gate failed')
    require(ex.get('reference_enabled') is False, 'no reference is allowed in action-margin stage')
    return request, ex, settlement


def timing_through(step):
    request, _, _ = completed_endpoint(step)
    for previous in PREVIOUS:
        if previous < step:
            prior, _, _ = completed_endpoint(previous)
            require(all(prior[k] == request[k] for k in ('config_sha256', 'parent_lineage', 'data_binding')),
                    'settled same-trajectory history differs')
    rows = [json.loads(line) for line in (Path(request['output_dir']) / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    require(len(rows) == step and [r.get('step') for r in rows] == list(range(1, step + 1)),
            'exact contiguous actual trace required before next segment')
    times = [r.get('optimizer_step_seconds') for r in rows]
    require(all(type(t) in (int, float) and math.isfinite(t) and t > 0 for t in times), 'finite measured times required')
    task = rows[1].get('task_gradient_evidence', {})
    require(all(task.get(k) is True for k in ('measured_before_optimizer', 'adapter_task_gradient_nonzero',
        'gate_task_gradient_nonzero', 'weight_decay_excluded', 'optimizer_momentum_excluded')),
        'profile must show actual task gradients, not only weight-decay/momentum changes')
    return max(times)


def resume_request(*, output, stop_after_step, max_seconds):
    import train_action_margin_teacher as entry
    step, output = stop_after_step, contained(output, ROOT)
    require(step in (8, 14, 20) and output == ROOT / job(step) and not output.exists(), 'new fixed continuation required')
    previous = PREVIOUS[step]
    prior, _, _ = completed_endpoint(previous)
    slowest = timing_through(previous)
    require(max_seconds == measured_seconds(slowest, step - previous), 'all completed steps must determine timing bound')
    require(entry.parent_binding() == prior['parent_lineage'], 'weights-only original parent binding changed')
    request = dict(prior, job_id=output.name, resume=str(paths(previous)['checkpoint']),
        protocol_cap_gpu_seconds=CAP,
        stop_after_step=step, max_seconds=max_seconds, predecessor_job=str(ROOT / job(previous)),
        predecessor_binding=record_binding(ROOT / job(previous)), measured_slowest_step_seconds=slowest)
    validate_request(request, output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'request.json', request)
    return request


def run(request, directory, lease):
    import train_action_margin_teacher as entry
    from training.demo.backend import validate_active_authority, validate_lease
    began = time.monotonic()
    directory = contained(directory, ROOT)
    validate_request(request, directory)
    validate_active_authority(lease)
    require(lease.get('gpu_index') == 2, 'only original Creator GPU2 guard')
    validate_lease(lease, max_seconds=request['max_seconds'], job_id=request['job_id'], request_sha256=sha256(directory / 'request.json'))
    require(sha256(request['config']) == request['config_sha256'], 'fixed config changed')
    step = request['stop_after_step']
    if step != 2:
        previous = PREVIOUS[step]
        require(record_binding(ROOT / job(previous)) == request['predecessor_binding']
                and timing_through(previous) == request['measured_slowest_step_seconds'], 'actual predecessor/timing changed')
    config = entry.load_config(request['config'])
    entry.validate_base_identity(request, config)
    execution = entry.launch(config, request['config'], lineage=request['parent_lineage'], data_binding=request['data_binding'],
        stop_after_step=step, resume=request['resume'], shared_lease=lease, shared_max_seconds=request['max_seconds'],
        shared_request_sha256=sha256(directory / 'request.json'))
    require(execution['stage'] == entry.STAGE and execution['step'] == step, 'wrong expanded training endpoint')
    return dict(status='completed', mode=MODE, execution=execution, elapsed_seconds=time.monotonic() - began,
        quality_improvement_verified=False, generalization_verified=False, default_model_changed=False, automatic_next_stage=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    first = commands.add_parser('prepare')
    first.add_argument('--output', required=True)
    first.add_argument('--data-binding-file')
    first.add_argument('--source-revision', default=SOURCE_REVISION)
    first.add_argument('--max-seconds', type=int, default=900)
    follow = commands.add_parser('resume')
    follow.add_argument('--output', required=True)
    follow.add_argument('--stop-after-step', type=int, required=True)
    follow.add_argument('--max-seconds', type=int, required=True)
    args = vars(parser.parse_args())
    operation = args.pop('operation')
    if operation == 'prepare':
        binding_file = args.pop('data_binding_file')
        args['data_binding'] = read(binding_file) if binding_file else None
    print(json.dumps(prepare(**args) if operation == 'prepare' else resume_request(**args)))
