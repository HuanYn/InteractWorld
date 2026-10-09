"""Complete-state TF20 -> TF160 continuation under the original Creator ledger.

Only this new module owns v2 requests.  It never edits a v1 record, silently
restarts sampling, or accepts caller-provided accounting as budget evidence.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.studio.controlled_action_train import MODEL, RUNTIME
from training.studio import completion_causal_train as old

ROOT = RUNTIME / 'abot-completion-20261007/causal-tf-sufficiency-v2'
MODE = 'completion_causal_sufficiency_train'
TRAIN_MODE = MODE
SOURCE_REVISION = 'creator-completion-tf-sufficiency-v2'
REVISION = 'tf20-complete-state-to160-same-absolute-trajectory-v2'
CAP, BYTES = 14 * 3600, 20_000_000_000
FOLLOWING_FIXED_GPU_SECONDS = (16 + 16 + 6 + 4) * 3600
GUARD_CONFIG = RUNTIME / 'guard-creator100-gpu2.json'
LEDGER = RUNTIME / 'budget.json'
SCHEDULE_SCHEMA = 'tf_sufficiency_remaining_schedule_v1'


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size < 16_000_000,
        'missing/unsafe small record: ' + str(path))
    value = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(value, dict), 'JSON object required: ' + str(path))
    return value


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def job(step):
    require(type(step) is int and 20 < step <= 160, 'continued TF endpoint21..160 required')
    return f'ccfs-to{step:07d}-v2'


def paths(step=160):
    require(type(step) is int and 20 <= step <= 160, 'TF20..160 endpoint required')
    if step == 20:
        return old.paths(20)
    return dict(profile=ROOT, config=ROOT/'training.yaml',
        checkpoint=ROOT/f'training/checkpoints/step-{step:07d}.pt', endpoint=ROOT/job(step))


def record_binding(directory):
    return {name+'_sha256': sha256(Path(directory)/(name+'.json'))
        for name in ('request', 'result', 'settlement')}


def source_binding():
    import train_completion_causal_tf_sufficiency as entry
    from training import causal_tf
    from utils import controlled_causal_action
    return {name:sha256(path) for name,path in dict(entry=entry.__file__, wrapper=__file__,
        v1_entry=entry.previous.__file__, causal_tf=causal_tf.__file__,
        controlled_causal_action=controlled_causal_action.__file__,
        causal_model=Path(entry.__file__).parent/'wan/modules/causal_model.py').items()}


def ensure_stage_records():
    """Persist immutable small bridge/config pins once, before any v2 worker."""
    import yaml
    import train_completion_causal_tf_sufficiency as entry
    require(ROOT == entry.ROOT, 'entry/studio stage roots differ')
    bridge = entry.bridge_source_binding()
    entry.validate_bridge_binding(bridge)
    config = entry.config_from_v1(entry.previous.load_config(bridge['config']))
    entry.validate_config(config, output_dir=ROOT/'training', data_binding=bridge['qualified_data_binding'])
    records = {'schema-bridge.json':bridge, 'source-binding.json':source_binding(),
        'data-binding.json':bridge['qualified_data_binding'], 'parent-lineage.json':bridge['parent_lineage']}
    ROOT.mkdir(parents=True, exist_ok=True)
    for name,value in records.items():
        target = ROOT/name
        if target.exists():
            require(read(target) == value, 'immutable v2 stage record changed: '+name)
        else:
            write_json(target, value)
    target = ROOT/'training.yaml'
    if target.exists():
        require(entry.load_config(target).to_dict() == config.to_dict(), 'immutable v2 configuration changed')
    else:
        with target.open('x', encoding='utf-8') as stream:
            stream.write(yaml.safe_dump(config.to_dict(), sort_keys=False))
    return config, bridge


def stage_binding():
    return {name+'_sha256':sha256(ROOT/name) for name in
        ('training.yaml', 'schema-bridge.json', 'source-binding.json', 'data-binding.json', 'parent-lineage.json')}


def ledger_snapshot():
    """Read-only original guard validation; no reserve, scan, migration or write."""
    from training.creator.guard import GuardConfig, CreatorGuard
    cfg = GuardConfig.load(GUARD_CONFIG)
    require(cfg.ledger == LEDGER and cfg.max_round_gpu_hours == 100
        and cfg.budget_scope == 'creator_100h_500gb_20261003'
        and cfg.gpu_index == 2 and cfg.supplemental_gpu_authority is None,
        'unchanged original Creator100/GPU2 guard and ledger required')
    guard = CreatorGuard(cfg)
    guard._authority()
    value = guard._ledger()
    require(value['storage_bytes_upper'] <= 500_000_000_000, 'Creator server500GB budget exceeded')
    return value, sha256(LEDGER)


def is_family_directory(value):
    path = Path(value).resolve()
    return any(path == root.resolve() or root.resolve() in path.parents for root in (old.ROOT,ROOT))


def fixed_formal_jobs():
    result = {}
    for step in (20,40,80,160):
        result[f'ccs-val-{step}-v2'] = dict(mode='completion_causal_sufficiency_validate', endpoint_step=step)
        if step > 20:
            for case in ('pair-j','holdout-l'):
                result[f'ccs-eval-{step}-{case}-v2'] = dict(mode='completion_causal_sufficiency_eval',
                    endpoint_step=step, case_id=case)
    return result


def formal_job_already_completed(identity, expected):
    directory = ROOT/identity
    if not (directory/'settlement.json').exists():
        return False
    request, result, settlement = (read(directory/(name+'.json'))
        for name in ('request','result','settlement'))
    require(request.get('job_id') == identity and all(request.get(k) == v for k,v in expected.items())
        and result.get('status') == 'completed' and result.get('mode') == expected['mode']
        and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed',
        'cannot omit an unfinished/failed fixed formal validation or video job')
    ledger,_ = ledger_snapshot()
    reservation = ledger['reservations'].get(settlement.get('reservation_id'),{})
    context = reservation.get('context',{})
    require(reservation.get('status') == 'settled' and context.get('job_id') == identity
        and context.get('job_directory') == str(directory)
        and context.get('request_sha256') == sha256(directory/'request.json')
        and reservation.get('charged_seconds') == settlement.get('charged_gpu_seconds')
        and reservation.get('settlement') == settlement,
        'omitted fixed formal job must have its original real ledger settlement')
    return True


def validate_complete_schedule(receipt):
    """Reject a falsely cheap plan with future checkpoints or videos deleted."""
    import train_completion_causal_tf_sufficiency as entry
    formal = fixed_formal_jobs()
    listed = set()
    cursor = receipt['last_completed_step']
    for row in receipt['jobs']:
        if row['mode'] == MODE:
            segment = entry.next_segment(cursor,receipt['measured_slowest_step_seconds'])
            require(row.get('start_step') == cursor and row.get('stop_after_step') == segment['stop_after_step']
                and row['max_seconds'] == segment['max_seconds'] and row['job_id'] == job(segment['stop_after_step']),
                'complete measured remaining training trajectory required, without missing/restarted segments')
            cursor = segment['stop_after_step']
        else:
            expected = formal.get(row['job_id'])
            require(expected is not None and all(row.get(k) == v for k,v in expected.items())
                and row['max_seconds'] == 1200 and row['endpoint_step'] <= cursor,
                'fixed formal validation/video must follow its actual trained endpoint')
            listed.add(row['job_id'])
    require(cursor == 160, 'remaining plan cannot silently omit TF160')
    for identity, expected in formal.items():
        require(identity in listed or formal_job_already_completed(identity,expected),
            'remaining plan omits fixed formal validation/video: '+identity)


def schedule_binding(request, directory):
    import train_completion_causal_tf_sufficiency as entry
    path = contained(request.get('schedule_receipt', ''), ROOT/'schedules')
    require(path == ROOT/'schedules'/f"schedule-{request['job_id']}.json"
        and sha256(path) == request.get('schedule_receipt_sha256'), 'fixed remaining-schedule receipt changed')
    receipt = read(path)
    expected = dict(schema=SCHEDULE_SCHEMA, root=str(ROOT), stage=entry.STAGE,
        source_revision=SOURCE_REVISION, gpu_index=2, current_job_id=request['job_id'],
        combined_v1_v2_cap_gpu_seconds=CAP, entry_sha256=sha256(entry.__file__),
        bridge_records=old.record_binding(old.paths(20)['endpoint']))
    require(all(receipt.get(k) == v for k,v in expected.items()), 'remaining schedule identity/source/bridge differs')
    require(type(receipt.get('last_completed_step')) is int
        and 20 <= receipt['last_completed_step'] <= 160
        and number(receipt.get('measured_slowest_step_seconds'))
        and receipt['measured_slowest_step_seconds'] > 0, 'measured schedule predecessor required')
    rows = receipt.get('jobs')
    require(isinstance(rows, list) and rows and all(isinstance(r, dict) for r in rows), 'complete remaining jobs required')
    require(len({r.get('job_id') for r in rows}) == len(rows), 'duplicate remaining global job identity')
    allowed = {MODE, 'completion_causal_sufficiency_validate', 'completion_causal_sufficiency_eval'}
    for row in rows:
        require(isinstance(row.get('job_id'), str) and row['job_id'] and '/' not in row['job_id']
            and row.get('mode') in allowed and type(row.get('max_seconds')) is int
            and 60 <= row['max_seconds'] <= 3600 and row.get('cleanup_seconds') == 60,
            'finite known remaining job contract required')
    current = rows[0]
    require(current['job_id'] == request['job_id'] and current['mode'] == request['mode']
        and current['max_seconds'] == request['max_seconds'] and current['cleanup_seconds'] == request['cleanup_seconds'],
        'schedule current request does not match first job')
    for key in ('start_step', 'stop_after_step', 'endpoint_step', 'case_id'):
        if key in current or key in request:
            require(current.get(key) == request.get(key), 'schedule current endpoint/case differs: '+key)
    validate_complete_schedule(receipt)
    remaining = sum(row['max_seconds']+row['cleanup_seconds'] for row in rows[1:])
    require(request.get('remaining_planned_gpu_seconds') == remaining,
        'remaining cost must be derived from the immutable complete schedule, not caller accounting')
    return receipt, remaining


def validate_combined_budget(request, directory, lease=None):
    """Reconcile both TF stages and all Creator reservations, counting each once."""
    import train_completion_causal_tf_sufficiency as entry
    schedule, remaining = schedule_binding(request, directory)
    ledger, ledger_sha = ledger_snapshot()
    family_charged, global_charged, active = 0., 0., []
    for rid, record in ledger['reservations'].items():
        require(record.get('status') in ('reserved', 'settled') and isinstance(record.get('context'), dict),
            'invalid original ledger reservation')
        context = record['context']
        family = is_family_directory(context.get('job_directory', ''))
        if record['status'] == 'settled':
            amount = record.get('charged_seconds')
            require(number(amount), 'invalid real settled charge')
            global_charged += amount
            if family:
                settlement = read(Path(context['job_directory'])/'settlement.json')
                require(settlement.get('status') == 'settled' and settlement.get('reservation_id') == rid
                    and settlement.get('charged_gpu_seconds') == amount
                    and settlement.get('outcome') == record.get('settlement', {}).get('outcome'),
                    'TF family disk settlement differs from original ledger')
                family_charged += amount
        else:
            require(number(record.get('reserved_seconds')), 'invalid real active reservation')
            active.append((rid,record))
    current_cost = request['max_seconds']+60
    if lease is None:
        require(not active, 'prepare requires zero active Creator reservations')
    else:
        require(len(active) == 1 and active[0][0] == lease.get('reservation_id'),
            'only this Creator request may be active during v2 execution')
        rid, current = active[0]
        context = current['context']
        require(current['lease'] == lease and context.get('job_id') == request['job_id']
            and context.get('job_directory') == str(directory)
            and context.get('request_sha256') == sha256(Path(directory)/'request.json')
            and context.get('max_seconds') == request['max_seconds']
            and current.get('reserved_seconds') == current_cost,
            'active ledger record must bind this exact request and cleanup reservation')
    # The current active reservation is included through current_cost, not added
    # a second time.  Other active work fails closed above.
    contract = dict(scope='completion_causal_tf_v1_and_sufficiency_v2_combined',
        combined_v1_v2_cap_gpu_seconds=CAP, combined_settled_gpu_seconds=family_charged,
        remaining_planned_gpu_seconds=remaining, current_job_cleanup_seconds=60)
    entry.validate_budget_contract(contract, shared_max_seconds=request['max_seconds'])
    require(global_charged+current_cost+remaining+FOLLOWING_FIXED_GPU_SECONDS <= 100*3600,
        'original100GPUh cannot cover TF plan and fixed ODE/DMD/runtime/evaluation remainder')
    return contract, dict(ledger=str(LEDGER), ledger_sha256=ledger_sha,
        global_settled_gpu_seconds=global_charged, family_settled_gpu_seconds=family_charged,
        current_reserved_gpu_seconds=current_cost, remaining_planned_gpu_seconds=remaining,
        following_fixed_gpu_seconds=FOLLOWING_FIXED_GPU_SECONDS,
        current_reservation_id=None if lease is None else lease['reservation_id'],
        current_active_count=len(active), schedule_receipt_sha256=request['schedule_receipt_sha256'])


def validate_request(request, directory):
    import train_completion_causal_tf_sufficiency as entry
    start, stop = request.get('start_step'), request.get('stop_after_step')
    segment = entry.next_segment(start, request.get('measured_slowest_step_seconds'))
    require(stop == segment['stop_after_step'], 'bounded measured next segment required')
    p = paths(stop)
    bridge = read(ROOT/'schema-bridge.json')
    expected = dict(mode=MODE, experiment_revision=REVISION, source_revision=SOURCE_REVISION,
        job_id=job(stop), config=str(p['config']), output_dir=str(ROOT/'training'), model_dir=str(MODEL),
        saved_tensors_on_cpu=True, cleanup_seconds=60, start_step=start, stop_after_step=stop,
        resume=str(paths(start)['checkpoint']), max_seconds=segment['max_seconds'],
        protocol_cap_gpu_seconds=CAP, protocol_incremental_bytes=BYTES,
        method_contract=entry.method_contract(), parent_teacher_qualified=False,
        schema_bridge=bridge, parent_lineage=bridge['parent_lineage'],
        data_binding=bridge['qualified_data_binding'], base_identity=bridge['parent_lineage']['base_identity'])
    require(Path(directory) == p['endpoint'] and all(request.get(k) == v for k,v in expected.items()),
        'fixed continued TF request differs')
    extra = {'config_sha256','stage_binding','predecessor_job','predecessor_binding',
        'measured_slowest_step_seconds','schedule_receipt','schedule_receipt_sha256',
        'remaining_planned_gpu_seconds','budget_contract','budget_preparation','headroom_bytes'}
    require(set(request) == set(expected)|extra and request['headroom_bytes'] == BYTES,
        'missing/unexpected v2 request or storage cap')
    require(request['config_sha256'] == sha256(p['config']) and request['stage_binding'] == stage_binding(),
        'immutable v2 config/bridge/source/data records changed')
    entry.validate_bridge_binding(bridge)
    require(request['predecessor_job'] == str(paths(start)['endpoint'])
        and request['predecessor_binding'] == record_binding(paths(start)['endpoint']),
        'immediate settled predecessor records changed')
    entry.validate_budget_contract(request['budget_contract'], shared_max_seconds=request['max_seconds'])
    schedule_binding(request, directory)


def completed_endpoint(step=160):
    if step == 20:
        # The return remains the original v1 execution, not a fabricated v2
        # optimizer run.  schema-bridge.json explicitly explains the transition.
        return old.completed_endpoint(20)
    import train_completion_causal_tf_sufficiency as entry
    p = paths(step)
    request, result, settlement = (read(p['endpoint']/(name+'.json'))
        for name in ('request','result','settlement'))
    validate_request(request, p['endpoint'])
    execution, q = result.get('execution',{}), result.get('execution',{}).get('qualification',{})
    require(result.get('status') == 'completed' and result.get('mode') == MODE
        and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed',
        'actual completed and settled v2 endpoint required')
    require(execution.get('stage') == entry.STAGE and execution.get('checkpoint_schema') == entry.CHECKPOINT_SCHEMA
        and execution.get('resume_step') == request['start_step']
        and all(execution.get(k) == v for k,v in entry.counters(step).items()), 'actual native step/cursor/history differs')
    require(execution.get('parent_lineage') == request['parent_lineage']
        and execution.get('base_identity') == request['base_identity']
        and execution.get('method_contract') == entry.method_contract()
        and execution.get('schema_migration') == request['schema_bridge']
        and execution.get('sampling_contract',{}).get('qualified_data_binding') == request['data_binding']
        and execution.get('sampling_contract',{}).get('namespace') == entry.NAMESPACE,
        'actual complete-state bridge/source/data binding differs')
    require(execution.get('checkpoint_path') == str(p['checkpoint']) and p['checkpoint'].is_file(),
        'actual complete v2 checkpoint missing')
    require(all(q.get(k) is True for k in ('lora_updated','gradients_finite','lora_B_task_gradient_nonzero',
        'lora_A_task_gradient_nonzero','adapter_unchanged','gate_unchanged','native_state_restore_matched'))
        and q.get('frozen_action_has_gradients') is False
        and 0 < q.get('peak_reserved_bytes',0) < 23*1024**3, 'actual continued TF update/resource gate failed')
    return request, execution, settlement


def timing_through(step):
    old.timing_through(20)
    times = []
    for path, first, last in ((Path(old.paths(20)['profile'])/'training/metrics.jsonl',1,20),
        (ROOT/'training/metrics.jsonl',21,step)):
        if last < first:
            continue
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        require([r.get('step') for r in rows] == list(range(first,last+1)), 'exact contiguous real step metrics required')
        for row in rows:
            value = row.get('optimizer_step_seconds')
            require(number(value) and value > 0, 'positive measured optimizer timing required')
            times.append(value)
    return max(times)


def prepare(*, output=None, job_id=None, max_seconds, headroom_bytes=BYTES, start_step,
        stop_after_step, measured_slowest_step_seconds, remaining_planned_gpu_seconds,
        schedule_receipt, source_revision=SOURCE_REVISION):
    import train_completion_causal_tf_sufficiency as entry
    output = paths(stop_after_step)['endpoint'] if output is None else contained(output, ROOT)
    require(output == paths(stop_after_step)['endpoint'] and not output.exists()
        and job_id in (None, output.name) and source_revision == SOURCE_REVISION,
        'new fixed v2 training identity required')
    prior, _, _ = completed_endpoint(start_step)
    slowest = timing_through(start_step)
    segment = entry.next_segment(start_step, slowest)
    require(measured_slowest_step_seconds == slowest and stop_after_step == segment['stop_after_step']
        and max_seconds == segment['max_seconds'] and headroom_bytes == BYTES,
        'use all actual predecessor timings and fixed20GB stage cap')
    config, bridge = ensure_stage_records()
    request = dict(mode=MODE, experiment_revision=REVISION, source_revision=SOURCE_REVISION,
        job_id=output.name, config=str(ROOT/'training.yaml'), config_sha256=sha256(ROOT/'training.yaml'),
        output_dir=str(ROOT/'training'), model_dir=str(MODEL), saved_tensors_on_cpu=True,
        cleanup_seconds=60, start_step=start_step, stop_after_step=stop_after_step,
        resume=str(paths(start_step)['checkpoint']), max_seconds=max_seconds,
        protocol_cap_gpu_seconds=CAP, protocol_incremental_bytes=BYTES, headroom_bytes=headroom_bytes,
        method_contract=entry.method_contract(), parent_teacher_qualified=False, schema_bridge=bridge,
        parent_lineage=bridge['parent_lineage'], data_binding=bridge['qualified_data_binding'],
        base_identity=bridge['parent_lineage']['base_identity'], stage_binding=stage_binding(),
        predecessor_job=str(paths(start_step)['endpoint']), predecessor_binding=record_binding(paths(start_step)['endpoint']),
        measured_slowest_step_seconds=slowest, schedule_receipt=str(schedule_receipt),
        schedule_receipt_sha256=sha256(schedule_receipt), remaining_planned_gpu_seconds=remaining_planned_gpu_seconds)
    contract, evidence = validate_combined_budget(request, output)
    request.update(budget_contract=contract, budget_preparation=evidence)
    validate_request(request, output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'request.json', request)
    write_json(output/'preparation.json', dict(status='cpu_prepared', gpu_launched=False,
        model_weights_loaded=False, complete_native_resume_required=True,
        training_length_adequacy_proven=False, budget_evidence=evidence))
    return request


def run(request, directory, lease):
    import train_completion_causal_tf_sufficiency as entry
    from training.demo.backend import validate_active_authority, validate_lease
    began = time.monotonic()
    directory = contained(directory, ROOT)
    validate_request(request, directory)
    validate_active_authority(lease)
    require(lease.get('gpu_index') == 2, 'finite formal TF chain is fixed to original Creator GPU2')
    validate_lease(lease, max_seconds=request['max_seconds'], job_id=request['job_id'],
        request_sha256=sha256(directory/'request.json'))
    completed_endpoint(request['start_step'])
    require(timing_through(request['start_step']) == request['measured_slowest_step_seconds'],
        'actual completed timing history changed')
    config, bridge = ensure_stage_records()
    contract, evidence = validate_combined_budget(request, directory, lease)
    require(contract == request['budget_contract'], 'actual family charges changed since preparation')
    write_json(directory/'budget-at-run.json', evidence)
    entry.previous.validate_base_identity(request, config)
    execution = entry.launch(config, request['config'], lineage=request['parent_lineage'],
        data_binding=request['data_binding'], bridge_binding=bridge, start_step=request['start_step'],
        stop_after_step=request['stop_after_step'], measured_slowest_step_seconds=request['measured_slowest_step_seconds'],
        resume=request['resume'], budget_contract=contract, shared_lease=lease,
        shared_max_seconds=request['max_seconds'], shared_request_sha256=sha256(directory/'request.json'))
    require(execution.get('stage') == entry.STAGE and execution.get('step') == request['stop_after_step'],
        'wrong formal TF native endpoint')
    return dict(status='completed', mode=MODE, execution=execution,
        elapsed_seconds=time.monotonic()-began, budget_evidence=evidence, parent_teacher_qualified=False,
        quality_improvement_verified=False, training_length_adequacy_proven=False,
        realtime_verified=False, default_model_changed=False)
