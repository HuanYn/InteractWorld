"""Guarded fixed two-arm controlled20 weights-only LoRA experiment."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.studio.controlled_action_train import DATA, MODEL, RUNTIME

ROOT = RUNTIME / 'abot-retrain-20261003/teacher-controlled-lora-pair-v1'
PARENT_JOB = RUNTIME / 'abot-retrain-20261003/teacher-controlled-v1/to20'
MODE = 'controlled_lora_pair_train'
REVISION = 'controlled-action-native97-lora-pair-v1'


def validate_request(request, directory):
    import train_controlled_lora_pair as entry
    require(request.get('mode') == MODE and request.get('experiment_revision') == REVISION,
        'explicit controlled LoRA-pair revision required')
    require(request.get('arm') in entry.ARMS, 'explicit frozen/trainable arm required')
    require(request.get('job_id') == directory.name and directory.name.startswith('clp-' + request['arm'] + '-'),
        'new globally unique arm identity required')
    require(type(request.get('max_seconds')) is int and 60 <= request['max_seconds'] <=3600,
        'finite <=3600s training watchdog required')
    require(type(request.get('stop_after_step')) is int and request['stop_after_step'] in (2,20),
        'pair endpoints are exactly2 and20')
    require((request.get('resume') is None and request['stop_after_step'] ==2)
        or (request.get('resume') is not None and request['stop_after_step'] ==20),
        'new arm first measures2 then strictly resumes to20')
    require(request.get('resume') is not None or request['max_seconds'] <=900,
        'two-step profile watchdog must not exceed900 seconds')
    require(request.get('saved_tensors_on_cpu') is True and request.get('data_root') == str(DATA)
        and request.get('model_dir') == str(MODEL), 'existing native97/base with CPU activation offload required')
    require(request.get('parent_job') == str(PARENT_JOB), 'only controlled20 parent job is accepted')
    require(isinstance(request.get('source_revision'), str) and bool(request['source_revision']), 'source revision required')
    require(not any(k in request for k in ('initialize_from', 'warm_start_from', 'length_transition_from', 'action_scale', 'ratio_cap')),
        'no historical-parent escape or action mechanism override allowed')
    entry._validate_lineage(request.get('parent_lineage'))
    require(request['parent_lineage']['parent_job'] == request['parent_job']
        and request['parent_lineage']['parent_checkpoint'] == request.get('parent_checkpoint')
        and request['parent_lineage']['parent_config'] == request.get('parent_config'), 'request parent lineage differs')
    require(request.get('method_contract') == entry.method_contract(request['arm']), 'fixed pair method changed')


def _parent_assets(config):
    import train_controlled_lora_pair as entry
    previous = json.loads((PARENT_JOB / 'request.json').read_text())
    result = json.loads((PARENT_JOB / 'result.json').read_text())
    settlement = json.loads((PARENT_JOB / 'settlement.json').read_text())
    require(result.get('status') == 'completed' and result.get('mode') == 'controlled_action_train'
        and result['execution']['step'] ==20 and result['execution']['stage'] == 'controlled_action_native97_v1'
        and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed',
        'parent controlled20 must be completed and settled')
    require(sha256(previous['config']) == previous['config_sha256'], 'parent training configuration changed')
    checkpoint = contained(result['execution']['checkpoint_path'], RUNTIME / 'abot-retrain-20261003/teacher-controlled-v1')
    require(checkpoint.is_file(), 'parent endpoint missing')
    state, lineage = entry.load_parent(parent_checkpoint=checkpoint, parent_config=previous['config'],
        parent_job=PARENT_JOB, config=config)
    del state
    return str(checkpoint), previous['config'], lineage


def prepare(*, output, arm, source_revision, max_seconds=900):
    import yaml
    import train_controlled_lora_pair as entry
    from training.config import ActionTeacherConfig
    require(arm in entry.ARMS, 'unknown LoRA arm')
    output = contained(output, ROOT)
    require(not output.exists(), 'new arm profile identity required')
    require(type(max_seconds) is int and 60 <= max_seconds <=900, 'profile maximum900 seconds')
    config = ActionTeacherConfig()
    config.model.base_model_path = str(MODEL)
    config.data.manifest_path = str(DATA / 'manifests/train.jsonl')
    config.data.manifest_sha256 = entry.MANIFEST_SHA256
    config.data.data_factory = entry.FACTORY
    config.data.num_frames, config.data.num_workers = 97, 0
    config.training.lora_parameter_dtype = 'float32'
    config.optimizer.adapter_lr = 1e-5
    config.optimizer.lora_lr = entry.LORA_LR if arm == 'trainable' else 0.
    config.training.max_steps = config.training.checkpoint_every =20
    config.training.output_dir = str(output / 'training')
    entry.validate_config(config, data_root=DATA, output_dir=output / 'training', model_dir=MODEL, arm=arm)
    parent_checkpoint, parent_config, lineage = _parent_assets(config)
    request = dict(job_id=output.name, mode=MODE, experiment_revision=REVISION, arm=arm,
        max_seconds=max_seconds, config=str(output / 'training.yaml'), data_root=str(DATA), model_dir=str(MODEL),
        output_dir=str(output / 'training'), resume=None, stop_after_step=2, saved_tensors_on_cpu=True,
        source_revision=source_revision, parent_job=str(PARENT_JOB), parent_checkpoint=parent_checkpoint,
        parent_config=parent_config, parent_lineage=lineage, method_contract=entry.method_contract(arm))
    validate_request(request, output)
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / 'training.yaml'
    config_path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding='utf-8')
    request['config_sha256'] = sha256(config_path)
    write_json(output / 'request.json', request)
    return request


def resume_request(*, original, output, stop_after_step, max_seconds):
    original, output = contained(original, ROOT), contained(output, ROOT)
    require(not output.exists(), 'new bounded continuation identity required')
    previous = json.loads((original / 'request.json').read_text())
    validate_request(previous, original)
    result = json.loads((original / 'result.json').read_text())
    settlement = json.loads((original / 'settlement.json').read_text())
    require(result.get('status') == 'completed' and result.get('mode') == MODE
        and result['execution']['step'] ==2 and stop_after_step ==20
        and result['execution']['arm'] == previous['arm']
        and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed',
        'pair continuation requires completed and settled same-arm step2')
    execution = result['execution']
    require(execution['parent_lineage'] == previous['parent_lineage']
        and execution['micro_batches_consumed'] ==176, 'profile lineage or absolute sample cursor differs')
    require(sha256(previous['config']) == previous['config_sha256'], 'pair training configuration changed')
    checkpoint = contained(execution['checkpoint_path'], previous['output_dir'])
    require(checkpoint.is_file(), 'profile checkpoint missing')
    request = dict(previous, job_id=output.name, resume=str(checkpoint), stop_after_step=20,
        max_seconds=max_seconds, predecessor_job=str(original))
    validate_request(request, output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'request.json', request)
    return request


def run(request, directory, lease):
    import train_controlled_lora_pair as entry
    from training.demo.backend import validate_active_authority
    began = time.monotonic()
    directory = contained(directory, ROOT)
    validate_request(request, directory)
    validate_active_authority(lease)
    config_path = contained(request['config'], ROOT)
    require(sha256(config_path) == request['config_sha256'], 'pair configuration changed')
    config = entry.load_config(config_path)
    output_dir = contained(request['output_dir'], ROOT)
    if request.get('resume'):
        contained(request['resume'], output_dir)
    for name in ('parent_checkpoint', 'parent_config', 'parent_job'):
        contained(request[name], RUNTIME / 'abot-retrain-20261003/teacher-controlled-v1')
    execution = entry.launch(config, str(config_path), data_root=DATA, output_dir=output_dir, model_dir=MODEL,
        arm=request['arm'], parent_checkpoint=request['parent_checkpoint'], parent_config=request['parent_config'],
        parent_job=request['parent_job'], parent_lineage=request['parent_lineage'],
        stop_after_step=request['stop_after_step'], resume=request.get('resume'), shared_lease=lease,
        shared_max_seconds=request['max_seconds'], shared_request_sha256=sha256(directory / 'request.json'))
    require(execution['step'] == request['stop_after_step'] and execution['arm'] ==request['arm'], 'pair endpoint mismatch')
    require(all(type(execution.get(k)) in (int,float) and math.isfinite(execution[k])
        for k in ('loss','grad_norm','optimizer_step_seconds')), 'actual pair measurements missing')
    return dict(status='completed', mode=MODE, arm=request['arm'], execution=execution,
        method_contract=entry.method_contract(request['arm']), elapsed_seconds=time.monotonic()-began,
        quality_improvement_verified=False, reference_memory_implemented=False, default_model_changed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    first = commands.add_parser('prepare')
    first.add_argument('--output', required=True)
    first.add_argument('--arm', choices=('frozen','trainable'), required=True)
    first.add_argument('--source-revision', required=True)
    first.add_argument('--max-seconds', type=int, default=900)
    follow = commands.add_parser('resume')
    follow.add_argument('--original', required=True)
    follow.add_argument('--output', required=True)
    follow.add_argument('--stop-after-step', type=int, required=True)
    follow.add_argument('--max-seconds', type=int, required=True)
    args = vars(parser.parse_args())
    operation = args.pop('operation')
    print(json.dumps(prepare(**args) if operation == 'prepare' else resume_request(**args)))


if __name__ == '__main__':
    main()
