"""Controlled action bootstrap, isolated from every historical R005 trajectory."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from training.demo.contracts import contained, require, sha256, write_json

RUNTIME = Path('/DATA_DISK/InteractWorld/creator-runtime')
ROOT = RUNTIME / 'abot-retrain-20261003/teacher-controlled-v1'
DATA = RUNTIME / 'native97-20260928/data'
MODEL = Path('/DATA_DISK/InteractWorld/legacy-assets/models/Wan2.2-TI2V-5B@921dbaf3f1674a56f47e83fb80a34bac8a8f203e')
MODE = 'controlled_action_train'
REVISION = 'controlled-action-wan-native97-gated20-v1'


def validate_request(request, directory):
    require(request.get('mode') == MODE and request.get('experiment_revision') == REVISION,
            'explicit controlled-action training stage required')
    require(request.get('job_id') == directory.name, 'fresh training job identity differs')
    require(type(request.get('max_seconds')) is int and 60 <= request['max_seconds'] <= 3600,
            'fresh training requires a finite <=3600s watchdog')
    require(type(request.get('stop_after_step')) is int and 2 <= request['stop_after_step'] <= 20,
            'controlled teacher trajectory is bounded to 20 optimizer steps')
    require(request.get('resume') is not None or request['stop_after_step'] == 2,
            'fresh execution must first measure exactly two optimizer steps')
    require(request.get('saved_tensors_on_cpu') is True, 'activation CPU offload is required')
    require(request.get('data_root') == str(DATA) and request.get('model_dir') == str(MODEL),
            'only existing official base and native97 feature assets are in scope')
    require(isinstance(request.get('source_revision'), str) and request['source_revision'],
            'actual source revision required')
    require(not any(key in request for key in ('initialize_from', 'warm_start_from', 'length_transition_from', 'parent_checkpoint')),
            'historical-parent initialization is forbidden for the new route')


def prepare(*, output, source_revision, max_seconds=1800):
    import yaml
    from training.config import ActionTeacherConfig
    import train_controlled_action_teacher as entry

    output = contained(output, ROOT)
    require(not output.exists(), 'fresh profile needs a new identity')
    config = ActionTeacherConfig()
    config.model.base_model_path = str(MODEL)
    config.data.manifest_path = str(DATA / 'manifests/train.jsonl')
    config.data.manifest_sha256 = 'fd47f92ad2c02aa2f19a2a1dac810c21320fc52bd05a56ff0d9443985764d4cb'
    config.data.data_factory = 'training.data.action_resampled:build_resampled_action_teacher_dataloader'
    config.data.num_frames = 97
    config.data.num_workers = 0
    config.training.lora_parameter_dtype = 'float32'
    config.optimizer.adapter_lr = 1e-5
    config.optimizer.lora_lr = 0.0
    config.training.max_steps = 20
    config.training.checkpoint_every = 20
    config.training.output_dir = str(output / 'training')
    entry.validate_config(config, data_root=DATA, output_dir=output / 'training', model_dir=MODEL)
    request = dict(job_id=output.name, mode=MODE, experiment_revision=REVISION,
        max_seconds=max_seconds, config=str(output / 'training.yaml'),
        data_root=str(DATA), model_dir=str(MODEL), output_dir=str(output / 'training'),
        resume=None, stop_after_step=2, saved_tensors_on_cpu=True,
        source_revision=source_revision, method_contract=entry.method_contract())
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
            and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed',
            'controlled teacher continuation needs its completed and settled predecessor')
    execution = result['execution']
    require(execution['step'] < stop_after_step <= 20, 'must advance the same fresh trajectory')
    require(sha256(previous['config']) == previous['config_sha256'], 'controlled teacher configuration changed')
    checkpoint = contained(execution['checkpoint_path'], previous['output_dir'])
    require(checkpoint.is_file(), 'fresh predecessor checkpoint is missing')
    request = dict(previous, job_id=output.name, resume=str(checkpoint),
        stop_after_step=stop_after_step, max_seconds=max_seconds, predecessor_job=str(original))
    validate_request(request, output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'request.json', request)
    return request


def run(request, directory, lease):
    import train_controlled_action_teacher as entry
    from training.demo.backend import validate_active_authority

    began = time.monotonic()
    directory = contained(directory, ROOT)
    validate_request(request, directory)
    validate_active_authority(lease)
    require(request['method_contract'] == entry.method_contract(), 'controlled teacher method changed')
    config_path = contained(request['config'], ROOT)
    require(sha256(config_path) == request['config_sha256'], 'training configuration changed')
    config = entry.load_config(config_path)
    output_dir = contained(request['output_dir'], ROOT)
    if request.get('resume'):
        contained(request['resume'], output_dir)
    execution = entry.launch(config, str(config_path), data_root=Path(request['data_root']),
        output_dir=output_dir, model_dir=Path(request['model_dir']),
        stop_after_step=request['stop_after_step'], resume=request.get('resume'), shared_lease=lease,
        shared_max_seconds=request['max_seconds'], shared_request_sha256=sha256(directory / 'request.json'))
    require(execution.get('step') == request['stop_after_step'], 'controlled teacher stopped before requested endpoint')
    require(all(type(execution.get(key)) in (int, float) and math.isfinite(execution[key])
                for key in ('loss', 'grad_norm', 'optimizer_step_seconds')),
            'missing actual training measurements')
    return dict(status='completed', mode=MODE, execution=execution, method_contract=entry.method_contract(),
        elapsed_seconds=time.monotonic() - began, quality_improvement_verified=False,
        reference_memory_implemented=False, default_model_changed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    first = commands.add_parser('prepare')
    first.add_argument('--output', required=True)
    first.add_argument('--source-revision', required=True)
    first.add_argument('--max-seconds', type=int, default=1800)
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
