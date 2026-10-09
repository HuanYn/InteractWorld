"""Prepare and execute bounded, explicitly new-data R005 fine-tuning jobs."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

from training.demo.contracts import contained, require, sha256, write_json

RUNTIME = Path('/DATA_DISK/InteractWorld/creator-runtime')
MIGRATION = Path('/DATA_DISK/InteractWorld/legacy-assets')
PARENT = MIGRATION / 'data/checkpoints/moba-parent-r005-1040.pt'
MODEL = MIGRATION / 'models/Wan2.2-TI2V-5B@921dbaf3f1674a56f47e83fb80a34bac8a8f203e'


def prepare(*, subset, output, variant, max_steps, stop_after_step, max_seconds, source_revision,
            parent=PARENT, saved_tensors_on_cpu=False):
    import torch
    import yaml
    from training.config import load_config
    from training.demo.relocation import ArtifactHashes
    from training.data.condition_robustness import POLICY
    require(variant in ('clean', 'robust'), 'variant must be clean or robust')
    require(type(max_steps) is int and 1 <= max_steps <= 40, 'pilot must have at most40 optimizer steps')
    require(type(stop_after_step) is int and 0 < stop_after_step <= max_steps, 'invalid stop step')
    require(type(max_seconds) is int and 60 <= max_seconds <= 3600, 'invalid finite job cap')
    output, subset = contained(output, RUNTIME), contained(subset, RUNTIME)
    require(not output.exists(), 'new job identity required')
    parent = Path(parent).resolve()
    contained(parent, RUNTIME.parent)
    digest = ArtifactHashes(RUNTIME / 'verified-artifacts.json', protected_paths=[parent])
    parent_sha = digest(parent)
    payload = torch.load(parent, map_location='cpu', weights_only=False)
    require(payload.get('stage') == 'action_teacher_lora_v1' and payload.get('step') == 1040,
            'this pilot starts from the retained R005 step1040')
    raw = copy.deepcopy(payload['config'])
    del payload
    raw['model']['base_model_path'] = str(MODEL)
    raw['data'].update(manifest_path=str(subset / 'manifests/train.jsonl'), manifest_sha256=None,
        prompt_cache_path=str(subset / 'features/scene-static-subset.pt'), num_workers=0,
        data_factory=('training.data.condition_robustness:build_condition_robustness_dataloader'
            if variant == 'robust' else 'training.data.action_resampled:build_resampled_action_teacher_dataloader'))
    raw['training'].update(max_steps=max_steps, checkpoint_every=20, output_dir=str(output / 'training'))
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / 'training.yaml'
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding='utf-8')
    load_config(config_path)
    request = dict(job_id=output.name, mode='subset_train', max_seconds=max_seconds,
        config=str(config_path), config_sha256=sha256(config_path), parent=str(parent), parent_sha256=parent_sha,
        resume=None, stop_after_step=stop_after_step, saved_tensors_on_cpu=bool(saved_tensors_on_cpu),
        source_revision=source_revision, experiment_revision='3090-subset-condition-v1',
        condition_policy=POLICY if variant == 'robust' else {'name': 'clean_condition_control'}, variant=variant)
    write_json(output / 'request.json', request)
    return request


def run(request, directory, lease):
    from train_action_teacher import launch
    from training.config import load_config
    from training.demo.backend import validate_active_authority
    from training.data.condition_robustness import POLICY
    began = time.monotonic()
    directory = Path(directory).resolve()
    validate_active_authority(lease)
    config_path = contained(request['config'], RUNTIME)
    require(sha256(config_path) == request['config_sha256'], 'training configuration changed')
    config = load_config(config_path)
    contained(config.training.output_dir, RUNTIME)
    variant = request['variant']
    require(variant in ('clean', 'robust'), 'invalid subset variant')
    if variant == 'robust':
        require(request['condition_policy'] == POLICY, 'conditioning policy differs from code')
    resume = request.get('resume')
    if resume is not None:
        contained(resume, config.training.output_dir)
    launch(config, str(config_path), resume, None,
        confirmed_gpu_index=lease['gpu_index'], confirmed_gpu_uuid=lease['gpu_uuid'],
        confirmed_at_utc='', allocation_profile=lease['profile'],
        weights_only_from=None if resume else request['parent'],
        weights_only_sha256=None if resume else request['parent_sha256'],
        base_model_path_relocated=not bool(resume),
        shared_lease=lease, shared_max_seconds=request['max_seconds'],
        shared_request_sha256=sha256(directory / 'request.json'),
        saved_tensors_on_cpu=request.get('saved_tensors_on_cpu', False),
        stop_after_step=request['stop_after_step'], profile_runtime=True)
    execution = json.loads((Path(config.training.output_dir) / 'execution_result.json').read_text())
    require(execution['step'] == request['stop_after_step'], 'training stopped before requested step')
    return dict(status='completed', mode='subset_train', variant=variant,
                execution=execution, elapsed_seconds=time.monotonic() - began,
                condition_policy=request['condition_policy'], quality_improvement='not_yet_evaluated')


def resume_request(*, original, output, stop_after_step, max_seconds):
    original, output = contained(original, RUNTIME), contained(output, RUNTIME)
    require(not output.exists(), 'new resume job identity required')
    previous = json.loads((original / 'request.json').read_text())
    result = json.loads((original / 'result.json').read_text())
    require(result.get('status') == 'completed', 'resume only after a completed measured segment')
    require(result['execution']['step'] < stop_after_step <= result['execution']['configured_max_steps'], 'invalid resume step')
    require(type(max_seconds) is int and 60 <= max_seconds <= 3600, 'invalid watchdog')
    request = dict(previous, job_id=output.name, max_seconds=max_seconds,
        resume=result['execution']['checkpoint_path'], stop_after_step=stop_after_step)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'request.json', request)
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='operation', required=True)
    create = modes.add_parser('prepare')
    create.add_argument('--subset', required=True, type=Path)
    create.add_argument('--output', required=True, type=Path)
    create.add_argument('--variant', choices=('clean', 'robust'), required=True)
    create.add_argument('--max-steps', type=int, default=20)
    create.add_argument('--stop-after-step', type=int, default=2)
    create.add_argument('--max-seconds', type=int, default=900)
    create.add_argument('--source-revision', required=True)
    create.add_argument('--saved-tensors-on-cpu', action='store_true')
    resume = modes.add_parser('resume')
    resume.add_argument('--original', required=True, type=Path)
    resume.add_argument('--output', required=True, type=Path)
    resume.add_argument('--stop-after-step', type=int, required=True)
    resume.add_argument('--max-seconds', type=int, required=True)
    args = vars(parser.parse_args())
    operation = args.pop('operation')
    print(json.dumps(prepare(**args) if operation == 'prepare' else resume_request(**args), ensure_ascii=False))


if __name__ == '__main__':
    main()
