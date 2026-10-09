"""Bounded TF+BID comparator; new objective and trajectory, old TF is unchanged."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from training.demo.contracts import contained, require, sha256, write_json

RUNTIME = Path('/DATA_DISK/InteractWorld/creator-runtime')
MODE = 'causal_migration_bid_train'
REVISION = 'native97-fp32-tf-plus-bid-paired20-v1'


def validate_request(request, directory):
    require(request.get('mode') == MODE and request.get('experiment_revision') == REVISION,
            'explicit new TF+BID objective required')
    require(request.get('job_id') == directory.name, 'TF+BID job identity differs')
    require(type(request.get('max_seconds')) is int and 60 <= request['max_seconds'] <= 3600,
            'TF+BID job must have a finite <=3600-second watchdog')
    require(type(request.get('stop_after_step')) is int and 1 <= request['stop_after_step'] <= 20,
            'fixed comparison ends no later than step20')
    require(request.get('saved_tensors_on_cpu') is True, 'TF+BID requires sequential CPU-offloaded activations')
    require(isinstance(request.get('source_revision'), str) and request['source_revision'], 'source revision required')


def prepare(*, template, output, source_revision, max_seconds=1800):
    import yaml
    from training.causal_moba import load_moba_config
    from train_causal_r005_bid_probe import validate_probe_config, method_contract

    output = contained(output, RUNTIME)
    require(not output.exists(), 'new TF+BID trajectory identity required')
    raw = yaml.safe_load(Path(template).read_text(encoding='utf-8'))
    raw['regularization'] = {'bidirectional_weight': 0.1}
    raw['training']['output_dir'] = str(output / 'training')
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / 'training.yaml'
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding='utf-8')
    config = load_moba_config(config_path)
    validate_probe_config(config)
    request = dict(job_id=output.name, mode=MODE, experiment_revision=REVISION,
        max_seconds=max_seconds, config=str(config_path), config_sha256=sha256(config_path),
        resume=None, stop_after_step=2, saved_tensors_on_cpu=True,
        source_revision=source_revision, method_contract=method_contract())
    validate_request(request, output)
    write_json(output / 'request.json', request)
    return request


def resume_request(*, original, output, stop_after_step, max_seconds):
    original, output = contained(original, RUNTIME), contained(output, RUNTIME)
    require(not output.exists(), 'new bounded continuation identity required')
    previous = json.loads((original / 'request.json').read_text())
    result = json.loads((original / 'result.json').read_text())
    settlement = json.loads((original / 'settlement.json').read_text())
    require(previous.get('mode') == MODE and result.get('status') == 'completed'
            and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed',
            'TF+BID continuation requires its own completed settled predecessor')
    execution = result['execution']
    require(execution['step'] < stop_after_step <= 20, 'must advance the fixed comparison trajectory')
    require(sha256(previous['config']) == previous['config_sha256'], 'TF+BID configuration changed')
    require(Path(execution['checkpoint_path']).is_file(), 'TF+BID predecessor checkpoint missing')
    request = dict(previous, job_id=output.name, resume=execution['checkpoint_path'],
        stop_after_step=stop_after_step, max_seconds=max_seconds, predecessor_job=str(original))
    validate_request(request, output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'request.json', request)
    return request


def run(request, directory, lease):
    from training.causal_moba import load_moba_config
    from train_causal_r005_bid_probe import launch, method_contract
    from training.demo.backend import validate_active_authority

    began = time.monotonic()
    directory = contained(directory, RUNTIME)
    validate_request(request, directory)
    validate_active_authority(lease)
    require(request.get('method_contract') == method_contract(), 'TF+BID objective changed')
    config_path = contained(request['config'], RUNTIME)
    require(sha256(config_path) == request['config_sha256'], 'TF+BID configuration changed')
    config = load_moba_config(config_path)
    contained(config.training.output_dir, RUNTIME)
    if request.get('resume'):
        contained(request['resume'], config.training.output_dir)
    execution = launch(config, str(config_path), shared_lease=lease,
        shared_max_seconds=request['max_seconds'], shared_request_sha256=sha256(directory / 'request.json'),
        source_revision=request['source_revision'], stop_after_step=request['stop_after_step'],
        resume=request.get('resume'), saved_tensors_on_cpu=True)
    require(execution.get('step') == request['stop_after_step'], 'TF+BID training stopped early')
    require(all(type(execution.get(key)) in (int, float) and math.isfinite(execution[key])
                for key in ('loss', 'loss_tf', 'loss_bid', 'loss_total', 'grad_norm', 'optimizer_step_seconds')),
            'missing actual TF/BID loss or resource measurements')
    return dict(status='completed', mode=MODE, execution=execution, method_contract=method_contract(),
        elapsed_seconds=time.monotonic() - began, quality_improvement_verified=False,
        realtime_verified=False, default_model_changed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    first = commands.add_parser('prepare')
    first.add_argument('--template', required=True)
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
