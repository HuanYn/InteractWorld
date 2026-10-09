"""Guarded, finite two-window fitting stage; no automatic parent continuation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.studio.controlled_action_train import MODEL, RUNTIME

ROOT = RUNTIME / 'abot-retrain-20261003/teacher-paired-turn-v1'
DATA = ROOT / 'data'
MODE = 'paired_turn_train'
REVISION = 'controlled-native97-same-episode-paired-turn-v1'
JOBS = {2: 'cpt-b-profile2-v1', 20: 'cpt-b-to20-v1'}


def validate_request(request, directory):
    import train_paired_turn_teacher as entry
    require(request.get('mode') == MODE and request.get('experiment_revision') == REVISION,
            'explicit tinyfit revision required')
    step = request.get('stop_after_step')
    require(type(step) is int and step in JOBS and request.get('job_id') == directory.name == JOBS[step],
            'only two unique bounded training identities permitted')
    maximum = request.get('max_seconds')
    require(type(maximum) is int and 60 <= maximum <= (900 if step == 2 else 3600),
            'profile/continuation watchdog out of bounds')
    require((step == 2 and request.get('resume') is None)
            or (step == 20 and request.get('resume') is not None), 'profile2 then strict2->20 only')
    require(request.get('data_root') == str(DATA) and request.get('model_dir') == str(MODEL)
        and request.get('saved_tensors_on_cpu') is True and request.get('arm') == 'frozen',
        'same native97/official base/FP32 frozen LoRA with saved activation offload required')
    require(request.get('config') == str(ROOT / JOBS[2] / 'training.yaml')
        and request.get('output_dir') == str(ROOT / JOBS[2] / 'training'),
        'both B segments must share the fixed new-stage config and training directory')
    require(isinstance(request.get('source_revision'), str) and bool(request['source_revision']), 'source revision required')
    require(not any(key in request for key in ('initialize_from', 'warm_start_from', 'length_transition_from',
        'action_scale', 'ratio_cap', 'gate_lr', 'adapter_lr')), 'no initialization or mechanism override allowed')
    entry._validate_lineage(request.get('parent_lineage'))
    entry._validate_pool(request.get('pool'))
    require(request.get('method_contract') == entry.method_contract(), 'paired-turn method changed')
    require(request.get('protocol_cap_gpu_seconds') == 16200
        and request.get('protocol_incremental_bytes') == 8_000_000_000,
        'whole-protocol budget must include completed cache/A costs')
    gate, reference = request.get('a_visual_gate', {}), request.get('a_reference', {})
    require(gate.get('review_path') == str(entry.REVIEW) and gate.get('b_training_allowed') is True
        and reference.get('parent_lineage') == request['parent_lineage']
        and reference.get('micro_batches') == 160 and reference.get('uses_A_weights') is False,
        'actual negative A review and equal-training common-parent record required')


def prepare(*, output, source_revision, max_seconds=900):
    import yaml
    import train_paired_turn_teacher as entry
    output = contained(output, ROOT)
    require(not output.exists() and output.name == JOBS[2], 'new fixed tinyfit profile identity required')
    gate = entry.visual_gate()  # Read actual review before any parent weights.
    source, _ = entry.a_eval.source_evaluation()
    config = entry.reference.load_config(source['training_config'])
    config.model.base_model_path = str(MODEL)
    config.data.manifest_path = str(DATA / 'manifests/train.jsonl')
    config.data.manifest_sha256 = sha256(DATA / 'manifests/train.jsonl')
    config.data.data_factory = entry.FACTORY
    config.data.num_frames, config.data.num_workers = 97, 0
    config.training.lora_parameter_dtype = 'float32'
    config.optimizer.adapter_lr, config.optimizer.lora_lr = 1e-5, 0.
    config.training.max_steps = config.training.checkpoint_every = 20
    config.training.output_dir = str(output / 'training')
    entry.validate_config(config, data_root=DATA, output_dir=output / 'training', model_dir=MODEL)
    state, lineage = entry.load_parent(config)
    del state
    pool = entry.pool_contract()
    _, a_reference = entry.reference_a_contract(config)
    request = dict(job_id=output.name, mode=MODE, experiment_revision=REVISION, arm='frozen',
        max_seconds=max_seconds, config=str(output / 'training.yaml'), data_root=str(DATA), model_dir=str(MODEL),
        output_dir=str(output / 'training'), resume=None, stop_after_step=2, saved_tensors_on_cpu=True,
        source_revision=source_revision, parent_lineage=lineage, pool=pool, method_contract=entry.method_contract(),
        a_visual_gate=gate, a_reference=a_reference,
        protocol_cap_gpu_seconds=16200, protocol_incremental_bytes=8_000_000_000,
        budget_scope='whole_paired_turn_protocol_including_cache_A_failures_and_B;not_reset')
    validate_request(request, output)
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'training.yaml'
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding='utf-8')
    request['config_sha256'] = sha256(path)
    write_json(output / 'request.json', request)
    write_json(output / 'preparation.json', dict(status='cpu_prepared', gpu_launched=False,
        parent_weights_loaded_cpu_only=True, new_optimizer_not_created_until_guarded_launch=True,
        stage=entry.STAGE, a_visual_gate=gate, a_reference=a_reference,
        parent_lineage=lineage, sampling_contract=entry.sampling_contract(config.training, pool),
        checkpoint_policy='newstep2/20_only', quality_verified=False))
    return request


def resume_request(*, original, output, stop_after_step=20, max_seconds):
    import train_paired_turn_teacher as entry
    original, output = contained(original, ROOT), contained(output, ROOT)
    require(original.name == JOBS[2] and output.name == JOBS[20] and not output.exists(),
            'new fixed continuation identity required')
    previous = json.loads((original / 'request.json').read_text(encoding='utf-8'))
    validate_request(previous, original)
    result = json.loads((original / 'result.json').read_text(encoding='utf-8'))
    settlement = json.loads((original / 'settlement.json').read_text(encoding='utf-8'))
    require(result.get('status') == 'completed' and result.get('mode') == MODE
        and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed',
        'strict continuation requires completed and settled two-step profile')
    require(entry.visual_gate() == previous['a_visual_gate'], 'A visual review changed')
    execution = result['execution']
    require(stop_after_step == 20 and execution['step'] == 2
        and execution['new_stage_micro_batches_consumed'] == execution['micro_batches_consumed'] == 16
        and execution['historical_micro_batches_before_stage'] == 320
        and execution['parent_lineage'] == previous['parent_lineage'], 'tinyfit stage cursor/lineage differs')
    qualification = execution.get('qualification', {})
    require(all(qualification.get(k) is True for k in ('adapter_updated', 'gate_updated', 'gradients_finite'))
        and qualification.get('lora_trainable') is False
        and 0 < qualification.get('peak_reserved_bytes', 0) < 23 * 1024 ** 3,
        'profile must show actual updates, finite gradients and reserved VRAM below23GiB')
    metrics = [json.loads(line) for line in (Path(previous['output_dir']) / 'metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    require([row.get('step') for row in metrics] == [1, 2], 'exact two-step profile metrics required')
    seconds = [row.get('optimizer_step_seconds') for row in metrics]
    require(all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in seconds), 'finite measured step times required')
    measured_limit = math.ceil(max(seconds) * 18 * 1.35 + 180)
    require(max_seconds == measured_limit and measured_limit <= 3600,
        'continuation watchdog must equal measured ceil(slowest*18*1.35+180), at most3600')
    require(sha256(previous['config']) == previous['config_sha256'], 'tinyfit training configuration changed')
    checkpoint = contained(execution['checkpoint_path'], previous['output_dir'])
    require(checkpoint.is_file() and checkpoint.name == 'step-0000002.pt', 'fixed profile checkpoint missing')
    request = dict(previous, job_id=output.name, resume=str(checkpoint), stop_after_step=20,
        max_seconds=max_seconds, predecessor_job=str(original))
    validate_request(request, output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'request.json', request)
    return request


def run(request, directory, lease):
    import train_paired_turn_teacher as entry
    from training.demo.backend import validate_active_authority
    began = time.monotonic()
    directory = contained(directory, ROOT)
    validate_request(request, directory)
    validate_active_authority(lease)
    path = contained(request['config'], ROOT)
    require(sha256(path) == request['config_sha256'], 'tinyfit configuration changed')
    config = entry.load_config(path)
    output = contained(request['output_dir'], ROOT)
    if request.get('resume'):
        contained(request['resume'], output)
    for field in ('parent_checkpoint', 'parent_config', 'parent_job'):
        contained(request['parent_lineage'][field], RUNTIME / 'abot-retrain-20261003/teacher-controlled-lora-pair-v1')
    execution = entry.launch(config, str(path), data_root=DATA, output_dir=output, model_dir=MODEL,
        parent_lineage=request['parent_lineage'], pool=request['pool'], stop_after_step=request['stop_after_step'],
        a_visual_gate=request['a_visual_gate'], a_reference=request['a_reference'],
        resume=request.get('resume'), shared_lease=lease, shared_max_seconds=request['max_seconds'],
        shared_request_sha256=sha256(directory / 'request.json'))
    require(execution['step'] == request['stop_after_step'] and execution['stage'] == entry.STAGE,
            'tinyfit endpoint changed')
    require(all(type(execution.get(key)) in (int, float) and math.isfinite(execution[key])
        for key in ('loss', 'grad_norm', 'optimizer_step_seconds')), 'missing actual tinyfit measurements')
    return dict(status='completed', mode=MODE, arm='frozen', execution=execution,
        method_contract=entry.method_contract(), elapsed_seconds=time.monotonic() - began,
        quality_improvement_verified=False, generalization_verified=False, reference_memory_implemented=False,
        data_coverage_causal_proof=False, default_model_changed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    first = commands.add_parser('prepare')
    first.add_argument('--output', required=True)
    first.add_argument('--source-revision', required=True)
    first.add_argument('--max-seconds', type=int, default=900)
    follow = commands.add_parser('resume')
    follow.add_argument('--original', required=True)
    follow.add_argument('--output', required=True)
    follow.add_argument('--stop-after-step', type=int, default=20)
    follow.add_argument('--max-seconds', type=int, required=True)
    args = vars(parser.parse_args())
    operation = args.pop('operation')
    print(json.dumps(prepare(**args) if operation == 'prepare' else resume_request(**args)))


if __name__ == '__main__':
    main()

