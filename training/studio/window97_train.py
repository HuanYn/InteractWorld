"""Explicit R005 49->97/resampled transfer, then strict same-config resume.

Preparation is CPU only. The guarded worker owns GPU allocation and watchdogs.
The fixed twenty-step configuration is not a promise of useful training quality.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.studio.subset_train import MODEL, PARENT, RUNTIME
from training.data.gt_prefix_mix import FACTORY as PREFIX_FACTORY, prefix_contract

FACTORY = 'training.data.action_resampled:build_resampled_action_teacher_dataloader'


def measured_fp32_continuation(original):
    """Bind one completed, settled resource pilot; never reload/hash its weights."""
    from train_action_teacher import FP32_LORA_PARENT_SHA, fp32_lora_precision_transition
    from training.config import load_config
    original = contained(original, RUNTIME)
    names = ('request.json', 'result.json', 'settlement.json', 'two-step-assessment.json')
    records = {name: json.loads((original / name).read_text()) for name in names}
    previous, result, settlement, assessment = (records[name] for name in names)
    require(previous.get('job_id') == original.name and previous.get('mode') == 'window97_train'
            and previous.get('resume') is None and previous.get('stop_after_step') == 2
            and previous.get('configured_max_steps') == 20
            and all(previous.get(key) is True for key in ('lora_fp32', 'gt_prefix_mix', 'freeze_action_adapter'))
            and previous.get('parent_sha256') == FP32_LORA_PARENT_SHA
            and previous.get('prefix_contract') == prefix_contract()
            and previous.get('precision_transition') == fp32_lora_precision_transition(),
            'measured continuation requires the fresh R005 FP32 GT97 stop2 pilot')
    config_path = contained(previous['config'], original)
    require(config_path == original / 'training.yaml' and sha256(config_path) == previous['config_sha256'],
            'measured pilot config binding changed')
    config = load_config(config_path)
    low_sigma = previous.get('low_sigma', False)
    from training.data.low_sigma import low_sigma_contract, summarize_samples, sigma_observation, validate_opt_in
    validate_opt_in(config, low_sigma)
    require(previous.get('low_sigma_contract') == (low_sigma_contract() if low_sigma else None),
            'measured pilot low-sigma contract changed')
    require(config.data.data_factory == PREFIX_FACTORY and config.data.num_frames == 97
            and config.training.lora_parameter_dtype == 'float32' and config.training.max_steps == 20
            and config.training.gradient_accumulation_steps == 8
            and config.optimizer.adapter_lr == 0 and config.optimizer.lora_lr == 2e-6
            and Path(config.training.output_dir).resolve() == original / 'training',
            'measured continuation must preserve the complete FP32 GT97 training configuration')
    execution = result.get('execution', {})
    require(result.get('status') == 'completed' and result.get('lora_fp32') is True
            and result.get('precision_transition') == fp32_lora_precision_transition()
            and execution.get('step') == 2 and execution.get('micro_batches_consumed') == 16
            and execution.get('configured_max_steps') == 20
            and execution.get('execution_stop_after_step') == 2,
            'measured continuation requires two actual completed optimizer steps')
    require(settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed'
            and isinstance(settlement.get('reservation_id'), str) and bool(settlement['reservation_id'])
            and type(settlement.get('charged_gpu_seconds')) in (int, float)
            and math.isfinite(settlement['charged_gpu_seconds']) and settlement['charged_gpu_seconds'] > 0,
            'measured pilot must already be settled')
    checkpoint = contained(execution['checkpoint_path'], original / 'training')
    require(checkpoint.is_file() and assessment.get('checkpoint') == str(checkpoint)
            and assessment.get('step') == 2 and assessment.get('micro_batches') == 16
            and assessment.get('status') == 'two_step_resource_gate_passed'
            and assessment.get('finite_loss_and_grad') is True,
            'measured pilot checkpoint or assessment does not match its completed step2')
    profile = execution.get('profile', {})
    rows = profile.get('steps', [])
    seconds = assessment.get('step_seconds', [])
    require(len(rows) == len(seconds) == 2 and [row.get('step') for row in rows] == [1, 2]
            and seconds == [row.get('seconds') for row in rows]
            and all(type(value) in (int, float) and math.isfinite(value) and 0 < value <= 120 for value in seconds)
            and profile.get('failure') is None, 'measured step timing or failure gate did not pass')
    peak, headroom = assessment.get('peak_allocated_bytes'), assessment.get('sampled_min_headroom_bytes')
    require(type(peak) is int and 0 < peak < 22 * 1024**3
            and peak == profile.get('final_memory', {}).get('cumulative_peak_allocated_bytes')
            and type(headroom) is int and headroom >= 1024**3
            and headroom == profile.get('driver_sampled_min_headroom_bytes'),
            'measured VRAM gate did not pass')
    losses = assessment.get('losses', [])
    require(len(losses) == 2 and all(type(value) in (int, float) and math.isfinite(value) for value in losses),
            'measured losses must be finite for both optimizer steps')
    if low_sigma:
        require(result.get('low_sigma_contract') == assessment.get('low_sigma_contract') == low_sigma_contract(),
                'measured low_sigma result/assessment policy differs')
        summaries = assessment.get('actual_low_sigma_summaries', [])
        require(len(summaries) == 2, 'low_sigma resource pilot needs both actual sampling summaries')
        for step, summary in enumerate(summaries):
            rows = summary.get('samples', [])
            require([row.get('sample_index') for row in rows] == list(range(step * 8, (step + 1) * 8))
                    and all(row == sigma_observation(row.get('schedule_index'), row['sample_index']) for row in rows)
                    and summary == summarize_samples(rows), 'measured low_sigma coverage differs from actual fixed policy')
        # Resume appends to this same training log: bind the immutable first two
        # rows, not the whole growing file. No checkpoint reload or tensor scan.
        metric_rows = [json.loads(line) for line in (original / 'training/metrics.jsonl').read_text().splitlines() if line.strip()]
        measured_rows = [row for row in metric_rows if row.get('step') in (1, 2)]
        require([row.get('step') for row in measured_rows] == [1, 2]
                and [row.get('low_sigma_summary') for row in measured_rows] == summaries
                and [row.get('loss') for row in measured_rows] == losses
                and [row.get('precision_summary') for row in measured_rows] == assessment.get('actual_gradient_precision')
                and all(type(row.get('grad_norm')) in (int, float) and math.isfinite(row['grad_norm'])
                        for row in measured_rows), 'low_sigma assessment differs from actual first-two-step metrics')
    else:
        require('low_sigma_contract' not in assessment and 'low_sigma_contract' not in result,
                'legacy FP32 assessment cannot relabel low_sigma evidence')
        groups = assessment.get('groups', {})
        for key in ('lora_a', 'lora_b'):
            group = groups.get(key, {})
            require(group.get('tensors') == 300 and group.get('elements') == 20152320
                    and group.get('dtypes') == ['torch.float32']
                    and type(group.get('changed')) is int and 0 < group['changed'] <= group['elements']
                    and type(group.get('recast_bf16_changed')) is int
                    and 0 < group['recast_bf16_changed'] <= group['elements'],
                    'measured FP32 LoRA A/B updates must be real, including nonzero BF16-visible updates')
        adapter = groups.get('adapter', {})
        require(adapter.get('tensors') == 6 and adapter.get('elements') == 270541824
                and adapter.get('dtypes') == ['torch.bfloat16'] and adapter.get('changed') == 0
                and adapter.get('recast_bf16_changed') == 0, 'measured fixed adapter changed')
    expected_gradient = dict(lora_gradient_dtype='float32', lora_parameter_dtype='float32',
        lora_gradient_tensors=600, lora_gradient_elements=40304640,
        observed_at='after_accumulation_before_clip_and_step')
    require(assessment.get('actual_gradient_precision') == [expected_gradient, expected_gradient],
            'measured actual gradients must be FP32 for all600 LoRA tensors on both steps')
    binding = dict(kind='low_sigma_measured_step2_to20_v1' if low_sigma else 'fp32_lora_measured_step2_to20_v1',
        source_job=str(original), checkpoint=str(checkpoint),
        config=str(config_path), config_sha256=previous['config_sha256'], from_step=2, to_step=20,
        micro_batches_consumed=16, parent_sha256=FP32_LORA_PARENT_SHA,
        record_sha256={name: sha256(original / name) for name in names})
    if low_sigma:
        binding['low_sigma_contract'] = low_sigma_contract()
        binding['first_two_metrics_sha256'] = hashlib.sha256(
            json.dumps(measured_rows, sort_keys=True).encode('utf-8')).hexdigest()
    return binding


def validate_training(config, config_path, *, gt_prefix_mix=False, lora_fp32=False, low_sigma=False):
    from train_action_teacher import _manifest_hashes
    from training.data.action_dataset import cache_index_path
    require(config.data.num_frames == 97 and config.data.prompt_cache_path is None,
            'native97 training requires inline static text, not a dev prompt sidecar')
    require(type(gt_prefix_mix) is bool, 'gt_prefix_mix must be an explicit boolean')
    require(type(lora_fp32) is bool and lora_fp32 == (config.training.lora_parameter_dtype == 'float32'),
            'lora_fp32 must explicitly match the configured LoRA parameter dtype')
    require(not lora_fp32 or gt_prefix_mix, 'FP32 LoRA requires the GT-prefix pilot')
    from training.data.low_sigma import validate_opt_in
    validate_opt_in(config, low_sigma)
    require(config.data.data_factory == (PREFIX_FACTORY if gt_prefix_mix else FACTORY),
            'native97 must retain its explicitly selected data factory')
    require(config.training.max_steps == 20, 'this pilot fixes configured max_steps at20')
    if gt_prefix_mix:
        require(config.optimizer.adapter_lr == 0 and config.optimizer.lora_lr == 2e-6,
                'GT prefix mix fixes adapter LR0 and LoRA LR2e-6')
    hashes = _manifest_hashes(config, config_path)
    manifest = [json.loads(line) for line in Path(config.data.manifest_path).read_text().splitlines() if line.strip()]
    index = [json.loads(line) for line in cache_index_path(config.data.manifest_path).read_text().splitlines() if line.strip()]
    ids = {row['episode_id'] for row in manifest}
    require(1 <= len(ids) <= 64 and len(ids) == len(manifest) == len(index), 'native97 pilot requires at most64 unique train episodes')
    require({row['episode_id'] for row in index} == ids, 'native97 feature IDs differ from training manifest')
    require(all(row.get('split') == 'train' for row in manifest + index), 'dev episodes must not enter native97 training')
    require(all(type(row.get('num_windows')) is int and 1 <= row['num_windows'] <= 4 for row in index),
            'native97 pilot permits one to four real cached windows per episode')
    return hashes


def prepare(*, data, output, source_revision, parent=PARENT, model=MODEL,
            stop_after_step=2, max_seconds=900, saved_tensors_on_cpu=False, freeze_action_adapter=False,
            gt_prefix_mix=False, lora_fp32=False, low_sigma=False):
    import torch
    import yaml
    from training.config import is_pinned_base_model, load_config
    from training.demo.relocation import ArtifactHashes
    require(type(stop_after_step) is int and 0 < stop_after_step <= 20, 'invalid pilot stop step')
    require(type(max_seconds) is int and 60 <= max_seconds <= 3600, 'invalid finite watchdog')
    require(type(saved_tensors_on_cpu) is bool, 'saved_tensors_on_cpu must be an explicit boolean')
    require(type(freeze_action_adapter) is bool, 'freeze_action_adapter must be an explicit boolean')
    require(type(gt_prefix_mix) is bool and (not gt_prefix_mix or freeze_action_adapter),
            'gt_prefix_mix requires explicit freeze_action_adapter')
    require(type(lora_fp32) is bool and (not lora_fp32 or (gt_prefix_mix and freeze_action_adapter
                                                        and stop_after_step == 2)),
            'lora_fp32 requires GT-prefix mix, fixed adapter and stop_after_step=2')
    require(type(low_sigma) is bool and (not low_sigma or lora_fp32),
            'low_sigma requires the explicit FP32 GT-prefix resource pilot')
    output, data = contained(output, RUNTIME), contained(data, RUNTIME)
    parent, model = contained(parent, RUNTIME.parent), contained(model, RUNTIME.parent)
    require(not output.exists(), 'new job identity required')
    verification_cache = RUNTIME / 'verified-artifacts.json'
    digest = ArtifactHashes(verification_cache, protected_paths=[parent])
    parent_sha = digest(parent)
    payload = torch.load(parent, map_location='cpu', weights_only=False)
    require(payload.get('stage') == 'action_teacher_lora_v1' and payload.get('step') == 1040,
            'native97 pilot starts from the original R005 step1040')
    if lora_fp32:
        from train_action_teacher import FP32_LORA_PARENT_SHA, _validate_lora_state_dtype
        require(parent_sha == FP32_LORA_PARENT_SHA, 'FP32 LoRA requires the pinned original R005 parent')
        require(payload['config']['training'].get('lora_parameter_dtype', 'bfloat16') == 'bfloat16',
                'FP32 LoRA requires the original BF16 parent')
        _validate_lora_state_dtype(payload.get('trainable_model'), 'bfloat16')
    raw = copy.deepcopy(payload['config'])
    del payload
    require(raw['data']['num_frames'] == 49 and is_pinned_base_model(raw['model']['base_model_path'])
            and is_pinned_base_model(str(model)), 'parent49 and exact pinned Wan base revision required')
    raw['model']['base_model_path'] = str(model)
    raw['data'].update(num_frames=97, manifest_path=str(data / 'manifests/train.jsonl'),
                       manifest_sha256=sha256(data / 'manifests/train.jsonl'), prompt_cache_path=None,
                       data_factory=PREFIX_FACTORY if gt_prefix_mix else FACTORY)
    # The explicit diagnostic changes only adapter_lr, not trainable membership.
    original_adapter_lr = raw['optimizer']['adapter_lr']
    original_lora_lr = raw['optimizer']['lora_lr']
    if freeze_action_adapter:
        require(type(original_adapter_lr) in (int, float) and 0 < original_adapter_lr < float('inf'),
                'adapter freeze requires a positive original parent learning rate')
        raw['optimizer']['adapter_lr'] = 0.0
    if gt_prefix_mix:
        raw['optimizer']['lora_lr'] = 2e-6
    # Preserve all other optimizer settings, accumulation, checkpoint interval and workers.
    raw['training'].update(max_steps=20, output_dir=str(output / 'training'))
    if lora_fp32:
        raw['training']['lora_parameter_dtype'] = 'float32'
    if low_sigma:
        raw['data']['low_sigma_stratification'] = True
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / 'training.yaml'
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding='utf-8')
    config = load_config(config_path)
    hashes = validate_training(config, config_path, gt_prefix_mix=gt_prefix_mix, lora_fp32=lora_fp32, low_sigma=low_sigma)
    request = dict(job_id=output.name, mode='window97_train', max_seconds=max_seconds,
        config=str(config_path), config_sha256=sha256(config_path), parent=str(parent), parent_sha256=parent_sha,
        verification_cache=str(verification_cache), resume=None, stop_after_step=stop_after_step,
        saved_tensors_on_cpu=saved_tensors_on_cpu, source_revision=source_revision,
        experiment_revision='r005-native97-resampled-v1', declared_changes=['rgb_window49_to97', 'absolute_index_resampled_windows'],
        training_manifest_hashes=hashes, configured_max_steps=20, quality_improvement_verified=False)
    if freeze_action_adapter:
        request.update(freeze_action_adapter=True,
            experiment_revision='r005-native97-resampled-adapter-lr-zero-v1',
            optimizer_override=dict(field='adapter_lr', source=original_adapter_lr, target=0.0,
                                    requires_grad_unchanged=True, complete_trainable_state_preserved=True))
        request['declared_changes'].append('action_adapter_lr_to_zero')
    if gt_prefix_mix:
        request.update(gt_prefix_mix=True, prefix_contract=prefix_contract(),
            experiment_revision='r005-native97-gt-prefix-mix-fixed-adapter-v1',
            lora_optimizer_override=dict(field='lora_lr', source=original_lora_lr, target=2e-6))
        request['declared_changes'].extend(['GT_clean_prefix_1_or2_equal_probability', 'lora_lr_to_2e-6'])
    if lora_fp32:
        from train_action_teacher import fp32_lora_precision_transition
        request.update(lora_fp32=True, precision_transition=fp32_lora_precision_transition(),
                       experiment_revision='r005-gt97-fp32-lora-parameters-stop2-v1')
        request['declared_changes'].append('FP32_LoRA_parameters_and_gradient_accumulation_BF16_compute')
    if low_sigma:
        from training.data.low_sigma import low_sigma_contract
        request.update(low_sigma=True, low_sigma_contract=low_sigma_contract(),
                       experiment_revision='r005-gt97-fp32-low-sigma-two-of-eight-v1')
        request['declared_changes'].append('two_of_eight_sigma_point2_times_original_u')
    write_json(output / 'request.json', request)
    return request


def run(request, directory, lease):
    from train_action_teacher import launch
    from training.config import load_config
    from training.demo.backend import validate_active_authority
    began = time.monotonic()
    directory = contained(directory, RUNTIME)
    validate_active_authority(lease)
    require(request.get('mode') == 'window97_train', 'wrong native97 worker mode')
    require(type(request.get('max_seconds')) is int and 60 <= request['max_seconds'] <= 3600, 'invalid finite watchdog')
    require(type(request.get('stop_after_step')) is int and 0 < request['stop_after_step'] <= 20, 'invalid native97 stop step')
    config_path = contained(request['config'], RUNTIME)
    require(sha256(config_path) == request['config_sha256'], 'native97 config changed')
    config = load_config(config_path)
    contained(config.training.output_dir, RUNTIME)
    gt_prefix_mix = request.get('gt_prefix_mix', False)
    require(type(gt_prefix_mix) is bool, 'gt_prefix_mix must be an explicit boolean')
    if gt_prefix_mix:
        require(request.get('prefix_contract') == prefix_contract() and request.get('freeze_action_adapter') is True,
                'GT prefix request contract or fixed-adapter declaration changed')
    lora_fp32 = request.get('lora_fp32', False)
    low_sigma = request.get('low_sigma', False)
    from training.data.low_sigma import low_sigma_contract, validate_opt_in
    validate_opt_in(config, low_sigma)
    require(request.get('low_sigma_contract') == (low_sigma_contract() if low_sigma else None),
            'low_sigma request policy differs')
    continuation = request.get('fp32_measured_continuation', False)
    require(type(continuation) is bool, 'fp32_measured_continuation must be an explicit boolean')
    measured = None
    if continuation:
        measured = request.get('measured_continuation', {})
        require(lora_fp32 is True and request['stop_after_step'] == 20 and request['max_seconds'] <= 2700,
                'measured FP32 continuation is limited to step20 and2700 seconds')
        require(measured == measured_fp32_continuation(measured.get('source_job', '')),
                'measured FP32 continuation evidence changed')
        require(request.get('resume') == measured['checkpoint'] and request['config'] == measured['config']
                and request['config_sha256'] == measured['config_sha256'],
                'measured continuation must resume the exact assessed checkpoint and configuration')
    else:
        require('measured_continuation' not in request, 'measured evidence requires its explicit continuation flag')
    if lora_fp32:
        from train_action_teacher import fp32_lora_precision_transition
        require(request.get('precision_transition') == fp32_lora_precision_transition()
                and (request['stop_after_step'] == 2 or continuation) and request.get('freeze_action_adapter') is True,
                'FP32 LoRA request requires exact precision transition and stop2 or measured continuation')
    else:
        require('precision_transition' not in request, 'legacy request must not relabel FP32 LoRA precision')
    require(validate_training(config, config_path, gt_prefix_mix=gt_prefix_mix,
                              lora_fp32=lora_fp32, low_sigma=low_sigma) == request['training_manifest_hashes'],
            'native97 training cache changed')
    resume = request.get('resume')
    freeze_action_adapter = request.get('freeze_action_adapter', False)
    require(type(freeze_action_adapter) is bool, 'freeze_action_adapter must be an explicit boolean')
    if freeze_action_adapter:
        require(type(config.optimizer.adapter_lr) in (int, float) and config.optimizer.adapter_lr == 0,
                'adapter-freeze request requires exact configured adapter_lr=0')
    if resume is not None:
        contained(resume, config.training.output_dir)
    else:
        contained(request['parent'], RUNTIME.parent)
        contained(request['verification_cache'], RUNTIME)
    launch(config, str(config_path), resume, None,
        length_transition_from=None if resume else request['parent'],
        length_transition_sha256=None if resume else request['parent_sha256'],
        length_transition_resampled=not bool(resume), base_model_path_relocated=not bool(resume),
        length_transition_verification_cache=None if resume else request['verification_cache'],
        freeze_action_adapter=freeze_action_adapter and not bool(resume),
        gt_prefix_mix=gt_prefix_mix and not bool(resume),
        lora_fp32=lora_fp32,
        low_sigma=low_sigma,
        fp32_measured_continuation=measured,
        shared_lease=lease, shared_max_seconds=request['max_seconds'],
        shared_request_sha256=sha256(directory / 'request.json'),
        saved_tensors_on_cpu=request.get('saved_tensors_on_cpu', False),
        stop_after_step=request['stop_after_step'], profile_runtime=True)
    execution = json.loads((Path(config.training.output_dir) / 'execution_result.json').read_text())
    require(execution['step'] == request['stop_after_step'], 'native97 training stopped before requested step')
    result = dict(status='completed', mode='window97_train', execution=execution,
                elapsed_seconds=time.monotonic() - began, training_rgb_frames=97,
                configured_max_steps=20, quality_improvement='not_yet_evaluated')
    if gt_prefix_mix:
        result.update(gt_prefix_mix=True, prefix_contract=prefix_contract())
    if lora_fp32:
        result.update(lora_fp32=True, precision_transition=fp32_lora_precision_transition())
    if continuation:
        result['measured_continuation'] = measured
    if low_sigma:
        result.update(low_sigma=True, low_sigma_contract=low_sigma_contract())
    return result


def resume_request(*, original, output, stop_after_step=20, max_seconds=3600,
                   fp32_measured_continuation=False, source_revision=None, low_sigma=False):
    from training.studio.subset_train import resume_request as prepare_resume
    original = contained(original, RUNTIME)
    previous = json.loads((original / 'request.json').read_text())
    require(type(low_sigma) is bool and low_sigma == previous.get('low_sigma', False),
            'low_sigma resume requires the same explicit policy as the original job')
    require(previous.get('mode') == 'window97_train' and previous.get('configured_max_steps') == 20,
            'resume requires this native97 pilot, not another branch')
    require(type(stop_after_step) is int and 0 < stop_after_step <= 20, 'invalid native97 resume step')
    require(type(fp32_measured_continuation) is bool, 'fp32_measured_continuation must be an explicit boolean')
    if source_revision is not None:
        require(isinstance(source_revision, str) and bool(source_revision.strip()), 'invalid resume source_revision')
    if fp32_measured_continuation:
        require(stop_after_step == 20 and type(max_seconds) is int and 60 <= max_seconds <= 2700,
                'measured FP32 continuation is limited to step20 and2700 seconds')
        require(isinstance(source_revision, str) and bool(source_revision.strip()),
                'measured continuation requires its actual deployment source_revision')
        binding = measured_fp32_continuation(original)
        output = contained(output, RUNTIME)
        require(not output.exists(), 'new resume job identity required')
        request = dict(previous, job_id=output.name, max_seconds=max_seconds, resume=binding['checkpoint'],
            stop_after_step=20, fp32_measured_continuation=True, measured_continuation=binding,
            profile_source_revision=previous['source_revision'], source_revision=source_revision,
            experiment_revision=('r005-gt97-fp32-low-sigma-measured-step2-to20-v1' if low_sigma
                                 else 'r005-gt97-fp32-lora-measured-step2-to20-v1'))
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'request.json', request)
        return request
    require(not previous.get('lora_fp32', False) or stop_after_step == 2,
            'FP32 LoRA pilot does not authorize continuation beyond step2')
    request = prepare_resume(original=original, output=output, stop_after_step=stop_after_step, max_seconds=max_seconds)
    if source_revision is not None:
        request.update(profile_source_revision=previous['source_revision'], source_revision=source_revision)
        write_json(Path(output) / 'request.json', request)
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    create = commands.add_parser('prepare')
    for name in ('data', 'output'):
        create.add_argument('--' + name, type=Path, required=True)
    create.add_argument('--source-revision', required=True)
    create.add_argument('--parent', type=Path, default=PARENT)
    create.add_argument('--model', type=Path, default=MODEL)
    create.add_argument('--stop-after-step', type=int, default=2)
    create.add_argument('--max-seconds', type=int, default=900)
    create.add_argument('--saved-tensors-on-cpu', action='store_true')
    create.add_argument('--freeze-action-adapter', action='store_true',
                        help='set only adapter_lr=0 for the new R005-to-native97 stage; retain complete state')
    create.add_argument('--gt-prefix-mix', action='store_true',
                        help='explicit GT1/2 clean-prefix pilot; requires fixed adapter; LoRA LR2e-6; not self-history')
    create.add_argument('--lora-fp32', action='store_true',
                        help='GT-prefix/fixed-adapter stop2 only; FP32 LoRA parameters/gradients, BF16 compute')
    create.add_argument('--low-sigma', action='store_true',
                        help='explicit FP32 GT97 candidate: exactly two of eight samples use sigma=0.2*u')
    resume = commands.add_parser('resume')
    for name in ('original', 'output'):
        resume.add_argument('--' + name, type=Path, required=True)
    resume.add_argument('--stop-after-step', type=int, default=20)
    resume.add_argument('--max-seconds', type=int, default=3600)
    resume.add_argument('--source-revision', help='actual deployed revision; required for measured FP32 continuation')
    resume.add_argument('--low-sigma', action='store_true', help='explicitly retain the low-sigma sampling contract')
    resume.add_argument('--fp32-measured-continuation', action='store_true',
                        help='explicit settled FP32 resource-pilot continuation; exact step2 to20, max2700s')
    options = vars(parser.parse_args())
    operation = options.pop('operation')
    print(json.dumps(prepare(**options) if operation == 'prepare' else resume_request(**options), ensure_ascii=False))


if __name__ == '__main__':
    main()
