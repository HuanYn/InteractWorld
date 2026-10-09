"""One opt-in6s evaluation of completed FP32-LoRA GT97 step20.

CPU preparation waits for completed and settled training. The original Demo,
legacy GT loaders, checkpoint bytes and BF16 Euler arithmetic remain unchanged.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import re
import shutil
import time

from training.demo.contracts import (ACTION_STAGE, ACTION_WINDOW6_METHOD, action_array, contained,
    load_initial, require, sha256, write_json)
from training.demo.action_backend import (GT_PREFIX_PARENT_SHA, GT_PREFIX_PARENT_SOURCE,
    load_action_inputs, validate_action_config)
from training.demo.relocation import ArtifactHashes, artifact_paths
from training.demo.latency_engine import ReusableActionEngine
from training.studio.quality_probe import DEFAULT_NEGATIVE_CACHE, QualityRuntime, annotate_report, load_negative
from training.studio.subset_eval import CONTROLS, FIXED_INPUTS
from training.studio.timestep_precision_probe import EXPECTED_INPUTS, check_original_input_digests

DATA_ROOT = Path('/DATA_DISK/InteractWorld')
NATIVE = DATA_ROOT / 'creator-runtime/native97-20260928'
PROFILE = NATIVE / 'native97-lora-fp32-profile-0929'
TRAINING_JOB = NATIVE / 'native97-lora-fp32-train20-0929'
CHECKPOINT = PROFILE / 'training/checkpoints/step-0000020.pt'
CONFIG = PROFILE / 'training.yaml'
REVISION = 'FP32-LoRA-GT97-completed20-original6s-BF16-inference-v1'
CASE = dict(name='fp32_lora_step20', mode='segment6', steps=40, cfg_scale=5,
            lora_strength=1, action_scale=.03, solver='flow_euler')
FIXED = dict(mode='fp32_lora_eval_probe', max_seconds=600, seed=42, width=832, height=480,
    future_frames=96, residency='co_resident', fastpath=False, negative_cache=DEFAULT_NEGATIVE_CACHE,
    parent_source_job=str(GT_PREFIX_PARENT_SOURCE), parent_checkpoint_sha256=GT_PREFIX_PARENT_SHA,
    training_job=str(TRAINING_JOB), training_config=str(CONFIG), checkpoint=str(CHECKPOINT),
    steps=40, cfg_scale=5, lora_strength=1, action_scale=.03, solver='flow_euler',
    fp32_lora_candidate=True, **EXPECTED_INPUTS)


def evaluation_scope(low_sigma=False):
    require(type(low_sigma) is bool, 'low-sigma evaluation must be an explicit boolean')
    if not low_sigma:
        return FIXED, TRAINING_JOB, CHECKPOINT, CONFIG, CASE, REVISION
    profile = NATIVE / 'native97-low-sigma-profile-0929'
    training_job = NATIVE / 'native97-low-sigma-train20-0929'
    checkpoint, config = profile / 'training/checkpoints/step-0000020.pt', profile / 'training.yaml'
    fixed = dict(FIXED, low_sigma_candidate=True, training_job=str(training_job),
                 training_config=str(config), checkpoint=str(checkpoint))
    return fixed, training_job, checkpoint, config, dict(CASE, name='low_sigma_step20'), 'low-sigma-FP32-GT20-original6s-v1'


def validate_request(request):
    require(type(request) is dict, 'evaluation request must be a mapping')
    fixed, _, _, _, _, revision = evaluation_scope(request.get('low_sigma_candidate', False))
    require(set(request) == set(fixed) |
        {'job_id', 'source_revision', 'experiment_revision', 'source_job', 'checkpoint_sha256', 'preparation_sha256'},
        'unexpected FP32 evaluation fields')
    require(all(request[key] == value and type(request[key]) is type(value) for key, value in fixed.items()),
            'single completedFP32step20/original6s/40EulerCFG5 scope is fixed')
    require(type(request['job_id']) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', request['job_id'])
            and type(request['source_revision']) is str and bool(request['source_revision'])
            and request['experiment_revision'] == revision, 'invalid FP32 evaluation identity')
    for field in ('checkpoint_sha256', 'preparation_sha256'):
        require(type(request[field]) is str and re.fullmatch('[0-9a-f]{64}', request[field]), 'invalid candidate binding')
    source = contained(request['source_job'], DATA_ROOT)
    require(source.name == 'source' and source.parent.name == request['job_id'], 'private candidate source required')


def completed_training(*, low_sigma=False):
    """Only small completion records here;actual tensor/config checks stay in loader."""
    _, training_job, checkpoint, config_path, _, _ = evaluation_scope(low_sigma)
    request = json.loads((training_job / 'request.json').read_text())
    result = json.loads((training_job / 'result.json').read_text())
    settlement = json.loads((training_job / 'settlement.json').read_text())
    require(request.get('low_sigma', False) == low_sigma, 'training low-sigma scope differs')
    execution = result.get('execution', {})
    require(request.get('mode') == 'window97_train' and request.get('lora_fp32') is True
            and request.get('stop_after_step') == request.get('configured_max_steps') == 20
            and Path(request.get('config', '')).resolve() == config_path.resolve()
            and request.get('parent_sha256') == GT_PREFIX_PARENT_SHA,
            'requires the explicit FP32 same-config resume20 request')
    require(result.get('status') == 'completed' and result.get('lora_fp32') is True
            and execution.get('step') == 20 and execution.get('micro_batches_consumed') == 160
            and Path(execution.get('checkpoint_path', '')).resolve() == checkpoint.resolve()
            and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed',
            'completed20-step training and successful settlement required before evaluation')
    require(sha256(config_path) == request.get('config_sha256') and checkpoint.is_file(),
            'completed candidate config/checkpoint is missing or changed')
    return dict(training_job=str(training_job), request_sha256=sha256(training_job / 'request.json'),
        result_sha256=sha256(training_job / 'result.json'), settlement_sha256=sha256(training_job / 'settlement.json'),
        completed_step=20, micro_batches_consumed=160, settled=True)


def prepare(*, output, dev_prompt_sidecar, source_revision, low_sigma=False):
    import numpy as np
    import yaml
    from training.config import load_config
    fixed, _, checkpoint, config_path, _, revision = evaluation_scope(low_sigma)
    directory, prompt_cache = contained(output, DATA_ROOT), contained(dev_prompt_sidecar, DATA_ROOT)
    require(not directory.exists(), 'new evaluation output required;do not overwrite old evidence')
    completion = completed_training(low_sigma=True) if low_sigma else completed_training()
    parent = GT_PREFIX_PARENT_SOURCE
    original = json.loads((parent / 'request.json').read_text())
    for filename, field in FIXED_INPUTS:
        require(sha256(parent / filename) == original[field], f'original source changed:{filename}')
    raw = yaml.safe_load((parent / 'rollout.yaml').read_text())
    validate_action_config(raw)
    require(raw['method'] == ACTION_WINDOW6_METHOD and len(raw['scenes']) == 1
            and raw['lineage']['checkpoint_sha256'] == original['checkpoint_sha256'] == GT_PREFIX_PARENT_SHA,
            'original R005 window6 source required')
    scene = copy.deepcopy(raw['scenes'][0])
    require(scene['source_episode_id'] == original['source_episode_id'] == '019285161503b1d49c72e81863eb17ef'
            and scene['prompt'] == original['prompt'] and scene['seed'] == original['seed'] == 42
            and scene['action_segments'] == original['action_segments'] == CONTROLS,
            'original image/text/control binding differs')
    initial, actions = load_initial(parent / 'initial.npy'), np.load(parent / 'actions.npy', allow_pickle=False)
    require(initial.shape == (480, 832, 3) and initial.dtype == np.uint8
            and actions.dtype == np.float32 and np.array_equal(actions, action_array(CONTROLS)), 'original geometry/actions differ')
    config = load_config(config_path)
    paths = artifact_paths(config, config_path, {})
    verified = raw['lineage'].get('verification_cache')
    require(verified and contained(verified, DATA_ROOT).is_file(), 'existing verified-artifact cache required')
    receipt = prompt_cache.with_suffix(prompt_cache.suffix + '.receipt.json')
    digest = ArtifactHashes(verified, protected_paths=[checkpoint, *paths.values(), prompt_cache, receipt,
        Path(config.model.base_model_path) / 'Wan2.2_VAE.pth'])
    candidate_sha = digest(checkpoint)
    source = directory / 'source'
    scene['initial_frame_path'] = str(source / 'initial.npy')
    frozen = copy.deepcopy(raw)
    frozen.update(run_id=source.name, scenes=[scene], output_root=str(source))
    frozen['lineage'] = dict(checkpoint_path=str(checkpoint), checkpoint_sha256=candidate_sha,
        expected_stage=ACTION_STAGE, expected_base_model_path=config.model.base_model_path,
        artifact_paths={key: str(value) for key, value in paths.items()}, verification_cache=verified,
        gt_prefix_mix_candidate=True, fp32_lora_candidate=True,
        eval_prompt_sidecar=dict(cache_path=str(prompt_cache), cache_sha256=digest(prompt_cache),
            receipt_path=str(receipt), receipt_sha256=digest(receipt)))
    if low_sigma:
        frozen['lineage']['low_sigma_candidate'] = True
    validated = (load_action_inputs(frozen, scene, allow_fp32_lora_candidate=True, allow_low_sigma_candidate=True)
                 if low_sigma else load_action_inputs(frozen, scene, allow_fp32_lora_candidate=True))
    lineage = copy.deepcopy(validated[3])
    del validated
    gc.collect()
    metadata = dict(status='cpu_evaluation_prepared', checkpoint=dict(path=str(checkpoint), sha256=candidate_sha, step=20),
        training_completion=completion, candidate_contract=lineage['fp32_lora_candidate_contract'],
        initialization=lineage['initialization'], training_config=str(config_path), prompt_condition=lineage['prompt_condition'],
        parent_source_job=str(parent), original_parent_checkpoint_sha256=GT_PREFIX_PARENT_SHA,
        fixed_inputs=dict(seed=42, prompt=scene['prompt'], source_episode_id=scene['source_episode_id'],
            initial_sha256=original['initial_sha256'], actions_sha256=original['actions_sha256'],
            action_segments=copy.deepcopy(CONTROLS), **EXPECTED_INPUTS),
        source_revision=source_revision, gpu_launched=False, video_generated=False,
        quality_improvement_verified=False, checkpoint_rewritten=False)
    directory.mkdir(parents=True, exist_ok=False)
    source.mkdir()
    for filename, _ in FIXED_INPUTS[1:]:
        shutil.copyfile(parent / filename, source / filename)
    rollout = source / 'rollout.yaml'
    rollout.write_text(yaml.safe_dump(frozen, sort_keys=False, allow_unicode=True), encoding='utf-8')
    write_json(source / 'request.json', dict(original, job_id=source.name, config_sha256=sha256(rollout),
        checkpoint_sha256=candidate_sha, parent_source_job=str(parent), parent_checkpoint_sha256=GT_PREFIX_PARENT_SHA,
        cpu_input_preparation_only=True, gt_prefix_mix_candidate=True, fp32_lora_candidate=True))
    write_json(directory / 'preparation.json', metadata)
    request = dict(copy.deepcopy(fixed), job_id=directory.name, source_revision=source_revision, experiment_revision=revision,
        source_job=str(source), checkpoint_sha256=candidate_sha, preparation_sha256=sha256(directory / 'preparation.json'))
    validate_request(request)
    write_json(directory / 'request.json', request)
    return dict(status='cpu_evaluation_prepared', request=str(directory / 'request.json'), checkpoint_sha256=candidate_sha,
                gpu_launched=False, video_generated=False)


class FixedInputQualityRuntime(QualityRuntime):
    def generate(self, first, prompt, keys, noise, steps, *, fastpath=False):
        require(steps == 40 and fastpath is False and self.forward_calls == 0,
                'one6s candidate only;no retry or changed sampler')
        check_original_input_digests(first, noise)
        return super().generate(first, prompt, keys, noise, steps, fastpath=False)


def inference_precision(model):
    import torch
    lora = [value for name, value in model.named_parameters() if name.endswith(('.lora_a.weight', '.lora_b.weight'))]
    adapter = [value for name, value in model.named_parameters() if 'act_control_adapter.' in name]
    require(len(lora) == 600 and len(adapter) == 6
            and all(value.dtype == torch.bfloat16 and not value.requires_grad for value in lora + adapter),
            'actual frozen inference parameters must be BF16')
    return dict(lora_tensors=len(lora), adapter_tensors=len(adapter), actual_inference_dtype='torch.bfloat16',
                checkpoint_lora_dtype='torch.float32', checkpoint_rewritten=False)


def run(request, directory, lease):
    import numpy as np
    from PIL import Image
    from training.demo.backend import validate_active_authority
    from training.studio.worker import read_source
    validate_request(request)
    low_sigma = request.get('low_sigma_candidate', False)
    _, _, _, _, CASE, _ = evaluation_scope(low_sigma)
    directory = contained(directory, DATA_ROOT)
    require(directory.name == request['job_id'] and Path(request['source_job']).resolve() == directory / 'source'
            and not any((directory / name).exists() for name in ('inputs.json', 'progress.json', CASE['name'])), 'new prepared evaluation required')
    require(sha256(directory / 'preparation.json') == request['preparation_sha256'], 'candidate preparation changed')
    preparation = json.loads((directory / 'preparation.json').read_text())
    completion = completed_training(low_sigma=True) if low_sigma else completed_training()
    require(completion == preparation['training_completion'], 'completed training records changed')
    source_record = json.loads((directory / 'source/request.json').read_text())
    require(all(source_record.get(field) == preparation['fixed_inputs'][field]
                for field in ('initial_sha256', 'actions_sha256')),
            'candidate source must retain the prepared original image/actions')
    began, runtime, engine, cold = time.monotonic(), None, None, 0.
    def check():
        require(time.monotonic() - began < 570, 'finite FP32 candidate deadline;no retry')
        validate_active_authority(lease)
    check()
    state, config, prompt, lineage, initial, actions, scene = (
        read_source(request, allow_fp32_lora_candidate=True, allow_low_sigma_candidate=True) if low_sigma else
        read_source(request, allow_fp32_lora_candidate=True))
    require(lineage['sha256'] == request['checkpoint_sha256'] == preparation['checkpoint']['sha256']
            and lineage['step'] == 20 and lineage['fp32_lora_candidate_contract'] == preparation['candidate_contract']
            and lineage['prompt_condition'] == preparation['prompt_condition'], 'candidate checkpoint/text contract changed')
    selected = actions[:96].copy()
    expected = np.zeros((96, 8), np.float32)
    expected[:, [0, 5]] = 1
    require(np.array_equal(selected, expected) and scene['seed'] == 42 and initial.shape == (480, 832, 3),
            'candidate first6s must retain original WJ96/seed/image geometry')
    negative, negative_source = load_negative(DEFAULT_NEGATIVE_CACHE, config.base_model_path)
    inputs = dict(mode=FIXED['mode'], prompt=scene['prompt'], prompt_condition=lineage['prompt_condition'],
        scene_id=scene['scene_id'], seed=42, width=832, height=480, checkpoint=lineage['path'],
        checkpoint_sha256=lineage['sha256'], checkpoint_step=20, candidate_contract=preparation['candidate_contract'],
        parent_checkpoint_sha256=GT_PREFIX_PARENT_SHA, checkpoint_initialization=lineage['initialization'],
        source_job=request['source_job'], parent_source_job=str(GT_PREFIX_PARENT_SOURCE), source_actions=scene['action_segments'],
        actual_actions=[dict(frames=96, keys=['W', 'J'])], negative_prompt=negative_source, cases=[CASE],
        expected_input_digests=EXPECTED_INPUTS, source_revision=request['source_revision'],
        training_completion=preparation['training_completion'], old_BF16_Euler_unchanged=True,
        checkpoint_rewritten=False, default_changed=False, quality_improvement_verified=False)
    if low_sigma:
        inputs['low_sigma_candidate'] = True
    write_json(directory / 'inputs.json', inputs)
    Image.fromarray(initial).save(directory / 'initial.png')
    target = directory / CASE['name']
    target.mkdir(exist_ok=False)
    write_json(directory / 'progress.json', dict(status='running', phase='loading'))
    try:
        check()
        tick = time.monotonic()
        runtime = FixedInputQualityRuntime(state, config, 'co_resident', negative)
        runtime.synchronize()
        cold = time.monotonic() - tick
        runtime.configure(CASE)
        inputs['inference_precision'] = inference_precision(runtime.model)
        write_json(directory / 'inputs.json', inputs)
        engine = ReusableActionEngine(state, config, residency='co_resident', fastpath=False, _runtime=runtime)
        engine.load_seconds = cold
        check()
        started = time.monotonic()
        write_json(directory / 'progress.json', dict(status='running', phase='generating', cases=[CASE]))
        report, endpoint = engine.run_segment(initial=initial, prompt=prompt, actions=selected,
            seed=42, output_path=target / 'raw.mp4', steps=40)
        del endpoint
        annotate_report(report, CASE, runtime, negative_source, cold)
        report.update(candidate_contract=preparation['candidate_contract'], inference_precision=inputs['inference_precision'],
                      checkpoint_sha256=lineage['sha256'], checkpoint_step=20)
        require(report['frames_written'] == 97 and runtime.forward_calls == 80
                and all(report[key] == value for key, value in EXPECTED_INPUTS.items()), '97frames/80calls/originalnoise mismatch')
        entry = dict(**CASE, status='completed', generation=report, prompt=scene['prompt'], seed=42,
            raw_video=str(target / 'raw.mp4'), elapsed_seconds=time.monotonic() - started,
            elapsed_seconds_scope='generation_excludes_one_time_model_load', quality_improvement_verified=False)
        write_json(target / 'result.json', entry)
    except BaseException as error:
        failed = copy.deepcopy(engine.last_run_report or {}) if engine else {}
        if runtime is not None:
            annotate_report(failed, CASE, runtime, negative_source, cold)
        entry = dict(**CASE, status='failed', generation=failed, error=f'{type(error).__name__}: {error}', automatic_retry=False)
        write_json(target / 'result.json', entry)
        write_json(directory / 'progress.json', dict(status='failed', cases=[entry]))
        raise
    finally:
        if engine is not None:
            engine.close()
        elif runtime is not None:
            runtime.close()
    result = dict(status='completed', mode=FIXED['mode'], cases=[entry], candidate_contract=preparation['candidate_contract'],
        checkpoint_sha256=lineage['sha256'], checkpoint_step=20, parent_checkpoint_sha256=GT_PREFIX_PARENT_SHA,
        elapsed_seconds=time.monotonic() - began, load_seconds=cold, default_changed=False,
        checkpoint_rewritten=False, automatic_retry=False, quality_improvement_verified=False, realtime=False)
    write_json(directory / 'progress.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dev-prompt-sidecar', required=True, type=Path)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--low-sigma', action='store_true', help='isolated completed low-sigma FP32 GT20 candidate')
    print(json.dumps(prepare(**vars(parser.parse_args())), ensure_ascii=False))


if __name__ == '__main__':
    main()
