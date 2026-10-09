"""A-gated B20 evaluation; identical cached conditions, never automatic training."""
from __future__ import annotations

import copy
import gc
import os
from pathlib import Path
import re
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.demo.relocation import ArtifactHashes
from training.studio import paired_turn_eval as a
from training.studio import paired_turn_train as training

MODE = 'paired_turn_b_eval'
REVISION = 'paired-turn-b20-fixed-a-conditions-v1'
STAGE = 'controlled_action_paired_turn_native97_v1'
CHECKPOINT_SCHEMA = 'controlled_action_paired_turn_checkpoint_v1'
ROOT = a.ROOT
ENDPOINT = ROOT / 'cpt-b-to20-v1'
CHECKPOINT = ROOT / 'cpt-b-profile2-v1/training/checkpoints/step-0000020.pt'
TRAINING_CONFIG = ROOT / 'cpt-b-profile2-v1/training.yaml'
CASES = {case: job.replace('cpt-a-', 'cpt-b-', 1) for case, job in a.CASES.items()}
read = a.read


def baseline_job(target, condition):
    require((target, condition) in CASES, 'fixed J/L original/swap case required')
    return ROOT / a.CASES[target, condition]


def completed_endpoint():
    import train_paired_turn_teacher as entry
    gate = entry.visual_gate()  # Missing/positive/technical-failure A review must fail closed.
    request, result, settlement = (read(ENDPOINT / (name + '.json'))
                                    for name in ('request', 'result', 'settlement'))
    training.validate_request(request, ENDPOINT)
    require(result.get('status') == 'completed' and result.get('mode') == training.MODE
            and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed',
            'B evaluation requires a completed and settled B training endpoint')
    execution = result['execution']
    require(execution.get('stage') == STAGE and execution.get('step') == 20
            and execution.get('checkpoint_path') == str(CHECKPOINT)
            and execution.get('micro_batches_consumed') == execution.get('new_stage_micro_batches_consumed') == 160
            and execution.get('historical_micro_batches_before_stage') == 320
            and execution.get('lifetime_micro_batches_consumed') == 480
            and request.get('stop_after_step') == 20 and request.get('config') == str(TRAINING_CONFIG),
            'only the actual B20 endpoint and new160/history320 exposures are permitted')
    require(execution['parent_lineage'] == request['parent_lineage']
            and execution['parent_lineage']['parent_checkpoint'] == str(a.PARENT_CHECKPOINT)
            and request['a_visual_gate'] == execution['a_visual_gate'] == gate,
            'B must retain common P and the unchanged valid negative A review')
    require(sha256(TRAINING_CONFIG) == request['config_sha256'] and CHECKPOINT.is_file(),
            'completed B checkpoint/config missing or changed')
    return gate, request, result


def checkpoint_digest():
    protected = [TRAINING_CONFIG, ENDPOINT / 'request.json', ENDPOINT / 'result.json', ROOT / 'a/review-result.json']
    return ArtifactHashes(ROOT / '.b-eval-checkpoint-hashes.json', protected_paths=protected)(CHECKPOINT)


def validate_request(request):
    case = (request.get('target'), request.get('condition'))
    require(request.get('mode') == MODE and request.get('experiment_revision') == REVISION
            and request.get('source_revision') == a.SOURCE_REVISION, 'explicit fixed B evaluation revision required')
    require(case in CASES and request.get('job_id') == CASES[case], 'only four fixed B cases permitted')
    expected = dict(arm='B_new_pool', model_arm='frozen', checkpoint=str(CHECKPOINT),
                    checkpoint_stage=STAGE, checkpoint_schema=CHECKPOINT_SCHEMA, checkpoint_step=20,
                    training_config=str(TRAINING_CONFIG), endpoint_job=str(ENDPOINT),
                    baseline_job=str(baseline_job(*case)), a_review=str(ROOT / 'a/review-result.json'),
                    seed=42, steps=40, cfg_scale=5, future_frames=96, solver='flow_euler', fastpath=False)
    require(all(request.get(key) == value for key, value in expected.items()), 'B source/sampling contract changed')
    require(type(request.get('max_seconds')) is int and 60 <= request['max_seconds'] <= 1200,
            'finite <=1200-second B evaluation required')
    for key in ('checkpoint_sha256', 'training_config_sha256', 'a_review_sha256',
                'baseline_request_sha256', 'baseline_result_sha256', 'baseline_settlement_sha256',
                'endpoint_request_sha256', 'endpoint_result_sha256', 'endpoint_settlement_sha256'):
        require(isinstance(request.get(key), str) and re.fullmatch('[0-9a-f]{64}', request[key]),
                'pinned completed source records required')
    require(not any(key in request for key in ('action_scale', 'ratio_cap', 'gate_scale', 'lora_strength',
            'initial_latent', 'future_clean_latents', 'prompt', 'actions', 'window_index', 'episode_id', 'source_start')),
            'no input, offset, scale, or future-latent overrides allowed')


def make_request(*, source_revision, target, condition, max_seconds=1200, job_id=None):
    require((target, condition) in CASES and (job_id is None or job_id == CASES[target, condition]),
            'fixed unique B identity required')
    gate, endpoint_request, _ = completed_endpoint()
    baseline = baseline_job(target, condition)
    request = dict(mode=MODE, experiment_revision=REVISION, source_revision=source_revision,
        job_id=CASES[target, condition], target=target, condition=condition, arm='B_new_pool', model_arm='frozen',
        checkpoint=str(CHECKPOINT), checkpoint_stage=STAGE, checkpoint_schema=CHECKPOINT_SCHEMA, checkpoint_step=20,
        checkpoint_sha256=checkpoint_digest(), training_config=str(TRAINING_CONFIG),
        training_config_sha256=endpoint_request['config_sha256'], endpoint_job=str(ENDPOINT),
        baseline_job=str(baseline), a_review=gate['review_path'], a_review_sha256=gate['review_sha256'],
        seed=42, steps=40, cfg_scale=5, future_frames=96, solver='flow_euler', fastpath=False, max_seconds=max_seconds)
    for prefix, directory in (('baseline', baseline), ('endpoint', ENDPOINT)):
        for name in ('request', 'result', 'settlement'):
            request[prefix + '_' + name + '_sha256'] = sha256(directory / (name + '.json'))
    validate_request(request)
    return request


def load_inputs(request):
    """Reuse A's prepared, z0-only path; no checkpoint is loaded on preparation."""
    validate_request(request)
    gate, _, endpoint_result = completed_endpoint()
    require(gate['review_sha256'] == request['a_review_sha256'], 'A visual decision changed')
    baseline = Path(request['baseline_job'])
    for prefix, directory in (('baseline', baseline), ('endpoint', ENDPOINT)):
        for name in ('request', 'result', 'settlement'):
            require(sha256(directory / (name + '.json')) == request[prefix + '_' + name + '_sha256'],
                    'completed source record changed: ' + prefix + '/' + name)
    before_request, before, settlement = (read(baseline / (name + '.json'))
                                           for name in ('request', 'result', 'settlement'))
    a.validate_request(before_request)
    require(before_request['target'] == request['target'] and before_request['condition'] == request['condition']
            and before.get('request') == before_request and before.get('status') == 'completed'
            and before.get('mode') == a.MODE and before.get('frames_written') == 97
            and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed',
            'matching completed A case required')
    first, prompt, actions, initial, negative, _, original_lineage, _ = a.load_inputs(before_request)
    for name in ('selection', 'cache_binding_sha256', 'checkpoint', 'checkpoint_sha256',
                 'first_latent_sha256_float32', 'prompt_sha256_float32', 'prompt_text', 'negative_source',
                 'original_action_sha256_float32', 'actual_action_sha256_float32', 'original_actions', 'actual_actions'):
        require(original_lineage[name] == before['lineage'][name], 'A/B input identity differs: ' + name)
    require(before['generation']['first_latent_sha256_float32'] == original_lineage['first_latent_sha256_float32']
            and isinstance(before['generation'].get('noise_sha256_float32'), str), 'A generation identity missing')
    lineage = copy.deepcopy(original_lineage)
    indices = list(range(0 if request['target'] == 'J' else 1, 160, 2))
    lineage.update(arm='B_new_pool', checkpoint=str(CHECKPOINT), checkpoint_sha256=request['checkpoint_sha256'],
        A_training_exposure=lineage['training_exposure'], A_checkpoint=before['lineage']['checkpoint'],
        A_checkpoint_sha256=before['lineage']['checkpoint_sha256'], A_baseline_job=str(baseline),
        A_baseline_request_sha256=request['baseline_request_sha256'], A_baseline_result_sha256=request['baseline_result_sha256'],
        A_noise_sha256_float32=before['generation']['noise_sha256_float32'], a_visual_gate=gate,
        training_exposure=dict(new_stage_sample_indices=indices, additional_exposures=80,
            source_start=a.STARTS[request['target']], original_recorded_training_actions=True,
            sampling='same_episode_J_L_exact_offsets_alternating', historical_micro_batches_before_stage=320,
            lifetime_micro_batches_consumed=480), actually_seen_by_endpoint=True,
        method_scope='equal_training_pool_replacement_single_episode_in_sample_not_generalization',
        overfit_diagnostic=True, data_coverage_causal_proof=False, automatic_further_training=False,
        automatic_B_training=False, teacher_ready=False, reference_memory_implemented=False)
    return first, prompt, actions, initial, negative, lineage, before, endpoint_result


def _load_checkpoint(request, endpoint_result):
    import torch
    import train_paired_turn_teacher as entry
    require(sha256(TRAINING_CONFIG) == request['training_config_sha256']
            and checkpoint_digest() == request['checkpoint_sha256'], 'B checkpoint/config changed')
    config = entry.load_config(TRAINING_CONFIG)
    hashes = entry.artifact_hashes(config, TRAINING_CONFIG)
    payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    entry.validate_checkpoint(payload, config=config, hashes=hashes)
    entry.validate_base_identity(payload, config)
    require(payload['stage'] == STAGE and payload['checkpoint_schema'] == CHECKPOINT_SCHEMA and payload['step'] == 20
            and payload['micro_batches_consumed'] == payload['new_stage_micro_batches_consumed'] == 160
            and payload['historical_micro_batches_before_stage'] == 320 and payload['lifetime_micro_batches_consumed'] == 480,
            'actual B checkpoint stage/exposure differs')
    execution = endpoint_result['execution']
    for key in ('parent_lineage', 'sampling_contract', 'a_visual_gate', 'a_reference'):
        require(payload[key] == execution[key], 'actual B endpoint metadata differs: ' + key)
    require(payload['parent_lineage']['parent_checkpoint'] == str(a.PARENT_CHECKPOINT), 'B inherited A instead of common P')
    lineage = dict(stage=STAGE, checkpoint_schema=CHECKPOINT_SCHEMA, step=20, arm='frozen',
        checkpoint=str(CHECKPOINT), checkpoint_sha256=request['checkpoint_sha256'],
        new_stage_micro_batches_consumed=160, historical_micro_batches_before_stage=320, lifetime_micro_batches_consumed=480,
        parent_lineage=payload['parent_lineage'], sampling_contract=payload['sampling_contract'],
        method_contract=payload['method_contract'], precision_contract=payload['precision_contract'],
        a_visual_gate=payload['a_visual_gate'], a_reference=payload['a_reference'], original_B_config=config.to_dict(),
        model_config_projection='factory_and_old_manifest_pin_only_for_unchanged_model_constructor;assets_validated_against_new_manifest',
        trained_lora=False, reference_memory_implemented=False, complete_abot_reproduction=False)
    return payload['trainable_model'], entry.model_config(config), lineage


def prepare(*, output, **options):
    output = contained(output, ROOT)
    request = make_request(**options)
    require(not output.exists() and output.name == request['job_id'], 'new matching B evaluation directory required')
    values = load_inputs(request)
    lineage = values[5]
    del values
    gc.collect()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'request.json', request)
    write_json(output / 'preparation.json', dict(status='cpu_prepared', request=request, lineage=lineage,
        weights_loaded=False, gpu_launched=False, video_generated=False, automatic_further_training=False))
    return request


def validate_generation(generation, forward_calls, before):
    stages = generation['stage_totals_seconds']
    require(generation['frames_written'] == 97 and forward_calls == 80
            and 'cached_z0_condition' in stages and 'encode_initial' not in stages and 'encode_endpoint' not in stages,
            'B must use cached z0 and97 frames/40 CFG pairs without encoding')
    for name in ('first_latent_sha256_float32', 'noise_sha256_float32'):
        require(generation[name] == before['generation'][name], 'A/B generation identity differs: ' + name)


def run(request, directory, lease):
    import numpy as np
    import torch
    from PIL import Image
    from training.demo.backend import validate_active_authority
    from training.eval.rollout15s import ffmpeg_writer_factory
    validate_request(request)
    directory = contained(directory, ROOT)
    require(directory.name == request['job_id'], 'B job directory differs')
    target = directory / 'matched-evaluation'
    target.mkdir(exist_ok=False)
    began = time.monotonic()
    runtime = engine = None
    report = dict(status='running', mode=MODE, revision=REVISION, request=request, arm='B_new_pool',
        future_clean_latents_supplied=False, independent_dev_evaluation=False, generalization_verified=False,
        quality_improvement_verified=False, action_correctness_verified=False, realtime=False,
        data_coverage_causal_proof=False, automatic_further_training=False)

    def check():
        require(time.monotonic() - began < request['max_seconds'] - 20, 'finite B evaluation deadline reached')
        require(a.matched.directory_bytes(directory) < a.matched.JOB_BYTES_LIMIT - 100_000, 'B job20MB cap reached')
        validate_active_authority(lease)

    try:
        check()
        require(os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'offline assets required')
        first, prompt, actions, initial, negative, lineage, before, endpoint_result = load_inputs(request)
        state, config, checkpoint_lineage = _load_checkpoint(request, endpoint_result)
        lineage['checkpoint_lineage'] = checkpoint_lineage
        report['lineage'] = lineage
        Image.fromarray(initial).save(target / 'initial.png')
        np.save(target / 'actions.npy', actions, allow_pickle=False)
        write_json(target / 'inputs.json', dict(request=request, lineage=lineage, actual_actions=actions.tolist()))
        check()
        tick = time.monotonic()
        runtime = a.paired.ControlledPairRuntime(state, config, negative, arm='frozen')
        state.clear()
        gc.collect()
        report['load_seconds'] = time.monotonic() - tick
        engine = a.matched.CachedInitialEngine({}, runtime.config, residency='phase', fastpath=False, _runtime=runtime,
            cached_first=first, _writer_factory=lambda *args:a.matched.BoundedPreservingWriter(ffmpeg_writer_factory(*args)))
        engine.load_seconds = report['load_seconds']
        generation, _ = engine.run_segment(initial=initial, prompt=prompt, actions=actions, seed=42,
            output_path=target / 'raw.mp4', steps=40)
        validate_generation(generation, runtime.forward_calls, before)
        generation.update(initial_condition='same_A_cached_z0;RGB0_display_only',
            continuation='independent_cached_z0_condition;no_future_ground_truth', vae_encoding_used=False)
        generation['sampler'].update(cfg='text_classifier_free', cfg_scale=5, model_forwards_per_step=2)
        report.update(status='completed', frames_written=97, generation=generation, raw_video=str(target / 'raw.mp4'),
            actual_model_forward_calls=runtime.forward_calls, peak_allocated_bytes=generation['peak_vram_bytes'],
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), quality_evaluation='awaiting_visual_review')
        check()
    except BaseException as error:
        report.update(status='failed', failure_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            if engine is not None:
                engine.close()
            elif runtime is not None:
                runtime.close()
        finally:
            report['elapsed_seconds'] = time.monotonic() - began
            write_json(target / 'result.json', report)
    return report
