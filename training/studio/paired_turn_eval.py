"""A-only evaluation on two exact new offsets, never automatic B training."""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import re
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.studio import controlled_matched_window_eval as matched
from training.studio import controlled_lora_pair_eval as paired
from training.studio import controlled_tinyfit_eval as tiny
from training.studio import paired_turn_cache as cache

MODE = 'paired_turn_eval'
REVISION = 'paired-turn-a20-new-offsets-original-jlswap-v1'
SOURCE_REVISION = 'creator-paired-turn-v1'
ROOT = matched.BASE / 'teacher-paired-turn-v1'
SOURCE_EVAL = tiny.ROOT / 'ctf-wj-original-v1/request.json'
SOURCE_RESULT = SOURCE_EVAL.with_name('result.json')
CHECKPOINT = tiny.ROOT / 'ctf-profile2-v1/training/checkpoints/step-0000020.pt'
TRAINING_CONFIG = tiny.ROOT / 'ctf-profile2-v1/training.yaml'
PARENT_CHECKPOINT = (matched.BASE / 'teacher-controlled-lora-pair-v1'
                     / 'clp-frozen-profile2-v2/training/checkpoints/step-0000020.pt')
EPISODE = '0328fbd39403bc7955f73eca6a9524f2'
STARTS = {'J': 0, 'L': 210}
CASES = {(target, condition): f'cpt-a-{target.lower()}-{suffix}-v1'
         for target in STARTS for condition, suffix in (('original', 'original'), ('jl_swap', 'swap'))}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def binding_digest(binding):
    # Only stable source/cache metadata; no full shard or weight hashing.
    value = {key: binding[key] for key in ('cache_index', 'cache_index_sha256',
             'cache_receipt', 'cache_receipt_sha256', 'windows', 'prompt_text', 'prompt_sha256_float32')}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def source_evaluation():
    source, result = read(SOURCE_EVAL), read(SOURCE_RESULT)
    tiny.validate_request(source)
    require(source['job_id'] == 'ctf-wj-original-v1' and source['checkpoint'] == str(CHECKPOINT)
            and source['training_config'] == str(TRAINING_CONFIG), 'only completed old-pool tinyfit20 A is permitted')
    require(result['status'] == 'completed' and result['request'] == source
            and result['lineage']['checkpoint'] == str(CHECKPOINT)
            and result['lineage']['checkpoint_sha256'] == source['checkpoint_sha256'],
            'completed A checkpoint provenance differs')
    checkpoint = result['lineage']['checkpoint_lineage']
    require(checkpoint['stage'] == tiny.STAGE and checkpoint['step'] == 20
            and checkpoint['parent_lineage']['parent_checkpoint'] == str(PARENT_CHECKPOINT),
            'A must be old tinyfit20, not the common parent or a new B endpoint')
    require(sha256(TRAINING_CONFIG) == source['training_config_sha256'], 'A training config changed')
    return source, result


def validate_request(request):
    case = (request.get('target'), request.get('condition'))
    require(request.get('mode') == MODE and request.get('experiment_revision') == REVISION
            and request.get('source_revision') == SOURCE_REVISION, 'fixed A-only paired-turn evaluation required')
    require(case in CASES and request.get('job_id') == CASES[case], 'only four fixed A cases permitted')
    expected = dict(arm='A_old_pool', model_arm='frozen', source_evaluation=str(SOURCE_EVAL),
                    source_result=str(SOURCE_RESULT), checkpoint=str(CHECKPOINT), training_config=str(TRAINING_CONFIG),
                    checkpoint_step=20, seed=42, steps=40, cfg_scale=5, future_frames=96,
                    solver='flow_euler', fastpath=False,
                    cache_index=str(ROOT / 'data/features/train.features.jsonl'),
                    cache_receipt=str(ROOT / 'data/features/train.features.jsonl.receipt.json'))
    require(all(request.get(key) == value for key, value in expected.items()), 'fixed A source/sampling/cache contract changed')
    require(type(request.get('max_seconds')) is int and 60 <= request['max_seconds'] <= 1200,
            'finite <=1200-second job required')
    for name in ('source_evaluation_sha256', 'source_result_sha256', 'checkpoint_sha256',
                 'training_config_sha256', 'cache_index_sha256', 'cache_receipt_sha256', 'cache_binding_sha256'):
        require(isinstance(request.get(name), str) and re.fullmatch('[0-9a-f]{64}', request[name]), 'source digest required')
    require(not any(name in request for name in ('action_scale', 'ratio_cap', 'gate_scale', 'lora_strength',
            'initial_latent', 'future_clean_latents', 'prompt', 'actions', 'window_index', 'episode_id', 'source_start')),
            'no condition, scale, future latent, or offset overrides accepted')


def make_request(*, source_revision, target, condition, max_seconds=1200, job_id=None):
    require((target, condition) in CASES, 'fixed J/L original/swap case required')
    require(job_id is None or job_id == CASES[target, condition], 'fixed A job identity differs')
    binding = cache.validate_completed_cache()
    source, _ = source_evaluation()
    request = dict(mode=MODE, experiment_revision=REVISION, source_revision=source_revision,
                   job_id=CASES[target, condition], target=target, condition=condition,
                   arm='A_old_pool', model_arm='frozen', source_evaluation=str(SOURCE_EVAL), source_result=str(SOURCE_RESULT),
                   source_evaluation_sha256=sha256(SOURCE_EVAL), source_result_sha256=sha256(SOURCE_RESULT),
                   checkpoint=str(CHECKPOINT), checkpoint_sha256=source['checkpoint_sha256'],
                   training_config=str(TRAINING_CONFIG), training_config_sha256=source['training_config_sha256'],
                   checkpoint_step=20, seed=42, steps=40, cfg_scale=5, future_frames=96,
                   solver='flow_euler', fastpath=False, max_seconds=max_seconds,
                   cache_binding_sha256=binding_digest(binding),
                   **{key: binding[key] for key in ('cache_index', 'cache_index_sha256', 'cache_receipt', 'cache_receipt_sha256')})
    validate_request(request)
    return request


def action_condition(original, condition):
    require(condition in ('original', 'jl_swap'), 'only recorded or J/L swap actions')
    actions = original.clone()
    if condition == 'jl_swap':
        actions[:, 5], actions[:, 7] = original[:, 7], original[:, 5]
    return actions


def extract_condition(payload, row):
    require(row['episode_id'] == EPISODE and row['source_start'] in STARTS.values()
            and row['window_index'] == (0 if row['source_start'] == 0 else 1), 'new shard local offset mapping differs')
    # Existing implementation performs [window_index, :1] BEFORE clone/cast.
    return matched._extract_condition(payload, row)


def exposure_metadata(config, row, checkpoint_lineage):
    """Map old *source starts*, not new local shard indices; no tensor access."""
    from training.data.action_dataset import cache_index_path, _read_jsonl
    manifest_path = Path(config.data.manifest_path)
    manifest = _read_jsonl(manifest_path)
    sources = [value for value in manifest if value['episode_id'] == row['episode_id'] and value['split'] == 'train']
    require(len(sources) == 1, 'candidate must belong to the original train manifest')
    source = sources[0]
    require(all(source[key] == row[key] for key in ('video_path', 'annotations_path')), 'source episode identity changed')
    starts = source['selected_source_starts']
    require(len(starts) == 4 and len(set(starts)) == 4, 'old endpoint must retain its actual four cached offsets')
    index = cache_index_path(manifest_path)
    episodes = [value for value in _read_jsonl(index) if value['episode_id'] == row['episode_id']]
    require(len(episodes) == 1 and episodes[0]['split'] == 'train' and episodes[0]['num_windows'] == 4,
            'old feature index episode binding differs')
    old_receipt = contained(index.parent / episodes[0]['episode_receipt']['path'], index.parent)
    require(read(old_receipt)['cache_binding']['selected_source_starts'] == starts, 'old cache source-start mapping differs')
    all_windows = []
    for window_index, source_start in enumerate(starts):
        value = matched.training_exposure(config, dict(episode_id=row['episode_id'], window_index=window_index))
        all_windows.append(dict(old_window_index=window_index, source_start=source_start,
                                parent_absolute_indices=value['parent_absolute_indices'],
                                paired_stage_absolute_indices=value['paired_stage_absolute_indices']))
    exact = [value for value in all_windows if value['source_start'] == row['source_start']]
    parent_indices = sorted(i for value in exact for i in value['parent_absolute_indices'])
    pair_indices = sorted(i for value in exact for i in value['paired_stage_absolute_indices'])
    pool = checkpoint_lineage['sampling_contract']['pool']
    require(pool['distinct_windows'] == 2 and pool['each_new_exposures'] == 80
            and pool['total_new_exposures'] == 160 and len(pool['windows']) == 2, 'A tinyfit pool/exposure differs')
    tiny_indices = [i for i in range(160) if
                    (pool['windows'][i % 2]['episode_id'], pool['windows'][i % 2]['source_start'])
                    == (row['episode_id'], row['source_start'])]
    seen = bool(parent_indices or pair_indices or tiny_indices)
    return dict(original_feature_index=str(index), original_selected_source_starts=starts,
                original_same_episode_window_exposures=all_windows,
                exact_source_start=row['source_start'], exact_parent_absolute_indices=parent_indices,
                exact_paired_stage_absolute_indices=pair_indices, exact_A_tinyfit_new_stage_indices=tiny_indices,
                actually_seen_by_A_endpoint=seen,
                same_episode_seen_by_parent_or_paired=any(value['parent_absolute_indices'] or value['paired_stage_absolute_indices']
                                                        for value in all_windows),
                new_cache_local_window_index_is_not_old_window_index=True,
                independent_dev_evaluation=False, metadata_only=True, future_latents_read=False)


def load_inputs(request):
    import numpy as np
    import torch
    import train_controlled_tinyfit as entry
    validate_request(request)
    require(sha256(SOURCE_EVAL) == request['source_evaluation_sha256']
            and sha256(SOURCE_RESULT) == request['source_result_sha256'], 'A source records changed')
    source, before = source_evaluation()
    require(source['checkpoint_sha256'] == request['checkpoint_sha256']
            and source['training_config_sha256'] == request['training_config_sha256'], 'A model bindings changed')
    binding = cache.validate_completed_cache()
    require(binding_digest(binding) == request['cache_binding_sha256']
            and all(binding[key] == request[key] for key in ('cache_index', 'cache_index_sha256', 'cache_receipt', 'cache_receipt_sha256')),
            'new exact-offset cache changed')
    row = binding['windows'][request['target']]
    require(row['source_start'] == STARTS[request['target']] and row['annotation_cache_actions_equal'] is True,
            'fixed new direction/offset differs')
    require(matched._stat(row['shard_path']) == row['shard_stat'], 'verified new shard stat changed')
    payload = torch.load(row['shard_path'], map_location='cpu', weights_only=True, mmap=True)
    first, prompt, original = extract_condition(payload, row)
    del payload
    require(matched.tensor_sha(first) == row['first_latent_sha256_float32']
            and matched.tensor_sha(original) == row['original_action_sha256_float32'],
            'cached first latent or original actions changed')
    require(matched.tensor_sha(prompt) == binding['prompt_sha256_float32'], 'reused static scene text changed')
    actions = action_condition(original, request['condition'])
    initial_path = contained(row['initial_path'], ROOT)
    require(sha256(initial_path) == row['initial_sha256'], 'cached display-only first RGB changed')
    initial = np.load(initial_path, allow_pickle=False)
    require(initial.shape == (480, 832, 3) and initial.dtype == np.uint8, 'display first RGB must be unannotated uint8')
    config = entry.load_config(TRAINING_CONFIG)  # No checkpoint or model is loaded during CPU preparation.
    parent_source = matched._source_evaluation()
    require(parent_source['checkpoint'] == str(PARENT_CHECKPOINT), 'fixed common parent changed')
    require(sha256(parent_source['negative_cache']) == parent_source['negative_cache_sha256']
            and sha256(Path(parent_source['negative_cache']).with_suffix('.json')) == parent_source['negative_receipt_sha256'],
            'negative text source changed')
    negative, negative_source = matched.load_negative(parent_source['negative_cache'], config.model.base_model_path)
    exposure = exposure_metadata(config, row, before['lineage']['checkpoint_lineage'])
    lineage = dict(arm='A_old_pool', checkpoint=str(CHECKPOINT), checkpoint_sha256=request['checkpoint_sha256'],
                   model_arm='frozen', common_parent_checkpoint=str(PARENT_CHECKPOINT),
                   source_evaluation=str(SOURCE_EVAL), source_evaluation_sha256=request['source_evaluation_sha256'],
                   source_result_sha256=request['source_result_sha256'], selection=row,
                   cache_index=binding['cache_index'], cache_index_sha256=binding['cache_index_sha256'],
                   cache_receipt=binding['cache_receipt'], cache_receipt_sha256=binding['cache_receipt_sha256'],
                   cache_binding_sha256=request['cache_binding_sha256'],
                   conditioning='new_exact_offset_cached_z0_and_reused_static_text_only',
                   future_clean_latents_read=False, future_clean_latents_supplied=False,
                   original_rgb_frames_decoded_by_eval=0, display_rgb_reused_from_cache=True, vae_encoding_used=False,
                   first_latent_sha256_float32=matched.tensor_sha(first), prompt_sha256_float32=matched.tensor_sha(prompt),
                   prompt_text=binding['prompt_text'], negative_source=negative_source,
                   original_action_sha256_float32=matched.tensor_sha(original), actual_action_sha256_float32=matched.tensor_sha(actions),
                   original_actions=original.tolist(), actual_actions=actions.tolist(), actual_action_condition=request['condition'],
                   training_manifest_window=True, independent_dev_evaluation=False, generalization_verified=False,
                   training_exposure=exposure, actually_seen_by_endpoint=exposure['actually_seen_by_A_endpoint'],
                   teacher_ready=False, reference_memory_implemented=False, automatic_B_training=False,
                   method_scope='same_training_episode_new_offset_A_qualification_not_independent_holdout')
    return first, prompt, actions.numpy(), initial, negative, config, lineage, source


def prepare(*, output, **options):
    output = contained(output, ROOT)
    request = make_request(**options)
    require(not output.exists() and output.name == request['job_id'], 'new matching A output directory required')
    values = load_inputs(request)
    lineage = values[-2]
    del values
    gc.collect()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'request.json', request)
    write_json(output / 'preparation.json', dict(status='cpu_prepared', request=request, lineage=lineage,
               gpu_launched=False, weights_loaded=False, video_generated=False, automatic_B_training=False))
    return request


def run(request, directory, lease):
    import numpy as np
    import torch
    import train_controlled_tinyfit as entry
    from PIL import Image
    from training.demo.backend import validate_active_authority
    from training.eval.rollout15s import ffmpeg_writer_factory
    validate_request(request)
    directory = contained(directory, ROOT)
    require(directory.name == request['job_id'], 'A job directory differs')
    target = directory / 'matched-evaluation'
    target.mkdir(exist_ok=False)
    began = time.monotonic()
    runtime = engine = None
    report = dict(status='running', mode=MODE, revision=REVISION, request=request, arm='A_old_pool',
                  future_clean_latents_supplied=False, independent_dev_evaluation=False, generalization_verified=False,
                  quality_improvement_verified=False, action_correctness_verified=False, realtime=False,
                  automatic_B_training=False)

    def check():
        require(time.monotonic() - began < request['max_seconds'] - 20, 'finite A evaluation deadline reached')
        require(matched.directory_bytes(directory) < matched.JOB_BYTES_LIMIT - 100_000, 'A job20MB output limit reached')
        validate_active_authority(lease)

    try:
        check()
        require(os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'offline assets required')
        first, prompt, actions, initial, negative, config, lineage, source = load_inputs(request)
        state, model_config, checkpoint_lineage = tiny._load_checkpoint(source, str(PARENT_CHECKPOINT))
        require(model_config.to_dict() == entry.model_config(config).to_dict(), 'A model config differs')
        before = read(SOURCE_RESULT)
        require(checkpoint_lineage['sampling_contract'] == before['lineage']['checkpoint_lineage']['sampling_contract'],
                'CPU exposure record differs from actual A checkpoint')
        lineage['checkpoint_lineage'] = checkpoint_lineage
        report['lineage'] = lineage
        Image.fromarray(initial).save(target / 'initial.png')
        np.save(target / 'actions.npy', actions, allow_pickle=False)
        write_json(target / 'inputs.json', dict(request=request, lineage=lineage, actual_actions=actions.tolist()))
        check()
        tick = time.monotonic()
        runtime = paired.ControlledPairRuntime(state, model_config, negative, arm='frozen')
        state.clear()
        gc.collect()
        report['load_seconds'] = time.monotonic() - tick
        engine = matched.CachedInitialEngine({}, runtime.config, residency='phase', fastpath=False, _runtime=runtime,
            cached_first=first, _writer_factory=lambda *args: matched.BoundedPreservingWriter(ffmpeg_writer_factory(*args)))
        engine.load_seconds = report['load_seconds']
        generation, _ = engine.run_segment(initial=initial, prompt=prompt, actions=actions, seed=42,
                                           output_path=target / 'raw.mp4', steps=40)
        stages = generation['stage_totals_seconds']
        require(generation['frames_written'] == 97 and runtime.forward_calls == 80
                and generation['first_latent_sha256_float32'] == lineage['first_latent_sha256_float32']
                and 'cached_z0_condition' in stages and 'encode_initial' not in stages and 'encode_endpoint' not in stages,
                'must use only cached z0 and97 frames/40 CFG pairs without VAE encoding')
        generation.update(initial_condition='new_exact_offset_cached_z0;RGB0_display_only',
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
