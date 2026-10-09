"""Actual Action-teacher UI inference: chunked by default, opt-in joint61.

This is a newly seeded browser request, not a claim of frozen-noise regression,
trained native15s context, reliable action control, or real-time generation. The initial
RGB is encoded now; static text comes from its exact bound precomputed embedding.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from training.demo.contracts import ACTION_METHOD, ACTION_JOINT_METHOD, ACTION_WINDOW6_METHOD, ACTION_STAGE, action_contract, require, sha256

WIDTH, HEIGHT = 832, 480
VAE_SHA256 = '20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36'
GT_PREFIX_INIT_MODE = 'length_gt_prefix_mix_transition_weights_only'
GT_PREFIX_PARENT_SHA = '9bc9f57cce1095cbbc472307bc3963da2e6153e9f0f19080b542b6bf118dfaee'
GT_PREFIX_PARENT_SOURCE = Path('/DATA_DISK/InteractWorld/creator-runtime/jobs/9be8bfe3ef4a499d9e9b1847a80af7fa')


def canonical(value):
    return json.dumps(value, sort_keys=True)


def compare_saved_config(saved, original, step):
    require(isinstance(saved, dict), 'Action checkpoint serialized config missing')
    previous, expected = copy.deepcopy(saved), copy.deepcopy(original)
    saved_max = previous['training'].pop('max_steps')
    original_max = expected['training'].pop('max_steps')
    require(type(saved_max) is int and saved_max >= step and saved_max >= original_max,
            'invalid same-run extended max_steps')
    require(canonical(previous) == canonical(expected), 'Action checkpoint changed fields beyond max_steps')


def tensor_sha(tensor):
    import torch
    return hashlib.sha256(tensor.detach().to(device='cpu', dtype=torch.float32).contiguous().numpy().tobytes()).hexdigest()


def validate_action_config(raw):
    adapter, expected_geometry = action_contract(raw.get('method'))
    require(raw.get('adapter_factory') == adapter,
            'expected the explicit Action UI backend/method')
    require(raw.get('sampler') == dict(solver='flow_euler', steps=40, shift=5.0, cfg='none'),
            'Action UI sampler differs from40-step flow Euler')
    require(raw['lineage'].get('expected_stage') == ACTION_STAGE, 'not an Action-teacher checkpoint')
    geometry = raw['geometry']
    require(all(geometry.get(key) == value for key, value in expected_geometry.items()),
            'Action UI geometry does not match the selected inference method')
    if raw['method'] != ACTION_METHOD:
        require(set(geometry) == set(expected_geometry), 'selected Action geometry must not contain other method fields')


def validate_gt_prefix_mix_contract(config, initialization, *, step):
    """Narrow completed-step20 GT-only candidate contract, not a quality gate."""
    return _validate_gt_prefix_mix_contract(config, initialization, step=step, fp32_lora=False)


def _validate_gt_prefix_mix_contract(config, initialization, *, step, fp32_lora, low_sigma=False):
    from training.data.gt_prefix_mix import FACTORY, prefix_contract
    require(isinstance(config, dict) and isinstance(initialization, dict), 'GT candidate config/initialization missing')
    expected_mode = 'length_gt_prefix_low_sigma_transition_weights_only' if low_sigma else GT_PREFIX_INIT_MODE
    require(type(step) is int and step == 20 and initialization.get('mode') == expected_mode,
            'GT candidate must be the completed step20 explicit prefix transfer')
    require(initialization.get('sha256') == GT_PREFIX_PARENT_SHA and initialization.get('source_step') == 1040
            and initialization.get('source_stage') == ACTION_STAGE
            and initialization.get('optimizer_restored') is False and initialization.get('rng_restored') is False
            and initialization.get('start_step') == 0, 'GT candidate must originate from fresh R005 step1040 transfer')
    require(initialization.get('prefix_contract') == prefix_contract(), 'GT prefix contract differs')
    data, training, optimizer = config['data'], config['training'], config['optimizer']
    require(type(low_sigma) is bool and type(data.get('low_sigma_stratification', False)) is bool
            and data.get('low_sigma_stratification', False) == low_sigma,
            'low-sigma candidate requires its isolated explicit evaluation contract')
    if low_sigma:
        from training.data.low_sigma import low_sigma_contract
        require(fp32_lora and initialization.get('low_sigma_contract') == low_sigma_contract(),
                'low-sigma initialization contract differs')
    else:
        require('low_sigma_contract' not in initialization, 'legacy GT candidate cannot relabel low-sigma training')
    if fp32_lora:
        from train_action_teacher import fp32_lora_precision_transition
        require(training.get('lora_parameter_dtype') == 'float32'
                and initialization.get('precision_transition') == fp32_lora_precision_transition(),
                'explicit FP32 LoRA precision transition differs')
    else:
        require(training.get('lora_parameter_dtype', 'bfloat16') == 'bfloat16'
                and 'precision_transition' not in initialization,
                'FP32 LoRA candidate is not enabled in the original GT inference contract')
    require(data['data_factory'] == FACTORY and data['num_frames'] == 97 and data['prompt_cache_path'] is None
            and training['max_steps'] == 20 and training['seed'] == 42 and training['precision'] == 'bf16'
            and training['gradient_accumulation_steps'] == 8, 'GT candidate training contract differs')
    source = initialization.get('source_config', {})
    source_model = copy.deepcopy(source.get('model', {}))
    target_model = config['model']
    if source_model.get('base_model_path') != target_model['base_model_path']:
        relocation = initialization.get('base_model_relocation', {})
        require(relocation.get('acknowledged') is True and relocation.get('changed') is True
                and relocation.get('source') == source_model.get('base_model_path')
                and relocation.get('target') == target_model['base_model_path'], 'GT candidate base relocation not acknowledged')
        source_model['base_model_path'] = target_model['base_model_path']
    require(source_model == target_model and target_model['action_scale'] == .03,
            'GT candidate changed the parent model contract')
    source_optimizer = source.get('optimizer', {})
    expected_optimizer = dict(source_optimizer, adapter_lr=0.0, lora_lr=2e-6)
    require(optimizer == expected_optimizer and type(optimizer['adapter_lr']) is not bool
            and type(optimizer['lora_lr']) is not bool, 'GT candidate requires adapter_lr0 and LoRA2e-6 only')
    expected_override = dict(kind='gt_prefix_mix_fixed_adapter_lora2e6_v1',
        changed_fields=['adapter_lr', 'lora_lr'], source_adapter_lr=source_optimizer.get('adapter_lr'),
        target_adapter_lr=0.0, source_lora_lr=source_optimizer.get('lora_lr'), target_lora_lr=2e-6,
        other_optimizer_fields_unchanged=True, requires_grad_unchanged=True, complete_trainable_state_preserved=True)
    require(initialization.get('optimizer_transition') == expected_override, 'GT optimizer transition differs')
    length = initialization.get('length_transition', {})
    require(length.get('source_rgb_frames') == 49 and length.get('target_rgb_frames') == 97
            and length.get('source_latent_frames') == 13 and length.get('target_latent_frames') == 25
            and length.get('model_contract_unchanged') is True, 'GT length transition differs')
    return dict(policy=prefix_contract(), completed_step=20, parent_checkpoint_sha256=GT_PREFIX_PARENT_SHA,
                adapter_lr=0.0, lora_lr=2e-6, quality_improvement_verified=False)


def validate_fp32_lora_eval_contract(payload, *, low_sigma=False):
    """Explicit completed FP32 step20 candidate; legacy GT validation stays closed."""
    import torch
    require(isinstance(payload, dict) and payload.get('format_version') == 1 and payload.get('stage') == ACTION_STAGE,
            'actual Action checkpoint payload required for FP32 inference')
    contract = _validate_gt_prefix_mix_contract(payload.get('config'), payload.get('initialization'),
                                               step=payload.get('step'), fp32_lora=True, low_sigma=low_sigma)
    require(payload.get('micro_batches_consumed') == 160, 'FP32 candidate requires20 steps/160microbatches')
    state = payload.get('trainable_model')
    require(isinstance(state, dict), 'actual FP32 candidate tensors missing')
    lora = {name: value for name, value in state.items() if name.endswith(('.lora_a.weight', '.lora_b.weight'))}
    adapter = {name: value for name, value in state.items() if 'act_control_adapter.' in name}
    require(len(lora) == 600 and len(adapter) == 6 and len(state) == 606
            and all(torch.is_tensor(value) and value.device.type == 'cpu' and value.dtype == torch.float32
                    for value in lora.values())
            and all(torch.is_tensor(value) and value.device.type == 'cpu' and value.dtype == torch.bfloat16
                    for value in adapter.values()), 'requires actual600 FP32 LoRA and6 BF16 adapter tensors only')
    contract.update(precision_transition=copy.deepcopy(payload['initialization']['precision_transition']),
        checkpoint_lora_tensors=600, checkpoint_lora_dtype='torch.float32', checkpoint_adapter_tensors=6,
        checkpoint_lora_elements=sum(value.numel() for value in lora.values()),
        checkpoint_adapter_dtype='torch.bfloat16', inference_parameter_dtype='torch.bfloat16',
        inference_cast='FP32 LoRA checkpoint loaded then cast to BF16 in-memory;checkpoint bytes unchanged',
        checkpoint_rewritten=False)
    if low_sigma:
        contract['low_sigma_contract'] = copy.deepcopy(payload['initialization']['low_sigma_contract'])
    return contract


def _eval_prompt_reference(spec, scene, config, hashes, initialization, digest):
    """Bind dev-only text separately from a train-only native97 feature cache."""
    reference = spec.get('eval_prompt_sidecar')
    require(config.data.num_frames == 97 and config.data.prompt_cache_path is None,
            'independent eval text is restricted to inline-text native97 training')
    require(isinstance(reference, dict) and set(reference) ==
            {'cache_path', 'cache_sha256', 'receipt_path', 'receipt_sha256'}, 'explicit dev text sidecar pins required')
    cache_path, receipt_path = Path(reference['cache_path']), Path(reference['receipt_path'])
    require(cache_path.is_absolute() and receipt_path.is_absolute()
            and receipt_path == cache_path.with_suffix(cache_path.suffix + '.receipt.json'), 'invalid dev text sidecar paths')
    require(digest(cache_path) == reference['cache_sha256'] and digest(receipt_path) == reference['receipt_sha256'],
            'dev text sidecar hash mismatch')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    require(receipt.get('schema_version') == 1 and receipt.get('kind') == 'window97_eval_text_reference_v1'
            and receipt.get('prompt_policy') == 'scene_static_only_v1'
            and receipt.get('scope') == 'dev_text_only_no_video_features', 'invalid independent dev text receipt')
    require(Path(receipt.get('cache_path', '')).resolve() == cache_path.resolve()
            and receipt.get('cache_sha256') == reference['cache_sha256'], 'dev text cache receipt binding differs')
    require(Path(receipt.get('training_manifest', '')).resolve() == Path(config.data.manifest_path).resolve()
            and receipt.get('training_manifest_sha256') == hashes['dataset_manifest'], 'dev text points to another training branch')
    allowed_modes = (('length_gt_prefix_low_sigma_transition_weights_only',) if spec.get('low_sigma_candidate', False) else
                     (GT_PREFIX_INIT_MODE,) if spec.get('gt_prefix_mix_candidate', False) else
                     ('length_transition_weights_only', 'length_resampled_transition_weights_only'))
    require(isinstance(initialization, dict) and initialization.get('mode') in allowed_modes,
            'dev text requires an explicit native97 parent transfer')
    parent_hashes = initialization.get('source_manifest_hashes', {})
    for field, key in (('source_manifest_sha256', 'dataset_manifest'),
                       ('source_prompt_cache_sha256', 'prompt_cache'),
                       ('source_prompt_receipt_sha256', 'prompt_cache_receipt'),
                       ('source_feature_index_sha256', 'feature_index'),
                       ('source_feature_receipt_sha256', 'feature_receipt')):
        require(receipt.get(field) and receipt[field] == parent_hashes.get(key),
                f'dev text does not belong to the original parent: {field}')
    for field in ('source_manifest', 'source_prompt_cache', 'source_prompt_receipt'):
        path = Path(receipt.get(field, ''))
        require(path.is_absolute() and digest(path) == receipt[field + '_sha256'], f'dev text source changed: {field}')
    source = json.loads(Path(receipt['source_prompt_receipt']).read_text(encoding='utf-8'))
    require(source.get('schema_version') == 1 and source.get('kind') == 'scene_static_prompt_cache'
            and source.get('prompt_policy') == 'scene_static_only_v1', 'invalid original static prompt receipt')
    for source_field, field in (('cache_sha256', 'source_prompt_cache_sha256'),
                               ('manifest_sha256', 'source_manifest_sha256'),
                               ('feature_index_sha256', 'source_feature_index_sha256'),
                               ('feature_receipt_sha256', 'source_feature_receipt_sha256')):
        require(source.get(source_field) == receipt[field], 'original static text binding differs')
    require(source.get('encoder') and source['encoder'] == receipt.get('encoder'), 'dev text encoder provenance changed')
    episodes = receipt.get('episodes')
    require(isinstance(episodes, dict) and episodes and scene['source_episode_id'] in episodes,
            'selected dev episode missing from independent text sidecar')
    for identity, binding in episodes.items():
        require(isinstance(binding, dict) and binding.get('split') == 'dev'
                and binding == source.get('episodes', {}).get(identity), 'dev text binding differs from original dev receipt')
        require(binding.get('prompt_sha256') == hashlib.sha256(binding.get('prompt', '').encode()).hexdigest(),
                'dev prompt text hash mismatch')
    train_rows = [json.loads(line) for line in Path(config.data.manifest_path).read_text().splitlines() if line.strip()]
    require(all(row.get('split') == 'train' and row.get('episode_id') not in episodes for row in train_rows),
            'independent dev episodes leaked into native97 training')
    require(episodes[scene['source_episode_id']]['prompt'] == scene['prompt'], 'selected dev text differs from displayed prompt')
    return cache_path, receipt, reference['cache_sha256']


def load_action_inputs(raw, scene, *, allow_fp32_lora_candidate=False, allow_low_sigma_candidate=False):
    """CPU-only exact checkpoint/artifact/text binding; no future video/shard read."""
    import torch
    from training.config import load_config
    from training.models.action_adapter import validate_action_scale
    from training.demo.relocation import ArtifactHashes, artifact_paths, relocate_path, relocated_manifest_hashes, validate_relocations
    validate_action_config(raw)
    spec = raw['lineage']
    require(type(allow_low_sigma_candidate) is bool and type(spec.get('low_sigma_candidate', False)) is bool
            and allow_low_sigma_candidate == spec.get('low_sigma_candidate', False)
            and (not allow_low_sigma_candidate or allow_fp32_lora_candidate),
            'low-sigma candidate requires the isolated explicit loader opt-in')
    require(type(allow_fp32_lora_candidate) is bool
            and type(spec.get('fp32_lora_candidate', False)) is bool
            and allow_fp32_lora_candidate == spec.get('fp32_lora_candidate', False),
            'FP32 candidate requires the isolated explicit loader opt-in')
    path = Path(spec['checkpoint_path'])
    verification_cache = spec.get('verification_cache')
    digest = (ArtifactHashes(verification_cache, protected_paths=[path, *spec['artifact_paths'].values(),
                  Path(spec['expected_base_model_path']) / 'Wan2.2_VAE.pth'])
              if verification_cache is not None else sha256)
    require(digest(path) == spec['checkpoint_sha256'].lower(), 'Action checkpoint SHA mismatch')
    config_path = Path(spec['artifact_paths']['training_config'])
    config = load_config(config_path)
    require(config.training.lora_parameter_dtype == ('float32' if allow_fp32_lora_candidate else 'bfloat16'),
            'FP32 LoRA training candidate is not enabled for video inference')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    require(payload.get('format_version') == 1 and payload.get('stage') == ACTION_STAGE
            and type(payload.get('step')) is int and payload['step'] > 0, 'not a completed self-trained Action checkpoint')
    compare_saved_config(payload.get('config'), config.to_dict(), payload['step'])
    candidate = spec.get('gt_prefix_mix_candidate', False)
    require(type(candidate) is bool, 'gt_prefix_mix_candidate must be an explicit boolean')
    fp32_contract = None
    if allow_fp32_lora_candidate:
        require(candidate and spec.get('eval_prompt_sidecar') is not None,
                'FP32 candidate requires explicit GT-prefix and dev-only text')
        fp32_contract = validate_fp32_lora_eval_contract(payload, low_sigma=allow_low_sigma_candidate)
    elif candidate:
        validate_gt_prefix_mix_contract(payload['config'], payload.get('initialization'), step=payload['step'])
        require(payload.get('micro_batches_consumed') == 160 and spec.get('eval_prompt_sidecar') is not None,
                'GT candidate must preserve completed20-step/160-microbatch state and dev text sidecar')
    else:
        require((payload.get('initialization') or {}).get('mode') != GT_PREFIX_INIT_MODE,
                'GT prefix checkpoint requires explicit gt_prefix_mix_candidate')
    # Compare original serialized fields before any in-memory relocation. The
    # map affects only file access, never checkpoint/YAML/receipt contents.
    relocations = validate_relocations(spec.get('path_relocations', {}))
    expected_paths = artifact_paths(config, config_path, relocations)
    separate_eval = spec.get('eval_prompt_sidecar') is not None
    require(config.data.prompt_cache_path is not None or separate_eval, 'Action UI requires the actual static prompt cache')
    require(not separate_eval or (config.data.num_frames == 97 and config.data.prompt_cache_path is None),
            'dev text sidecar must not replace an existing training prompt cache')
    require(set(expected_paths) == set(spec['artifact_paths']), 'Action artifact schema changed')
    require(all(Path(spec['artifact_paths'][key]).resolve() == value.resolve() for key, value in expected_paths.items()),
            'Action artifact paths differ from original training contract after explicit relocation')
    base_model_path = relocate_path(config.model.base_model_path, relocations)
    require(Path(spec['expected_base_model_path']).resolve() == base_model_path.resolve(), 'Action base-model pin changed')
    if relocations or verification_cache is not None:
        hashes = relocated_manifest_hashes(config, config_path, relocations,
                                          verifier=digest if verification_cache is not None else None)
    else:
        from train_action_teacher import _manifest_hashes
        hashes = _manifest_hashes(config, config_path)
    require(payload.get('manifest_hashes') == hashes, 'checkpoint data/config/static-prompt hashes changed')
    scale = validate_action_scale(config.model.action_scale)
    require(scale > 0, 'Action demo requires a nonzero actual conditioning scale')
    state = payload.get('trainable_model')
    require(isinstance(state, dict) and state and any('.lora_a.' in name for name in state),
            'Action checkpoint has no actual trained LoRA state')
    require(any('act_control_adapter.' in name for name in state), 'Action adapter weights missing')
    lineage = dict(path=str(path.resolve()), sha256=spec['checkpoint_sha256'], stage=ACTION_STAGE,
                   step=payload['step'], config=payload['config'], manifest_hashes=hashes,
                   initialization=payload.get('initialization'), action_scale=scale)
    if fp32_contract is not None:
        lineage['fp32_lora_candidate_contract'] = fp32_contract
    if relocations:
        lineage.update(path_relocations=relocations,
                       resolved_artifact_paths={key: str(value.resolve()) for key, value in expected_paths.items()},
                       resolved_base_model_path=str(base_model_path.resolve()))
    del payload
    gc.collect()
    if separate_eval:
        cache_path, receipt, prompt_cache_sha = _eval_prompt_reference(
            spec, scene, config, hashes, lineage['initialization'], digest)
    else:
        cache_path = expected_paths['prompt_cache']
        receipt = json.loads(cache_path.with_suffix('.pt.receipt.json').read_text(encoding='utf-8'))
        prompt_cache_sha = hashes['prompt_cache']
    binding = receipt.get('episodes', {}).get(scene['source_episode_id'])
    require(binding and binding.get('split') == 'dev' and binding.get('prompt') == scene['prompt'],
            'displayed Action prompt does not match the actual static embedding text/source')
    # _manifest_hashes already validated the full immutable cache/receipt. Only
    # inspect the selected tensor here; no all-episode numerical audit or T5 load.
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    require(cache.get('schema_version') == 1 and cache.get('kind') == 'scene_static_prompt_cache',
            'unexpected static prompt payload')
    if separate_eval:
        require(set(cache.get('prompt_embeds', {})) == set(receipt['episodes']), 'dev text payload/receipt episode IDs differ')
    prompt = cache['prompt_embeds'][scene['source_episode_id']]
    require(torch.is_tensor(prompt) and prompt.device.type == 'cpu' and prompt.is_floating_point()
            and prompt.ndim == 2 and 1 <= prompt.shape[0] <= 512 and prompt.shape[1] == 4096
            and bool(torch.isfinite(prompt).all()), 'selected static embedding is invalid')
    if separate_eval:
        require(receipt.get('selected_embedding_sha256_float32', {}).get(scene['source_episode_id']) == tensor_sha(prompt),
                'selected dev embedding differs from its copied original tensor receipt')
    prompt = prompt.detach().to(dtype=torch.bfloat16).clone()
    del cache
    gc.collect()
    lineage['prompt_condition'] = dict(policy='scene_static_only_v1', text=scene['prompt'],
                                      episode_id=scene['source_episode_id'], cache_sha256=prompt_cache_sha,
                                      embedding_sha256_float32=tensor_sha(prompt), shape=list(prompt.shape), t5_loaded=False)
    if separate_eval:
        lineage['prompt_condition'].update(storage='independent_dev_text_sidecar',
            training_prompt_storage='inline_train97_feature_shards', reference=copy.deepcopy(spec['eval_prompt_sidecar']),
            original_prompt_cache_sha256=receipt['source_prompt_cache_sha256'])
    vae_path = base_model_path / 'Wan2.2_VAE.pth'
    require(digest(vae_path) == VAE_SHA256, 'Wan VAE differs from the verified Action cache/inference VAE')
    lineage['vae'] = dict(path=str(vae_path), sha256=VAE_SHA256)
    runtime_model = copy.deepcopy(config.model)
    runtime_model.base_model_path = str(base_model_path)
    return state, runtime_model, prompt, lineage


def rollout_chunks(*, initial_latent, initial_rgb, prompt, actions, seed, generate, decode, encode, emit, window6=False):
    """CPU-testable scheduling; callbacks own real model/VAE computation."""
    import torch
    require(tuple(initial_latent.shape) == (1, 1, 48, 30, 52), 'initial VAE latent shape mismatch')
    require(actions.shape == (240, 8) and actions.dtype == np.float32, 'Action UI needs240x8 float32 actions')
    rng = torch.Generator(device='cpu').manual_seed(seed)
    future_counts = (96, 96, 48) if window6 else (48,) * 5
    mapped_noise = (joint_noise_from_chunks(torch.stack([
        torch.randn((1, 13, 48, 30, 52), generator=rng, dtype=torch.float32, device='cpu') for _ in range(5)]))
        if window6 else None)
    first = initial_latent.detach().cpu().to(torch.bfloat16).clone()
    rows, written, action_start = [], 0, 0
    for index, future_count in enumerate(future_counts):
        action_stop = action_start + future_count
        latent_frames = future_count // 4 + 1
        keys = torch.from_numpy(actions[action_start:action_stop].copy()).unsqueeze(0)
        noise = (torch.cat([mapped_noise[:, :1], mapped_noise[:, 1 + action_start // 4:1 + action_stop // 4]], dim=1)
                 if window6 else torch.randn((1, 13, 48, 30, 52), generator=rng, dtype=torch.float32, device='cpu'))
        row = dict(chunk=index, action_start=action_start, action_stop=action_stop,
                   latent_frames=latent_frames, emitted_frames=future_count + (1 if index == 0 else 0),
                   first_latent_sha256_float32=tensor_sha(first), noise_sha256_float32=tensor_sha(noise),
                   action_sha256_float32=tensor_sha(keys),
                   first_latent_source='submitted_RGB_encoded' if index == 0 else 'previous_generated_float_RGB_reencoded')
        latent = generate(first, prompt, keys, noise, index)
        require(tuple(latent.shape) == (1, latent_frames, 48, 30, 52) and bool(torch.isfinite(latent).all()), 'generated chunk latent frame count or values changed')
        require(torch.equal(latent[:, :1].cpu(), first.to(latent.dtype).cpu()), 'sampler altered the clean first latent')
        pixels = decode(latent, index)
        require(tuple(pixels.shape) == (1, future_count + 1, 3, HEIGHT, WIDTH) and bool(torch.isfinite(pixels).all())
                and pixels.min() >= -1 and pixels.max() <= 1, 'VAE window frame count or finite float RGB values in[-1,1] changed')
        # Frame0 is the actual submitted condition. Every subsequent frame is
        # genuinely generated; decoded conditioning frames are not duplicated.
        if index == 0:
            emit(initial_rgb[None], 0)
            written = 1
        rgb = pixels[0, 1:].permute(0, 2, 3, 1).add(1).mul(127.5).round().clamp(0, 255).byte().numpy()
        emit(rgb, written)
        written += future_count
        endpoint = pixels[:, -1:].detach().permute(0, 2, 1, 3, 4).contiguous()
        row.update(written_frames=written, endpoint_float_rgb_sha256=tensor_sha(endpoint))
        if index + 1 < len(future_counts):
            first = encode(endpoint, index).detach().cpu().to(torch.bfloat16)
            require(tuple(first.shape) == (1, 1, 48, 30, 52), 'endpoint was not independently reencoded')
        rows.append(row)
        action_start = action_stop
        print(f'Action generation: chunk {index + 1}/{len(future_counts)} complete; {written}/241 frames written', flush=True)
    require(written == 241, 'Action UI generated an incorrect frame count')
    return dict(method=ACTION_WINDOW6_METHOD if window6 else ACTION_METHOD, chunks=rows, frames_written=written,
                noise_policy=('one_CPU_generator_seeded_by_request;five_consecutive_FP32_draws;future_slots_repartitioned24_24_12'
                              if window6 else 'one_CPU_generator_seeded_by_request;five_consecutive_FP32_noise_draws'),
                frozen_noise_regression=False, initial_output='submitted_RGB_at_frame0',
                continuation='previous_generated_float_RGB_before_uint8_reencoded',
                native_long_context=False, realtime=False)


def euler_rollout(model, *, first, noise, conditions, sigmas, fp32_accumulation=False):
    """Flow Euler with opt-in FP32 accumulation and unchanged model conditioning.

    The default preserves the original solver arithmetic. With the explicit
    flag, the supplied (already quantized) noise/first-frame values are promoted
    to FP32 without a new draw. Only sigma differences and state updates use
    FP32; model video inputs retain ``noise.dtype``. Timesteps are still computed
    from the ORIGINAL supplied sigmas, exactly as in the default branch (even
    BF16 multiply rounding). Conditions are passed through unchanged. The opt-in
    result stays FP32; callers retain their existing decoder-input cast.

    For a single-variable comparison, pass the same BF16 noise AND BF16 sigma
    nodes to both modes; this flag does not construct a different schedule.
    """
    import torch
    require(type(fp32_accumulation) is bool, 'fp32_accumulation must be an explicit boolean')
    current = noise.clone()
    current[:, :1] = first
    if fp32_accumulation:
        # Promote after the original assignment so the initial condition has
        # exactly the same numerical values, including its original dtype cast.
        current = current.float()
        fixed_first = current[:, :1].clone()
    for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        timestep = (sigma * 1000).expand(current.shape[:2]).clone()
        timestep[:, 0] = 0
        model_input = current.to(dtype=noise.dtype) if fp32_accumulation else current
        velocity, _ = model(model_input, conditional_dict=conditions, timestep=timestep,
                            replace_first_timestep_and_noise_latents=True)
        if fp32_accumulation:
            current = current + (next_sigma.float() - sigma.float()) * velocity.float()
            current[:, :1] = fixed_first
        else:
            current = current + (next_sigma - sigma) * velocity
            current[:, :1] = first
        require(bool(torch.isfinite(current).all()), 'nonfinite Action Euler state')
    return current


def joint_noise_from_chunks(noises):
    """Map five independent13-slot draws to one61-slot state without redrawing.

    All five old first slots are unused conditioning slots. Preserve the first
    placeholder (Euler overwrites it) and all five ordered12-slot futures.
    """
    import torch
    require(torch.is_tensor(noises) and noises.ndim == 6 and tuple(noises.shape[:4]) == (5, 1, 13, 48)
            and noises.dtype == torch.float32 and noises.device.type == 'cpu'
            and bool(torch.isfinite(noises).all()), 'joint noise requires five finite CPU FP32 13-slot draws')
    return torch.cat([noises[0, :, :1], *[noises[index, :, 1:] for index in range(5)]], dim=1)


def rollout_joint(*, initial_latent, initial_rgb, prompt, actions, seed, generate, decode, emit):
    """One61-latent solve, then continuous cached VAE decode; no RGB feedback.

    ``decode`` resets its VAE cache only at index0 and retains it thereafter.
    Its first decoded frame warms the cache but is replaced by submitted RGB,
    matching the existing browser protocol, not the frozen regression protocol.
    """
    import torch
    require(tuple(initial_latent.shape) == (1, 1, 48, 30, 52), 'initial VAE latent shape mismatch')
    require(actions.shape == (240, 8) and actions.dtype == np.float32
            and np.isfinite(actions).all() and np.isin(actions, [0, 1]).all(),
            'joint Action UI needs240x8 finite binary float32 actions')
    rng = torch.Generator(device='cpu').manual_seed(seed)
    noises = torch.stack([torch.randn((1, 13, 48, 30, 52), generator=rng, dtype=torch.float32, device='cpu')
                          for _ in range(5)])
    noise = joint_noise_from_chunks(noises)
    first = initial_latent.detach().cpu().to(torch.bfloat16).clone()
    keys = torch.from_numpy(actions.copy()).unsqueeze(0)
    input_hashes = dict(first_latent_sha256_float32=tensor_sha(first), noise_sha256_float32=tensor_sha(noise),
                        action_sha256_float32=tensor_sha(keys),
                        source_noise_sha256_float32=[tensor_sha(value) for value in noises])
    del noises
    latent = generate(first, prompt, keys, noise, 0)
    require(tuple(latent.shape) == (1, 61, 48, 30, 52) and bool(torch.isfinite(latent).all()),
            'joint generation must return61 finite latent frames')
    require(torch.equal(latent[:, :1].cpu(), first.to(latent.dtype).cpu()), 'sampler altered the clean first latent')
    written, rows = 0, []
    slices = [(0, 1, 1), *[(1 + index * 3, 4 + index * 3, 12) for index in range(20)]]
    for index, (start, stop, expected_frames) in enumerate(slices):
        pixels = decode(latent[:, start:stop], index)
        require(tuple(pixels.shape) == (1, expected_frames, 3, HEIGHT, WIDTH)
                and bool(torch.isfinite(pixels).all()) and pixels.min() >= -1 and pixels.max() <= 1,
                'continuous VAE decode has invalid frame count/geometry/pixels')
        rgb = (initial_rgb[None] if index == 0 else
               pixels[0].permute(0, 2, 3, 1).add(1).mul(127.5).round().clamp(0, 255).byte().numpy())
        emit(rgb, written)
        written += expected_frames
        rows.append(dict(latent_start=start, latent_stop=stop, frames=expected_frames,
                         total_frames=written, use_cache=True, previous_generated_cache_retained=index != 0))
    require(written == 241, 'joint Action UI generated an incorrect frame count')
    print('Action joint generation: one61-latent solve and continuous decode complete;241/241 frames written', flush=True)
    return dict(method=ACTION_JOINT_METHOD, input_hashes=input_hashes, decode_stream=rows,
                frames_written=written, joint_sequence=True, latent_frames=61, action_rows=240,
                initial_latent_unchanged=True, is_causal=False,
                noise_policy='one_CPU_generator_seeded_by_request;five_consecutive_FP32_draws;ordered12_future_slots_each',
                frozen_noise_regression=False, initial_output='submitted_RGB_at_frame0',
                continuation='none;one_joint61_solve', rgb_endpoint_reencoding=False,
                native_long_context=False, quality_evaluation='not_run', realtime=False)


def generate_action_video(**kwargs):
    """Default legacy5x49 generation, unchanged by joint-mode opt-in."""
    return _generate_action_video(**kwargs, joint=False)


def generate_action_joint_video(**kwargs):
    """Opt-in one61-latent async generation, not real-time or quality certification."""
    return _generate_action_video(**kwargs, joint=True)


def generate_action_window6_video(**kwargs):
    """Opt-in6+6+3s windows; unchanged floating RGB feedback and Euler sampler."""
    return _generate_action_video(**kwargs, joint=False, window6=True)


def _generate_action_video(*, state, model_config, prompt, initial, actions, seed, output_path, joint, window6=False):
    """GPU-only real model path; caller must already pass standing/GPU/budget gates."""
    import torch
    import utils.wan_wrapper as wrappers
    from training.models.lora import configure_action_teacher, load_trainable_state_dict
    from training.models.action_adapter import build_action_context
    from training.longforcing_lite import euler_sigmas
    from training.eval.rollout15s import ffmpeg_writer_factory
    require(initial.shape == (HEIGHT, WIDTH, 3) and initial.dtype == np.uint8, 'actual initial RGB is invalid')
    print('Action loading: base model load starting', flush=True)
    model = wrappers.WanDiffusionWrapper(model_name=model_config.base_model_path, model_type='ci2v',
                                         is_causal=False, timestep_shift=5.0, downscale_factor_control_adapter=16)
    print('Action loading: base model load complete', flush=True)
    if joint or window6:
        require(model.is_causal is False and model.uniform_timestep is True, 'Action windows require the noncausal wrapper')
    summary = configure_action_teacher(model.model, rank=model_config.lora_rank,
                                       alpha=model_config.lora_alpha, dropout=model_config.lora_dropout)
    require({name for name, value in model.named_parameters() if value.requires_grad} == set(state), 'Action trainable tensor names mismatch')
    load_trainable_state_dict(model, state)
    state.clear()
    print('Action loading: trained LoRA and action adapter restored', flush=True)
    model.requires_grad_(False).eval().to(dtype=torch.bfloat16)
    print('Action loading: model BF16 conversion complete', flush=True)
    print('Action loading: VAE load starting', flush=True)
    vae = wrappers.WanVAEWrapper(pretrained_path=str(Path(model_config.base_model_path) / 'Wan2.2_VAE.pth'))
    vae.requires_grad_(False).eval().to(dtype=torch.bfloat16)
    print('Action loading: VAE load and BF16 conversion complete', flush=True)
    peak = 0

    def capture_peak():
        nonlocal peak
        peak = max(peak, torch.cuda.max_memory_allocated())

    def vae_on_gpu():
        model.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
        vae.to('cuda')

    def encode(pixel, index):
        print(f'Action encoding: {"initial frame" if index < 0 else "window " + str(index + 1) + " endpoint"} starting', flush=True)
        vae_on_gpu()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            value = vae.encode_to_latent(pixel.to(device='cuda', dtype=torch.bfloat16)).detach().cpu()
        capture_peak()
        print('Action encoding: complete', flush=True)
        return value

    def generate(first, text, keys, noise, index):
        print(f'Action generation: window {index + 1} starting', flush=True)
        vae.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
        model.to('cuda')
        condition = first.to(device='cuda', dtype=torch.bfloat16)
        conditions = dict(prompt_embeds=[text.to(device='cuda', dtype=torch.bfloat16)],
                          act_context_scale=model_config.action_scale,
                          act_context=build_action_context(keys, height=HEIGHT, width=WIDTH, device='cuda', dtype=torch.bfloat16))
        sigmas = euler_sigmas(40, shift=5.0, device=torch.device('cuda'), dtype=torch.bfloat16)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            current = euler_rollout(model, first=condition, noise=noise.to(device='cuda', dtype=torch.bfloat16),
                                    conditions=conditions, sigmas=sigmas)
        capture_peak()
        result = current.detach().cpu()
        print(f'Action generation: window {index + 1} complete', flush=True)
        return result

    def decode(latent, index):
        print(f'Action decoding: {"slice" if joint else "window"} {index + 1} starting', flush=True)
        vae_on_gpu()
        if joint and index == 0:
            vae.model.clear_cache()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            value = vae.decode_to_pixel(latent.to(device='cuda', dtype=torch.bfloat16), use_cache=joint, return_in_cpu=True)
        capture_peak()
        print(f'Action decoding: {"slice" if joint else "window"} {index + 1} complete', flush=True)
        return value

    writer = ffmpeg_writer_factory(output_path, WIDTH, HEIGHT, 16)
    began = time.monotonic()
    try:
        pixel = torch.from_numpy(initial.copy()).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).float().div(127.5).sub(1)
        first = encode(pixel, -1)
        options = dict(initial_latent=first, initial_rgb=initial, prompt=prompt, actions=actions,
                       seed=seed, generate=generate, decode=decode, emit=lambda frames, offset: writer.write(frames))
        report = rollout_joint(**options) if joint else rollout_chunks(**options, encode=encode, window6=window6)
        writer.close()
    except BaseException:
        writer.abort()
        raise
    finally:
        if joint:
            vae.model.clear_cache()
        model.to('cpu')
        vae.to('cpu')
        del model, vae
        gc.collect()
        torch.cuda.empty_cache()
    report.update(elapsed_seconds=time.monotonic() - began, peak_vram_bytes=peak,
                  trainable_parameters=summary.trainable_parameters,
                  sampler=dict(solver='flow_euler', steps=40, shift=5.0, cfg='none'),
                  t5_loaded=False, action_scale=model_config.action_scale)
    return report
