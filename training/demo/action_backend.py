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


def load_action_inputs(raw, scene):
    """CPU-only exact checkpoint/artifact/text binding; no future video/shard read."""
    import torch
    from training.config import load_config
    from training.models.action_adapter import validate_action_scale
    from training.demo.relocation import ArtifactHashes, artifact_paths, relocate_path, relocated_manifest_hashes, validate_relocations
    validate_action_config(raw)
    spec = raw['lineage']
    path = Path(spec['checkpoint_path'])
    verification_cache = spec.get('verification_cache')
    digest = (ArtifactHashes(verification_cache, protected_paths=[path, *spec['artifact_paths'].values(),
                  Path(spec['expected_base_model_path']) / 'Wan2.2_VAE.pth'])
              if verification_cache is not None else sha256)
    require(digest(path) == spec['checkpoint_sha256'].lower(), 'Action checkpoint SHA mismatch')
    config_path = Path(spec['artifact_paths']['training_config'])
    config = load_config(config_path)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    require(payload.get('format_version') == 1 and payload.get('stage') == ACTION_STAGE
            and type(payload.get('step')) is int and payload['step'] > 0, 'not a completed self-trained Action checkpoint')
    compare_saved_config(payload.get('config'), config.to_dict(), payload['step'])
    # Compare original serialized fields before any in-memory relocation. The
    # map affects only file access, never checkpoint/YAML/receipt contents.
    relocations = validate_relocations(spec.get('path_relocations', {}))
    expected_paths = artifact_paths(config, config_path, relocations)
    require(config.data.prompt_cache_path is not None, 'Action UI requires the actual static prompt cache')
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
    if relocations:
        lineage.update(path_relocations=relocations,
                       resolved_artifact_paths={key: str(value.resolve()) for key, value in expected_paths.items()},
                       resolved_base_model_path=str(base_model_path.resolve()))
    del payload
    gc.collect()
    cache_path = expected_paths['prompt_cache']
    receipt = json.loads(cache_path.with_suffix('.pt.receipt.json').read_text(encoding='utf-8'))
    binding = receipt.get('episodes', {}).get(scene['source_episode_id'])
    require(binding and binding.get('split') == 'dev' and binding.get('prompt') == scene['prompt'],
            'displayed Action prompt does not match the actual static embedding text/source')
    # _manifest_hashes already validated the full immutable cache/receipt. Only
    # inspect the selected tensor here; no all-episode numerical audit or T5 load.
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    require(cache.get('schema_version') == 1 and cache.get('kind') == 'scene_static_prompt_cache',
            'unexpected static prompt payload')
    prompt = cache['prompt_embeds'][scene['source_episode_id']]
    require(torch.is_tensor(prompt) and prompt.device.type == 'cpu' and prompt.is_floating_point()
            and prompt.ndim == 2 and 1 <= prompt.shape[0] <= 512 and prompt.shape[1] == 4096
            and bool(torch.isfinite(prompt).all()), 'selected static embedding is invalid')
    prompt = prompt.detach().to(dtype=torch.bfloat16).clone()
    del cache
    gc.collect()
    lineage['prompt_condition'] = dict(policy='scene_static_only_v1', text=scene['prompt'],
                                      episode_id=scene['source_episode_id'], cache_sha256=hashes['prompt_cache'],
                                      embedding_sha256_float32=tensor_sha(prompt), shape=list(prompt.shape), t5_loaded=False)
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
