"""One bounded native97 causal checkpoint evaluation; no launcher or GPU grant.

``run(request, directory, lease)`` is called by the existing leased worker.
The official causal model, cache allocators, Euler solver and streaming VAE are
reused. CFG branches retain separate full histories; an explicit CPU cache
option trades transfer time for a single GPU cache allocation. Failed outputs
are retained. This resource probe makes no visual-quality or speedup claim.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from training.demo.contracts import action_array, contained, load_initial, require, sha256, write_json
from training.studio.subset_eval import CONTROLS, FIXED_INPUTS
from training.studio.quality_probe import DEFAULT_NEGATIVE_CACHE

MODE = 'causal_native_eval'
REVISION = 'native97-causal-full-history-euler40-cfg5-v1'
ZERO_UPDATE_PURPOSE = 'zero_update_causal_initialization'
TF_STAGE = 'causal_r005_native97_fp32_v1'
BID_STAGE = 'causal_r005_native97_fp32_bid_v1'
DATA_ROOT = Path('/DATA_DISK/InteractWorld')


def tensor_sha(value):
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def normalize_initial_latent(latent):
    """The official VAE returns FP32 even under BF16 autocast; pin KV dtype."""
    import torch
    require(tuple(latent.shape) == (1, 1, 48, 30, 52) and latent.is_floating_point()
            and bool(torch.isfinite(latent).all()), 'initial latent geometry or values differ')
    return latent.to(dtype=torch.bfloat16)


def restore_fp32_checkpoint(model, state):
    """Undo construction-time casts before loading the original trained values."""
    import torch
    import train_action_teacher as action_entry
    from training.models.lora import load_trainable_state_dict
    action_entry._apply_lora_parameter_dtype(model, 'float32')
    load_trainable_state_dict(model, state)
    lora = [p for name, p in model.named_parameters() if '.lora_a.' in name or '.lora_b.' in name]
    require(bool(lora) and all(p.dtype == torch.float32 for p in lora), 'FP32 trained LoRA was rounded')
    return model.requires_grad_(False).eval()


def close_video_preserving(writer):
    """Finalize the existing FFmpeg writer without its failure-time unlink."""
    import subprocess
    if writer.process.stdin is not None:
        try:
            writer.process.stdin.close()
        except BrokenPipeError:
            pass
    try:
        code = writer.process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        writer.process.kill()
        writer.process.wait(timeout=5)
        raise RuntimeError('video finalization timed out; partial output preserved')
    require(code == 0, f'ffmpeg failed with exit code {code}; partial output preserved')
    writer.partial.replace(writer.path)


def validate_request(request):
    require(request.get('mode') == MODE and request.get('experiment_revision') == REVISION,
            'explicit native97 causal evaluation revision required')
    require(type(request.get('max_seconds')) is int and 60 <= request['max_seconds'] <= 3600,
            'causal evaluation requires a finite <=3600-second deadline')
    require(request.get('future_frames') in (96, 240) and type(request['future_frames']) is int,
            'evaluate exactly 6s qualification or 15s fixed rollout')
    require(request.get('trained_stage', TF_STAGE) in (TF_STAGE, BID_STAGE),
            'explicit supported trained_stage required; checkpoint auto-detection is disabled')
    if 'purpose' in request:
        require(request['purpose'] == ZERO_UPDATE_PURPOSE,
                'unknown causal evaluation purpose')
        require(request['future_frames'] == 96,
                'zero-update causal initialization is a 96-future-frame diagnostic only')
        require(request.get('trained_stage', TF_STAGE) == TF_STAGE,
                'zero-update diagnostic is restricted to the original TF parent contract')
    require(request.get('steps') == 40 and request.get('cfg_scale') == 5
            and request.get('seed') == 42 and request.get('solver') == 'flow_euler',
            'fixed 40-step Euler CFG5 seed42 required')
    require(type(request.get('cfg_cache_offload')) is bool, 'choose CFG cache offload explicitly')
    require(isinstance(request.get('job_id'), str) and Path(request['job_id']).name == request['job_id']
            and request['job_id'] not in ('', '.', '..'), 'invalid evaluation job identity')
    for field in ('checkpoint', 'training_config', 'source_job', 'prompt_cache', 'negative_cache'):
        require(isinstance(request.get(field), str) and Path(request[field]).is_absolute(), f'absolute {field} required')
    for field in ('checkpoint_sha256', 'training_config_sha256', 'prompt_cache_sha256',
                  'prompt_receipt_sha256', 'negative_cache_sha256', 'negative_receipt_sha256',
                  'config_sha256', 'initial_sha256', 'actions_sha256'):
        value = request.get(field)
        require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
                f'exact {field} required')


def make_request(*, job_id, source_revision, checkpoint, training_config, source_job,
                 dev_prompt_sidecar, future_frames=240, cfg_cache_offload=True,
                 max_seconds=3600, negative_cache=DEFAULT_NEGATIVE_CACHE, purpose=None,
                 trained_stage=None):
    """CPU-only request binding; does not create outputs, load models or reserve GPUs."""
    source, prompt, negative = map(Path, (source_job, dev_prompt_sidecar, negative_cache))
    request = dict(mode=MODE, experiment_revision=REVISION, job_id=job_id,
        source_revision=source_revision, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        training_config=str(training_config), training_config_sha256=sha256(training_config),
        source_job=str(source), prompt_cache=str(prompt), prompt_cache_sha256=sha256(prompt),
        prompt_receipt_sha256=sha256(prompt.with_suffix(prompt.suffix + '.receipt.json')),
        negative_cache=str(negative), negative_cache_sha256=sha256(negative),
        negative_receipt_sha256=sha256(negative.with_suffix('.json')),
        max_seconds=max_seconds, future_frames=future_frames, cfg_cache_offload=cfg_cache_offload,
        steps=40, cfg_scale=5, seed=42, solver='flow_euler')
    request.update({field: sha256(source / name) for name, field in FIXED_INPUTS})
    if purpose is not None:
        request['purpose'] = purpose
    if trained_stage is not None:
        request['trained_stage'] = trained_stage
    validate_request(request)
    return request


def _stage_interface(request):
    """Select a trusted entry and parser from the request, never from weights."""
    validate_request(request)
    if request.get('trained_stage', TF_STAGE) == BID_STAGE:
        import train_causal_r005_bid_probe as entry
        from training.causal_moba import load_moba_config as load_config
    else:
        import train_causal_r005_probe as entry
        from training.causal_tf import load_causal_config as load_config
    return entry, load_config


def _load_evaluation_weights(request, config):
    """Separate the strict trained stage from an explicitly untrained control.

    The control loads the actual SHA-pinned R005 parent through the same trusted
    migration used before training. It neither creates a step-zero checkpoint
    nor relaxes the trained checkpoint's FP32/configuration checks.
    """
    import torch
    import train_action_teacher as action_entry

    entry, _ = _stage_interface(request)
    entry.validate_probe_config(config)
    if request.get('purpose') == ZERO_UPDATE_PURPOSE:
        require(request['checkpoint'] == str(config.lineage.checkpoint_path)
                and request['checkpoint_sha256'] == config.lineage.checkpoint_sha256,
                'zero-update checkpoint must exactly match the configured R005 parent path and SHA256')
        parent, initialization, hashes = entry.load_parent(config, request['training_config'])
        require(hashes.get('training_config') == request['training_config_sha256']
                and hashes.get('teacher_checkpoint') == request['checkpoint_sha256'],
                'zero-update parent/configuration binding differs')
        # These are source checkpoint facts, not a fictitious new training step.
        lineage = dict(stage=ZERO_UPDATE_PURPOSE, purpose=ZERO_UPDATE_PURPOSE,
            trained_checkpoint=False, optimization_steps_this_round=0,
            source_stage=initialization['source_stage'], source_step=initialization['source_step'],
            checkpoint=request['checkpoint'], checkpoint_sha256=request['checkpoint_sha256'],
            initialization=initialization, manifest_hashes=hashes,
            method_contract=dict(method=ZERO_UPDATE_PURPOSE, new_causal_training_performed=False,
                independent_first_frame=True, future_latent_frames_per_block=3,
                kv_inference_history='generated_latents_at_t0', fp32_lora_storage=True,
                source_lora_dtype='bfloat16', source_values_unchanged=True,
                quality_validated=False))
        return parent['trainable_model'], lineage

    payload = torch.load(request['checkpoint'], map_location='cpu', weights_only=False)
    require(payload.get('format_version') == 1 and payload.get('stage') == entry.STAGE_NAME
            and payload.get('method_contract') == entry.method_contract(), 'new native97 FP32 causal stage required')
    require(type(payload.get('step')) is int and 0 < payload['step'] <= 20
            and payload.get('micro_batches_consumed') == payload['step'] * 8,
            'checkpoint must contain completed optimizer steps')
    require(json.dumps(payload.get('config'), sort_keys=True) == json.dumps(config.to_dict(), sort_keys=True)
            and payload.get('manifest_hashes', {}).get('training_config') == request['training_config_sha256'],
            'serialized training configuration differs')
    state = payload.get('trainable_model')
    require(isinstance(state, dict) and bool(state) and all(torch.is_tensor(v) and bool(torch.isfinite(v).all())
            for v in state.values()), 'checkpoint weights must be named finite tensors')
    action_entry._validate_lora_state_dtype(state, 'float32')
    initialization = payload.get('initialization', {})
    require(initialization.get('mode') == 'r005_native97_causal_fp32_weights_only'
            and initialization.get('sha256') == config.lineage.checkpoint_sha256,
            'causal checkpoint parent lineage differs')
    return state, dict(stage=payload['stage'], step=payload['step'], checkpoint=request['checkpoint'],
        checkpoint_sha256=request['checkpoint_sha256'], initialization=initialization,
        method_contract=payload['method_contract'], manifest_hashes=payload['manifest_hashes'])


def load_inputs(request):
    """Selected checkpoint and dev text only; no teacher/model download or T5 load."""
    import numpy as np
    import torch
    import yaml
    from training.studio.quality_probe import load_negative

    validate_request(request)
    source = Path(request['source_job'])
    for name, field in FIXED_INPUTS:
        require(sha256(source / name) == request[field], f'fixed source changed: {name}')
    raw = yaml.safe_load((source / 'rollout.yaml').read_text(encoding='utf-8'))
    require(len(raw['scenes']) == 1, 'one fixed evaluation scene required')
    scene = raw['scenes'][0]
    require(scene['seed'] == 42 and scene['action_segments'] == CONTROLS,
            'original WJ120 then WL120 scene/seed required')
    initial = load_initial(source / 'initial.npy')
    actions = np.load(source / 'actions.npy', allow_pickle=False)
    require(initial.shape == (480, 832, 3) and initial.dtype == np.uint8
            and actions.dtype == np.float32 and np.array_equal(actions, action_array(CONTROLS)),
            'fixed RGB or exact future actions differ')
    require(sha256(request['training_config']) == request['training_config_sha256']
            and sha256(request['checkpoint']) == request['checkpoint_sha256'], 'checkpoint/config changed')
    _, load_config = _stage_interface(request)
    config = load_config(request['training_config'])
    state, lineage = _load_evaluation_weights(request, config)
    prompt_path = Path(request['prompt_cache'])
    receipt_path = prompt_path.with_suffix(prompt_path.suffix + '.receipt.json')
    require(sha256(prompt_path) == request['prompt_cache_sha256']
            and sha256(receipt_path) == request['prompt_receipt_sha256'], 'dev prompt source changed')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    require(receipt.get('kind') == 'window97_eval_text_reference_v1'
            and receipt.get('scope') == 'dev_text_only_no_video_features'
            and receipt.get('prompt_policy') == 'scene_static_only_v1'
            and receipt.get('cache_sha256') == request['prompt_cache_sha256']
            and Path(receipt.get('training_manifest', '')).resolve() == Path(config.data.manifest_path).resolve()
            and receipt.get('training_manifest_sha256') == lineage['manifest_hashes']['dataset_manifest'],
            'dev text receipt belongs to another training source')
    episode = scene['source_episode_id']
    binding = receipt.get('episodes', {}).get(episode, {})
    require(binding.get('split') == 'dev' and binding.get('prompt') == scene['prompt'],
            'scene text is not the selected held-out static prompt')
    cache = torch.load(prompt_path, map_location='cpu', weights_only=True)
    prompt = cache['prompt_embeds'][episode]
    require(prompt.ndim == 2 and 1 <= prompt.shape[0] <= 512 and prompt.shape[1] == 4096
            and bool(torch.isfinite(prompt).all()) and tensor_sha(prompt) ==
            receipt.get('selected_embedding_sha256_float32', {}).get(episode), 'dev embedding differs')
    negative_path = Path(request['negative_cache'])
    require(sha256(negative_path) == request['negative_cache_sha256']
            and sha256(negative_path.with_suffix('.json')) == request['negative_receipt_sha256'],
            'negative text source changed')
    negative, negative_source = load_negative(negative_path, config.model.base_model_path)
    lineage.update(prompt_text=scene['prompt'], prompt_episode=episode, prompt_sha256_float32=tensor_sha(prompt),
        negative_source=negative_source, training_rgb_frames=97, t5_loaded=False)
    return state, config, prompt.detach().to(torch.bfloat16), negative, initial, actions[:request['future_frames']], lineage


class FullHistoryCFG:
    """Thin CFG/cache ownership around the existing CausalWanModel KV forward.

    The GPU scratch cache is overwritten from the selected branch before every
    forward. Only its written prefix is uploaded; only the current block is
    copied back. Clean commits update both branches at t=0, without evictions.
    """
    def __init__(self, student, config, first, positive, negative, *, capacity, offload,
                 check=lambda: None, cache_initializers=None):
        import torch
        self.student, self.config, self.check = student, config, check
        require(first.dtype == torch.bfloat16, 'causal KV caches require an explicitly BF16 initial latent')
        backbone = student.model
        require(backbone.local_attn_size == -1 and not backbone.use_relative_rope
                and backbone.independent_first_frame and backbone.num_frame_per_block == 3,
                'full-history absolute-position block3 model required')
        self.tokens = first.shape[-2] * first.shape[-1] // (backbone.patch_size[1] * backbone.patch_size[2])
        self.device, self.offload, self.position = first.device, offload, 0
        self.prompts = [positive.to(first.device), negative.to(first.device)]
        self.stats = dict(forward_calls=0, host_transfer_bytes=0, host_transfer_seconds=0.,
                          cfg_cache_offload=offload, slow_path=offload, evictions=0,
                          capacity_latents=capacity, tokens_per_frame=self.tokens, commits=[])
        if cache_initializers is None:
            from pipeline.causal_inference import CausalInferencePipeline
            cache_initializers = (CausalInferencePipeline._initialize_kv_cache,
                                  CausalInferencePipeline._initialize_crossattn_cache)
        def allocate(device):
            holder = SimpleNamespace(generator=student, local_attn_size=-1, frame_seq_length=self.tokens,
                num_transformer_blocks=len(backbone.blocks), args=SimpleNamespace(image_or_video_shape=[1, capacity]))
            cache_initializers[0](holder, 1, first.dtype, device)
            cache_initializers[1](holder, 1, first.dtype, device)
            return holder.kv_cache1, holder.crossattn_cache
        self.branches = [allocate('cpu' if offload else first.device) for _ in range(2)]
        self.scratch = allocate(first.device) if offload else None
        self.stats['branch_cache_bytes'] = sum(v.numel() * v.element_size() for cache in self.branches[0]
            for layer in cache for v in layer.values() if torch.is_tensor(v))
        bytes_per_frame = sum(layer[key][:, :self.tokens].numel() * layer[key].element_size()
                              for layer in self.branches[0][0] for key in ('k', 'v'))
        chunks = (capacity - 1) // 3
        # 40 denoising calls and one clean commit per branch/block. Current
        # blocks overwrite the same absolute positions on every solver step.
        transfer_frames = 2 + 2 * sum((1 + 3 * i) + 40 * (4 + 3 * i) + 41 * 3 for i in range(chunks))
        self.stats['projected_host_transfer_bytes'] = transfer_frames * bytes_per_frame if offload else 0
        self.actions = None
        self.pending = None
        for branch in range(2):
            self._call(branch, first, 0, 0, None)
        self.position = 1
        self.stats['commits'].append(dict(start=0, stop=1, timestep=0))

    def _copy(self, destination, source):
        destination.copy_(source)
        self.stats['host_transfer_bytes'] += source.numel() * source.element_size()

    def _call(self, branch, latents, timestep, start, action):
        import torch
        self.check()
        require(not torch.is_grad_enabled(), 'KV inference requires no_grad')
        stored_kv, cross = self.branches[branch]
        tick = time.monotonic()
        if self.offload:
            kv, scratch_cross = self.scratch
            for src, dst in zip(stored_kv, kv):
                end = int(src['local_end_index'].item())
                for key in ('k', 'v'):
                    self._copy(dst[key][:, :end], src[key][:, :end])
                for key in ('global_end_index', 'local_end_index', 'ref_token_len'):
                    dst[key].copy_(src[key])
            # Cross attention is small and also text-specific. Keep its own
            # branch tensors resident so no prompt cache can leak across CFG.
            for layer in cross:
                for key in ('k', 'v'):
                    if layer[key].device != self.device:
                        layer[key] = layer[key].to(self.device)
            self.stats['host_transfer_seconds'] += time.monotonic() - tick
        else:
            kv = stored_kv
        time_tensor = torch.as_tensor(timestep, device=latents.device, dtype=torch.float32).reshape(-1, 1)
        time_tensor = time_tensor.expand(latents.shape[0], latents.shape[1])
        flow = self.student.model(latents.permute(0, 2, 1, 3, 4), t=time_tensor,
            context=[self.prompts[branch]], seq_len=None, act_context=action,
            act_context_scale=self.config.model.action_scale, kv_cache=kv,
            crossattn_cache=cross, current_start=start * self.tokens)
        if isinstance(flow, tuple):
            flow = flow[0]
        flow = flow.permute(0, 2, 1, 3, 4)
        expected = (start + latents.shape[1]) * self.tokens
        require(all(int(layer['global_end_index'].item()) == expected
                    and int(layer['local_end_index'].item()) == expected for layer in kv),
                'causal KV position did not advance exactly or history was evicted')
        if self.offload:
            tick = time.monotonic()
            for src, dst in zip(kv, stored_kv):
                for key in ('k', 'v'):
                    self._copy(dst[key][:, start * self.tokens:expected], src[key][:, start * self.tokens:expected])
                for key in ('global_end_index', 'local_end_index', 'ref_token_len'):
                    dst[key].copy_(src[key])
            self.stats['host_transfer_seconds'] += time.monotonic() - tick
        self.stats['forward_calls'] += 1
        return flow

    def velocity(self, *, noisy_block, history, action_block, timestep, block_index, **_):
        from training.models.action_adapter import build_action_context
        start = 1 + block_index * 3
        require(self.position == start and history.shape[1] == start and noisy_block.shape[1] == 3,
                'out-of-order or incompatible causal block')
        if self.pending != block_index:
            require(self.pending is None, 'interleaved causal block')
            self.actions = build_action_context(action_block, height=self.config.data.height,
                width=self.config.data.width, device=noisy_block.device, dtype=noisy_block.dtype)
            self.pending = block_index
        positive = self._call(0, noisy_block, timestep, start, self.actions)
        negative = self._call(1, noisy_block, timestep, start, self.actions)
        return negative + 5 * (positive - negative)

    def commit(self, endpoint, block_index):
        start = 1 + block_index * 3
        require(self.pending == block_index and self.position == start, 'cannot commit an unsolved block')
        for branch in range(2):
            self._call(branch, endpoint, 0, start, self.actions)
        self.position = start + 3
        self.pending, self.actions = None, None
        self.stats['commits'].append(dict(start=start, stop=self.position, timestep=0))

    def close(self):
        self.branches.clear()
        self.scratch = self.prompts = self.actions = None


def run(request, directory, lease):
    """Generate one 97/241-frame resource probe inside an already leased worker."""
    import numpy as np
    import torch
    from PIL import Image
    import train_causal_teacher_forcing as tf_entry
    from training.demo.backend import validate_active_authority
    from training.longforcing_lite import _solve_block
    from training.eval.rollout15s import ffmpeg_writer_factory

    validate_request(request)
    directory = contained(directory, DATA_ROOT)
    require(directory.name == request['job_id'], 'worker output does not match request')
    target = directory / 'causal-evaluation'
    target.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    report = dict(status='running', mode=MODE, revision=REVISION, request=request,
        blocks=[], quality_improvement_verified=False, realtime=False, automatic_retry=False,
        future_ground_truth_used=False, failed_outputs_preserved=True)
    model = vae = flow = writer = None
    cuda_started = False
    def check():
        require(time.monotonic() - began < request['max_seconds'] - 15, 'finite causal evaluation deadline reached')
        validate_active_authority(lease)
    try:
        check()
        for field in ('checkpoint', 'training_config', 'source_job', 'prompt_cache', 'negative_cache'):
            contained(request[field], DATA_ROOT)
        require(os.environ.get('HF_HUB_OFFLINE') == '1' and os.environ.get('TRANSFORMERS_OFFLINE') == '1',
                'causal probe requires offline existing model execution')
        state, config, positive, negative, initial, actions, lineage = load_inputs(request)
        contained(config.model.base_model_path, DATA_ROOT)
        report['lineage'] = lineage
        Image.fromarray(initial).save(target / 'initial.png')
        np.save(target / 'actions.npy', actions, allow_pickle=False)
        write_json(target / 'progress.json', dict(report, phase='encoding_initial'))
        from utils.wan_wrapper import WanVAEWrapper
        device = torch.device('cuda')
        torch.cuda.reset_peak_memory_stats(device)
        cuda_started = True
        vae = WanVAEWrapper(vae_type='Wan2.2_VAE', pretrained_path=str(Path(config.model.base_model_path) / 'Wan2.2_VAE.pth'))
        vae.requires_grad_(False).eval().to(device=device, dtype=torch.bfloat16)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            pixel = torch.from_numpy(initial.copy()).to(device, torch.bfloat16).permute(2, 0, 1)[None, :, None]
            first = normalize_initial_latent(vae.encode_to_latent(pixel.div(127.5).sub(1)))
        del pixel
        vae.to('cpu')
        vae.model.clear_cache()
        torch.cuda.empty_cache()
        check()
        tick = time.monotonic()
        model, _ = tf_entry._build_model(config, {'trainable_model': state}, device)
        # _build_model casts the model to BF16. Restore FP32 storage BEFORE
        # loading the original checkpoint tensors, retaining small trained deltas.
        restore_fp32_checkpoint(model, state)
        state.clear()
        gc.collect()
        report['load_seconds'] = time.monotonic() - tick
        report['inference_precision'] = dict(lora='float32', base='bfloat16', compute='bfloat16', checkpoint_rewritten=False)
        chunks = request['future_frames'] // 12
        rng = torch.Generator(device='cpu').manual_seed(42)
        history = first
        tick = time.monotonic()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            flow = FullHistoryCFG(model, config, first, positive, negative, capacity=1 + chunks * 3,
                                  offload=request['cfg_cache_offload'], check=check)
            report['kv'] = flow.stats
            for index in range(chunks):
                block_tick = time.monotonic()
                noise = torch.randn((1, 3, 48, 30, 52), generator=rng, dtype=torch.float32).to(device, torch.bfloat16)
                action = torch.from_numpy(actions[index * 12:(index + 1) * 12].copy())[None].to(device, torch.bfloat16)
                endpoint = _solve_block(flow.velocity, initial_noise=noise, history=history,
                    action_block=action, block_index=index, conditions={}, steps=40, timestep_shift=5.)
                require(bool(torch.isfinite(endpoint).all()), 'nonfinite generated latent')
                flow.commit(endpoint, index)
                history = torch.cat((history, endpoint), dim=1)
                torch.cuda.synchronize()
                report['blocks'].append(dict(index=index, action_start=index * 12, action_stop=(index + 1) * 12,
                    latent_start=1 + index * 3, latent_stop=4 + index * 3, seconds=time.monotonic() - block_tick,
                    noise_sha256=tensor_sha(noise), action_sha256=tensor_sha(action), endpoint_sha256=tensor_sha(endpoint)))
                if flow.stats['host_transfer_bytes']:
                    flow.stats['linear_transfer_seconds_estimate'] = (flow.stats['host_transfer_seconds']
                        * flow.stats['projected_host_transfer_bytes'] / flow.stats['host_transfer_bytes'])
                write_json(target / 'progress.json', dict(report, phase='generating'))
        report['latent_generation_seconds'] = time.monotonic() - tick
        report['generation_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['generation_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        latents = history.detach().cpu()
        torch.save(dict(latents=latents, seed=42, checkpoint_sha256=request['checkpoint_sha256']), target / 'generated_latents.pt')
        flow.close()
        model.to('cpu')
        del history, first, endpoint, noise, action
        gc.collect()
        torch.cuda.empty_cache()
        check()
        vae.to(device=device, dtype=torch.bfloat16)
        vae.model.clear_cache()
        writer = ffmpeg_writer_factory(target / 'raw.mp4', 832, 480, 16)
        written = 0
        tick = time.monotonic()
        for start, stop, count in [(0, 1, 1), *[(1 + i * 3, 4 + i * 3, 12) for i in range(chunks)]]:
            check()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                decoded = vae.decode_to_pixel(latents[:, start:stop].to(device, torch.bfloat16), use_cache=True, return_in_cpu=True)
            require(tuple(decoded.shape) == (1, count, 3, 480, 832) and bool(torch.isfinite(decoded).all()),
                    'continuous VAE decode frame count or values differ')
            rgb = decoded[0].permute(0, 2, 3, 1).add(1).mul(127.5).round().clamp(0, 255).byte().numpy()
            if start == 0:
                rgb[0] = initial  # Output carries the exact submitted first RGB.
            writer.write(rgb)
            written += count
        close_video_preserving(writer)
        writer = None
        require(written == request['future_frames'] + 1, 'decoded video length differs')
        report.update(status='completed', frames_written=written, raw_video=str(target / 'raw.mp4'),
            decode_seconds=time.monotonic() - tick)
    except BaseException as error:
        report.update(status='failed', failure_type=type(error).__name__, error=str(error))
        raise
    finally:
        if writer is not None:
            # The standard close()/abort() remove failed video output.
            try:
                close_video_preserving(writer)
            except Exception as error:
                report['writer_close_error'] = repr(error)
        if flow is not None:
            flow.close()
        for item in (model, vae):
            if item is not None:
                try:
                    item.to('cpu')
                except Exception as error:
                    report.setdefault('cleanup_errors', []).append(repr(error))
        if cuda_started:
            report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            torch.cuda.empty_cache()
        report['elapsed_seconds'] = time.monotonic() - began
        write_json(target / 'result.json', report)
    return report
