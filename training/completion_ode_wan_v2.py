"""Real Wan adapters for packed ODE supervision and positive-only KV rollout.

No alternative ODE mathematics lives here. The shared CPU core owns the grid,
Euler trajectories, endpoint target and four-round endpoint/re-noise sampler.
The real-model probe remains a runtime gate, not a CPU fixture success claim.
"""
from __future__ import annotations

from contextlib import nullcontext
import inspect
import math
import time
from types import SimpleNamespace

import torch

import train_completion_ode_sufficiency as entry

require = entry.require


def expected_packed_token_mask(q, k, *, latent_frames, tokens_per_frame):
    """Independent all-token clean/own-noisy block predicate, padding diagonal."""
    n = latent_frames*tokens_per_frame
    frame = q.remainder(n).div(tokens_per_frame, rounding_mode='floor')
    block_start = torch.where(frame == 0, torch.zeros_like(frame), 1+3*((frame-1)//3))
    block_end = torch.where(frame == 0, torch.ones_like(frame), (block_start+3).clamp(max=latent_frames))
    start, end = block_start*tokens_per_frame, block_end*tokens_per_frame
    clean = (q < n) & (k < end)
    noisy = (q >= n) & (q < 2*n) & ((k < start) | ((k >= n+start) & (k < n+end)))
    return (q == k) | clean | noisy


def audit_installed_token_mask(backbone, *, device, latent_frames=25, tokens_per_frame=390,
                               query_chunk=256, check=lambda: None):
    """Exhaustively compare the actual installed mask, not just boundary samples."""
    require(type(backbone).__module__ == 'wan.modules.causal_model'
            and type(backbone).__name__ == 'CausalWanModel', 'actual installed Wan causal backbone required')
    require(backbone.local_attn_size == -1 and backbone.independent_first_frame
            and backbone.num_frame_per_block == 3 and not backbone.use_relative_rope
            and getattr(backbone, 'controlled_action_enabled', False) is True,
            'fixed absolute-position controlled causal architecture required')
    key = (str(device), latent_frames, tokens_per_frame, 3, -1, True, True, 0)
    require(backbone._block_mask_cache_key == key and backbone.block_mask is not None
            and backbone._block_mask_cache.get(key) is backbone.block_mask
            and all(getattr(block.self_attn, '_is_teacher_forcing', False) is True for block in backbone.blocks),
            'real latest forward must have installed packed teacher-forcing no-reference mask')
    require(type(query_chunk) is int and 1 <= query_chunk <= 512, 'bounded audit scratch chunk required')
    total = math.ceil(2*latent_frames*tokens_per_frame/128)*128
    k = torch.arange(total, device=device)[None, :]
    zero = torch.tensor(0, device=device)
    pairs, began = 0, time.monotonic()
    for start in range(0, total, query_chunk):
        check()
        q = torch.arange(start, min(start+query_chunk, total), device=device)[:, None]
        actual = backbone.block_mask.mask_mod(zero, zero, q, k)
        expected = expected_packed_token_mask(q, k, latent_frames=latent_frames, tokens_per_frame=tokens_per_frame)
        require(actual.dtype == torch.bool and actual.shape == expected.shape and torch.equal(actual, expected),
                'installed token mask differs from independent causal predicate')
        pairs += expected.numel()
        del actual, expected, q
    require(pairs == total**2, 'all padded token pairs must be audited')
    return dict(actual_token_mask_audited=True, exhaustive_token_pairs=pairs, padded_token_length=total,
        latent_frames=latent_frames, tokens_per_frame=tokens_per_frame, mask_cache_key=list(key),
        source_sha256=entry.previous.sha256_file(inspect.getfile(type(backbone))),
        audit_seconds=time.monotonic()-began, future_clean_visible=False,
        earlier_or_future_noisy_visible=False, padding_diagonal_only=True)


class WanPackedAdapter:
    def __init__(self, model, config, device, *, check=lambda: None, positive_only=False):
        self.model, self.config, self.device, self.check = model, config, torch.device(device), check
        self.positive_only, self.forward_count = positive_only, 0
        require(getattr(model, 'is_causal', False) is True, 'Wan causal wrapper required, never bidirectional adapter')

    def callback(self, *, noisy_latents, clean_context, actions, clean_visibility,
                 sigma, timestep, branch, conditions):
        from training.models.action_adapter import build_action_context
        self.check()
        require(branch in ('positive', 'negative') and (not self.positive_only or branch == 'positive'),
                'student positive-only / teacher two-CFG branches required')
        require(noisy_latents.shape == clean_context.shape and noisy_latents.ndim == 5
                and noisy_latents.dtype == clean_context.dtype == torch.bfloat16
                and noisy_latents.device == clean_context.device == self.device,
                'matching native BF16 packed noisy/independent clean buffers required')
        require(torch.equal(clean_visibility, entry.core().clean_visibility_mask(noisy_latents.shape[1], device=self.device)),
                'shared clean visibility contract differs')
        require(sigma.shape == timestep.shape == (noisy_latents.shape[0],)
                and conditions is not None and conditions.ndim == 3,
                'actual rounded grid values and batched text required')
        vector = timestep[:, None].expand(-1, noisy_latents.shape[1]).clone()
        vector[:, :1] = 0
        text = conditions.to(device=self.device, dtype=torch.bfloat16)
        action = build_action_context(actions.to(device=self.device, dtype=torch.bfloat16),
            height=self.config.data.height, width=self.config.data.width, device=self.device, dtype=torch.bfloat16)
        value = self.model(noisy_latents, conditional_dict=dict(prompt_embeds=list(text), act_context=action,
            act_context_scale=self.config.model.action_scale), timestep=vector, clean_x=clean_context,
            aug_t=torch.zeros_like(vector), replace_first_timestep_and_noise_latents=True)
        self.forward_count += 1
        flow = value[0] if isinstance(value, tuple) else value
        require(flow.shape == noisy_latents.shape and flow.dtype == torch.bfloat16
                and bool(torch.isfinite(flow).all()), 'finite same-shape native Wan velocity required')
        return flow

    def verify_causality(self, *, noisy, clean_context, actions, positive_conditions):
        require(all(not p.requires_grad and p.grad is None for p in self.model.parameters()),
                'real teacher must be completely frozen for independent input-VJP probe')
        before = {name: p._version for name, p in self.model.named_parameters()}
        started, calls = time.monotonic(), self.forward_count
        amp = torch.autocast('cuda', dtype=torch.bfloat16) if self.device.type == 'cuda' else nullcontext()
        saved = torch.autograd.graph.save_on_cpu(pin_memory=True) if self.device.type == 'cuda' else nullcontext()
        with torch.enable_grad(), amp, saved:
            probe = entry.core().probe_packed_causality(self.callback, noisy=noisy, clean_context=clean_context,
                actions=actions, positive_conditions=positive_conditions)
        require(self.forward_count-calls == 32 and probe['logical_forward_calls'] == 32,
                'all eight blocks need the actual clean/earlier-noisy/future-noisy/VJP checks')
        audit = audit_installed_token_mask(self.model.model, device=self.device,
            latent_frames=noisy.shape[1], tokens_per_frame=noisy.shape[-2]*noisy.shape[-1]//4, check=self.check)
        require(all(p.grad is None and p._version == before[name] for name,p in self.model.named_parameters()),
                'teacher weights/gradients changed during causality probe')
        return dict(causal_future_perturbation_passed=True, causal_gradient_probe_passed=True,
            actual_token_mask_audited=True, input_probe=probe, installed_mask=audit,
            actual_probe_forwards=32, actual_input_vjps=8, probe_seconds=time.monotonic()-started,
            finite_probe_is_not_full_jacobian_proof=True)


def build_teacher(parent, device, *, check=lambda: None):
    """Actual validated parent weights; caller must already own the original lease."""
    state = entry.load_parent(parent)
    config = entry.tf.load_config(parent['parent_config'])
    check()
    model, summary = entry.previous.build_model(config, device, frozen_action_model=state['frozen_action_model'],
        trainable_model=state['trainable_model'])
    require(entry.previous.model_signature(model) == state['model_signature'], 'teacher original LoRA identity changed')
    model.requires_grad_(False).eval()
    del state
    return model, config, summary


class WanSingleBranchKV:
    """One positive KV cache; no negative branch and no future-GT argument."""
    branch_count, external_cfg_scale = 1, 1.

    def __init__(self, model, config, *, capacity=25, check=lambda: None, cache_initializers=None):
        backbone = model.model
        require(backbone.local_attn_size == -1 and not backbone.use_relative_rope
                and backbone.independent_first_frame and backbone.num_frame_per_block == 3,
                'fixed full-history absolute-position model required')
        require(type(capacity) is int and capacity >= 4 and (capacity-1)%3 == 0, 'bounded complete cache capacity required')
        self.model, self.config, self.capacity, self.check = model, config, capacity, check
        self.cache_initializers, self.forward_count, self.position, self.pending = cache_initializers, 0, 0, None
        self.kv = self.cross = self.actions = self.text = None
        self.commits = []

    def _call(self, value, timestep, start, action):
        self.check()
        require(not torch.is_grad_enabled(), 'actual KV route requires no_grad, never detached training shortcut')
        require(start+value.shape[1] <= self.capacity, 'bounded history capacity exhausted')
        times = torch.as_tensor(timestep, device=value.device, dtype=torch.float32).reshape(-1, 1)
        times = times.expand(value.shape[0], value.shape[1])
        # The official backbone/activations are BF16 but our trained LoRA stays
        # FP32. Its ordinary nn.Linear modules need actual CUDA autocast even
        # when an online caller only entered no_grad (not a training loop).
        amp = torch.autocast('cuda', dtype=torch.bfloat16) if value.device.type == 'cuda' else nullcontext()
        with amp:
            flow = self.model.model(value.permute(0, 2, 1, 3, 4), t=times, context=[self.text], seq_len=None,
                act_context=action, act_context_scale=self.config.model.action_scale, kv_cache=self.kv,
                crossattn_cache=self.cross, current_start=start*self.tokens)
        flow = (flow[0] if isinstance(flow, tuple) else flow).permute(0, 2, 1, 3, 4)
        expected = (start+value.shape[1])*self.tokens
        require(flow.shape == value.shape and flow.dtype == value.dtype and bool(torch.isfinite(flow).all())
                and all(int(layer['global_end_index']) == expected and int(layer['local_end_index']) == expected
                        for layer in self.kv), 'single-branch causal KV positions/velocity differ')
        self.forward_count += 1
        return flow

    def initialize(self, first, conditions):
        require(self.forward_count == self.position == 0 and self.kv is None and first.shape[0:2] == (1, 1)
                and first.dtype == torch.bfloat16, 'fresh one-branch BF16 first-latent session required')
        self.text = conditions.squeeze(0).to(device=first.device, dtype=first.dtype)
        require(self.text.ndim == 2 and self.text.shape[-1] == 4096, 'single positive embedding required')
        backbone = self.model.model
        self.tokens = first.shape[-2]*first.shape[-1]//(backbone.patch_size[1]*backbone.patch_size[2])
        initializers = self.cache_initializers
        if initializers is None:
            from pipeline.causal_inference import CausalInferencePipeline
            initializers = (CausalInferencePipeline._initialize_kv_cache, CausalInferencePipeline._initialize_crossattn_cache)
        holder = SimpleNamespace(generator=self.model, local_attn_size=-1, frame_seq_length=self.tokens,
            num_transformer_blocks=len(backbone.blocks), args=SimpleNamespace(image_or_video_shape=[1, self.capacity]))
        initializers[0](holder, 1, first.dtype, first.device)
        initializers[1](holder, 1, first.dtype, first.device)
        self.kv, self.cross = holder.kv_cache1, holder.crossattn_cache
        self._call(first, 0, 0, None)
        self.position = 1
        self.commits.append(dict(start=0, stop=1, timestep=0))

    def velocity(self, *, noisy_block, history, action_block, timestep, block_index,
                 branch, cfg_scale, conditions, **unused):
        from training.models.action_adapter import build_action_context
        require(branch == 'positive' and cfg_scale == 1. and not unused.keys()-{'sigma'},
                'actual single positive branch only')
        start = 1+block_index*3
        require(start == self.position == history.shape[1] and noisy_block.shape[1] == 3,
                'own committed history only, ordered three-latent blocks')
        require(torch.equal(conditions.squeeze(0).to(self.text), self.text), 'positive conditions changed mid-session')
        if self.pending != block_index:
            require(self.pending is None, 'interleaved blocks prohibited')
            self.actions = build_action_context(action_block, height=self.config.data.height,
                width=self.config.data.width, device=noisy_block.device, dtype=noisy_block.dtype)
            self.pending = block_index
        return self._call(noisy_block, timestep, start, self.actions)

    def commit(self, endpoint, block_index):
        start = 1+block_index*3
        require(self.pending == block_index and self.position == start, 'unsolved block cannot commit')
        self._call(endpoint, 0, start, self.actions)
        self.position, self.pending, self.actions = start+3, None, None
        self.commits.append(dict(start=start, stop=self.position, timestep=0))

    def close(self):
        self.kv = self.cross = self.actions = self.text = None
