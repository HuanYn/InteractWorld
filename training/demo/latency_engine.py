"""Reusable Action inference for explicit latency experiments, not the demo contract.

The caller owns GPU authorization, a finite session watchdog and the shared lease.
Importing this module does not load models or initialize CUDA. Construction loads
the already verified Action state; ``close`` releases the resident modules.
"""
from __future__ import annotations

import copy
from functools import partial
import gc
from pathlib import Path
import threading
import time

import numpy as np

from training.demo import action_backend as action
from training.demo.contracts import require


def _validate_steps(steps):
    require(type(steps) is int and steps in (40, 20, 10, 6),
            'latency experiments support explicit steps=40, 20, 10 or 6 only')


class _WanRuntime:
    """Actual CUDA implementation; the engine can instead inject a CPU test double."""

    timing_kind = 'cuda_synchronized_host_wall_clock'

    def __init__(self, state, model_config, residency):
        import torch
        import utils.wan_wrapper as wrappers
        from training.models.lora import configure_action_teacher, load_trainable_state_dict

        self.torch = torch
        self.config = copy.deepcopy(model_config)
        self.residency = residency
        self.model = self.vae = None
        try:
            self.model = wrappers.WanDiffusionWrapper(
                model_name=self.config.base_model_path, model_type='ci2v',
                is_causal=False, timestep_shift=5.0, downscale_factor_control_adapter=16)
            require(self.model.is_causal is False and self.model.uniform_timestep is True,
                    'Action latency experiments require the noncausal wrapper')
            summary = configure_action_teacher(self.model.model, rank=self.config.lora_rank,
                alpha=self.config.lora_alpha, dropout=self.config.lora_dropout)
            require({name for name, value in self.model.named_parameters() if value.requires_grad}
                    == set(state), 'Action trainable tensor names mismatch')
            load_trainable_state_dict(self.model, state)
            self.trainable_parameters = summary.trainable_parameters
            self.model.requires_grad_(False).eval().to(dtype=torch.bfloat16)
            self.vae = wrappers.WanVAEWrapper(
                pretrained_path=str(Path(self.config.base_model_path) / 'Wan2.2_VAE.pth'))
            self.vae.requires_grad_(False).eval().to(dtype=torch.bfloat16)
            if residency == 'co_resident':
                self.model.to('cuda')
                self.vae.to('cuda')
                self.synchronize()
        except BaseException:
            try:
                self.close()
            except Exception:
                pass  # Keep the original construction/OOM error.
            raise

    def synchronize(self):
        self.torch.cuda.synchronize()

    def begin(self):
        self.torch.cuda.reset_peak_memory_stats()
        self.vae.model.clear_cache()

    def peak_bytes(self):
        return self.torch.cuda.max_memory_allocated()

    def transfer(self, target):
        if self.residency == 'co_resident':
            return
        inactive, active = ((self.model, self.vae) if target == 'vae'
                            else (self.vae, self.model))
        inactive.to('cpu')
        gc.collect()
        self.torch.cuda.empty_cache()
        active.to('cuda')

    def encode(self, pixels):
        torch = self.torch
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            return self.vae.encode_to_latent(
                pixels.to(device='cuda', dtype=torch.bfloat16)).detach().cpu()

    def generate(self, first, prompt, keys, noise, steps, *, fastpath=False):
        from training.models.action_adapter import build_action_context
        from training.longforcing_lite import euler_sigmas

        torch = self.torch
        condition = first.to(device='cuda', dtype=torch.bfloat16)
        conditions = dict(prompt_embeds=[prompt.to(device='cuda', dtype=torch.bfloat16)],
            act_context_scale=self.config.action_scale,
            act_context=build_action_context(keys, height=first.shape[-2] * 16, width=first.shape[-1] * 16,
                                            device='cuda', dtype=torch.bfloat16))
        sigmas = euler_sigmas(steps, shift=5.0, device=torch.device('cuda'), dtype=torch.bfloat16)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            if fastpath:
                # This object is never retained across windows or changes of input.
                with self.model.model.inference_conditioning_cache() as cache:
                    flow_model = partial(self.model, velocity_only=True,
                                         inference_conditioning_cache=cache)
                    current = action.euler_rollout(flow_model, first=condition,
                        noise=noise.to(device='cuda', dtype=torch.bfloat16),
                        conditions=conditions, sigmas=sigmas)
            else:
                current = action.euler_rollout(self.model, first=condition,
                    noise=noise.to(device='cuda', dtype=torch.bfloat16),
                    conditions=conditions, sigmas=sigmas)
        return current.detach().cpu()

    def decode(self, latent):
        torch = self.torch
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            return self.vae.decode_to_pixel(latent.to(device='cuda', dtype=torch.bfloat16),
                                           use_cache=False, return_in_cpu=True)

    def close(self):
        # Ownership ends here; dropping modules avoids a pointless full CPU copy.
        self.model = self.vae = None
        gc.collect()
        if self.torch.cuda.is_initialized():
            self.torch.cuda.empty_cache()


class ReusableActionEngine:
    """One bounded-session model load, sequential independent runs.

    ``state``/``model_config`` must come from ``load_action_inputs``. They are not
    mutated. ``phase`` preserves the old model/VAE transfers; ``co_resident``
    keeps both on CUDA and propagates OOM without changing modes.

    ``on_chunk(frames, offset)`` receives uint8 RGB after those frames have been
    sent to the raw writer. The initial frame is a separate callback at offset
    zero, then future windows at 1/97/193 (window6). The raw MP4 is finalized only
    at writer.close; the callback must encode/finalize its own playable segment.
    ``_runtime`` and ``_writer_factory`` are CPU testing hooks only.
    """

    def __init__(self, state, model_config, *, residency='phase', fastpath=False,
                 width=None, height=None,
                 _runtime=None, _writer_factory=None):
        require(residency in ('phase', 'co_resident'), 'unknown residency mode')
        require(type(fastpath) is bool, 'fastpath must be an explicit boolean')
        self.width = action.WIDTH if width is None else width
        self.height = action.HEIGHT if height is None else height
        if width is not None or height is not None:
            require(type(self.width) is int and type(self.height) is int
                    and 32 <= self.width <= action.WIDTH and 32 <= self.height <= action.HEIGHT
                    and self.width % 32 == self.height % 32 == 0,
                    'preview dimensions must be positive multiples of32, at most832x480')
        self.resolution_reduced = (self.width, self.height) != (action.WIDTH, action.HEIGHT)
        self.residency = residency
        self.fastpath = fastpath
        self.action_scale = model_config.action_scale
        self._lock = threading.Lock()
        self._closed = False
        self._runs = 0
        self.last_run_report = None
        started = time.monotonic()
        self._runtime = _runtime if _runtime is not None else _WanRuntime(state, model_config, residency)
        self.load_seconds = time.monotonic() - started
        if _writer_factory is None:
            from training.eval.rollout15s import ffmpeg_writer_factory
            _writer_factory = ffmpeg_writer_factory
        self._writer_factory = _writer_factory

    def _timed(self, name, call):
        self._runtime.synchronize()
        started = time.monotonic()
        status = 'failed'
        try:
            value = call()
            self._runtime.synchronize()
            status = 'completed'
            return value
        finally:
            elapsed = time.monotonic() - started
            self._timings.append(dict(stage=name, elapsed_seconds=elapsed, status=status))

    def _encode(self, pixels, index):
        self._timed('transfer_to_vae', lambda: self._runtime.transfer('vae'))
        return self._timed('encode_initial' if index < 0 else 'encode_endpoint',
                           lambda: self._runtime.encode(pixels))

    def _generate(self, first, prompt, keys, noise, index):
        self._timed('transfer_to_model', lambda: self._runtime.transfer('model'))
        options = {'fastpath': True} if self.fastpath else {}
        return self._timed('sampling_and_conditioning',
            lambda: self._runtime.generate(first, prompt, keys, noise, self._steps, **options))

    def _decode(self, latent, index):
        self._timed('transfer_to_vae', lambda: self._runtime.transfer('vae'))
        pixels = self._timed('decode', lambda: self._runtime.decode(latent))
        # A contiguous last-frame view can still retain the entire decoded window.
        self._endpoint = pixels[:, -1:].detach().permute(0, 2, 1, 3, 4).float().cpu().contiguous().clone()
        return pixels

    def run(self, initial, prompt, actions, seed, output_path, *, steps=40,
            window6=True, on_chunk=None):
        """Return JSON-serializable timing/identity metadata for exactly 241 frames."""
        require(not self.resolution_reduced, 'reduced resolution is allowed only for run_segment')
        require(type(window6) is bool, 'window6 must be an explicit boolean')
        return self._run(initial, prompt, actions, seed, output_path, steps=steps,
                         window6=window6, on_chunk=on_chunk, segment=False)[0]

    def run_segment(self, initial, prompt, actions, seed, output_path, *, steps=40,
                    on_chunk=None):
        """Return (report, final_float_rgb) for 48 or 96 future frames.

        The CPU float32 endpoint has shape [1,3,1,H,W], range [-1,1]. Pass it as
        the next segment's initial condition to avoid a uint8 feedback roundtrip.
        Each segment independently VAE-encodes that RGB; it is not world state or
        a retained Transformer/VAE history. Its first displayed frame is uint8.
        Explicit smaller width/height applies only here. The caller must supply
        an already resized initial condition; this method never silently resizes.
        """
        return self._run(initial, prompt, actions, seed, output_path, steps=steps,
                         window6=False, on_chunk=on_chunk, segment=True)

    def _run(self, initial, prompt, actions, seed, output_path, *, steps, window6,
             on_chunk, segment):
        import torch

        _validate_steps(steps)
        require(type(seed) is int and 0 <= seed <= 2**32 - 1, 'invalid seed')
        require(on_chunk is None or callable(on_chunk), 'on_chunk must be callable')
        require(isinstance(actions, np.ndarray) and actions.ndim == 2
                and actions.shape[1] == 8 and actions.dtype == np.float32
                and np.isfinite(actions).all() and np.isin(actions, [0, 1]).all(),
                'actions must be finite binary float32 [N,8]')
        require(len(actions) in ((48, 96) if segment else (240,)), 'unsupported action frame count')
        require(torch.is_tensor(prompt) and prompt.ndim == 2 and 1 <= prompt.shape[0] <= 512
                and prompt.shape[1] == 4096 and prompt.is_floating_point()
                and bool(torch.isfinite(prompt).all()), 'invalid precomputed prompt embedding')
        if segment and torch.is_tensor(initial):
            require(initial.device.type == 'cpu' and initial.is_floating_point()
                    and tuple(initial.shape) == (1, 3, 1, self.height, self.width)
                    and bool(torch.isfinite(initial).all()) and initial.min() >= -1 and initial.max() <= 1,
                    'segment float RGB condition must be finite CPU [1,3,1,H,W] in [-1,1]')
            pixel = initial.detach().float().clone()
            initial_rgb = pixel[0, :, 0].permute(1, 2, 0).add(1).mul(127.5).round().clamp(0, 255).byte().numpy()
            initial_kind = 'previous_generated_float_RGB_reencoded'
        else:
            require(isinstance(initial, np.ndarray) and initial.shape == (self.height, self.width, 3)
                    and initial.dtype == np.uint8, 'initial RGB must be uint8 [H,W,3]')
            initial_rgb = initial.copy()
            pixel = torch.from_numpy(initial_rgb).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).float().div(127.5).sub(1)
            initial_kind = 'submitted_RGB_encoded'
        output_path = Path(output_path)
        require(not output_path.exists(), 'refusing to overwrite an existing video')
        require(self._lock.acquire(blocking=False), 'engine already running or closing')
        writer = None
        try:
            require(not self._closed, 'engine is closed')
            self._runs += 1
            self._steps, self._timings, self._endpoint = steps, [], None
            started = time.monotonic()
            report = dict(outcome='failed', run_index=self._runs, model_reused=self._runs > 1,
                model_load_count=1, load_seconds=self.load_seconds,
                load_seconds_scope='one_time_engine_construction_excluded_from_run_time',
                residency=self.residency, sampler=dict(solver='flow_euler', steps=steps, shift=5.0, cfg='none'),
                fastpath=self.fastpath,
                fastpath_components=['window_action_features', 'window_text_projection', 'velocity_only'] if self.fastpath else [],
                experimental_steps=steps != 40, is_causal=False, realtime=False,
                action_scale=self.action_scale, trainable_parameters=self._runtime.trainable_parameters,
                t5_loaded=False, timing_kind=self._runtime.timing_kind,
                elapsed_seconds_scope='run_only_excludes_engine_construction_and_close',
                fps=16, width=self.width, height=self.height,
                resolution_reduced=self.resolution_reduced,
                reference_resolution=dict(width=action.WIDTH, height=action.HEIGHT),
                same_resolution_comparison=not self.resolution_reduced,
                quality_evaluation='not_run', initial_condition=initial_kind, world_state=False)
            self.last_run_report = report
            try:
                self._runtime.begin()
                writer = self._timed('raw_writer_open', lambda: self._writer_factory(output_path, self.width, self.height, 16))
                first = self._encode(pixel, -1)
                if self.resolution_reduced:
                    require(tuple(first.shape) == (1, 1, 48, self.height // 16, self.width // 16),
                            'preview VAE latent geometry differs from explicit dimensions')

                def emit(frames, offset):
                    self._timed('raw_write', lambda: writer.write(frames))
                    if on_chunk is not None:
                        self._timed('chunk_callback', lambda: on_chunk(frames, offset))

                options = dict(initial_latent=first, initial_rgb=initial_rgb, prompt=prompt,
                    actions=actions.copy(), seed=seed, generate=self._generate,
                    decode=self._decode, emit=emit)
                if segment:
                    generated = self._segment_rollout(**options)
                else:
                    generated = action.rollout_chunks(**options, encode=self._encode, window6=window6)
                self._timed('raw_finalize', writer.close)
                report.update(generated, outcome='completed',
                    method='action_latency_segment_v1' if segment else 'action_latency_full_v1',
                    reference_method=generated.get('method'))
            except BaseException as error:
                report.update(error_type=type(error).__name__, error=str(error))
                if writer is not None:
                    try:
                        writer.abort()
                    except Exception:
                        pass  # Preserve the actual inference/callback error.
                raise
            finally:
                report['elapsed_seconds'] = time.monotonic() - started
                report['stage_timings'] = list(self._timings)
                totals = {}
                for row in self._timings:
                    totals[row['stage']] = totals.get(row['stage'], 0.0) + row['elapsed_seconds']
                report['stage_totals_seconds'] = totals
                report['peak_vram_bytes'] = self._runtime.peak_bytes()
            return report, self._endpoint
        finally:
            self._lock.release()

    def _segment_rollout(self, *, initial_latent, initial_rgb, prompt, actions, seed, generate, decode, emit):
        import torch

        count = len(actions)
        require(initial_latent.ndim == 5 and tuple(initial_latent.shape[:3]) == (1, 1, 48)
                and min(initial_latent.shape[-2:]) > 0, 'invalid initial latent shape')
        latent_height, latent_width = initial_latent.shape[-2:]
        noise_shape = (1, 13, 48, latent_height, latent_width)
        rng = torch.Generator(device='cpu').manual_seed(seed)
        if count == 96:
            noises = [torch.randn(noise_shape, generator=rng, dtype=torch.float32, device='cpu')
                      for _ in range(5)]
            # Same five-draw mapping as window6, with resolution-specific shapes.
            mapped = torch.cat([noises[0][:, :1], *[value[:, 1:] for value in noises]], dim=1)
            noise = mapped[:, :25].clone()
            noise_policy = 'independent_segment_seed;first_window_of_window6_five_CPU_FP32_draws'
        else:
            noise = torch.randn(noise_shape, generator=rng, dtype=torch.float32, device='cpu')
            noise_policy = 'independent_segment_seed;one_CPU_FP32_draw'
        first = initial_latent.detach().cpu().to(torch.bfloat16).clone()
        latent = generate(first, prompt, torch.from_numpy(actions.copy()).unsqueeze(0), noise, 0)
        require(tuple(latent.shape) == (1, count // 4 + 1, 48, latent_height, latent_width)
                and bool(torch.isfinite(latent).all()) and torch.equal(latent[:, :1].cpu(), first.to(latent.dtype)),
                'invalid segment latent or modified conditioning latent')
        pixels = decode(latent, 0)
        require(tuple(pixels.shape) == (1, count + 1, 3, self.height, self.width)
                and bool(torch.isfinite(pixels).all()) and pixels.min() >= -1 and pixels.max() <= 1,
                'invalid segment decoded pixels')
        emit(initial_rgb[None], 0)
        rgb = pixels[0, 1:].permute(0, 2, 3, 1).add(1).mul(127.5).round().clamp(0, 255).byte().numpy()
        emit(rgb, 1)
        return dict(frames_written=count + 1, future_frames=count, fps=16,
                    future_duration_seconds=count / 16, noise_policy=noise_policy,
                    latent_height=latent_height, latent_width=latent_width,
                    noise_comparison='same_geometry_only;different_resolutions_use_different_tensor_shapes',
                    first_latent_sha256_float32=action.tensor_sha(first),
                    noise_sha256_float32=action.tensor_sha(noise),
                    continuation='independent_RGB_condition;no_retained_temporal_cache')

    def close(self):
        require(self._lock.acquire(blocking=False), 'cannot close a running engine')
        try:
            if not self._closed:
                try:
                    self._runtime.close()
                finally:
                    self._closed = True
        finally:
            self._lock.release()

    def __enter__(self):
        require(not self._closed, 'engine is closed')
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
