"""Concrete Wan2.2 streaming adapter for the rollout15s contract.

Imports that construct Wan/T5/VAE models are intentionally delayed until the
factory is called.  The CLI resolves this factory only after the dedicated GPU
gate succeeds.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.eval.rollout15s import (
    EXPECTED_STAGE,
    LATENT_FRAMES_PER_CHUNK,
    PINNED_BASE_MODEL,
    RGB_FRAMES_PER_CHUNK,
    CausalChunk,
    RolloutCursor,
    SceneSpec,
)
from training.runtime import sha256_file
from training.paths import is_pinned_base_model

LONGFORCING_STAGE = "longforcing_lite_v1"


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@dataclass(frozen=True)
class _Session:
    identity: str
    generation: int


class WanCausalRolloutAdapter:
    """One-model adapter with a resettable, strictly sequential stream."""

    context_mode = "full_history_kv"

    def __init__(
        self,
        *,
        pipeline: Any,
        torch_module: Any,
        device: Any,
        checkpoint_path: str,
        checkpoint_sha256: str,
        checkpoint_stage: str,
        width: int = 832,
        height: int = 480,
    ) -> None:
        self.pipeline = pipeline
        self.torch = torch_module
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = checkpoint_sha256
        self.checkpoint_stage = checkpoint_stage
        self.width = width
        self.height = height
        self.latent_height = height // 16
        self.latent_width = width // 16
        self.latent_channels = 48
        self._session: _Session | None = None
        self._rng = None
        self._generation = 0
        self._prompt_cache: dict[str, Any] = {}

    def _reset_vae_cache(self) -> None:
        vae = self.pipeline.vae
        if hasattr(vae, "model") and callable(getattr(vae.model, "clear_cache", None)):
            vae.model.clear_cache()
        elif hasattr(vae, "taehv") and callable(getattr(vae.taehv, "reset", None)):
            vae.taehv.reset()
        else:
            raise RuntimeError("selected VAE exposes no resettable streaming cache")

    def _encode_initial(self, initial_frame: np.ndarray):
        torch = self.torch
        pixel = torch.from_numpy(np.ascontiguousarray(initial_frame)).to(
            device=self.device, dtype=torch.bfloat16
        )
        pixel = pixel.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        pixel = pixel.div(127.5).sub(1.0)
        latent = self.pipeline.vae.encode_to_latent(pixel)
        expected = (1, 1, self.latent_channels, self.latent_height, self.latent_width)
        if tuple(latent.shape) != expected:
            raise RuntimeError(f"Wan VAE initial latent must be {expected}, got {tuple(latent.shape)}")
        return latent.to(device=self.device, dtype=torch.bfloat16)

    def _prime_transformer(self, initial_latent: Any) -> None:
        torch = self.torch
        self.pipeline.reset_stream(
            batch_size=1,
            dtype=torch.bfloat16,
            device=self.device,
            initial_latent=initial_latent,
        )
        condition = self.pipeline.get_condition_split(self.pipeline.conditional_dict, 0, 1)
        timestep = torch.zeros((1, 1), device=self.device, dtype=torch.int64)
        # The Wan model uses global grad mode to select training versus KV inference.
        # eval() and frozen parameters alone do not select the inference branch.
        with torch.no_grad():
            self.pipeline.generator(
                noisy_image_or_video=initial_latent,
                conditional_dict=condition,
                timestep=timestep,
                kv_cache=self.pipeline.kv_cache1,
                crossattn_cache=self.pipeline.crossattn_cache,
                current_start=0,
            )
        if self._cache_position() != self.pipeline.frame_seq_length:
            raise RuntimeError("initial frame did not prime exactly one frame of Wan KV cache")
        # The initial frame is now already represented in the real KV cache.
        # Streaming blocks contain future latents only, so disable the helper's
        # first-block replacement path and start at absolute latent frame one.
        self.pipeline.current_start_frame = 1
        self.pipeline.num_input_frames = 0
        self.pipeline.conditional_dict["first_frame_latents"] = None

    def _prime_vae(self, initial_latent: Any) -> None:
        decoded = self.pipeline.vae.decode_to_pixel(
            initial_latent, use_cache=True, return_in_cpu=True
        )
        if decoded.ndim != 5 or decoded.shape[0] != 1:
            raise RuntimeError("VAE failed to prime its temporal cache from the initial latent")

    def _cache_position(self) -> int:
        caches = self.pipeline.kv_cache1
        if not isinstance(caches, list) or not caches:
            raise RuntimeError("Wan causal KV cache is not initialized")
        positions = tuple(int(cache["global_end_index"].item()) for cache in caches)
        if len(set(positions)) != 1:
            raise RuntimeError("Wan transformer blocks disagree on KV cache position")
        return positions[0]

    def _initialize_generation(self, initial_latent: Any) -> None:
        self._prime_transformer(initial_latent)

    def _context_position(self) -> int:
        return self._cache_position()

    def _generate_latent_block(self, noise: Any, action_tensor: Any, chunk_index: int):
        torch = self.torch
        from training.models.action_adapter import pack_canonical_actions

        packed = pack_canonical_actions(action_tensor, device=self.device, dtype=torch.bfloat16)
        context = packed.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        context = context.expand(-1, -1, -1, self.height, self.width).contiguous()
        self.pipeline.conditional_dict["act_context"] = context
        before = self._cache_position()
        latent_block = self.pipeline.generate_next_block(noise)
        if self._cache_position() <= before:
            raise RuntimeError("Wan KV cache did not advance after a generated causal block")
        return latent_block

    def begin(
        self,
        *,
        scene: SceneSpec,
        initial_frame: np.ndarray,
        variant: str,
        seed: int,
    ) -> RolloutCursor:
        torch = self.torch
        if initial_frame.shape != (self.height, self.width, 3) or initial_frame.dtype != np.uint8:
            raise ValueError("Wan adapter received an invalid initial RGB frame")
        self._generation += 1
        identity = f"{scene.scene_id}:{variant}:{seed}:{self._generation}:{uuid.uuid4().hex}"
        self._session = _Session(identity=identity, generation=self._generation)
        self._rng = torch.Generator(device="cpu").manual_seed(seed)
        self._reset_vae_cache()
        if scene.prompt not in self._prompt_cache:
            self.pipeline.text_encoder.to(device=self.device, dtype=torch.bfloat16)
            condition = self.pipeline.text_encoder(
                text_prompts=[scene.prompt], device=self.device
            )
            self._prompt_cache[scene.prompt] = condition["prompt_embeds"].detach()
            self.pipeline.text_encoder.to(device="cpu", dtype=torch.bfloat16)
            torch.cuda.empty_cache()
        self.pipeline.conditional_dict = {
            "prompt_embeds": self._prompt_cache[scene.prompt]
        }
        initial_latent = self._encode_initial(initial_frame)
        self.pipeline.conditional_dict["act_context_scale"] = 1.0
        self.pipeline.conditional_dict["act_context"] = None
        self._initialize_generation(initial_latent)
        self._prime_vae(initial_latent)
        position = self._context_position()
        token = hashlib.sha256(f"{identity}:0:{position}".encode()).hexdigest()
        return RolloutCursor(
            session_id=identity,
            next_chunk_index=0,
            state_token=token,
            last_frame_sha256=_array_sha256(initial_frame),
            opaque=self._session,
        )

    def _check_cursor(self, cursor: RolloutCursor, chunk_index: int) -> _Session:
        if (
            self._session is None
            or cursor.opaque != self._session
            or cursor.session_id != self._session.identity
            or cursor.next_chunk_index != chunk_index
        ):
            raise RuntimeError("stale, interleaved, or out-of-order Wan rollout cursor")
        return self._session

    def generate_next(
        self,
        *,
        cursor: RolloutCursor,
        actions: np.ndarray,
        chunk_index: int,
        latent_frames: int,
        rgb_frames: int,
    ) -> CausalChunk:
        torch = self.torch
        session = self._check_cursor(cursor, chunk_index)
        if (latent_frames, rgb_frames) != (LATENT_FRAMES_PER_CHUNK, RGB_FRAMES_PER_CHUNK):
            raise ValueError("Wan v1 adapter requires 3 latent / 12 RGB frames per chunk")
        if actions.shape != (RGB_FRAMES_PER_CHUNK, 8):
            raise ValueError("Wan action block must be [12,8]")
        if self._rng is None:
            raise RuntimeError("rollout RNG was not initialized")

        action_tensor = torch.from_numpy(np.ascontiguousarray(actions)).unsqueeze(0)
        action_tensor = action_tensor.to(device=self.device, dtype=torch.bfloat16)

        noise = torch.randn(
            (
                1,
                LATENT_FRAMES_PER_CHUNK,
                self.latent_channels,
                self.latent_height,
                self.latent_width,
            ),
            generator=self._rng,
            device="cpu",
            dtype=torch.float32,
        ).to(device=self.device, dtype=torch.bfloat16)
        latent_block = self._generate_latent_block(noise, action_tensor, chunk_index)
        after = self._context_position()

        decoded = self.pipeline.vae.decode_to_pixel(
            latent_block, use_cache=True, return_in_cpu=True
        )
        expected = (1, RGB_FRAMES_PER_CHUNK, 3, self.height, self.width)
        if tuple(decoded.shape) != expected:
            raise RuntimeError(f"streaming VAE must decode exactly 12 frames, got {tuple(decoded.shape)}")
        frames = (
            decoded[0]
            .permute(0, 2, 3, 1)
            .add(1.0)
            .mul(127.5)
            .clamp(0, 255)
            .to(torch.uint8)
            .numpy()
        )
        latent_digest = hashlib.sha256(
            latent_block.detach().to(device="cpu", dtype=torch.float32).numpy().tobytes()
        ).hexdigest()
        token = hashlib.sha256(
            f"{session.identity}:{chunk_index + 1}:{after}:{latent_digest}".encode()
        ).hexdigest()
        next_cursor = RolloutCursor(
            session_id=session.identity,
            next_chunk_index=chunk_index + 1,
            state_token=token,
            last_frame_sha256=_array_sha256(frames[-1]),
            opaque=session,
        )
        return CausalChunk(
            parent_state_token=cursor.state_token,
            cursor=next_cursor,
            frames=frames,
        )


def _denoising_steps(stage: str) -> list[int]:
    if stage == LONGFORCING_STAGE:
        return [1000, 750, 500, 250]
    if stage == EXPECTED_STAGE:
        return list(range(1000, 0, -25))
    raise ValueError(f"unsupported rollout checkpoint stage: {stage!r}")


def create_wan_causal_adapter(
    *,
    checkpoint_path: str,
    checkpoint_sha256: str,
    base_model_path: str,
    device: str,
) -> WanCausalRolloutAdapter:
    """Load the pinned base plus adapter/LoRA state into the real stream."""

    if not is_pinned_base_model(base_model_path):
        raise ValueError("Wan rollout requires the pinned base-model revision")
    if sha256_file(checkpoint_path) != checkpoint_sha256.lower():
        raise ValueError("rollout checkpoint changed after the CPU lineage gate")

    # Delayed heavy imports: this factory is called only after the GPU gate.
    import torch
    from omegaconf import OmegaConf

    from pipeline.causal_inference import CausalInferencePipeline
    from training.models.lora import configure_action_teacher, load_trainable_state_dict

    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the concrete Wan rollout adapter requires CUDA")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("rollout checkpoint must contain a mapping")
    stage = str(payload.get("stage"))
    steps = _denoising_steps(stage)
    longforcing_config = None
    if stage == LONGFORCING_STAGE:
        from training.longforcing_lite import longforcing_config_from_dict

        longforcing_config = longforcing_config_from_dict(payload.get("config", {}))

    base = Path(base_model_path)
    config = OmegaConf.create(
        {
            "model_type": "ci2v",
            "denoising_step_list": steps,
            "warp_denoising_step": True,
            # Our causal checkpoint predicts flow velocity; LongForcing also
            # trains its student through deterministic Euler rollouts. Neither
            # uses the upstream distilled x0/re-noise transition at inference.
            "streaming_solver": "flow_euler",
            "context_noise": 0,
            "image_or_video_shape": [1, 13 if longforcing_config else 61, 48, 30, 52],
            "independent_first_frame": True,
            "num_frame_per_block": 3,
            "ref_num_slots": 0,
            "ref_resolution": 512,
            "model_kwargs": {
                "model_name": str(base),
                "timestep_shift": 5.0,
                "local_attn_size": -1,
                "use_relative_rope": False,
                "downscale_factor_control_adapter": 16,
            },
            "text_encoder_kwargs": {
                "tokenizer_path": str(base / "google" / "umt5-xxl"),
                "encoder_pth_path": str(base / "models_t5_umt5-xxl-enc-bf16.pth"),
            },
            "vae_kwargs": {
                "vae_type": "Wan2.2_VAE",
                "pretrained_path": str(base / "Wan2.2_VAE.pth"),
            },
        }
    )
    pipeline = CausalInferencePipeline(config, device=target)
    pipeline.generator.model.independent_first_frame = True
    model_config = payload.get("config", {}).get("model", {})
    configure_action_teacher(
        pipeline.generator.model,
        rank=int(model_config.get("lora_rank", 16)),
        alpha=float(model_config.get("lora_alpha", 16.0)),
        dropout=float(model_config.get("lora_dropout", 0.0)),
    )
    state = payload.get("trainable_model")
    if not isinstance(state, dict) or not state:
        raise ValueError("rollout checkpoint has no named trainable_model state")
    load_trainable_state_dict(pipeline.generator, state)
    pipeline.requires_grad_(False).eval()

    # Keep T5 on CPU between scenes; prompts are encoded briefly on CUDA.  The
    # causal generator and streaming VAE remain resident to preserve caches.
    pipeline.generator.to(device=target, dtype=torch.bfloat16)
    pipeline.vae.to(device=target, dtype=torch.bfloat16)
    pipeline.text_encoder.to(device="cpu", dtype=torch.bfloat16)
    pipeline.torch_dtype = torch.bfloat16
    if longforcing_config is not None:
        from training.eval.wan_longforcing_adapter import WanLongForcingRolloutAdapter

        return WanLongForcingRolloutAdapter(
            pipeline=pipeline,
            torch_module=torch,
            device=target,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_stage=stage,
            longforcing_config=longforcing_config,
        )
    return WanCausalRolloutAdapter(
        pipeline=pipeline,
        torch_module=torch,
        device=target,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_stage=stage,
    )
