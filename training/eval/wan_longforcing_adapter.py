"""Stage-3 streaming RGB output with the exact training window/Euler rollout.

The transformer recomputes recent generated latents, rather than carrying KV
representations that contain evicted history. Only the VAE keeps a stream cache.
"""

from __future__ import annotations

from typing import Any

from training.eval.wan_causal_adapter import LONGFORCING_STAGE, WanCausalRolloutAdapter
from training.models.action_adapter import validate_action_scale
from training.longforcing_lite import (
    LongForcingConfig,
    WanLongForcingWindowStudent,
    _solve_block,
    euler_sigmas,
)


class WanLongForcingRolloutAdapter(WanCausalRolloutAdapter):
    context_mode = "sliding_window_recompute"

    def __init__(self, *, longforcing_config: LongForcingConfig, **kwargs: Any):
        requested_scale = validate_action_scale(
            kwargs.pop("action_scale", longforcing_config.model.action_scale)
        )
        if requested_scale != longforcing_config.model.action_scale:
            raise ValueError("LongForcing inference action_scale must match the saved configuration")
        kwargs["action_scale"] = longforcing_config.model.action_scale
        super().__init__(**kwargs)
        if self.checkpoint_stage != LONGFORCING_STAGE:
            raise ValueError("window-recompute inference requires a LongForcing checkpoint")
        if (longforcing_config.data.height, longforcing_config.data.width) != (self.height, self.width):
            raise ValueError("LongForcing inference must match the trained spatial geometry")
        if (longforcing_config.model.num_frame_per_block, longforcing_config.rollout.student_steps) != (3, 4):
            raise ValueError("LongForcing inference requires the trained 3-latent / 4-Euler-step contract")
        self.longforcing_config = longforcing_config
        # This helper owns only the already-loaded student, never a teacher.
        self._window_student = WanLongForcingWindowStudent(
            student=self.pipeline.generator, config=longforcing_config
        )
        self._latent_history = None
        self._action_blocks = []
        self._window_conditions = {}
        # The inherited pipeline is only a model/VAE/text container in this mode.
        # Report the exact BF16 schedule used by training, not its legacy table.
        sigmas = euler_sigmas(
            4, shift=longforcing_config.model.timestep_shift,
            device=self.torch.device("cpu"), dtype=self.torch.bfloat16,
        )
        self.pipeline.denoising_step_list = (sigmas[:-1] * 1000.0).float()
        self.pipeline.args.context_mode = self.context_mode

    def _initialize_generation(self, initial_latent: Any) -> None:
        # Do not call reset_stream/_prime_transformer: either would allocate KV.
        if self.pipeline.kv_cache1 is not None or self.pipeline.crossattn_cache is not None:
            raise RuntimeError("LongForcing window-recompute stream must not retain transformer KV")
        self._latent_history = initial_latent.detach().clone()
        self._action_blocks = []
        self._window_conditions = {
            "prompt_embeds": self.pipeline.conditional_dict["prompt_embeds"]
        }
        self.pipeline.current_start_frame = 1
        self.pipeline.num_input_frames = 0
        self.pipeline.conditional_dict["first_frame_latents"] = None

    def _context_position(self) -> int:
        if self._latent_history is None:
            raise RuntimeError("LongForcing generated-latent history is not initialized")
        return int(self._latent_history.shape[1]) * self.pipeline.frame_seq_length

    def _generate_latent_block(self, noise: Any, action_tensor: Any, chunk_index: int):
        if self._latent_history is None or len(self._action_blocks) != chunk_index:
            raise RuntimeError("LongForcing history/action cursor is not sequential")
        if self._latent_history.shape[1] != 1 + 3 * chunk_index:
            raise RuntimeError("LongForcing latent history does not match the current chunk")
        torch = self.torch
        all_actions = torch.stack([*self._action_blocks, action_tensor], dim=1)
        self._window_conditions["all_block_actions"] = all_actions
        with torch.no_grad():
            latent_block = _solve_block(
                self._window_student.student_velocity,
                initial_noise=noise,
                history=self._latent_history,
                action_block=action_tensor,
                block_index=chunk_index,
                conditions=self._window_conditions,
                steps=self.longforcing_config.rollout.student_steps,
                timestep_shift=self.longforcing_config.model.timestep_shift,
            )
        self._latent_history = torch.cat((self._latent_history, latent_block.detach()), dim=1).detach()
        self._action_blocks.append(action_tensor.detach())
        self.pipeline.current_start_frame = int(self._latent_history.shape[1])
        return latent_block
