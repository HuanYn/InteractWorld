"""Small CPU checks of shared training/streaming history and stage dispatch."""

from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from torch import nn

from training.eval.rollout15s import SceneSpec
from training.eval.wan_causal_adapter import (
    EXPECTED_STAGE, LONGFORCING_STAGE, PINNED_BASE_MODEL,
    WanCausalRolloutAdapter, create_wan_causal_adapter,
)
from training.eval.wan_longforcing_adapter import WanLongForcingRolloutAdapter
from training.longforcing_lite import (
    WanLongForcingBackend, WanLongForcingWindowStudent, _solve_block,
    euler_sigmas, load_longforcing_config, longforcing_config_from_dict,
)
from training.models.action_adapter import build_action_context


CONFIG_PATH = Path(__file__).parents[1] / "configs/train/longforcing_lite_5090_week.yaml"


class _Backbone(nn.Module):
    patch_size = (1, 2, 2)

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, **kwargs):
        self.calls.append((x.clone(), kwargs, torch.is_grad_enabled()))
        assert "kv_cache" not in kwargs
        return x * 0.125 + 0.5


class _VAE:
    def __init__(self):
        self.model = SimpleNamespace(clear_cache=Mock())
        self.decode_calls = []

    def encode_to_latent(self, pixel):
        return torch.full((1, 1, 48, 2, 4), 2.0, dtype=torch.bfloat16)

    def decode_to_pixel(self, latent, *, use_cache, return_in_cpu):
        self.decode_calls.append((latent.clone(), use_cache))
        frames = 1 if latent.shape[1] == 1 else 12
        return torch.zeros(1, frames, 3, 32, 64)


class _Text:
    def __init__(self):
        self.calls = 0
        self.prompt = torch.arange(8, dtype=torch.bfloat16).reshape(1, 2, 4)

    def to(self, **kwargs):
        return self

    def __call__(self, **kwargs):
        self.calls += 1
        return {"prompt_embeds": self.prompt}


def _fixture():
    config = load_longforcing_config(CONFIG_PATH)
    config.data.height, config.data.width = 32, 64
    pipeline = SimpleNamespace(
        args=SimpleNamespace(streaming_solver="flow_euler"),
        generator=SimpleNamespace(model=_Backbone()), vae=_VAE(), text_encoder=_Text(),
        kv_cache1=None, crossattn_cache=None, frame_seq_length=2,
        reset_stream=Mock(side_effect=AssertionError("must not allocate KV")),
        generate_next_block=Mock(side_effect=AssertionError("must use training solver")),
    )
    adapter = WanLongForcingRolloutAdapter(
        pipeline=pipeline, torch_module=torch, device="cpu",
        checkpoint_path="unused", checkpoint_sha256="unused", checkpoint_stage=LONGFORCING_STAGE,
        width=64, height=32, longforcing_config=config,
    )
    return adapter, pipeline, config


def test_shared_training_methods_and_exact_four_step_endpoint():
    adapter, pipeline, config = _fixture()
    assert WanLongForcingBackend.student_velocity is WanLongForcingWindowStudent.student_velocity
    assert WanLongForcingBackend._window_inputs is WanLongForcingWindowStudent._window_inputs
    assert not hasattr(adapter._window_student, "teacher")
    initial = torch.full((1, 1, 48, 2, 4), 2.0, dtype=torch.bfloat16)
    pipeline.conditional_dict = {"prompt_embeds": pipeline.text_encoder.prompt}
    adapter._initialize_generation(initial)
    noise = torch.linspace(-1, 1, 3 * 48 * 2 * 4).reshape(1, 3, 48, 2, 4).bfloat16()
    actions = torch.zeros(1, 12, 8, dtype=torch.bfloat16)
    actions[:, :, 0] = 1
    conditions = {"prompt_embeds": pipeline.text_encoder.prompt, "all_block_actions": actions[:, None]}
    shared = WanLongForcingWindowStudent(student=SimpleNamespace(model=_Backbone()), config=config)
    with torch.no_grad():
        expected = _solve_block(
            shared.student_velocity, initial_noise=noise.clone(), history=initial,
            action_block=actions, block_index=0, conditions=conditions,
            steps=4, timestep_shift=config.model.timestep_shift,
        )
    actual = adapter._generate_latent_block(noise.clone(), actions, 0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    sigma = euler_sigmas(4, shift=5.0, device=torch.device("cpu"), dtype=torch.bfloat16)
    torch.testing.assert_close(pipeline.denoising_step_list, (sigma[:-1] * 1000).float())
    assert len(pipeline.generator.model.calls) == 4


def test_stream_crosses_four_to_five_block_boundary_without_kv_or_future_data():
    adapter, pipeline, config = _fixture()
    scene = SceneSpec("toy", "shared prompt", "unused", "never-read", 42, ())
    initial = np.zeros((32, 64, 3), dtype=np.uint8)
    cursor = adapter.begin(scene=scene, initial_frame=initial, variant="preview", seed=42)
    all_actions = []
    for block in range(5):
        history = adapter._latent_history.clone()
        actions = np.zeros((12, 8), dtype=np.float32)
        actions[:, block] = 1
        all_actions.append(torch.from_numpy(actions).bfloat16())
        chunk = adapter.generate_next(
            cursor=cursor, actions=actions, chunk_index=block, latent_frames=3, rgb_frames=12,
        )
        assert chunk.frames.shape == (12, 32, 64, 3)
        assert chunk.parent_state_token == cursor.state_token
        cursor = chunk.cursor
        x, kwargs, grad_enabled = pipeline.generator.model.calls[block * 4]
        assert not grad_enabled
        assert len(pipeline.generator.model.calls) == (block + 1) * 4
        expected_history = history if block < 4 else history[:, -9:]
        torch.testing.assert_close(x[:, :, :-3].permute(0, 2, 1, 3, 4), expected_history)
        expected_frames = 1 + (block + 1) * 3 if block < 4 else 12
        assert x.shape[2] == expected_frames
        assert kwargs["current_start"] == (0 if block < 4 else 4 * pipeline.frame_seq_length)
        assert torch.all(kwargs["t"][:, :-3] == 0)
        first_block = 0 if block < 4 else block - 3
        action_frames = torch.stack(all_actions[first_block:]).flatten(0, 1).unsqueeze(0)
        expected_context = build_action_context(
            action_frames, height=32, width=64, device="cpu", dtype=torch.bfloat16,
        )
        torch.testing.assert_close(kwargs["act_context"][0], expected_context[0])
        torch.testing.assert_close(kwargs["context"][0], pipeline.text_encoder.prompt[0])
        assert len(adapter._window_conditions["_longforcing_action_context_cache"]) == 1
        assert pipeline.kv_cache1 is None and pipeline.crossattn_cache is None
    assert adapter._latent_history.shape[1] == 16
    assert len(pipeline.vae.decode_calls) == 6
    assert all(cached for _, cached in pipeline.vae.decode_calls)
    pipeline.reset_stream.assert_not_called()
    pipeline.generate_next_block.assert_not_called()
    assert adapter.context_mode == "sliding_window_recompute"
    old_cursor = cursor
    adapter.begin(scene=scene, initial_frame=initial, variant="preview", seed=42)
    assert adapter._latent_history.shape[1] == 1
    assert adapter._action_blocks == []
    assert pipeline.text_encoder.calls == 1  # Same shared prompt cache, new VAE/session state.
    with pytest.raises(RuntimeError, match="stale"):
        adapter.generate_next(cursor=old_cursor, actions=actions, chunk_index=5, latent_frames=3, rgb_frames=12)


def test_factory_dispatch_keeps_causal_full_kv_and_long_window_only():
    def namespace(value):
        return SimpleNamespace(**{key: namespace(item) for key, item in value.items()}) if isinstance(value, dict) else value

    class Component:
        def __init__(self):
            self.model = SimpleNamespace()

        def to(self, **kwargs):
            return self

    class FakePipeline:
        def __init__(self, config, device):
            self.args = config
            self.generator, self.vae, self.text_encoder = Component(), Component(), Component()
            self.kv_cache1 = self.crossattn_cache = None

        def requires_grad_(self, enabled):
            return self

        def eval(self):
            return self

    fake_pipeline_module = ModuleType("pipeline.causal_inference")
    fake_pipeline_module.CausalInferencePipeline = FakePipeline
    fake_omegaconf = ModuleType("omegaconf")
    fake_omegaconf.OmegaConf = SimpleNamespace(create=namespace)
    config = load_longforcing_config(CONFIG_PATH)
    assert longforcing_config_from_dict(config.to_dict()).to_dict() == config.to_dict()
    for stage in (EXPECTED_STAGE, LONGFORCING_STAGE):
        payload = {"stage": stage, "config": config.to_dict(), "trainable_model": {"toy": torch.zeros(1)}}
        with patch.dict("sys.modules", {"pipeline.causal_inference": fake_pipeline_module, "omegaconf": fake_omegaconf}), \
             patch("training.eval.wan_causal_adapter.sha256_file", return_value="abc"), \
             patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch, "load", return_value=payload), \
             patch("training.models.lora.configure_action_teacher"), \
             patch("training.models.lora.load_trainable_state_dict"):
            adapter = create_wan_causal_adapter(
                checkpoint_path="unused", checkpoint_sha256="abc", base_model_path=PINNED_BASE_MODEL, device="cuda",
            )
        if stage == EXPECTED_STAGE:
            assert type(adapter) is WanCausalRolloutAdapter
            assert adapter.context_mode == "full_history_kv"
            assert adapter.pipeline.args.image_or_video_shape[1] == 61
        else:
            assert type(adapter) is WanLongForcingRolloutAdapter
            assert adapter.pipeline.args.image_or_video_shape[1] == 13
        assert adapter.pipeline.kv_cache1 is None and adapter.pipeline.crossattn_cache is None
