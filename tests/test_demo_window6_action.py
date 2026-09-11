"""CPU6+6+3s candidate contracts; not model/quality evidence."""
import copy

import numpy as np
import pytest
import torch

from training.demo import action_backend as action
from training.demo.contracts import (ACTION_ADAPTER, ACTION_JOINT_ADAPTER, ACTION_STAGE,
                                     ACTION_WINDOW6_METHOD, action_contract)


def test_window6_contract_rejects_mixed_modes_and_short_action_geometry():
    adapter, geometry = action_contract(ACTION_WINDOW6_METHOD)
    config = dict(method=ACTION_WINDOW6_METHOD, adapter_factory=adapter, geometry=geometry,
                  lineage={'expected_stage': ACTION_STAGE},
                  sampler=dict(solver='flow_euler', steps=40, shift=5.0, cfg='none'))
    action.validate_action_config(config)
    assert geometry['chunk_latent_frames'] == [25, 25, 13]
    for updates in ({'adapter_factory': ACTION_ADAPTER}, {'adapter_factory': ACTION_JOINT_ADAPTER},
                    {'geometry': {**geometry, 'chunk_future_frames': [48, 48, 48]}},
                    {'geometry': {**geometry, 'joint_latent_frames': 61}}):
        changed = copy.deepcopy(config)
        changed.update(updates)
        with pytest.raises(ValueError):
            action.validate_action_config(changed)


def test_window6_noise_boundaries_raw_rgb_feedback_and_241_emitted_frames(monkeypatch):
    monkeypatch.setattr(action, 'HEIGHT', 2)
    monkeypatch.setattr(action, 'WIDTH', 2)
    initial = np.full((2, 2, 3), 17, dtype=np.uint8)
    first = torch.zeros(1, 1, 48, 30, 52, dtype=torch.bfloat16)
    actions = np.zeros((240, 8), dtype=np.float32)
    actions[:96, 0], actions[96:192, 7], actions[192:, 2] = 1, 1, 1
    rng = torch.Generator(device='cpu').manual_seed(42)
    expected_noise = action.joint_noise_from_chunks(torch.stack([
        torch.randn((1, 13, 48, 30, 52), generator=rng) for _ in range(5)]))
    plans = [(0, 96, 25), (96, 192, 25), (192, 240, 13)]
    generated, encoded, emitted = [], [], []

    def generate(condition, prompt, keys, noise, index):
        start, stop, latent_frames = plans[index]
        assert bool((condition == index).all())
        assert noise.shape == (1, latent_frames, 48, 30, 52)
        assert torch.equal(noise[:, :1], expected_noise[:, :1])
        assert torch.equal(noise[:, 1:], expected_noise[:, 1 + start // 4:1 + stop // 4])
        np.testing.assert_array_equal(keys[0].numpy(), actions[start:stop])
        generated.append(index)
        result = noise.clone()
        result[:, :1] = condition.float()
        return result

    def decode(latent, index):
        start, stop, _ = plans[index]
        pixels = torch.zeros(1, stop - start + 1, 3, 2, 2)
        pixels[:, -1] = .123 + index / 10
        return pixels

    def encode(endpoint, index):
        assert endpoint.shape == (1, 3, 1, 2, 2)
        assert torch.allclose(endpoint, torch.full_like(endpoint, .123 + index / 10))
        encoded.append(endpoint.clone())
        return torch.full_like(first, index + 1)

    report = action.rollout_chunks(initial_latent=first, initial_rgb=initial,
        prompt=torch.ones(2, 4096), actions=actions, seed=42, generate=generate,
        decode=decode, encode=encode, emit=lambda frames, offset: emitted.append((offset, frames.copy())),
        window6=True)
    assert generated == [0, 1, 2] and len(encoded) == 2
    assert [offset for offset, _ in emitted] == [0, 1, 97, 193]
    assert [len(frames) for _, frames in emitted] == [1, 96, 96, 48]
    np.testing.assert_array_equal(emitted[0][1][0], initial)
    assert report['frames_written'] == 241 and report['method'] == ACTION_WINDOW6_METHOD
    assert [row['emitted_frames'] for row in report['chunks']] == [97, 96, 48]
    assert [row['action_start'] for row in report['chunks']] == [0, 96, 192]
    assert [row['action_stop'] for row in report['chunks']] == [96, 192, 240]
    assert report['frozen_noise_regression'] is False and report['realtime'] is False
    assert report['continuation'] == 'previous_generated_float_RGB_before_uint8_reencoded'
