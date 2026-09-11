"""CPU contracts/scheduling only; these fakes establish no video quality."""
import copy

import numpy as np
import pytest
import torch

from training.demo import action_backend as action
from training.demo.contracts import (ACTION_ADAPTER, ACTION_JOINT_ADAPTER, ACTION_JOINT_METHOD,
                                     ACTION_METHOD, ACTION_STAGE, action_contract)


def config_for(method):
    adapter, geometry = action_contract(method)
    return dict(method=method, adapter_factory=adapter, geometry=geometry,
                lineage={'expected_stage': ACTION_STAGE},
                sampler=dict(solver='flow_euler', steps=40, shift=5.0, cfg='none'))


def test_joint_contract_is_opt_in_and_rejects_mixed_methods():
    legacy, joint = config_for(ACTION_METHOD), config_for(ACTION_JOINT_METHOD)
    action.validate_action_config(legacy)
    action.validate_action_config(joint)
    assert legacy['adapter_factory'] == ACTION_ADAPTER and legacy['geometry']['num_chunks'] == 5
    assert joint['adapter_factory'] == ACTION_JOINT_ADAPTER
    assert joint['geometry']['joint_latent_frames'] == 61 and 'num_chunks' not in joint['geometry']
    for updates in ({'adapter_factory': ACTION_ADAPTER}, {'geometry': legacy['geometry']},
                    {'method': 'browser_invented_mode'},
                    {'geometry': {**joint['geometry'], 'num_chunks': 5}},
                    {'sampler': dict(solver='flow_euler', steps=4, shift=5.0, cfg='none')}):
        changed = copy.deepcopy(joint)
        changed.update(updates)
        with pytest.raises(ValueError):
            action.validate_action_config(changed)


def test_joint_noise_preserves_all_sixty_future_slots_and_never_changes_source():
    noises = torch.arange(5 * 13 * 48, dtype=torch.float32).reshape(5, 1, 13, 48, 1, 1)
    before = noises.clone()
    joint = action.joint_noise_from_chunks(noises)
    assert joint.shape == (1, 61, 48, 1, 1)
    assert torch.equal(joint[:, :1], noises[0, :, :1])
    for index in range(5):
        assert torch.equal(joint[:, 1 + 12 * index:13 + 12 * index], noises[index, :, 1:])
    assert torch.equal(noises, before)
    for invalid in (noises.to(torch.bfloat16), noises[:4], noises + float('nan')):
        with pytest.raises(ValueError):
            action.joint_noise_from_chunks(invalid)


def test_joint_scheduling_one_solve_full_actions_and_continuous_decode(monkeypatch):
    monkeypatch.setattr(action, 'HEIGHT', 2)
    monkeypatch.setattr(action, 'WIDTH', 2)
    initial = np.full((2, 2, 3), 17, dtype=np.uint8)
    first = torch.full((1, 1, 48, 30, 52), 3, dtype=torch.bfloat16)
    actions = np.zeros((240, 8), dtype=np.float32)
    actions[:48, 0], actions[192:, 7] = 1, 1
    prompt = torch.ones(2, 4096)
    before_actions, before_first = actions.copy(), first.clone()
    generated, decoded, emitted = [], [], []

    def generate(condition, text, keys, noise, index):
        assert index == 0 and text is prompt
        assert condition.dtype == torch.bfloat16 and noise.shape == (1, 61, 48, 30, 52)
        np.testing.assert_array_equal(keys.numpy()[0], actions)
        rng = torch.Generator(device='cpu').manual_seed(42)
        draws = torch.stack([torch.randn((1, 13, 48, 30, 52), generator=rng) for _ in range(5)])
        assert torch.equal(noise, action.joint_noise_from_chunks(draws))
        generated.append(index)
        result = torch.arange(61, dtype=torch.float32).reshape(1, 61, 1, 1, 1).expand_as(noise).clone()
        result[:, :1] = condition.float()
        return result

    def decode(latent, index):
        assert index == len(decoded)
        if index == 0:
            assert latent.shape[1] == 1 and bool((latent == 3).all())
        else:
            assert latent.shape[1] == 3
            assert list(latent[0, :, 0, 0, 0]) == list(range(1 + (index - 1) * 3, 4 + (index - 1) * 3))
        decoded.append(index)
        return torch.zeros(1, 1 if index == 0 else 12, 3, 2, 2)

    result = action.rollout_joint(initial_latent=first, initial_rgb=initial, prompt=prompt, actions=actions,
                                  seed=42, generate=generate, decode=decode,
                                  emit=lambda frames, offset: emitted.append((offset, frames.copy())))
    assert generated == [0] and len(decoded) == 21
    assert [offset for offset, _ in emitted] == [0, *range(1, 241, 12)]
    assert sum(len(frames) for _, frames in emitted) == 241
    np.testing.assert_array_equal(emitted[0][1][0], initial)
    np.testing.assert_array_equal(actions, before_actions)
    assert torch.equal(first, before_first)
    assert result['method'] == ACTION_JOINT_METHOD and result['action_rows'] == 240
    assert not result['rgb_endpoint_reencoding'] and not result['frozen_noise_regression']
    assert not result['realtime'] and result['quality_evaluation'] == 'not_run'
    assert all(row['use_cache'] for row in result['decode_stream'])
    assert all(row['previous_generated_cache_retained'] for row in result['decode_stream'][1:])


def test_joint_refuses_sampler_that_changes_condition(monkeypatch):
    first = torch.full((1, 1, 48, 30, 52), 3, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match='clean first latent'):
        action.rollout_joint(initial_latent=first, initial_rgb=np.zeros((480, 832, 3), np.uint8),
                             prompt=torch.ones(2, 4096), actions=np.zeros((240, 8), np.float32), seed=42,
                             generate=lambda first, prompt, keys, noise, index: torch.zeros_like(noise),
                             decode=lambda *args: pytest.fail('invalid latents must not reach the VAE'),
                             emit=lambda *args: pytest.fail('invalid latents must not create video'))
