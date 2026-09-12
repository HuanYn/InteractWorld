"""Four small CPU tests for teacher-free history noise, not a GPU smoke suite."""
import copy

import pytest
import torch

from training.causal_tf import CausalTeacherForcingConfig, causal_teacher_forcing_loss, validate_causal_batch
from training.history_noise import (HistoryNoiseConfig, history_noise_contract, history_noise_loss,
                                    max_history_sigma, prepare_history_context)


def sample():
    config = CausalTeacherForcingConfig()
    config.data.num_frames = 97
    config.data.height = config.data.width = 32
    config.model.latent_channels = 2
    clean = torch.arange(200).reshape(1, 25, 2, 2, 2).float() / 200
    target = torch.full_like(clean, .25)
    target[:, 0].zero_()
    t = torch.tensor([[0.] + [731.25] * 24])
    batch = dict(noisy_latents=clean + (t / 1000).view(1, 25, 1, 1, 1) * target,
                 target_flow=target, timesteps=t, actions=torch.zeros(1, 96, 8),
                 prompt_embeds=torch.arange(32).reshape(1, 2, 16).float())
    return config, batch


class FakeCausal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.1))
        self.calls = []

    def forward(self, noisy, **kwargs):
        self.calls.append((noisy.detach().clone(), kwargs))
        output = noisy * self.scale + kwargs['clean_x'] * .01
        return output, noisy


def test_versioned_contract_and_bounded_curriculum():
    config = HistoryNoiseConfig.from_dict({})
    assert max_history_sigma(config, 0) == .05
    assert max_history_sigma(config, 40) == pytest.approx(.1)
    assert max_history_sigma(config, 80) == pytest.approx(.15)
    assert max_history_sigma(config, 200) == pytest.approx(.15)
    assert history_noise_contract(config)['teacher_model'] is False
    assert history_noise_contract(config)['student_rollout'] is False
    with pytest.raises(ValueError, match='unknown'):
        HistoryNoiseConfig.from_dict({'teacher': 'anything'})
    with pytest.raises(ValueError, match='0.15'):
        HistoryNoiseConfig.from_dict({'sigma_end': .5})


def test_deterministic_absolute_cursor_no_global_rng_initial_and_time_preserved():
    config, batch = sample()
    clean = validate_causal_batch(batch, config)
    options = HistoryNoiseConfig(clean_probability=0)
    global_before = torch.get_rng_state().clone()
    first = prepare_history_context(clean, noise_config=options, absolute_microbatch_index=480, stage_optimizer_step=0)
    replay = prepare_history_context(clean, noise_config=options, absolute_microbatch_index=480, stage_optimizer_step=0)
    other = prepare_history_context(clean, noise_config=options, absolute_microbatch_index=481, stage_optimizer_step=0)
    assert torch.equal(global_before, torch.get_rng_state())
    assert torch.equal(first[0], replay[0]) and torch.equal(first[1], replay[1]) and first[2] == replay[2]
    assert not torch.equal(first[0][:, 1:], other[0][:, 1:])
    assert torch.equal(first[0][:, :1], clean[:, :1]) and not bool(first[1][:, :1].any())
    sigma = first[2]['samples'][0]['sigma']
    assert 0 < sigma <= .05
    assert torch.allclose(first[1][:, 1:], torch.full((1, 24), 1000 * sigma))
    assert not torch.equal(first[0][:, 1:], clean[:, 1:])


def test_disabled_and_clean_mixture_use_bitwise_original_loss():
    config, batch = sample()
    baseline = FakeCausal()
    expected = causal_teacher_forcing_loss(baseline, batch, config, torch.device('cpu'))
    for options in (HistoryNoiseConfig(enabled=False), HistoryNoiseConfig(clean_probability=1)):
        model, metrics = FakeCausal(), {}
        loss = history_noise_loss(model, batch, config, torch.device('cpu'), noise_config=options,
                                  absolute_microbatch_index=480, stage_optimizer_step=0, metrics=metrics)
        assert torch.equal(loss, expected)
        assert len(model.calls) == 1
        assert torch.equal(model.calls[0][0], baseline.calls[0][0])
        assert torch.equal(model.calls[0][1]['clean_x'], baseline.calls[0][1]['clean_x'])
        assert torch.equal(model.calls[0][1]['aug_t'], baseline.calls[0][1]['aug_t'])
        assert metrics['forward_path'] == 'unchanged_original_clean_history_loss'


def test_noised_forward_changes_only_context_keeps_GT_target_and_gradients():
    config, batch = sample()
    snapshots = {key: value.clone() for key, value in batch.items()}
    model, metrics = FakeCausal(), {}
    loss = history_noise_loss(model, batch, config, torch.device('cpu'),
        noise_config=HistoryNoiseConfig(clean_probability=0), absolute_microbatch_index=640,
        stage_optimizer_step=20, metrics=metrics)
    for key in snapshots:
        assert torch.equal(batch[key], snapshots[key])
    noisy, args = model.calls[0]
    clean = validate_causal_batch(batch, config)
    assert torch.equal(noisy, batch['noisy_latents'].bfloat16())
    assert torch.equal(args['timestep'], batch['timesteps'])
    assert torch.equal(args['conditional_dict']['prompt_embeds'][0], batch['prompt_embeds'][0].bfloat16())
    assert not torch.equal(args['clean_x'][:, 1:], clean[:, 1:].bfloat16())
    assert torch.equal(args['clean_x'][:, :1], clean[:, :1].bfloat16())
    assert not bool(args['aug_t'][:, :1].any()) and bool((args['aug_t'][:, 1:] > 0).all())
    assert args['replace_first_timestep_and_noise_latents'] is True
    expected_prediction = noisy * model.scale + args['clean_x'] * .01
    expected_loss = torch.nn.functional.mse_loss(expected_prediction[:, 1:].float(), batch['target_flow'][:, 1:].bfloat16().float())
    assert torch.equal(loss, expected_loss)
    loss.backward()
    assert model.scale.grad is not None and bool(torch.isfinite(model.scale.grad))
    assert len(model.calls) == 1 and metrics['noised_samples'] == 1
