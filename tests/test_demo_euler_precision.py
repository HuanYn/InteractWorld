"""CPU numerical/interface regressions, not evidence of generated video quality."""
import pytest
import torch

from training.demo.action_backend import euler_rollout
from training.longforcing_lite import euler_sigmas


def sigma_nodes():
    return euler_sigmas(40, shift=5.0, device=torch.device('cpu'), dtype=torch.bfloat16)


class RecordingFlow:
    def __init__(self, constant=None):
        self.constant = constant
        self.calls = []

    def __call__(self, value, *, conditional_dict, timestep, replace_first_timestep_and_noise_latents):
        self.calls.append((value.clone(), timestep.clone(), conditional_dict,
                           replace_first_timestep_and_noise_latents))
        velocity = (value * .03125 + .0078125 if self.constant is None
                    else torch.full_like(value, self.constant))
        return velocity, None


def legacy_reference(model, *, first, noise, conditions, sigmas):
    # Original pre-change body: locks down default operation order and rounding.
    current = noise.clone()
    current[:, :1] = first
    for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        timestep = (sigma * 1000).expand(current.shape[:2]).clone()
        timestep[:, 0] = 0
        velocity, _ = model(current, conditional_dict=conditions, timestep=timestep,
                            replace_first_timestep_and_noise_latents=True)
        current = current + (next_sigma - sigma) * velocity
        current[:, :1] = first
        assert bool(torch.isfinite(current).all())
    return current


def test_small_constant_flow_is_accumulated_without_changing_noise_or_sigma_nodes():
    noise = torch.ones(1, 2, 1, 1, 1, dtype=torch.bfloat16)
    first = noise[:, :1].clone()
    sigmas = sigma_nodes()
    before_noise, before_sigmas = noise.clone(), sigmas.clone()
    legacy = euler_rollout(RecordingFlow(.01), first=first, noise=noise, conditions={}, sigmas=sigmas)
    precise = euler_rollout(RecordingFlow(.01), first=first, noise=noise, conditions={},
                            sigmas=sigmas, fp32_accumulation=True)
    # The fake model remains BF16 in BOTH modes; use its actual quantized flow.
    actual_flow = torch.tensor(.01, dtype=torch.bfloat16).float().item()
    expected = 1 + (sigmas[-1].float()-sigmas[0].float()).item() * actual_flow
    assert legacy[0, 1].item() == 1.0
    assert precise.dtype == torch.float32
    assert precise[0, 1].item() == pytest.approx(expected, abs=5e-7)
    assert torch.equal(noise, before_noise) and torch.equal(sigmas, before_sigmas)


def test_model_dtype_timestep_values_first_frame_and_conditions_are_exactly_preserved():
    noise = torch.tensor([3., 1., -2.], dtype=torch.bfloat16).reshape(1, 3, 1, 1, 1)
    first = torch.full_like(noise[:, :1], .375)
    conditions = {'act_context': [torch.ones(32, 2, 1, 1, dtype=torch.bfloat16)], 'act_context_scale': .03}
    action_before = conditions['act_context'][0].clone()
    sigmas = sigma_nodes()
    first_before, noise_before, sigma_before = first.clone(), noise.clone(), sigmas.clone()
    old, new = RecordingFlow(.01), RecordingFlow(.01)
    legacy_reference(old, first=first, noise=noise, conditions=conditions, sigmas=sigmas)
    result = euler_rollout(new, first=first, noise=noise, conditions=conditions,
                           sigmas=sigmas, fp32_accumulation=True)
    assert len(old.calls) == len(new.calls) == 40
    assert torch.equal(old.calls[0][0], new.calls[0][0])
    for original, actual in zip(old.calls, new.calls):
        video, timestep, observed_conditions, first_flag = actual
        assert video.dtype == noise.dtype == torch.bfloat16
        assert timestep.dtype == original[1].dtype == torch.bfloat16
        assert torch.equal(timestep, original[1])  # Not a FP32 multiply followed by BF16 cast.
        assert torch.equal(video[:, :1], first) and torch.equal(timestep[:, :1], torch.zeros_like(timestep[:, :1]))
        assert observed_conditions is conditions and first_flag is True
    assert torch.equal(result[:, :1], first.float())
    for actual, before in ((first,first_before),(noise,noise_before),(sigmas,sigma_before),
                           (conditions['act_context'][0],action_before)):
        assert torch.equal(actual,before)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('explicit_flag', [False, True])
def test_default_and_explicit_false_are_bitwise_legacy_compatible(dtype, explicit_flag):
    noise = torch.tensor([.25, 1., -2.],dtype=dtype).reshape(1,3,1,1,1)
    first, sigmas = noise[:, :1].clone(), sigma_nodes().to(dtype)
    expected = legacy_reference(RecordingFlow(), first=first, noise=noise, conditions={}, sigmas=sigmas)
    kwargs = {'fp32_accumulation':False} if explicit_flag else {}
    actual = euler_rollout(RecordingFlow(), first=first, noise=noise, conditions={}, sigmas=sigmas, **kwargs)
    assert actual.dtype == expected.dtype and torch.equal(actual, expected)


def test_opt_in_does_not_consume_rng_and_rejects_implicit_truthy_flags():
    noise = torch.ones(1,2,1,1,1,dtype=torch.bfloat16)
    before = torch.get_rng_state().clone()
    euler_rollout(RecordingFlow(.01),first=noise[:, :1],noise=noise,conditions={},
                  sigmas=sigma_nodes(),fp32_accumulation=True)
    assert torch.equal(torch.get_rng_state(),before)
    with pytest.raises(ValueError,match='explicit boolean'):
        euler_rollout(RecordingFlow(),first=noise[:, :1],noise=noise,conditions={},
                      sigmas=sigma_nodes(),fp32_accumulation='true')
