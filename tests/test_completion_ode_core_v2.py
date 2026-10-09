"""Small CPU tensors, not real Wan trajectories or GPU qualification."""

import copy
from dataclasses import replace

import pytest
import torch

from training import completion_ode_core_v2 as core


def data(blocks=2):
    clean = torch.arange(1 + blocks * 3, dtype=torch.bfloat16).reshape(1, -1, 1, 1, 1) / 8
    noisy = torch.ones_like(clean)
    noisy[:, :1] = clean[:, :1]
    actions = torch.zeros(1, blocks * 12, 8)
    actions[:, :, 5] = 1.
    return clean, noisy, actions


def masked_callback(**kwargs):
    clean = kwargs['clean_context']
    visible = kwargs['clean_visibility'].float()
    context = torch.einsum('ij,bjchw->bichw', visible, clean.float())
    branch = .25 if kwargs['branch'] == 'positive' else 0.
    return (kwargs['noisy_latents'].float() * .1 + context * .02 + branch).to(torch.bfloat16)


def trajectory(blocks=2):
    clean, noisy, actions = data(blocks)
    result = core.packed_teacher_trajectory(masked_callback, noisy_initial=noisy,
        clean_context=clean, actions=actions, positive_conditions='p', negative_conditions='n')
    return result, clean, actions


def test_actual_grid_matches_existing_solver_and_rounding():
    # Read-only reuse of the existing schedule, not a separately tuned four-step scheduler.
    from training.longforcing_lite import euler_sigmas
    expected = euler_sigmas(40, shift=5., device=torch.device('cpu'), dtype=torch.bfloat16)
    grid = core.teacher_grid()
    assert grid.sigmas == tuple(expected.tolist())
    assert grid.model_timesteps == tuple((expected * 1000.).tolist())
    assert [grid.sigmas[i] for i in core.GRID_INDICES] == [1., .9375, .83203125, .625, 0.]
    assert [grid.model_timesteps[i] for i in core.GRID_INDICES] == [1000., 936., 832., 624., 0.]
    assert grid.model_timesteps[10] != grid.sigmas[10] * 1000.
    assert core.grid_from_record(grid.record()) == grid


@pytest.mark.parametrize('key,value', [
    ('teacher_steps', 4), ('teacher_cfg', 1.), ('timestep_shift', 1.),
    ('native_dtype', 'float32'), ('indices', [0, 1, 2, 3, 4]),
])
def test_changed_grid_metadata_rejected(key, value):
    record = core.teacher_grid().record()
    record[key] = value
    with pytest.raises(ValueError, match='grid'):
        core.grid_from_record(record)


def test_fp32_reconstructed_timestep_is_not_actual_model_timestep():
    record = core.teacher_grid().record()
    record['actual_model_timesteps'][10] = 937.5
    with pytest.raises(ValueError):
        core.grid_from_record(record)
    grid = replace(core.teacher_grid(), indices=(0, 1, 2, 3, 40))
    with pytest.raises(ValueError):
        core.validate_grid(grid)


def test_reference_mask_and_private_past_prefix():
    clean, _, _ = data(3)
    mask = core.clean_visibility_mask(10)
    assert mask.dtype == torch.bool
    assert mask[:, 0].all()
    assert not mask[0, 1:].any()
    assert not mask[1:4, 1:].any()
    assert mask[4:7, 1:4].all() and not mask[4:7, 4:].any()
    prefix = core.clean_history_prefix(clean.requires_grad_(True), 2)
    assert prefix.shape[1] == 7 and not prefix.requires_grad
    prefix.zero_()
    assert clean[:, 1:].sum() > 0
    with pytest.raises(ValueError):
        core.clean_history_prefix(clean, 3)


def test_future_only_loss_is_fp32_and_teacher_state_detached():
    _, noisy, _ = data(1)
    noisy = noisy.requires_grad_(True)
    velocity = torch.full_like(noisy, .25, requires_grad=True)
    endpoint = torch.zeros_like(noisy, requires_grad=True)
    with torch.no_grad():
        velocity[:, 0] = 1000.
        endpoint[:, 0] = -1000.
    loss = core.future_endpoint_loss(noisy, velocity, endpoint, .5)
    assert loss.dtype == torch.float32
    assert loss.detach().item() == pytest.approx((1. - .5 * .25)**2)
    loss.backward()
    assert velocity.grad[:, 0].eq(0).all()
    assert velocity.grad[:, 1:].ne(0).all()
    assert endpoint.grad is None and noisy.grad is None


def test_endpoint_batch_sigma_and_nonfinite_rejection():
    noisy = torch.ones(2, 3, 1, 1, 1, dtype=torch.bfloat16)
    velocity = torch.full_like(noisy, 2.)
    endpoint = core.endpoint_from_velocity(noisy, velocity, torch.tensor([1., .5]))
    assert endpoint[:, 0, 0, 0, 0].tolist() == [-1., 0.]
    for bad in (1.1, -1., float('nan'), torch.ones(3)):
        with pytest.raises(ValueError):
            core.endpoint_from_velocity(noisy, velocity, bad)
    with pytest.raises(ValueError):
        core.endpoint_from_velocity(noisy, velocity[:, :2], .5)


def test_packed_teacher_uses_80_calls_exact_bf16_grid_and_fixed_first():
    clean, noisy, actions = data()
    calls = []
    def callback(**kwargs):
        calls.append((kwargs['branch'], float(kwargs['sigma'][0]), float(kwargs['timestep'][0])))
        assert not torch.is_grad_enabled()
        assert kwargs['clean_context'] is not clean
        return torch.full_like(kwargs['noisy_latents'], 2. if kwargs['branch'] == 'positive' else 1.)
    result = core.packed_teacher_trajectory(callback, noisy_initial=noisy,
        clean_context=clean, actions=actions, positive_conditions='p', negative_conditions='n')
    expected = noisy.clone()
    sigmas = torch.tensor(core.teacher_grid().sigmas, dtype=torch.bfloat16)
    saved = [expected.clone()]
    for index in range(40):
        expected[:, 1:] = expected[:, 1:] + (sigmas[index + 1] - sigmas[index]) * 6.
        if index + 1 in core.GRID_INDICES:
            saved.append(expected.clone())
    assert len(calls) == result.denoising_forwards == 80
    assert calls[20:22] == [('positive', .9375, 936.), ('negative', .9375, 936.)]
    assert all(torch.equal(a, b) for a, b in zip(result.states, saved))
    assert torch.equal(result.endpoint[:, :1], clean[:, :1])
    assert not torch.equal(result.endpoint, clean)  # Integrated target is not GT clean context.
    assert not noisy.requires_grad and not clean.requires_grad


@pytest.mark.parametrize('bad', ['fp32', 'wrongfirst', 'badshape', 'actions', 'nonfinite'])
def test_teacher_input_contract_fails_closed(bad):
    clean, noisy, actions = data()
    if bad == 'fp32':
        noisy = noisy.float()
    elif bad == 'wrongfirst':
        noisy[:, 0] = 3.
    elif bad == 'badshape':
        noisy = noisy[:, :-1]
    elif bad == 'actions':
        actions = actions[:, :-1]
    else:
        noisy[:, 2] = float('inf')
    with pytest.raises(ValueError):
        core.packed_teacher_trajectory(masked_callback, noisy_initial=noisy,
            clean_context=clean, actions=actions, positive_conditions='p', negative_conditions='n')


def test_teacher_rejects_mutating_context_or_wrong_callback_dtype():
    clean, noisy, actions = data()
    def mutating(**kwargs):
        kwargs['clean_context'].add_(1.)
        return torch.zeros_like(kwargs['noisy_latents'])
    for callback in (mutating, lambda **kw: torch.zeros_like(kw['noisy_latents']).float()):
        with pytest.raises(ValueError):
            core.packed_teacher_trajectory(callback, noisy_initial=noisy,
                clean_context=clean, actions=actions, positive_conditions='p', negative_conditions='n')


def test_student_training_one_positive_branch_and_separate_endpoint():
    cached, clean, actions = trajectory()
    weight = torch.tensor(.2, requires_grad=True)
    calls = []
    def callback(**kwargs):
        calls.append(kwargs)
        return (kwargs['noisy_latents'].float() * weight).to(torch.bfloat16)
    loss = core.packed_student_loss(callback, trajectory=cached, clean_context=clean,
        actions=actions, positive_conditions='p', round_index=1)
    loss.backward()
    assert weight.grad is not None and weight.grad != 0
    assert len(calls) == 1 and calls[0]['branch'] == 'positive'
    assert float(calls[0]['timestep'][0]) == 936.
    assert not calls[0]['clean_context'].requires_grad
    assert calls[0]['clean_context'].shape == cached.endpoint.shape
    expected = core.future_endpoint_loss(cached.states[1],
        (cached.states[1].float() * weight.detach()).to(torch.bfloat16), cached.endpoint, .9375)
    assert loss.detach() == expected


@pytest.mark.parametrize('index', [0, 10, 20, 30])
def test_masked_callback_passes_finite_causal_perturbation_and_vjp(index):
    clean, noisy, actions = data()
    result = core.probe_packed_causality(masked_callback, noisy=noisy, clean_context=clean,
        actions=actions, positive_conditions='p', teacher_index=index)
    assert result['passed'] and result['blocks_checked'] == 2
    assert result['logical_forward_calls'] == 8
    assert result['max_forbidden_clean_vjp'] == 0.
    assert result['max_future_noisy_vjp'] == 0.
    assert result['max_earlier_noisy_vjp'] == 0.
    assert result['max_earlier_noisy_perturbation_difference'] == 0.


@pytest.mark.parametrize('leak', ['clean', 'noisy'])
def test_causal_probe_rejects_unmasked_callback(leak):
    clean, noisy, actions = data()
    def callback(**kwargs):
        source = kwargs['clean_context'] if leak == 'clean' else kwargs['noisy_latents']
        return (kwargs['noisy_latents'].float() + source.float().mean(dim=1, keepdim=True)).to(torch.bfloat16)
    with pytest.raises(ValueError, match='leaked'):
        core.probe_packed_causality(callback, noisy=noisy, clean_context=clean,
            actions=actions, positive_conditions='p')


@pytest.mark.parametrize('teacher_index', [0, 10, 20, 30])
def test_probe_rejects_current_block_reading_earlier_noisy_block(teacher_index):
    # Regression: block 1 reads block 0 noisy while never reading future noisy
    # or forbidden clean. The earlier future-only probe incorrectly accepted it.
    clean, noisy, actions = data()
    def callback(**kwargs):
        value = masked_callback(**kwargs).float()
        past_noisy = kwargs['noisy_latents'][:, 1:4].float().mean(dim=1, keepdim=True)
        return torch.cat((value[:, :4], value[:, 4:7] + past_noisy), dim=1).to(torch.bfloat16)
    with pytest.raises(ValueError, match='out-of-block noisy'):
        core.probe_packed_causality(callback, noisy=noisy, clean_context=clean,
            actions=actions, positive_conditions='p', teacher_index=teacher_index)


def test_probe_rejects_reading_first_noisy_position_instead_of_clean_history():
    clean, noisy, actions = data()
    def callback(**kwargs):
        value = masked_callback(**kwargs).float()
        return (value + kwargs['noisy_latents'][:, :1].float()).to(torch.bfloat16)
    with pytest.raises(ValueError, match='out-of-block noisy'):
        core.probe_packed_causality(callback, noisy=noisy, clean_context=clean,
            actions=actions, positive_conditions='p')


def test_earlier_noisy_vjp_check_is_not_replaced_by_perturbation_only():
    clean, noisy, actions = data()
    def callback(**kwargs):
        value = masked_callback(**kwargs).float()
        past = kwargs['noisy_latents'][:, 1:4].float().mean(dim=1, keepdim=True)
        # Forward perturbations cancel, but a forbidden training gradient remains.
        return torch.cat((value[:, :4], value[:, 4:7] + past - past.detach()), dim=1).to(torch.bfloat16)
    with pytest.raises(ValueError, match='out-of-block noisy'):
        core.probe_packed_causality(callback, noisy=noisy, clean_context=clean,
            actions=actions, positive_conditions='p')


def test_noise_streams_reproducible_independent_and_do_not_touch_global_rng():
    global_state = torch.random.get_rng_state().clone()
    a = core.fixed_noise_streams(seed=42, block_index=0, shape=(1, 3, 2, 2, 2))
    b = core.fixed_noise_streams(seed=42, block_index=0, shape=(1, 3, 2, 2, 2))
    c = core.fixed_noise_streams(seed=42, block_index=1, shape=(1, 3, 2, 2, 2))
    assert torch.equal(global_state, torch.random.get_rng_state())
    assert len(set(a.seeds)) == 4 and len(set(a.seeds + c.seeds)) == 8
    assert torch.equal(a.initial, b.initial)
    assert all(torch.equal(x, y) for x, y in zip(a.renoise, b.renoise))
    assert a.initial.dtype == torch.float32 and a.initial.device.type == 'cpu'
    assert all(not torch.equal(a.initial, noise) for noise in a.renoise)
    other = core.fixed_noise_streams(seed=42, block_index=0, shape=(1, 3, 2, 2, 2), namespace='other')
    assert other.seeds != a.seeds


def test_sampler_is_endpoint_flow_renoise_not_four_step_euler_or_vp():
    first = torch.zeros(1, 1, 1, 1, 1, dtype=torch.bfloat16)
    initial = torch.ones(1, 3, 1, 1, 1, dtype=torch.bfloat16)
    noises = tuple(torch.full_like(initial, n).float() for n in (2., 3., 4.))
    calls = []
    def callback(**kwargs):
        calls.append(copy.copy(kwargs))
        return torch.full_like(kwargs['noisy_block'], .25)
    output = core.sample_endpoint_block(callback, initial_noise=initial, renoise=noises,
        history=first, action_block=torch.zeros(1, 12, 8), block_index=0, positive_conditions='p')
    grid = core.teacher_grid()
    state = initial.clone()
    for round_index, index in enumerate(core.GRID_INDICES[:-1]):
        assert torch.equal(calls[round_index]['noisy_block'], state)
        assert calls[round_index]['branch'] == 'positive' and calls[round_index]['cfg_scale'] == 1.
        assert float(calls[round_index]['timestep'][0]) == grid.model_timesteps[index]
        endpoint = state.float() - grid.sigmas[index] * .25
        if round_index < 3:
            level = grid.sigmas[core.GRID_INDICES[round_index + 1]]
            state = ((1. - level) * endpoint + level * noises[round_index]).to(torch.bfloat16)
    assert len(calls) == 4 and torch.equal(output, endpoint.to(torch.bfloat16))
    assert not torch.equal(output, initial - .25)  # Constant-velocity Euler endpoint.
    assert all('clean_context' not in call for call in calls)


class ToySingleBackend:
    branch_count = 1
    external_cfg_scale = 1.
    forward_count = 0

    def __init__(self):
        self.committed = []
        self.calls = []

    def initialize(self, first, conditions):
        self.forward_count += 1
        self.committed.append(first.clone())

    def velocity(self, **kwargs):
        self.forward_count += 1
        assert kwargs['branch'] == 'positive' and kwargs['cfg_scale'] == 1.
        assert torch.equal(kwargs['history'], torch.cat(self.committed, dim=1))
        self.calls.append(kwargs)
        return torch.zeros_like(kwargs['noisy_block'])

    def commit(self, endpoint, block_index):
        assert block_index + 1 == len(self.committed)
        self.forward_count += 1
        self.committed.append(endpoint.clone())


def test_full_clip_41_actual_reported_calls_own_history_and_matching_noise():
    first = torch.zeros(1, 1, 1, 1, 1, dtype=torch.bfloat16)
    actions = torch.arange(96).float().reshape(1, 96, 1).expand(-1, -1, 8)
    backend = ToySingleBackend()
    result = core.sample_causal_clip(backend, first=first, actions=actions,
                                    positive_conditions='p', seed=42)
    other = core.sample_causal_clip(ToySingleBackend(), first=first, actions=actions,
                                   positive_conditions='p', seed=42)
    assert result['denoising_forwards'] == 32 and result['commit_forwards'] == 9
    assert result['total_forwards'] == backend.forward_count == 41
    assert result['latents'].shape == (1, 25, 1, 1, 1)
    assert torch.equal(result['latents'], other['latents'])
    assert torch.equal(result['latents'][:, :1], first)
    assert result['future_clean_latents_supplied'] is False
    assert result['history_source'] == 'own_committed_outputs'
    for block in range(8):
        assert torch.equal(backend.calls[block * 4]['action_block'], actions[:, block * 12:(block + 1) * 12])


@pytest.mark.parametrize('bad', ['dualbranch', 'cfg5', 'used', 'hiddenforward'])
def test_two_branch_cfg1_alias_or_incorrect_forward_accounting_rejected(bad):
    backend = ToySingleBackend()
    if bad == 'dualbranch':
        backend.branch_count = 2
    elif bad == 'cfg5':
        backend.external_cfg_scale = 5.
    elif bad == 'used':
        backend.forward_count = 1
    else:
        def initialize(first, conditions):
            backend.forward_count += 2
        backend.initialize = initialize
    with pytest.raises(ValueError):
        core.sample_causal_clip(backend, first=torch.zeros(1, 1, 1, 1, 1, dtype=torch.bfloat16),
            actions=torch.zeros(1, 12, 8), positive_conditions='p', seed=42, blocks=1)


@pytest.mark.parametrize('bad', ['history', 'actions', 'renoise', 'dtype'])
def test_sampler_rejects_future_history_or_malformed_inputs(bad):
    state = torch.zeros(1, 3, 1, 1, 1, dtype=torch.bfloat16)
    history = torch.zeros(1, 1, 1, 1, 1, dtype=torch.bfloat16)
    actions, noises = torch.zeros(1, 12, 8), (state, state, state)
    if bad == 'history':
        history = torch.zeros(1, 4, 1, 1, 1, dtype=torch.bfloat16)
    elif bad == 'actions':
        actions = actions[:, :-1]
    elif bad == 'renoise':
        noises = noises[:2]
    else:
        state = state.float()
    with pytest.raises(ValueError):
        core.sample_endpoint_block(lambda **kw: torch.zeros_like(kw['noisy_block']),
            initial_noise=state, renoise=noises, history=history,
            action_block=actions, block_index=0, positive_conditions='p')
