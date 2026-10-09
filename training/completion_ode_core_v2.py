"""CPU-testable ODE contracts; no model, checkpoint, CUDA, or launch imports.

The production candidate is a *packed* causal teacher: 40 native-BF16 Euler
updates, two CFG branches, and a separately supplied GT clean-history buffer.
The callback must enforce the existing causal mask.  This module cannot make
an arbitrary callback causal; ``probe_packed_causality`` is an adapter probe,
not permission to cache real trajectories before that adapter is verified.

Student inference predicts x0 and independently re-noises at the next level.
It is not four-step Euler.  The clean history at inference is exclusively the
student's committed output; GT clean context is only an offline-training input.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Mapping, Protocol

import torch

from utils.scheduler import FlowMatchScheduler

TEACHER_STEPS = 40
TEACHER_CFG = 5.0
STUDENT_CFG = 1.0
GRID_INDICES = (0, 10, 20, 30, 40)
BLOCK_LATENTS = 3
RGB_PER_LATENT = 4
ACTION_DIM = 8
GRID_SCHEMA = "completion_ode_bf16_teacher_grid_v2"
NOISE_NAMESPACE = "completion_ode_endpoint_renoise_v2"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_tensor(value: torch.Tensor, name: str) -> None:
    _require(isinstance(value, torch.Tensor) and value.is_floating_point(),
             f"{name} must be a floating tensor")
    _require(bool(torch.isfinite(value).all()), f"{name} contains nonfinite values")


def _video(value: torch.Tensor, name: str) -> int:
    _finite_tensor(value, name)
    _require(value.ndim == 5 and all(size > 0 for size in value.shape),
             f"{name} must have B,T,C,H,W dimensions")
    _require(value.shape[1] >= 4 and (value.shape[1] - 1) % BLOCK_LATENTS == 0,
             f"{name} must contain one first latent and complete three-latent blocks")
    return (value.shape[1] - 1) // BLOCK_LATENTS


def _actions(value: torch.Tensor, batch: int, blocks: int) -> None:
    _finite_tensor(value, "actions")
    _require(tuple(value.shape) == (batch, blocks * BLOCK_LATENTS * RGB_PER_LATENT, ACTION_DIM),
             "actions must have exactly 12 rows per block in W,A,S,D,I,J,K,L order")


@dataclass(frozen=True)
class TeacherGrid:
    """Record both integration sigma and the actually rounded model timestep."""

    sigmas: tuple[float, ...]
    model_timesteps: tuple[float, ...]
    indices: tuple[int, ...] = GRID_INDICES

    def record(self) -> dict[str, Any]:
        return dict(schema=GRID_SCHEMA, teacher_steps=40, teacher_cfg=5.0,
                    timestep_shift=5.0, native_dtype="bfloat16", indices=list(self.indices),
                    integration_sigmas=list(self.sigmas),
                    actual_model_timesteps=list(self.model_timesteps),
                    timestep_rule="BF16(sigma_BF16 * 1000), then adapter may cast to FP32")

    def levels(self, state: torch.Tensor, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        _require(type(index) is int and 0 <= index <= TEACHER_STEPS, "teacher index outside 0..40")
        # These are already rounded values. Do not recover sigma from timestep,
        # reapply the shift, or multiply an FP32 sigma by 1000 instead.
        return (torch.full((state.shape[0],), self.sigmas[index], device=state.device, dtype=torch.bfloat16),
                torch.full((state.shape[0],), self.model_timesteps[index], device=state.device, dtype=torch.bfloat16))


def teacher_grid() -> TeacherGrid:
    """Match longforcing_lite.euler_sigmas/_solve_block without importing Wan."""
    with torch.device("cpu"):
        scheduler = FlowMatchScheduler(num_inference_steps=40, shift=5.0,
                                       sigma_min=0.0, extra_one_step=True)
    sigmas = torch.cat((scheduler.sigmas.to(dtype=torch.bfloat16),
                        torch.zeros(1, dtype=torch.bfloat16)))
    actual_t = sigmas * 1000.0  # Native BF16 multiplication is part of the contract.
    return TeacherGrid(tuple(sigmas.tolist()), tuple(actual_t.tolist()))


def validate_grid(grid: TeacherGrid) -> None:
    _require(isinstance(grid, TeacherGrid) and grid == teacher_grid(),
             "actual 40-step BF16 integration/model grid changed")


def grid_from_record(record: Mapping[str, Any]) -> TeacherGrid:
    expected = teacher_grid()
    _require(dict(record) == expected.record(), "trajectory grid record differs from actual BF16 teacher grid")
    return expected


def clean_visibility_mask(latent_count: int, *, device: Any = "cpu") -> torch.Tensor:
    """Reference latent-level clean visibility, not an installed DiT token mask.

    All video queries may read the first clean latent. Other clean keys must
    belong to a strictly earlier block. The first query reads only itself.
    The real adapter must separately verify that its token-level mask agrees.
    """
    _require(type(latent_count) is int and latent_count >= 4 and (latent_count - 1) % 3 == 0,
             "invalid one-plus-three-block latent count")
    pos = torch.arange(latent_count, device=device)
    block = (pos - 1) // 3
    mask = (pos[None, :] == 0) | ((pos[:, None] > 0) & (pos[None, :] > 0)
                                                & (block[None, :] < block[:, None]))
    return mask


def clean_history_prefix(clean_context: torch.Tensor, block_index: int) -> torch.Tensor:
    """Detached private past-only view for semantic tests; not a 640-call wrapper."""
    blocks = _video(clean_context, "clean context")
    _require(type(block_index) is int and 0 <= block_index < blocks, "invalid block index")
    return clean_context[:, :1 + block_index * 3].detach().clone()


def endpoint_from_velocity(noisy: torch.Tensor, velocity: torch.Tensor,
                           sigma: float | torch.Tensor) -> torch.Tensor:
    """Flow endpoint x0 = x_sigma - sigma*v, computed in FP32."""
    _finite_tensor(noisy, "noisy state")
    _finite_tensor(velocity, "velocity")
    _require(noisy.shape == velocity.shape and noisy.device == velocity.device,
             "velocity/state shape or device mismatch")
    level = torch.as_tensor(sigma, device=noisy.device, dtype=torch.float32)
    _require(level.ndim in (0, 1) and (level.ndim == 0 or level.numel() == noisy.shape[0])
             and bool(torch.isfinite(level).all()) and bool(((level >= 0) & (level <= 1)).all()),
             "integration sigma must be scalar or batch vector in [0,1]")
    if level.ndim:
        level = level.reshape(-1, *([1] * (noisy.ndim - 1)))
    endpoint = noisy.float() - level * velocity.float()
    _finite_tensor(endpoint, "predicted endpoint")
    return endpoint


def future_endpoint_loss(noisy: torch.Tensor, velocity: torch.Tensor,
                         teacher_endpoint: torch.Tensor,
                         sigma: float | torch.Tensor) -> torch.Tensor:
    """Only future latents are supervised; teacher target and sampled state detach.

    ``teacher_endpoint`` is the last *ODE-integrated* trajectory state, never
    the dataset GT clean history. This mathematical helper cannot distinguish
    identically shaped mislabelled tensors; cache lineage must bind their roles.
    """
    _video(noisy, "noisy state")
    _video(teacher_endpoint, "teacher endpoint")
    _require(noisy.shape == teacher_endpoint.shape and noisy.device == teacher_endpoint.device,
             "teacher endpoint/state mismatch")
    prediction = endpoint_from_velocity(noisy.detach(), velocity, sigma)
    loss = (prediction[:, 1:] - teacher_endpoint.detach().float()[:, 1:]).square().mean()
    _finite_tensor(loss, "ODE endpoint loss")
    return loss


def _packed_call(callback: Callable[..., torch.Tensor], *, state: torch.Tensor,
                 clean: torch.Tensor, actions: torch.Tensor, grid: TeacherGrid,
                 index: int, branch: str, conditions: Any) -> torch.Tensor:
    sigma, timestep = grid.levels(state, index)
    value = callback(noisy_latents=state, clean_context=clean, actions=actions,
                     clean_visibility=clean_visibility_mask(state.shape[1], device=state.device),
                     sigma=sigma, timestep=timestep, branch=branch, conditions=conditions)
    _finite_tensor(value, "packed velocity")
    _require(value.shape == state.shape and value.dtype == state.dtype and value.device == state.device,
             "packed callback must return same-shape/native-dtype velocity")
    return value


@dataclass(frozen=True)
class PackedTrajectory:
    """Five sampled states; GT clean context is intentionally not a sixth state."""

    grid: TeacherGrid
    states: tuple[torch.Tensor, ...]
    denoising_forwards: int = 80

    @property
    def endpoint(self) -> torch.Tensor:
        return self.states[-1]

    def validate(self) -> None:
        validate_grid(self.grid)
        _require(len(self.states) == 5 and self.denoising_forwards == 80,
                 "packed trajectory must contain five states and 80 teacher callbacks")
        first = self.states[0]
        _video(first, "trajectory state")
        for state in self.states:
            _video(state, "trajectory state")
            _require(state.shape == first.shape and state.dtype == torch.bfloat16
                     and state.device == first.device and not state.requires_grad
                     and torch.equal(state[:, :1], first[:, :1]),
                     "trajectory shape, BF16 dtype, detached state or fixed first latent changed")


@torch.no_grad()
def packed_teacher_trajectory(callback: Callable[..., torch.Tensor], *,
                              noisy_initial: torch.Tensor, clean_context: torch.Tensor,
                              actions: torch.Tensor, positive_conditions: Any,
                              negative_conditions: Any, grid: TeacherGrid | None = None) -> PackedTrajectory:
    """Candidate packed80 trajectory, conditional on a separately verified adapter.

    Current/future GT exists in the clean buffer but MUST be masked by callback.
    No KV commits are hidden in these 80 logical denoising callback invocations.
    """
    grid = teacher_grid() if grid is None else grid
    validate_grid(grid)
    blocks = _video(noisy_initial, "initial state")
    _video(clean_context, "GT clean context")
    _actions(actions, noisy_initial.shape[0], blocks)
    _require(noisy_initial.dtype == clean_context.dtype == torch.bfloat16
             and noisy_initial.shape == clean_context.shape and noisy_initial.device == clean_context.device,
             "packed teacher requires matching native-BF16 state/GT context")
    _require(torch.equal(noisy_initial[:, :1], clean_context[:, :1]), "initial first latent must be clean and fixed")
    state = noisy_initial.detach().clone()
    clean = clean_context.detach().clone()
    clean_snapshot = clean.clone()
    states = [state.clone()]
    sigmas = torch.tensor(grid.sigmas, dtype=torch.bfloat16, device=state.device)
    for index in range(40):
        positive = _packed_call(callback, state=state, clean=clean, actions=actions,
                                grid=grid, index=index, branch="positive", conditions=positive_conditions)
        negative = _packed_call(callback, state=state, clean=clean, actions=actions,
                                grid=grid, index=index, branch="negative", conditions=negative_conditions)
        velocity = negative + TEACHER_CFG * (positive - negative)
        # Exactly the native Euler arithmetic, but never update the first latent.
        updated = state + (sigmas[index + 1] - sigmas[index]) * velocity
        state = torch.cat((state[:, :1], updated[:, 1:]), dim=1)
        _finite_tensor(state, "teacher integrated state")
        _require(torch.equal(clean, clean_snapshot), "packed callback mutated GT clean history")
        if index + 1 in GRID_INDICES:
            states.append(state.clone())
    result = PackedTrajectory(grid, tuple(states))
    result.validate()
    return result


def packed_student_loss(callback: Callable[..., torch.Tensor], *, trajectory: PackedTrajectory,
                        clean_context: torch.Tensor, actions: torch.Tensor,
                        positive_conditions: Any, round_index: int) -> torch.Tensor:
    """One positive branch at one cached level, under the same clean-history mask."""
    trajectory.validate()
    _require(type(round_index) is int and 0 <= round_index < 4, "student training round must be 0..3")
    state = trajectory.states[round_index]
    blocks = _video(clean_context, "GT clean context")
    _require(clean_context.shape == state.shape and clean_context.dtype == state.dtype
             and clean_context.device == state.device
             and torch.equal(clean_context[:, :1], state[:, :1]), "separate GT context/state contract changed")
    _actions(actions, state.shape[0], blocks)
    velocity = _packed_call(callback, state=state.detach().clone(), clean=clean_context.detach().clone(), actions=actions,
                            grid=trajectory.grid, index=GRID_INDICES[round_index],
                            branch="positive", conditions=positive_conditions)
    return future_endpoint_loss(state, velocity, trajectory.endpoint,
                                trajectory.grid.sigmas[GRID_INDICES[round_index]])


def probe_packed_causality(callback: Callable[..., torch.Tensor], *, noisy: torch.Tensor,
                           clean_context: torch.Tensor, actions: torch.Tensor,
                           positive_conditions: Any, teacher_index: int = 0) -> dict[str, Any]:
    """Perturb and VJP-probe this callback's current/future access at a fixed level.

    Checks every block, current+future clean GT and all out-of-block noisy
    positions (earlier noisy prefix including position zero, and future noisy
    suffix). Past information must come from clean context, not past noisy. The
    deterministic nonuniform VJP is a finite probe, not a full Jacobian proof;
    the actual adapter must additionally audit its installed token-level mask.
    Running this on a toy callback does not validate a Wan adapter or GPU path.
    """
    grid = teacher_grid()
    blocks = _video(noisy, "probe noisy state")
    _video(clean_context, "probe GT context")
    _require(noisy.shape == clean_context.shape and noisy.dtype == clean_context.dtype == torch.bfloat16,
             "probe needs matching BF16 state/context")
    _actions(actions, noisy.shape[0], blocks)
    clean_diff, noisy_diff, clean_grad, noisy_grad, calls = 0., 0., 0., 0., 0
    earlier_noisy_diff, earlier_noisy_grad = 0., 0.
    for block in range(blocks):
        start, end = 1 + 3 * block, 4 + 3 * block
        clean = clean_context.detach().clone().requires_grad_(True)
        state = noisy.detach().clone().requires_grad_(True)
        baseline = _packed_call(callback, state=state, clean=clean, actions=actions,
                                grid=grid, index=teacher_index, branch="positive", conditions=positive_conditions)
        weights = torch.linspace(.5, 1.5, baseline[:, start:end].numel(),
                                 device=baseline.device).reshape_as(baseline[:, start:end])
        scalar = (baseline[:, start:end].float() * weights).sum()
        gradients = torch.autograd.grad(scalar, (clean, state), allow_unused=True) if scalar.requires_grad else (None, None)
        if gradients[0] is not None:
            clean_grad = max(clean_grad, float(gradients[0][:, start:].abs().max()))
        if gradients[1] is not None:
            earlier_noisy_grad = max(earlier_noisy_grad, float(gradients[1][:, :start].abs().max()))
        if end < state.shape[1] and gradients[1] is not None:
            noisy_grad = max(noisy_grad, float(gradients[1][:, end:].abs().max()))
        with torch.no_grad():
            changed_clean = clean_context.detach().clone()
            changed_clean[:, start:] += 7.
            value = _packed_call(callback, state=noisy.detach(), clean=changed_clean, actions=actions,
                                 grid=grid, index=teacher_index, branch="positive", conditions=positive_conditions)
            clean_diff = max(clean_diff, float((value[:, start:end] - baseline[:, start:end]).abs().max()))
            changed_noisy = noisy.detach().clone()
            changed_noisy[:, end:] += 7.
            value = _packed_call(callback, state=changed_noisy, clean=clean_context.detach(), actions=actions,
                                 grid=grid, index=teacher_index, branch="positive", conditions=positive_conditions)
            noisy_diff = max(noisy_diff, float((value[:, start:end] - baseline[:, start:end]).abs().max()))
            changed_earlier_noisy = noisy.detach().clone()
            changed_earlier_noisy[:, :start] += 7.
            value = _packed_call(callback, state=changed_earlier_noisy, clean=clean_context.detach(), actions=actions,
                                 grid=grid, index=teacher_index, branch="positive", conditions=positive_conditions)
            earlier_noisy_diff = max(earlier_noisy_diff, float((value[:, start:end] - baseline[:, start:end]).abs().max()))
        calls += 4
    _require(all(value == 0. for value in (clean_diff, noisy_diff, clean_grad, noisy_grad,
                                          earlier_noisy_diff, earlier_noisy_grad)),
             "packed callback leaked current/future GT or out-of-block noisy state")
    return dict(passed=True, teacher_index=teacher_index, blocks_checked=blocks,
                logical_forward_calls=calls, max_clean_perturbation_difference=clean_diff,
                max_future_noisy_perturbation_difference=noisy_diff,
                max_earlier_noisy_perturbation_difference=earlier_noisy_diff,
                max_forbidden_clean_vjp=clean_grad, max_future_noisy_vjp=noisy_grad,
                max_earlier_noisy_vjp=earlier_noisy_grad,
                proof_scope="finite callback perturbation/VJP probe; token-mask audit additionally required")


@dataclass(frozen=True)
class NoiseStreams:
    initial: torch.Tensor
    renoise: tuple[torch.Tensor, ...]
    seeds: tuple[int, ...]
    namespace: str


def fixed_noise_streams(*, seed: int, block_index: int, shape: tuple[int, ...],
                        namespace: str = NOISE_NAMESPACE) -> NoiseStreams:
    """Four independent CPU-FP32 streams; never consume global RNG state."""
    _require(type(seed) is int and seed >= 0 and type(block_index) is int and block_index >= 0,
             "noise seed/block must be nonnegative integers")
    _require(len(shape) == 5 and shape[1] == 3 and all(type(n) is int and n > 0 for n in shape),
             "noise must describe a B,3,C,H,W block")
    _require(isinstance(namespace, str) and bool(namespace), "noise namespace required")
    seeds, noises = [], []
    for stream in ("initial", "renoise-1", "renoise-2", "renoise-3"):
        digest = hashlib.sha256(f"{namespace}|{seed}|{block_index}|{stream}".encode()).digest()
        actual_seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
        generator = torch.Generator(device="cpu").manual_seed(actual_seed)
        seeds.append(actual_seed)
        noises.append(torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu"))
    return NoiseStreams(noises[0], tuple(noises[1:]), tuple(seeds), namespace)


@torch.no_grad()
def sample_endpoint_block(callback: Callable[..., torch.Tensor], *, initial_noise: torch.Tensor,
                          renoise: tuple[torch.Tensor, ...], history: torch.Tensor,
                          action_block: torch.Tensor, block_index: int,
                          positive_conditions: Any, grid: TeacherGrid | None = None) -> torch.Tensor:
    """Four positive CFG1 predictions, three flow re-noises, final x0 output."""
    grid = teacher_grid() if grid is None else grid
    validate_grid(grid)
    _finite_tensor(initial_noise, "initial block")
    _require(initial_noise.ndim == 5 and initial_noise.shape[1] == 3
             and initial_noise.dtype == torch.bfloat16, "student model block must be native BF16 B,3,C,H,W")
    _finite_tensor(history, "own clean history")
    _require(type(block_index) is int and block_index >= 0
             and history.shape == (initial_noise.shape[0], 1 + 3 * block_index, *initial_noise.shape[2:])
             and history.dtype == initial_noise.dtype and history.device == initial_noise.device,
             "student may read only its completed past prefix")
    _actions(action_block, initial_noise.shape[0], 1)
    _require(len(renoise) == 3, "exactly three independent re-noise tensors required")
    for noise in renoise:
        _finite_tensor(noise, "re-noise tensor")
        _require(noise.shape == initial_noise.shape, "re-noise block shape mismatch")
    state = initial_noise.detach().clone()
    prefix = history.detach().clone()
    for round_index, index in enumerate(GRID_INDICES[:-1]):
        sigma, timestep = grid.levels(state, index)
        velocity = callback(noisy_block=state, history=prefix, action_block=action_block,
                            sigma=sigma, timestep=timestep, block_index=block_index,
                            branch="positive", cfg_scale=STUDENT_CFG, conditions=positive_conditions)
        _require(isinstance(velocity, torch.Tensor) and velocity.dtype == state.dtype,
                 "single-branch callback must preserve native BF16 velocity dtype")
        _require(torch.equal(prefix, history.detach()), "callback mutated read-only own history")
        endpoint = endpoint_from_velocity(state, velocity, sigma)
        if round_index == 3:
            return endpoint.to(dtype=state.dtype)
        next_sigma = grid.sigmas[GRID_INDICES[round_index + 1]]
        noise = renoise[round_index].to(device=state.device, dtype=torch.float32)
        # Flow interpolation, deliberately neither VP sqrt nor an Euler delta.
        state = ((1. - next_sigma) * endpoint + next_sigma * noise).to(dtype=state.dtype)
        _finite_tensor(state, "student re-noised state")
    raise AssertionError("unreachable")


class SingleBranchBackend(Protocol):
    """Each initialize/velocity/commit must report exactly one actual model call."""

    branch_count: int
    external_cfg_scale: float
    forward_count: int

    def initialize(self, first: torch.Tensor, conditions: Any) -> None: ...
    def velocity(self, **kwargs: Any) -> torch.Tensor: ...
    def commit(self, endpoint: torch.Tensor, block_index: int) -> None: ...


@torch.no_grad()
def sample_causal_clip(backend: SingleBranchBackend, *, first: torch.Tensor,
                       actions: torch.Tensor, positive_conditions: Any, seed: int,
                       blocks: int = 8, noise_namespace: str = NOISE_NAMESPACE) -> dict[str, Any]:
    """Inference-only orchestration: 8 blocks => 32 denoise + 9 commit calls.

    There is no future-GT argument. A new positive-only KV adapter is needed;
    existing FullHistoryCFG cannot satisfy this contract just by setting CFG1.
    """
    _finite_tensor(first, "first latent")
    _require(first.ndim == 5 and first.shape[1] == 1 and first.dtype == torch.bfloat16,
             "one native-BF16 first latent required")
    _require(type(blocks) is int and blocks > 0, "positive block count required")
    _actions(actions, first.shape[0], blocks)
    _require(backend.branch_count == 1 and backend.external_cfg_scale == 1.
             and type(backend.forward_count) is int and backend.forward_count == 0,
             "fresh single-positive-branch CFG1 backend required")
    backend.initialize(first.detach().clone(), positive_conditions)
    _require(backend.forward_count == 1, "first clean KV commit must be one actual forward")
    history, streams_record = first.detach().clone(), []
    block_shape = (first.shape[0], 3, *first.shape[2:])
    for block in range(blocks):
        streams = fixed_noise_streams(seed=seed, block_index=block, shape=block_shape, namespace=noise_namespace)
        before = backend.forward_count
        endpoint = sample_endpoint_block(backend.velocity,
            initial_noise=streams.initial.to(device=first.device, dtype=torch.bfloat16),
            renoise=streams.renoise, history=history,
            action_block=actions[:, block * 12:(block + 1) * 12], block_index=block,
            positive_conditions=positive_conditions)
        _require(backend.forward_count == before + 4, "four denoising calls must mean four actual model forwards")
        backend.commit(endpoint, block)
        _require(backend.forward_count == before + 5, "clean KV commit must be one actual positive forward")
        history = torch.cat((history, endpoint), dim=1)
        streams_record.append(dict(block_index=block, namespace=noise_namespace, seeds=list(streams.seeds)))
    return dict(latents=history, denoising_forwards=blocks * 4, commit_forwards=blocks + 1,
                total_forwards=backend.forward_count, cfg_scale=1., branch_count=1,
                grid=teacher_grid().record(), noise_streams=streams_record,
                future_clean_latents_supplied=False, history_source="own_committed_outputs")
