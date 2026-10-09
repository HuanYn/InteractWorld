"""Fixed action-margin micro step with one autograd graph resident at a time.

This is an independent project experiment, not an official ABot loss.  The
default loss remains ``train_action_teacher._forward_loss`` (FP32 future-only
MSE).  The caller owns clipping, master writeback, optimizer steps and the
eight-micro accumulation boundary; this helper never steps an optimizer.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from typing import Any, Callable

import numpy as np
import torch

LAMBDA = 0.1
RELATIVE_MARGIN = 0.01
ACCUMULATION_STEPS = 8
RECOMPUTE_RTOL = 1e-5
RECOMPUTE_ATOL = 1e-7


def margin_contract() -> dict[str, Any]:
    return dict(
        kind="jl_relative_flow_margin_v1", auxiliary_weight=LAMBDA,
        relative_margin=RELATIVE_MARGIN, margin_stop_gradient=True,
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        wrong_action_swap_indices=[5, 7], identical_actions_policy="correct_only",
        base_loss="train_action_teacher._forward_loss:fp32_future_only_mse",
        hinge_active_rule="fp32((1.01 * detach(C)) - W) > 0; boundary_inactive",
        one_live_autograd_graph=True, wrong_recompute_rtol=RECOMPUTE_RTOL,
        wrong_recompute_atol=RECOMPUTE_ATOL, wrong_rng_side_effects=False,
        optimizer_step_in_helper=False,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def swapped_action_batch(batch: Mapping[str, Any], config: Any) -> tuple[dict[str, Any], bool]:
    """Copy only the small action tensor; all latent/text inputs stay identical."""
    _require(isinstance(batch, Mapping), "batch must be a mapping")
    _require(not any(key in batch for key in ("y", "clip_fea", "ref_latents", "ref_mask")),
             "action-margin experiment has no additional/reference conditioning")
    actions = batch.get("actions")
    expected = (config.training.micro_batch_size, config.data.num_frames - 1, 8)
    _require(torch.is_tensor(actions) and tuple(actions.shape) == expected,
             f"actions must have shape {expected}")
    _require(not actions.requires_grad, "observed actions must not require gradients")
    _require(bool(torch.isfinite(actions).all())
             and bool(((actions == 0) | (actions == 1)).all()), "actions must be finite binary keys")
    changed = not torch.equal(actions[..., 5], actions[..., 7])
    wrong_actions = actions.clone()
    wrong_actions[..., 5] = actions[..., 7]
    wrong_actions[..., 7] = actions[..., 5]
    wrong = dict(batch)
    wrong["actions"] = wrong_actions
    return wrong, not changed


def _rng_state(device: torch.device) -> dict[str, Any]:
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def _restore_rng(state: Mapping[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


@contextmanager
def _wrong_rng(device: torch.device, replay: Mapping[str, Any]):
    """Wrong passes replay the same draw without changing the correct-pass RNG."""
    ambient = _rng_state(device)
    _restore_rng(replay, device)
    try:
        yield
    finally:
        _restore_rng(ambient, device)


def _input_versions(batch: Mapping[str, Any]) -> list[tuple[torch.Tensor, int]]:
    # Metadata only: no clone/hash/scan of large latents, prompts or model weights.
    found: list[tuple[torch.Tensor, int]] = []

    def visit(value: Any) -> None:
        if torch.is_tensor(value):
            found.append((value, value._version))
        elif isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(batch)
    return found


def _unchanged(versions: list[tuple[torch.Tensor, int]]) -> None:
    _require(all(tensor._version == version for tensor, version in versions),
             "forward/backward mutated an input tensor in place")


def _scalar(loss: torch.Tensor, *, branch: str, gradient: bool) -> float:
    _require(torch.is_tensor(loss) and loss.ndim == 0 and loss.dtype == torch.float32,
             f"{branch} must return scalar FP32 future MSE")
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError(f"non-finite {branch} loss")
    value = float(loss.detach())
    _require(value >= 0, f"{branch} MSE must be nonnegative")
    _require(loss.requires_grad is gradient, f"{branch} unexpected gradient mode")
    return value


def accumulate_margin_micro(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    config: Any,
    device: torch.device | str,
    profiler: Any = None,
    *,
    forward_loss: Callable | None = None,
) -> dict[str, Any]:
    """Accumulate one of eight micros, returning detached *true* objective metrics.

    Wrong-probe/no-grad -> correct/backward/release -> active wrong/recompute/
    negative-backward.  A failure clears the whole incomplete accumulated step
    and raises; callers must not catch that failure and continue optimization.
    ``forward_loss`` injection is only for focused CPU mathematical tests.
    """
    device = torch.device(device)
    try:
        _require(device.type in ("cpu", "cuda"), "only CPU test or CUDA training is supported")
        _require(config.training.gradient_accumulation_steps == ACCUMULATION_STEPS,
                 "fixed action-margin accumulation must equal eight")
        wrong, identical = swapped_action_batch(batch, config)
        if forward_loss is None:
            from train_action_teacher import _forward_loss
            forward_loss = _forward_loss
        versions = _input_versions(batch) + _input_versions(wrong)
        replay = _rng_state(device)
        forwards = backwards = 0

        def phase(name: str) -> None:
            if profiler is not None:
                profiler.set_phase(name)

        def evaluate(current: Mapping[str, Any], name: str, *, gradient: bool) -> tuple[torch.Tensor, float]:
            nonlocal forwards
            phase(name)
            saved = torch.autograd.graph.save_on_cpu(pin_memory=True) if device.type == "cuda" and gradient else nullcontext()
            cast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with saved, cast:
                result = forward_loss(model, current, config, device)
            forwards += 1
            _unchanged(versions)
            return result, _scalar(result, branch=name, gradient=gradient)

        w_tensor = None
        w = None
        if not identical:
            with _wrong_rng(device, replay), torch.no_grad():
                w_tensor, w = evaluate(wrong, "margin_wrong_probe", gradient=False)
        c_tensor, c = evaluate(batch, "margin_correct_forward", gradient=True)
        if identical:
            active, hinge, correct_weight = False, 0.0, 1.0
        else:
            # FP32 threshold implements the fixed strict > boundary rule.  The
            # margin is detached: dC coefficient is 1, not 1.01, when active.
            residual = (1.0 + RELATIVE_MARGIN) * c_tensor.detach() - w_tensor
            hinge = max(0.0, float(residual))
            active = bool(residual > 0)
            correct_weight = 1.0 + LAMBDA * int(active)
        true_loss = c + LAMBDA * hinge
        if not math.isfinite(true_loss):
            raise FloatingPointError("non-finite full margin objective")
        phase("margin_correct_backward")
        (correct_weight * c_tensor / ACCUMULATION_STEPS).backward()
        backwards += 1
        _unchanged(versions)
        del c_tensor, w_tensor

        recompute_error = None
        recompute_relative_error = None
        w_recomputed = None
        if active:
            with _wrong_rng(device, replay):
                w_gradient, w_recomputed = evaluate(wrong, "margin_wrong_recompute", gradient=True)
                recompute_error = abs(w_recomputed - w)
                recompute_relative_error = recompute_error / max(abs(w), RECOMPUTE_ATOL)
                _require(math.isclose(w_recomputed, w, rel_tol=RECOMPUTE_RTOL, abs_tol=RECOMPUTE_ATOL),
                         "wrong no-grad probe and recomputed loss disagree")
                phase("margin_wrong_backward")
                (-LAMBDA * w_gradient / ACCUMULATION_STEPS).backward()
                backwards += 1
                _unchanged(versions)
                del w_gradient
        return dict(
            loss=true_loss, true_loss=true_loss, C=c, W=w, correct_loss=c, wrong_loss=w,
            hinge=hinge, hinge_active=active, identical_actions=identical,
            correct_gradient_weight=correct_weight, wrong_gradient_weight=-LAMBDA if active else 0.0,
            backprop_weighted_loss=correct_weight*c - (LAMBDA*w_recomputed if active else 0.0),
            recomputed_wrong_loss=w_recomputed, recompute_error=recompute_error,
            recompute_relative_error=recompute_relative_error,
            recompute_rtol=RECOMPUTE_RTOL, recompute_atol=RECOMPUTE_ATOL,
            forward_calls=forwards, backward_calls=backwards,
            accumulation_divisor=ACCUMULATION_STEPS,
        )
    except BaseException:
        # No optimizer exists here; discard all partial micro gradients and fail.
        model.zero_grad(set_to_none=True)
        raise
