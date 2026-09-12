"""Bounded, resumable context-only recycling of detached model errors.

Inspired by NVlabs/LongLive model/diffusion.py at commit
6b36d20ec6f7958d29d11a704dfa64611a9f2572 (SVI-style error recycling).
This is an independent, deliberately smaller implementation: no DMD, auxiliary
model, noise/target corruption, position buckets, or distributed buffer sharing.
BFCHW tensors use flow = noise - clean. Only the separate teacher-forcing
``clean_x`` context is changed; its initial latent is always preserved.

Call prepare_context, the training forward, then observe, in that order. The
buffer contains only older forwards when preparing a context. Context replay
samples across sigma buckets (not the target's current noise level), as in the
upstream context branch. The caller must retain its causal visibility mask.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

METHOD_NAME = "context_error_recycling_v1"
SOURCE_COMMIT = "6b36d20ec6f7958d29d11a704dfa64611a9f2572"


@dataclass
class ErrorRecyclingConfig:
    enabled: bool = False
    seed: int = 1419
    context_inject_prob: float = 0.5
    clean_prob: float = 0.25
    error_scale: float = 0.25
    num_buckets: int = 10
    entries_per_bucket: int = 8
    max_buffer_bytes: int = 64 * 1024 * 1024
    warmup_observations: int = 16
    frames_per_block: int = 3

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("error_recycling.enabled must be bool")
        for name in ("seed", "num_buckets", "entries_per_bucket", "max_buffer_bytes",
                     "warmup_observations", "frames_per_block"):
            value = getattr(self, name)
            minimum = 0 if name in ("seed", "warmup_observations") else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"error_recycling.{name} must be an integer >= {minimum}")
        if self.seed >= 2**63 or self.num_buckets > 1000:
            raise ValueError("error_recycling seed/bucket count is out of range")
        if self.max_buffer_bytes > 64 * 1024 * 1024:
            raise ValueError("context-only v1 caps its buffer at 64 MiB")
        for name in ("context_inject_prob", "clean_prob", "error_scale"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"error_recycling.{name} must be finite in [0,1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _shape_and_sigma(value: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != 5 or not value.is_floating_point():
        raise ValueError("latents must be floating point [B,T,C,H,W]")
    if any(size <= 0 for size in value.shape) or value.shape[1] < 2:
        raise ValueError("latents require positive dimensions and at least two frames")
    if not torch.is_tensor(sigma):
        raise ValueError("sigma must be a tensor in [0,1]")
    sigma_cpu = sigma.detach().to(device="cpu", dtype=torch.float32)
    if sigma_cpu.ndim == 5 and tuple(sigma_cpu.shape[2:]) == (1, 1, 1):
        sigma_cpu = sigma_cpu[..., 0, 0, 0]
    if sigma_cpu.ndim == 1 and value.shape[0] == 1:
        sigma_cpu = sigma_cpu.unsqueeze(0)
    if tuple(sigma_cpu.shape) != tuple(value.shape[:2]):
        raise ValueError("sigma must have shape [B,T], [T] for B=1, or [B,T,1,1,1]")
    if not bool(torch.isfinite(sigma_cpu).all()) or bool(((sigma_cpu < 0) | (sigma_cpu > 1)).any()):
        raise ValueError("sigma must be finite in [0,1], not raw 0..1000 timesteps")
    if bool(torch.count_nonzero(sigma_cpu[:, 0])):
        raise ValueError("the protected initial latent must have sigma zero")
    return sigma_cpu


class ContextErrorRecycling:
    """CPU BF16 block bank, independent RNG, bounded FIFO retention.

    The byte limit applies to live stored tensor payloads. Serialization makes a
    detached snapshot, so checkpoint construction may temporarily use one extra
    buffer's worth of CPU memory. No full-video FP32 GPU copy is constructed.
    """

    def __init__(self, config: ErrorRecyclingConfig):
        config.validate()
        self.config = copy.deepcopy(config)
        self._generator = torch.Generator(device="cpu").manual_seed(config.seed)
        self._buckets: list[list[tuple[int, torch.Tensor]]] = [[] for _ in range(config.num_buckets)]
        self._bytes = 0
        self._next_id = 0
        self.observations = 0
        self._dropped = 0
        self._total_injected_forwards = 0
        self._total_injected_blocks = 0
        self._pending = False
        self._last_receipt: dict[str, Any] = {}

    def _blocks(self, frames: int):
        for start in range(1, frames, self.config.frames_per_block):
            yield start, min(frames, start + self.config.frames_per_block)

    def _random(self) -> float:
        return float(torch.rand((), generator=self._generator))

    def prepare_context(self, clean: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        _shape_and_sigma(clean, sigma)
        if self._pending:
            raise RuntimeError("observe must finish the previous prepare_context before another prepare")
        self._pending = True
        context = clean.detach()
        receipt: dict[str, Any] = {"method": METHOD_NAME, "injected": False,
                                  "injected_blocks": 0, "source_entry_ids": [],
                                  "source_buckets": [], "observations_before": self.observations,
                                  "error_rms": 0.0, "reason": "empty"}
        if not self.config.enabled:
            receipt["reason"] = "disabled"
        elif self.observations < self.config.warmup_observations:
            receipt["reason"] = "warmup"
        elif self.config.error_scale == 0 or self.config.context_inject_prob == 0:
            receipt["reason"] = "zero_perturbation"
        elif self._bytes:
            if self._random() < self.config.clean_prob:
                receipt["reason"] = "clean_probability"
            elif self._random() >= self.config.context_inject_prob:
                receipt["reason"] = "injection_probability"
            else:
                squared, elements = 0.0, 0
                for batch_idx in range(clean.shape[0]):
                    for start, end in self._blocks(clean.shape[1]):
                        shape = (end - start, *clean.shape[2:])
                        candidates = [(bucket_idx, entry_id, error)
                                      for bucket_idx, bucket in enumerate(self._buckets)
                                      for entry_id, error in bucket if tuple(error.shape) == shape]
                        if not candidates:
                            continue
                        idx = int(torch.randint(len(candidates), (), generator=self._generator))
                        bucket_idx, entry_id, error = candidates[idx]
                        if not receipt["injected"]:
                            context = context.clone()
                        perturbation = error.to(device=context.device, dtype=context.dtype) * self.config.error_scale
                        context[batch_idx, start:end] += perturbation
                        receipt["injected"] = True
                        receipt["injected_blocks"] += 1
                        receipt["source_entry_ids"].append(entry_id)
                        receipt["source_buckets"].append(bucket_idx)
                        squared += float(error.float().square().sum()) * self.config.error_scale**2
                        elements += error.numel()
                receipt["reason"] = "replayed" if receipt["injected"] else "shape_mismatch"
                receipt["error_rms"] = math.sqrt(squared / elements) if elements else 0.0
        self._last_receipt = copy.deepcopy(receipt)
        return context, receipt

    def _remove_front(self, bucket_idx: int) -> None:
        _, value = self._buckets[bucket_idx].pop(0)
        self._bytes -= value.numel() * value.element_size()
        self._dropped += 1

    def _store(self, bucket_idx: int, error: torch.Tensor) -> None:
        size = error.numel() * error.element_size()
        if size > self.config.max_buffer_bytes:
            self._dropped += 1
            return
        while len(self._buckets[bucket_idx]) >= self.config.entries_per_bucket:
            self._remove_front(bucket_idx)
        while self._bytes + size > self.config.max_buffer_bytes:
            oldest_bucket = min((i for i, bucket in enumerate(self._buckets) if bucket),
                                key=lambda i: self._buckets[i][0][0])
            self._remove_front(oldest_bucket)
        self._buckets[bucket_idx].append((self._next_id, error))
        self._next_id += 1
        self._bytes += size

    @torch.no_grad()
    def observe(self, noisy: torch.Tensor, target: torch.Tensor, pred: torch.Tensor,
                sigma: torch.Tensor, *, clean: torch.Tensor | None = None) -> dict[str, Any]:
        sigma_cpu = _shape_and_sigma(noisy, sigma)
        if not self._pending:
            raise RuntimeError("prepare_context must precede observe: only older errors may be replayed")
        for name, tensor in (("target", target), ("pred", pred), ("clean", clean)):
            if tensor is not None and (not torch.is_tensor(tensor) or tensor.shape != noisy.shape
                                       or not tensor.is_floating_point()):
                raise ValueError(f"{name} must match noisy [B,T,C,H,W]")
        if bool(torch.count_nonzero(target[:, 0])):
            raise ValueError("the protected initial latent must have zero flow target")
        if self.config.enabled:
            for batch_idx in range(noisy.shape[0]):
                for start, end in self._blocks(noisy.shape[1]):
                    index = (batch_idx, slice(start, end))
                    block_sigma = sigma_cpu[batch_idx, start:end].view(-1, 1, 1, 1)
                    # Copy only this small detached block from prediction's device.
                    prediction = pred[index].detach().to(device="cpu", dtype=torch.float32)
                    noisy_cpu = noisy[index].detach().to(device="cpu", dtype=torch.float32)
                    clean_pred = noisy_cpu - block_sigma * prediction
                    clean_cpu = (clean[index].detach().to(device="cpu", dtype=torch.float32)
                                 if clean is not None else noisy_cpu - block_sigma *
                                 target[index].detach().to(device="cpu", dtype=torch.float32))
                    error = (clean_pred - clean_cpu).to(dtype=torch.bfloat16).contiguous()
                    if not bool(torch.isfinite(error).all()):
                        raise FloatingPointError("non-finite model error must not enter the replay buffer")
                    bucket = min(self.config.num_buckets - 1,
                                 int(float(block_sigma.mean()) * self.config.num_buckets))
                    self._store(bucket, error)
        self._total_injected_forwards += int(self._last_receipt.get("injected", False))
        self._total_injected_blocks += int(self._last_receipt.get("injected_blocks", 0))
        self.observations += 1
        self._pending = False
        return self.metrics()

    def metrics(self) -> dict[str, Any]:
        return {"observations": self.observations, "buffer_bytes": self._bytes,
                "buffer_entries": sum(map(len, self._buckets)),
                "filled_buckets": sum(bool(bucket) for bucket in self._buckets),
                "total_added": self._next_id, "total_dropped": self._dropped,
                "total_injected_forwards": self._total_injected_forwards,
                "total_injected_blocks": self._total_injected_blocks,
                "injected": bool(self._last_receipt.get("injected", False)),
                "injected_blocks": int(self._last_receipt.get("injected_blocks", 0)),
                "error_rms": float(self._last_receipt.get("error_rms", 0.0))}

    def state_dict(self) -> dict[str, Any]:
        if self._pending:
            raise RuntimeError("cannot checkpoint an incomplete prepare/observe pair")
        return {"format_version": 1, "method": METHOD_NAME, "source_commit": SOURCE_COMMIT,
                "config": self.config.to_dict(), "rng_state": self._generator.get_state().clone(),
                "observations": self.observations, "next_id": self._next_id,
                "total_injected_forwards": self._total_injected_forwards,
                "total_injected_blocks": self._total_injected_blocks,
                "total_dropped": self._dropped, "last_receipt": copy.deepcopy(self._last_receipt),
                "buckets": [[{"id": entry_id, "error": error.clone()}
                             for entry_id, error in bucket] for bucket in self._buckets]}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if self._pending:
            raise RuntimeError("cannot restore during an incomplete prepare/observe pair")
        if (not isinstance(state, Mapping) or state.get("format_version") != 1
                or state.get("method") != METHOD_NAME or state.get("source_commit") != SOURCE_COMMIT
                or state.get("config") != self.config.to_dict()):
            raise ValueError("error recycling state/config/method mismatch")
        for name in ("observations", "next_id", "total_dropped", "total_injected_forwards",
                     "total_injected_blocks"):
            if isinstance(state.get(name), bool) or not isinstance(state.get(name), int) or state[name] < 0:
                raise ValueError(f"invalid error recycling {name}")
        if (state["total_injected_forwards"] > state["observations"]
                or state["total_injected_blocks"] < state["total_injected_forwards"]):
            raise ValueError("invalid error recycling cumulative injection counts")
        raw = state.get("buckets")
        if not isinstance(raw, list) or len(raw) != self.config.num_buckets:
            raise ValueError("error recycling bucket count mismatch")
        buckets, seen, total = [], set(), 0
        for bucket in raw:
            if not isinstance(bucket, list) or len(bucket) > self.config.entries_per_bucket:
                raise ValueError("error recycling per-bucket limit exceeded")
            restored, previous = [], -1
            for entry in bucket:
                if not isinstance(entry, Mapping):
                    raise ValueError("invalid error recycling entry")
                entry_id, error = entry.get("id"), entry.get("error")
                if (isinstance(entry_id, bool) or not isinstance(entry_id, int)
                        or entry_id <= previous or entry_id >= state["next_id"] or entry_id in seen):
                    raise ValueError("invalid error recycling entry order/id")
                if (not torch.is_tensor(error) or error.device.type != "cpu"
                        or error.dtype != torch.bfloat16 or error.ndim != 4 or error.requires_grad
                        or any(size <= 0 for size in error.shape)
                        or error.shape[0] > self.config.frames_per_block
                        or not bool(torch.isfinite(error).all())):
                    raise ValueError("error recycling entries must be finite detached CPU BF16 blocks")
                total += error.numel() * error.element_size()
                if total > self.config.max_buffer_bytes:
                    raise ValueError("error recycling byte limit exceeded")
                restored.append((entry_id, error.clone().contiguous()))
                previous = entry_id
                seen.add(entry_id)
            buckets.append(restored)
        generator = torch.Generator(device="cpu")
        try:
            generator.set_state(state["rng_state"])
        except (KeyError, TypeError, RuntimeError) as exc:
            raise ValueError("invalid error recycling RNG state") from exc
        if not isinstance(state.get("last_receipt"), dict):
            raise ValueError("invalid error recycling last receipt")
        self._buckets, self._bytes, self._generator = buckets, total, generator
        self.observations = state["observations"]
        self._next_id, self._dropped = state["next_id"], state["total_dropped"]
        self._total_injected_forwards = state["total_injected_forwards"]
        self._total_injected_blocks = state["total_injected_blocks"]
        self._last_receipt = copy.deepcopy(state["last_receipt"])
