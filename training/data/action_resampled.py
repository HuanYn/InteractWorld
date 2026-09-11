"""Absolute-index action-teacher sampling over the unchanged feature cache.

This is an opt-in factory, not a silent change to the legacy dataset/resume
contract. A run has enough virtual samples for its full optimizer-step budget,
so crossing the physical cache length never resets its diffusion RNG index.
The sample at an absolute index is independent of worker state and run length.
"""

from __future__ import annotations

from functools import lru_cache
from operator import index as integer_index
from typing import Any

import torch
from torch.utils.data import DataLoader

from .action_dataset import PrecomputedActionDataset, _seed64, cache_index_path

SAMPLING_NAMESPACE = "action_resampled_absolute_v1"
INDEX_POLICY = "absolute_index_episode_round_robin_window_permutation_v1"


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = integer_index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def sampling_contract(training: Any) -> dict[str, Any]:
    """Serializable policy metadata; no cache reads or CUDA queries."""
    steps = _positive_integer(training.max_steps, "max_steps")
    accumulation = _positive_integer(
        training.gradient_accumulation_steps, "gradient_accumulation_steps"
    )
    batch = _positive_integer(training.micro_batch_size, "micro_batch_size")
    return {
        "version": 1,
        "namespace": SAMPLING_NAMESPACE,
        "index_policy": INDEX_POLICY,
        "training_seed": int(training.seed),
        "data_seed": _seed64(int(training.seed), SAMPLING_NAMESPACE),
        "virtual_samples": steps * accumulation * batch,
        "micro_batches": steps * accumulation,
        "first_absolute_sample_index": 0,
        "last_absolute_sample_index": steps * accumulation * batch - 1,
        "diffusion_seed_policy": "seed64(data_seed, diffusion, absolute_sample_index)",
        "window_sampling": "per_episode_without_replacement_within_window_epoch",
    }


class ResampledActionDataset(PrecomputedActionDataset):
    """Episode-balanced virtual views with freshly indexed noise and timesteps.

    Each episode's windows are visited without replacement before reshuffling.
    Noise and the timestep draw use a separate RNG seeded by absolute index;
    a timestep value can legitimately collide between different random draws.
    Existing cache/receipt/prompt checks and read-only shard loading are inherited.
    """

    def __init__(self, index_path, *, num_samples: int, **kwargs: Any) -> None:
        self.num_samples = _positive_integer(num_samples, "num_samples")
        super().__init__(index_path, **kwargs)
        self.data_seed = _seed64(self.seed, SAMPLING_NAMESPACE)

    def __len__(self) -> int:
        return self.num_samples

    @lru_cache(maxsize=2048)
    def _window_order(self, episode_id: str, count: int, epoch: int) -> tuple[int, ...]:
        return tuple(sorted(
            range(count),
            key=lambda window: (
                _seed64(self.data_seed, "window-permutation", episode_id, epoch, window),
                window,
            ),
        ))

    def sampling_spec(self, index: int) -> dict[str, Any]:
        """Resolve an index without loading shards or consuming global RNG."""
        if isinstance(index, bool):
            raise TypeError("sample index must be an integer, not bool")
        index = integer_index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        cycle, slot = divmod(index, len(self.episodes))
        episode = self.episodes[slot]
        count = int(episode["num_windows"])
        epoch, position = divmod(cycle, count)
        window = self._window_order(episode["episode_id"], count, epoch)[position]
        return {
            "absolute_sample_index": index,
            "episode_slot": slot,
            "episode_id": episode["episode_id"],
            "episode_cycle": cycle,
            "window_epoch": epoch,
            "window_index": window,
            "diffusion_seed": _seed64(self.data_seed, "diffusion", index),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        spec = self.sampling_spec(index)
        cached = self._cached_window(
            self.episodes[spec["episode_slot"]], spec["window_index"]
        )
        clean = cached.pop("clean_latents")
        generator = torch.Generator(device="cpu").manual_seed(spec["diffusion_seed"])
        noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
        schedule_index = int(torch.randint(0, 1000, (1,), generator=generator).item())
        sigma_linear = 1.0 - schedule_index / 1000.0
        sigma = self.timestep_shift * sigma_linear / (
            1.0 + (self.timestep_shift - 1.0) * sigma_linear
        )
        noisy = (1.0 - sigma) * clean + sigma * noise
        target = noise - clean
        # Preserve the legacy TI2V clean-first-latent flow objective exactly.
        noisy[0] = clean[0]
        target[0].zero_()
        timesteps = torch.full((clean.shape[0],), sigma * 1000.0, dtype=torch.float32)
        timesteps[0] = 0.0
        return {
            "noisy_latents": noisy,
            "target_flow": target,
            "timesteps": timesteps,
            **cached,
        }


def build_resampled_action_teacher_dataloader(*, config: Any, training: Any) -> DataLoader:
    """Opt-in absolute-index factory, sized to the complete configured run.

    ``len(loader) == max_steps * accumulation``. Thus the trainer's legacy
    resume skip modulo this length is the identity for every unfinished
    checkpoint. Extending max_steps also preserves all prior index meanings.
    No raw video, VAE, text encoder, cache writes, or mutable worker epoch state.
    """
    if not getattr(config, "precomputed_latents", False):
        raise ValueError("action teacher requires precomputed_latents=true")
    if not getattr(config, "precomputed_text_embeddings", False):
        raise ValueError("action teacher requires precomputed_text_embeddings=true")
    policy = sampling_contract(training)
    dataset = ResampledActionDataset(
        cache_index_path(config.manifest_path),
        num_samples=policy["virtual_samples"],
        manifest_path=config.manifest_path,
        split="train",
        seed=training.seed,
        timestep_shift=5.0,
        prompt_cache_path=getattr(config, "prompt_cache_path", None),
    )
    return DataLoader(
        dataset,
        batch_size=training.micro_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        drop_last=True,
    )
