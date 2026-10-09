"""Opt-in, small first-condition perturbation; not self-generated-history training.

The clean control uses the exact same absolute-index sampling and diffusion RNG.
Only the first (conditioning) latent is perturbed on half the visits. Future
latents, flow targets, actions and time labels stay identical. This tests local
conditioning robustness, not geometric consistency or an on-policy objective.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .action_dataset import _seed64, cache_index_path
from .action_resampled import ResampledActionDataset, sampling_contract

POLICY = dict(version=1, name="first_condition_gaussian_clean_mix_v1",
              standard_deviation=.03, clean_fraction=.5,
              scope="first_latent_only", future_targets_unchanged=True,
              self_generated_history=False)


def perturb_condition(batch, *, sample_index, episode_count, seed):
    """Alternate clean/corrupt visits per episode, using independent CPU RNG."""
    if episode_count <= 0 or sample_index < 0:
        raise ValueError("invalid absolute sample index or episode count")
    result = dict(batch)
    if (sample_index // episode_count) % 2 == 0:
        return result
    source = batch["noisy_latents"]
    if source.device.type != "cpu" or source.ndim != 4:
        raise ValueError("expected a CPU unbatched FCHW sample")
    if float(batch["timesteps"][0]) != 0 or bool(torch.count_nonzero(batch["target_flow"][0])):
        raise ValueError("the TI2V first-slot time and target must remain zero")
    generator = torch.Generator(device="cpu").manual_seed(
        _seed64(seed, POLICY["name"], sample_index))
    noise = torch.randn(source[0].shape, generator=generator, dtype=torch.float32)
    result["noisy_latents"] = source.clone()
    result["noisy_latents"][0] += noise.to(source.dtype) * POLICY["standard_deviation"]
    return result


class ConditionRobustnessDataset(ResampledActionDataset):
    def __getitem__(self, index):
        spec = self.sampling_spec(index)
        return perturb_condition(super().__getitem__(index),
            sample_index=spec["absolute_sample_index"], episode_count=len(self.episodes), seed=self.seed)


def build_condition_robustness_dataloader(*, config, training):
    if not (config.precomputed_latents and config.precomputed_text_embeddings):
        raise ValueError("this factory only accepts precomputed features")
    if config.num_frames != 49:
        raise ValueError("this bounded pilot fixes the original49-frame contract")
    policy = sampling_contract(training)
    dataset = ConditionRobustnessDataset(cache_index_path(config.manifest_path),
        num_samples=policy["virtual_samples"], manifest_path=config.manifest_path,
        split="train", seed=training.seed, timestep_shift=5.,
        prompt_cache_path=config.prompt_cache_path, num_frames=config.num_frames)
    return DataLoader(dataset, batch_size=training.micro_batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0, drop_last=True)
