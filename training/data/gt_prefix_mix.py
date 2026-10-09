"""Opt-in GT one/two-clean-latent adaptation over unchanged continuous97 caches.

This is NOT generated-history training or error recovery. Two clean latents
represent5 GT RGB frames and leave92 future frames, unlike the inference
candidate's5+96 RGB/26-latent continuation. Old factories are unchanged.
"""
from __future__ import annotations

import copy
import torch
from torch.utils.data import DataLoader

from .action_dataset import _seed64, cache_index_path
from .action_resampled import ResampledActionDataset, sampling_contract

FACTORY = 'training.data.gt_prefix_mix:build_gt_prefix_mix_dataloader'
POLICY = dict(version=1, name='gt_clean_prefix_1_2_mix_v1',
    clean_latents=[1, 2], probabilities=[0.5, 0.5],
    selection='independent_CPU_rng_seed64(data_seed,policy,absolute_sample_index)',
    source='ground_truth_continuous97_cache_prefix_only',
    source_rgb_frames=97, latent_frames=25, condition_rgb_frames=[1, 5], future_rgb_frames=[96, 92],
    action_rows=96, action_alignment='all24_tokens_right_aligned_to_slots1_to24;first4_actions_are_history_for_k2',
    loss='mean_per_sample_future_only_MSE_then_batch_mean',
    self_generated_history=False, error_recovery=False, cache_reencoded=False,
    inference_length_boundary='k2_training5+92_RGB_is_not_inference5+96_RGB_26_latents')


def prefix_contract():
    return copy.deepcopy(POLICY)


def prefix_count(data_seed, sample_index):
    if type(sample_index) is not int or sample_index < 0:
        raise ValueError('prefix selection requires a nonnegative absolute index')
    generator = torch.Generator(device='cpu').manual_seed(
        _seed64(data_seed, POLICY['name'], sample_index))
    return 1 + int(torch.randint(0, 2, (), generator=generator).item())


def validate_prefix_counts(counts, batch_size):
    if (not torch.is_tensor(counts) or counts.shape != (batch_size,)
            or counts.dtype != torch.int64 or not bool(((counts == 1) | (counts == 2)).all())):
        raise ValueError('clean_prefix_count must be int64 [B] with values1 or2')


def validate_prefix_batch(batch):
    target = batch['target_flow']
    counts = batch.get('clean_prefix_count')
    validate_prefix_counts(counts, target.shape[0])
    times = batch['timesteps']
    if target.shape[1] != 25 or tuple(times.shape) != tuple(target.shape[:2]):
        raise ValueError('GT prefix batch requires25 latents and per-frame timesteps')
    mask = torch.arange(25, device=times.device)[None] < counts.to(times.device)[:, None]
    if (not bool(torch.isfinite(times).all()) or bool((times[mask] != 0).any())
            or not bool((times[~mask] > 0).all())
            or bool(torch.count_nonzero(target[mask.to(target.device)]))):
        raise ValueError('GT prefix requires clean t0/zero target and positive future timesteps')


def future_only_mse(prediction, target, counts):
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError('prefix loss requires equal [B,F,C,H,W] predictions and targets')
    validate_prefix_counts(counts, prediction.shape[0])
    if prediction.shape[1] <= 2:
        raise ValueError('prefix loss needs nonempty future frames')
    counts = counts.to(device=prediction.device)
    frame_error = (prediction.float() - target.float()).square().flatten(2).mean(2)
    future = torch.arange(prediction.shape[1], device=prediction.device)[None] >= counts[:, None]
    # Each sample has equal weight despite24 versus23 predicted latent slots.
    return ((frame_error * future).sum(1) / future.sum(1)).mean()


class GTPrefixMixDataset(ResampledActionDataset):
    def __init__(self, *args, **kwargs):
        if kwargs.get('num_frames') != 97:
            raise ValueError('GT prefix mix requires unchanged97-frame feature caches')
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        spec = self.sampling_spec(index)
        # Reuse the original sampler/noise/sigma path verbatim. The independent
        # prefix draw cannot advance its RNG or change window selection.
        result = super().__getitem__(index)
        count = prefix_count(self.data_seed, spec['absolute_sample_index'])
        if count == 2:
            clean = self._cached_window(self.episodes[spec['episode_slot']], spec['window_index'])['clean_latents']
            result['noisy_latents'][:2] = clean[:2]
            result['target_flow'][:2].zero_()
            result['timesteps'][:2] = 0
        result['clean_prefix_count'] = torch.tensor(count, dtype=torch.int64)
        return result


def build_gt_prefix_mix_dataloader(*, config, training):
    if (config.data_factory != FACTORY or config.num_frames != 97
            or not config.precomputed_latents or not config.precomputed_text_embeddings
            or config.prompt_cache_path is not None or training.max_steps != 20):
        raise ValueError('GT prefix pilot requires explicit factory/inline cached97/max_steps20')
    policy = sampling_contract(training)
    dataset_class = GTPrefixMixDataset
    if getattr(config, 'low_sigma_stratification', False):
        from .low_sigma import LowSigmaGTPrefixMixDataset
        dataset_class = LowSigmaGTPrefixMixDataset
    dataset = dataset_class(cache_index_path(config.manifest_path),
        num_samples=policy['virtual_samples'], manifest_path=config.manifest_path,
        split='train', seed=training.seed, timestep_shift=5., prompt_cache_path=None, num_frames=97)
    return DataLoader(dataset, batch_size=training.micro_batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0, drop_last=True)
