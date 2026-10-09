"""One opt-in time-sampling change; original noise, prefix and target stay intact."""
from __future__ import annotations

import copy
from functools import lru_cache
import torch

from .gt_prefix_mix import FACTORY, GTPrefixMixDataset
from .action_dataset import _seed64

INIT_MODE = 'length_gt_prefix_low_sigma_transition_weights_only'
POLICY = dict(version=1, name='gt97_low_sigma_two_of_eight_block_rng_v1',
    period=8, selected_per_block=2, selection_seed=42,
    selected_residues='independent_CPU_randperm(8)[:2];seed64(42,policy_name,absolute_sample_index//8)',
    selected_sigma='0.2 * u',
    original_sigma='5 * u / (1 + 4 * u)', u='1 - original_schedule_index / 1000',
    schedule_indices=1000, original_rng_draw_order_unchanged=True,
    noise_data_prefix_target_unchanged=True, loss_reweighting=False)


def low_sigma_contract():
    return copy.deepcopy(POLICY)


def validate_opt_in(config, enabled):
    if (type(enabled) is not bool or type(config.data.low_sigma_stratification) is not bool
            or enabled != config.data.low_sigma_stratification):
        raise ValueError('low_sigma requires an explicit flag matching its serialized data configuration')
    if enabled and (config.data.data_factory != FACTORY or config.data.num_frames != 97
            or config.training.lora_parameter_dtype != 'float32' or config.training.max_steps != 20
            or config.training.micro_batch_size != 1 or config.training.gradient_accumulation_steps != 8
            or config.training.seed != 42 or config.optimizer.adapter_lr != 0 or config.optimizer.lora_lr != 2e-6):
        raise ValueError('low_sigma is limited to FP32 GT97/fixed-adapter/LoRA2e-6/20steps/accum8')


def validate_initialization(config, initialization):
    enabled = config.data.low_sigma_stratification
    validate_opt_in(config, enabled)
    if enabled:
        if initialization.get('mode') != INIT_MODE or initialization.get('low_sigma_contract') != low_sigma_contract():
            raise ValueError('low_sigma checkpoint initialization or sampling contract differs')
    elif 'low_sigma_contract' in initialization or initialization.get('mode') == INIT_MODE:
        raise ValueError('legacy training cannot relabel a low_sigma checkpoint')


@lru_cache(maxsize=20)
def selected_residues(block_index):
    if type(block_index) is not int or not 0 <= block_index < 20:
        raise ValueError('low_sigma block index must be within the fixed20-step run')
    generator = torch.Generator(device='cpu').manual_seed(_seed64(42, POLICY['name'], block_index))
    return tuple(sorted(torch.randperm(8, generator=generator)[:2].tolist()))


def sigma_observation(schedule_index, sample_index):
    if type(schedule_index) is not int or not 0 <= schedule_index < 1000:
        raise ValueError('invalid low_sigma schedule index')
    if type(sample_index) is not int or not 0 <= sample_index < 160:
        raise ValueError('low_sigma sample index must be within the fixed160-sample run')
    u = 1.0 - schedule_index / 1000.0
    original = 5.0 * u / (1.0 + 4.0 * u)
    selected = sample_index % 8 in selected_residues(sample_index // 8)
    return dict(sample_index=sample_index, schedule_index=schedule_index, original_sigma=original,
                sigma=0.2 * u if selected else original, stratified=selected)


class LowSigmaGTPrefixMixDataset(GTPrefixMixDataset):
    def noise_level(self, schedule_index, sample_index):
        if self.timestep_shift != 5.0:
            raise ValueError('low_sigma candidate keeps the original shift5 mapping')
        row = sigma_observation(schedule_index, sample_index)
        return row['sigma'], {'low_sigma_sample': row}


def observed_sample(batch, expected_sample_index):
    """Validate the real collated CPU batch, not a predicted sampling histogram."""
    raw = batch.get('low_sigma_sample')
    if not isinstance(raw, dict):
        raise ValueError('low_sigma batch is missing its actual sampling observation')
    row = {key: value.item() if torch.is_tensor(value) and value.numel() == 1 else value
           for key, value in raw.items()}
    if row != sigma_observation(row.get('schedule_index'), expected_sample_index):
        raise ValueError('low_sigma observation differs from the absolute sample or fixed policy')
    count = int(batch['clean_prefix_count'].item())
    times = batch['timesteps']
    expected = torch.full_like(times, row['sigma'] * 1000)
    expected[:, :count] = 0
    if not torch.equal(times, expected):
        raise ValueError('low_sigma observation does not match the actual model timesteps')
    return row


def summarize_samples(rows):
    if (len(rows) != 8 or rows[0]['sample_index'] % 8 != 0
            or [r['sample_index'] for r in rows] != list(range(rows[0]['sample_index'], rows[0]['sample_index'] + 8))):
        raise ValueError('low_sigma metrics require the actual eight consecutive accumulated samples')
    if sum(row['stratified'] for row in rows) != 2:
        raise ValueError('low_sigma accumulated batch must contain exactly two stratified samples')
    return dict(samples=copy.deepcopy(rows), sample_count=8, stratified_samples=2,
                sigma_le_02=sum(row['sigma'] <= .2 for row in rows),
                original_sigma_le_02=sum(row['original_sigma'] <= .2 for row in rows))
