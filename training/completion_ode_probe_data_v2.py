"""Independent CPU supervision sources for the fixed 60 + 4 ODE contract.

This module is deliberately not an inference loader. Its four probes reuse the
already pinned TF full-GT validator, never an inference z0 reader. No model or
CUDA initialization is needed to prepare or validate the source manifest.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import torch

import train_completion_ode_sufficiency as entry

require = entry.require
SCHEMA = 'completion_ode_formal64_selection_v2'
PROBE_ORIGIN = 'balanced_qualification4'


def tensor_sha(value):
    from training.studio.balanced_lora_pair_eval import matched
    return matched.tensor_sha(value)


def original_probe_contract(parent):
    """Bind the real immutable TF qualification assets, not new caller flags."""
    from training.studio import completion_causal_sufficiency_eval as supervision
    path = entry.tf.ROOT/'validation-contract.json'
    value = entry.read(path)
    supervision.validate_validation_binding(value)
    require(value['training_data_binding'] == parent['qualified_data_binding'],
            'ODE probes must exclude the actual parent sixty training windows')
    return value, dict(path=str(path), sha256=entry.previous.sha256_file(path))


def load_probe_window(row, parent):
    """Return actual full original GT only to the offline supervised path."""
    from training.studio import completion_causal_sufficiency_eval as supervision
    binding = row.get('source_asset_binding', {})
    require(row.get('cache_origin') == PROBE_ORIGIN
            and binding.get('future_GT_supervision_only') is True
            and binding.get('future_GT_used_for_generation') is False,
            'independent original-GT probe source required')
    contract = entry.small_pin(binding.get('validation_contract'))
    supervision.validate_validation_binding(contract)
    require(contract['training_data_binding'] == parent['qualified_data_binding']
            and contract == entry.read(entry.tf.ROOT/'validation-contract.json'),
            'actual immutable probe/training exclusion contract differs')
    case = row.get('case_id')
    require(case in supervision.VALIDATION_CASES and binding.get('source_window') == contract['windows'][case],
            'probe original source cannot be reselected')
    original = contract['windows'][case]
    require(all(row.get(key) == original[key] for key in
                ('episode_id', 'window_index', 'source_start', 'target')),
            'probe exact source offsets changed')
    clean, text, actions = supervision.load_supervised_window(original)
    require(tensor_sha(clean) == original['supervised_clean_latents_sha256_float32'],
            'full independent probe GT hash changed')
    return dict(clean_latents=clean.float().clone(), prompt_embeds=text.float().clone(),
                actions=actions.float().clone())


def trajectory_seed(row):
    fields = entry.previous.common.data_module().IDENTITY_FIELDS
    text = entry.TRAJECTORY_NAMESPACE+'|'+'|'.join(str(row[key]) for key in fields)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little') % (2**63-1)


class SourcePool:
    """Exact original sources. Never pass this object to a generator."""
    def __init__(self, parent, selection=None):
        entry.validate_parent_binding(parent)
        self.parent = parent
        self.train = entry.tf.build_dataset(parent['qualified_data_binding'])
        module = entry.previous.common.data_module()
        self.train_rows = {module.window_key(row): row for row in self.train.episodes}
        self.selection = selection
        if selection is not None:
            entry.validate_probe_manifest(selection, entry.choose_exact_windows(parent['qualified_data_binding']))

    def load(self, row):
        if row.get('split') == 'probe':
            value = load_probe_window(row, self.parent)
        else:
            key = entry.previous.common.data_module().window_key(row)
            require(row.get('split') == 'train' and key in self.train_rows,
                    'source absent from the actual reviewed sixty windows')
            source = self.train_rows[key]
            value = self.train._cached_window(source, source['window_index'])
        require(set(value) == {'clean_latents', 'actions', 'prompt_embeds'}
                and value['clean_latents'].shape == (25, 48, 30, 52)
                and value['actions'].shape == (96, 8)
                and value['prompt_embeds'].ndim == 2 and value['prompt_embeds'].shape[-1] == 4096
                and all(t.device.type == 'cpu' and bool(torch.isfinite(t).all()) for t in value.values()),
                'finite exact97 CPU source tensors required')
        require(bool(((value['actions'] == 0) | (value['actions'] == 1)).all()), 'binary original actions required')
        return {key: value[key].detach().float().clone() for key in value}


def load_negative(parent):
    from training.studio import balanced_fresh_eval as source
    from training.studio.quality_probe import load_negative as actual_loader
    binding = source.negative_binding()
    config = entry.tf.load_config(parent['parent_config'])
    value, metadata = actual_loader(binding['path'], config.model.base_model_path)
    require(value.device.type == 'cpu' and value.ndim == 2 and value.shape[-1] == 4096
            and bool(torch.isfinite(value).all()), 'actual original negative embedding required')
    return value.detach().bfloat16().clone(), binding, metadata


def make_selection(parent):
    """CPU-only, exclusive selection input: all sources checked before reserve."""
    contract, pin = original_probe_contract(parent)
    train = entry.choose_exact_windows(parent['qualified_data_binding'])
    probes = []
    for slot, case in enumerate(('pair-j', 'pair-l', 'holdout-j', 'holdout-l')):
        source = copy.deepcopy(contract['windows'][case])
        probes.append(dict(cache_origin=PROBE_ORIGIN, episode_id=source['episode_id'],
            window_index=source['window_index'], source_start=source['source_start'], target=source['target'],
            split='probe', slot=slot, case_id=case, source_asset_binding=dict(validation_contract=pin,
                source_window=source, future_GT_supervision_only=True, future_GT_used_for_generation=False)))
    result = dict(schema=SCHEMA, training_rows=train, probe_rows=probes,
        independent_final_test=False, repeated_development_qualification=True,
        future_GT_used_for_generation=False, teacher_binding=copy.deepcopy(parent))
    entry.validate_probe_manifest(result, train)
    return result


def preflight_selection(parent, selection):
    """Read all 64 real sources and pin tensors; zero model/CUDA activity."""
    pool = SourcePool(parent, selection)
    negative, binding, metadata = load_negative(parent)
    rows = []
    for global_slot, wanted in enumerate(selection['training_rows'] + selection['probe_rows']):
        value = pool.load(wanted)
        rows.append(dict(**copy.deepcopy(wanted), global_slot=global_slot,
            trajectory_seed=trajectory_seed(wanted), clean_context_sha256_float32=tensor_sha(value['clean_latents']),
            action_sha256_float32=tensor_sha(value['actions']), positive_text_sha256=tensor_sha(value['prompt_embeds']),
            negative_text_sha256=tensor_sha(negative), future_GT_supervision_only=True))
    require(len(rows) == 64 and len({r['trajectory_seed'] for r in rows}) == 64,
            'fixed64 unique source/noise identities required')
    return dict(rows=rows, negative_binding=binding, negative_metadata=metadata,
                negative_text_sha256=tensor_sha(negative), CPU_sources_verified=64,
                model_loaded=False, cuda_initialized=False)
