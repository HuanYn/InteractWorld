"""Explicit sampler conversion must not become a general resume bypass."""
from __future__ import annotations

import copy
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import train_action_teacher as entry
from training.config import load_config
from training.runtime import load_checkpoint, sha256_file

CONFIG = Path(__file__).parents[1] / 'configs/train/action_teacher_5090_repair_r005.yaml'


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.act_control_adapter = nn.Linear(1, 1)


def fixture(tmp_path):
    cfg = load_config(CONFIG)
    cfg.training.output_dir = str(tmp_path / 'new-sampling-run')
    cfg.training.max_steps = 260
    source_config = cfg.to_dict()
    source_config['training'].update(max_steps=1040, output_dir=str(tmp_path / 'source-run'))
    cfg.data.data_factory = entry.RESAMPLED_ACTION_FACTORY
    hashes = dict(dataset_manifest='dataset', feature_index='features', feature_receipt='features-receipt',
                  prompt_cache='static-prompt', prompt_cache_receipt='static-prompt-receipt',
                  training_config='new-config')
    model = TinyTeacher()
    state = {name: torch.full_like(value, 0.25) for name, value in model.state_dict().items()}
    payload = dict(format_version=1, stage='action_teacher_lora_v1', step=1040,
        micro_batches_consumed=8320, config=source_config,
        manifest_hashes={**hashes, 'training_config': 'source-config'}, trainable_model=state,
        optimizer={'must_not_restore': True}, torch_rng_state=torch.tensor([255], dtype=torch.uint8),
        cuda_rng_state_all=[torch.tensor([255], dtype=torch.uint8)],
        initialization={'mode': 'original-parent-lineage'})
    path = tmp_path / 'source1040.pt'
    torch.save(payload, path)
    return cfg, hashes, path, payload


def load(cfg, hashes, path):
    return entry._load_sampling_transition_checkpoint(path, config=cfg, current_manifest_hashes=hashes)


@pytest.mark.parametrize('other', ['--resume', '--initialize-from', '--warm-start-from'])
def test_explicit_cli_is_mutually_exclusive(other):
    assert entry.parse_args(['--sampling-transition-from', 'source.pt']).sampling_transition_from == 'source.pt'
    with pytest.raises(SystemExit):
        entry.parse_args(['--sampling-transition-from', 'source.pt', other, 'other.pt'])


def test_weights_only_and_truthful_transition_metadata(tmp_path):
    cfg, hashes, path, payload = fixture(tmp_path)
    original_sha = sha256_file(path)
    source_config = copy.deepcopy(payload['config'])
    target_config = cfg.to_dict()
    state, init = load(cfg, hashes, path)
    assert init['mode'] == 'sampling_transition_weights_only'
    assert init['source_step'] == 1040
    assert init['start_step'] == init['micro_batches_consumed'] == 0
    assert not init['optimizer_restored'] and not init['rng_restored']
    assert init['source_config'] == source_config
    assert init['source_manifest_hashes'] == payload['manifest_hashes']
    assert init['target_manifest_hashes'] == hashes
    assert init['source_initialization'] == payload['initialization']
    assert init['sampling_transition']['source_factory'] == entry.LEGACY_ACTION_FACTORY
    assert init['sampling_transition']['target_factory'] == entry.RESAMPLED_ACTION_FACTORY
    assert init['sampling_transition']['changed_data_fields'] == ['data_factory']
    assert init['sampling_transition']['source_micro_batches_consumed'] == 8320
    assert init['sampling_transition']['target_sampling_namespace'] == 'action_resampled_absolute_v1'
    assert init['sampling_transition']['target_virtual_sample_count'] == 2080
    assert init['sampling_transition']['target_absolute_sample_start'] == 0
    assert not init['sampling_transition']['source_position_restored']
    assert not init['prompt_transition']['changed']
    assert set(state) == set(payload['trainable_model'])
    assert all(torch.equal(value, payload['trainable_model'][name]) for name, value in state.items())
    assert sha256_file(path) == original_sha == init['sha256']
    assert cfg.to_dict() == target_config


@pytest.mark.parametrize(('section', 'field', 'value'), [
    ('model', 'action_scale', 0.5), ('model', 'gradient_checkpointing', False),
    ('model', 'base_model_path', '/different-base'), ('model', 'lora_alpha', 32),
    ('optimizer', 'adapter_lr', 0.1), ('optimizer', 'lora_lr', 0.2),
    ('optimizer', 'betas', (0.8, 0.99)),
    ('data', 'prompt_cache_path', '/different/static.pt'), ('data', 'height', 256),
    ('data', 'num_workers', 0), ('data', 'manifest_path', '/different/train.jsonl'),
    ('training', 'seed', 43), ('training', 'gradient_accumulation_steps', 4),
    ('training', 'checkpoint_every', 50),
])
def test_rejects_unrelated_configuration_change(tmp_path, section, field, value):
    cfg, hashes, path, _ = fixture(tmp_path)
    setattr(getattr(cfg, section), field, value)
    with pytest.raises(ValueError, match=f'{section} contract'):
        load(cfg, hashes, path)


@pytest.mark.parametrize('side', ['source', 'target'])
@pytest.mark.parametrize('factory', [entry.LEGACY_ACTION_FACTORY, entry.RESAMPLED_ACTION_FACTORY, 'arbitrary:loader'])
def test_only_exact_factory_pair_allowed(tmp_path, side, factory):
    cfg, hashes, path, payload = fixture(tmp_path)
    if side == 'source':
        payload['config']['data']['data_factory'] = factory
        torch.save(payload, path)
        valid = factory == entry.LEGACY_ACTION_FACTORY
    else:
        cfg.data.data_factory = factory
        valid = factory == entry.RESAMPLED_ACTION_FACTORY
    if valid:
        load(cfg, hashes, path)
    else:
        with pytest.raises(ValueError, match='exact legacy-to-resampled factory pair'):
            load(cfg, hashes, path)


@pytest.mark.parametrize('key', ['dataset_manifest', 'feature_index', 'feature_receipt',
                               'prompt_cache', 'prompt_cache_receipt'])
def test_rejects_artifact_changes(tmp_path, key):
    cfg, hashes, path, _ = fixture(tmp_path)
    hashes[key] = 'changed'
    with pytest.raises(ValueError, match='mismatch|prompt contract'):
        load(cfg, hashes, path)


def test_rejects_fake_old_config_hash_and_same_output(tmp_path):
    cfg, hashes, path, payload = fixture(tmp_path)
    hashes['training_config'] = 'source-config'
    with pytest.raises(ValueError, match='new configuration hash'):
        load(cfg, hashes, path)
    hashes['training_config'] = 'new-config'
    cfg.training.output_dir = payload['config']['training']['output_dir']
    with pytest.raises(ValueError, match='fresh output directory'):
        load(cfg, hashes, path)


@pytest.mark.parametrize(('field', 'value'), [('stage', 'causal_teacher_forcing_v1'),
    ('step', True), ('step', 0), ('micro_batches_consumed', 0), ('trainable_model', {})])
def test_rejects_invalid_parent(tmp_path, field, value):
    cfg, hashes, path, payload = fixture(tmp_path)
    payload[field] = value
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load(cfg, hashes, path)


def test_ordinary_warm_start_and_strict_resume_remain_strict(tmp_path):
    cfg, hashes, path, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='data contract differs'):
        entry._load_warm_start_checkpoint(path, config=cfg, current_manifest_hashes=hashes)
    with pytest.raises(ValueError, match='manifest hashes'):
        load_checkpoint(path, expected_manifest_hashes=hashes)


def test_launch_fresh_optimizer_rng_step_and_checkpoint_lineage(tmp_path, monkeypatch):
    cfg, hashes, path, payload = fixture(tmp_path)
    cfg.training.max_steps = 1
    model = TinyTeacher()
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    def reject(*args, **kwargs):
        raise AssertionError('sampler transition must not restore parent optimizer/RNG')
    monkeypatch.setattr(optimizer, 'load_state_dict', reject)
    monkeypatch.setattr(torch, 'set_rng_state', reject)
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', reject)
    monkeypatch.setattr(entry, 'validate_confirmation', lambda *a: None)
    monkeypatch.setattr(entry, 'query_dedicated_gpu', lambda **k: SimpleNamespace(as_dict=lambda: {}))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    for name in ('set_device', 'reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    for name in ('memory_allocated', 'memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, lambda *a: 0)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *a: 'CPU stand-in')
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    monkeypatch.setattr(entry, 'peak_vram_bytes', lambda *a: 0)
    seeds, batches, factories = [], [], []
    monkeypatch.setattr(entry, 'seed_everything', seeds.append)
    monkeypatch.setattr(entry, '_manifest_hashes', lambda *a: hashes)
    monkeypatch.setattr(entry, '_build_model', lambda *a: (model, SimpleNamespace(
        trainable_parameters=2, total_parameters=2, replaced_linear_layers=[])))
    monkeypatch.setattr(entry, '_build_optimizer', lambda *a: optimizer)
    def factory(spec):
        factories.append(spec)
        return lambda **k: list(range(8))
    monkeypatch.setattr(entry, '_import_factory', factory)
    monkeypatch.setattr(torch, 'autocast', lambda *a, **k: nullcontext())
    def loss(model, batch, *args):
        batches.append(batch)
        assert all(torch.equal(p, payload['trainable_model'][name]) for name, p in model.named_parameters())
        return sum(p.square().sum() for p in model.parameters())
    monkeypatch.setattr(entry, '_forward_loss', loss)
    entry.launch(cfg, CONFIG, None, None, confirmed_gpu_index=0, confirmed_gpu_uuid='CPU',
        confirmed_at_utc='CPU', allocation_profile='CPU', sampling_transition_from=str(path))
    assert seeds == [42] and batches == list(range(8))
    assert factories == [entry.RESAMPLED_ACTION_FACTORY]
    saved = torch.load(Path(cfg.training.output_dir) / 'checkpoints/step-0000001.pt', weights_only=False)
    assert saved['step'] == 1 and saved['micro_batches_consumed'] == 8
    assert saved['optimizer']['state'] == {}
    assert saved['initialization']['start_step'] == 0
    assert saved['initialization']['source_step'] == 1040
    assert saved['config']['data']['data_factory'] == entry.RESAMPLED_ACTION_FACTORY
    metadata = json.loads((Path(cfg.training.output_dir) / 'run_metadata.json').read_text())
    assert metadata['initialization'] == json.loads(json.dumps(saved['initialization']))


def test_conflicting_api_modes_rejected_before_gpu_gate(monkeypatch):
    monkeypatch.setattr(entry, 'validate_confirmation', lambda *a: pytest.fail('must not reach GPU gate'))
    cfg = load_config(CONFIG)
    with pytest.raises(ValueError, match='mutually exclusive'):
        entry.launch(cfg, CONFIG, 'old.pt', None, sampling_transition_from='source.pt',
            confirmed_gpu_index=0, confirmed_gpu_uuid='CPU', confirmed_at_utc='CPU', allocation_profile='CPU')
