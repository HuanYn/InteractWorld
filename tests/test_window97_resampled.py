"""97-frame resampling regression: CPU cache fixtures, not GPU quality evidence."""
import copy
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import train_action_teacher as trainer
from training.config import load_config
from training.data import action_dataset as cache_data
from training.data.action_resampled import (
    ResampledActionDataset, build_resampled_action_teacher_dataloader,
)


def test_real97_factory_shape_static_prompt_and_read_only_cache(tmp_path, monkeypatch):
    from test_window97_data import setup_episode, Encoder, fake_decoder
    from scripts import cache_abot_window97_features as builder

    manifests = tmp_path / 'manifests'
    manifests.mkdir()
    manifest, _ = setup_episode(manifests, monkeypatch)
    root = tmp_path / 'features'
    receipt = builder.cache_window97_manifest(manifest, root, encoder=Encoder(),
        decoder=fake_decoder, max_total_windows=2, max_windows_per_episode=2)
    assert receipt['windows'] == 2
    files = [manifest, *sorted(root.rglob('*'))]
    hashes = {path: cache_data.sha256_file(path) for path in files if path.is_file()}
    config = SimpleNamespace(manifest_path=str(manifest), num_frames=97, num_workers=0,
        precomputed_latents=True, precomputed_text_embeddings=True, prompt_cache_path=None)
    training = SimpleNamespace(max_steps=2, gradient_accumulation_steps=8, micro_batch_size=1, seed=42)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    loader = build_resampled_action_teacher_dataloader(config=config, training=training)
    assert loader.dataset.num_frames == 97 and len(loader) == 16
    first, next_pass = loader.dataset[0], loader.dataset[2]
    assert first['noisy_latents'].shape == first['target_flow'].shape == (25,48,30,52)
    assert first['actions'].shape == (96,8) and first['timesteps'].shape == (25,)
    assert first['prompt_embeds'].shape == (3,4096)
    assert torch.all(first['noisy_latents'][0] == .5)
    assert torch.count_nonzero(first['target_flow'][0]) == 0 and first['timesteps'][0] == 0
    assert torch.equal(first['prompt_embeds'], next_pass['prompt_embeds'])
    assert not torch.equal(first['target_flow'][1:], next_pass['target_flow'][1:])
    assert {path: cache_data.sha256_file(path) for path in hashes} == hashes


@pytest.fixture
def index256(monkeypatch):
    def initialize(self, index_path, *, seed=42, timestep_shift=5., num_frames=49, **kwargs):
        self.seed, self.timestep_shift, self.num_frames = seed, timestep_shift, num_frames
        self.episodes = [{'episode_id': f'train-{i:03d}', 'num_windows':2} for i in range(128)]
        self.samples_per_episode = 2
    monkeypatch.setattr(cache_data.PrecomputedActionDataset, '__init__', initialize)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    return ResampledActionDataset('metadata-only-256-window-fixture', num_samples=2048, seed=42, num_frames=97)


def test_256_windows_without_replacement_and_fresh_diffusion_each_pass(index256):
    dataset = index256
    all_seeds = set()
    for epoch in range(4):
        rows = [dataset.sampling_spec(i) for i in range(epoch*256, (epoch+1)*256)]
        assert len({(row['episode_id'], row['window_index']) for row in rows}) == 256
        assert {row['window_epoch'] for row in rows} == {epoch}
        assert len({row['diffusion_seed'] for row in rows}) == 256
        assert not all_seeds.intersection(row['diffusion_seed'] for row in rows)
        all_seeds.update(row['diffusion_seed'] for row in rows)
    assert len(all_seeds) == 1024


@pytest.mark.parametrize('batch_size', [1,2])
def test_resume_uses_absolute_micro_batch_position_not_physical_cache(index256, monkeypatch, batch_size):
    config = SimpleNamespace(manifest_path='/unused/manifests/train.jsonl', num_workers=0, num_frames=97,
        precomputed_latents=True, precomputed_text_embeddings=True, prompt_cache_path=None)
    training = SimpleNamespace(max_steps=80, gradient_accumulation_steps=8, micro_batch_size=batch_size, seed=42)
    monkeypatch.setattr(ResampledActionDataset, '__getitem__',
        lambda self, index: self.sampling_spec(index)['absolute_sample_index'])
    loader = build_resampled_action_teacher_dataloader(config=config, training=training)
    assert len(loader) == 640 and loader.dataset.num_frames == 97
    _, iterator = trainer._iterator_at_micro_batch(loader, 40*8)
    batches = list(islice(iterator,2))
    assert torch.cat(batches).tolist() == list(range(320*batch_size, 322*batch_size))
    training.max_steps = 120
    extended = build_resampled_action_teacher_dataloader(config=config, training=training)
    assert loader.dataset.sampling_spec(511) == extended.dataset.sampling_spec(511)
    _, iterator = trainer._iterator_at_micro_batch(extended, 100*8)
    assert next(iterator).tolist() == list(range(800*batch_size, 801*batch_size))


def test_sampling_transition_retains_actual97_length_transition_lineage(tmp_path):
    from test_window97_trainer import transition_fixture
    cfg97, hashes97, original_path, _ = transition_fixture(tmp_path)
    state97, initialization97 = trainer._load_length_transition_checkpoint(original_path,
        expected_sha256=cache_data.sha256_file(original_path), config=cfg97, current_manifest_hashes=hashes97)
    source_config = cfg97.to_dict()
    source_config['training']['max_steps'] = 2000
    native97 = dict(format_version=1, stage='action_teacher_lora_v1', step=100,
        micro_batches_consumed=800, config=source_config, manifest_hashes=hashes97,
        trainable_model=state97, initialization=initialization97)
    path = tmp_path/'native97-step100.pt'
    torch.save(native97, path)
    before = cache_data.sha256_file(path)
    cfg97.data.data_factory = trainer.RESAMPLED_ACTION_FACTORY
    cfg97.training.output_dir = str(tmp_path/'new-resampled97')
    target_hashes = {**hashes97, 'training_config':'new-resampled-training-config'}
    state, lineage = trainer._load_sampling_transition_checkpoint(path, config=cfg97,
        current_manifest_hashes=target_hashes)
    assert lineage['source_step'] == 100 and lineage['sha256'] == before
    assert lineage['source_initialization'] == initialization97
    assert lineage['source_initialization']['length_transition']['target_rgb_frames'] == 97
    assert lineage['source_initialization']['length_transition']['source_rgb_frames'] == 49
    assert lineage['source_config']['data']['num_frames'] == 97
    assert cfg97.data.num_frames == 97 and cfg97.data.prompt_cache_path is None
    assert lineage['source_manifest_hashes'] == hashes97
    assert lineage['target_manifest_hashes'] == target_hashes
    assert lineage['start_step'] == lineage['micro_batches_consumed'] == 0
    assert not lineage['optimizer_restored'] and not lineage['rng_restored']
    assert lineage['sampling_transition']['target_absolute_sample_start'] == 0
    assert lineage['sampling_transition']['changed_data_fields'] == ['data_factory']
    assert cache_data.sha256_file(path) == before
    assert all(torch.equal(value, state97[name]) for name, value in state.items())
    # The new checkpoint must keep the whole nested 49->97->resampling chain.
    model = torch.nn.Module()
    model.register_parameter('weight', torch.nn.Parameter(torch.ones(1)))
    original_trainable = trainer.trainable_state_dict
    try:
        trainer.trainable_state_dict = lambda _: state
        payload = trainer._checkpoint_payload(model, torch.optim.SGD(model.parameters(),lr=0),
            config=cfg97, step=20, micro_batches_consumed=160, metrics={},
            manifest_hashes=target_hashes, initialization=lineage)
    finally:
        trainer.trainable_state_dict = original_trainable
    assert payload['initialization']['source_initialization'] == initialization97
