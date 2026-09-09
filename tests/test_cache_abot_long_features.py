import json
from types import SimpleNamespace

import pytest
import torch

from scripts import cache_abot_long_features as cache
from training.data.action_schema import ACTION_KEYS
from training.longforcing_lite import load_longforcing_config


class Encoder:
    provenance = {'type': 'cpu-test-encoder'}
    disabled = False

    def encode_text(self, prompts):
        assert not self.disabled
        return torch.zeros(1, 3, 8)

    def encode_video(self, pixels):
        assert not self.disabled
        return torch.zeros(1, 61, 48, 30, 52, dtype=torch.bfloat16)


def setup_episode(tmp_path, monkeypatch):
    annotation = tmp_path / 'annotation-fixture'
    annotation.write_bytes(b'CPU fixture; parser is injected')
    action = {'fps': 30, 'total_frames': 480, 'frames': [
        {'frame_id': f'frame_{i:06d}', 'timestamp': i / 30,
         'keys': {key: int(key == 'W' and i % 2 == 0) for key in ACTION_KEYS}}
        for i in range(480)]}
    monkeypatch.setattr(cache, 'read_annotation_bundle', lambda _: SimpleNamespace(action=action, caption='test scene'))
    monkeypatch.setattr(cache, 'probe_video', lambda _: {'fps': 30, 'frames': 480, 'video_sha256': 'cpu-fixture'})
    monkeypatch.setattr(cache, '_selected_starts', lambda *args: [0, 1])
    manifest = tmp_path / 'train.jsonl'
    manifest.write_text(json.dumps({'episode_id': 'a', 'split': 'train',
                                   'annotations_path': str(annotation),
                                   'video_path': str(tmp_path / 'fake-video')}) + '\n', encoding='utf-8')
    return manifest


def test_completed_long_cache_reuse_and_legacy_rebuild(tmp_path, monkeypatch):
    manifest = setup_episode(tmp_path, monkeypatch)
    encoder = Encoder()

    def decoder(path, **kwargs):
        assert not encoder.disabled
        if kwargs['start_seconds'] == 0:
            raise cache.IncompleteVideoWindowError('fixture EOF', expected_frames=451, actual_frames=450)
        return torch.zeros(3, 241, 2, 2, dtype=torch.uint8), 'cpu_fixture'

    root = tmp_path / 'cache'
    first = cache.cache_long_manifest(manifest, root, encoder=encoder, decoder=decoder)
    assert first['windows'] == 1 and first['excluded_window_count'] == 1
    receipt_path = root / 'episodes/a/long241-receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['cache_binding']['config']['seek_policy'].endswith('_v1')
    assert receipt['split'] == 'train'
    original_hash = cache.sha256_file(root / receipt['shards'][0]['path'])
    encoder.disabled = True
    reused = cache.cache_long_manifest(manifest, root, encoder=encoder, decoder=decoder, reuse_completed=True)
    assert reused['reused_episode_count'] == 1
    assert reused['index_sha256'] == first['index_sha256']
    assert cache.sha256_file(root / receipt['shards'][0]['path']) == original_hash
    del receipt['cache_binding']
    receipt_path.write_text(json.dumps(receipt), encoding='utf-8')
    encoder.disabled = False
    legacy = cache.cache_long_manifest(manifest, root, encoder=encoder, decoder=decoder, reuse_completed=True)
    assert legacy['recomputed_legacy_episodes'] == ['a']
    assert legacy['reused_episode_count'] == 0


def test_long_cache_does_not_swallow_non_eof_errors(tmp_path, monkeypatch):
    manifest = setup_episode(tmp_path, monkeypatch)

    def broken(*args, **kwargs):
        raise RuntimeError('decoder unavailable')

    with pytest.raises(RuntimeError, match='decoder unavailable'):
        cache.cache_long_manifest(manifest, tmp_path / 'cache', encoder=Encoder(), decoder=broken)


def test_week_long_config_reaches_15_seconds_and_keeps_original_data():
    root = cache.ROOT / 'configs/train'
    old = load_longforcing_config(root / 'longforcing_lite_v1.yaml')
    week = load_longforcing_config(root / 'longforcing_lite_5090_week.yaml')
    assert week.data == old.data
    assert week.training.max_steps == 80
    assert week.training.checkpoint_every == 20
    assert list(week.rollout.curriculum_start_steps) == [0, 10, 25, 50]
    assert '/abot-week-v1/action-teacher/' in week.lineage.teacher_checkpoint_path
    assert '/abot-week-v1/causal-teacher-forcing/' in week.lineage.causal_checkpoint_path
    assert week.training.output_dir.endswith('/abot-week-v1/longforcing-lite')
