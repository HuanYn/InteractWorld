"""CPU-only contract tests; injected video/VAE are not GPU feasibility evidence."""
import json
from types import SimpleNamespace

import pytest
import torch

from scripts import cache_abot_window97_features as cache
from training.config import ActionTeacherConfig
from training.data.action_dataset import FeatureCacheError, PrecomputedActionDataset, action_teacher_shapes
from training.data.action_schema import ACTION_KEYS, parse_action_document


def setup_episode(tmp_path, monkeypatch, *, identity="train-a", split="train"):
    annotation = tmp_path / "annotation-fixture"
    annotation.write_bytes(b"CPU fixture: injected parser")
    action = {"fps": 30, "total_frames": 240, "frames": [
        {"frame_id": f"frame_{i:06d}", "timestamp": i / 30,
         "keys": {key: int((i + ACTION_KEYS.index(key)) % 3 == 0) for key in ACTION_KEYS}}
        for i in range(240)]}
    monkeypatch.setattr(cache, "read_annotation_bundle", lambda _: SimpleNamespace(
        action=action, caption={"scene_static": "A fixed outdoor scene.", "narrative": "must not encode this"}))
    monkeypatch.setattr(cache, "probe_video", lambda _: {"fps": 30, "frames": 240, "video_sha256": "cpu-fixture"})
    manifest = tmp_path / "train97.jsonl"
    manifest.write_text(json.dumps(dict(episode_id=identity, split=split, annotations_path=str(annotation),
                                       video_path=str(tmp_path / "fake-video"))) + "\n", encoding="utf-8")
    return manifest, parse_action_document(action)


class Encoder:
    provenance = {"type": "cpu-test-encoder"}
    calls = 0
    disabled = False

    def encode_text(self, prompts):
        assert not self.disabled
        assert prompts == ["A fixed outdoor scene."]
        return torch.ones(1, 3, 4096)

    def encode_video(self, pixels):
        assert not self.disabled
        assert pixels.shape == (1, 3, 97, 480, 832)
        self.calls += 1
        value = torch.zeros(1, 25, 48, 30, 52, dtype=torch.bfloat16)
        value[:, 0].fill_(0.5)
        return value


def fake_decoder(*args, **kwargs):
    assert kwargs["num_frames"] == 97
    assert kwargs["target_fps"] == 16
    return torch.zeros(1, dtype=torch.uint8).expand(3, 97, 480, 832), "cpu_fixture"


@pytest.mark.parametrize("frames,shape", [(49, (13, 48, 30, 52)), (97, (25, 48, 30, 52))])
def test_explicit_config_and_shape_contract(frames, shape):
    config = ActionTeacherConfig()
    config.data.num_frames = frames
    config.validate()
    assert action_teacher_shapes(frames) == (shape, (frames - 1, 8))


def test_other_lengths_and_wrong_compression_rejected():
    config = ActionTeacherConfig()
    config.data.num_frames = 145
    with pytest.raises(ValueError, match="exactly 49 or 97"):
        config.validate()
    with pytest.raises(FeatureCacheError, match="exactly 49 or 97"):
        action_teacher_shapes(145)
    config.data.num_frames = 97
    config.model.temporal_compression = 8
    with pytest.raises(ValueError, match="compression"):
        config.validate()


def test_action_alignment_is96_future_rows_and_deterministic(tmp_path, monkeypatch):
    _, sequence = setup_episode(tmp_path, monkeypatch)
    starts = cache.selected_starts(sequence, "train-a", 2, 42)
    assert starts == cache.selected_starts(sequence, "train-a", 2, 42)
    offsets = sequence.resampled_offsets(97, output_fps=16)
    assert offsets[0] == 0 and offsets[-1] == 180
    actions = cache.sampled_actions(sequence, starts[0])
    assert actions.shape == (96, 8)
    assert actions.tolist() == [list(sequence.frames[starts[0] + offset].keys) for offset in offsets[1:]]


@pytest.mark.parametrize("split,identity", [("dev", "some-dev"), ("test", "some-test"), ("train", cache.FROZEN_DEMO_EPISODE)])
def test_cache_rejects_dev_test_and_frozen_episode_before_encoding(tmp_path, monkeypatch, split, identity):
    manifest, _ = setup_episode(tmp_path, monkeypatch, split=split, identity=identity)
    with pytest.raises(ValueError, match="training-only"):
        cache.cache_window97_manifest(manifest, tmp_path / "features", encoder=Encoder(), decoder=fake_decoder)


def test_original_split_binding_cannot_relabel_dev(tmp_path, monkeypatch):
    manifest, _ = setup_episode(tmp_path, monkeypatch)
    source = tmp_path / "original898.jsonl"
    row = json.loads(manifest.read_text())
    source.write_text(json.dumps({**row, "split": "dev"}))
    with pytest.raises(ValueError, match="training-only"):
        cache.validated_training_records(manifest, source)
    source.write_text(json.dumps(row) + "\n" + json.dumps({**row, "episode_id": "dev-other", "split": "dev"}))
    assert cache.validated_training_records(manifest, source) == [row]


def test_cpu_manifest_selection_preserves_original_rows_hash_and_refuses_overwrite(tmp_path, monkeypatch):
    manifest, _ = setup_episode(tmp_path, monkeypatch)
    base = json.loads(manifest.read_text())
    rows = [{**base, "episode_id": f"episode-{i}", "split": "train" if i < 4 else "dev",
             "source_fps": 30, "output_fps": 16, "total_action_frames": 240} for i in range(5)]
    source = tmp_path / "original.jsonl"
    source.write_bytes(cache._jsonl_bytes(rows))
    output = tmp_path / "selected/manifests/train.jsonl"
    first = cache.prepare_training_manifest(source, output, max_episodes=2)
    assert first["source_episode_count"] == 5
    assert first["selected_episode_count"] == 2
    assert first["source_manifest_sha256"] == cache.sha256_file(source)
    selected = [json.loads(line) for line in output.read_text().splitlines()]
    assert all(row in rows[:4] for row in selected)
    assert cache.prepare_training_manifest(source, output, max_episodes=2) == first
    with pytest.raises(FileExistsError, match="different content"):
        cache.prepare_training_manifest(source, output, max_episodes=3)
    with pytest.raises(ValueError, match="traversal"):
        cache.prepare_training_manifest(source, tmp_path / "../bad.jsonl")


def test_continuous_cache_loader_and_hash_verified_resume(tmp_path, monkeypatch):
    manifest, _ = setup_episode(tmp_path, monkeypatch)
    root = tmp_path / "features"
    encoder = Encoder()
    first = cache.cache_window97_manifest(manifest, root, encoder=encoder, decoder=fake_decoder, max_total_windows=1)
    assert first["windows"] == 1 and encoder.calls == 1
    assert first["tensor_shard_bytes"] > 3_744_000
    index = cache.cache_index_path(manifest, root)
    dataset = PrecomputedActionDataset(index, manifest_path=manifest, num_frames=97)
    sample = dataset[0]
    assert sample["noisy_latents"].shape == (25, 48, 30, 52)
    assert sample["actions"].shape == (96, 8)
    assert sample["prompt_embeds"].shape == (3, 4096)
    assert bool((sample["noisy_latents"][0] == 0.5).all())
    assert bool((sample["target_flow"][0] == 0).all())
    assert sample["timesteps"][0] == 0
    assert torch.equal(sample["noisy_latents"], dataset[0]["noisy_latents"])
    with pytest.raises(FeatureCacheError, match="old49"):
        PrecomputedActionDataset(index, manifest_path=manifest)
    encoder.disabled = True
    resumed = cache.cache_window97_manifest(manifest, root, encoder=encoder, decoder=fake_decoder,
        max_total_windows=1, reuse_completed=True)
    assert resumed["index_sha256"] == first["index_sha256"]
    assert resumed["reused_episodes"] == ["train-a"]
    with pytest.raises(ValueError, match="different binding"):
        cache.cache_window97_manifest(manifest, root, encoder=encoder, decoder=fake_decoder,
            max_total_windows=2, reuse_completed=True)
    shard = root / json.loads(index.read_text())["shards"][0]["path"]
    with shard.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        cache.cache_window97_manifest(manifest, root, encoder=encoder, decoder=fake_decoder,
            max_total_windows=1, reuse_completed=True)


def test_incomplete_rgb_is_excluded_not_padded_and_other_errors_propagate(tmp_path, monkeypatch):
    manifest, _ = setup_episode(tmp_path, monkeypatch)
    def eof(*args, **kwargs):
        raise cache.IncompleteVideoWindowError("fixture EOF", expected_frames=181, actual_frames=180)
    with pytest.raises(ValueError, match="no complete97"):
        cache.cache_window97_manifest(manifest, tmp_path / "features", encoder=Encoder(), decoder=eof, max_total_windows=1)
    receipt = json.loads((tmp_path / "features/episodes/train-a/window97-receipt.json").read_text())
    assert receipt["excluded_windows"][0]["reason"] == "incomplete_video_window"
    assert not cache.cache_index_path(manifest, tmp_path / "features").exists()
    def broken(*args, **kwargs):
        raise RuntimeError("codec unavailable")
    with pytest.raises(RuntimeError, match="codec unavailable"):
        cache.cache_window97_manifest(manifest, tmp_path / "other", encoder=Encoder(), decoder=broken)


def test_wrong_rgb_length_is_rejected_and_cli_plan_never_queries_cuda(tmp_path, monkeypatch, capsys):
    manifest, _ = setup_episode(tmp_path, monkeypatch)
    def short(*args, **kwargs):
        return torch.zeros(3, 49, 2, 2, dtype=torch.uint8), "invalid"
    with pytest.raises(ValueError, match="complete continuous RGB"):
        cache.cache_window97_manifest(manifest, tmp_path / "features", encoder=Encoder(), decoder=short)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(AssertionError("CUDA queried")))
    assert cache.main(["--manifest", str(manifest)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "cpu_plan" and not plan["cuda_queried"] and not plan["weights_loaded"]
