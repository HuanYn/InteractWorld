"""Tiny CPU fixtures: shared static text for rollout and replay, no GPU/cache writes outside tmp."""
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from training.data import longforcing_dataset as module
from training.data.action_dataset import FeatureCacheError, sha256_file


def _fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "LONG_LATENT_SHAPE", (61, 1, 1, 1))
    manifest, short, long = (tmp_path / name for name in ("train.jsonl", "train.features.jsonl", "train.long241.features.jsonl"))
    rows = [{"episode_id": "one", "split": "train"}, {"episode_id": "two", "split": "dev"}]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    short.write_text(manifest.read_text(), encoding="utf-8")
    episode_receipt = tmp_path / "one.receipt.json"
    episode_receipt.write_text("{}", encoding="utf-8")
    shard = tmp_path / "long.pt"
    torch.save({"schema_version": 1, "kind": module.LONG_CACHE_KIND,
                "clean_latents": torch.zeros(1, 61, 1, 1, 1), "actions": torch.zeros(1, 240, 8),
                "prompt_embeds": torch.zeros(2, 4096)}, shard)
    long.write_text(json.dumps({**rows[0], "num_windows": 1,
        "shards": [{"path": shard.name, "sha256": sha256_file(shard), "samples": 1}],
        "episode_receipt": {"path": episode_receipt.name, "sha256": sha256_file(episode_receipt)}}) + "\n", encoding="utf-8")
    for index in (short, long):
        receipt = {"schema_version": 1, "index": str(index), "manifest": str(manifest),
                   "index_sha256": sha256_file(index), "manifest_sha256": sha256_file(manifest)}
        if index == long:
            receipt.update(kind=module.LONG_CACHE_KIND, config={"num_frames": 241, "target_fps": 16})
        index.with_suffix(index.suffix + ".receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    cache = tmp_path / "static.pt"
    payload = {"schema_version": 1, "kind": "scene_static_prompt_cache", "prompt_embeds": {
        row["episode_id"]: torch.ones(2, 4096, dtype=torch.bfloat16) for row in rows}}
    text = "A quiet outdoor scene."
    receipt = {"schema_version": 1, "kind": "scene_static_prompt_cache", "prompt_policy": "scene_static_only_v1",
        "cache_path": str(cache), "manifest_sha256": sha256_file(manifest), "feature_index_sha256": sha256_file(short),
        "feature_receipt_sha256": sha256_file(short.with_suffix(short.suffix + ".receipt.json")),
        "encoder": {"kind": "unit-test"}, "episodes": {row["episode_id"]: {"split": row["split"], "prompt": text,
        "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()} for row in rows}}
    _write_cache(cache, payload, receipt)
    return cache, payload, receipt, dict(index_path=long, manifest_path=manifest, short_index_path=short)


def _write_cache(path, payload, receipt):
    torch.save(payload, path)
    receipt["cache_sha256"] = sha256_file(path)
    path.with_suffix(".pt.receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


def test_static_override_is_shared_by_rollout_and_replay_and_preserves_noise(tmp_path, monkeypatch):
    path, _, receipt, kwargs = _fixture(tmp_path, monkeypatch)
    before = sha256_file(path)
    legacy = module.PrecomputedLongForcingDataset(**kwargs)[0]
    dataset = module.PrecomputedLongForcingDataset(**kwargs, prompt_cache_path=path)
    current = dataset[0]
    assert dataset.prompt_cache_receipt == receipt
    assert current["conditions"]["prompt_embeds"] is current["short_window"]["prompt_embeds"]
    assert torch.all(current["conditions"]["prompt_embeds"] == 1)
    assert torch.all(legacy["conditions"]["prompt_embeds"] == 0)
    for key in ("initial_latent", "rollout_noise", "block_actions"):
        assert torch.equal(current[key], legacy[key])
    for key in ("noisy_latents", "target_flow", "timesteps", "actions"):
        assert torch.equal(current["short_window"][key], legacy["short_window"][key])
    assert sha256_file(path) == before
    # The factory must forward the opt-in path; a None config preserves legacy.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    loader = module.build_longforcing_dataloader(config=SimpleNamespace(
        precomputed_latents=True, precomputed_text_embeddings=True, num_workers=0,
        manifest_path=kwargs["manifest_path"], feature_index_path=kwargs["short_index_path"],
        long_feature_index_path=kwargs["index_path"], prompt_cache_path=str(path)),
        training=SimpleNamespace(seed=42, micro_batch_size=1))
    assert torch.all(next(iter(loader))["conditions"]["prompt_embeds"] == 1)


@pytest.mark.parametrize("change", ["missing", "missing_episode", "payload_hash", "policy", "long_binding", "split"])
def test_static_override_fails_closed_even_when_general_hash_checks_are_disabled(tmp_path, monkeypatch, change):
    path, payload, receipt, kwargs = _fixture(tmp_path, monkeypatch)
    if change == "missing":
        path = tmp_path / "missing.pt"
    elif change == "missing_episode":
        payload["prompt_embeds"].pop("one")
        receipt["episodes"].pop("one")
    elif change == "policy":
        receipt["prompt_policy"] = "original_feature_cache_prompt"
    elif change == "long_binding":
        receipt["feature_index_sha256"] = sha256_file(kwargs["index_path"])
    elif change == "split":
        receipt["episodes"]["one"]["split"] = "dev"
    if change != "missing":
        _write_cache(path, payload, receipt)
        if change == "payload_hash":
            receipt["cache_sha256"] = "changed"
            path.with_suffix(".pt.receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises((FeatureCacheError, FileNotFoundError)):
        module.PrecomputedLongForcingDataset(**kwargs, prompt_cache_path=path, verify_hashes=False)
