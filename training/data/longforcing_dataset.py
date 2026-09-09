"""Deterministic 241-frame cache reader for LongForcing-lite."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Dataset

from training.data.action_dataset import (
    CACHE_SCHEMA_VERSION,
    FeatureCacheError,
    sha256_file,
    validate_feature_cache_binding,
)

LONG_CACHE_KIND = "abot_long241_features"
LONG_LATENT_SHAPE = (61, 48, 30, 52)
LONG_ACTION_SHAPE = (240, 8)


def _seed64(*parts: object) -> int:
    raw = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FeatureCacheError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise FeatureCacheError(f"{path}:{line_number}: record must be an object")
            records.append(record)
    return records


def validate_long_cache_binding(
    index_path: str | Path,
    manifest_path: str | Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    index = Path(index_path).resolve()
    manifest = Path(manifest_path).resolve()
    receipt_path = index.with_suffix(index.suffix + ".receipt.json")
    if not index.is_file() or not receipt_path.is_file() or not manifest.is_file():
        raise FileNotFoundError("long241 index, receipt, or manifest is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise FeatureCacheError("unsupported long241 cache schema")
    if receipt.get("kind") != LONG_CACHE_KIND:
        raise FeatureCacheError("cache receipt is not a LongForcing long241 cache")
    if Path(str(receipt.get("index", ""))).resolve() != index:
        raise FeatureCacheError("long241 receipt index path mismatch")
    if Path(str(receipt.get("manifest", ""))).resolve() != manifest:
        raise FeatureCacheError("long241 receipt manifest path mismatch")
    if verify_hashes and receipt.get("index_sha256") != sha256_file(index):
        raise FeatureCacheError("long241 index hash mismatch")
    if verify_hashes and receipt.get("manifest_sha256") != sha256_file(manifest):
        raise FeatureCacheError("long241 manifest hash mismatch")
    contract = receipt.get("config", {})
    if (contract.get("num_frames"), contract.get("target_fps")) != (241, 16):
        raise FeatureCacheError("long241 cache has the wrong frame/fps contract")
    return receipt


class PrecomputedLongForcingDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        index_path: str | Path,
        *,
        manifest_path: str | Path,
        short_index_path: str | Path,
        split: str = "train",
        seed: int = 42,
        verify_hashes: bool = True,
    ) -> None:
        self.index_path = Path(index_path).resolve()
        self.seed = int(seed)
        validate_long_cache_binding(
            self.index_path,
            manifest_path,
            verify_hashes=verify_hashes,
        )
        # The replay source is lineage-bound to stages 1/2 even though its
        # tensors are sliced from the corresponding long window.
        validate_feature_cache_binding(
            short_index_path,
            manifest_path,
            verify_hashes=verify_hashes,
        )
        episodes = [
            record for record in _read_jsonl(self.index_path) if record.get("split") == split
        ]
        episodes.sort(key=lambda record: str(record.get("episode_id", "")))
        if not episodes:
            raise FeatureCacheError(f"long241 cache has no episodes for split {split!r}")
        for episode in episodes:
            if not isinstance(episode.get("episode_id"), str):
                raise FeatureCacheError("long241 episode_id must be a string")
            shards = episode.get("shards")
            if not isinstance(shards, list) or not shards:
                raise FeatureCacheError(f"episode {episode['episode_id']} has no long241 shards")
            total = sum(int(shard.get("samples", -1)) for shard in shards)
            if total != episode.get("num_windows") or total <= 0:
                raise FeatureCacheError(f"episode {episode['episode_id']} window count mismatch")
            receipt = episode.get("episode_receipt")
            if not isinstance(receipt, dict):
                raise FeatureCacheError(f"episode {episode['episode_id']} has no receipt")
            path = (self.index_path.parent / str(receipt.get("path", ""))).resolve()
            try:
                path.relative_to(self.index_path.parent)
            except ValueError as exc:
                raise FeatureCacheError("episode receipt escapes long241 cache root") from exc
            if verify_hashes and sha256_file(path) != receipt.get("sha256"):
                raise FeatureCacheError(f"episode receipt hash mismatch: {path}")
        self.episodes = episodes
        self.samples_per_episode = max(int(record["num_windows"]) for record in episodes)
        self.verify_hashes = bool(verify_hashes)

    def __len__(self) -> int:
        return len(self.episodes) * self.samples_per_episode

    @lru_cache(maxsize=2)
    def _load_shard(self, relative_path: str, expected_hash: str) -> Mapping[str, Any]:
        path = (self.index_path.parent / relative_path).resolve()
        try:
            path.relative_to(self.index_path.parent)
        except ValueError as exc:
            raise FeatureCacheError("long241 shard escapes cache root") from exc
        if self.verify_hashes and sha256_file(path) != expected_hash:
            raise FeatureCacheError(f"long241 shard hash mismatch: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise FeatureCacheError(f"invalid long241 shard: {path}")
        if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise FeatureCacheError(f"unsupported long241 shard schema: {path}")
        if payload.get("kind") != LONG_CACHE_KIND:
            raise FeatureCacheError(f"wrong cache kind in long241 shard: {path}")
        return payload

    def _window(self, episode: Mapping[str, Any], index: int) -> dict[str, Any]:
        offset = index
        for shard in episode["shards"]:
            count = int(shard["samples"])
            if offset < count:
                payload = self._load_shard(str(shard["path"]), str(shard["sha256"]))
                clean = payload["clean_latents"][offset].float()
                actions = payload["actions"][offset].float()
                if tuple(clean.shape) != LONG_LATENT_SHAPE:
                    raise FeatureCacheError(
                        f"long latent must be {LONG_LATENT_SHAPE}, got {tuple(clean.shape)}"
                    )
                if tuple(actions.shape) != LONG_ACTION_SHAPE:
                    raise FeatureCacheError(
                        f"long actions must be {LONG_ACTION_SHAPE}, got {tuple(actions.shape)}"
                    )
                result = {
                    "clean_latents": clean,
                    "actions": actions,
                    "prompt_embeds": payload["prompt_embeds"].float(),
                }
                for key in ("y", "clip_fea"):
                    if key in payload:
                        value = payload[key]
                        result[key] = (
                            value[offset]
                            if torch.is_tensor(value) and value.shape[0] == count
                            else value
                        )
                return result
            offset -= count
        raise IndexError(index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        cycle, slot = divmod(index, len(self.episodes))
        episode = self.episodes[slot]
        window_index = _seed64(self.seed, cycle, episode["episode_id"]) % int(
            episode["num_windows"]
        )
        cached = self._window(episode, window_index)
        clean = cached.pop("clean_latents")
        actions = cached.pop("actions")
        generator = torch.Generator(device="cpu").manual_seed(
            _seed64(self.seed, "longforcing", index)
        )
        rollout_noise = torch.randn(
            (20, 3, *clean.shape[1:]),
            generator=generator,
            dtype=torch.float32,
        )

        short_clean = clean[:13]
        short_noise = torch.randn(short_clean.shape, generator=generator, dtype=torch.float32)
        schedule_index = int(torch.randint(0, 1000, (1,), generator=generator).item())
        sigma_linear = 1.0 - schedule_index / 1000.0
        sigma = 5.0 * sigma_linear / (1.0 + 4.0 * sigma_linear)
        noisy = (1.0 - sigma) * short_clean + sigma * short_noise
        target = short_noise - short_clean
        noisy[0] = short_clean[0]
        target[0].zero_()
        timesteps = torch.full((13,), sigma * 1000.0, dtype=torch.float32)
        timesteps[0] = 0.0
        conditions = {"prompt_embeds": cached["prompt_embeds"]}
        short_window = {
            "noisy_latents": noisy,
            "target_flow": target,
            "timesteps": timesteps,
            "prompt_embeds": cached["prompt_embeds"],
            "actions": actions[:48],
        }
        for key in ("y", "clip_fea"):
            if key in cached:
                conditions[key] = cached[key]
                short_window[key] = cached[key]
        return {
            "initial_latent": clean[:1],
            "rollout_noise": rollout_noise,
            "block_actions": actions.reshape(20, 12, 8),
            "conditions": conditions,
            "short_window": short_window,
        }


def build_longforcing_dataloader(*, config: Any, training: Any) -> DataLoader:
    if not getattr(config, "precomputed_latents", False):
        raise ValueError("LongForcing-lite requires precomputed_latents=true")
    if not getattr(config, "precomputed_text_embeddings", False):
        raise ValueError("LongForcing-lite requires precomputed_text_embeddings=true")
    dataset = PrecomputedLongForcingDataset(
        config.long_feature_index_path,
        manifest_path=config.manifest_path,
        short_index_path=config.feature_index_path,
        split="train",
        seed=training.seed,
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
