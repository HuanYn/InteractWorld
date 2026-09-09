"""Read-only, deterministic training data backed by precomputed ABot features.

The cache is trusted project output rather than arbitrary downloaded tensors.  Every
index and shard is nevertheless checked against the hashes written by the caching
job before it is consumed.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Dataset

CACHE_SCHEMA_VERSION = 1
DEFAULT_TARGET_FPS = 16
EXPECTED_LATENT_SHAPE = (13, 48, 30, 52)
EXPECTED_ACTION_SHAPE = (48, 8)


class FeatureCacheError(ValueError):
    """Raised when a feature cache is incomplete, stale, or malformed."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_cache_root(manifest_path: str | Path) -> Path:
    """Use ``<data-root>/features`` for ``<data-root>/manifests/*.jsonl``."""
    manifest = Path(manifest_path)
    return manifest.parent.parent / "features"


def cache_index_path(manifest_path: str | Path, cache_root: str | Path | None = None) -> Path:
    manifest = Path(manifest_path)
    root = default_cache_root(manifest) if cache_root is None else Path(cache_root)
    return root / f"{manifest.stem}.features.jsonl"


def validate_feature_cache_binding(
    index_path: str | Path,
    manifest_path: str | Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Fail closed when an index/receipt belongs to a different manifest."""
    index = Path(index_path).resolve()
    manifest = Path(manifest_path).resolve()
    receipt_path = index.with_suffix(index.suffix + ".receipt.json")
    if not index.is_file() or not receipt_path.is_file() or not manifest.is_file():
        raise FileNotFoundError("precomputed cache index, receipt, or bound manifest is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise FeatureCacheError("unsupported feature-cache receipt schema")
    if Path(str(receipt.get("index", ""))).resolve() != index:
        raise FeatureCacheError("feature-cache receipt index path mismatch")
    if Path(str(receipt.get("manifest", ""))).resolve() != manifest:
        raise FeatureCacheError("feature-cache receipt manifest path mismatch")
    if verify_hashes and receipt.get("index_sha256") != sha256_file(index):
        raise FeatureCacheError("feature-cache index hash mismatch")
    if verify_hashes and receipt.get("manifest_sha256") != sha256_file(manifest):
        raise FeatureCacheError("feature-cache manifest hash mismatch")
    return receipt


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FeatureCacheError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise FeatureCacheError(f"{path}:{line_number}: record must be an object")
            records.append(value)
    return records


def _seed64(*parts: object) -> int:
    raw = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


class PrecomputedActionDataset(Dataset[dict[str, torch.Tensor]]):
    """Episode-balanced deterministic views over immutable feature shards."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        manifest_path: str | Path | None = None,
        split: str = "train",
        seed: int = 42,
        timestep_shift: float = 5.0,
        verify_hashes: bool = True,
    ) -> None:
        self.index_path = Path(index_path).resolve()
        self.seed = int(seed)
        self.timestep_shift = float(timestep_shift)
        if self.timestep_shift <= 0:
            raise ValueError("timestep_shift must be positive")
        receipt_path = self.index_path.with_suffix(self.index_path.suffix + ".receipt.json")
        if manifest_path is None:
            if not receipt_path.is_file():
                raise FileNotFoundError(f"precomputed cache receipt is missing: {receipt_path}")
            receipt_hint = json.loads(receipt_path.read_text(encoding="utf-8"))
            manifest_path = receipt_hint.get("manifest", "")
        validate_feature_cache_binding(
            self.index_path, manifest_path, verify_hashes=verify_hashes
        )

        episodes = [record for record in _read_jsonl(self.index_path) if record.get("split") == split]
        episodes.sort(key=lambda item: str(item.get("episode_id", "")))
        if not episodes:
            raise FeatureCacheError(f"feature cache has no episodes for split {split!r}")
        for episode in episodes:
            if not isinstance(episode.get("episode_id"), str):
                raise FeatureCacheError("cache episode_id must be a string")
            shards = episode.get("shards")
            if not isinstance(shards, list) or not shards:
                raise FeatureCacheError(f"episode {episode['episode_id']} has no shards")
            total = 0
            for shard in shards:
                if not isinstance(shard, dict) or not isinstance(shard.get("samples"), int):
                    raise FeatureCacheError("invalid shard index entry")
                total += shard["samples"]
            if total != episode.get("num_windows") or total <= 0:
                raise FeatureCacheError(f"episode {episode['episode_id']} window count mismatch")
            episode_receipt = episode.get("episode_receipt")
            if not isinstance(episode_receipt, dict):
                raise FeatureCacheError(f"episode {episode['episode_id']} has no receipt binding")
            receipt_file = (self.index_path.parent / str(episode_receipt.get("path", ""))).resolve()
            try:
                receipt_file.relative_to(self.index_path.parent)
            except ValueError as exc:
                raise FeatureCacheError("episode receipt escapes cache root") from exc
            if verify_hashes and sha256_file(receipt_file) != episode_receipt.get("sha256"):
                raise FeatureCacheError(f"episode receipt hash mismatch: {receipt_file}")
        self.episodes = episodes
        self.samples_per_episode = max(int(item["num_windows"]) for item in episodes)
        self.verify_hashes = bool(verify_hashes)

    def __len__(self) -> int:
        return len(self.episodes) * self.samples_per_episode

    @lru_cache(maxsize=2)
    def _load_shard(self, relative_path: str, expected_hash: str) -> Mapping[str, Any]:
        path = (self.index_path.parent / relative_path).resolve()
        try:
            path.relative_to(self.index_path.parent)
        except ValueError as exc:
            raise FeatureCacheError(f"shard escapes cache root: {relative_path!r}") from exc
        if self.verify_hashes and sha256_file(path) != expected_hash:
            raise FeatureCacheError(f"feature shard hash mismatch: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch < 2.0 compatibility
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping) or payload.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise FeatureCacheError(f"invalid feature shard: {path}")
        return payload

    def _cached_window(self, episode: Mapping[str, Any], window_index: int) -> dict[str, Any]:
        offset = window_index
        for shard in episode["shards"]:
            count = int(shard["samples"])
            if offset < count:
                payload = self._load_shard(str(shard["path"]), str(shard["sha256"]))
                latent = payload["clean_latents"][offset].float()
                actions = payload["actions"][offset].float()
                prompt = payload["prompt_embeds"].float()
                if tuple(latent.shape) != EXPECTED_LATENT_SHAPE:
                    raise FeatureCacheError(
                        f"latent must be {EXPECTED_LATENT_SHAPE}, got {tuple(latent.shape)}"
                    )
                if tuple(actions.shape) != EXPECTED_ACTION_SHAPE:
                    raise FeatureCacheError(
                        f"actions must be {EXPECTED_ACTION_SHAPE}, got {tuple(actions.shape)}"
                    )
                result: dict[str, Any] = {
                    "clean_latents": latent,
                    "actions": actions,
                    "prompt_embeds": prompt,
                }
                for key in ("y", "clip_fea"):
                    if key in payload:
                        value = payload[key]
                        result[key] = value[offset] if torch.is_tensor(value) and value.shape[0] == count else value
                return result
            offset -= count
        raise IndexError(window_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        cycle, episode_slot = divmod(index, len(self.episodes))
        episode = self.episodes[episode_slot]
        window_index = _seed64(self.seed, cycle, episode["episode_id"]) % int(
            episode["num_windows"]
        )
        cached = self._cached_window(episode, window_index)
        clean = cached.pop("clean_latents")
        generator = torch.Generator(device="cpu").manual_seed(
            _seed64(self.seed, "diffusion", index)
        )
        noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
        schedule_index = int(torch.randint(0, 1000, (1,), generator=generator).item())
        sigma_linear = 1.0 - schedule_index / 1000.0
        sigma = self.timestep_shift * sigma_linear / (
            1.0 + (self.timestep_shift - 1.0) * sigma_linear
        )
        noisy = (1.0 - sigma) * clean + sigma * noise
        target = noise - clean

        # Wan2.2 TI2V conditions on a clean first latent frame.
        noisy[0] = clean[0]
        target[0].zero_()
        timesteps = torch.full((clean.shape[0],), sigma * 1000.0, dtype=torch.float32)
        timesteps[0] = 0.0
        result = {
            "noisy_latents": noisy,
            "target_flow": target,
            "timesteps": timesteps,
            **cached,
        }
        return result


def build_action_teacher_dataloader(*, config: Any, training: Any) -> DataLoader:
    """Factory consumed by :mod:`train_action_teacher`.

    This function deliberately has no raw-video path: a missing cache is an error,
    preventing accidental VAE/T5 loading or RGB materialization in training workers.
    """
    if not getattr(config, "precomputed_latents", False):
        raise ValueError("action teacher requires precomputed_latents=true")
    if not getattr(config, "precomputed_text_embeddings", False):
        raise ValueError("action teacher requires precomputed_text_embeddings=true")
    index = cache_index_path(config.manifest_path)
    dataset = PrecomputedActionDataset(
        index,
        manifest_path=config.manifest_path,
        split="train",
        seed=training.seed,
        timestep_shift=5.0,
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
