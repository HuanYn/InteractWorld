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
from typing import Any, Iterable, Mapping

import torch
from torch.utils.data import DataLoader, Dataset

CACHE_SCHEMA_VERSION = 1
DEFAULT_TARGET_FPS = 16
EXPECTED_LATENT_SHAPE = (13, 48, 30, 52)
EXPECTED_ACTION_SHAPE = (48, 8)
SCENE_STATIC_PROMPT_CACHE_KIND = "scene_static_prompt_cache"
SCENE_STATIC_PROMPT_POLICY = "scene_static_only_v1"


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


def validate_scene_static_prompt_cache_binding(
    path: str | Path, *, index_path: str | Path, manifest_path: str | Path,
) -> dict[str, Any]:
    """Validate the static-text sidecar without loading its tensor payload.

    Hashes establish provenance/integrity, not linguistic correctness: the
    generator records the static-text policy and the exact text it encoded.
    The payload is hashed once here; callers can reuse ``cache_sha256`` for
    checkpoint lineage rather than reading the potentially large file again.
    """
    cache = Path(path).resolve()
    index = Path(index_path).resolve()
    manifest = Path(manifest_path).resolve()
    receipt_path = cache.with_suffix(cache.suffix + ".receipt.json")
    if not cache.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("scene-static prompt cache or receipt is missing")
    source = validate_feature_cache_binding(index, manifest, verify_hashes=True)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise FeatureCacheError("unsupported scene-static prompt receipt schema")
    if receipt.get("kind") != SCENE_STATIC_PROMPT_CACHE_KIND or receipt.get("prompt_policy") != SCENE_STATIC_PROMPT_POLICY:
        raise FeatureCacheError("scene-static prompt kind/policy mismatch")
    bound_path = receipt.get("cache_path")
    if not isinstance(bound_path, str) or not Path(bound_path).is_absolute() or Path(bound_path).resolve() != cache:
        raise FeatureCacheError("scene-static prompt cache path mismatch")
    expected = {
        "manifest_sha256": source["manifest_sha256"],
        "feature_index_sha256": source["index_sha256"],
        "feature_receipt_sha256": sha256_file(index.with_suffix(index.suffix + ".receipt.json")),
    }
    for key, digest in expected.items():
        if receipt.get(key) != digest:
            raise FeatureCacheError(f"scene-static prompt {key} mismatch")
    if not isinstance(receipt.get("encoder"), Mapping) or not receipt["encoder"]:
        raise FeatureCacheError("scene-static prompt receipt has no encoder provenance")
    source_episodes: dict[str, str] = {}
    for row in _read_jsonl(index):
        identity, split = row.get("episode_id"), row.get("split")
        if not isinstance(identity, str) or not identity or not isinstance(split, str) or not split or identity in source_episodes:
            raise FeatureCacheError("feature index contains invalid or duplicate episode IDs")
        source_episodes[identity] = split
    episodes = receipt.get("episodes")
    if not isinstance(episodes, dict) or not episodes:
        raise FeatureCacheError("scene-static prompt receipt has no episode bindings")
    for identity, binding in episodes.items():
        if identity not in source_episodes or not isinstance(binding, Mapping):
            raise FeatureCacheError(f"scene-static prompt has unknown/unbound episode: {identity}")
        if binding.get("split") != source_episodes[identity]:
            raise FeatureCacheError(f"scene-static prompt split mismatch: {identity}")
        prompt = binding.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise FeatureCacheError(f"scene-static prompt text is missing: {identity}")
        if binding.get("prompt_sha256") != hashlib.sha256(prompt.encode("utf-8")).hexdigest():
            raise FeatureCacheError(f"scene-static prompt text hash mismatch: {identity}")
    if receipt.get("cache_sha256") != sha256_file(cache):
        raise FeatureCacheError("scene-static prompt payload hash mismatch")
    return receipt


def load_scene_static_prompt_cache(
    path: str | Path, *, index_path: str | Path, manifest_path: str | Path,
    required_episodes: Iterable[str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load trusted, hash-bound CPU text features without touching video shards.

    ``None`` requires every episode in the source index. A split-specific
    consumer may pass its required IDs; extra known splits are reusable, but
    no required episode may silently fall back to the old narrative prompt.
    """
    receipt = validate_scene_static_prompt_cache_binding(
        path, index_path=index_path, manifest_path=manifest_path,
    )
    source_ids = {row["episode_id"] for row in _read_jsonl(Path(index_path))}
    if required_episodes is None:
        required = source_ids
    else:
        if isinstance(required_episodes, (str, bytes)):
            raise FeatureCacheError("required_episodes must contain episode-ID strings")
        values = list(required_episodes)
        if any(not isinstance(value, str) or not value for value in values):
            raise FeatureCacheError("required_episodes must contain episode-ID strings")
        required = set(values)
    if not required.issubset(source_ids):
        raise FeatureCacheError("required prompt episodes are absent from the source index")
    missing = required.difference(receipt["episodes"])
    if missing:
        raise FeatureCacheError(f"scene-static prompt cache missing required episodes: {sorted(missing)}")
    payload = torch.load(Path(path).resolve(), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != CACHE_SCHEMA_VERSION or payload.get("kind") != SCENE_STATIC_PROMPT_CACHE_KIND:
        raise FeatureCacheError("unsupported scene-static prompt payload schema/kind")
    embeddings = payload.get("prompt_embeds")
    if not isinstance(embeddings, Mapping) or set(embeddings) != set(receipt["episodes"]):
        raise FeatureCacheError("scene-static prompt payload/receipt episode IDs differ")
    for identity, tensor in embeddings.items():
        if (not torch.is_tensor(tensor) or tensor.layout != torch.strided or not tensor.is_floating_point()
                or tensor.ndim != 2 or not 1 <= tensor.shape[0] <= 512 or tensor.shape[1] != 4096):
            raise FeatureCacheError(f"scene-static prompt embedding must be floating [L,4096], 1<=L<=512: {identity}")
        if tensor.device.type != "cpu" or not bool(torch.isfinite(tensor).all()):
            raise FeatureCacheError(f"scene-static prompt embedding must be finite CPU data: {identity}")
    return {identity: tensor.detach() for identity, tensor in embeddings.items()}, receipt


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
        prompt_cache_path: str | Path | None = None,
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
        self.prompt_cache_path = Path(prompt_cache_path).resolve() if prompt_cache_path is not None else None
        self.prompt_cache_receipt: dict[str, Any] | None = None
        self._prompt_overrides: dict[str, torch.Tensor] | None = None
        if self.prompt_cache_path is not None:
            self._prompt_overrides, self.prompt_cache_receipt = load_scene_static_prompt_cache(
                self.prompt_cache_path, index_path=self.index_path, manifest_path=manifest_path,
                required_episodes=[episode["episode_id"] for episode in episodes],
            )

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
                prompt = (
                    payload["prompt_embeds"].float() if self._prompt_overrides is None
                    else self._prompt_overrides[episode["episode_id"]].to(dtype=torch.float32, copy=True)
                )
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
        prompt_cache_path=getattr(config, "prompt_cache_path", None),
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
