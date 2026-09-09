"""Deterministic, storage-bounded manifest construction for ABot episodes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .action_schema import (
    ACTION_KEYS,
    ActionSchemaError,
    OFFICIAL_SOURCE_FPS,
    SUPPORTED_WINDOWS,
    TRAINING_OUTPUT_FPS,
    parse_action_document,
)
from .safe_annotations import AnnotationArchiveError, read_annotation_bundle

DEFAULT_MAX_STORAGE_BYTES = 80_000_000_000
SPLIT_NAMES = ("train", "dev", "test")
_SAFE_SAMPLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FIRST_PERSON_TERMS = (
    "first person",
    "first-person",
    "firstperson",
    "pov",
    "ego view",
    "egocentric",
    "cockpit view",
)
_MINECRAFT_TERMS = ("minecraft", "mine craft", "voxel sandbox")


class MetadataError(ValueError):
    pass


class StorageBudgetError(ValueError):
    pass


@dataclass(frozen=True)
class ManifestConfig:
    max_storage_bytes: int = DEFAULT_MAX_STORAGE_BYTES
    required_frames: int = 241
    split_seed: str = "abot-v1"
    train_fraction: float = 0.90
    dev_fraction: float = 0.05
    require_first_person: bool = True
    exclude_minecraft: bool = True
    unknown_video_reservation_bytes: int | None = None
    source_fps: int = OFFICIAL_SOURCE_FPS
    output_fps: int = TRAINING_OUTPUT_FPS

    def validate(self) -> None:
        if self.max_storage_bytes <= 0:
            raise ValueError("max_storage_bytes must be positive")
        if self.required_frames not in SUPPORTED_WINDOWS:
            raise ValueError(f"required_frames must be one of {SUPPORTED_WINDOWS}")
        if not (0 < self.train_fraction < 1):
            raise ValueError("train_fraction must be between 0 and 1")
        if not (0 <= self.dev_fraction < 1):
            raise ValueError("dev_fraction must be between 0 and 1")
        if self.train_fraction + self.dev_fraction >= 1:
            raise ValueError("train_fraction + dev_fraction must be below 1")
        if self.unknown_video_reservation_bytes is not None and self.unknown_video_reservation_bytes <= 0:
            raise ValueError("unknown_video_reservation_bytes must be positive")
        if self.source_fps != OFFICIAL_SOURCE_FPS:
            raise ValueError(f"source_fps must be the official {OFFICIAL_SOURCE_FPS}")
        if self.output_fps != TRAINING_OUTPUT_FPS:
            raise ValueError(f"output_fps must be the training rate {TRAINING_OUTPUT_FPS}")


@dataclass
class StorageBudget:
    limit_bytes: int
    used_bytes: int = 0

    def reserve(self, size_bytes: int) -> None:
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise StorageBudgetError(f"invalid byte count: {size_bytes!r}")
        if self.used_bytes + size_bytes > self.limit_bytes:
            raise StorageBudgetError(
                f"storage budget exceeded: {self.used_bytes} + {size_bytes} > {self.limit_bytes}"
            )
        self.used_bytes += size_bytes


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metadata_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    metadata_path = Path(path)
    seen: set[str] = set()
    with metadata_path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise MetadataError(f"line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise MetadataError(f"line {line_number}: record must be an object")
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not _SAFE_SAMPLE_ID.fullmatch(sample_id):
                raise MetadataError(f"line {line_number}: invalid sample_id {sample_id!r}")
            if sample_id in seen:
                raise MetadataError(f"line {line_number}: duplicate sample_id {sample_id!r}")
            video = record.get("video")
            if not isinstance(video, dict) or not isinstance(video.get("path"), str):
                raise MetadataError(f"line {line_number}: video.path must be a string")
            if not isinstance(record.get("annotations"), str):
                raise MetadataError(f"line {line_number}: annotations must be a string")
            seen.add(sample_id)
            yield record


def deterministic_split(
    sample_id: str,
    *,
    seed: str = "abot-v1",
    train_fraction: float = 0.90,
    dev_fraction: float = 0.05,
) -> str:
    digest = hashlib.sha256(f"{seed}\0{sample_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < train_fraction:
        return "train"
    if value < train_fraction + dev_fraction:
        return "dev"
    return "test"


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(_flatten_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten_text(item) for item in value)
    return ""


def first_person_non_minecraft(
    caption: Any,
    *,
    require_first_person: bool = True,
    exclude_minecraft: bool = True,
    first_person_terms: Iterable[str] = _FIRST_PERSON_TERMS,
    minecraft_terms: Iterable[str] = _MINECRAFT_TERMS,
) -> bool:
    """Caption-level selection interface; callers may supply audited vocabularies."""
    text = " ".join(_flatten_text(caption).lower().split())
    if exclude_minecraft and any(term.lower() in text for term in minecraft_terms):
        return False
    if require_first_person:
        perspective = caption.get("perspective") if isinstance(caption, Mapping) else None
        if perspective is not None:
            if not isinstance(perspective, str) or perspective.strip().lower() != "first":
                return False
        elif not any(term.lower() in text for term in first_person_terms):
            return False
    return True


def _episode_paths(payload_root: Path, sample_id: str) -> tuple[Path, Path]:
    episode = payload_root / "data" / sample_id[:2] / sample_id
    return episode / "video.mp4", episode / "annotations.tar"


def _video_size(record: Mapping[str, Any], video_path: Path, config: ManifestConfig) -> tuple[int, str]:
    if video_path.is_file():
        return video_path.stat().st_size, "local_file"
    raw_size = record["video"].get("bytes")
    if isinstance(raw_size, int) and not isinstance(raw_size, bool) and raw_size >= 0:
        return raw_size, "metadata"
    if config.unknown_video_reservation_bytes is not None:
        return config.unknown_video_reservation_bytes, "explicit_reservation"
    raise StorageBudgetError("unknown_video_size")


def _jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for record in records
    )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build_manifest(
    metadata_path: str | Path,
    payload_root: str | Path,
    output_path: str | Path,
    *,
    config: ManifestConfig | None = None,
) -> dict[str, Any]:
    """Inspect local annotation payloads and write a deterministic episode manifest."""
    effective = config or ManifestConfig()
    effective.validate()
    metadata = Path(metadata_path)
    payload = Path(payload_root).resolve()
    output = Path(output_path)
    rejected: Counter[str] = Counter()
    budget = StorageBudget(effective.max_storage_bytes)
    manifest_records: list[dict[str, Any]] = []

    for record in sorted(read_metadata_jsonl(metadata), key=lambda item: item["sample_id"]):
        sample_id = record["sample_id"]
        video_path, annotations_path = _episode_paths(payload, sample_id)
        if not annotations_path.is_file():
            rejected["missing_annotations"] += 1
            continue
        try:
            bundle = read_annotation_bundle(annotations_path)
            actions = parse_action_document(bundle.action)
        except (AnnotationArchiveError, ActionSchemaError):
            rejected["invalid_annotations"] += 1
            continue
        if actions.fps != effective.source_fps:
            rejected["wrong_source_fps"] += 1
            continue
        if not first_person_non_minecraft(
            bundle.caption,
            require_first_person=effective.require_first_person,
            exclude_minecraft=effective.exclude_minecraft,
        ):
            rejected["scene_filter"] += 1
            continue
        windows = actions.eligible_resampled_window_counts(output_fps=effective.output_fps)
        if windows[effective.required_frames] == 0:
            rejected["insufficient_contiguous_frames"] += 1
            continue
        try:
            video_bytes, size_source = _video_size(record, video_path, effective)
            episode_bytes = video_bytes + annotations_path.stat().st_size
            budget.reserve(episode_bytes)
        except StorageBudgetError as exc:
            rejection = "unknown_video_size" if str(exc) == "unknown_video_size" else "storage_budget_exceeded"
            rejected[rejection] += 1
            continue

        manifest_records.append(
            {
                "schema_version": 1,
                "episode_id": sample_id,
                "split": deterministic_split(
                    sample_id,
                    seed=effective.split_seed,
                    train_fraction=effective.train_fraction,
                    dev_fraction=effective.dev_fraction,
                ),
                "video_path": str(video_path),
                "annotations_path": str(annotations_path),
                "video_uri": record["video"]["path"],
                "annotations_uri": record["annotations"],
                "source_fps": actions.fps,
                "output_fps": effective.output_fps,
                "total_action_frames": actions.total_frames,
                "eligible_resampled_windows": {
                    str(size): windows[size] for size in SUPPORTED_WINDOWS
                },
                "allocated_bytes": episode_bytes,
                "video_size_source": size_source,
            }
        )

    manifest_data = _jsonl_bytes(manifest_records)
    _atomic_write(output, manifest_data)
    manifest_sha = hashlib.sha256(manifest_data).hexdigest()
    split_counts = Counter(record["split"] for record in manifest_records)
    receipt = {
        "schema_version": 1,
        "action_keys": list(ACTION_KEYS),
        "manifest": str(output.resolve()),
        "manifest_sha256": manifest_sha,
        "manifest_bytes": len(manifest_data),
        "metadata": str(metadata.resolve()),
        "metadata_sha256": _sha256_file(metadata),
        "records": len(manifest_records),
        "split_counts": {name: split_counts[name] for name in SPLIT_NAMES},
        "rejected": dict(sorted(rejected.items())),
        "storage": {"limit_bytes": budget.limit_bytes, "allocated_bytes": budget.used_bytes},
        "config": asdict(effective),
    }
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    _atomic_write(
        receipt_path,
        (json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    return receipt
