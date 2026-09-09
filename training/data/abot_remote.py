"""Pinned, selective acquisition of the official ABot Explorer dataset.

The module deliberately separates planning from execution.  Reading a local
``metadata.jsonl`` is enough to build a plan; network access and filesystem
writes happen only when :func:`execute_subset` is called explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence
from urllib.parse import quote, unquote, urlsplit

from .abot_manifest import first_person_non_minecraft, read_metadata_jsonl
from .action_schema import (
    OFFICIAL_SOURCE_FPS,
    SUPPORTED_WINDOWS,
    TRAINING_OUTPUT_FPS,
    ActionSchemaError,
    parse_action_document,
)
from .safe_annotations import AnnotationArchiveError, read_annotation_bundle
from training.paths import project_root

OFFICIAL_REPO_ID = "acvlab/ABot-World-Explorer-500h"
OFFICIAL_REVISION = "49118ecb23a069abdab522b3cbd1f3d0588d040c"
DEFAULT_ENDPOINT = "https://huggingface.co"
ALLOWED_ENDPOINTS = (DEFAULT_ENDPOINT, "https://hf-mirror.com")
DEFAULT_ROOT = project_root()
DEFAULT_METADATA_PATH = DEFAULT_ROOT / "data/index/metadata.jsonl"
DEFAULT_PAYLOAD_ROOT = DEFAULT_ROOT / "data/ABot-World-Explorer-500h"
DEFAULT_STATE_DIR = DEFAULT_ROOT / "data/index/abot-first-person-80gb"

# GB is intentionally decimal here.  The values are serialized into every plan
# and receipt so a later run cannot silently reinterpret them as GiB.
DEFAULT_VIDEO_CAP_BYTES = 80_000_000_000
DEFAULT_MIN_FREE_BYTES = 40_000_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


class RemoteDatasetError(RuntimeError):
    """Base error for pinned remote dataset acquisition."""


class RemoteMetadataError(RemoteDatasetError):
    """Raised when local metadata is not pinned to the expected Hub files."""


class RemoteCatalogError(RemoteDatasetError):
    """Raised when paths-info is incomplete or internally inconsistent."""


class DownloadIntegrityError(RemoteDatasetError):
    """Raised when a local or downloaded artifact does not match paths-info."""


class DiskReserveError(RemoteDatasetError):
    """Raised before a write would cross the configured free-space reserve."""


@dataclass(frozen=True)
class SubsetConfig:
    repo_id: str = OFFICIAL_REPO_ID
    revision: str = OFFICIAL_REVISION
    endpoint: str = DEFAULT_ENDPOINT
    video_cap_bytes: int = DEFAULT_VIDEO_CAP_BYTES
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES
    paths_info_batch_size: int = 500
    required_output_frames: int = 241
    source_fps: int = OFFICIAL_SOURCE_FPS
    output_fps: int = TRAINING_OUTPUT_FPS
    target_fill_ratio: float = 0.995
    require_first_person: bool = True
    max_candidates: int | None = None
    require_lfs_sha256: bool = True

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.repo_id):
            raise ValueError(f"invalid repo_id: {self.repo_id!r}")
        if not _REVISION.fullmatch(self.revision):
            raise ValueError("revision must be an immutable 40-character lowercase commit")
        if self.endpoint not in ALLOWED_ENDPOINTS:
            raise ValueError(f"endpoint must be one of {ALLOWED_ENDPOINTS}")
        if self.video_cap_bytes <= 0:
            raise ValueError("video_cap_bytes must be positive")
        if self.min_free_bytes < 0:
            raise ValueError("min_free_bytes cannot be negative")
        if not 1 <= self.paths_info_batch_size <= 1_000:
            raise ValueError("paths_info_batch_size must be in [1, 1000]")
        if self.required_output_frames not in SUPPORTED_WINDOWS:
            raise ValueError(f"required_output_frames must be one of {SUPPORTED_WINDOWS}")
        if self.source_fps != OFFICIAL_SOURCE_FPS:
            raise ValueError(f"source_fps must be {OFFICIAL_SOURCE_FPS}")
        if self.output_fps != TRAINING_OUTPUT_FPS:
            raise ValueError(f"output_fps must be {TRAINING_OUTPUT_FPS}")
        if not 0 < self.target_fill_ratio <= 1:
            raise ValueError("target_fill_ratio must be in (0, 1]")
        if self.max_candidates is not None and self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")


@dataclass(frozen=True)
class EpisodeSpec:
    sample_id: str
    video_path: str
    annotation_path: str
    video_uri: str
    annotation_uri: str


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    sha256: str | None
    blob_oid: str | None = None

    def validate(self, *, require_sha256: bool = True) -> None:
        _validate_remote_path(self.path)
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise RemoteCatalogError(f"invalid size for {self.path!r}: {self.size!r}")
        if self.sha256 is not None and not _SHA256.fullmatch(self.sha256):
            raise RemoteCatalogError(f"invalid LFS SHA-256 for {self.path!r}")
        if require_sha256 and self.sha256 is None:
            raise RemoteCatalogError(f"missing LFS SHA-256 for {self.path!r}")


@dataclass(frozen=True)
class DownloadResult:
    remote_path: str
    local_path: str
    status: str
    file_bytes: int
    transferred_bytes: int
    sha256: str


class DownloadResponse(Protocol):
    status: int
    headers: Mapping[str, str]

    def read(self, amount: int = -1) -> bytes: ...

    def close(self) -> None: ...


class RemoteClient(Protocol):
    def get_paths_info(self, paths: Sequence[str]) -> Sequence[RemoteFile]: ...

    def open_file(self, remote: RemoteFile, *, offset: int) -> DownloadResponse: ...


class HuggingFaceClient:
    """Thin Hub adapter; heavyweight imports are delayed until execution."""

    def __init__(
        self,
        *,
        repo_id: str,
        revision: str,
        endpoint: str = DEFAULT_ENDPOINT,
        token: str | bool | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        from huggingface_hub import HfApi
        import requests

        self.repo_id = repo_id
        self.revision = revision
        if endpoint not in ALLOWED_ENDPOINTS:
            raise ValueError(f"endpoint must be one of {ALLOWED_ENDPOINTS}")
        self.endpoint = endpoint.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self._api = HfApi(endpoint=self.endpoint, token=token)
        self._session = requests.Session()

    def get_paths_info(self, paths: Sequence[str]) -> Sequence[RemoteFile]:
        entries = self._api.get_paths_info(
            self.repo_id,
            list(paths),
            repo_type="dataset",
            revision=self.revision,
            token=self.token,
        )
        files: list[RemoteFile] = []
        for entry in entries:
            if not hasattr(entry, "size"):
                raise RemoteCatalogError(f"paths-info returned a folder for {entry.path!r}")
            lfs = getattr(entry, "lfs", None)
            sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
            files.append(
                RemoteFile(
                    path=entry.path,
                    size=entry.size,
                    sha256=sha256,
                    blob_oid=getattr(entry, "blob_id", None),
                )
            )
        return files

    def open_file(self, remote: RemoteFile, *, offset: int) -> DownloadResponse:
        url = (
            f"{self.endpoint}/datasets/"
            f"{self.repo_id}/resolve/{self.revision}/{quote(remote.path, safe='/')}"
        )
        headers: dict[str, str] = {"Accept-Encoding": "identity"}
        if isinstance(self.token, str) and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if offset:
            headers["Range"] = f"bytes={offset}-"
        response = self._session.get(
            url,
            headers=headers,
            stream=True,
            timeout=(self.timeout_seconds, self.timeout_seconds),
        )
        if response.status_code not in (200, 206):
            try:
                response.raise_for_status()
            finally:
                response.close()
        response.raw.decode_content = True
        return _RequestsResponse(response)


class _RequestsResponse:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.status = int(response.status_code)
        self.headers = response.headers

    def read(self, amount: int = -1) -> bytes:
        return self._response.raw.read(amount)

    def close(self) -> None:
        self._response.close()


def _validate_remote_path(path: str) -> None:
    if not isinstance(path, str) or not path or "\\" in path:
        raise RemoteMetadataError(f"unsafe remote path: {path!r}")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise RemoteMetadataError(f"unsafe remote path: {path!r}")


def _path_from_hf_uri(uri: str, *, repo_id: str, revision: str) -> str:
    prefix = f"hf://datasets/{repo_id}@{revision}/"
    if not uri.startswith(prefix):
        raise RemoteMetadataError(
            f"URI is not pinned to datasets/{repo_id}@{revision}: {uri!r}"
        )
    path = unquote(uri[len(prefix) :])
    _validate_remote_path(path)
    return path


def _path_from_annotation_uri(uri: str, *, repo_id: str, revision: str) -> str:
    if uri.startswith("hf://"):
        return _path_from_hf_uri(uri, repo_id=repo_id, revision=revision)
    parsed = urlsplit(uri)
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
        raise RemoteMetadataError(f"annotation URI is not on huggingface.co: {uri!r}")
    prefix = f"/datasets/{repo_id}/resolve/{revision}/"
    if not parsed.path.startswith(prefix):
        raise RemoteMetadataError(
            f"annotation URI is not pinned to {repo_id}@{revision}: {uri!r}"
        )
    path = unquote(parsed.path[len(prefix) :])
    _validate_remote_path(path)
    return path


def read_pinned_episode_specs(
    metadata_path: str | Path,
    *,
    repo_id: str = OFFICIAL_REPO_ID,
    revision: str = OFFICIAL_REVISION,
) -> list[EpisodeSpec]:
    specs: list[EpisodeSpec] = []
    seen_paths: set[str] = set()
    for record in sorted(read_metadata_jsonl(metadata_path), key=lambda item: item["sample_id"]):
        sample_id = record["sample_id"]
        video_uri = record["video"]["path"]
        annotation_uri = record["annotations"]
        video_path = _path_from_hf_uri(video_uri, repo_id=repo_id, revision=revision)
        annotation_path = _path_from_annotation_uri(
            annotation_uri, repo_id=repo_id, revision=revision
        )
        expected_dir = f"data/{sample_id[:2]}/{sample_id}"
        expected_video = f"{expected_dir}/video.mp4"
        expected_annotation = f"{expected_dir}/annotations.tar"
        if video_path != expected_video or annotation_path != expected_annotation:
            raise RemoteMetadataError(
                f"sample {sample_id!r} does not use the canonical episode paths"
            )
        if video_path in seen_paths or annotation_path in seen_paths:
            raise RemoteMetadataError(f"duplicate remote path for sample {sample_id!r}")
        seen_paths.update((video_path, annotation_path))
        specs.append(
            EpisodeSpec(
                sample_id=sample_id,
                video_path=video_path,
                annotation_path=annotation_path,
                video_uri=video_uri,
                annotation_uri=annotation_uri,
            )
        )
    return specs


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
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


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode("utf-8")
        for record in records
    )


def _chunks(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def build_subset_plan(
    metadata_path: str | Path,
    *,
    payload_root: str | Path = DEFAULT_PAYLOAD_ROOT,
    state_dir: str | Path = DEFAULT_STATE_DIR,
    config: SubsetConfig | None = None,
) -> dict[str, Any]:
    """Build a local-only plan.  This function performs no network calls or writes."""
    effective = config or SubsetConfig()
    effective.validate()
    metadata = Path(metadata_path)
    specs = read_pinned_episode_specs(
        metadata, repo_id=effective.repo_id, revision=effective.revision
    )
    candidate_count = len(specs)
    if effective.max_candidates is not None:
        candidate_count = min(candidate_count, effective.max_candidates)
    remote_paths = candidate_count * 2
    return {
        "schema_version": 1,
        "mode": "plan-only",
        "network_access": False,
        "filesystem_writes": False,
        "metadata": str(metadata.resolve()),
        "metadata_sha256": _sha256_file(metadata),
        "payload_root": str(Path(payload_root).resolve()),
        "state_dir": str(Path(state_dir).resolve()),
        "candidate_episodes": candidate_count,
        "remote_paths": remote_paths,
        "paths_info_batches_at_most": math.ceil(
            remote_paths / effective.paths_info_batch_size
        ),
        "selection_order": "sample_id ascending; greedy without exceeding video_cap_bytes",
        "execution_order": [
            "paths-info in bounded batches",
            "annotations download and CPU validation",
            "selection manifest atomically committed",
            "selected videos downloaded, ffprobe/alignment validated, and failures replenished",
        ],
        "config": asdict(effective),
    }


def _catalog_record(repo_id: str, revision: str, remote: RemoteFile) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "repo_id": repo_id,
        "revision": revision,
        "path": remote.path,
        "size": remote.size,
        "sha256": remote.sha256,
        "blob_oid": remote.blob_oid,
    }


def _load_catalog(
    path: Path, *, repo_id: str, revision: str, require_sha256: bool
) -> dict[str, RemoteFile]:
    if not path.is_file():
        return {}
    catalog: dict[str, RemoteFile] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RemoteCatalogError(f"catalog line {line_number}: invalid JSON") from exc
            if record.get("repo_id") != repo_id or record.get("revision") != revision:
                raise RemoteCatalogError("catalog repo/revision does not match this execution")
            remote = RemoteFile(
                path=record.get("path"),
                size=record.get("size"),
                sha256=record.get("sha256"),
                blob_oid=record.get("blob_oid"),
            )
            remote.validate(require_sha256=require_sha256)
            if remote.path in catalog:
                raise RemoteCatalogError(f"duplicate catalog path {remote.path!r}")
            catalog[remote.path] = remote
    return catalog


def _write_catalog(
    path: Path, *, repo_id: str, revision: str, catalog: Mapping[str, RemoteFile]
) -> None:
    records = (
        _catalog_record(repo_id, revision, catalog[name]) for name in sorted(catalog)
    )
    _atomic_write(path, _jsonl_bytes(records))


def _query_missing(
    client: RemoteClient,
    paths: Sequence[str],
    *,
    catalog: dict[str, RemoteFile],
    config: SubsetConfig,
) -> None:
    missing = [path for path in paths if path not in catalog]
    for batch in _chunks(missing, config.paths_info_batch_size):
        requested = set(batch)
        response = list(client.get_paths_info(batch))
        found: dict[str, RemoteFile] = {}
        for remote in response:
            remote.validate(require_sha256=config.require_lfs_sha256)
            if remote.path not in requested:
                raise RemoteCatalogError(
                    f"paths-info returned unrequested path {remote.path!r}"
                )
            if remote.path in found:
                raise RemoteCatalogError(f"paths-info duplicated {remote.path!r}")
            found[remote.path] = remote
        absent = requested.difference(found)
        if absent:
            raise RemoteCatalogError(f"paths-info omitted {sorted(absent)!r}")
        catalog.update(found)


def _episode_destination(payload_root: Path, spec: EpisodeSpec, *, annotation: bool) -> Path:
    name = "annotations.tar" if annotation else "video.mp4"
    return payload_root / "data" / spec.sample_id[:2] / spec.sample_id / name


def _nearest_existing(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise DiskReserveError(f"cannot find an existing ancestor for {path}")
        candidate = parent
    return candidate


def _ensure_free_space(
    path: Path,
    additional_bytes: int,
    min_free_bytes: int,
    disk_usage: Callable[[str | os.PathLike[str]], Any],
) -> None:
    usage = disk_usage(_nearest_existing(path))
    if usage.free - additional_bytes < min_free_bytes:
        raise DiskReserveError(
            f"write blocked: free={usage.free}, needed={additional_bytes}, "
            f"reserve={min_free_bytes}"
        )


def _part_paths(destination: Path) -> tuple[Path, Path]:
    part = destination.with_name(destination.name + ".part")
    return part, part.with_name(part.name + ".meta.json")


def _expected_part_metadata(remote: RemoteFile) -> dict[str, Any]:
    return {"path": remote.path, "size": remote.size, "sha256": remote.sha256}


def _validate_part(remote: RemoteFile, destination: Path) -> int:
    part, meta = _part_paths(destination)
    if not part.exists():
        return 0
    if not part.is_file() or not meta.is_file():
        raise DownloadIntegrityError(f"untrusted partial state for {destination}")
    try:
        recorded = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DownloadIntegrityError(f"invalid partial metadata for {destination}") from exc
    if recorded != _expected_part_metadata(remote):
        raise DownloadIntegrityError(f"partial metadata mismatch for {destination}")
    size = part.stat().st_size
    if size > remote.size:
        raise DownloadIntegrityError(f"partial file is oversized for {destination}")
    return size


def _verify_complete(remote: RemoteFile, destination: Path) -> DownloadResult | None:
    if not destination.exists():
        return None
    if not destination.is_file() or destination.stat().st_size != remote.size:
        raise DownloadIntegrityError(f"existing destination size mismatch: {destination}")
    actual_sha256 = _sha256_file(destination)
    if remote.sha256 is not None and actual_sha256 != remote.sha256:
        raise DownloadIntegrityError(f"existing destination hash mismatch: {destination}")
    return DownloadResult(
        remote_path=remote.path,
        local_path=str(destination.resolve()),
        status="reused",
        file_bytes=remote.size,
        transferred_bytes=0,
        sha256=actual_sha256,
    )


def remaining_download_bytes(remote: RemoteFile, destination: Path) -> int:
    existing = _verify_complete(remote, destination)
    if existing is not None:
        return 0
    return remote.size - _validate_part(remote, destination)


def download_remote_file(
    client: RemoteClient,
    remote: RemoteFile,
    destination: str | Path,
    *,
    min_free_bytes: int,
    disk_usage: Callable[[str | os.PathLike[str]], Any] = shutil.disk_usage,
    chunk_bytes: int = 4 * 1024 * 1024,
) -> DownloadResult:
    """Resume into ``.part`` and expose the final path only after size/hash checks."""
    remote.validate(require_sha256=False)
    target = Path(destination)
    reused = _verify_complete(remote, target)
    if reused is not None:
        return reused

    target.parent.mkdir(parents=True, exist_ok=True)
    part, meta = _part_paths(target)
    offset = _validate_part(remote, target)
    _ensure_free_space(target, remote.size - offset, min_free_bytes, disk_usage)
    if not meta.exists():
        _atomic_write(meta, _json_bytes(_expected_part_metadata(remote)))

    response = client.open_file(remote, offset=offset)
    initial_offset = offset
    try:
        if offset and response.status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                raise DownloadIntegrityError(
                    f"invalid Content-Range for {remote.path!r}: {content_range!r}"
                )
            mode = "ab"
        elif response.status == 200:
            # Some storage backends ignore Range.  Restart only the private part;
            # the verified destination is never touched.
            offset = 0
            mode = "wb"
        elif not offset and response.status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith("bytes 0-"):
                raise DownloadIntegrityError(
                    f"invalid Content-Range for {remote.path!r}: {content_range!r}"
                )
            mode = "wb"
        else:
            raise DownloadIntegrityError(
                f"unexpected HTTP status {response.status} for {remote.path!r}"
            )

        written = offset
        transferred = 0
        with part.open(mode) as stream:
            while True:
                chunk = response.read(chunk_bytes)
                if not chunk:
                    break
                stream.write(chunk)
                written += len(chunk)
                transferred += len(chunk)
                if written > remote.size:
                    raise DownloadIntegrityError(f"download is oversized: {remote.path!r}")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        response.close()

    if part.stat().st_size != remote.size:
        raise DownloadIntegrityError(
            f"incomplete download for {remote.path!r}: {part.stat().st_size}/{remote.size}"
        )
    actual_sha256 = _sha256_file(part)
    if remote.sha256 is not None and actual_sha256 != remote.sha256:
        raise DownloadIntegrityError(f"SHA-256 mismatch for {remote.path!r}")
    os.replace(part, target)
    try:
        meta.unlink()
    except FileNotFoundError:
        pass
    return DownloadResult(
        remote_path=remote.path,
        local_path=str(target.resolve()),
        status="resumed" if initial_offset and response.status == 206 else "downloaded",
        file_bytes=remote.size,
        transferred_bytes=transferred,
        sha256=actual_sha256,
    )


def _classify_annotation(path: Path, config: SubsetConfig) -> tuple[str | None, dict[str, Any]]:
    try:
        bundle = read_annotation_bundle(path)
    except AnnotationArchiveError:
        return "invalid_annotation_archive", {}
    try:
        actions = parse_action_document(bundle.action)
    except ActionSchemaError as exc:
        message = str(exc)
        if "unsupported active action key" in message:
            return "active_Q_E_or_Space", {}
        if "unknown action key" in message:
            return "unknown_action_key", {}
        return "invalid_action_document", {}
    if actions.fps != config.source_fps:
        return "wrong_source_fps", {"source_fps": actions.fps}
    if not first_person_non_minecraft(
        bundle.caption, require_first_person=False, exclude_minecraft=True
    ):
        return "minecraft", {}
    if config.require_first_person and not first_person_non_minecraft(
        bundle.caption, require_first_person=True, exclude_minecraft=False
    ):
        return "not_first_person", {}
    windows = actions.eligible_resampled_window_counts(
        (config.required_output_frames,), output_fps=config.output_fps
    )
    eligible = windows[config.required_output_frames]
    if eligible == 0:
        return "insufficient_resampled_frames", {
            "source_fps": actions.fps,
            "output_fps": config.output_fps,
            "required_output_frames": config.required_output_frames,
        }
    return None, {
        "source_fps": actions.fps,
        "output_fps": config.output_fps,
        "total_action_frames": actions.total_frames,
        "required_output_frames": config.required_output_frames,
        "eligible_resampled_windows": eligible,
        "resampled_source_span": actions.resampled_offsets(
            config.required_output_frames, output_fps=config.output_fps
        )[-1]
        + 1,
    }


def _download_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "files": len(records),
        "transferred_bytes": sum(int(item["transferred_bytes"]) for item in records),
        "materialized_bytes": sum(int(item["file_bytes"]) for item in records),
        "reused_bytes": sum(
            int(item["file_bytes"]) for item in records if item["status"] == "reused"
        ),
        "by_kind": {},
    }
    for kind in ("annotation", "video"):
        chosen = [record for record in records if record["kind"] == kind]
        summary["by_kind"][kind] = {
            "files": len(chosen),
            "transferred_bytes": sum(int(item["transferred_bytes"]) for item in chosen),
            "materialized_bytes": sum(int(item["file_bytes"]) for item in chosen),
            "reused_bytes": sum(
                int(item["file_bytes"]) for item in chosen if item["status"] == "reused"
            ),
        }
    return summary


def _download_record(result: DownloadResult, *, kind: str, sample_id: str) -> dict[str, Any]:
    return {"kind": kind, "sample_id": sample_id, **asdict(result)}


def validate_downloaded_episode(video_path: Path, annotation_path: Path) -> Mapping[str, Any]:
    """Run the same strict CPU video gate used by feature caching.

    The import is delayed so plan-only mode does not import torch or inspect CUDA.
    """
    from scripts.cache_abot_features import probe_video, validate_action_video_alignment

    sequence = parse_action_document(read_annotation_bundle(annotation_path).action)
    probe = probe_video(video_path)
    validate_action_video_alignment(sequence, probe)
    return probe


def execute_subset(
    metadata_path: str | Path,
    *,
    payload_root: str | Path = DEFAULT_PAYLOAD_ROOT,
    state_dir: str | Path = DEFAULT_STATE_DIR,
    config: SubsetConfig | None = None,
    client: RemoteClient,
    disk_usage: Callable[[str | os.PathLike[str]], Any] = shutil.disk_usage,
    video_validator: Callable[[Path, Path], Mapping[str, Any]] = validate_downloaded_episode,
) -> dict[str, Any]:
    """Fetch annotations, select under a hard cap, then fetch and validate videos.

    A video that fails the codec/FPS/frame-alignment gate is rejected locally and
    selection continues with later annotations.  Thus one corrupt upstream item
    cannot poison the later feature-cache job.
    """
    effective = config or SubsetConfig()
    effective.validate()
    metadata = Path(metadata_path)
    payload = Path(payload_root)
    state = Path(state_dir)
    specs = read_pinned_episode_specs(
        metadata, repo_id=effective.repo_id, revision=effective.revision
    )
    if effective.max_candidates is not None:
        specs = specs[: effective.max_candidates]
    metadata_sha256 = _sha256_file(metadata)

    payload.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    catalog_path = state / "remote_catalog.jsonl"
    selected_path = state / "selected_metadata.jsonl"
    downloads_path = state / "downloads.jsonl"
    receipt_path = state / "subset.receipt.json"
    catalog = _load_catalog(
        catalog_path,
        repo_id=effective.repo_id,
        revision=effective.revision,
        require_sha256=effective.require_lfs_sha256,
    )
    if not catalog_path.is_file():
        _write_catalog(
            catalog_path,
            repo_id=effective.repo_id,
            revision=effective.revision,
            catalog=catalog,
        )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    validated_ids: set[str] = set()
    invalid_video_ids: set[str] = set()
    validation_failures: list[dict[str, str]] = []
    overflow: list[tuple[EpisodeSpec, RemoteFile, RemoteFile, dict[str, Any]]] = []
    selected_video_bytes = 0
    selected_annotation_bytes = 0
    rejected_counts: Counter[str] = Counter()
    rejected_video_bytes: Counter[str] = Counter()
    rejected_annotation_bytes: Counter[str] = Counter()
    download_records: list[dict[str, Any]] = []
    considered = 0
    target_bytes = math.ceil(effective.video_cap_bytes * effective.target_fill_ratio)
    episodes_per_query_wave = max(1, effective.paths_info_batch_size // 2)
    next_spec_index = 0
    catalog_wave_end = 0
    spec_by_id = {spec.sample_id: spec for spec in specs}

    def reject(reason: str, annotation: RemoteFile, video: RemoteFile) -> None:
        rejected_counts[reason] += 1
        rejected_annotation_bytes[reason] += annotation.size
        rejected_video_bytes[reason] += video.size

    def add_selected(
        spec: EpisodeSpec,
        annotation_remote: RemoteFile,
        video_remote: RemoteFile,
        facts: dict[str, Any],
    ) -> bool:
        nonlocal selected_video_bytes, selected_annotation_bytes
        if spec.sample_id in invalid_video_ids or spec.sample_id in selected_ids:
            return False
        if selected_video_bytes + video_remote.size > effective.video_cap_bytes:
            return False
        selected.append(
            {
                "sample_id": spec.sample_id,
                "video": {
                    "path": spec.video_uri,
                    "bytes": video_remote.size,
                    "sha256": video_remote.sha256,
                },
                "annotations": spec.annotation_uri,
                "annotation_bytes": annotation_remote.size,
                "annotation_sha256": annotation_remote.sha256,
                "selection": facts,
            }
        )
        selected_ids.add(spec.sample_id)
        selected_video_bytes += video_remote.size
        selected_annotation_bytes += annotation_remote.size
        return True

    def ensure_next_catalog_wave() -> None:
        nonlocal catalog_wave_end
        if next_spec_index < catalog_wave_end or next_spec_index >= len(specs):
            return
        catalog_wave_end = min(len(specs), next_spec_index + episodes_per_query_wave)
        wave = specs[next_spec_index:catalog_wave_end]
        paths = [path for spec in wave for path in (spec.annotation_path, spec.video_path)]
        _query_missing(client, paths, catalog=catalog, config=effective)
        _write_catalog(
            catalog_path,
            repo_id=effective.repo_id,
            revision=effective.revision,
            catalog=catalog,
        )

    def fill_selection() -> int:
        """Fill from cached overflow first, then inspect later annotations."""
        nonlocal next_spec_index, considered
        added = 0
        still_overflow: list[tuple[EpisodeSpec, RemoteFile, RemoteFile, dict[str, Any]]] = []
        for candidate in overflow:
            if selected_video_bytes >= target_bytes:
                still_overflow.append(candidate)
                continue
            if add_selected(*candidate):
                added += 1
            else:
                still_overflow.append(candidate)
        overflow[:] = still_overflow

        while selected_video_bytes < target_bytes and next_spec_index < len(specs):
            ensure_next_catalog_wave()
            spec = specs[next_spec_index]
            next_spec_index += 1
            considered += 1
            annotation_remote = catalog[spec.annotation_path]
            video_remote = catalog[spec.video_path]
            annotation_destination = _episode_destination(payload, spec, annotation=True)
            annotation_result = download_remote_file(
                client,
                annotation_remote,
                annotation_destination,
                min_free_bytes=effective.min_free_bytes,
                disk_usage=disk_usage,
            )
            download_records.append(
                _download_record(annotation_result, kind="annotation", sample_id=spec.sample_id)
            )
            reason, facts = _classify_annotation(annotation_destination, effective)
            if reason is not None:
                reject(reason, annotation_remote, video_remote)
                continue
            candidate = (spec, annotation_remote, video_remote, facts)
            if add_selected(*candidate):
                added += 1
            else:
                overflow.append(candidate)
        return added

    def receipt(phase: str) -> dict[str, Any]:
        selected_data = _jsonl_bytes(selected)
        rejection_by_reason = {
            reason: {
                "count": rejected_counts[reason],
                "annotation_bytes": rejected_annotation_bytes[reason],
                "video_bytes": rejected_video_bytes[reason],
            }
            for reason in sorted(rejected_counts)
        }
        return {
            "schema_version": 1,
            "phase": phase,
            "repo_id": effective.repo_id,
            "revision": effective.revision,
            "metadata": str(metadata.resolve()),
            "metadata_sha256": metadata_sha256,
            "catalog": str(catalog_path.resolve()),
            "catalog_sha256": _sha256_file(catalog_path),
            "selected_metadata": str(selected_path.resolve()),
            "selected_metadata_sha256": hashlib.sha256(selected_data).hexdigest(),
            "selection": {
                "candidates_available": len(specs),
                "candidates_considered": considered,
                "candidates_uninspected": len(specs) - considered,
                "selected_count": len(selected),
                "selected_video_bytes": selected_video_bytes,
                "selected_annotation_bytes": selected_annotation_bytes,
                "hard_video_cap_bytes": effective.video_cap_bytes,
                "target_fill_bytes": target_bytes,
                "target_reached": selected_video_bytes >= target_bytes,
            },
            "rejected": {
                "count": sum(rejected_counts.values()),
                "video_bytes": sum(rejected_video_bytes.values()),
                "annotation_bytes": sum(rejected_annotation_bytes.values()),
                "by_reason": rejection_by_reason,
                "video_validation_failures": validation_failures,
            },
            "downloads": _download_summary(download_records),
            "config": asdict(effective),
        }

    try:
        while True:
            added = fill_selection()
            if selected_video_bytes > effective.video_cap_bytes:
                raise AssertionError("selector exceeded the hard video cap")

            # Publish a complete candidate list before any newly selected video
            # is fetched.  A failed video is removed and this phase repeats.
            _atomic_write(selected_path, _jsonl_bytes(selected))
            _atomic_write(downloads_path, _jsonl_bytes(download_records))
            _atomic_write(receipt_path, _json_bytes(receipt("selection_complete")))

            pending = [record for record in selected if record["sample_id"] not in validated_ids]
            missing_video_bytes = 0
            for record in pending:
                spec = spec_by_id[record["sample_id"]]
                missing_video_bytes += remaining_download_bytes(
                    catalog[spec.video_path],
                    _episode_destination(payload, spec, annotation=False),
                )
            _ensure_free_space(payload, missing_video_bytes, effective.min_free_bytes, disk_usage)

            failed_this_round: set[str] = set()
            for record in pending:
                sample_id = record["sample_id"]
                spec = spec_by_id[sample_id]
                video_destination = _episode_destination(payload, spec, annotation=False)
                result = download_remote_file(
                    client,
                    catalog[spec.video_path],
                    video_destination,
                    min_free_bytes=effective.min_free_bytes,
                    disk_usage=disk_usage,
                )
                download_records.append(_download_record(result, kind="video", sample_id=sample_id))
                try:
                    probe = dict(
                        video_validator(
                            video_destination,
                            _episode_destination(payload, spec, annotation=True),
                        )
                    )
                except Exception as exc:
                    failed_this_round.add(sample_id)
                    invalid_video_ids.add(sample_id)
                    reject("invalid_video", catalog[spec.annotation_path], catalog[spec.video_path])
                    validation_failures.append(
                        {
                            "sample_id": sample_id,
                            "error_type": type(exc).__name__,
                            "message": str(exc)[:500],
                        }
                    )
                else:
                    record["video_validation"] = probe
                    validated_ids.add(sample_id)

            if failed_this_round:
                kept: list[dict[str, Any]] = []
                for record in selected:
                    if record["sample_id"] not in failed_this_round:
                        kept.append(record)
                        continue
                    selected_ids.remove(record["sample_id"])
                    selected_video_bytes -= int(record["video"]["bytes"])
                    selected_annotation_bytes -= int(record["annotation_bytes"])
                selected[:] = kept
                continue

            if selected_video_bytes >= target_bytes:
                break
            if next_spec_index >= len(specs) and added == 0:
                break
    except BaseException as exc:
        failed = receipt("video_download_failed") if catalog_path.is_file() else {
            "schema_version": 1,
            "phase": "video_download_failed",
        }
        failed["failure_type"] = type(exc).__name__
        _atomic_write(downloads_path, _jsonl_bytes(download_records))
        _atomic_write(receipt_path, _json_bytes(failed))
        raise

    # Only candidates still unable to fit are final cap rejections.  They were
    # not counted earlier because a failed video may have opened capacity.
    for _, annotation_remote, video_remote, _ in overflow:
        reject("video_cap_exceeded", annotation_remote, video_remote)
    _atomic_write(selected_path, _jsonl_bytes(selected))
    _atomic_write(downloads_path, _jsonl_bytes(download_records))
    final_receipt = receipt("complete")
    final_receipt.update(
        {
            "downloads_manifest": str(downloads_path.resolve()),
            "downloads_manifest_sha256": _sha256_file(downloads_path),
        }
    )
    _atomic_write(receipt_path, _json_bytes(final_receipt))
    return final_receipt
