"""Read ABot annotation tarballs without extracting untrusted paths."""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

DEFAULT_MAX_MEMBER_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_MEMBERS = 20_000


class AnnotationArchiveError(ValueError):
    """Raised when an annotation archive is unsafe or malformed."""


@dataclass(frozen=True)
class AnnotationBundle:
    action: dict[str, Any]
    caption: Any | None
    member_names: tuple[str, ...]


def _validate_member(member: tarfile.TarInfo) -> None:
    name = member.name
    if not name or "\\" in name:
        raise AnnotationArchiveError(f"unsafe archive member name: {name!r}")
    path = PurePosixPath(name)
    has_windows_drive = bool(path.parts and path.parts[0].endswith(":"))
    if path.is_absolute() or has_windows_drive or ".." in path.parts:
        raise AnnotationArchiveError(f"archive path traversal rejected: {name!r}")
    if member.issym() or member.islnk():
        raise AnnotationArchiveError(f"archive link rejected: {name!r}")
    if not (member.isfile() or member.isdir()):
        raise AnnotationArchiveError(f"unsupported archive member type: {name!r}")


def _read_json_member(
    archive: tarfile.TarFile,
    members: list[tarfile.TarInfo],
    basename: str,
    *,
    required: bool,
) -> Any | None:
    matches = [member for member in members if member.isfile() and PurePosixPath(member.name).name == basename]
    if not matches:
        if required:
            raise AnnotationArchiveError(f"archive is missing required {basename}")
        return None
    if len(matches) != 1:
        raise AnnotationArchiveError(f"archive contains duplicate {basename} members")
    stream = archive.extractfile(matches[0])
    if stream is None:
        raise AnnotationArchiveError(f"cannot read {matches[0].name!r}")
    try:
        return json.loads(stream.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnnotationArchiveError(f"invalid JSON in {matches[0].name!r}: {exc}") from exc


def read_annotation_bundle(
    path: str | Path,
    *,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
    max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
    max_members: int = DEFAULT_MAX_MEMBERS,
) -> AnnotationBundle:
    """Validate and read action/caption JSON without writing archive contents to disk."""
    archive_path = Path(path)
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            members = archive.getmembers()
            if len(members) > max_members:
                raise AnnotationArchiveError(
                    f"archive has {len(members)} members; limit is {max_members}"
                )
            total_bytes = 0
            for member in members:
                _validate_member(member)
                if member.size < 0 or member.size > max_member_bytes:
                    raise AnnotationArchiveError(
                        f"member {member.name!r} size {member.size} exceeds limit"
                    )
                total_bytes += member.size
                if total_bytes > max_archive_bytes:
                    raise AnnotationArchiveError(
                        f"uncompressed archive size exceeds {max_archive_bytes} bytes"
                    )
            action = _read_json_member(archive, members, "action.json", required=True)
            if not isinstance(action, dict):
                raise AnnotationArchiveError("action.json must contain a JSON object")
            caption = _read_json_member(archive, members, "caption.json", required=False)
            return AnnotationBundle(
                action=action,
                caption=caption,
                member_names=tuple(member.name for member in members),
            )
    except (tarfile.TarError, OSError) as exc:
        raise AnnotationArchiveError(f"cannot read annotation archive {archive_path}: {exc}") from exc
