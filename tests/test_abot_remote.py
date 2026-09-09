from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.data.abot_remote import (
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    DiskReserveError,
    DownloadIntegrityError,
    RemoteFile,
    RemoteMetadataError,
    SubsetConfig,
    build_subset_plan,
    download_remote_file,
    execute_subset,
    read_pinned_episode_specs,
)


def _annotation(*, perspective: str = "first", active_aux: str | None = None) -> bytes:
    frames = []
    for index in range(451):
        keys = {key: False for key in "WASDIJKL"}
        keys.update({"Q": False, "E": False, "Space": False})
        if active_aux:
            keys[active_aux] = True
        frames.append(
            {
                "frame_id": f"frame_{index + 1:06d}",
                "timestamp": index / 30,
                "keys": keys,
            }
        )
    action = {"fps": 30, "total_frames": len(frames), "frames": frames}
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, value in (
            ("episode/action.json", action),
            ("episode/caption.json", {"perspective": perspective, "narrative": "road"}),
        ):
            payload = json.dumps(value).encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return stream.getvalue()


def _metadata_record(sample_id: str) -> dict:
    episode = f"data/{sample_id[:2]}/{sample_id}"
    return {
        "sample_id": sample_id,
        "video": {
            "path": (
                f"hf://datasets/{OFFICIAL_REPO_ID}@{OFFICIAL_REVISION}/"
                f"{episode}/video.mp4"
            ),
            "bytes": None,
        },
        "annotations": (
            f"https://huggingface.co/datasets/{OFFICIAL_REPO_ID}/resolve/"
            f"{OFFICIAL_REVISION}/{episode}/annotations.tar"
        ),
    }


def _write_metadata(path: Path, sample_ids: list[str]) -> None:
    path.write_text(
        "".join(json.dumps(_metadata_record(sample_id)) + "\n" for sample_id in sample_ids),
        encoding="utf-8",
    )


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, offset: int):
        super().__init__(payload[offset:])
        self.status = 206 if offset else 200
        self.headers = (
            {"Content-Range": f"bytes {offset}-{len(payload) - 1}/{len(payload)}"}
            if offset
            else {}
        )


class _FakeClient:
    def __init__(self, payloads: dict[str, bytes]):
        self.payloads = payloads
        self.events: list[tuple[str, object]] = []

    def get_paths_info(self, paths: list[str]):
        self.events.append(("paths_info", tuple(paths)))
        return [
            RemoteFile(
                path=path,
                size=len(self.payloads[path]),
                sha256=hashlib.sha256(self.payloads[path]).hexdigest(),
                blob_oid="fake",
            )
            for path in reversed(paths)
        ]

    def open_file(self, remote: RemoteFile, *, offset: int):
        self.events.append(("download", remote.path))
        return _Response(self.payloads[remote.path], offset)


def _fake_payloads(sample_payloads: dict[str, tuple[bytes, bytes]]) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for sample_id, (annotation, video) in sample_payloads.items():
        episode = f"data/{sample_id[:2]}/{sample_id}"
        result[f"{episode}/annotations.tar"] = annotation
        result[f"{episode}/video.mp4"] = video
    return result


def test_plan_only_has_no_network_and_creates_no_output(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.jsonl"
    _write_metadata(metadata, ["aa0001", "bb0002"])
    output = tmp_path / "never-created"

    plan = build_subset_plan(
        metadata,
        payload_root=output / "payload",
        state_dir=output / "state",
        config=SubsetConfig(video_cap_bytes=100, min_free_bytes=0),
    )

    assert plan["mode"] == "plan-only"
    assert plan["network_access"] is False
    assert plan["candidate_episodes"] == 2
    assert not output.exists()


def test_subset_endpoint_is_restricted_to_known_hub_hosts() -> None:
    SubsetConfig(endpoint="https://hf-mirror.com").validate()
    with pytest.raises(ValueError, match="endpoint must be one of"):
        SubsetConfig(endpoint="https://example.invalid").validate()


def test_explicit_explorer_mode_accepts_third_person(
    tmp_path: Path,
) -> None:
    third = tmp_path / "third.tar"
    third.write_bytes(_annotation(perspective="third"))
    from training.data.abot_remote import _classify_annotation

    reason, _ = _classify_annotation(
        third,
        SubsetConfig(
            video_cap_bytes=100,
            min_free_bytes=0,
            require_first_person=False,
        ),
    )
    assert reason is None


def test_metadata_must_match_immutable_official_revision(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.jsonl"
    record = _metadata_record("aa0001")
    record["video"]["path"] = record["video"]["path"].replace(OFFICIAL_REVISION, "main")
    metadata.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(RemoteMetadataError, match="not pinned"):
        read_pinned_episode_specs(metadata)


def test_execute_uses_paths_info_cap_and_annotation_first_policy(tmp_path: Path) -> None:
    samples = {
        "aa0001": (_annotation(), b"a" * 60),
        "bb0002": (_annotation(perspective="third"), b"b" * 30),
        "cc0003": (_annotation(active_aux="Q"), b"c" * 20),
        "dd0004": (_annotation(), b"d" * 40),
    }
    metadata = tmp_path / "metadata.jsonl"
    _write_metadata(metadata, list(samples))
    client = _FakeClient(_fake_payloads(samples))

    receipt = execute_subset(
        metadata,
        payload_root=tmp_path / "payload",
        state_dir=tmp_path / "state",
        config=SubsetConfig(
            video_cap_bytes=100,
            min_free_bytes=0,
            paths_info_batch_size=4,
            target_fill_ratio=1.0,
        ),
        client=client,
        disk_usage=lambda _: SimpleNamespace(free=1_000_000_000),
        video_validator=lambda video, annotation: {"validated": True},
    )

    assert receipt["phase"] == "complete"
    assert receipt["selection"]["selected_video_bytes"] == 100
    assert receipt["selection"]["selected_video_bytes"] <= 100
    assert receipt["rejected"]["by_reason"]["not_first_person"]["count"] == 1
    assert receipt["rejected"]["by_reason"]["active_Q_E_or_Space"]["count"] == 1
    assert receipt["downloads"]["by_kind"]["video"]["materialized_bytes"] == 100

    downloads = [event[1] for event in client.events if event[0] == "download"]
    first_video = next(index for index, path in enumerate(downloads) if str(path).endswith("video.mp4"))
    assert all(str(path).endswith("annotations.tar") for path in downloads[:first_video])
    assert not any("bb0002/video.mp4" in str(path) or "cc0003/video.mp4" in str(path) for path in downloads)
    selected = [
        json.loads(line)
        for line in (tmp_path / "state/selected_metadata.jsonl").read_text().splitlines()
    ]
    assert [item["sample_id"] for item in selected] == ["aa0001", "dd0004"]


def test_download_resumes_private_part_and_checks_hash(tmp_path: Path) -> None:
    payload = b"0123456789"
    remote = RemoteFile(
        path="data/aa/aa0001/video.mp4",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    destination = tmp_path / "video.mp4"
    part = tmp_path / "video.mp4.part"
    part.write_bytes(payload[:4])
    (tmp_path / "video.mp4.part.meta.json").write_text(
        json.dumps({"path": remote.path, "size": remote.size, "sha256": remote.sha256})
        + "\n",
        encoding="utf-8",
    )
    client = _FakeClient({remote.path: payload})

    result = download_remote_file(
        client,
        remote,
        destination,
        min_free_bytes=0,
        disk_usage=lambda _: SimpleNamespace(free=1_000_000),
    )
    assert result.status == "resumed"
    assert result.transferred_bytes == 6
    assert destination.read_bytes() == payload
    assert not part.exists()


def test_bad_video_is_rejected_and_later_candidate_replenishes_cap(tmp_path: Path) -> None:
    samples = {
        "aa0001": (_annotation(), b"a" * 60),
        "bb0002": (_annotation(), b"b" * 40),
        "cc0003": (_annotation(), b"c" * 60),
    }
    metadata = tmp_path / "metadata.jsonl"
    _write_metadata(metadata, list(samples))
    client = _FakeClient(_fake_payloads(samples))

    def validate(video: Path, _: Path) -> dict:
        if video.read_bytes().startswith(b"a"):
            raise ValueError("synthetic invalid video")
        return {"validated": True}

    receipt = execute_subset(
        metadata,
        payload_root=tmp_path / "payload",
        state_dir=tmp_path / "state",
        config=SubsetConfig(
            video_cap_bytes=100,
            min_free_bytes=0,
            paths_info_batch_size=4,
            target_fill_ratio=1.0,
        ),
        client=client,
        disk_usage=lambda _: SimpleNamespace(free=1_000_000_000),
        video_validator=validate,
    )

    assert receipt["selection"]["selected_video_bytes"] == 100
    assert receipt["rejected"]["by_reason"]["invalid_video"]["count"] == 1
    selected = [
        json.loads(line)["sample_id"]
        for line in (tmp_path / "state/selected_metadata.jsonl").read_text().splitlines()
    ]
    assert selected == ["bb0002", "cc0003"]


def test_existing_wrong_hash_is_never_overwritten(tmp_path: Path) -> None:
    destination = tmp_path / "video.mp4"
    destination.write_bytes(b"wrong")
    expected = b"right"
    remote = RemoteFile(
        path="data/aa/aa0001/video.mp4",
        size=len(expected),
        sha256=hashlib.sha256(expected).hexdigest(),
    )
    with pytest.raises(DownloadIntegrityError, match="hash mismatch"):
        download_remote_file(
            _FakeClient({remote.path: expected}),
            remote,
            destination,
            min_free_bytes=0,
            disk_usage=lambda _: SimpleNamespace(free=1_000_000),
        )
    assert destination.read_bytes() == b"wrong"


def test_download_refuses_to_cross_free_space_reserve(tmp_path: Path) -> None:
    payload = b"0123456789"
    remote = RemoteFile(
        path="data/aa/aa0001/video.mp4",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    client = _FakeClient({remote.path: payload})
    with pytest.raises(DiskReserveError, match="write blocked"):
        download_remote_file(
            client,
            remote,
            tmp_path / "video.mp4",
            min_free_bytes=5,
            disk_usage=lambda _: SimpleNamespace(free=14),
        )
    assert not any(event[0] == "download" for event in client.events)
    assert not (tmp_path / "video.mp4").exists()
