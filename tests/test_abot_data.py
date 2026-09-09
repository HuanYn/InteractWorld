from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from training.data.abot_manifest import (
    ManifestConfig,
    StorageBudget,
    StorageBudgetError,
    build_manifest,
    deterministic_split,
    first_person_non_minecraft,
)
from training.data.action_schema import ACTION_KEYS, ActionSchemaError, parse_action_document
from training.data.safe_annotations import AnnotationArchiveError, read_annotation_bundle


def _actions(
    frame_ids: list[int],
    *,
    extra_key: str | None = None,
    timestamps: list[float] | None = None,
) -> dict:
    frames = []
    for frame_id in frame_ids:
        keys = {key: key == "W" for key in ACTION_KEYS}
        if extra_key:
            keys[extra_key] = True
        keys.update({"Q": False, "E": False, "Space": False})
        timestamp = timestamps[len(frames)] if timestamps is not None else len(frames) / 30
        frames.append(
            {"frame_id": f"frame_{frame_id:06d}", "timestamp": timestamp, "keys": keys}
        )
    return {"fps": 30.0, "total_frames": len(frames), "frames": frames}


def _write_tar(path: Path, *, action: dict, caption: object, malicious: tarfile.TarInfo | None = None) -> None:
    with tarfile.open(path, "w") as archive:
        for name, value in (("episode/action.json", action), ("episode/caption.json", caption)):
            data = json.dumps(value).encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        if malicious is not None:
            archive.addfile(malicious, io.BytesIO(b"x") if malicious.isreg() else None)


def test_action_schema_rejects_unknown_and_counts_only_contiguous_windows() -> None:
    sequence = parse_action_document(_actions(list(range(30)) + list(range(31, 80))))
    assert sequence.frames[0].keys == (1, 0, 0, 0, 0, 0, 0, 0)
    assert sequence.eligible_window_counts() == {49: 1, 241: 0}
    assert sequence.eligible_window_starts(49) == (30,)
    with pytest.raises(ActionSchemaError, match="unknown action key"):
        parse_action_document(_actions([0], extra_key="SHIFT"))


def test_action_schema_allows_inactive_released_auxiliary_keys_but_rejects_active() -> None:
    sequence = parse_action_document(_actions([1, 2]))
    assert sequence.fps == 30
    document = _actions([1])
    document["frames"][0]["keys"]["Q"] = True
    with pytest.raises(ActionSchemaError, match="unsupported active action key"):
        parse_action_document(document)


def test_window_eligibility_detects_timestamp_gap() -> None:
    timestamps = [index / 30 for index in range(49)]
    timestamps[25:] = [value + 1.0 for value in timestamps[25:]]
    sequence = parse_action_document(_actions(list(range(1, 50)), timestamps=timestamps))
    assert sequence.eligible_window_counts()[49] == 0


def test_30_to_16_fps_windows_use_full_three_and_fifteen_second_spans() -> None:
    sequence = parse_action_document(_actions(list(range(1, 452))))
    assert sequence.resampled_offsets(49)[-1] == 90
    assert sequence.resampled_offsets(241)[-1] == 450
    assert sequence.eligible_resampled_window_counts() == {49: 361, 241: 1}


@pytest.mark.parametrize(
    "unsafe_name",
    ["../action.json", "/absolute/action.json", "C:/absolute/action.json", "dir\\action.json"],
)
def test_annotation_reader_rejects_unsafe_paths(tmp_path: Path, unsafe_name: str) -> None:
    archive_path = tmp_path / "unsafe.tar"
    info = tarfile.TarInfo(unsafe_name)
    info.size = 1
    _write_tar(archive_path, action=_actions([0]), caption="first person", malicious=info)
    with pytest.raises(AnnotationArchiveError):
        read_annotation_bundle(archive_path)


@pytest.mark.parametrize("link_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_annotation_reader_rejects_links(tmp_path: Path, link_type: bytes) -> None:
    archive_path = tmp_path / "link.tar"
    link = tarfile.TarInfo("episode/link")
    link.type = link_type
    link.linkname = "../../outside"
    _write_tar(archive_path, action=_actions([0]), caption="first person", malicious=link)
    with pytest.raises(AnnotationArchiveError, match="link rejected"):
        read_annotation_bundle(archive_path)


def test_scene_filter_is_explicit_and_case_insensitive() -> None:
    assert first_person_non_minecraft({"perspective": "first", "narrative": "driving on a road"})
    assert not first_person_non_minecraft({"perspective": "third", "narrative": "first-person-like camera"})
    assert not first_person_non_minecraft("First-person exploration in Minecraft")
    assert not first_person_non_minecraft("Third-person character walking")
    assert first_person_non_minecraft("Third-person walking", require_first_person=False)


def test_episode_split_is_deterministic() -> None:
    ids = [f"episode-{index}" for index in range(1000)]
    first = [deterministic_split(item, seed="fixed") for item in ids]
    second = [deterministic_split(item, seed="fixed") for item in reversed(ids)]
    assert first == list(reversed(second))
    assert set(first) == {"train", "dev", "test"}


def test_storage_budget_never_overcommits() -> None:
    budget = StorageBudget(10)
    budget.reserve(7)
    with pytest.raises(StorageBudgetError, match="exceeded"):
        budget.reserve(4)
    assert budget.used_bytes == 7


def test_manifest_and_receipt_are_reproducible(tmp_path: Path) -> None:
    sample_id = "ab0123456789"
    payload = tmp_path / "payload"
    episode = payload / "data" / "ab" / sample_id
    episode.mkdir(parents=True)
    (episode / "video.mp4").write_bytes(b"video-bytes")
    _write_tar(
        episode / "annotations.tar",
        action=_actions(list(range(451))),
        caption={"caption": "A first-person POV drive through a city"},
    )
    metadata = tmp_path / "metadata.jsonl"
    record = {
        "sample_id": sample_id,
        "video": {"path": f"hf://dataset/data/ab/{sample_id}/video.mp4", "bytes": None},
        "annotations": f"https://example.invalid/data/ab/{sample_id}/annotations.tar",
    }
    metadata.write_text(json.dumps(record) + "\n", encoding="utf-8")
    output = tmp_path / "manifest.jsonl"

    receipt = build_manifest(metadata, payload, output, config=ManifestConfig(max_storage_bytes=1_000_000))
    line = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["records"] == 1
    assert line["eligible_resampled_windows"] == {"49": 361, "241": 1}
    assert line["source_fps"] == 30
    assert line["output_fps"] == 16
    assert line["video_size_source"] == "local_file"
    assert receipt["manifest_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert output.with_suffix(".jsonl.receipt.json").is_file()


def test_unknown_video_size_is_rejected_without_explicit_reservation(tmp_path: Path) -> None:
    sample_id = "cd0123456789"
    episode = tmp_path / "payload" / "data" / "cd" / sample_id
    episode.mkdir(parents=True)
    _write_tar(
        episode / "annotations.tar",
        action=_actions(list(range(91))),
        caption="first-person POV",
    )
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "video": {"path": "hf://video", "bytes": None},
                "annotations": "https://example.invalid/annotations.tar",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "manifest.jsonl"
    receipt = build_manifest(
        metadata,
        tmp_path / "payload",
        output,
        config=ManifestConfig(required_frames=49, max_storage_bytes=1_000_000),
    )
    assert receipt["records"] == 0
    assert receipt["rejected"] == {"unknown_video_size": 1}
