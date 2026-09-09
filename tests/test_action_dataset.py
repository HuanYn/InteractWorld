from __future__ import annotations

import io
import importlib.util
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from training.data.action_dataset import (
    FeatureCacheError,
    PrecomputedActionDataset,
    build_action_teacher_dataloader,
    cache_index_path,
    load_scene_static_prompt_cache,
    validate_scene_static_prompt_cache_binding,
)
from training.data.action_schema import ACTION_KEYS, parse_action_document

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "cache_abot_features.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location("abot_cache_features", _SCRIPT_PATH)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT)
NUM_FRAMES = _SCRIPT.NUM_FRAMES
_sampled_actions = _SCRIPT._sampled_actions
cache_manifest = _SCRIPT.cache_manifest
decode_video_window = _SCRIPT.decode_video_window
main = _SCRIPT.main
probe_video = _SCRIPT.probe_video
validate_action_video_alignment = _SCRIPT.validate_action_video_alignment


def _action_document(count: int) -> dict:
    frames = []
    for index in range(count):
        frames.append(
            {
                "frame_id": f"frame_{index:06d}",
                "timestamp": index / 30,
                "keys": {key: int(key == "W" and index % 2 == 0) for key in ACTION_KEYS},
            }
        )
    return {"fps": 30, "total_frames": count, "frames": frames}


def _write_annotations(path: Path, count: int = 120) -> None:
    with tarfile.open(path, "w") as archive:
        for name, value in (
            ("episode/action.json", _action_document(count)),
            ("episode/caption.json", {"caption": "first-person drive on a quiet road"}),
        ):
            data = json.dumps(value).encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


@pytest.fixture
def small_video(tmp_path: Path) -> Path:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe are unavailable")
    path = tmp_path / "fixture.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=48x24:rate=30:duration=4",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


def test_video_decoder_resamples_exact_30_to_16_indices_and_center_crops(small_video: Path) -> None:
    frames, backend = decode_video_window(
        small_video,
        start_seconds=0.0,
        num_frames=49,
        source_fps=30,
        target_fps=16,
        height=16,
        width=32,
        backend="cpu",
    )
    assert frames.shape == (3, 49, 16, 32)
    assert frames.dtype == torch.uint8
    assert backend == "ffmpeg_cpu"


def test_probe_is_strict_and_action_alignment_is_bound(small_video: Path) -> None:
    metadata = probe_video(small_video)
    assert metadata["fps"] == pytest.approx(30.0)
    assert metadata["frames"] == 120
    assert metadata["codec"]
    assert metadata["pixel_format"]
    assert len(metadata["video_sha256"]) == 64
    validate_action_video_alignment(parse_action_document(_action_document(120)), metadata)
    with pytest.raises(ValueError, match="length mismatch"):
        validate_action_video_alignment(parse_action_document(_action_document(100)), metadata)


def test_probe_rejects_non_30fps_before_decode(tmp_path: Path) -> None:
    class Result:
        returncode = 0
        stderr = b""
        stdout = json.dumps(
            {
                "streams": [
                    {
                        "codec_name": "h264",
                        "pix_fmt": "yuv420p",
                        "width": 32,
                        "height": 16,
                        "r_frame_rate": "24/1",
                        "avg_frame_rate": "24/1",
                        "nb_frames": "120",
                        "duration": "5.0",
                    }
                ]
            }
        ).encode()

    video = tmp_path / "not-opened.mp4"
    video.write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="must be 30 fps"):
        probe_video(video, runner=lambda *args, **kwargs: Result())


def test_future_actions_use_round_i_times_30_over_16() -> None:
    sequence = parse_action_document(_action_document(91))
    actions = _sampled_actions(sequence, 0)
    indices = list(sequence.resampled_offsets(49, output_fps=16)[1:])
    assert actions.shape == (48, 8)
    assert actions[:, 0].tolist() == [float(index % 2 == 0) for index in indices]
    assert indices[11] == 23  # 12 * 30 / 16 = 22.5, nearest-half-up rather than banker's round.
    assert indices[-1] == 90


class _FakeEncoder:
    def encode_video(self, rgb_bcfhw: torch.Tensor) -> torch.Tensor:
        assert rgb_bcfhw.shape == (1, 3, 49, 16, 32)
        return torch.zeros(1, 13, 48, 30, 52)

    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        assert prompts == ["first-person drive on a quiet road"]
        return torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)


def test_cache_receipt_and_read_only_loader_are_deterministic(
    tmp_path: Path, small_video: Path
) -> None:
    annotations = tmp_path / "annotations.tar"
    _write_annotations(annotations)
    manifest = tmp_path / "manifests" / "train.jsonl"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "episode_id": "episode-a",
                "split": "train",
                "video_path": str(small_video),
                "annotations_path": str(annotations),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_decode(*args, **kwargs):
        return torch.zeros(3, NUM_FRAMES, 16, 32, dtype=torch.uint8), "fixture_cpu"

    root = tmp_path / "features"
    receipt = cache_manifest(
        manifest,
        root,
        encoder=_FakeEncoder(),
        max_windows_per_episode=1,
        shard_size=1,
        decoder=fake_decode,
        height=16,
        width=32,
    )
    assert receipt["episodes"] == 1
    assert receipt["windows"] == 1
    assert receipt["decode_backends"] == {"fixture_cpu": 1}
    assert len(receipt["source_bindings_sha256"]) == 64

    dataset = PrecomputedActionDataset(cache_index_path(manifest, root), seed=42)
    first = dataset[0]
    second = dataset[0]
    assert first["noisy_latents"].shape == (13, 48, 30, 52)
    assert first["actions"].shape == (48, 8)
    assert first["prompt_embeds"].shape == (3, 8)
    assert torch.equal(first["noisy_latents"], second["noisy_latents"])
    assert torch.all(first["noisy_latents"][0] == 0)
    assert torch.all(first["target_flow"][0] == 0)
    assert first["timesteps"][0] == 0
    assert torch.all(first["timesteps"][1:] > 0)

    loader = build_action_teacher_dataloader(
        config=SimpleNamespace(
            manifest_path=str(manifest),
            precomputed_latents=True,
            precomputed_text_embeddings=True,
            num_workers=0,
        ),
        training=SimpleNamespace(seed=42, micro_batch_size=1),
    )
    batch = next(iter(loader))
    assert batch["noisy_latents"].shape == (1, 13, 48, 30, 52)
    assert batch["actions"].shape == (1, 48, 8)

    manifest.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(FeatureCacheError, match="manifest hash mismatch"):
        PrecomputedActionDataset(cache_index_path(manifest, root), manifest_path=manifest)
    manifest.write_text(
        json.dumps(
            {
                "episode_id": "episode-a",
                "split": "train",
                "video_path": str(small_video),
                "annotations_path": str(annotations),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    index_record = json.loads(cache_index_path(manifest, root).read_text(encoding="utf-8"))
    shard = root / index_record["shards"][0]["path"]
    with shard.open("ab") as stream:
        stream.write(b"tampered")
    reloaded = PrecomputedActionDataset(cache_index_path(manifest, root), seed=42)
    with pytest.raises(FeatureCacheError, match="shard hash mismatch"):
        reloaded[0]


def test_cli_default_is_cpu_plan_and_does_not_query_cuda(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("", encoding="utf-8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(AssertionError()))
    assert main(["--manifest", str(manifest)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "cpu_plan"
    assert report["cuda_queried"] is False
    assert report["weights_loaded"] is False


def test_cli_launch_requires_fresh_gpu_authorization(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="fresh GPU authorization"):
        main(["--manifest", str(manifest), "--launch"])


def test_cache_corrected_seek_short_window_and_verified_resume(tmp_path: Path, monkeypatch) -> None:
    rounded = _action_document(1800)
    rounded["frames"][1709]["timestamp"] = 56.9667
    sequence = parse_action_document(rounded)
    assert _SCRIPT._window_start_seconds(sequence, 1709) == 56.966666
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture")
    annotations = tmp_path / "annotations.tar"
    _write_annotations(annotations, count=120)
    source_probe = {"frames": 120, "fps": 30, "video_sha256": _SCRIPT.sha256_file(video)}
    monkeypatch.setattr(_SCRIPT, "probe_video", lambda path: source_probe)
    manifest = tmp_path / "manifests" / "train.jsonl"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({
        "episode_id": "episode-a", "split": "dev", "video_path": str(video),
        "annotations_path": str(annotations),
    }) + "\n")
    sequence = parse_action_document(_action_document(120))
    starts = _SCRIPT._selected_starts(sequence, "episode-a", 8, 42)
    short_start = starts[3]
    encoded: list[bool] = []
    forbid_encoding = False

    class Encoder:
        provenance = {"type": "fixture_encoder"}

        def encode_text(self, prompts):
            assert not forbid_encoding
            return torch.zeros(1, 3, 8)

        def encode_video(self, pixels):
            assert not forbid_encoding
            encoded.append(True)
            return torch.zeros(1, 13, 48, 30, 52)

    def decode(path, **kwargs):
        assert not forbid_encoding
        if kwargs["start_seconds"] == _SCRIPT._window_start_seconds(sequence, short_start):
            raise _SCRIPT.IncompleteVideoWindowError("90 of 91", expected_frames=91, actual_frames=90)
        return torch.zeros(3, 49, 2, 2, dtype=torch.uint8), "fixture_cpu"

    cache = tmp_path / "features"
    first = cache_manifest(manifest, cache, encoder=Encoder(), decoder=decode, reuse_completed=True)
    assert first["windows"] == 7 and len(encoded) == 7
    assert first["excluded_window_count"] == 1
    assert first["excluded_windows"][0]["source_start"] == short_start
    episode_path = cache / "episodes" / "episode-a" / "receipt.json"
    episode = json.loads(episode_path.read_text())
    assert episode["split"] == "dev" and episode["num_windows"] == 7
    actual_starts = []
    for shard in episode["shards"]:
        payload = torch.load(cache / shard["path"], weights_only=True)
        actual_starts.extend(payload["source_starts"].tolist())
    assert actual_starts == [start for start in starts if start != short_start]
    forbid_encoding = True
    reused = cache_manifest(manifest, cache, encoder=Encoder(), decoder=decode, reuse_completed=True)
    assert reused["reused_episode_count"] == 1 and reused["windows"] == 7
    assert reused["excluded_window_count"] == 1
    episode.pop("cache_binding")
    episode_path.write_text(json.dumps(episode))
    forbid_encoding = False
    rebuilt = cache_manifest(manifest, cache, encoder=Encoder(), decoder=decode, reuse_completed=True)
    assert rebuilt["reused_episode_count"] == 0 and rebuilt["recomputed_legacy_episode_count"] == 1
    assert len(encoded) == 14

    def failure(*args, **kwargs):
        raise RuntimeError("unrelated codec failure")

    with pytest.raises(RuntimeError, match="unrelated codec failure"):
        cache_manifest(manifest, cache, encoder=Encoder(), decoder=failure)

    monkeypatch.setattr(_SCRIPT.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=bytes(90 * 2 * 2 * 3), stderr=b"",
    ))
    with pytest.raises(_SCRIPT.IncompleteVideoWindowError):
        decode_video_window(video, start_seconds=0, height=2, width=2)
    monkeypatch.setattr(_SCRIPT.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout=bytes(90 * 2 * 2 * 3), stderr=b"codec error",
    ))
    with pytest.raises(RuntimeError, match="video decode failed") as error:
        decode_video_window(video, start_seconds=0, height=2, width=2)
    assert not isinstance(error.value, _SCRIPT.IncompleteVideoWindowError)


def _static_prompt_fixture(tmp_path):
    root = tmp_path / "features"
    root.mkdir()
    manifest = tmp_path / "manifests" / "manifest.jsonl"
    manifest.parent.mkdir()
    records = [{"episode_id": "train-a", "split": "train"},
               {"episode_id": "train-b", "split": "train"},
               {"episode_id": "dev-c", "split": "dev"}]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    shard = root / "unchanged.pt"
    torch.save({"schema_version": 1, "clean_latents": torch.zeros(1, 13, 48, 30, 52, dtype=torch.bfloat16),
                "actions": torch.zeros(1, 48, 8), "prompt_embeds": torch.ones(3, 8)}, shard)
    index = root / "manifest.features.jsonl"
    index_rows = []
    for row in records:
        episode_receipt = root / (row["episode_id"] + ".json")
        episode_receipt.write_text(json.dumps(row), encoding="utf-8")
        index_rows.append({**row, "num_windows": 1,
                           "shards": [{"path": shard.name, "sha256": _SCRIPT.sha256_file(shard), "samples": 1}],
                           "episode_receipt": {"path": episode_receipt.name, "sha256": _SCRIPT.sha256_file(episode_receipt)}})
    index.write_text("".join(json.dumps(row) + "\n" for row in index_rows), encoding="utf-8")
    feature_receipt = index.with_suffix(index.suffix + ".receipt.json")
    feature_receipt.write_text(json.dumps({"schema_version": 1, "index": str(index), "manifest": str(manifest),
                                          "index_sha256": _SCRIPT.sha256_file(index), "manifest_sha256": _SCRIPT.sha256_file(manifest)}), encoding="utf-8")
    path = root / "static-prompts.pt"
    payload = {"schema_version": 1, "kind": "scene_static_prompt_cache", "prompt_embeds": {
        row["episode_id"]: torch.full((4, 4096), position + 2, dtype=torch.bfloat16)
        for position, row in enumerate(records)}}
    receipt = {"schema_version": 1, "kind": "scene_static_prompt_cache", "prompt_policy": "scene_static_only_v1",
               "cache_path": str(path), "manifest_sha256": _SCRIPT.sha256_file(manifest),
               "feature_index_sha256": _SCRIPT.sha256_file(index), "feature_receipt_sha256": _SCRIPT.sha256_file(feature_receipt),
               "encoder": {"kind": "unit-test-only"}, "episodes": {}}
    for row in records:
        prompt = "A static outdoor scene. " + row["episode_id"]
        receipt["episodes"][row["episode_id"]] = {"split": row["split"], "prompt": prompt,
                                                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    _write_static_prompt_fixture(path, payload, receipt)
    return path, index, manifest, shard, payload, receipt


def _write_static_prompt_fixture(path, payload, receipt):
    torch.save(payload, path)
    receipt["cache_sha256"] = _SCRIPT.sha256_file(path)
    path.with_suffix(path.suffix + ".receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


def test_static_prompt_override_changes_only_prompt_and_never_mutates_cache(tmp_path, monkeypatch):
    path, index, manifest, shard, _, receipt = _static_prompt_fixture(tmp_path)
    original_shard_sha = _SCRIPT.sha256_file(shard)
    original_prompt_sha = _SCRIPT.sha256_file(path)
    baseline = PrecomputedActionDataset(index, manifest_path=manifest)
    override = PrecomputedActionDataset(index, manifest_path=manifest, prompt_cache_path=path)
    assert override.prompt_cache_receipt == receipt
    for sample in range(len(baseline)):
        old, new = baseline[sample], override[sample]
        for key in ("noisy_latents", "target_flow", "timesteps", "actions"):
            assert torch.equal(old[key], new[key])
        assert old["prompt_embeds"].shape == (3, 8)
        assert new["prompt_embeds"].shape == (4, 4096)
        assert torch.all(new["prompt_embeds"] == sample + 2)
        new["prompt_embeds"].fill_(99)
        assert torch.all(override[sample]["prompt_embeds"] == sample + 2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    loader = build_action_teacher_dataloader(
        config=SimpleNamespace(manifest_path=str(manifest), precomputed_latents=True,
                               precomputed_text_embeddings=True, num_workers=0, prompt_cache_path=str(path)),
        training=SimpleNamespace(seed=42, micro_batch_size=1))
    assert torch.all(next(iter(loader))["prompt_embeds"] == 2)
    assert _SCRIPT.sha256_file(shard) == original_shard_sha
    assert _SCRIPT.sha256_file(path) == original_prompt_sha


def test_static_prompt_binding_helper_does_not_load_tensors_or_query_cuda(tmp_path, monkeypatch):
    path, index, manifest, _, _, expected = _static_prompt_fixture(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("binding verification must not load tensors or query CUDA")
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    assert validate_scene_static_prompt_cache_binding(path, index_path=index, manifest_path=manifest) == expected


@pytest.mark.parametrize("field", ["manifest_sha256", "feature_index_sha256", "feature_receipt_sha256",
                                  "cache_sha256", "cache_path", "prompt_policy"])
def test_static_prompt_rejects_stale_provenance_even_when_shard_checks_disabled(tmp_path, field):
    path, index, manifest, _, _, receipt = _static_prompt_fixture(tmp_path)
    receipt[field] = "wrong"
    path.with_suffix(path.suffix + ".receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(FeatureCacheError, match="scene-static prompt"):
        PrecomputedActionDataset(index, manifest_path=manifest, prompt_cache_path=path, verify_hashes=False)


def test_static_prompt_requires_selected_split_coverage_without_fallback(tmp_path):
    path, index, manifest, _, payload, receipt = _static_prompt_fixture(tmp_path)
    payload["prompt_embeds"].pop("train-b")
    receipt["episodes"].pop("train-b")
    _write_static_prompt_fixture(path, payload, receipt)
    with pytest.raises(FeatureCacheError, match="missing required episodes.*train-b"):
        PrecomputedActionDataset(index, manifest_path=manifest, prompt_cache_path=path)
    embeddings, _ = load_scene_static_prompt_cache(path, index_path=index, manifest_path=manifest,
                                                   required_episodes=["dev-c"])
    assert set(embeddings) == {"train-a", "dev-c"}
    assert PrecomputedActionDataset(index, manifest_path=manifest, split="dev", prompt_cache_path=path)[0]["prompt_embeds"].shape == (4, 4096)
    with pytest.raises(FeatureCacheError, match="missing required episodes"):
        load_scene_static_prompt_cache(path, index_path=index, manifest_path=manifest)


@pytest.mark.parametrize("damage", ["unknown_id", "split", "text_hash", "payload_ids"])
def test_static_prompt_rejects_episode_or_text_misbinding(tmp_path, damage):
    path, index, manifest, _, payload, receipt = _static_prompt_fixture(tmp_path)
    if damage == "unknown_id":
        receipt["episodes"]["not-in-source"] = dict(receipt["episodes"]["train-a"])
        payload["prompt_embeds"]["not-in-source"] = payload["prompt_embeds"]["train-a"]
    elif damage == "split":
        receipt["episodes"]["train-a"]["split"] = "dev"
    elif damage == "text_hash":
        receipt["episodes"]["train-a"]["prompt"] = "a different static scene"
    else:
        payload["prompt_embeds"].pop("train-a")
    _write_static_prompt_fixture(path, payload, receipt)
    with pytest.raises(FeatureCacheError, match="scene-static prompt"):
        load_scene_static_prompt_cache(path, index_path=index, manifest_path=manifest)


@pytest.mark.parametrize("tensor", [torch.zeros(4, 8), torch.zeros(0, 4096), torch.zeros(513, 4096),
                                   torch.zeros(4, 4096, dtype=torch.int64), torch.full((4, 4096), float("nan"))])
def test_static_prompt_rejects_invalid_embeddings(tmp_path, tensor):
    path, index, manifest, _, payload, receipt = _static_prompt_fixture(tmp_path)
    payload["prompt_embeds"]["train-a"] = tensor
    _write_static_prompt_fixture(path, payload, receipt)
    with pytest.raises(FeatureCacheError, match="scene-static prompt embedding"):
        load_scene_static_prompt_cache(path, index_path=index, manifest_path=manifest)
