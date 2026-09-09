#!/usr/bin/env python3
"""Precompute Wan2.2 VAE/T5 features for an ABot manifest.

The default invocation prints a CPU-only plan.  CUDA is not queried and model
weights are not opened unless both ``--launch`` and ``--confirm-single-5090`` are
present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.data.action_dataset import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    cache_index_path,
    default_cache_root,
    sha256_file,
)
from training.data.action_schema import (  # noqa: E402
    CanonicalActionSequence,
    parse_action_document,
)
from training.data.safe_annotations import read_annotation_bundle  # noqa: E402
from training.gpu_gate import query_dedicated_gpu, validate_confirmation  # noqa: E402

TARGET_FPS = 16
NUM_FRAMES = 49
HEIGHT = 480
WIDTH = 832
LATENT_SHAPE = (13, 48, 30, 52)
SOURCE_FPS = 30


class IncompleteVideoWindowError(RuntimeError):
    """A successful decoder reached EOF before returning the requested frames."""

    def __init__(self, message: str, *, expected_frames: int, actual_frames: int):
        super().__init__(message)
        self.expected_frames = expected_frames
        self.actual_frames = actual_frames


class FeatureEncoder(Protocol):
    def encode_video(self, rgb_bcfhw: torch.Tensor) -> torch.Tensor: ...
    def encode_text(self, prompts: list[str]) -> torch.Tensor: ...


def _path_fingerprint(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if resolved.is_file():
        return {"path": str(resolved), "kind": "file", "sha256": sha256_file(resolved)}
    if resolved.is_dir():
        entries = []
        for child in sorted(item for item in resolved.rglob("*") if item.is_file()):
            entries.append(
                {
                    "path": child.relative_to(resolved).as_posix(),
                    "bytes": child.stat().st_size,
                    "sha256": sha256_file(child),
                }
            )
        digest = hashlib.sha256(
            json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {"path": str(resolved), "kind": "directory", "sha256": digest, "files": len(entries)}
    raise FileNotFoundError(resolved)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        torch.save(dict(payload), temporary)
        with open(temporary, "rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _jsonl_bytes(records: list[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for record in records
    )


def _flatten_caption(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        preferred = value.get("caption") or value.get("narrative") or value.get("description")
        if isinstance(preferred, str) and preferred.strip():
            return preferred.strip()
        return " ".join(filter(None, (_flatten_caption(item) for item in value.values())))
    if isinstance(value, (list, tuple)):
        return " ".join(filter(None, (_flatten_caption(item) for item in value)))
    return ""


def _ffmpeg_command(
    video_path: Path,
    *,
    start_seconds: float,
    num_frames: int,
    decode_fps: int,
    height: int,
    width: int,
    nvdec: bool,
) -> list[str]:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if nvdec:
        command += ["-hwaccel", "cuda"]
    command += ["-ss", f"{start_seconds:.9f}", "-i", str(video_path)]
    command += [
        "-vf",
        (
            f"fps={decode_fps},"
            f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={width}:{height},format=rgb24"
        ),
        "-frames:v",
        str(num_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    return command


def decode_video_window(
    video_path: str | Path,
    *,
    start_seconds: float,
    num_frames: int = NUM_FRAMES,
    target_fps: int = TARGET_FPS,
    source_fps: int = SOURCE_FPS,
    height: int = HEIGHT,
    width: int = WIDTH,
    backend: str = "auto",
) -> tuple[torch.Tensor, str]:
    """Decode directly to memory, preferring NVDEC and explicitly reporting fallback."""
    if backend not in {"auto", "nvdec", "cpu"}:
        raise ValueError("backend must be auto, nvdec, or cpu")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    mapping = CanonicalActionSequence(fps=source_fps, frames=())
    source_indices = list(mapping.resampled_offsets(num_frames, output_fps=target_fps))
    decoded_frames = source_indices[-1] + 1
    expected_bytes = decoded_frames * height * width * 3
    attempts = [True, False] if backend == "auto" else [backend == "nvdec"]
    failures: list[str] = []
    short_decode: int | None = None
    for nvdec in attempts:
        process = subprocess.run(
            _ffmpeg_command(
                Path(video_path),
                start_seconds=start_seconds,
                num_frames=decoded_frames,
                decode_fps=source_fps,
                height=height,
                width=width,
                nvdec=nvdec,
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if process.returncode == 0 and len(process.stdout) == expected_bytes:
            frames = torch.frombuffer(bytearray(process.stdout), dtype=torch.uint8)
            frames = frames.reshape(decoded_frames, height, width, 3)
            selected = frames[source_indices].permute(3, 0, 1, 2).contiguous()
            return selected, "nvdec" if nvdec else "ffmpeg_cpu"
        failures.append(
            f"{'nvdec' if nvdec else 'cpu'}: rc={process.returncode}, "
            f"bytes={len(process.stdout)}/{expected_bytes}, stderr={process.stderr[-400:].decode(errors='replace')}"
        )
        frame_bytes = height * width * 3
        short_decode = (
            len(process.stdout) // frame_bytes
            if process.returncode == 0 and len(process.stdout) < expected_bytes
            and len(process.stdout) % frame_bytes == 0 else None
        )
    if short_decode is not None:
        raise IncompleteVideoWindowError(
            "incomplete video window; " + " | ".join(failures),
            expected_frames=decoded_frames, actual_frames=short_decode,
        )
    raise RuntimeError("video decode failed; " + " | ".join(failures))


def _source_window_size(source_fps: int) -> int:
    mapping = CanonicalActionSequence(fps=source_fps, frames=())
    return mapping.resampled_offsets(NUM_FRAMES, output_fps=TARGET_FPS)[-1] + 1


def _rate(value: object) -> float:
    try:
        result = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid video frame rate {value!r}") from exc
    if result <= 0:
        raise ValueError(f"invalid video frame rate {value!r}")
    return result


def probe_video(
    video_path: str | Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> dict[str, Any]:
    """Return strict, JSON-serializable ffprobe metadata for the first video track."""
    path = Path(video_path)
    process = runner(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode:
        raise ValueError(f"ffprobe failed for {path}: {process.stderr[-400:].decode(errors='replace')}")
    try:
        payload = json.loads(process.stdout)
        stream = payload["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"ffprobe returned no usable video stream for {path}") from exc
    fps = _rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    duration = float(stream.get("duration") or 0.0)
    raw_count = stream.get("nb_frames")
    frame_count_source = "nb_frames"
    try:
        frame_count = int(raw_count)
    except (TypeError, ValueError):
        frame_count = round(duration * fps)
        frame_count_source = "duration_x_fps"
    if not stream.get("codec_name") or not stream.get("pix_fmt") or width <= 0 or height <= 0:
        raise ValueError(f"video stream lacks codec/pixel/dimension metadata: {path}")
    if abs(fps - SOURCE_FPS) > 0.05:
        raise ValueError(f"video must be 30 fps, got {fps:.6f}: {path}")
    if frame_count < _source_window_size(SOURCE_FPS):
        raise ValueError(f"video has only {frame_count} frames; at least 91 are required")
    return {
        "codec": str(stream["codec_name"]),
        "pixel_format": str(stream["pix_fmt"]),
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frame_count,
        "frame_count_source": frame_count_source,
        "duration_seconds": duration,
        "file_bytes": path.stat().st_size,
        "video_sha256": sha256_file(path),
    }


def validate_action_video_alignment(
    sequence: CanonicalActionSequence, probe: Mapping[str, Any], *, tolerance_frames: int = 2
) -> None:
    if sequence.fps != SOURCE_FPS:
        raise ValueError(f"action annotation must be 30 fps, got {sequence.fps}")
    if not sequence.frames:
        raise ValueError("action annotation is empty")
    video_frames = int(probe["frames"])
    action_span = sequence.frames[-1].frame_id - sequence.frames[0].frame_id + 1
    if abs(video_frames - action_span) > tolerance_frames:
        raise ValueError(
            f"action/video length mismatch: video={video_frames}, action_span={action_span}"
        )


def _sampled_actions(sequence: CanonicalActionSequence, start: int) -> torch.Tensor:
    offsets = sequence.resampled_offsets(NUM_FRAMES, output_fps=TARGET_FPS)
    indices = [start + offset for offset in offsets[1:]]
    return torch.tensor([sequence.frames[index].keys for index in indices], dtype=torch.float32)


def _selected_starts(
    sequence: CanonicalActionSequence, episode_id: str, max_windows: int, seed: int
) -> list[int]:
    candidates = list(
        sequence.eligible_resampled_window_starts(NUM_FRAMES, output_fps=TARGET_FPS)
    )
    rng = random.Random(int.from_bytes(hashlib.sha256(f"{seed}\0{episode_id}".encode()).digest()[:8], "big"))
    rng.shuffle(candidates)
    return sorted(candidates[:max_windows])


def _window_start_seconds(sequence: CanonicalActionSequence, start: int) -> float:
    frame = sequence.frames[start]
    # Exported annotation timestamps are rounded (e.g. 56.9667), which can
    # seek past frame 1709 at true PTS 1709/30 and silently skip that frame.
    source_frame = round(frame.timestamp * sequence.fps) if frame.timestamp is not None else frame.frame_id
    # FFmpeg seeks at microsecond precision: floor, never round above the PTS.
    return (source_frame * 1_000_000 // sequence.fps) / 1_000_000


def _reuse_episode(
    receipt_path: Path, *, root: Path, record: Mapping[str, Any],
    source: Mapping[str, Any], binding: Mapping[str, Any],
    sequence: CanonicalActionSequence,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if "cache_binding" not in receipt:
        # Legacy receipts do not prove the corrected seek policy or encoder.
        # They must be recomputed, not relabelled as newly verified features.
        return None
    if (
        receipt.get("schema_version") != CACHE_SCHEMA_VERSION
        or receipt.get("episode_id") != record["episode_id"]
        or receipt.get("source") != source
    ):
        raise ValueError(f"completed episode source binding mismatch: {receipt_path}")
    if receipt["cache_binding"] != binding or receipt.get("split") != record["split"]:
        raise ValueError(f"completed episode encoder/config/split mismatch: {receipt_path}")
    reuse = {"kind": "verified_encoder_config_seek_binding"}
    selected = binding["selected_source_starts"]
    exclusions = receipt.get("excluded_windows", [])
    excluded_starts = [item["source_start"] for item in exclusions]
    if len(set(excluded_starts)) != len(excluded_starts) or any(
        start not in selected for start in excluded_starts
    ) or any(item.get("reason") != "incomplete_video_window" for item in exclusions):
        raise ValueError(f"completed episode has invalid window exclusions: {receipt_path}")
    expected = [start for start in selected if start not in excluded_starts]
    cached_starts: list[int] = []
    config = binding["config"]
    latent_shape = (13, 48, config["height"] // 16, config["width"] // 16)
    for shard in receipt["shards"]:
        path = (root / shard["path"]).resolve()
        path.relative_to(receipt_path.parent)
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(f"completed shard hash mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        count = int(shard["samples"])
        if not 0 < count <= config["shard_size"] or (
            payload.get("schema_version") != CACHE_SCHEMA_VERSION
            or payload.get("episode_id") != record["episode_id"]
            or tuple(payload["clean_latents"].shape) != (count, *latent_shape)
            or tuple(payload["actions"].shape) != (count, 48, 8)
            or tuple(payload["source_starts"].shape) != (count,)
            or payload["prompt_embeds"].ndim != 2
        ):
            raise ValueError(f"completed shard shape/identity mismatch: {path}")
        starts = [int(value) for value in payload["source_starts"].tolist()]
        if any(start not in expected for start in starts):
            raise ValueError(f"completed shard selection mismatch: {path}")
        actual_actions = torch.stack([_sampled_actions(sequence, start) for start in starts])
        if not torch.equal(payload["actions"].float(), actual_actions):
            raise ValueError(f"completed shard actions differ from source annotations: {path}")
        cached_starts.extend(starts)
    if cached_starts != expected or receipt.get("num_windows") != len(expected):
        raise ValueError(f"completed episode deterministic starts mismatch: {receipt_path}")
    return receipt, reuse


def cache_manifest(
    manifest_path: str | Path,
    cache_root: str | Path,
    *,
    encoder: FeatureEncoder,
    max_windows_per_episode: int = 8,
    shard_size: int = 2,
    seed: int = 42,
    decoder: Callable[..., tuple[torch.Tensor, str]] = decode_video_window,
    height: int = HEIGHT,
    width: int = WIDTH,
    execution_metadata: Mapping[str, Any] | None = None,
    reuse_completed: bool = False,
) -> dict[str, Any]:
    if max_windows_per_episode <= 0 or shard_size <= 0:
        raise ValueError("window and shard counts must be positive")
    manifest = Path(manifest_path).resolve()
    root = Path(cache_root).resolve()
    index = cache_index_path(manifest, root)
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    index_records: list[dict[str, Any]] = []
    backend_counts: dict[str, int] = {}
    exclusions: list[dict[str, Any]] = []
    reused_episodes: list[dict[str, Any]] = []
    recomputed_legacy_episodes: list[str] = []
    cache_config = {
        "target_fps": TARGET_FPS, "num_frames": NUM_FRAMES,
        "height": height, "width": width,
        "max_windows_per_episode": max_windows_per_episode,
        "shard_size": shard_size, "seed": seed,
        "resize": "aspect-ratio-preserving Lanczos then center crop",
        "seek_policy": "annotation_time_rounded_to_frame_then_floor_microseconds_v1",
    }
    encoder_provenance = getattr(encoder, "provenance", {"type": type(encoder).__name__})

    for episode_number, record in enumerate(sorted(records, key=lambda item: item["episode_id"]), 1):
        bundle = read_annotation_bundle(record["annotations_path"])
        sequence = parse_action_document(bundle.action)
        video_probe = probe_video(record["video_path"])
        validate_action_video_alignment(sequence, video_probe)
        starts = _selected_starts(sequence, record["episode_id"], max_windows_per_episode, seed)
        if not starts:
            continue
        prompt = _flatten_caption(bundle.caption) or "first-person world exploration"
        source_binding = {
            "video": video_probe,
            "annotations_sha256": sha256_file(record["annotations_path"]),
        }
        binding = {
            "config": cache_config, "encoder": encoder_provenance,
            "selected_source_starts": starts,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }
        episode_dir = root / "episodes" / record["episode_id"]
        episode_receipt_path = episode_dir / "receipt.json"
        if reuse_completed and episode_receipt_path.is_file():
            completed_reuse = _reuse_episode(
                episode_receipt_path, root=root, record=record, source=source_binding,
                binding=binding, sequence=sequence,
            )
            if completed_reuse is not None:
                completed, reused = completed_reuse
                reused_episodes.append({"episode_id": record["episode_id"], **reused})
                exclusions.extend(
                    {"episode_id": record["episode_id"], **item}
                    for item in completed.get("excluded_windows", [])
                )
                if completed["num_windows"]:
                    index_records.append({
                        "schema_version": CACHE_SCHEMA_VERSION,
                        "episode_id": record["episode_id"], "split": record["split"],
                        "num_windows": completed["num_windows"], "shards": completed["shards"],
                        "source": source_binding,
                        "episode_receipt": {
                            "path": episode_receipt_path.relative_to(root).as_posix(),
                            "sha256": sha256_file(episode_receipt_path),
                        },
                        "cache_reuse": reused,
                    })
                print(f"Episode {episode_number}/{len(records)} reused: {record['episode_id']}", flush=True)
                continue
            recomputed_legacy_episodes.append(record["episode_id"])
            print(f"Recomputing legacy episode with corrected seek policy: {record['episode_id']}", flush=True)
        prompt_embeds = encoder.encode_text([prompt]).detach().cpu()
        if prompt_embeds.ndim >= 3 and prompt_embeds.shape[0] == 1:
            prompt_embeds = prompt_embeds[0]
        shard_entries: list[dict[str, Any]] = []
        episode_exclusions: list[dict[str, Any]] = []
        for shard_number in range(0, len(starts), shard_size):
            shard_starts = starts[shard_number : shard_number + shard_size]
            valid_starts: list[int] = []
            latents: list[torch.Tensor] = []
            actions: list[torch.Tensor] = []
            backends: list[str] = []
            for start in shard_starts:
                try:
                    rgb, used_backend = decoder(
                        record["video_path"],
                        start_seconds=_window_start_seconds(sequence, start),
                        num_frames=NUM_FRAMES,
                        target_fps=TARGET_FPS,
                        source_fps=sequence.fps,
                        height=height,
                        width=width,
                        backend="auto",
                    )
                except IncompleteVideoWindowError as exc:
                    exclusion = {
                        "source_start": start,
                        "start_seconds": _window_start_seconds(sequence, start),
                        "reason": "incomplete_video_window",
                        "expected_source_frames": exc.expected_frames,
                        "decoded_source_frames": exc.actual_frames,
                        "detail": str(exc),
                    }
                    episode_exclusions.append(exclusion)
                    exclusions.append({"episode_id": record["episode_id"], **exclusion})
                    print(f"Excluded incomplete window {record['episode_id']} start={start}", flush=True)
                    continue
                backend_counts[used_backend] = backend_counts.get(used_backend, 0) + 1
                pixels = rgb.unsqueeze(0).float().div_(127.5).sub_(1.0)
                latent = encoder.encode_video(pixels).detach().cpu()
                if latent.ndim == 5 and latent.shape[0] == 1:
                    latent = latent[0]
                if (height, width) == (HEIGHT, WIDTH) and tuple(latent.shape) != LATENT_SHAPE:
                    raise ValueError(f"VAE returned {tuple(latent.shape)}, expected {LATENT_SHAPE}")
                latents.append(latent.to(torch.bfloat16))
                actions.append(_sampled_actions(sequence, start))
                valid_starts.append(start)
                backends.append(used_backend)
                del rgb, pixels
            if not valid_starts:
                continue
            shard_path = episode_dir / f"shard-{shard_number // shard_size:05d}.pt"
            _atomic_torch_save(
                shard_path,
                {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "episode_id": record["episode_id"],
                    "clean_latents": torch.stack(latents),
                    "actions": torch.stack(actions),
                    "prompt_embeds": prompt_embeds.to(torch.bfloat16),
                    "source_starts": torch.tensor(valid_starts, dtype=torch.int64),
                    "decode_backends": backends,
                },
            )
            shard_entries.append(
                {
                    "path": shard_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(shard_path),
                    "samples": len(valid_starts),
                }
            )
        num_windows = sum(shard["samples"] for shard in shard_entries)
        episode_receipt = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "episode_id": record["episode_id"],
            "source": source_binding,
            "split": record["split"],
            "num_windows": num_windows,
            "shards": shard_entries,
            "cache_binding": binding,
            "excluded_windows": episode_exclusions,
        }
        _atomic_write(
            episode_receipt_path,
            (json.dumps(episode_receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
        print(
            f"Episode {episode_number}/{len(records)} completed: {record['episode_id']} "
            f"windows={num_windows} excluded={len(episode_exclusions)}",
            flush=True,
        )
        if not num_windows:
            continue
        index_records.append(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "episode_id": record["episode_id"],
                "split": record["split"],
                "num_windows": num_windows,
                "shards": shard_entries,
                "source": source_binding,
                "episode_receipt": {
                    "path": episode_receipt_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(episode_receipt_path),
                },
            }
        )

    index_data = _jsonl_bytes(index_records)
    _atomic_write(index, index_data)
    receipt = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "index": str(index),
        "index_sha256": hashlib.sha256(index_data).hexdigest(),
        "episodes": len(index_records),
        "windows": sum(item["num_windows"] for item in index_records),
        "decode_backends": dict(sorted(backend_counts.items())),
        "encoder": encoder_provenance,
        "reused_episode_count": len(reused_episodes),
        "reused_episodes": reused_episodes,
        "recomputed_legacy_episode_count": len(recomputed_legacy_episodes),
        "recomputed_legacy_episodes": recomputed_legacy_episodes,
        "excluded_window_count": len(exclusions),
        "excluded_windows": exclusions,
        "source_bindings_sha256": hashlib.sha256(
            json.dumps(
                [item["source"] for item in index_records], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "config": cache_config,
        "execution": dict(execution_metadata or {"mode": "test_or_cpu_injected"}),
    }
    _atomic_write(
        index.with_suffix(index.suffix + ".receipt.json"),
        (json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return receipt


class WanFeatureEncoder:
    def __init__(self, *, vae_path: str, t5_path: str, tokenizer_path: str, device: torch.device):
        from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper

        self.device = device
        self.provenance = {
            "type": "Wan2.2-VAE+UMT5",
            "vae": _path_fingerprint(vae_path),
            "t5": _path_fingerprint(t5_path),
            "tokenizer": _path_fingerprint(tokenizer_path),
        }
        self.vae = WanVAEWrapper(pretrained_path=vae_path).to(device=device, dtype=torch.bfloat16)
        self.text = WanTextEncoder(tokenizer_path=tokenizer_path, encoder_pth_path=t5_path).to(
            device=device, dtype=torch.bfloat16
        )

    @torch.inference_mode()
    def encode_video(self, rgb_bcfhw: torch.Tensor) -> torch.Tensor:
        return self.vae.encode_to_latent(rgb_bcfhw.to(self.device, dtype=torch.bfloat16))

    @torch.inference_mode()
    def encode_text(self, prompts: list[str]) -> torch.Tensor:
        return self.text(text_prompts=prompts, device=self.device)["prompt_embeds"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--max-windows-per-episode", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reuse-completed", action="store_true",
        help="reuse verified encoder/config/seek-bound receipts; recompute legacy receipts",
    )
    from training.paths import pinned_base_model_path
    model_root = pinned_base_model_path().as_posix()
    parser.add_argument("--vae-path", default=f"{model_root}/Wan2.2_VAE.pth")
    parser.add_argument("--t5-path", default=f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--tokenizer-path", default=f"{model_root}/google/umt5-xxl")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--confirmed-gpu-index", type=int)
    parser.add_argument("--confirmed-gpu-uuid")
    parser.add_argument("--confirmed-at-utc")
    parser.add_argument("--allocation-profile")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.cache_root or default_cache_root(args.manifest)
    plan = {
        "mode": "cuda_cache" if args.launch else "cpu_plan",
        "manifest": str(args.manifest.resolve()),
        "cache_root": str(root.resolve()),
        "target": {"frames": NUM_FRAMES, "fps": TARGET_FPS, "height": HEIGHT, "width": WIDTH},
        "cuda_queried": False,
        "weights_loaded": False,
        "reuse_completed": args.reuse_completed,
    }
    if not args.launch:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    required = {
        "--confirmed-gpu-index": args.confirmed_gpu_index,
        "--confirmed-gpu-uuid": args.confirmed_gpu_uuid,
        "--confirmed-at-utc": args.confirmed_at_utc,
        "--allocation-profile": args.allocation_profile,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"--launch requires fresh GPU authorization fields: {missing}")
    validate_confirmation(args.confirmed_at_utc)
    snapshot = query_dedicated_gpu(
        confirmed_index=args.confirmed_gpu_index,
        confirmed_uuid=args.confirmed_gpu_uuid,
        profile=args.allocation_profile,
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("expected exactly one visible CUDA device")
    name = torch.cuda.get_device_name(0)
    if "5090" not in name:
        raise SystemExit(f"confirmed profile requires RTX 5090, found {name!r}")
    device = torch.device("cuda", 0)
    encoder = WanFeatureEncoder(
        vae_path=args.vae_path,
        t5_path=args.t5_path,
        tokenizer_path=args.tokenizer_path,
        device=device,
    )
    receipt = cache_manifest(
        args.manifest,
        root,
        encoder=encoder,
        max_windows_per_episode=args.max_windows_per_episode,
        shard_size=args.shard_size,
        seed=args.seed,
        reuse_completed=args.reuse_completed,
        execution_metadata={
            "mode": "authorized_cuda_cache",
            "confirmed_at_utc": args.confirmed_at_utc,
            "allocation_profile": args.allocation_profile,
            "prelaunch_gpu_snapshot": snapshot.as_dict(),
        },
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
