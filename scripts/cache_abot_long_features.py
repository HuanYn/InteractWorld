#!/usr/bin/env python3
"""Cache continuous 241-frame ABot windows for LongForcing-lite.

The default mode is a CPU-only plan.  Actual VAE/T5 work requires ``--launch``
plus the same fresh dedicated-GPU authorization as the other cache job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cache_abot_features import (  # noqa: E402
    HEIGHT,
    IncompleteVideoWindowError,
    SOURCE_FPS,
    TARGET_FPS,
    WIDTH,
    WanFeatureEncoder,
    _atomic_torch_save,
    _atomic_write,
    _flatten_caption,
    _jsonl_bytes,
    _window_start_seconds,
    decode_video_window,
    probe_video,
    validate_action_video_alignment,
)
from training.data.action_dataset import CACHE_SCHEMA_VERSION, sha256_file  # noqa: E402
from training.data.action_schema import CanonicalActionSequence, parse_action_document  # noqa: E402
from training.data.longforcing_dataset import LONG_CACHE_KIND  # noqa: E402
from training.data.safe_annotations import read_annotation_bundle  # noqa: E402
from training.gpu_gate import query_dedicated_gpu, validate_confirmation  # noqa: E402

NUM_FRAMES = 241
LATENT_SHAPE = (61, 48, 30, 52)
ACTION_SHAPE = (240, 8)


def long_cache_index_path(manifest_path: str | Path, cache_root: str | Path) -> Path:
    manifest = Path(manifest_path)
    return Path(cache_root) / f"{manifest.stem}.long241.features.jsonl"


def _sampled_actions(sequence: CanonicalActionSequence, start: int) -> torch.Tensor:
    offsets = sequence.resampled_offsets(NUM_FRAMES, output_fps=TARGET_FPS)
    indices = [start + offset for offset in offsets[1:]]
    actions = torch.tensor(
        [sequence.frames[index].keys for index in indices],
        dtype=torch.float32,
    )
    if tuple(actions.shape) != ACTION_SHAPE:
        raise ValueError(f"sampled actions must be {ACTION_SHAPE}, got {tuple(actions.shape)}")
    return actions


def _selected_starts(
    sequence: CanonicalActionSequence,
    episode_id: str,
    max_windows: int,
    seed: int,
) -> list[int]:
    candidates = list(
        sequence.eligible_resampled_window_starts(NUM_FRAMES, output_fps=TARGET_FPS)
    )
    rng = random.Random(
        int.from_bytes(hashlib.sha256(f"{seed}\0{episode_id}\0long241".encode()).digest()[:8], "big")
    )
    rng.shuffle(candidates)
    return sorted(candidates[:max_windows])


def _reuse_long_episode(
    receipt_path: Path, *, root: Path, record: Mapping[str, Any],
    source: Mapping[str, Any], binding: Mapping[str, Any],
    sequence: CanonicalActionSequence,
) -> dict[str, Any] | None:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if "cache_binding" not in receipt:
        return None
    if (
        receipt.get("schema_version") != CACHE_SCHEMA_VERSION
        or receipt.get("kind") != LONG_CACHE_KIND
        or receipt.get("episode_id") != record["episode_id"]
        or receipt.get("source") != source
        or receipt.get("cache_binding") != binding
        or receipt.get("split") != record["split"]
    ):
        raise ValueError(f"completed long episode source/config/encoder/split mismatch: {receipt_path}")
    selected = binding["selected_source_starts"]
    exclusions = receipt.get("excluded_windows", [])
    excluded = [item["source_start"] for item in exclusions]
    if (len(set(excluded)) != len(excluded)
            or any(start not in selected for start in excluded)
            or any(item.get("reason") != "incomplete_video_window" for item in exclusions)):
        raise ValueError(f"invalid completed long-window exclusions: {receipt_path}")
    expected = [start for start in selected if start not in excluded]
    cached = []
    config = binding["config"]
    latent_shape = (61, 48, config["height"] // 16, config["width"] // 16)
    for shard in receipt["shards"]:
        path = (root / shard["path"]).resolve()
        path.relative_to(receipt_path.parent.resolve())
        if sha256_file(path) != shard["sha256"]:
            raise ValueError(f"completed long shard hash mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        count = int(shard["samples"])
        if (not 0 < count <= config["shard_size"]
                or payload.get("schema_version") != CACHE_SCHEMA_VERSION
                or payload.get("kind") != LONG_CACHE_KIND
                or payload.get("episode_id") != record["episode_id"]
                or tuple(payload["clean_latents"].shape) != (count, *latent_shape)
                or tuple(payload["actions"].shape) != (count, 240, 8)
                or tuple(payload["source_starts"].shape) != (count,)
                or payload["prompt_embeds"].ndim != 2):
            raise ValueError(f"completed long shard shape/identity mismatch: {path}")
        starts = [int(value) for value in payload["source_starts"].tolist()]
        if any(start not in expected for start in starts):
            raise ValueError(f"completed long shard selection mismatch: {path}")
        actions = torch.stack([_sampled_actions(sequence, start) for start in starts])
        if not torch.equal(payload["actions"].float(), actions):
            raise ValueError(f"completed long shard actions differ from annotations: {path}")
        cached.extend(starts)
    if cached != expected or receipt.get("num_windows") != len(expected):
        raise ValueError(f"completed long episode window selection mismatch: {receipt_path}")
    return receipt


def cache_long_manifest(
    manifest_path: str | Path,
    cache_root: str | Path,
    *,
    encoder: Any,
    max_windows_per_episode: int = 2,
    shard_size: int = 1,
    seed: int = 42,
    decoder=decode_video_window,
    execution_metadata: Mapping[str, Any] | None = None,
    reuse_completed: bool = False,
) -> dict[str, Any]:
    if max_windows_per_episode <= 0 or shard_size <= 0:
        raise ValueError("window and shard counts must be positive")
    manifest = Path(manifest_path).resolve()
    root = Path(cache_root).resolve()
    index = long_cache_index_path(manifest, root)
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    index_records = []
    backend_counts: dict[str, int] = {}
    exclusions: list[dict[str, Any]] = []
    reused_episodes: list[str] = []
    recomputed_legacy_episodes: list[str] = []
    cache_config = {
        "target_fps": TARGET_FPS, "num_frames": NUM_FRAMES,
        "height": HEIGHT, "width": WIDTH,
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
        starts = _selected_starts(
            sequence,
            record["episode_id"],
            max_windows_per_episode,
            seed,
        )
        if not starts:
            continue
        prompt = _flatten_caption(bundle.caption) or "first-person world exploration"
        source = {"video": video_probe, "annotations_sha256": sha256_file(record["annotations_path"])}
        binding = {
            "config": cache_config, "encoder": encoder_provenance,
            "selected_source_starts": starts,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }
        episode_dir = root / "episodes" / record["episode_id"]
        episode_receipt_path = episode_dir / "long241-receipt.json"
        if reuse_completed and episode_receipt_path.is_file():
            completed = _reuse_long_episode(
                episode_receipt_path, root=root, record=record,
                source=source, binding=binding, sequence=sequence,
            )
            if completed is not None:
                reused_episodes.append(record["episode_id"])
                exclusions.extend({"episode_id": record["episode_id"], **item}
                                  for item in completed.get("excluded_windows", []))
                if completed["num_windows"]:
                    index_records.append({
                        "schema_version": CACHE_SCHEMA_VERSION, "kind": LONG_CACHE_KIND,
                        "episode_id": record["episode_id"], "split": record["split"],
                        "num_windows": completed["num_windows"], "shards": completed["shards"],
                        "source": source,
                        "episode_receipt": {"path": episode_receipt_path.relative_to(root).as_posix(),
                                            "sha256": sha256_file(episode_receipt_path)},
                    })
                print(f"Episode {episode_number}/{len(records)} reused: {record['episode_id']}", flush=True)
                continue
            recomputed_legacy_episodes.append(record["episode_id"])
        prompt_embeds = encoder.encode_text([prompt]).detach().cpu()
        if prompt_embeds.ndim >= 3 and prompt_embeds.shape[0] == 1:
            prompt_embeds = prompt_embeds[0]
        shards = []
        episode_exclusions = []
        for shard_start in range(0, len(starts), shard_size):
            shard_starts = starts[shard_start : shard_start + shard_size]
            latents, actions, backends = [], [], []
            completed_starts = []
            for start in shard_starts:
                try:
                    rgb, backend = decoder(
                        record["video_path"],
                        start_seconds=_window_start_seconds(sequence, start),
                        num_frames=NUM_FRAMES,
                        target_fps=TARGET_FPS,
                        source_fps=sequence.fps,
                        height=HEIGHT,
                        width=WIDTH,
                        backend="auto",
                    )
                except IncompleteVideoWindowError as exc:
                    excluded = {"source_start": start, "reason": "incomplete_video_window",
                                "expected_source_frames": exc.expected_frames,
                                "decoded_source_frames": exc.actual_frames, "detail": str(exc)}
                    episode_exclusions.append(excluded)
                    exclusions.append({"episode_id": record["episode_id"], **excluded})
                    continue
                pixels = rgb.unsqueeze(0).float().div_(127.5).sub_(1.0)
                latent = encoder.encode_video(pixels).detach().cpu()
                if latent.ndim == 5 and latent.shape[0] == 1:
                    latent = latent[0]
                if tuple(latent.shape) != LATENT_SHAPE:
                    raise ValueError(f"VAE returned {tuple(latent.shape)}, expected {LATENT_SHAPE}")
                latents.append(latent.to(torch.bfloat16))
                actions.append(_sampled_actions(sequence, start))
                backends.append(backend)
                completed_starts.append(start)
                backend_counts[backend] = backend_counts.get(backend, 0) + 1
                del rgb, pixels
            if not completed_starts:
                continue
            shard_path = episode_dir / f"long241-shard-{shard_start // shard_size:05d}.pt"
            _atomic_torch_save(
                shard_path,
                {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "kind": LONG_CACHE_KIND,
                    "episode_id": record["episode_id"],
                    "clean_latents": torch.stack(latents),
                    "actions": torch.stack(actions),
                    "prompt_embeds": prompt_embeds.to(torch.bfloat16),
                    "source_starts": torch.tensor(completed_starts, dtype=torch.int64),
                    "decode_backends": backends,
                },
            )
            shards.append(
                {
                    "path": shard_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(shard_path),
                    "samples": len(completed_starts),
                }
            )
        num_windows = sum(shard["samples"] for shard in shards)
        episode_receipt = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "kind": LONG_CACHE_KIND,
            "episode_id": record["episode_id"],
            "source": source,
            "num_windows": num_windows,
            "shards": shards,
            "split": record["split"],
            "cache_binding": binding,
            "excluded_windows": episode_exclusions,
        }
        _atomic_write(
            episode_receipt_path,
            (
                json.dumps(episode_receipt, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8"),
        )
        print(f"Episode {episode_number}/{len(records)} cached: {record['episode_id']} ({num_windows} windows)", flush=True)
        if not num_windows:
            continue
        index_records.append(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "kind": LONG_CACHE_KIND,
                "episode_id": record["episode_id"],
                "split": record["split"],
                "num_windows": num_windows,
                "shards": shards,
                "source": source,
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
        "kind": LONG_CACHE_KIND,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "index": str(index),
        "index_sha256": hashlib.sha256(index_data).hexdigest(),
        "episodes": len(index_records),
        "windows": sum(record["num_windows"] for record in index_records),
        "decode_backends": dict(sorted(backend_counts.items())),
        "encoder": encoder_provenance,
        "config": cache_config,
        "reused_episode_count": len(reused_episodes),
        "reused_episodes": reused_episodes,
        "recomputed_legacy_episodes": recomputed_legacy_episodes,
        "excluded_window_count": len(exclusions),
        "excluded_windows": exclusions,
        "execution": dict(execution_metadata or {"mode": "test_or_cpu_injected"}),
    }
    _atomic_write(
        index.with_suffix(index.suffix + ".receipt.json"),
        (json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--max-windows-per-episode", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-completed", action="store_true")
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
    cache_root = args.cache_root or args.manifest.resolve().parent.parent / "features"
    plan = {
        "mode": "authorized_cuda_cache" if args.launch else "cpu_plan",
        "manifest": str(args.manifest.resolve()),
        "index": str(long_cache_index_path(args.manifest, cache_root).resolve()),
        "target": {"frames": NUM_FRAMES, "fps": TARGET_FPS, "height": HEIGHT, "width": WIDTH},
        "max_windows_per_episode": args.max_windows_per_episode,
        "reuse_completed": args.reuse_completed,
        "cuda_queried": False,
        "weights_loaded": False,
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
    device = torch.device("cuda", 0)
    encoder = WanFeatureEncoder(
        vae_path=args.vae_path,
        t5_path=args.t5_path,
        tokenizer_path=args.tokenizer_path,
        device=device,
    )
    receipt = cache_long_manifest(
        args.manifest,
        cache_root,
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
