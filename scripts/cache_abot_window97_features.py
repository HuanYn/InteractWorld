#!/usr/bin/env python3
"""Cache independent continuous RGB97 windows for the action teacher.

CPU planning is the default. Never concatenate old49/long241 latent caches.
Each selected window is decoded from raw video and independently VAE-encoded;
text comes only from explicit scene_static annotations. Completed episodes may
be reused only after their source, encoder, configuration and shard hashes pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cache_abot_features import (
    HEIGHT, WIDTH, TARGET_FPS, IncompleteVideoWindowError, WanFeatureEncoder,
    _atomic_torch_save, _atomic_write, _jsonl_bytes, _window_start_seconds,
    decode_video_window, probe_video, validate_action_video_alignment,
)
from scripts.cache_scene_static_prompts import static_caption
from training.data.action_dataset import (
    CACHE_SCHEMA_VERSION, FROZEN_DEMO_EPISODE, SCENE_STATIC_PROMPT_POLICY,
    WINDOW97_CACHE_KIND, WINDOW97_ENCODING_POLICY, action_teacher_shapes,
    cache_index_path, sha256_file,
)
from training.data.action_schema import CanonicalActionSequence, parse_action_document
from training.data.safe_annotations import read_annotation_bundle
from training.gpu_gate import query_dedicated_gpu, validate_confirmation

NUM_FRAMES = 97
LATENT_SHAPE, ACTION_SHAPE = action_teacher_shapes(NUM_FRAMES)
POLICIES = {
    "kind": WINDOW97_CACHE_KIND,
    "prompt_policy": SCENE_STATIC_PROMPT_POLICY,
    "encoding_policy": WINDOW97_ENCODING_POLICY,
    "split_policy": "train_only_excluding_frozen_demo_v1",
    "excluded_episode_ids": [FROZEN_DEMO_EPISODE],
}


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    identities = [row.get("episode_id") for row in rows]
    if (not rows or any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value)
                        for value in identities) or len(set(identities)) != len(identities)):
        raise ValueError("manifest must contain unique, path-safe episode IDs")
    return rows


def validated_training_records(manifest: Path, source_manifest: Path) -> list[dict[str, Any]]:
    """Preserve original episode split/path bindings; never relabel dev/test."""
    rows = _manifest_rows(manifest)
    source = {row["episode_id"]: row for row in _manifest_rows(source_manifest)}
    for row in rows:
        identity = row["episode_id"]
        original = source.get(identity)
        if (row.get("split") != "train" or identity == FROZEN_DEMO_EPISODE
                or original is None or original.get("split") != "train"):
            raise ValueError(f"window97 requires original training-only episodes: {identity}")
        for key in ("video_path", "annotations_path"):
            if not isinstance(row.get(key), str) or row.get(key) != original.get(key):
                raise ValueError(f"source manifest {key} mismatch: {identity}")
    return rows


def prepare_training_manifest(source_manifest: Path, output: Path, *, max_episodes: int = 128, seed: int = 42) -> dict:
    """CPU metadata selection only; raw decode/annotation continuity are checked at cache time."""
    if max_episodes <= 0 or ".." in output.parts or ".." in source_manifest.parts:
        raise ValueError("positive max_episodes and paths without traversal required")
    source_manifest, output = source_manifest.resolve(), output.resolve()
    if source_manifest == output:
        raise ValueError("selected manifest must not overwrite its source")
    source = _manifest_rows(source_manifest)
    candidates = [row for row in source if row.get("split") == "train"
                  and row["episode_id"] != FROZEN_DEMO_EPISODE
                  and row.get("source_fps") == 30 and row.get("output_fps") == 16
                  and isinstance(row.get("total_action_frames"), int) and row["total_action_frames"] >= 182]
    candidates.sort(key=lambda row: hashlib.sha256(f"{seed}\0{row['episode_id']}".encode()).hexdigest())
    selected = candidates[:max_episodes]
    if not selected:
        raise ValueError("source metadata contains no training episodes eligible for two97-frame windows")
    data = _jsonl_bytes(selected)
    digest = hashlib.sha256(data).hexdigest()
    receipt = dict(schema_version=1, kind="window97_training_manifest_selection_v1",
                   source_manifest=str(source_manifest), source_manifest_sha256=sha256_file(source_manifest),
                   source_episode_count=len(source), eligible_train_episode_count=len(candidates),
                   selected_episode_count=len(selected), manifest=str(output), manifest_sha256=digest,
                   seed=seed, max_episodes=max_episodes, excluded_episode_ids=[FROZEN_DEMO_EPISODE],
                   eligibility="source_metadata_30fps_at_least182frames_raw_continuity_checked_at_cache_time")
    receipt_path = output.with_suffix(output.suffix + ".selection.json")
    if output.exists() and output.read_bytes() != data:
        raise FileExistsError("selected manifest exists with different content")
    if receipt_path.exists() and json.loads(receipt_path.read_text(encoding="utf-8")) != receipt:
        raise FileExistsError("selected manifest receipt exists with different source/options")
    if not output.exists():
        _atomic_write(output, data)
    if not receipt_path.exists():
        _atomic_write(receipt_path, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
    return receipt


def sampled_actions(sequence: CanonicalActionSequence, start: int) -> torch.Tensor:
    offsets = sequence.resampled_offsets(NUM_FRAMES, output_fps=TARGET_FPS)
    value = torch.tensor([sequence.frames[start + offset].keys for offset in offsets[1:]], dtype=torch.float32)
    if tuple(value.shape) != ACTION_SHAPE:
        raise ValueError(f"window97 actions must be {ACTION_SHAPE}")
    return value


def selected_starts(sequence: CanonicalActionSequence, episode_id: str, count: int, seed: int) -> list[int]:
    candidates = list(sequence.eligible_resampled_window_starts(NUM_FRAMES, output_fps=TARGET_FPS))
    rng = random.Random(int.from_bytes(hashlib.sha256(f"{seed}\0{episode_id}\0window97".encode()).digest()[:8], "big"))
    rng.shuffle(candidates)
    return sorted(candidates[:count])


def _reuse_episode(path: Path, root: Path, expected: Mapping[str, Any], sequence: CanonicalActionSequence) -> dict:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(f"window97 completed episode source/config/encoder mismatch: {path}")
    selected = expected["cache_binding"]["selected_source_starts"]
    excluded = receipt.get("excluded_windows", [])
    excluded_starts = [item.get("source_start") for item in excluded]
    if (len(set(excluded_starts)) != len(excluded_starts)
            or any(start not in selected for start in excluded_starts)
            or any(item.get("reason") != "incomplete_video_window" for item in excluded)):
        raise ValueError("window97 completed exclusions mismatch")
    wanted = [start for start in selected if start not in excluded_starts]
    cached = []
    for shard in receipt["shards"]:
        shard_path = (root / shard["path"]).resolve()
        shard_path.relative_to(path.parent.resolve())
        if sha256_file(shard_path) != shard["sha256"]:
            raise ValueError(f"window97 completed shard hash mismatch: {shard_path}")
        value = torch.load(shard_path, map_location="cpu", weights_only=True)
        count = shard["samples"]
        if (not 0 < count <= expected["cache_binding"]["config"]["shard_size"]
                or any(value.get(key) != expected[key] for key in ("kind", "episode_id", "prompt_policy", "encoding_policy"))
                or tuple(value["clean_latents"].shape) != (count, *LATENT_SHAPE)
                or tuple(value["actions"].shape) != (count, *ACTION_SHAPE)
                or tuple(value["source_starts"].shape) != (count,)):
            raise ValueError("window97 completed shard contract mismatch")
        starts = value["source_starts"].tolist()
        if any(start not in wanted for start in starts):
            raise ValueError("window97 completed shard selection mismatch")
        if not torch.equal(value["actions"].float(), torch.stack([sampled_actions(sequence, start) for start in starts])):
            raise ValueError("window97 completed actions differ from annotations")
        cached.extend(starts)
    if cached != wanted or receipt.get("num_windows") != len(wanted):
        raise ValueError("window97 completed window count/order mismatch")
    return receipt


def cache_window97_manifest(
    manifest_path: str | Path, cache_root: str | Path, *, encoder: Any,
    source_manifest_path: str | Path | None = None,
    max_windows_per_episode: int = 2, max_total_windows: int = 256,
    shard_size: int = 2, seed: int = 42, decoder=decode_video_window,
    reuse_completed: bool = False, execution_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if min(max_windows_per_episode, max_total_windows, shard_size) <= 0 or max_total_windows > 256:
        raise ValueError("positive window/shard counts required; first cache is capped at256 windows")
    started = time.monotonic()
    manifest, root = Path(manifest_path).resolve(), Path(cache_root).resolve()
    source_manifest = Path(source_manifest_path or manifest).resolve()
    records = validated_training_records(manifest, source_manifest)
    records.sort(key=lambda row: hashlib.sha256(f"{seed}\0{row['episode_id']}".encode()).hexdigest())
    index = cache_index_path(manifest, root)
    config = dict(num_frames=97, target_fps=16, height=HEIGHT, width=WIDTH,
                  max_windows_per_episode=max_windows_per_episode, max_total_windows=max_total_windows,
                  shard_size=shard_size, seed=seed,
                  resize="aspect-ratio-preserving Lanczos then center crop",
                  seek_policy="annotation_time_rounded_to_frame_then_floor_microseconds_v1")
    provenance = getattr(encoder, "provenance", {"type": type(encoder).__name__})
    run_binding = dict(schema_version=CACHE_SCHEMA_VERSION, **POLICIES, config=config,
                       encoder=provenance, manifest=str(manifest), manifest_sha256=sha256_file(manifest),
                       source_manifest=str(source_manifest), source_manifest_sha256=sha256_file(source_manifest))
    binding_path = index.with_suffix(index.suffix + ".building.json")
    if binding_path.exists():
        if not reuse_completed or json.loads(binding_path.read_text(encoding="utf-8")) != run_binding:
            raise ValueError("window97 output exists with different binding or without --reuse-completed")
    elif index.exists() or index.with_suffix(index.suffix + ".receipt.json").exists():
        raise FileExistsError("refusing to overwrite an existing unbound feature index")
    else:
        _atomic_write(binding_path, (json.dumps(run_binding, indent=2, sort_keys=True) + "\n").encode())
    index_rows, reused, all_excluded = [], [], []
    backend_counts: dict[str, int] = {}
    remaining = max_total_windows
    for record in records:
        if remaining == 0:
            break
        identity = record["episode_id"]
        bundle = read_annotation_bundle(record["annotations_path"])
        sequence = parse_action_document(bundle.action)
        video_probe = probe_video(record["video_path"])
        validate_action_video_alignment(sequence, video_probe)
        starts = selected_starts(sequence, identity, min(max_windows_per_episode, remaining), seed)
        if not starts:
            continue
        remaining -= len(starts)  # Budget attempted windows too; EOF exclusions do not change selection on resume.
        prompt = static_caption(bundle.caption)
        source = dict(video=video_probe, annotations_sha256=sha256_file(record["annotations_path"]))
        binding = dict(config=config, encoder=provenance, selected_source_starts=starts,
                       prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        episode_dir = root / "episodes" / identity
        receipt_path = episode_dir / "window97-receipt.json"
        expected = dict(schema_version=CACHE_SCHEMA_VERSION, **POLICIES, episode_id=identity,
                        split="train", source=source, cache_binding=binding)
        if reuse_completed and receipt_path.is_file():
            episode_receipt = _reuse_episode(receipt_path, root, expected, sequence)
            reused.append(identity)
        else:
            if receipt_path.exists():
                raise FileExistsError(receipt_path)
            prompt_embeds = encoder.encode_text([prompt]).detach().cpu()
            if prompt_embeds.ndim == 3 and prompt_embeds.shape[0] == 1:
                prompt_embeds = prompt_embeds[0]
            if (prompt_embeds.ndim != 2 or not 1 <= prompt_embeds.shape[0] <= 512
                    or prompt_embeds.shape[1] != 4096 or not bool(torch.isfinite(prompt_embeds).all())):
                raise ValueError("window97 text encoder must return finite [L,4096]")
            shards, excluded = [], []
            for offset in range(0, len(starts), shard_size):
                latents, actions, valid_starts, backends = [], [], [], []
                for start in starts[offset:offset + shard_size]:
                    try:
                        rgb, backend = decoder(record["video_path"], start_seconds=_window_start_seconds(sequence, start),
                            num_frames=97, target_fps=16, source_fps=sequence.fps, height=HEIGHT, width=WIDTH, backend="auto")
                    except IncompleteVideoWindowError as exc:
                        excluded.append(dict(source_start=start, reason="incomplete_video_window",
                                             expected_source_frames=exc.expected_frames, decoded_source_frames=exc.actual_frames))
                        continue
                    if tuple(rgb.shape) != (3, 97, HEIGHT, WIDTH):
                        raise ValueError("window97 decoder must return one complete continuous RGB[3,97,480,832] window")
                    pixels = rgb.unsqueeze(0).float().div_(127.5).sub_(1.0)
                    latent = encoder.encode_video(pixels).detach().cpu()
                    if latent.ndim == 5 and latent.shape[0] == 1:
                        latent = latent[0]
                    if tuple(latent.shape) != LATENT_SHAPE or not bool(torch.isfinite(latent).all()):
                        raise ValueError(f"window97 VAE must return finite {LATENT_SHAPE}")
                    latents.append(latent.to(torch.bfloat16))
                    actions.append(sampled_actions(sequence, start))
                    valid_starts.append(start)
                    backends.append(backend)
                    backend_counts[backend] = backend_counts.get(backend, 0) + 1
                    del pixels, rgb
                if not valid_starts:
                    continue
                path = episode_dir / f"window97-shard-{offset // shard_size:05d}.pt"
                _atomic_torch_save(path, dict(schema_version=CACHE_SCHEMA_VERSION, **POLICIES,
                    episode_id=identity, clean_latents=torch.stack(latents), actions=torch.stack(actions),
                    prompt_embeds=prompt_embeds.to(torch.bfloat16), source_starts=torch.tensor(valid_starts), decode_backends=backends))
                shards.append(dict(path=path.relative_to(root).as_posix(), sha256=sha256_file(path), samples=len(valid_starts)))
            episode_receipt = dict(**expected, shards=shards, num_windows=sum(row["samples"] for row in shards), excluded_windows=excluded)
            _atomic_write(receipt_path, (json.dumps(episode_receipt, indent=2, sort_keys=True) + "\n").encode())
        all_excluded.extend(dict(episode_id=identity, **item) for item in episode_receipt["excluded_windows"])
        if episode_receipt["num_windows"]:
            index_rows.append(dict(schema_version=CACHE_SCHEMA_VERSION, **POLICIES, episode_id=identity, split="train",
                source=source, num_windows=episode_receipt["num_windows"], shards=episode_receipt["shards"],
                episode_receipt=dict(path=receipt_path.relative_to(root).as_posix(), sha256=sha256_file(receipt_path))))
        print(json.dumps(dict(episode_id=identity, windows=episode_receipt["num_windows"], reused=identity in reused)), flush=True)
    if not index_rows:
        raise ValueError("no complete97-frame training windows; no trainable cache index published")
    index_rows.sort(key=lambda row: row["episode_id"])
    index_data = _jsonl_bytes(index_rows)
    _atomic_write(index, index_data)
    receipt = dict(**run_binding, index=str(index), index_sha256=hashlib.sha256(index_data).hexdigest(),
                   episodes=len(index_rows), windows=sum(row["num_windows"] for row in index_rows),
                   source_manifest_episode_count=len(_manifest_rows(source_manifest)),
                   selected_manifest_episode_count=len(records),
                   tensor_shard_bytes=sum((root / shard["path"]).stat().st_size for row in index_rows for shard in row["shards"]),
                   tensor_storage_dtype="bfloat16_latents_and_text_float32_binary_actions",
                   decode_backends=backend_counts, reused_episodes=reused, excluded_windows=all_excluded,
                   elapsed_seconds=time.monotonic() - started,
                   execution=dict(execution_metadata or {"mode": "test_or_cpu_injected"}))
    _atomic_write(index.with_suffix(index.suffix + ".receipt.json"), (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--prepare-manifest", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=128)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--max-windows-per-episode", type=int, default=2)
    parser.add_argument("--max-total-windows", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-completed", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--confirmed-gpu-index", type=int)
    parser.add_argument("--confirmed-gpu-uuid")
    parser.add_argument("--confirmed-at-utc")
    parser.add_argument("--allocation-profile")
    from training.paths import pinned_base_model_path
    base = pinned_base_model_path().as_posix()
    parser.add_argument("--vae-path", default=f"{base}/Wan2.2_VAE.pth")
    parser.add_argument("--t5-path", default=f"{base}/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--tokenizer-path", default=f"{base}/google/umt5-xxl")
    args = parser.parse_args(argv)
    if args.prepare_manifest:
        if args.launch or args.source_manifest is None:
            raise ValueError("--prepare-manifest requires --source-manifest and forbids --launch")
        print(json.dumps(prepare_training_manifest(args.source_manifest, args.manifest,
                                                  max_episodes=args.max_episodes, seed=args.seed), indent=2))
        return 0
    root = args.cache_root or args.manifest.resolve().parent.parent / "features"
    if not args.launch:
        if min(args.max_total_windows, args.max_windows_per_episode, args.shard_size) <= 0 or args.max_total_windows > 256:
            raise ValueError("positive window/shard counts required; first cache is capped at256 windows")
        windows_per_episode = min(args.max_windows_per_episode, args.max_total_windows)
        episodes = (args.max_total_windows + windows_per_episode - 1) // windows_per_episode
        full_episodes, last_windows = divmod(args.max_total_windows, windows_per_episode)
        shards = full_episodes * ((windows_per_episode + args.shard_size - 1) // args.shard_size)
        shards += (last_windows + args.shard_size - 1) // args.shard_size
        print(json.dumps(dict(mode="cpu_plan", **POLICIES, manifest=str(args.manifest.resolve()),
            source_manifest=str((args.source_manifest or args.manifest).resolve()), index=str(cache_index_path(args.manifest, root)),
            frames=97, latent_shape=LATENT_SHAPE, action_shape=ACTION_SHAPE, max_total_windows=args.max_total_windows,
            shard_size=args.shard_size, estimated_episodes_for_cap=episodes,
            estimated_latent_bytes=args.max_total_windows * 3_744_000,
            estimated_text_bytes_upper=shards * 512 * 4096 * 2,
            estimated_action_bytes=args.max_total_windows * 96 * 8 * 4,
            estimate_excludes="torch containers, source starts, manifests, receipts, checkpoints, temporary encoding memory",
            cuda_queried=False, weights_loaded=False), indent=2))
        return 0
    if any(value is None for value in (args.confirmed_gpu_index, args.confirmed_gpu_uuid, args.confirmed_at_utc, args.allocation_profile)):
        raise ValueError("--launch requires confirmed GPU index/UUID/time/allocation profile")
    validate_confirmation(args.confirmed_at_utc)
    snapshot = query_dedicated_gpu(confirmed_index=args.confirmed_gpu_index, confirmed_uuid=args.confirmed_gpu_uuid, profile=args.allocation_profile)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("expected exactly one visible CUDA device")
    encoder = WanFeatureEncoder(vae_path=args.vae_path, t5_path=args.t5_path,
                               tokenizer_path=args.tokenizer_path, device=torch.device("cuda", 0))
    receipt = cache_window97_manifest(args.manifest, root, source_manifest_path=args.source_manifest, encoder=encoder,
        max_windows_per_episode=args.max_windows_per_episode, max_total_windows=args.max_total_windows,
        shard_size=args.shard_size, seed=args.seed, reuse_completed=args.reuse_completed,
        execution_metadata=dict(mode="authorized_cuda_cache", prelaunch_gpu_snapshot=snapshot.as_dict(),
                                confirmed_at_utc=args.confirmed_at_utc, script_sha256=sha256_file(__file__)))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
