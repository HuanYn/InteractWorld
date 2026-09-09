#!/usr/bin/env python3
"""Encode only explicit scene_static captions; reuse immutable video features.

CPU planning is the default. No narrative fallback, VAE, future RGB, action
rewriting, or modification of the original feature shards is performed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cache_abot_features import _atomic_torch_save, _atomic_write, _path_fingerprint
from training.data.action_dataset import sha256_file, validate_feature_cache_binding
from training.data.safe_annotations import read_annotation_bundle
from training.gpu_gate import query_dedicated_gpu, validate_confirmation


def static_caption(caption: object) -> str:
    if not isinstance(caption, dict):
        raise ValueError("caption must contain an explicit scene_static string")
    text = caption.get("scene_static")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("missing nonempty scene_static; narrative fallback is forbidden")
    return text.strip()


def collect_prompts(manifest: Path, index: Path) -> dict:
    validate_feature_cache_binding(index, manifest)
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    selected = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    by_id = {record["episode_id"]: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate manifest episode ID")
    prompts = {}
    for episode in selected:
        identity = episode["episode_id"]
        record = by_id[identity]
        if identity in prompts or record["split"] != episode["split"]:
            raise ValueError("duplicate episode or split mismatch")
        archive = Path(record["annotations_path"])
        digest = sha256_file(archive)
        if digest != episode["source"]["annotations_sha256"]:
            raise ValueError(f"annotation hash differs from video cache: {identity}")
        text = static_caption(read_annotation_bundle(archive).caption)
        prompts[identity] = {
            "split": episode["split"], "prompt": text,
            "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "annotations_sha256": digest,
        }
    if not prompts:
        raise ValueError("empty feature index")
    return prompts


def encode_prompts(prompts: dict, encoder) -> dict:
    import torch
    outputs = {}
    for count, (identity, info) in enumerate(sorted(prompts.items()), 1):
        with torch.inference_mode():
            value = encoder([info["prompt"]])["prompt_embeds"]
        if tuple(value.shape) != (1, 512, 4096) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"invalid T5 output for {identity}: {tuple(value.shape)}")
        outputs[identity] = value[0].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        if count % 25 == 0 or count == len(prompts):
            print(json.dumps({"encoded": count, "total": len(prompts)}), flush=True)
    return outputs


def inside(path: Path, root: Path) -> Path:
    path = path.resolve()
    path.relative_to(root.resolve())
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--confirmed-gpu-index", type=int)
    parser.add_argument("--confirmed-gpu-uuid")
    parser.add_argument("--confirmed-at-utc")
    parser.add_argument("--allocation-profile", default="dedicated_local_single_gpu")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    for field in ("manifest", "feature_index", "output", "base_model"):
        setattr(args, field, inside(getattr(args, field), root))
    receipt_path = Path(str(args.output) + ".receipt.json")
    failure_path = Path(str(args.output) + ".failure.json")
    if any(path.exists() for path in (args.output, receipt_path, failure_path)):
        raise FileExistsError("output/receipt/failure evidence exists; use a new output")
    started = time.monotonic()
    report = {"schema_version": 1, "kind": "scene_static_prompt_cache",
              "prompt_policy": "scene_static_only_v1", "cache_path": str(args.output),
              "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "command": sys.argv, "seed": 42,
              "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "script_sha256": sha256_file(__file__), "video_features_modified": False}
    try:
        prompts = collect_prompts(args.manifest, args.feature_index)
        print(json.dumps({"mode": "launch" if args.launch else "cpu_plan",
                          "episodes": len(prompts), "prompt_policy": report["prompt_policy"],
                          "estimated_tensor_bytes": len(prompts) * 512 * 4096 * 2}), flush=True)
        if not args.launch:
            return 0
        for key in ("TMPDIR", "HF_HOME", "XDG_CACHE_HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
            if not os.environ.get(key):
                raise ValueError(f"missing cache environment: {key}")
            inside(Path(os.environ[key]), root)
        validate_confirmation(args.confirmed_at_utc)
        report["gpu_gate"] = query_dedicated_gpu(
            confirmed_index=args.confirmed_gpu_index, confirmed_uuid=args.confirmed_gpu_uuid,
            profile=args.allocation_profile).as_dict()
        import torch
        from utils.wan_wrapper import WanTextEncoder
        os.environ["CUDA_VISIBLE_DEVICES"] = args.confirmed_gpu_uuid
        torch.manual_seed(42)
        t5 = args.base_model / "models_t5_umt5-xxl-enc-bf16.pth"
        tokenizer = args.base_model / "google/umt5-xxl"
        report["encoder"] = {"kind": "WanTextEncoder", "base_model": str(args.base_model),
                             "t5": _path_fingerprint(t5), "tokenizer": _path_fingerprint(tokenizer)}
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        encoder = WanTextEncoder(tokenizer_path=str(tokenizer), encoder_pth_path=str(t5)).eval().requires_grad_(False).to("cuda", torch.bfloat16)
        values = encode_prompts(prompts, encoder)
        del encoder
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _atomic_torch_save(args.output, {"schema_version": 1, "kind": report["kind"], "prompt_embeds": values})
        report.update(cache_sha256=sha256_file(args.output), episodes=prompts,
                      manifest_sha256=sha256_file(args.manifest), feature_index_sha256=sha256_file(args.feature_index),
                      feature_receipt_sha256=sha256_file(Path(str(args.feature_index) + ".receipt.json")),
                      peak_vram_bytes=torch.cuda.max_memory_allocated(), elapsed_seconds=time.monotonic() - started,
                      finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), outcome="completed")
        _atomic_write(receipt_path, (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        print(json.dumps({"outcome": "completed", "elapsed_seconds": report["elapsed_seconds"],
                          "cache_sha256": report["cache_sha256"], "peak_vram_bytes": report["peak_vram_bytes"]}), flush=True)
        return 0
    except BaseException as exc:
        if args.launch:
            report.update(outcome="failed", failure_type=type(exc).__name__, error=str(exc),
                          traceback=traceback.format_exc(), elapsed_seconds=time.monotonic() - started)
            _atomic_write(failure_path, (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
