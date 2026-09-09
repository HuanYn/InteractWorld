#!/usr/bin/env python3
"""Generate continuous 15-second trained previews, without a smoke/evaluation suite."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def add_scene_input_display(path, scene, initial, actions):
    """CPU-only presentation copy; raw model output remains the scoring artifact."""
    from training.eval.input_header import annotate_video
    from training.runtime import sha256_file

    display_path = path.with_name(path.stem + ".inputs.mp4")
    metadata = annotate_video(
        path, display_path, initial_frame=initial, prompt=scene.prompt,
        actions=actions, seed=scene.seed, fps=16,
    )
    metadata.update(video=display_path.name, video_sha256=sha256_file(display_path))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, help="Data root (or set INTERACTWORLD_ROOT)")
    parser.add_argument("--scene-count", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--confirmed-gpu-index", type=int)
    parser.add_argument("--confirmed-gpu-uuid")
    parser.add_argument("--confirmed-at-utc", help="Operator availability-check time under standing authorization")
    parser.add_argument("--allocation-profile")
    parser.add_argument("--authorization-record", type=Path)
    args = parser.parse_args()
    if args.launch:
        required = (args.confirmed_gpu_index, args.confirmed_gpu_uuid, args.confirmed_at_utc, args.allocation_profile)
        if any(value is None for value in required):
            raise ValueError("launch requires physical GPU identity and fresh operator availability check")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.confirmed_gpu_uuid

    from training.eval.rollout15s import (
        _run_variant,
        action_script,
        default_image_loader,
        ffmpeg_writer_factory,
        load_rollout_config,
        resolve_adapter_factory,
        verify_checkpoint_lineage,
    )
    from training.gpu_gate import query_dedicated_gpu, validate_confirmation
    from training.runtime import git_revision, sha256_file
    from training.paths import project_root

    config = load_rollout_config(args.config)
    if not args.launch:
        print(json.dumps({"mode": "cpu_plan", "scenes": args.scene_count, "seconds_per_scene": 15,
                          "checkpoint": config.lineage.checkpoint_path, "output": str(args.output)}, indent=2))
        return
    output = args.output.resolve()
    project = project_root(args.project_root).resolve()
    if project not in output.parents:
        raise ValueError("demo outputs must remain under the authorized configured project")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if args.authorization_record is None or not args.authorization_record.is_file():
        raise ValueError("standing authorization record is required")
    object.__setattr__(config.lineage, "checkpoint_sha256", sha256_file(config.lineage.checkpoint_path))
    lineage = verify_checkpoint_lineage(config)
    validate_confirmation(args.confirmed_at_utc)
    snapshot = query_dedicated_gpu(confirmed_index=args.confirmed_gpu_index,
                                   confirmed_uuid=args.confirmed_gpu_uuid, profile=args.allocation_profile)
    output.mkdir(parents=True)
    started = time.monotonic()
    receipt = {"kind": "trained_continuous_preview", "quality_evaluation": "not_run",
               "stage": config.lineage.expected_stage, "lineage": lineage, "git_revision": git_revision(ROOT),
               "config_sha256": sha256_file(args.config), "gpu": snapshot.as_dict(),
               "authorization_record_sha256": sha256_file(args.authorization_record),
               "operator_checked_at_utc": args.confirmed_at_utc,
               "frames": 241, "fps": 16, "future_duration_seconds": 15, "scenes": [], "outcome": "running"}
    status = output / "receipt.json"
    status.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    try:
        adapter = resolve_adapter_factory(config.adapter_factory)(
            checkpoint_path=config.lineage.checkpoint_path,
            checkpoint_sha256=config.lineage.checkpoint_sha256,
            base_model_path=config.lineage.expected_base_model_path, device="cuda")
        pipeline = getattr(adapter, "pipeline", None)
        if pipeline is not None:
            receipt["sampler"] = {
                "solver": getattr(pipeline.args, "streaming_solver", "renoise"),
                "timesteps": pipeline.denoising_step_list.tolist(),
                "context_mode": getattr(adapter, "context_mode", "unspecified"),
                "action_scale": getattr(adapter, "action_scale", None),
            }
        for scene in config.scenes[:args.scene_count]:
            import torch
            # Seed all stochastic components, including legacy adapter paths.
            torch.manual_seed(scene.seed)
            torch.cuda.manual_seed_all(scene.seed)
            # Only the source initial frame conditions generation. Held-out future references are never loaded.
            initial = default_image_loader(scene.initial_frame_path)
            path = output / f"{scene.scene_id}.mp4"
            partial = path.with_suffix(".partial.mp4")
            actions = action_script(scene)
            _run_variant(config=config, scene=scene, variant="trained_preview", actions=actions,
                         initial=initial, adapter=adapter, output_path=partial, writer_factory=ffmpeg_writer_factory)
            partial.replace(path)
            receipt["scenes"].append({"scene_id": scene.scene_id, "seed": scene.seed,
                                      "video": path.name, "video_sha256": sha256_file(path),
                                      "initial_sha256": sha256_file(scene.initial_frame_path)})
            receipt["scenes"][-1]["display_video"] = add_scene_input_display(path, scene, initial, actions)
            print(f"completed 15-second preview: {path}", flush=True)
        receipt["outcome"] = "completed"
        cards = "".join(f'<article><h2>{html.escape(item["scene_id"])}</h2><video controls src="{html.escape(item["display_video"]["video"])}"></video></article>'
                        for item in receipt["scenes"])
        (output / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>ABot-inspired trained preview</title>'
            '<style>body{background:#10151c;color:#eee;font:16px sans-serif;max-width:960px;margin:40px auto}video{width:100%}</style>'
            '<h1>Continuous world exploration</h1><p>Trained model preview · 15 seconds · quality not yet assessed</p>' + cards,
            encoding="utf-8")
    except BaseException as error:
        receipt["outcome"] = "failed"
        receipt["failure_type"] = type(error).__name__
        receipt["error"] = str(error)
        raise
    finally:
        import torch
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
        status.write_text(json.dumps(receipt, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
