#!/usr/bin/env python3
"""Prepare three fixed held-out, real-action 15-second scenes on CPU only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from scripts.cache_abot_features import (  # noqa: E402
    _flatten_caption,
    _window_start_seconds,
    decode_video_window,
)
from training.data.action_schema import ACTION_KEYS, parse_action_document  # noqa: E402
from training.data.safe_annotations import read_annotation_bundle  # noqa: E402
from training.eval.rollout15s import load_rollout_config  # noqa: E402
from training.runtime import sha256_file  # noqa: E402
from training.paths import project_root as resolve_project_root  # noqa: E402


def action_segments(rows):
    result = []
    for row in rows:
        keys = [key for key, on in zip(ACTION_KEYS, row, strict=True) if on]
        if result and result[-1]["keys"] == keys:
            result[-1]["frames"] += 1
        else:
            result.append({"frames": 1, "keys": keys})
    return result


def lineage_from_training(manifest: Path, training_config: Path, stage: str) -> dict:
    """Bind to the actual config file and parent paths, never synthesize hashes."""
    if stage == "causal":
        from training.causal_tf import load_causal_config
        config = load_causal_config(training_config)
        teacher_path = config.lineage.checkpoint_path
        expected_stage = "causal_teacher_forcing_v1"
    elif stage == "longforcing":
        from training.longforcing_lite import load_longforcing_config
        config = load_longforcing_config(training_config)
        teacher_path = config.lineage.teacher_checkpoint_path
        expected_stage = "longforcing_lite_v1"
    else:
        raise ValueError("stage must be causal or longforcing")
    if Path(config.data.manifest_path).resolve() != manifest.resolve():
        raise ValueError("demo manifest must be the exact manifest named by the training config")
    artifacts = {
        "dataset_manifest": str(manifest.resolve()),
        "feature_index": config.data.feature_index_path,
        "feature_receipt": config.data.feature_receipt_path,
        "training_config": str(training_config.resolve()),
        "teacher_checkpoint": teacher_path,
    }
    if stage == "longforcing":
        artifacts.update(
            causal_checkpoint=config.lineage.causal_checkpoint_path,
            long_feature_index=config.data.long_feature_index_path,
            long_feature_receipt=config.data.long_feature_receipt_path,
        )
    prompt_path = getattr(config.data, "prompt_cache_path", None)
    if prompt_path is not None:
        artifacts.update(
            prompt_cache=prompt_path,
            prompt_cache_receipt=str(Path(prompt_path).with_suffix(".pt.receipt.json")),
        )
    return {
        "checkpoint_path": str(Path(config.training.output_dir) / "checkpoints/best.pt"),
        "checkpoint_sha256": None,
        "expected_stage": expected_stage,
        "expected_base_model_path": config.model.base_model_path,
        "artifact_paths": artifacts,
    }


def prompt_receipt_from_lineage(lineage: dict) -> dict | None:
    """Reuse the exact text already encoded for training, without loading T5."""
    artifacts = lineage["artifact_paths"]
    path = artifacts.get("prompt_cache")
    if path is None:
        return None
    from training.data.action_dataset import validate_scene_static_prompt_cache_binding
    return validate_scene_static_prompt_cache_binding(
        path, index_path=artifacts["feature_index"], manifest_path=artifacts["dataset_manifest"],
    )


def prompt_for_episode(record: dict, caption: object, receipt: dict | None) -> str:
    if receipt is None:
        return _flatten_caption(caption) or "Third-person world exploration."
    from scripts.cache_scene_static_prompts import static_caption
    identity = record["episode_id"]
    binding = receipt["episodes"].get(identity)
    if not binding or binding["split"] != record["split"]:
        raise ValueError(f"static prompt is missing or has wrong split: {identity}")
    text = static_caption(caption)
    if text != binding["prompt"] or sha256_file(record["annotations_path"]) != binding["annotations_sha256"]:
        raise ValueError(f"demo static caption/annotations differ from the training prompt cache: {identity}")
    return binding["prompt"]


def prepare(manifest: Path, output: Path, template: Path, *, training_config: Path | None = None,
            stage: str = "causal", project_root: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    project = resolve_project_root(
        project_root if project_root is not None else os.environ.get("INTERACTWORLD_ROOT", str(manifest.parent.parent.parent))
    ).resolve()
    if project not in manifest.resolve().parents or project not in output.resolve().parents:
        raise ValueError("demo manifest and assets must remain under the selected data project root")
    if training_config is None:
        name = "causal_teacher_forcing_5090_week.yaml" if stage == "causal" else "longforcing_lite_5090_week.yaml"
        training_config = ROOT / "configs/train" / name
    lineage = lineage_from_training(manifest, training_config, stage)
    prompt_receipt = prompt_receipt_from_lineage(lineage)
    records = sorted(
        (json.loads(line) for line in manifest.read_text().splitlines() if line.strip()),
        key=lambda item: item["episode_id"],
    )
    selected = []
    opposite = ({"W", "S"}, {"A", "D"}, {"I", "K"}, {"J", "L"})
    for record in records:
        if record["split"] != "dev":
            continue
        bundle = read_annotation_bundle(record["annotations_path"])
        sequence = parse_action_document(bundle.action)
        starts = sequence.eligible_resampled_window_starts(241, output_fps=16)
        offsets = sequence.resampled_offsets(241, output_fps=16)
        for candidate in np.linspace(0, len(starts) - 1, min(32, len(starts)), dtype=int):
            start = starts[int(candidate)]
            rows = [sequence.frames[start + offset].keys for offset in offsets[1:]]
            segments = action_segments(rows)
            if sum(any(row) for row in rows) < 60 or len(segments) < 2:
                continue
            if any(pair.issubset(segment["keys"]) for pair in opposite for segment in segments):
                continue
            selected.append((record, sequence, start, segments,
                             prompt_for_episode(record, bundle.caption, prompt_receipt)))
            break
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise ValueError("fewer than three eligible held-out scenes with nontrivial actions")

    output.mkdir(parents=True)
    config = yaml.safe_load(template.read_text())
    config["run_id"] = f"abot-week-{stage}-dev3"
    config["output_root"] = str(project / "eval" / "rollout15s")
    config["lineage"] = lineage
    config["scenes"] = []
    receipt = {"kind": "held_out_source_assets_not_generated_video", "manifest_sha256": sha256_file(manifest),
               "prompt_policy": "scene_static_only_v1" if prompt_receipt else "original_feature_cache_prompt",
               "prompt_artifacts": {name: sha256_file(path) for name, path in lineage["artifact_paths"].items()
                                    if name in ("prompt_cache", "prompt_cache_receipt")}, "scenes": []}
    for index, (record, sequence, start, segments, caption) in enumerate(selected):
        scene_id = f"dev{index + 1}-{record['episode_id']}"
        scene_dir = output / scene_id
        scene_dir.mkdir()
        rgb, backend = decode_video_window(
            record["video_path"], start_seconds=_window_start_seconds(sequence, start),
            num_frames=241, source_fps=sequence.fps, target_fps=16,
            height=480, width=832, backend="auto",
        )
        frames = rgb.permute(1, 2, 3, 0).numpy()
        initial = scene_dir / "initial.npy"
        anchors = scene_dir / "late_anchors.npz"
        np.save(initial, frames[0], allow_pickle=False)
        np.savez_compressed(anchors, frames=frames[[120, 180, 240]], frame_indices=np.array([120, 180, 240]))
        config["scenes"].append({
            "scene_id": scene_id, "source_episode_id": record["episode_id"], "prompt": caption,
            "initial_frame_path": str(initial), "reference_frames_path": str(anchors),
            "seed": 4201 + index, "action_segments": segments,
        })
        receipt["scenes"].append({
            "episode_id": record["episode_id"], "split": "dev", "source_start": start,
            "source_video": record["video_path"], "annotations_sha256": sha256_file(record["annotations_path"]),
            "initial_sha256": sha256_file(initial), "anchors_sha256": sha256_file(anchors),
            "prompt_sha256": hashlib.sha256(caption.encode("utf-8")).hexdigest(),
            "decode_backend": backend, "reference_use": "scoring_only_never_model_condition",
        })
        del frames, rgb
        print(f"prepared {scene_id}", flush=True)
    config_path = output / f"rollout15s_{stage}_week.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    load_rollout_config(config_path).validate()
    receipt["config"] = str(config_path)
    receipt["config_sha256"] = sha256_file(config_path)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=ROOT / "configs/eval/rollout15s_v1.yaml")
    parser.add_argument("--training-config", type=Path, help="Exact causal/LongForcing training YAML used for this checkpoint")
    parser.add_argument("--stage", choices=("causal", "longforcing"), default="causal")
    parser.add_argument("--project-root", type=Path, help="Data root (or set INTERACTWORLD_ROOT)")
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest.resolve(), args.output.resolve(), args.template.resolve(),
                             training_config=args.training_config, stage=args.stage, project_root=args.project_root), indent=2))


if __name__ == "__main__":
    main()
