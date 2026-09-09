#!/usr/bin/env python3
"""Generate portable, unexecuted configs for the three-stage training recipe.

This command does not download data, create input footage, authorize GPU use,
or start training.  The eval scenes are placeholders: prepare real held-out
assets before using the generated eval configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


PINNED_MODEL_DIR = "Wan2.2-TI2V-5B@921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
TEMPLATES = {
    "action": "configs/train/action_teacher_5090_week.yaml",
    "causal": "configs/train/causal_teacher_forcing_5090_week.yaml",
    "longforcing": "configs/train/longforcing_lite_5090_week.yaml",
    "eval": "configs/eval/rollout15s_v1.yaml",
}


def repository_root() -> Path:
    """Work both under public_repro/scripts and exported repository/scripts."""
    for candidate in Path(__file__).resolve().parents:
        if all((candidate / name).is_file() for name in TEMPLATES.values()):
            return candidate
    raise FileNotFoundError("cannot find repository training/eval templates")


def _data_root(value: str | Path) -> Path:
    text = str(value)
    path = Path(value)
    if text.startswith("~") or not path.is_absolute():
        raise ValueError("--data-root must be an explicit absolute path (no ~ or relative paths)")
    resolved = path.resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError("--data-root must not be a filesystem root")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError("--data-root must be a directory, not an existing file")
    return resolved


def _rebind(value: Any, old_roots: set[str], data_root: Path) -> Any:
    if isinstance(value, dict):
        # Digests are recorded from real artifacts by the training entry points.
        return {
            key: None if key.endswith("sha256") else _rebind(item, old_roots, data_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rebind(item, old_roots, data_root) for item in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        for old_root in sorted(old_roots, key=len, reverse=True):
            if normalized == old_root or normalized.startswith(old_root + "/"):
                relative = normalized[len(old_root):].lstrip("/")
                return str(data_root.joinpath(*PurePosixPath(relative).parts))
    return value


def configure(
    data_root: Path,
    output_dir: Path,
    eval_stage: str = "longforcing",
    templates_root: Path | None = None,
) -> dict[str, Path]:
    """Write four new configs; leave model/data directories and artifacts alone.

    ``templates_root`` is the repository root, not its ``configs`` directory.
    The returned values are absolute paths to the generated YAML files.
    """
    root = _data_root(data_root)
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    output = output.resolve()
    if eval_stage not in ("causal", "longforcing"):
        raise ValueError("eval_stage must be causal or longforcing")
    templates = Path(templates_root) if templates_root is not None else repository_root()
    raw = {
        stage: yaml.safe_load((templates / relative).read_text(encoding="utf-8"))
        for stage, relative in TEMPLATES.items()
    }
    old_roots = {
        str(PurePosixPath(raw[stage]["model"]["base_model_path"].replace("\\", "/")).parent.parent)
        for stage in ("action", "causal", "longforcing")
    }
    old_roots.add(str(PurePosixPath(
        raw["eval"]["lineage"]["expected_base_model_path"].replace("\\", "/")
    ).parent.parent))
    configs = {stage: _rebind(config, old_roots, root) for stage, config in raw.items()}
    paths = {stage: output / f"{stage}.yaml" for stage in TEMPLATES}
    model = str(root / "models" / PINNED_MODEL_DIR)
    run_names = {
        "action": "action-teacher",
        "causal": "causal-teacher-forcing",
        "longforcing": "longforcing-lite",
    }
    runs = {stage: root / "runs" / "abot-week-v1" / name for stage, name in run_names.items()}
    checkpoints = {stage: str(run / "checkpoints" / "best.pt") for stage, run in runs.items()}
    for stage in runs:
        configs[stage]["model"]["base_model_path"] = model
        configs[stage]["training"]["output_dir"] = str(runs[stage])
    configs["causal"]["lineage"]["checkpoint_path"] = checkpoints["action"]
    configs["longforcing"]["lineage"].update(
        teacher_checkpoint_path=checkpoints["action"],
        causal_checkpoint_path=checkpoints["causal"],
    )
    artifacts = {
        "dataset_manifest": str(root / "data/manifests/train.jsonl"),
        "feature_index": str(root / "data/features/train.features.jsonl"),
        "feature_receipt": str(root / "data/features/train.features.jsonl.receipt.json"),
        "training_config": str(paths[eval_stage]),
        "teacher_checkpoint": checkpoints["action"],
    }
    if eval_stage == "longforcing":
        artifacts.update(
            causal_checkpoint=checkpoints["causal"],
            long_feature_index=str(root / "data/features/train.long241.features.jsonl"),
            long_feature_receipt=str(root / "data/features/train.long241.features.jsonl.receipt.json"),
        )
    configs["eval"]["lineage"].update(
        checkpoint_path=checkpoints[eval_stage],
        expected_stage=("longforcing_lite_v1" if eval_stage == "longforcing" else "causal_teacher_forcing_v1"),
        expected_base_model_path=model,
        artifact_paths=artifacts,
    )
    configs["eval"]["output_root"] = str(root / "eval/rollout15s")
    for scene in configs["eval"]["scenes"]:
        scene_id = scene["scene_id"]
        if not scene_id or scene_id in (".", "..") or any(c in scene_id for c in "/\\:"):
            raise ValueError(f"unsafe template scene_id: {scene_id!r}")
        scene_root = root / "demo_inputs" / scene_id
        scene["initial_frame_path"] = str(scene_root / "initial.npy")
        scene["reference_frames_path"] = str(scene_root / "late_anchors.npz")
    # Serialize before creating the directory; malformed templates leave no output.
    serialized = {
        stage: yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        for stage, config in configs.items()
    }
    output.mkdir(parents=True, exist_ok=False)
    for stage, path in paths.items():
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized[stage])
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Explicit absolute data/model/run root")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for generated YAML")
    parser.add_argument("--eval-stage", choices=("causal", "longforcing"), default="longforcing")
    args = parser.parse_args(argv)
    try:
        paths = configure(args.data_root, args.output_dir, args.eval_stage)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "configs": {stage: str(path) for stage, path in paths.items()},
        "eval_scenes": "DRAFT ONLY: prepare real held-out assets with prepare_abot_demo_assets.py before evaluation",
        "gpu_launched": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
