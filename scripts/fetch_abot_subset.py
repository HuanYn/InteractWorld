#!/usr/bin/env python3
"""Plan or explicitly execute the pinned 80 GB ABot subset acquisition."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.data.abot_remote import (  # noqa: E402
    DEFAULT_METADATA_PATH,
    DEFAULT_ENDPOINT,
    DEFAULT_MIN_FREE_BYTES,
    DEFAULT_PAYLOAD_ROOT,
    DEFAULT_STATE_DIR,
    DEFAULT_VIDEO_CAP_BYTES,
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    HuggingFaceClient,
    SubsetConfig,
    build_subset_plan,
    execute_subset,
)

from training.paths import project_root

EXECUTION_ROOT = project_root()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, help="Data root (or set INTERACTWORLD_ROOT)")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--payload-root", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--repo-id", default=OFFICIAL_REPO_ID)
    parser.add_argument("--revision", default=OFFICIAL_REVISION)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT),
        help="Pinned Hub endpoint (huggingface.co or the approved hf-mirror.com mirror)",
    )
    parser.add_argument("--video-cap-bytes", type=int, default=DEFAULT_VIDEO_CAP_BYTES)
    parser.add_argument("--min-free-bytes", type=int, default=DEFAULT_MIN_FREE_BYTES)
    parser.add_argument("--paths-info-batch-size", type=int, default=500)
    parser.add_argument("--required-output-frames", type=int, choices=(49, 241), default=241)
    parser.add_argument("--target-fill-ratio", type=float, default=0.995)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument(
        "--allow-third-person",
        action="store_true",
        help="Accept official third-person explorer clips while still excluding Minecraft",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable containing an optional Hub token; its value is never logged",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Enable paths-info calls and payload writes (default is local plan-only)",
    )
    args = parser.parse_args(argv)
    args.project_root = project_root(args.project_root)
    args.metadata = args.metadata or args.project_root / "data/index/metadata.jsonl"
    args.payload_root = args.payload_root or args.project_root / "data/ABot-World-Explorer-500h"
    args.state_dir = args.state_dir or args.project_root / "data/index/abot-first-person-80gb"
    return args


def _assert_under_execution_root(path: Path, root: Path | None = None) -> None:
    resolved = path.resolve()
    root = project_root(root if root is not None else EXECUTION_ROOT).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"--execute path must stay under {root}: {resolved}") from exc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = SubsetConfig(
        repo_id=args.repo_id,
        revision=args.revision,
        endpoint=args.endpoint,
        video_cap_bytes=args.video_cap_bytes,
        min_free_bytes=args.min_free_bytes,
        paths_info_batch_size=args.paths_info_batch_size,
        required_output_frames=args.required_output_frames,
        target_fill_ratio=args.target_fill_ratio,
        max_candidates=args.max_candidates,
        require_first_person=not args.allow_third_person,
    )
    if not args.execute:
        result = build_subset_plan(
            args.metadata,
            payload_root=args.payload_root,
            state_dir=args.state_dir,
            config=config,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    for path in (args.metadata, args.payload_root, args.state_dir):
        _assert_under_execution_root(path, args.project_root)
    token = os.environ.get(args.token_env) or None
    client = HuggingFaceClient(
        repo_id=config.repo_id,
        revision=config.revision,
        endpoint=config.endpoint,
        token=token,
    )
    result = execute_subset(
        args.metadata,
        payload_root=args.payload_root,
        state_dir=args.state_dir,
        config=config,
        client=client,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
