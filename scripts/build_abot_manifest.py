#!/usr/bin/env python3
"""Build a verified ABot episode manifest from metadata and local payloads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.data.abot_manifest import (  # noqa: E402
    DEFAULT_MAX_STORAGE_BYTES,
    ManifestConfig,
    build_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="Official metadata.jsonl")
    parser.add_argument("--payload-root", type=Path, required=True, help="Local dataset root containing data/<prefix>/<id>/")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL manifest")
    parser.add_argument("--max-storage-bytes", type=int, default=DEFAULT_MAX_STORAGE_BYTES)
    parser.add_argument("--required-frames", type=int, choices=(49, 241), default=241)
    parser.add_argument("--split-seed", default="abot-v1")
    parser.add_argument("--unknown-video-reservation-bytes", type=int)
    parser.add_argument("--allow-non-first-person", action="store_true")
    parser.add_argument("--allow-minecraft", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = build_manifest(
        args.metadata,
        args.payload_root,
        args.output,
        config=ManifestConfig(
            max_storage_bytes=args.max_storage_bytes,
            required_frames=args.required_frames,
            split_seed=args.split_seed,
            require_first_person=not args.allow_non_first_person,
            exclude_minecraft=not args.allow_minecraft,
            unknown_video_reservation_bytes=args.unknown_video_reservation_bytes,
        ),
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
