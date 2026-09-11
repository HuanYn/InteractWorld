#!/usr/bin/env python3
"""Plan or launch the fixed three-scene 15-second rollout evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.eval.rollout15s import (  # noqa: E402
    build_plan,
    load_rollout_config,
    resolve_adapter_factory,
    run_rollout_suite,
    verify_checkpoint_lineage,
)
from training.gpu_gate import query_dedicated_gpu, validate_confirmation  # noqa: E402
from training.runtime import sha256_file  # noqa: E402

DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "eval" / "rollout15s_v1.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--launch", action="store_true", help="run inference; default is CPU-only plan")
    parser.add_argument("--output-root")
    parser.add_argument("--run-id")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument(
        "--expected-stage",
        choices=("longforcing_lite_v1", "causal_teacher_forcing_v1", "causal_moba_regularized_v1"),
        help="explicit checkpoint stage; source configuration and lineage must match",
    )
    parser.add_argument("--confirmed-gpu-index", type=int)
    parser.add_argument("--confirmed-gpu-uuid")
    parser.add_argument("--confirmed-at-utc")
    parser.add_argument("--allocation-profile")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_rollout_config(args.config)
    if args.output_root:
        object.__setattr__(config, "output_root", args.output_root)
    if args.run_id:
        object.__setattr__(config, "run_id", args.run_id)
    if args.checkpoint:
        object.__setattr__(config.lineage, "checkpoint_path", args.checkpoint)
    if args.checkpoint_sha256:
        object.__setattr__(config.lineage, "checkpoint_sha256", args.checkpoint_sha256.lower())
    if args.expected_stage:
        object.__setattr__(config.lineage, "expected_stage", args.expected_stage)
    config.validate()
    if not args.launch:
        print(json.dumps(build_plan(config, args.config), indent=2, sort_keys=True))
        return 0

    final = (Path(config.output_root) / config.run_id).resolve()
    if final.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {final}")
    lineage = verify_checkpoint_lineage(config)
    required = (
        args.confirmed_gpu_index,
        args.confirmed_gpu_uuid,
        args.confirmed_at_utc,
        args.allocation_profile,
    )
    if any(value is None for value in required):
        raise ValueError("--launch requires fresh GPU index, UUID, UTC time, and allocation profile")
    validate_confirmation(args.confirmed_at_utc)
    snapshot = query_dedicated_gpu(
        confirmed_index=args.confirmed_gpu_index,
        confirmed_uuid=args.confirmed_gpu_uuid,
        profile=args.allocation_profile,
    )
    factory = resolve_adapter_factory(config.adapter_factory)
    output, receipt = run_rollout_suite(
        config,
        lineage={**lineage, "gpu": snapshot.as_dict()},
        adapter_factory=factory,
        device="cuda",
        config_sha256=sha256_file(args.config),
    )
    print(json.dumps({"output_dir": str(output), "passed": receipt["passed"]}, indent=2))
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
