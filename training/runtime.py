"""Reproducible runtime utilities shared by training entry points."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def collect_manifest_hashes(paths: dict[str, str | Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"manifest {name!r} does not exist: {resolved}")
        hashes[name] = sha256_file(resolved)
    return hashes


def git_revision(root: str | Path = ".") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@dataclass
class ThroughputTracker:
    started_at: float
    samples: int = 0

    @classmethod
    def start(cls) -> "ThroughputTracker":
        return cls(started_at=time.perf_counter())

    def update(self, samples: int) -> None:
        self.samples += samples

    @property
    def elapsed_seconds(self) -> float:
        return max(time.perf_counter() - self.started_at, 1e-9)

    @property
    def samples_per_second(self) -> float:
        return self.samples / self.elapsed_seconds


def peak_vram_bytes(device: torch.device | None = None) -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated(device))


class CheckpointManager:
    """Save resumable trainable state while retaining best 1 + last N."""

    def __init__(self, output_dir: str | Path, *, keep_last: int = 2) -> None:
        if keep_last < 1:
            raise ValueError("keep_last must be positive")
        self.output_dir = Path(output_dir).resolve()
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        existing = sorted(self.checkpoint_dir.glob("step-*.pt"))
        self._recent: deque[Path] = deque(existing[-keep_last:])
        for stale in existing[:-keep_last]:
            self._safe_unlink(stale)
        self.best_metric = float("inf")
        best_path = self.checkpoint_dir / "best.pt"
        if best_path.is_file():
            best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
            self.best_metric = float(best_payload.get("metrics", {}).get("loss", float("inf")))

    def _safe_unlink(self, path: Path) -> None:
        resolved = path.resolve()
        if self.checkpoint_dir not in resolved.parents:
            raise RuntimeError(f"refusing to remove checkpoint outside {self.checkpoint_dir}: {resolved}")
        resolved.unlink(missing_ok=True)

    def save(self, payload: dict[str, Any], *, step: int, metric: float) -> Path:
        target = self.checkpoint_dir / f"step-{step:07d}.pt"
        temporary = target.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(target)
        self._recent.append(target)
        while len(self._recent) > self.keep_last:
            self._safe_unlink(self._recent.popleft())
        if metric < self.best_metric:
            self.best_metric = metric
            best_tmp = self.checkpoint_dir / "best.tmp"
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(payload, best_tmp)
            best_tmp.replace(best_path)
        return target


def load_checkpoint(
    path: str | Path,
    *,
    expected_manifest_hashes: dict[str, str],
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    actual = checkpoint.get("manifest_hashes")
    if actual != expected_manifest_hashes:
        raise ValueError(
            "resume manifest hashes do not match the current run: "
            f"checkpoint={actual}, current={expected_manifest_hashes}"
        )
    return checkpoint
