"""Reproducible runtime utilities shared by training entry points."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import threading
import time
from collections import deque
from contextlib import contextmanager
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


class OptimizerStepProfiler:
    """Optional low-frequency measurements of complete optimizer steps.

    Training boundaries synchronize CUDA; driver memory is independently sampled,
    not confused with tensor allocation or claimed to be an exact continuous peak.
    Each invocation writes new session files so strict resumes preserve old data.
    Performance warmup never changes learning rates or discards parameter updates.
    """

    def __init__(
        self, device: torch.device, *, warmup_steps: int = 5,
        output_dir: str | Path | None = None, gpu_uuid: str | None = None,
        driver_interval_seconds: float = 1.0, samples_per_step: int = 8,
        enabled: bool = True,
    ) -> None:
        if warmup_steps < 0 or driver_interval_seconds <= 0 or samples_per_step < 1:
            raise ValueError("invalid optimizer profiling settings")
        self.enabled = enabled
        self.device = torch.device(device)
        self.warmup_steps = warmup_steps
        self.samples_per_step = samples_per_step
        self.output_dir = Path(output_dir) if enabled and output_dir is not None else None
        self.session_id = str(time.time_ns())
        self.started_at = time.perf_counter()
        self.current_phase = "startup"
        self.steps: list[dict[str, Any]] = []
        self.checkpoints: list[dict[str, Any]] = []
        self.phases: list[dict[str, Any]] = []
        self.driver_samples: list[dict[str, Any]] = []
        self.driver_errors: list[str] = []
        self.failure: dict[str, Any] | None = None
        self._step_started: tuple[int, float] | None = None
        self._checkpoint_started: tuple[int, float] | None = None
        self._closed = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.driver_interval_seconds = driver_interval_seconds
        self.gpu_uuid = gpu_uuid
        self.initial_memory = self._memory()
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self._record({"kind": "profile_start", "warmup_steps": warmup_steps,
                      "samples_per_optimizer_step": samples_per_step,
                      "memory": self.initial_memory})
        if self._cuda_active() and isinstance(gpu_uuid, str) and gpu_uuid.startswith("GPU-"):
            self._thread = threading.Thread(target=self._sample_driver, daemon=True)
            self._thread.start()

    def _cuda_active(self) -> bool:
        return self.enabled and self.device.type == "cuda" and torch.cuda.is_initialized()

    def _synchronize(self) -> None:
        if self._cuda_active():
            torch.cuda.synchronize(self.device)

    def _memory(self) -> dict[str, Any]:
        values: dict[str, Any] = {"cuda_active": self._cuda_active()}
        if not self.enabled:
            return values
        if values["cuda_active"]:
            values.update(
                allocated_bytes=int(torch.cuda.memory_allocated(self.device)),
                reserved_bytes=int(torch.cuda.memory_reserved(self.device)),
                cumulative_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(self.device)),
                cumulative_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(self.device)),
            )
        # Linux /proc gives process RAM without an additional dependency or CUDA call.
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith(("VmRSS:", "VmHWM:")):
                    name, number, _unit = line.split()
                    values["host_rss_bytes" if name == "VmRSS:" else "host_peak_rss_bytes"] = int(number) * 1024
        except OSError:
            pass
        return values

    def _record(self, record: dict[str, Any]) -> None:
        if self.output_dir is not None:
            append_jsonl(self.output_dir / f"profile-{self.session_id}.jsonl", {
                "session_id": self.session_id, "timestamp_unix": time.time(), **record,
            })

    def _sample_driver(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", f"--id={self.gpu_uuid}",
                     "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3, check=True,
                )
                used, total, utilization = [int(value.strip()) for value in result.stdout.strip().split(",")]
                sample = {"timestamp_unix": time.time(), "phase": self.current_phase,
                          "used_bytes": used * 1024**2, "total_bytes": total * 1024**2,
                          "gpu_utilization_percent": utilization}
                self.driver_samples.append(sample)
                self._record({"kind": "driver_sample", **sample})
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                message = f"{type(exc).__name__}: {exc}"
                self.driver_errors.append(message)
                self._record({"kind": "driver_sampling_error", "error": message})
            self._stop.wait(self.driver_interval_seconds)

    def set_phase(self, name: str) -> None:
        if self.enabled:
            self.current_phase = name

    @contextmanager
    def phase(self, name: str):
        if not self.enabled:
            yield
            return
        previous = self.current_phase
        self.set_phase(name)
        self._synchronize()
        started = time.perf_counter()
        try:
            yield
            self._synchronize()
        except BaseException as exc:
            self._failure(exc)
            raise
        finally:
            record = {"kind": "phase", "name": name,
                      "seconds": time.perf_counter() - started, "memory": self._memory()}
            self.phases.append(record)
            self._record(record)
            self.set_phase(previous)

    def begin_step(self, step: int) -> None:
        if not self.enabled:
            return
        if self._step_started is not None:
            raise RuntimeError("optimizer step measurement already active")
        self._synchronize()
        self.set_phase("data_wait")
        self._step_started = (step, time.perf_counter())

    def end_step(self, step: int) -> dict[str, Any]:
        if not self.enabled:
            return {}
        if self._step_started is None or self._step_started[0] != step:
            raise RuntimeError("optimizer step measurement does not match")
        self._synchronize()
        seconds = time.perf_counter() - self._step_started[1]
        self._step_started = None
        record = {"kind": "optimizer_step", "step": step, "seconds": seconds,
                  "session_ordinal": len(self.steps) + 1,
                  "performance_warmup": len(self.steps) < self.warmup_steps,
                  "samples": self.samples_per_step, "memory": self._memory(),
                  "timing_scope": "data_wait+all_accumulated_forward_backward+optimizer_update; excludes_checkpoint"}
        self.steps.append(record)
        self._record(record)
        self.set_phase("between_steps")
        return record

    def begin_checkpoint(self, step: int) -> None:
        if not self.enabled:
            return
        if self._step_started is not None or self._checkpoint_started is not None:
            raise RuntimeError("checkpoint must be timed separately from optimizer steps")
        self._synchronize()
        self.set_phase("checkpoint_save")
        self._checkpoint_started = (step, time.perf_counter())

    def end_checkpoint(self, step: int, checkpoint_path: str | Path | None = None) -> dict[str, Any]:
        if not self.enabled:
            return {}
        if self._checkpoint_started is None or self._checkpoint_started[0] != step:
            raise RuntimeError("checkpoint measurement does not match")
        self._synchronize()
        seconds = time.perf_counter() - self._checkpoint_started[1]
        self._checkpoint_started = None
        record = {"kind": "checkpoint_save", "step": step, "seconds": seconds,
                  "memory": self._memory()}
        if checkpoint_path is not None:
            path = Path(checkpoint_path)
            record.update(path=str(path), bytes=path.stat().st_size)
        self.checkpoints.append(record)
        self._record(record)
        self.set_phase("between_steps")
        return record

    def _failure(self, exc: BaseException) -> None:
        if not self.enabled:
            return
        if self.failure is None:
            self.failure = {"type": type(exc).__name__, "message": str(exc),
                            "phase": self.current_phase, "memory": self._memory()}
            self._record({"kind": "failure", **self.failure})

    def summary(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "completed_optimizer_steps": None}
        steady = [row["seconds"] for row in self.steps if not row["performance_warmup"]]
        samples = list(self.driver_samples)
        peak = max((row["used_bytes"] for row in samples), default=None)
        return {
            "schema_version": 1, "enabled": True, "session_id": self.session_id,
            "completed_optimizer_steps": len(self.steps), "steady_optimizer_steps": len(steady),
            "performance_warmup_steps": min(len(self.steps), self.warmup_steps),
            "steady_median_seconds": float(np.median(steady)) if steady else None,
            "steady_p90_seconds": float(np.percentile(steady, 90)) if steady else None,
            "checkpoint_seconds_total": sum(row["seconds"] for row in self.checkpoints),
            "checkpoint_seconds_median": float(np.median([row["seconds"] for row in self.checkpoints])) if self.checkpoints else None,
            "initial_memory": self.initial_memory, "final_memory": self._memory(),
            "driver_sampled_peak_bytes": peak,
            "driver_sampled_min_headroom_bytes": min((row["total_bytes"] - row["used_bytes"] for row in samples), default=None),
            "driver_sample_count": len(samples), "driver_sampling_errors": list(self.driver_errors),
            "driver_sampling_interval_requested_seconds": self.driver_interval_seconds,
            "driver_sampling_note": "Periodic whole-GPU samples include desktop; sampled maximum, not a continuous peak. Query latency adds to interval.",
            "phase_memory_note": "PyTorch peak counters are cumulative since caller reset; phase endpoints are not isolated phase peaks.",
            "steps": self.steps, "checkpoints": self.checkpoints, "phases": self.phases,
            "failure": self.failure, "elapsed_seconds": time.perf_counter() - self.started_at,
            "quality_claim": "No quality or convergence claim is supported by resource profiling alone.",
        }

    def close(self, error: BaseException | None = None) -> None:
        if self._closed:
            return
        if error is not None:
            self._failure(error)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=4)
        self._closed = True
        if self.output_dir is not None:
            (self.output_dir / f"profile-{self.session_id}.summary.json").write_text(
                json.dumps(self.summary(), indent=2, sort_keys=True), encoding="utf-8",
            )


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
