import json

import pytest
import torch

from training.runtime import OptimizerStepProfiler


def test_full_step_warmup_and_checkpoint_are_separate(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("training.runtime.time.perf_counter", lambda: clock[0])
    profiler = OptimizerStepProfiler(torch.device("cpu"), output_dir=tmp_path, warmup_steps=1)
    for step, seconds in [(1, 50), (2, 10), (3, 20)]:
        profiler.begin_step(step)
        clock[0] += seconds
        profiler.end_step(step)
    profiler.begin_checkpoint(3)
    clock[0] += 100
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    profiler.end_checkpoint(3, checkpoint)
    profiler.close()
    summary = json.loads(next(tmp_path.glob("*.summary.json")).read_text())
    assert summary["completed_optimizer_steps"] == 3
    assert summary["steady_optimizer_steps"] == 2
    assert summary["steady_median_seconds"] == 15
    assert summary["steady_p90_seconds"] == 19
    assert summary["checkpoint_seconds_total"] == 100
    assert summary["driver_sampled_peak_bytes"] is None
    assert summary["steps"][0]["samples"] == 8
    assert summary["checkpoints"][0]["bytes"] == 10
    assert summary["final_memory"]["cuda_active"] is False


def test_short_profile_has_no_fabricated_steady_speed():
    profiler = OptimizerStepProfiler(torch.device("cpu"))
    profiler.begin_step(1)
    profiler.end_step(1)
    assert profiler.summary()["steady_median_seconds"] is None
    assert profiler.summary()["steady_p90_seconds"] is None
    profiler.close()


def test_boundaries_reject_overlap_and_wrong_step():
    profiler = OptimizerStepProfiler(torch.device("cpu"))
    profiler.begin_step(1)
    with pytest.raises(RuntimeError):
        profiler.begin_checkpoint(1)
    with pytest.raises(RuntimeError):
        profiler.end_step(2)
    profiler.end_step(1)
    profiler.begin_checkpoint(1)
    with pytest.raises(RuntimeError):
        profiler.end_checkpoint(2)
    profiler.end_checkpoint(1)
    profiler.close()


def test_failure_phase_and_resume_files_retained(tmp_path):
    first = OptimizerStepProfiler(torch.device("cpu"), output_dir=tmp_path)
    with pytest.raises(RuntimeError):
        with first.phase("backward"):
            raise RuntimeError("synthetic oom")
    first.close()
    second = OptimizerStepProfiler(torch.device("cpu"), output_dir=tmp_path)
    second.close()
    summaries = [json.loads(path.read_text()) for path in tmp_path.glob("*.summary.json")]
    assert len(summaries) == 2
    assert first.summary()["failure"]["phase"] == "backward"
    assert second.summary()["failure"] is None


def test_cuda_samples_are_not_inferred_without_initialized_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: pytest.fail("initialized CUDA required"))
    profiler = OptimizerStepProfiler(torch.device("cuda"), gpu_uuid="GPU-test")
    profiler.begin_step(1)
    profiler.end_step(1)
    assert profiler._thread is None
    profiler.close()


def test_disabled_profiler_has_no_sync_or_files(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: pytest.fail("profiling disabled"))
    profiler = OptimizerStepProfiler(torch.device("cuda"), output_dir=tmp_path / "unused",
                                     gpu_uuid="GPU-test", enabled=False)
    with profiler.phase("model_load"):
        pass
    profiler.begin_step(1)
    assert profiler.end_step(1) == {}
    profiler.begin_checkpoint(1)
    assert profiler.end_checkpoint(1) == {}
    profiler.close()
    assert profiler.summary()["enabled"] is False
    assert not (tmp_path / "unused").exists()
