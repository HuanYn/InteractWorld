"""Teacher-free, paired history-noise training from a pinned causal checkpoint.

The original YAML is immutable historical evidence. A separate JSON stage
contract declares a fresh optimizer and new local step counter, an absolute
dataset offset, and the sole candidate/control augmentation difference.
Without --launch this entry validates configuration on CPU only. CUDA requires
an external authorizer that checks the fresh reservation/GPU/storage grant.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import train_causal_teacher_forcing as tf_entry
from training.causal_tf import construct_causal_wrapper, load_causal_config
from training.history_noise import HistoryNoiseConfig, history_noise_contract, history_noise_loss
from training.models.lora import configure_action_teacher, load_trainable_state_dict, trainable_state_dict
from training.runtime import CheckpointManager, append_jsonl, git_revision, load_checkpoint, seed_everything, sha256_file

STAGE_NAME = "causal_history_noise_v1"
INITIALIZATION = "same_causal60_weights_only_fresh_optimizer_rng_local_step0_absolute_data480"
FACTORY = "training.data.action_resampled:build_resampled_action_teacher_dataloader"
STAGE_KEYS = {"schema_version", "stage", "base_config_sha256", "parent_checkpoint_path",
              "parent_checkpoint_sha256", "parent_expected_stage", "parent_expected_step",
              "data_start_absolute_index", "max_steps", "output_dir", "history_noise"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def source_provenance(authorization):
    """Frozen archives have no .git; only an authorizer may bind their revision."""
    queried = git_revision(Path(__file__).parent)
    bound = authorization.get("source_revision")
    if bound is not None:
        _require(isinstance(bound, str) and len(bound) == 40
                 and all(c in "0123456789abcdef" for c in bound), "invalid authorized source revision")
        _require(queried == "unknown" or queried == bound, "local Git revision differs from authorized source")
    return {"git_revision": bound if bound is not None else queried,
            "git_query_revision": queried,
            "source_revision_origin": "verified_external_authorizer" if bound is not None else "local_git_query"}


def load_stage(config_path, stage_config_path):
    config = load_causal_config(config_path)
    stage = json.loads(Path(stage_config_path).read_text(encoding="utf-8"))
    _require(isinstance(stage, dict) and set(stage) == STAGE_KEYS, "stage JSON has unexpected keys")
    _require(stage["schema_version"] == 1 and stage["stage"] == STAGE_NAME, "history-noise stage/version mismatch")
    _require(_digest(stage["base_config_sha256"]) and sha256_file(config_path) == stage["base_config_sha256"],
             "original base YAML SHA differs")
    _require(_digest(stage["parent_checkpoint_sha256"]), "parent checkpoint requires exact SHA")
    _require(stage["parent_expected_stage"] == "causal_teacher_forcing_v1"
             and type(stage["parent_expected_step"]) is int and stage["parent_expected_step"] == 60,
             "new history-noise stage requires actual clean causal60 parent")
    _require(type(stage["data_start_absolute_index"]) is int and stage["data_start_absolute_index"] == 480,
             "both paired branches must continue absolute data at index480")
    _require(type(stage["max_steps"]) is int and 20 <= stage["max_steps"] <= 200,
             "new stage configured horizon must be20..200 local optimizer steps")
    _require(bool(stage["output_dir"]) and bool(stage["parent_checkpoint_path"]), "explicit parent/output paths required")
    _require(config.data.num_frames == 97 and config.data.data_factory == FACTORY
             and config.data.prompt_cache_path is None and not config.error_recycling.enabled,
             "base must be unchanged native97 resampled clean-history config")
    _require(config.training.checkpoint_every == 20 and config.training.micro_batch_size == 1
             and config.training.gradient_accumulation_steps == 8, "paired checkpoint/batch contract differs")
    HistoryNoiseConfig.from_dict(stage["history_noise"]).validate()
    config.training.max_steps = stage["max_steps"]
    config.training.output_dir = stage["output_dir"]
    config.validate()
    return config, stage


def method_contract(stage):
    return {
        "method": STAGE_NAME, "history_noise": HistoryNoiseConfig.from_dict(stage["history_noise"]).to_dict(),
        "history_noise_contract": history_noise_contract(HistoryNoiseConfig.from_dict(stage["history_noise"])),
        "loss": "MSE(predicted_flow[:,1:],original_GT_flow[:,1:])",
        "context": "only_clean_x_context_and_its_aug_t_are_augmented",
        "student_visibility": "completed_clean_history_and_current_noisy_block",
        "initial_latent_preserved": True, "future_GT_as_inference_condition": False,
        "noisy_target_actions_prompt_unchanged": True,
        "noise_rng": "stateless_history_namespace_seed_plus_absolute_microbatch_index",
        "teacher_model_loaded": False, "teacher_labels": False, "DMD": False,
        "full_self_rollout": False, "quality_validated": False,
        "initialization": INITIALIZATION,
    }


def artifact_hashes(config, stage, config_path, stage_config_path):
    hashes = {
        "dataset_manifest": sha256_file(config.data.manifest_path),
        "feature_index": sha256_file(config.data.feature_index_path),
        "feature_receipt": sha256_file(config.data.feature_receipt_path),
        "base_training_config": sha256_file(config_path),
        "stage_training_config": sha256_file(stage_config_path),
        "parent_causal_checkpoint": sha256_file(stage["parent_checkpoint_path"]),
    }
    _require(hashes["base_training_config"] == stage["base_config_sha256"], "base config changed")
    _require(hashes["parent_causal_checkpoint"] == stage["parent_checkpoint_sha256"], "parent causal checkpoint changed")
    if config.data.manifest_sha256:
        _require(hashes["dataset_manifest"] == config.data.manifest_sha256, "dataset manifest changed")
    from training.data.action_dataset import validate_feature_cache_binding, validate_window97_contract
    validate_window97_contract(validate_feature_cache_binding(config.data.feature_index_path, config.data.manifest_path))
    return hashes


def load_parent_checkpoint(config_path, stage, hashes):
    """No Action/teacher model load; validate this actual causal60 source."""
    base = load_causal_config(config_path)
    payload = torch.load(stage["parent_checkpoint_path"], map_location="cpu", weights_only=False)
    _require(payload.get("format_version") == 1 and payload.get("stage") == stage["parent_expected_stage"]
             and payload.get("step") == 60 and payload.get("micro_batches_consumed") == 480,
             "parent is not the exact clean causal60/micro480 checkpoint")
    _require(_canonical(payload.get("config")) == _canonical(base.to_dict()), "parent config differs from original YAML")
    expected = {key: hashes[key] for key in ("dataset_manifest", "feature_index", "feature_receipt")}
    expected.update(training_config=stage["base_config_sha256"], teacher_checkpoint=base.lineage.checkpoint_sha256)
    _require(payload.get("manifest_hashes") == expected, "parent historical artifact lineage mismatch")
    _require("error_recycling_state" not in payload, "parent must not contain a recycling buffer")
    state = payload.get("trainable_model")
    _require(isinstance(state, dict) and bool(state), "parent has no trainable state")
    _require(all(torch.is_tensor(value) and bool(torch.isfinite(value).all()) for value in state.values()),
             "parent parameters must be finite tensors")
    lineage = {"kind": "causal_weights_only_parent", "path": str(Path(stage["parent_checkpoint_path"]).resolve()),
               "sha256": stage["parent_checkpoint_sha256"], "stage": payload["stage"], "step": 60,
               "micro_batches_consumed": 480, "manifest_hashes": copy.deepcopy(payload["manifest_hashes"]),
               "historical_action_parent": copy.deepcopy(payload.get("parent_teacher"))}
    return state, lineage


def _build_model(config, state, device):
    from utils.wan_wrapper import WanDiffusionWrapper
    model = construct_causal_wrapper(config, WanDiffusionWrapper)
    if config.model.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    # Historical helper name configures adapters/LoRA only, and loads no teacher.
    summary = configure_action_teacher(model.model, rank=config.model.lora_rank,
                                       alpha=config.model.lora_alpha, dropout=config.model.lora_dropout)
    _require({name for name, parameter in model.named_parameters() if parameter.requires_grad} == set(state),
             "new causal model parameter names differ from actual parent")
    load_trainable_state_dict(model, state)
    return model.to(device=device, dtype=torch.bfloat16).train(), summary


def build_offset_dataloader(config, stage, *, consumed_micro=0):
    from training.data.action_resampled import ResampledActionDataset
    start = stage["data_start_absolute_index"]
    stop = start + config.training.max_steps * config.training.gradient_accumulation_steps
    _require(type(consumed_micro) is int and 0 <= consumed_micro < stop - start, "invalid local microbatch cursor")
    dataset = ResampledActionDataset(config.data.feature_index_path, num_samples=stop,
        manifest_path=config.data.manifest_path, split="train", seed=config.training.seed,
        timestep_shift=5.0, prompt_cache_path=None, num_frames=97)
    # Stateless base sampling preserves global index meanings; do not load/skip
    # 480 historical tensor batches or reset index0 on a fresh stage.
    subset = Subset(dataset, range(start + consumed_micro, stop))
    return DataLoader(subset, batch_size=1, shuffle=False, num_workers=config.data.num_workers,
                      pin_memory=True, persistent_workers=config.data.num_workers > 0, drop_last=True)


def _checkpoint_payload(model, optimizer, *, config, stage, hashes, lineage, step, micro, metrics):
    _require(micro == step * config.training.gradient_accumulation_steps, "checkpoint local data cursor differs")
    return {
        "format_version": 1, "stage": STAGE_NAME, "step": step, "micro_batches_consumed": micro,
        "absolute_micro_batches_consumed": stage["data_start_absolute_index"] + micro,
        "next_absolute_sample_index": stage["data_start_absolute_index"] + micro,
        "data_start_absolute_index": stage["data_start_absolute_index"],
        "trainable_model": trainable_state_dict(model), "optimizer": optimizer.state_dict(),
        "config": config.to_dict(), "stage_config": copy.deepcopy(stage),
        "base_config_sha256": hashes["base_training_config"], "stage_config_sha256": hashes["stage_training_config"],
        "manifest_hashes": hashes, "parent_causal": lineage, "method_contract": method_contract(stage),
        "initialization_mode": INITIALIZATION, "metrics": metrics,
        "torch_rng_state": torch.get_rng_state(), "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
    }


def validate_resume_checkpoint(path, *, config, stage, hashes):
    payload = load_checkpoint(path, expected_manifest_hashes=hashes, map_location="cpu")
    _require(payload.get("format_version") == 1 and payload.get("stage") == STAGE_NAME,
             "strict resume requires a real history-noise stage checkpoint")
    _require(_canonical(payload.get("config")) == _canonical(config.to_dict()) and payload.get("stage_config") == stage,
             "strict resume effective/base stage config changed")
    _require(payload.get("method_contract") == method_contract(stage) and payload.get("initialization_mode") == INITIALIZATION,
             "strict resume history-noise method changed")
    _require(payload.get("base_config_sha256") == hashes["base_training_config"]
             and payload.get("stage_config_sha256") == hashes["stage_training_config"], "strict resume config SHA changed")
    step, micro = payload.get("step"), payload.get("micro_batches_consumed")
    _require(type(step) is int and 0 < step <= config.training.max_steps and type(micro) is int
             and micro == step * config.training.gradient_accumulation_steps, "strict resume local optimizer cursor differs")
    absolute = stage["data_start_absolute_index"] + micro
    _require(payload.get("data_start_absolute_index") == stage["data_start_absolute_index"]
             and payload.get("absolute_micro_batches_consumed") == absolute
             and payload.get("next_absolute_sample_index") == absolute, "strict resume absolute data cursor differs")
    parent = payload.get("parent_causal", {})
    _require(parent.get("kind") == "causal_weights_only_parent"
             and parent.get("sha256") == stage["parent_checkpoint_sha256"]
             and parent.get("stage") == "causal_teacher_forcing_v1" and parent.get("step") == 60,
             "strict resume causal parent changed")
    _require(all(key in payload for key in ("trainable_model", "optimizer", "torch_rng_state", "cuda_rng_state_all",
                                           "python_rng_state", "numpy_rng_state")), "strict resume missing optimizer/RNG state")
    _require("parent_teacher" not in payload and "teacher_flow" not in payload and "error_recycling_state" not in payload,
             "unexpected teacher/recycling stage state")
    return payload


def _iterator_preserving_rng(loader):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            return iter(loader)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def launch(config, stage, config_path, stage_config_path, *, authorization, stop_after_step=20, resume=None):
    _require(isinstance(authorization, dict) and bool(authorization), "fresh external authorization receipt required")
    target = stop_after_step
    _require(type(target) is int and 1 <= target <= config.training.max_steps, "execution cap exceeds configured horizon")
    hashes = artifact_hashes(config, stage, config_path, stage_config_path)
    source_state, lineage = load_parent_checkpoint(config_path, stage, hashes)
    restored = validate_resume_checkpoint(resume, config=config, stage=stage, hashes=hashes) if resume else None
    output = Path(config.training.output_dir)
    _require(resume is not None or not output.exists() or not any(output.iterdir()), "fresh output directory must be empty")
    if resume:
        _require(output.resolve() in Path(resume).resolve().parents, "resume checkpoint is outside this run")
        _require(restored["parent_causal"] == lineage and restored["step"] < target, "resume parent differs or target already reached")
    _require(torch.cuda.is_available(), "authorized launch requires CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(config.training.seed)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"command": sys.argv, **source_provenance(authorization), "stage": STAGE_NAME,
        "config": config.to_dict(), "base_config_path": str(config_path), "stage_config": stage,
        "manifest_hashes": hashes, "parent_causal": lineage, "method_contract": method_contract(stage),
        "initialization_mode": INITIALIZATION, "execution_mode": "strict_resume" if resume else "weights_only_new_stage",
        "authorization": authorization, "started_unix": time.time(), "execution_stop_after_step": target}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    model, summary = _build_model(config, source_state, device)
    del source_state
    optimizer = tf_entry._build_optimizer(model, config)
    start, micro = 0, 0
    if restored:
        load_trainable_state_dict(model, restored["trainable_model"])
        optimizer.load_state_dict(restored["optimizer"])
        torch.set_rng_state(restored["torch_rng_state"])
        torch.cuda.set_rng_state_all(restored["cuda_rng_state_all"])
        random.setstate(restored["python_rng_state"])
        np.random.set_state(restored["numpy_rng_state"])
        start, micro = restored["step"], restored["micro_batches_consumed"]
        del restored
    loader = build_offset_dataloader(config, stage, consumed_micro=micro)
    iterator = _iterator_preserving_rng(loader)
    manager = CheckpointManager(output, keep_last=2)
    noise_config = HistoryNoiseConfig.from_dict(stage["history_noise"])
    model.zero_grad(set_to_none=True)
    step, last_saved, accumulated, latest = start, start, 0.0, {}
    stage_started = time.perf_counter()
    print(json.dumps({"status": "launching", "stage": STAGE_NAME, "resume_step": start,
        "next_absolute_sample_index": 480 + micro, "trainable_parameters": summary.trainable_parameters,
        "history_noise": noise_config.to_dict(), "teacher_model_loaded": False}, sort_keys=True), flush=True)
    while step < target:
        if micro % 8 == 0:
            torch.cuda.synchronize(device)
            step_started, wait_seconds, augmentation = time.perf_counter(), 0.0, []
        waited = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration as exc:
            raise RuntimeError("absolute dataloader exhausted before declared optimizer target") from exc
        wait_seconds += time.perf_counter() - waited
        noise_metrics = {}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = history_noise_loss(model, batch, config, device, noise_config=noise_config,
                absolute_microbatch_index=stage["data_start_absolute_index"] + micro,
                stage_optimizer_step=step, metrics=noise_metrics)
        _require(bool(torch.isfinite(loss)), "non-finite history-noise loss")
        (loss / 8).backward()
        accumulated += float(loss.detach())
        augmentation.append(noise_metrics)
        micro += 1
        if micro % 8:
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.optimizer.max_grad_norm)
        _require(bool(torch.isfinite(grad_norm)), "non-finite history-noise gradient")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - step_started
        latest = {"loss": accumulated / 8, "grad_norm": float(grad_norm), "optimizer_step_seconds": elapsed,
            "data_wait_seconds": wait_seconds, "optimizer_compute_seconds_excluding_data_wait": elapsed - wait_seconds,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "samples_per_second": (micro - start * 8) / (time.perf_counter() - stage_started),
            "absolute_micro_batches_consumed": 480 + micro, "history_noise_microbatches": augmentation}
        accumulated = 0.0
        record = {"step": step, **latest, "unix": time.time()}
        append_jsonl(output / "metrics.jsonl", record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if step % 20 == 0:
            payload = _checkpoint_payload(model, optimizer, config=config, stage=stage, hashes=hashes,
                lineage=lineage, step=step, micro=micro, metrics=latest)
            manager.save(payload, step=step, metric=latest["loss"])
            del payload
            last_saved = step
    if last_saved != step:
        payload = _checkpoint_payload(model, optimizer, config=config, stage=stage, hashes=hashes,
            lineage=lineage, step=step, micro=micro, metrics=latest)
        manager.save(payload, step=step, metric=latest["loss"])
    result = {"status": "complete" if step == config.training.max_steps else "segment_complete", "stage": STAGE_NAME,
        "step": step, "resume_step": start, "micro_batches_consumed": micro,
        "absolute_micro_batches_consumed": 480 + micro, "next_absolute_sample_index": 480 + micro,
        "configured_max_steps": config.training.max_steps, "execution_stop_after_step": target,
        "checkpoint_path": str((output / "checkpoints" / f"step-{step:07d}.pt").resolve()),
        "output_dir": str(output.resolve()), "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "parent_causal": lineage, "method_contract": method_contract(stage), "final_metrics": latest}
    (output / "execution_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage-config", required=True)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--authorizer")
    parser.add_argument("--stop-after-step", type=int, default=20)
    parser.add_argument("--resume")
    args = parser.parse_args(argv)
    config, stage = load_stage(args.config, args.stage_config)
    _require(1 <= args.stop_after_step <= config.training.max_steps, "execution cap exceeds configured horizon")
    if not args.launch:
        print(json.dumps({"status": "cpu_configuration_valid", "stage": STAGE_NAME, "config": config.to_dict(),
            "stage_config": stage, "method_contract": method_contract(stage), "cuda_queried": False,
            "weights_loaded": False, "execution_stop_after_step": args.stop_after_step}, indent=2, sort_keys=True))
        return 0
    _require(args.authorizer and ":" in args.authorizer, "--launch requires explicit --authorizer module:function")
    module, name = args.authorizer.split(":", 1)
    receipt = getattr(importlib.import_module(module), name)(args=args, config=config, stage_config=stage)
    launch(config, stage, args.config, args.stage_config, authorization=receipt,
           stop_after_step=args.stop_after_step, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
