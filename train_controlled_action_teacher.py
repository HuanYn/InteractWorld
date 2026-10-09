"""Fresh bounded action teacher with zero gate and frozen zero-residual LoRA.

Independent official-base trajectory; no R005 or failed teacher inheritance.
The fixed20-step stage must first measure exactly two optimizer steps.
Without --launch only CPU configuration validation is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import random
import sys
import time

import numpy as np
import torch

import train_action_teacher as teacher
from training.config import ActionTeacherConfig, load_config as _load_config
from training.data.action_resampled import sampling_contract
from training.models.lora import LoRALinear, load_trainable_state_dict, trainable_state_dict
from training.runtime import OptimizerStepProfiler, append_jsonl, git_revision, seed_everything, sha256_file

STAGE = STAGE_NAME = "controlled_action_native97_v1"
CHECKPOINT_SCHEMA = "controlled_action_teacher_checkpoint_v1"
FACTORY = teacher.RESAMPLED_ACTION_FACTORY
MANIFEST_SHA256 = "fd47f92ad2c02aa2f19a2a1dac810c21320fc52bd05a56ff0d9443985764d4cb"
MAX_STEPS = 20
RATIO_CAP = 0.1
GATE_LR = 1e-2
GATE_CHANNELS = 3072
SAMPLE_ELEMENTS = 4096


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def precision_contract():
    return dict(backbone_parameter_dtype="bfloat16", adapter_parameter_dtype="bfloat16",
                frozen_lora_parameter_dtype="float32", gate_parameter_dtype="float32",
                adapter_gradient_dtype="bfloat16", gate_gradient_dtype="float32",
                forward_autocast_dtype="bfloat16", separate_master_copy=False)

def method_contract():
    return dict(method=STAGE, objective="bidirectional_clean_first_latent_flow_matching",
                official_abot_reproduction=False, hyperparameters_source="project_specific_controlled_action_v1",
                frozen_official_backbone=True, train_action_adapter=True, train_lora=False,
                frozen_lora_zero_residual=True, train_controlled_action_gate=True,
                causal=False, history_kv_training=False, distillation=False,
                first_latent_in_loss=False, latent_frames=25, future_action_frames=96,
                action_scale=1.0, adapter_lr=1e-5, lora_lr=0.0, gate_lr=GATE_LR,
                adapter_weight_decay=0.01, gate_weight_decay=0.0, ratio_cap=RATIO_CAP,
                gate_channels=GATE_CHANNELS, gate_initialization="float32_zeros",
                residual="ratio_cap*RMS(video.detach)*tanh(gate)*action/RMS(action)",
                normalization_axis="per_token_channel", ratio_cap_is_quality_evidence=False,
                max_optimizer_steps=MAX_STEPS, prompt_policy="scene_static_only_v1",
                prompt_storage="inline_97_frame_feature_shards",
                saved_tensors_on_cpu=True, gradient_checkpointing=True,
                checkpoint_retention="all_step_endpoints_no_pruning_no_best_overwrite",
                frozen_lora_checkpoint_storage=False,
                precision=precision_contract(), quality_validated=False)

def initialization_contract(config):
    return dict(mode="official_base_fresh_adapter_zero_gate_frozen_lora",
                base_model_path=config.model.base_model_path, source_checkpoint=None,
                adapter_seed=0, lora_seed=42,
                adapter_initialization="existing_complete_missing_adapter_reset_parameters",
                lora_initialization="FP32_kaiming_uniform_A_zero_B_seed42_frozen",
                gate_initialization="FP32_zeros_3072", optimizer_restored=False, rng_restored=False,
                start_step=0, micro_batches_consumed=0, training_seed=42)

def validate_probe_config(config):
    _require(type(config) is ActionTeacherConfig, "controlled teacher requires ActionTeacherConfig only")
    config.validate()
    _require(config.data.num_frames == 97 and config.data.data_factory == FACTORY
             and config.data.prompt_cache_path is None and not config.data.low_sigma_stratification,
             "controlled teacher requires native97 inline-static resampled data, not GT-prefix")
    _require(config.data.manifest_sha256 == MANIFEST_SHA256
             and config.data.precomputed_latents and config.data.precomputed_text_embeddings,
             "controlled teacher requires the pinned64-episode/256-window cache")
    _require(config.model.action_scale == 1 and config.model.lora_alpha == 16
             and config.model.lora_dropout == 0 and config.model.timestep_shift == 5
             and config.model.gradient_checkpointing, "controlled teacher model hyperparameters changed")
    _require(config.optimizer.name == "adamw8bit" and config.optimizer.adapter_lr == 1e-5
             and config.optimizer.lora_lr == 0 and tuple(config.optimizer.betas) == (0.9, 0.95)
             and config.optimizer.weight_decay == 0.01 and config.optimizer.max_grad_norm == 1,
             "controlled teacher optimizer hyperparameters changed")
    _require(config.training.lora_parameter_dtype == "float32" and config.training.max_steps == MAX_STEPS
             and config.training.checkpoint_every == 20 and config.training.log_every == 1,
             "controlled teacher fixes frozen FP32 LoRA/max20/checkpoint20/log1")


def load_config(path):
    config = _load_config(path)
    validate_probe_config(config)
    return config


def _absolute(value):
    raw = str(value)
    path = PureWindowsPath(raw) if PureWindowsPath(raw).drive else PurePosixPath(raw)
    _require(path.is_absolute() and ".." not in path.parts, "paths must be absolute without traversal")
    return path


def validate_config(config, *, data_root, output_dir, model_dir):
    validate_probe_config(config)
    data, output, model = map(_absolute, (data_root, output_dir, model_dir))
    _require(_absolute(config.data.manifest_path) == data / "manifests" / "train.jsonl",
             "data-root differs from the configured native97 manifest")
    _require(_absolute(config.training.output_dir) == output and _absolute(config.model.base_model_path) == model,
             "output-dir/model-dir must match the fixed serialized configuration")
    _require(output != data and output not in data.parents and output != model and output not in model.parents,
             "training output must not contain the input data or official base")


def artifact_hashes(config, config_path):
    validate_probe_config(config)
    hashes = teacher._manifest_hashes(config, config_path)
    from training.data.action_dataset import cache_index_path
    index = cache_index_path(config.data.manifest_path)
    receipt = json.loads(index.with_suffix(index.suffix + ".receipt.json").read_text(encoding="utf-8"))
    _require(receipt.get("episodes") == 64 and receipt.get("windows") == 256,
             "controlled teacher requires the existing64-episode/256-window receipt")
    return hashes


def base_identity(config):
    """Check local base headers/stat identity, not a new full-weight byte audit."""
    from safetensors import safe_open
    root = Path(config.model.base_model_path).resolve()
    config_file = root / "config.json"
    _require(config_file.is_file(), "official base config.json is missing")
    shards = sorted(root.glob("diffusion_pytorch_model*.safetensors"))
    _require(bool(shards), "official local diffusion safetensors are missing")
    records, keys = [], set()
    for shard in shards:
        _require(shard.resolve().parent == root, "base shard escapes the selected model directory")
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            names = sorted(handle.keys())
        _require(not any("act_control_adapter" in name or "lora_" in name for name in names),
                 "controlled teacher refuses a base containing learned action-adapter/LoRA tensors")
        _require(not keys.intersection(names), "base has duplicate diffusion tensor names")
        keys.update(names)
        stat = shard.stat()
        records.append(dict(name=shard.name, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                            tensor_names_sha256=hashlib.sha256("\n".join(names).encode()).hexdigest()))
    _require("patch_embedding.weight" in keys and "head.head.weight" in keys, "not a complete Wan base header")
    return dict(path=str(root), config_sha256=sha256_file(config_file), diffusion_shards=records,
                verification="existing_local_pinned_revision_plus_headers_and_stat_not_full_weight_hash")


def validate_base_identity(payload, config):
    _require(payload.get("base_identity") == base_identity(config), "official base identity changed")


def _build_model(config, device):
    """Rebuild deterministic zero-residual frozen LoRA and install the zero gate."""
    model, summary = teacher._build_model(config, device)
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device="cpu").manual_seed(42)
        for module in model.modules():
            if type(module) is LoRALinear:
                # CPU draws make the initialization identical in CPU eval and GPU training.
                value = torch.empty(module.lora_a.weight.shape, dtype=torch.float32)
                torch.nn.init.kaiming_uniform_(value, a=math.sqrt(5), generator=generator)
                with torch.no_grad():
                    module.lora_a.weight.copy_(value)
                    module.lora_b.weight.zero_()
    for module in model.modules():
        if type(module) is LoRALinear:
            module.lora_a.requires_grad_(False)
            module.lora_b.requires_grad_(False)
    from utils.controlled_action import install_controlled_action
    from dataclasses import replace
    install_controlled_action(model.model, ratio_cap=RATIO_CAP)
    _model_signature(model)
    summary = replace(summary,
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      total_parameters=sum(p.numel() for p in model.parameters()))
    return model, summary


def _group(name):
    if "act_control_adapter." in name:
        return "adapter"
    if name == "model.controlled_action_gate":
        return "gate"
    raise ValueError(f"unexpected trainable parameter: {name}")

def _model_signature(model):
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = _group(name)
            dtype = torch.float32 if group == "gate" else torch.bfloat16
            _require(parameter.dtype == dtype, f"incorrect actual {group} parameter dtype")
            if group == "gate":
                _require(parameter.shape == (GATE_CHANNELS,), "gate must have exactly3072 channels")
            result[name] = dict(shape=list(parameter.shape), dtype=str(dtype), group=group)
    _require({row["group"] for row in result.values()} == {"adapter", "gate"}, "Adapter and gate required")
    return result

def validate_checkpoint(payload, *, config, hashes):
    validate_probe_config(config)
    _require(isinstance(payload, dict) and payload.get("format_version") == 1
             and payload.get("checkpoint_schema") == CHECKPOINT_SCHEMA and payload.get("stage") == STAGE,
             "checkpoint must be this fresh official-base stage, never R005/causal")
    for name, expected in dict(config=config.to_dict(), manifest_hashes=hashes,
                               initialization=initialization_contract(config), method_contract=method_contract(),
                               precision_contract=precision_contract(), sampling_contract=sampling_contract(config.training)).items():
        _require(payload.get(name) == expected, f"strict fresh checkpoint {name} changed")
    step, cursor = payload.get("step"), payload.get("micro_batches_consumed")
    _require(type(step) is int and 2 <= step <= MAX_STEPS and type(cursor) is int and cursor == step * 8,
             "fresh checkpoint optimizer/data cursor is inconsistent")
    state, signature = payload.get("trainable_model"), payload.get("model_signature")
    _require(isinstance(state, dict) and bool(state) and isinstance(signature, dict)
             and set(state) == set(signature), "checkpoint model tensor names/signature differ")
    groups = set()
    for name, value in state.items():
        group = _group(name)
        groups.add(group)
        if group == "gate":
            _require(torch.is_tensor(value) and value.shape == (GATE_CHANNELS,), "checkpoint gate geometry differs")
        dtype = torch.float32 if group == "gate" else torch.bfloat16
        _require(torch.is_tensor(value) and value.dtype == dtype
                 and signature[name] == dict(shape=list(value.shape), dtype=str(dtype), group=group),
                 f"checkpoint shape/precision differs: {name}")
        _require(bool(torch.isfinite(value).all()), f"nonfinite checkpoint parameter: {name}")
    _require(groups == {"adapter", "gate"}, "checkpoint is missing a trainable group")
    _require(isinstance(payload.get("base_identity"), dict), "checkpoint lacks official base identity")
    return payload


def validate_resume(path, *, config, hashes):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint(payload, config=config, hashes=hashes)
    optimizer = payload.get("optimizer")
    _require(isinstance(optimizer, dict) and isinstance(optimizer.get("state"), dict)
             and isinstance(optimizer.get("param_groups"), list), "resume lacks optimizer state")
    groups = optimizer["param_groups"]
    _require(len(groups) == 2 and [row.get("name") for row in groups] == ["action_adapter", "controlled_action_gate"],
             "resume optimizer groups changed")
    ids = []
    for group, name, lr, decay in zip(groups, ("adapter", "gate"), (1e-5, GATE_LR), (0.01, 0.0)):
        expected_count = sum(row["group"] == name for row in payload["model_signature"].values())
        _require(group.get("lr") == lr and tuple(group.get("betas", ())) == (0.9, 0.95)
                 and group.get("weight_decay") == decay and len(group.get("params", [])) == expected_count,
                 "resume optimizer configuration/parameter count changed")
        ids.extend(group["params"])
    _require(len(ids) == len(set(ids)) and set(ids) == set(optimizer["state"])
             and all(bool(optimizer["state"][key]) for key in ids), "resume optimizer tensor state is incomplete")
    try:
        torch.Generator(device="cpu").set_state(payload["torch_rng_state"])
        random.Random().setstate(payload["python_rng_state"])
        np.random.RandomState().set_state(payload["numpy_rng_state"])
        cuda = payload["cuda_rng_state_all"]
        _require(isinstance(cuda, list) and len(cuda) == 1
                 and all(torch.is_tensor(x) and x.dtype == torch.uint8 and x.ndim == 1 and x.numel() > 0 for x in cuda),
                 "resume requires the complete single-visible-GPU RNG state")
    except (KeyError, TypeError, RuntimeError) as exc:
        raise ValueError("resume RNG state is missing or invalid") from exc
    return payload


def _sample_parameters(model):
    return {name: parameter.detach().reshape(-1)[::max(1, math.ceil(parameter.numel() / SAMPLE_ELEMENTS))]
            [:SAMPLE_ELEMENTS].to(device="cpu").clone()
            for name, parameter in model.named_parameters() if parameter.requires_grad}


def _update_evidence(model, before, *, step):
    after = _sample_parameters(model)
    _require(set(before) == set(after), "parameter sample names changed")
    groups = {key: dict(sampled_elements=0, changed_elements=0, update_l2=0.0) for key in ("adapter", "gate")}
    for name, value in after.items():
        _require(bool(torch.isfinite(value).all()), "nonfinite updated parameter sample")
        difference = value.float() - before[name].float()
        row = groups[_group(name)]
        row["sampled_elements"] += difference.numel()
        row["changed_elements"] += int(torch.count_nonzero(difference))
        row["update_l2"] += float(difference.double().square().sum())
    for row in groups.values():
        row["update_l2"] = math.sqrt(row["update_l2"])
    _require(groups["gate"]["changed_elements"] > 0, "optimizer produced no actual gate update")
    if step >= 2:
        _require(groups["adapter"]["changed_elements"] > 0,
                 "step2 onward requires a sampled representable Adapter update")
    return dict(method="deterministic_per_tensor_strided_cpu_sample_v1",
                maximum_elements_per_tensor=SAMPLE_ELEMENTS, groups=groups, gate_updated=True,
                adapter_updated=groups["adapter"]["changed_elements"] > 0,
                adapter_zero_first_gradient_expected=step == 1, frozen_lora_updated=False,
                sampled_not_full_parameter_census=True)

def _gradient_precision(model):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            _require(p.grad is None, "frozen backbone/LoRA unexpectedly has gradients")
            continue
        group = _group(name)
        dtype = torch.float32 if group == "gate" else torch.bfloat16
        _require(p.dtype == dtype and p.grad is not None and p.grad.dtype == dtype,
                 f"each {group} parameter requires its declared actual gradient dtype")
        sample = p.grad.detach().reshape(-1)[::max(1, math.ceil(p.numel() / SAMPLE_ELEMENTS))][:SAMPLE_ELEMENTS].float()
        _require(bool(torch.isfinite(sample).all()), "nonfinite sampled gradient")
        row = groups.setdefault(group, dict(parameter_dtype=str(dtype), gradient_dtype=str(dtype),
                    gradient_tensors=0, sampled_gradient_elements=0, sampled_nonzero_gradients=0,
                    sampled_gradient_l2_squared=0.0))
        row["gradient_tensors"] += 1
        row["sampled_gradient_elements"] += sample.numel()
        row["sampled_nonzero_gradients"] += int(torch.count_nonzero(sample))
        row["sampled_gradient_l2_squared"] += float(sample.double().square().sum())
    _require(set(groups) == {"adapter", "gate"}, "Adapter and gate gradient groups required")
    for row in groups.values():
        row["sampled_gradient_l2"] = math.sqrt(row.pop("sampled_gradient_l2_squared"))
    return dict(observed_at="after_accumulation_before_clip_and_step", groups=groups,
                gradient_value_evidence="deterministic_per_tensor_strided_sample",
                frozen_lora_has_gradients=False)


def _gate_statistics(model):
    gate = model.model.controlled_action_gate.detach().float()
    _require(bool(torch.isfinite(gate).all()), "nonfinite gate")
    return dict(dtype=str(gate.dtype), channels=gate.numel(), abs_max=float(gate.abs().max()),
                rms=float(gate.square().mean().sqrt()), tanh_abs_max=float(gate.tanh().abs().max()),
                residual_ratio_upper_bound=RATIO_CAP * float(gate.tanh().abs().max()),
                measured_injected_residual_ratio=False)


def _build_optimizer(model, config):
    _model_signature(model)
    parameters = {group: [p for name, p in model.named_parameters()
                          if p.requires_grad and _group(name) == group] for group in ("adapter", "gate")}
    import bitsandbytes as bnb
    return bnb.optim.AdamW8bit([
        dict(params=parameters["adapter"], name="action_adapter", lr=config.optimizer.adapter_lr,
             weight_decay=config.optimizer.weight_decay),
        dict(params=parameters["gate"], name="controlled_action_gate", lr=GATE_LR, weight_decay=0.0),
    ], betas=config.optimizer.betas)

def _save_checkpoint(output, payload):
    """Exclusive endpoint creation; no pruning or mutable best.pt alias."""
    directory = Path(output) / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"step-{payload['step']:07d}.pt"
    temporary = target.with_suffix(".partial")
    _require(not target.exists() and not temporary.exists(), "refusing to overwrite checkpoint or partial evidence")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, target)  # Atomic and fails rather than overwriting an existing endpoint.
    temporary.unlink()  # Only our newly written, fully linked temporary file.
    return target


def _checkpoint(model, optimizer, *, config, hashes, identity, step, metrics):
    return dict(format_version=1, checkpoint_schema=CHECKPOINT_SCHEMA, stage=STAGE,
                config=config.to_dict(), manifest_hashes=hashes, base_identity=identity,
                initialization=initialization_contract(config), method_contract=method_contract(),
                precision_contract=precision_contract(), sampling_contract=sampling_contract(config.training),
                model_signature=_model_signature(model), trainable_model=trainable_state_dict(model),
                optimizer=optimizer.state_dict(), step=step, micro_batches_consumed=step * 8, metrics=metrics,
                torch_rng_state=torch.get_rng_state(), cuda_rng_state_all=torch.cuda.get_rng_state_all(),
                python_rng_state=random.getstate(), numpy_rng_state=np.random.get_state())


def launch(config, config_path, *, data_root, output_dir, model_dir, stop_after_step,
           resume=None, shared_lease, shared_max_seconds, shared_request_sha256):
    validate_config(config, data_root=data_root, output_dir=output_dir, model_dir=model_dir)
    _require(type(stop_after_step) is int and 2 <= stop_after_step <= MAX_STEPS, "execution cap must be2..20")
    _require(type(shared_max_seconds) is int and 1 <= shared_max_seconds <= 14400, "finite watchdog must be<=4h")
    output = Path(output_dir).resolve()
    if resume is None:
        _require(stop_after_step == 2, "fresh execution must first measure exactly two optimizer steps")
        _require(not output.exists() or not any(output.iterdir()), "fresh output must be empty")
    else:
        _require(Path(resume).resolve().parent == output / "checkpoints", "resume must belong to this run")
    hashes, identity = artifact_hashes(config, config_path), base_identity(config)
    restored = validate_resume(resume, config=config, hashes=hashes) if resume else None
    if restored:
        _require(restored["base_identity"] == identity, "resume official base identity changed")
        _require(Path(resume).name == f"step-{restored['step']:07d}.pt", "resume checkpoint filename/cursor differs")
    start = restored["step"] if restored else 0
    _require(start < stop_after_step, "execution cap must exceed the restored optimizer step")
    gpu = teacher._shared_training_gpu(shared_lease, max_seconds=shared_max_seconds,
                                       request_sha256=shared_request_sha256)
    _require(torch.cuda.is_available(), "leased launch requires CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(42)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(stage=STAGE, config=config.to_dict(), manifest_hashes=hashes, base_identity=identity,
                    method_contract=method_contract(), initialization=initialization_contract(config),
                    command=sys.argv, git_revision=git_revision(Path(__file__).parent),
                    entry_sha256=sha256_file(__file__), gpu_state=gpu.as_dict(),
                    shared_guard=dict(lease=shared_lease, max_seconds=shared_max_seconds,
                                      request_sha256=shared_request_sha256, watchdog_owner="external_guard_worker"),
                    resume=resume, resume_step=start, stop_after_step=stop_after_step, started_unix=time.time())
    if not resume:
        with (output / "run_metadata.json").open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
    append_jsonl(output / "executions.jsonl", metadata)
    profiler = OptimizerStepProfiler(device, output_dir=output / "profile_output", gpu_uuid=shared_lease["gpu_uuid"])
    began = time.monotonic()
    try:
        with profiler.phase("model_load"):
            model, summary = _build_model(config, device)
        optimizer = _build_optimizer(model, config)
        if restored:
            _require(restored["model_signature"] == _model_signature(model), "resume tensor names/shapes differ from real model")
            load_trainable_state_dict(model, restored["trainable_model"])
            optimizer.load_state_dict(restored["optimizer"])
        loader = teacher._import_factory(FACTORY)(config=config.data, training=config.training)
        _, iterator = teacher._iterator_at_micro_batch(loader, start * 8)
        if restored:
            teacher._restore_rng_state(restored, require_complete=True)
            del restored
        model.zero_grad(set_to_none=True)
        latest = {}
        for step in range(start + 1, stop_after_step + 1):
            before = _sample_parameters(model)
            profiler.begin_step(step)
            losses = []
            for _ in range(8):
                profiler.set_phase("data_wait")
                batch = next(iterator)
                profiler.set_phase("forward")
                with torch.autograd.graph.save_on_cpu(pin_memory=True), torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = teacher._forward_loss(model, batch, config, device)
                _require(bool(torch.isfinite(loss)), "nonfinite controlled-action-teacher loss")
                profiler.set_phase("backward")
                (loss / 8).backward()
                losses.append(float(loss.detach()))
                del loss
            precision = _gradient_precision(model)
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            _require(bool(torch.isfinite(norm)), "nonfinite controlled-action-teacher accumulated gradient")
            precision["gradients_finite"] = True
            profiler.set_phase("optimizer_update")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            timing = profiler.end_step(step)
            evidence = _update_evidence(model, before, step=step)
            del before
            memory = timing["memory"]
            latest = dict(loss=sum(losses) / 8, grad_norm=float(norm), precision_summary=precision,
                          update_evidence=evidence, gate_statistics=_gate_statistics(model), optimizer_step_seconds=timing["seconds"], step_profile=timing,
                          elapsed_seconds=time.monotonic() - began,
                          peak_vram_bytes=memory["cumulative_peak_allocated_bytes"],
                          peak_reserved_vram_bytes=memory["cumulative_peak_reserved_bytes"],
                          host_peak_rss_bytes=memory.get("host_peak_rss_bytes"),
                          samples_per_second=8 / timing["seconds"])
            append_jsonl(output / "metrics.jsonl", dict(step=step, micro_batches_consumed=step * 8, **latest))
            print(json.dumps(dict(step=step, **latest)), flush=True)
            if step in (2, MAX_STEPS) or step == stop_after_step:
                profiler.begin_checkpoint(step)
                target = _save_checkpoint(output, _checkpoint(model, optimizer, config=config, hashes=hashes,
                                           identity=identity, step=step, metrics=latest))
                profiler.end_checkpoint(step, checkpoint_path=str(target))
        result = dict(status="completed", stage=STAGE, step=stop_after_step, micro_batches_consumed=stop_after_step * 8,
                      resume_step=start, checkpoint_path=str(target), configured_max_steps=MAX_STEPS,
                      trainable_parameters=summary.trainable_parameters, **latest,
                      qualification=dict(frozen_lora=True, gate_updated=True, adapter_updated=True,
                                         gradients_finite=True, peak_reserved_bytes=latest["peak_reserved_vram_bytes"]),
                      quality_improvement_verified=False)
        append_jsonl(output / "execution_segments.jsonl", result)
        (output / "execution_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    except BaseException as exc:
        profiler.close(error=exc)
        append_jsonl(output / "failures.jsonl", dict(status="failed", stage=STAGE, failure_type=type(exc).__name__,
                                                    error=str(exc), timestamp_unix=time.time(), resume_step=start))
        raise
    finally:
        profiler.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data-root", "output-dir", "model-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--stop-after-step", type=int, required=True)
    parser.add_argument("--resume")
    parser.add_argument("--shared-gpu-lease", help="JSON file containing the active shared lease")
    parser.add_argument("--max-seconds", type=int)
    parser.add_argument("--request-sha256")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    validate_config(config, data_root=args.data_root, output_dir=args.output_dir, model_dir=args.model_dir)
    if not args.launch:
        report = teacher.cpu_dry_run(config, args.config)
        report.update(stage=STAGE, method_contract=method_contract(), initialization=initialization_contract(config),
                      sampling_contract=sampling_contract(config.training), execution_stop_after_step=args.stop_after_step)
        print(json.dumps(report, indent=2))
        return 0
    if not all(value is not None for value in (args.shared_gpu_lease, args.max_seconds, args.request_sha256)):
        parser.error("--launch requires --shared-gpu-lease --max-seconds --request-sha256")
    lease = json.loads(Path(args.shared_gpu_lease).read_text(encoding="utf-8"))
    launch(config, args.config, data_root=args.data_root, output_dir=args.output_dir, model_dir=args.model_dir,
           stop_after_step=args.stop_after_step, resume=args.resume, shared_lease=lease,
           shared_max_seconds=args.max_seconds, shared_request_sha256=args.request_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
