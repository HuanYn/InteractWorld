"""Step20 qualification for the fresh, RMS-bounded zero-gated action teacher.

This entry never accepts old step64/R005/causal weights, inference-only scaling,
or ablations. It reuses the training constructor and controlled injection, then
the existing phase-offloaded VAE/CFG/Euler runtime. No reference memory, causal
generation, distillation, quality gain or real-time performance is implied.
"""
from __future__ import annotations

import copy
import gc
import json
import os
from pathlib import Path
import re
import time

from training.demo.contracts import action_array, contained, load_initial, require, sha256, write_json
from training.demo.latency_engine import ReusableActionEngine
from training.demo.relocation import ArtifactHashes
from training.studio.abot_fresh_eval import ACTION_CASES, PreservingWriter, action_case_array
from training.studio.causal_native_eval import tensor_sha
from training.studio.quality_probe import DEFAULT_NEGATIVE_CACHE, QualityRuntime, capture_lora_scales, load_negative
from training.studio.subset_eval import CONTROLS, FIXED_INPUTS

MODE = 'controlled_action_eval'
REVISION = 'controlled-action-native97-step20-euler40-cfg5-v1'
STAGE = 'controlled_action_native97_v1'
DATA_ROOT = Path('/DATA_DISK/InteractWorld')
CHECKPOINT_VERIFICATION_CACHE = DATA_ROOT / 'creator-runtime/abot-retrain-20261003/.controlled-eval-checkpoint-hashes.json'
PATH_FIELDS = ('checkpoint', 'training_config', 'source_job', 'prompt_cache', 'negative_cache')
HASH_FIELDS = ('checkpoint_sha256', 'training_config_sha256', 'prompt_cache_sha256',
               'prompt_receipt_sha256', 'negative_cache_sha256', 'negative_receipt_sha256',
               'config_sha256', 'initial_sha256', 'actions_sha256')


def validate_request(request):
    require(request.get('mode') == MODE and request.get('experiment_revision') == REVISION,
            'explicit controlled-action evaluation revision required')
    require(type(request.get('checkpoint_step')) is int and request['checkpoint_step'] == 20,
            'controlled-action qualification is fixed at step20')
    require(request.get('action_case') in ACTION_CASES, 'explicit forward=WJ or reverse=WL case required')
    require(request.get('action_segments') == [dict(frames=96, keys=ACTION_CASES[request['action_case']])],
            'actual action schedule must match the fixed six-second case')
    require(type(request.get('future_frames')) is int and request['future_frames'] == 96
            and request.get('seed') == 42 and request.get('steps') == 40
            and request.get('cfg_scale') == 5 and request.get('solver') == 'flow_euler'
            and request.get('fastpath') is False, 'fixed native97/seed42/Euler40/CFG5 route required')
    require(type(request.get('max_seconds')) is int and 60 <= request['max_seconds'] <= 3600,
            'finite <=3600-second watchdog required')
    require(isinstance(request.get('source_revision'), str) and bool(request['source_revision'].strip()),
            'deployed source revision required')
    require(isinstance(request.get('job_id'), str)
            and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}', request['job_id']) is not None,
            'unsafe controlled-action evaluation identity')
    for key in PATH_FIELDS:
        require(isinstance(request.get(key), str) and Path(request[key]).is_absolute(), f'absolute {key} required')
    for key in HASH_FIELDS:
        require(isinstance(request.get(key), str)
                and re.fullmatch(r'[0-9a-f]{64}', request[key]) is not None, f'exact {key} required')
    forbidden = ('component_ablation', 'action_scale', 'effective_action_scale', 'calibrated_action_scale',
                 'ratio_cap', 'gate_scale', 'lora_strength', 'purpose', 'trained_stage', 'parent_checkpoint',
                 'allow_fp32_lora_candidate')
    require(not any(key in request for key in forbidden),
            'no legacy checkpoint escape hatch or inference-only control override is accepted')


def checkpoint_digest(request):
    prompt, negative, source = map(Path, (request['prompt_cache'], request['negative_cache'], request['source_job']))
    protected = [request[key] for key in PATH_FIELDS]
    protected.extend((prompt.with_suffix(prompt.suffix + '.receipt.json'), negative.with_suffix('.json')))
    protected.extend(source / name for name, _ in FIXED_INPUTS)
    return ArtifactHashes(CHECKPOINT_VERIFICATION_CACHE, protected_paths=protected)(request['checkpoint'])


def make_request(*, job_id, source_revision, checkpoint, training_config, source_job,
                 dev_prompt_sidecar, checkpoint_step=20, action_case, max_seconds=1800,
                 negative_cache=DEFAULT_NEGATIVE_CACHE):
    require(action_case in ACTION_CASES, 'unknown action case')
    source, prompt, negative = map(Path, (source_job, dev_prompt_sidecar, negative_cache))
    request = dict(mode=MODE, experiment_revision=REVISION, job_id=job_id, source_revision=source_revision,
        checkpoint=str(checkpoint), checkpoint_step=checkpoint_step,
        training_config=str(training_config), training_config_sha256=sha256(training_config),
        source_job=str(source), prompt_cache=str(prompt), prompt_cache_sha256=sha256(prompt),
        prompt_receipt_sha256=sha256(prompt.with_suffix(prompt.suffix + '.receipt.json')),
        negative_cache=str(negative), negative_cache_sha256=sha256(negative),
        negative_receipt_sha256=sha256(negative.with_suffix('.json')),
        action_case=action_case, action_segments=[dict(frames=96, keys=list(ACTION_CASES[action_case]))],
        future_frames=96, seed=42, steps=40, cfg_scale=5, solver='flow_euler', fastpath=False,
        max_seconds=max_seconds)
    request['checkpoint_sha256'] = checkpoint_digest(request)
    request.update({field: sha256(source / name) for name, field in FIXED_INPUTS})
    validate_request(request)
    return request


def _load_weights(request):
    import torch
    import train_controlled_action_teacher as entry
    validate_request(request)
    require(sha256(request['training_config']) == request['training_config_sha256']
            and checkpoint_digest(request) == request['checkpoint_sha256'], 'controlled checkpoint/config changed')
    config = entry.load_config(request['training_config'])
    entry.validate_probe_config(config)
    hashes = entry.artifact_hashes(config, request['training_config'])
    payload = torch.load(request['checkpoint'], map_location='cpu', weights_only=False)
    entry.validate_checkpoint(payload, config=config, hashes=hashes)
    entry.validate_base_identity(payload, config)
    require(payload['stage'] == STAGE and payload['step'] == 20,
            'actual checkpoint is not the new controlled-action step20 endpoint')
    lineage = dict(stage=payload['stage'], step=payload['step'], checkpoint=request['checkpoint'],
        checkpoint_sha256=request['checkpoint_sha256'], checkpoint_schema=payload['checkpoint_schema'],
        initialization=payload['initialization'], manifest_hashes=hashes,
        method_contract=payload['method_contract'], precision_contract=payload['precision_contract'],
        sampling_contract=payload['sampling_contract'], base_identity=payload['base_identity'],
        reference_memory_implemented=False, complete_abot_reproduction=False,
        historical_r005_used_for_initialization=False, historical_step64_used_for_initialization=False,
        trained_lora=False, checkpoint_contains_lora=False)
    return payload['trainable_model'], config, lineage


def load_inputs(request):
    """CPU-only checkpoint plus held-out input checks; no T5/model downloads."""
    import numpy as np
    import torch
    import yaml
    validate_request(request)
    source = Path(request['source_job'])
    for name, field in FIXED_INPUTS:
        require(sha256(source / name) == request[field], f'fixed source changed: {name}')
    raw = yaml.safe_load((source / 'rollout.yaml').read_text(encoding='utf-8'))
    require(len(raw['scenes']) == 1, 'one fixed dev scene required')
    scene = raw['scenes'][0]
    require(scene['seed'] == 42 and scene['action_segments'] == CONTROLS, 'original fixed dev source differs')
    initial = load_initial(source / 'initial.npy')
    historical_actions = np.load(source / 'actions.npy', allow_pickle=False)
    require(initial.shape == (480, 832, 3) and initial.dtype == np.uint8
            and historical_actions.dtype == np.float32
            and np.array_equal(historical_actions, action_array(CONTROLS)), 'source RGB/actions differ')
    state, config, lineage = _load_weights(request)
    prompt_path = Path(request['prompt_cache'])
    receipt_path = prompt_path.with_suffix(prompt_path.suffix + '.receipt.json')
    require(sha256(prompt_path) == request['prompt_cache_sha256']
            and sha256(receipt_path) == request['prompt_receipt_sha256'], 'dev text cache changed')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    require(receipt.get('kind') == 'window97_eval_text_reference_v1'
            and receipt.get('scope') == 'dev_text_only_no_video_features'
            and receipt.get('prompt_policy') == 'scene_static_only_v1'
            and receipt.get('cache_sha256') == request['prompt_cache_sha256']
            and Path(receipt.get('training_manifest', '')).resolve() == Path(config.data.manifest_path).resolve()
            and receipt.get('training_manifest_sha256') == lineage['manifest_hashes']['dataset_manifest'],
            'dev text receipt does not match the controlled-action training dataset')
    episode = scene['source_episode_id']
    binding = receipt.get('episodes', {}).get(episode, {})
    require(binding.get('split') == 'dev' and binding.get('prompt') == scene['prompt'],
            'selected scene is not the held-out dev static prompt')
    for line in Path(config.data.manifest_path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            require(episode not in (row.get('episode_id'), row.get('source_episode_id'), row.get('sample_id')),
                    'evaluation episode is present in the training manifest')
    cache = torch.load(prompt_path, map_location='cpu', weights_only=True)
    prompt = cache['prompt_embeds'][episode]
    require(prompt.ndim == 2 and 1 <= prompt.shape[0] <= 512 and prompt.shape[1] == 4096
            and bool(torch.isfinite(prompt).all()) and tensor_sha(prompt) ==
            receipt.get('selected_embedding_sha256_float32', {}).get(episode), 'dev prompt tensor differs')
    negative_path = Path(request['negative_cache'])
    require(sha256(negative_path) == request['negative_cache_sha256']
            and sha256(negative_path.with_suffix('.json')) == request['negative_receipt_sha256'],
            'negative text source changed')
    negative, negative_source = load_negative(negative_path, config.model.base_model_path)
    actions = action_case_array(request['action_case'])
    require(actions.shape == (96, 8) and actions.dtype == np.float32, 'actual action tensor geometry differs')
    lineage.update(prompt_text=scene['prompt'], prompt_episode=episode,
        prompt_sha256_float32=tensor_sha(prompt), negative_source=negative_source,
        source_action_segments=CONTROLS, actual_action_segments=request['action_segments'],
        actual_action_sha256_float32=tensor_sha(torch.from_numpy(actions)),
        training_rgb_frames=97, evaluation_rgb_frames=97, t5_loaded=False,
        future_ground_truth_used=False, action_pairing='same_image_text_seed_noise;WJ96_vs_WL96',
        injection_consistency='same_training_constructor_and_controlled_action_helper;no_inference_override')
    return state, config, prompt.detach().to(torch.bfloat16), negative, initial, actions, lineage


def prepare(*, output, **options):
    output = Path(output)
    require(not output.exists() and output.name == options['job_id'], 'new matching evaluation directory required')
    request = make_request(**options)
    inputs = load_inputs(request)
    lineage = inputs[-1]
    del inputs
    gc.collect()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'request.json', request)
    write_json(output / 'preparation.json', dict(status='cpu_prepared', request=request, lineage=lineage,
        gpu_launched=False, video_generated=False, quality_verified=False))
    return request


def build_inference_model(config, state):
    import torch
    import train_controlled_action_teacher as entry
    from training.models.lora import load_trainable_state_dict
    from utils.controlled_action import controlled_action_contract
    model, summary = entry._build_model(config, torch.device('cpu'))
    require(model.is_causal is False and model.uniform_timestep is True,
            'controlled teacher evaluation must be bidirectional')
    expected = {name: value for name, value in model.named_parameters() if value.requires_grad}
    require(set(state) == set(expected) and bool(expected), 'controlled trainable tensor names differ')
    for name, value in expected.items():
        require(name == 'model.controlled_action_gate' or name.startswith('model.act_control_adapter.'),
                'only controlled gate and Adapter may be trainable')
        tensor = state[name]
        require(torch.is_tensor(tensor) and tensor.shape == value.shape and tensor.dtype == value.dtype
                and bool(torch.isfinite(tensor).all()), 'controlled checkpoint tensor differs: ' + name)
    load_trainable_state_dict(model, state)
    gate = getattr(model.model, 'controlled_action_gate', None)
    adapters = [p for name, p in model.named_parameters() if name.startswith('model.act_control_adapter.')]
    lora = [(name, p) for name, p in model.named_parameters() if name.endswith(('.lora_a.weight', '.lora_b.weight'))]
    require(gate is not None and gate.dtype == torch.float32 and tuple(gate.shape) == (3072,)
            and bool(adapters) and all(p.dtype == torch.bfloat16 for p in adapters),
            'controlled gate/Adapter precision or shape differs')
    require(bool(lora) and all(p.dtype == torch.float32 and not p.requires_grad for _, p in lora)
            and all(not bool(torch.count_nonzero(p)) for name, p in lora if name.endswith('.lora_b.weight')),
            'fresh frozen LoRA must retain a zero B residual')
    require(getattr(model.model, 'controlled_action_enabled', False) is True
            and getattr(model.model, 'controlled_action_ratio_cap', None) == .1
            and getattr(model.model, 'controlled_action_eps', None) == 1e-6
            and config.model.action_scale == 1.0, 'training/inference controlled injection differs')
    require(not any(bool(getattr(model.model, flag, False)) for flag in
                    ('enable_teacache', 'use_teacache', 'enable_fbcache', 'use_fbcache')),
            'approximate inference caches must be disabled')
    model.controlled_action_receipt = controlled_action_contract(model.model)
    return model.requires_grad_(False).eval(), summary


class ControlledRuntime(QualityRuntime):
    def __init__(self, state, config, negative):
        import torch
        from utils.wan_wrapper import WanVAEWrapper
        self.torch, self.config, self.residency = torch, copy.deepcopy(config.model), 'phase'
        self.model = self.vae = None
        self.lora_scales, self.negative, self.forward_calls = [], negative, 0
        try:
            self.model, summary = build_inference_model(config, state)
            self.controlled_action_receipt = self.model.controlled_action_receipt
            self.trainable_parameters = summary.trainable_parameters
            self.vae = WanVAEWrapper(pretrained_path=str(Path(config.model.base_model_path) / 'Wan2.2_VAE.pth'))
            self.vae.requires_grad_(False).eval().to(dtype=torch.bfloat16)
            self.lora_scales = capture_lora_scales(self.model)
            self.configure(dict(cfg_scale=5, lora_strength=1, action_scale=1, solver='flow_euler'))
        except BaseException:
            self.close()
            raise


def run(request, directory, lease):
    import numpy as np
    import torch
    from PIL import Image
    from training.demo.backend import validate_active_authority
    from training.eval.rollout15s import ffmpeg_writer_factory
    validate_request(request)
    directory = contained(directory, DATA_ROOT)
    require(directory.name == request['job_id'], 'controlled evaluation job directory differs')
    target = directory / 'controlled-evaluation'
    target.mkdir(exist_ok=False)
    began = time.monotonic()
    report = dict(status='running', mode=MODE, revision=REVISION, request=request,
        reference_memory_implemented=False, complete_abot_reproduction=False,
        fastpath=False, teacache=False, automatic_retry=False, failed_outputs_preserved=True,
        quality_improvement_verified=False, realtime=False, future_ground_truth_used=False,
        trained_lora=False, checkpoint_contains_lora=False, inference_override=False)
    runtime = engine = None
    def check():
        require(time.monotonic() - began < request['max_seconds'] - 20, 'finite controlled evaluation deadline reached')
        validate_active_authority(lease)
    try:
        check()
        for name in PATH_FIELDS:
            contained(request[name], DATA_ROOT)
        require(os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1',
                'existing offline model assets required')
        state, config, positive, negative, initial, actions, lineage = load_inputs(request)
        contained(config.model.base_model_path, DATA_ROOT)
        report['lineage'] = lineage
        Image.fromarray(initial).save(target / 'initial.png')
        np.save(target / 'actions.npy', actions, allow_pickle=False)
        check()
        tick = time.monotonic()
        runtime = ControlledRuntime(state, config, negative)
        report['controlled_action'] = runtime.controlled_action_receipt
        write_json(target / 'inputs.json', dict(request=request, lineage=lineage, actual_actions=actions.tolist(),
            initial_rgb_sha256_float32=tensor_sha(torch.from_numpy(initial)), actual_action_case=request['action_case'],
            controlled_action=runtime.controlled_action_receipt, inference_override=False))
        state.clear()
        gc.collect()
        report['load_seconds'] = time.monotonic() - tick
        engine = ReusableActionEngine({}, runtime.config, residency='phase', fastpath=False, _runtime=runtime,
            _writer_factory=lambda *args: PreservingWriter(ffmpeg_writer_factory(*args)))
        engine.load_seconds = report['load_seconds']
        generation, endpoint = engine.run_segment(initial=initial, prompt=positive, actions=actions,
            seed=42, output_path=target / 'raw.mp4', steps=40)
        np.save(target / 'endpoint.npy', endpoint.numpy(), allow_pickle=False)
        require(generation['frames_written'] == 97 and runtime.forward_calls == 80,
                'controlled qualification must perform40 CFG pairs and produce97 frames')
        require(generation.get('fastpath') is False and generation.get('is_causal') is False,
                'controlled evaluation unexpectedly used a fast or causal route')
        generation['sampler'].update(cfg='text_classifier_free', cfg_scale=5, model_forwards_per_step=2)
        generation['controlled_action'] = runtime.controlled_action_receipt
        report.update(status='completed', frames_written=97, generation=generation,
            raw_video=str(target / 'raw.mp4'), actual_model_forward_calls=runtime.forward_calls,
            peak_allocated_bytes=generation['peak_vram_bytes'], peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            inference_precision=dict(gate='float32', frozen_lora='float32', adapter='bfloat16',
                base='bfloat16', compute='bfloat16', checkpoint_rewritten=False),
            quality_evaluation='awaiting_visual_review')
        check()
    except BaseException as error:
        report.update(status='failed', failure_type=type(error).__name__, error=str(error))
        if engine is not None:
            report['partial_generation'] = copy.deepcopy(engine.last_run_report)
        raise
    finally:
        try:
            if engine is not None:
                engine.close()
            elif runtime is not None:
                runtime.close()
        finally:
            report['elapsed_seconds'] = time.monotonic() - began
            write_json(target / 'result.json', report)
    return report
