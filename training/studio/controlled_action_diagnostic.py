"""Fixed no-grad action/flow diagnostic for controlled step20, not video quality.

Preparation reads eight mmap action slices, never model weights or latent data.
Execution requires the existing Creator launcher lease; there is no GPU CLI.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time

from training.demo.contracts import contained, require, sha256, write_json

MODE = 'controlled_action_diagnostic'
REVISION = 'controlled20-fixed4-sigma2-action4-flow-diagnostic-v1'
ROOT = Path('/DATA_DISK/InteractWorld/creator-runtime/abot-retrain-20261003/teacher-controlled-v1')
CONDITIONS = ('correct', 'jl_swap', 'zero', 'gate_off')
SIGMAS = (.25, .75)
SELECTION = 'top4_JL_unequal_frame_count_from_consumed_absolute_indices0to7_tie_index_v1'


def protocol():
    return dict(selection=SELECTION, candidate_absolute_indices=list(range(8)), window_count=4,
                sigmas=list(SIGMAS), conditions=list(CONDITIONS), seed=42, model_forward_calls=32,
                training_windows_only=True, cfg=False, decoding=False, backward=False,
                first_latent_loss=False, target='noise-clean; BF16 target as existing teacher loss',
                shared_noise_across_sigmas_and_conditions=True, ratio_cap=.1,
                zero_actions_are_not_gate_off=True, visual_quality_verified=False)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _stat(path):
    info = Path(path).stat()
    return dict(bytes=info.st_size, mtime_ns=info.st_mtime_ns)


def _tensor_digest(value):
    # Only tiny action tensors, not full shards or model weights.
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def _action_counts(actions):
    import torch
    require(tuple(actions.shape) == (96, 8) and bool(torch.isfinite(actions).all())
            and bool(((actions == 0) | (actions == 1)).all()), 'expected binary96x8 canonical actions')
    return dict(jl_unequal_frames=int((actions[:, 5] != actions[:, 7]).sum()),
                WJ_frames=int(((actions[:, 0] == 1) & (actions[:, 5] == 1)).sum()),
                WL_frames=int(((actions[:, 0] == 1) & (actions[:, 7] == 1)).sum()),
                zero_frames=int((actions.sum(1) == 0).sum()),
                key_frames=dict(zip('WASDIJKL', [int(x) for x in actions.sum(0)])))


def _choose(candidates):
    require([r['absolute_index'] for r in candidates] == list(range(8)), 'exact first8 consumed candidates required')
    usable = [r for r in candidates if r['action_counts']['jl_unequal_frames'] > 0]
    require(len(usable) >= 4, 'fewer than4 swap-informative windows; do not expand sample search automatically')
    top = sorted(usable, key=lambda r: (-r['action_counts']['jl_unequal_frames'], r['absolute_index']))[:4]
    return sorted(top, key=lambda r: r['absolute_index'])


def _candidates(config):
    import torch
    from training.data.action_dataset import _seed64, cache_index_path
    from training.data.action_resampled import ResampledActionDataset, SAMPLING_NAMESPACE
    index_path = cache_index_path(config.data.manifest_path)
    episodes = sorted([json.loads(line) for line in index_path.read_text(encoding='utf-8').splitlines()
                       if line.strip()], key=lambda r: r['episode_id'])
    require(len(episodes) == 64 and sum(r['num_windows'] for r in episodes) == 256,
            'expected existing64-episode/256-window cache')
    # Resolve the real sampler without its all-shard hash/read constructor.
    sampler = ResampledActionDataset.__new__(ResampledActionDataset)
    sampler.episodes, sampler.num_samples = episodes, 160
    sampler.data_seed = _seed64(42, SAMPLING_NAMESPACE)
    candidates = []
    for index in range(8):
        spec = sampler.sampling_spec(index)
        episode, offset = episodes[spec['episode_slot']], spec['window_index']
        for shard in episode['shards']:
            if offset < shard['samples']:
                path = contained(index_path.parent / shard['path'], index_path.parent)
                payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
                require(payload['episode_id'] == episode['episode_id'], 'cached episode differs')
                actions = payload['actions'][offset]
                candidates.append(dict(absolute_index=index, episode_id=episode['episode_id'],
                    window_index=spec['window_index'], shard=str(path), shard_offset=offset,
                    shard_stat=_stat(path), shard_recorded_sha256=shard['sha256'],
                    source_start=int(payload['source_starts'][offset]),
                    action_counts=_action_counts(actions), action_sha256=_tensor_digest(actions)))
                del actions, payload
                break
            offset -= shard['samples']
        else:
            raise ValueError('sampler window not found in shards')
    return candidates, index_path


def validate_request(request, directory):
    require(request.get('mode') == MODE and request.get('experiment_revision') == REVISION
            and request.get('protocol') == protocol(), 'explicit fixed diagnostic protocol required')
    require(request.get('job_id') == Path(directory).name, 'diagnostic job identity differs')
    require(type(request.get('max_seconds')) is int and 60 <= request['max_seconds'] <= 600,
            'diagnostic requires finite <=600second watchdog')
    require(request.get('selected_windows') == _choose(request.get('candidate_windows', [])),
            'frozen label-only window selection changed')
    for name in ('evaluation_request', 'training_config', 'manifest', 'feature_index'):
        require(isinstance(request.get(name), str) and Path(request[name]).is_absolute(), f'absolute{name} required')
    require(isinstance(request.get('source_revision'), str) and bool(request['source_revision']), 'source revision required')


def prepare(*, output, evaluation_request, source_revision, max_seconds=600):
    import train_controlled_action_teacher as entry
    from training.studio import controlled_action_eval as evaluation
    output = contained(output, ROOT)
    require(not output.exists(), 'new diagnostic identity required')
    source = Path(evaluation_request).resolve()
    require(source.is_file(), 'existing controlled20 evaluation request required')
    evaluation_request_value = _read(source)
    evaluation.validate_request(evaluation_request_value)
    config_path = Path(evaluation_request_value['training_config'])
    require(sha256(config_path) == evaluation_request_value['training_config_sha256'], 'training config changed')
    config = entry.load_config(config_path)
    candidates, index_path = _candidates(config)
    manifest = Path(config.data.manifest_path)
    request = dict(mode=MODE, experiment_revision=REVISION, job_id=output.name,
        source_revision=source_revision, max_seconds=max_seconds, protocol=protocol(),
        evaluation_request=str(source), evaluation_request_sha256=sha256(source),
        training_config=str(config_path), training_config_sha256=sha256(config_path),
        manifest=str(manifest), manifest_stat=_stat(manifest),
        feature_index=str(index_path), feature_index_stat=_stat(index_path),
        candidate_windows=candidates, selected_windows=_choose(candidates),
        preparation=dict(weights_loaded=False,latent_values_read=False,action_slices_read=8,
                         full_shard_hashes_recomputed=False))
    validate_request(request, output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'request.json', request)
    return request


def _window(row):
    import torch
    require(_stat(row['shard']) == row['shard_stat'], 'selected shard stat identity changed')
    payload = torch.load(row['shard'], map_location='cpu', weights_only=True, mmap=True)
    require(payload['episode_id'] == row['episode_id'], 'selected episode differs')
    offset = row['shard_offset']
    actions = payload['actions'][offset].float().clone()
    require(_tensor_digest(actions) == row['action_sha256'] and _action_counts(actions) == row['action_counts'],
            'selected action tensor changed')
    clean, prompt = payload['clean_latents'][offset].float().clone(), payload['prompt_embeds'].float().clone()
    require(tuple(clean.shape) == (25, 48, 30, 52) and bool(torch.isfinite(clean).all()), 'native97 clean latent differs')
    require(prompt.ndim == 2 and 1 <= prompt.shape[0] <= 512 and prompt.shape[1] == 4096
            and bool(torch.isfinite(prompt).all()), 'static prompt geometry differs')
    require(not any(key in payload for key in ('clip_fea', 'y')), 'unexpected extra conditioning in selected cache')
    return clean, prompt, actions


def make_batch(clean, prompt, actions, *, absolute_index, sigma):
    import torch
    from training.data.action_dataset import _seed64
    require(sigma in SIGMAS, 'fixed sigma required')
    seed = _seed64(42, REVISION, absolute_index)
    noise = torch.randn(clean.shape, generator=torch.Generator(device='cpu').manual_seed(seed))
    noisy, target = (1 - sigma) * clean + sigma * noise, noise - clean
    noisy[0], target[0] = clean[0], 0
    time_labels = torch.full((1, clean.shape[0]), sigma * 1000., dtype=torch.float32)
    time_labels[:, 0] = 0
    return dict(noisy_latents=noisy[None], target_flow=target[None], timesteps=time_labels,
                prompt_embeds=prompt[None], actions=actions[None]), seed


def action_condition(actions, condition):
    require(condition in CONDITIONS, 'unknown action condition')
    value = actions.clone()
    if condition == 'jl_swap':
        value[..., 5], value[..., 7] = actions[..., 7], actions[..., 5]
    elif condition == 'zero':
        value.zero_()
    return value


def _rms(value):
    return float(value.float().square().mean().sqrt())


class _ForwardTrace:
    """Observe the same forward; no additional denoiser calls or output mutation."""
    def __init__(self, wrapper):
        self.model, self.handles = wrapper.model, []
        self.reset()
        self.handles.append(self.model.patch_embedding.register_forward_hook(self._patch))
        self.handles.append(self.model.act_control_adapter.register_forward_hook(self._action))
        self.handles.append(wrapper.register_forward_hook(self._velocity))

    def reset(self):
        self.patch = self.velocity = None
        self.metrics = {}

    def _patch(self, module, inputs, output):
        self.patch = output.detach()

    def _action(self, module, inputs, output):
        from utils.controlled_action import inject_controlled_action_features
        require(self.patch is not None, 'action forward preceded video patch embedding')
        offset = self.patch.shape[2] - output.shape[2]
        future = self.patch[:, :, offset:]
        controlled = inject_controlled_action_features(self.model, self.patch, output)
        delta = controlled[:, :, offset:].float() - future.float()
        video_rms = _rms(future)
        self.metrics = dict(raw_action_feature_rms=_rms(output), video_patch_rms=video_rms,
            actual_bf16_injected_residual_rms=_rms(delta),
            actual_global_rms_ratio=_rms(delta) / max(video_rms, 1e-30),
            action_input_rms=_rms(inputs[0]),
            residual_measurement='exact helper recomputation including BF16 cast/add; not just theoretical gate bound')
        self.patch = None

    def _velocity(self, module, inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        self.velocity = value.detach().float().cpu()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.reset()


def run_diagnostic(request, directory, lease, request_sha):
    import torch
    import train_action_teacher as teacher
    from training.demo.backend import validate_active_authority
    from training.runtime import append_jsonl
    from training.studio import controlled_action_eval as evaluation
    directory = contained(directory, ROOT)
    validate_request(request, directory)
    require(sha256(directory / 'request.json') == request_sha, 'leased diagnostic request differs')
    require(not (directory / 'diagnostic-metrics.jsonl').exists(), 'diagnostic cannot overwrite previous observations')
    validate_active_authority(lease)
    teacher._shared_training_gpu(lease, max_seconds=request['max_seconds'], request_sha256=request_sha)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one leased visible GPU required')
    require(sha256(request['evaluation_request']) == request['evaluation_request_sha256']
            and sha256(request['training_config']) == request['training_config_sha256'], 'model source request/config changed')
    require(_stat(request['manifest']) == request['manifest_stat']
            and _stat(request['feature_index']) == request['feature_index_stat'], 'data metadata changed')
    began, device, records, model, trace = time.monotonic(), torch.device('cuda', 0), [], None, None
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        state, config, lineage = evaluation._load_weights(_read(request['evaluation_request']))
        require(config.data.manifest_path == request['manifest'], 'checkpoint data manifest differs')
        model, summary = evaluation.build_inference_model(config, state)
        del state
        gc.collect()
        model.to(device)
        trace = _ForwardTrace(model)
        gate = model.model.controlled_action_gate
        original_gate = gate.detach().clone()
        with torch.inference_mode():
            for selected in request['selected_windows']:
                clean, prompt, actions = _window(selected)
                for sigma in SIGMAS:
                    batch, seed = make_batch(clean, prompt, actions, absolute_index=selected['absolute_index'], sigma=sigma)
                    correct_velocity = None
                    correct_loss = None
                    for condition in CONDITIONS:
                        require(time.monotonic() - began < request['max_seconds'], 'finite diagnostic deadline reached')
                        trace.reset()
                        local = dict(batch, actions=action_condition(batch['actions'], condition))
                        gate.copy_(original_gate)
                        if condition == 'gate_off':
                            gate.zero_()
                        tick = time.monotonic()
                        try:
                            with torch.autocast('cuda', dtype=torch.bfloat16):
                                loss = teacher._forward_loss(model, local, config, device)
                        finally:
                            gate.copy_(original_gate)
                        require(trace.velocity is not None and bool(trace.metrics) and bool(torch.isfinite(trace.velocity).all())
                                and bool(torch.isfinite(loss)), 'nonfinite/missing diagnostic flow')
                        if condition == 'gate_off':
                            require(trace.metrics['actual_bf16_injected_residual_rms'] == 0,
                                    'gate-off condition still injected an action residual')
                        velocity = trace.velocity[:, 1:]
                        if condition == 'correct':
                            correct_velocity, correct_loss = velocity.clone(), float(loss)
                        row = dict(absolute_index=selected['absolute_index'], episode_id=selected['episode_id'],
                            window_index=selected['window_index'], sigma=sigma, noise_seed=seed,
                            condition=condition, flow_loss=float(loss), loss_minus_correct=float(loss)-correct_loss,
                            velocity_rms=_rms(velocity), velocity_delta_vs_correct_rms=_rms(velocity-correct_velocity),
                            target_rms=_rms(batch['target_flow'][:, 1:].bfloat16()),
                            condition_action_sha256=_tensor_digest(local['actions']),
                            forward_seconds=time.monotonic()-tick, **trace.metrics)
                        records.append(row)
                        append_jsonl(directory / 'diagnostic-metrics.jsonl', row)
                        print(json.dumps(row), flush=True)
                    del correct_velocity, batch
                del clean, prompt, actions
            require(torch.equal(gate, original_gate), 'diagnostic must restore original gate')
        require(len(records) == 32, 'diagnostic requires exactly32 fixed model forwards')
        comparisons = {name: dict(mean_loss_delta=sum(r['loss_minus_correct'] for r in records if r['condition']==name)/8,
            pairs_correct_loss_lower=sum(r['loss_minus_correct']>0 for r in records if r['condition']==name),
            pairs=8, mean_velocity_delta_rms=sum(r['velocity_delta_vs_correct_rms'] for r in records if r['condition']==name)/8)
            for name in CONDITIONS[1:]}
        result = dict(status='completed', mode=MODE, experiment_revision=REVISION, protocol=protocol(),
            lineage=lineage, selected_windows=request['selected_windows'], actual_model_forward_calls=len(records),
            comparisons=comparisons, metrics_path=str(directory/'diagnostic-metrics.jsonl'),
            peak_vram_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(device), elapsed_seconds=time.monotonic()-began,
            model_weights_updated=False, gate_restored=True, visual_quality_verified=False,
            action_correctness_verified=False, independent_dev_evaluation=False)
        write_json(directory / 'diagnostic-summary.json', result)
        return result
    finally:
        if trace is not None:
            trace.close()
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
