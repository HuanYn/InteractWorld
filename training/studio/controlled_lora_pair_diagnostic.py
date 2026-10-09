"""Matched 32-forward action diagnostic for both new LoRA arms.

The earlier controlled20 selection and noise generator are reused exactly, not
selected again after seeing new results. Lower unconditional loss is not action
control or visual success. Only the shared GPU worker may invoke execution.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.studio import controlled_action_diagnostic as previous
from training.studio import controlled_lora_pair_eval as evaluation

MODE = 'controlled_lora_pair_diagnostic'
REVISION = 'controlled-lora-pair-fixed4-sigma2-action4-flow-v1'
ROOT = Path('/DATA_DISK/InteractWorld/creator-runtime/abot-retrain-20261003/teacher-controlled-lora-pair-v1')
TEMPLATE = previous.ROOT / 'controlled20-actiondiag-v1/request.json'
CONDITIONS, SIGMAS = previous.CONDITIONS, previous.SIGMAS
make_batch, action_condition = previous.make_batch, previous.action_condition


def protocol():
    return dict(previous.protocol(), noise_revision=previous.REVISION,
        selection_reused_from='controlled20-actiondiag-v1', selection_recomputed=False,
        interpretation='paired_condition_margins_not_unconditional_loss_or_visual_success')


def validate_request(request, directory):
    require(request.get('mode') == MODE and request.get('experiment_revision') == REVISION
            and request.get('protocol') == protocol(), 'explicit paired diagnostic protocol required')
    require(request.get('arm') in evaluation.ARMS, 'explicit diagnostic LoRA arm required')
    projected = dict(request, mode=previous.MODE, experiment_revision=previous.REVISION,
                     protocol=previous.protocol())
    previous.validate_request(projected, directory)
    require(request.get('selection_source') == str(TEMPLATE), 'fixed original selection source required')
    require(isinstance(request.get('selection_source_sha256'), str)
            and len(request['selection_source_sha256']) == 64, 'original selection source identity required')


def _verify_template(request):
    require(sha256(TEMPLATE) == request['selection_source_sha256'], 'original diagnostic selection changed')
    template = previous._read(TEMPLATE)
    previous.validate_request(template, TEMPLATE.parent)
    for field in ('manifest', 'manifest_stat', 'feature_index', 'feature_index_stat',
                  'candidate_windows', 'selected_windows'):
        require(request.get(field) == template[field], 'paired diagnostic changed frozen ' + field)
    return template


def prepare(*, output, evaluation_request, source_revision, max_seconds=600):
    import train_controlled_lora_pair as entry
    output = contained(output, ROOT)
    require(not output.exists(), 'new paired diagnostic identity required')
    source = Path(evaluation_request).resolve()
    require(source.is_file(), 'existing paired endpoint evaluation request required')
    evaluation_value = previous._read(source)
    evaluation.validate_request(evaluation_value)
    config_path = Path(evaluation_value['training_config'])
    require(sha256(config_path) == evaluation_value['training_config_sha256'], 'paired training config changed')
    config = entry.load_config(config_path)
    template = previous._read(TEMPLATE)
    previous.validate_request(template, TEMPLATE.parent)
    require(str(Path(config.data.manifest_path)) == template['manifest'], 'paired dataset differs from fixed diagnostic')
    require(previous._stat(template['manifest']) == template['manifest_stat']
            and previous._stat(template['feature_index']) == template['feature_index_stat'], 'data metadata changed')
    request = dict(mode=MODE, experiment_revision=REVISION, job_id=output.name, arm=evaluation_value['arm'],
        source_revision=source_revision, max_seconds=max_seconds, protocol=protocol(),
        evaluation_request=str(source), evaluation_request_sha256=sha256(source),
        training_config=str(config_path), training_config_sha256=sha256(config_path),
        selection_source=str(TEMPLATE), selection_source_sha256=sha256(TEMPLATE),
        preparation=dict(weights_loaded=False, latent_values_read=False, action_slices_read=0,
            selection_reused=True, full_shard_hashes_recomputed=False))
    for field in ('manifest', 'manifest_stat', 'feature_index', 'feature_index_stat',
                  'candidate_windows', 'selected_windows'):
        request[field] = template[field]
    validate_request(request, output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'request.json', request)
    return request


def run_diagnostic(request, directory, lease, request_sha):
    import torch
    import train_action_teacher as teacher
    from training.demo.backend import validate_active_authority
    from training.runtime import append_jsonl
    directory = contained(directory, ROOT)
    validate_request(request, directory)
    require(sha256(directory / 'request.json') == request_sha, 'leased diagnostic request differs')
    require(not (directory / 'diagnostic-metrics.jsonl').exists(), 'diagnostic cannot overwrite previous observations')
    _verify_template(request)
    validate_active_authority(lease)
    teacher._shared_training_gpu(lease, max_seconds=request['max_seconds'], request_sha256=request_sha)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one leased visible GPU required')
    require(sha256(request['evaluation_request']) == request['evaluation_request_sha256']
            and sha256(request['training_config']) == request['training_config_sha256'], 'model source/config changed')
    require(previous._stat(request['manifest']) == request['manifest_stat']
            and previous._stat(request['feature_index']) == request['feature_index_stat'], 'data metadata changed')
    evaluation_value = previous._read(request['evaluation_request'])
    require(evaluation_value['arm'] == request['arm'], 'diagnostic/evaluation arm differs')
    began, device, records, model, trace = time.monotonic(), torch.device('cuda', 0), [], None, None
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        state, config, lineage = evaluation._load_weights(evaluation_value)
        require(config.data.manifest_path == request['manifest'], 'checkpoint data manifest differs')
        model, _ = evaluation.build_inference_model(config, state, arm=request['arm'])
        del state
        gc.collect()
        model.to(device)
        trace = previous._ForwardTrace(model)
        gate = model.model.controlled_action_gate
        original_gate = gate.detach().clone()
        with torch.inference_mode():
            for selected in request['selected_windows']:
                clean, prompt, actions = previous._window(selected)
                for sigma in SIGMAS:
                    batch, seed = make_batch(clean, prompt, actions, absolute_index=selected['absolute_index'], sigma=sigma)
                    correct_velocity = correct_loss = None
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
                                    'gate-off condition still injected action residual')
                        velocity = trace.velocity[:, 1:]
                        if condition == 'correct':
                            correct_velocity, correct_loss = velocity.clone(), float(loss)
                        row = dict(arm=request['arm'], absolute_index=selected['absolute_index'],
                            episode_id=selected['episode_id'], window_index=selected['window_index'],
                            sigma=sigma, noise_seed=seed, condition=condition, flow_loss=float(loss),
                            loss_minus_correct=float(loss)-correct_loss, velocity_rms=previous._rms(velocity),
                            velocity_delta_vs_correct_rms=previous._rms(velocity-correct_velocity),
                            target_rms=previous._rms(batch['target_flow'][:, 1:].bfloat16()),
                            condition_action_sha256=previous._tensor_digest(local['actions']),
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
            arm=request['arm'], lineage=lineage, selected_windows=request['selected_windows'],
            selection_source=request['selection_source'], selection_source_sha256=request['selection_source_sha256'],
            actual_model_forward_calls=len(records), comparisons=comparisons,
            metrics_path=str(directory/'diagnostic-metrics.jsonl'),
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
