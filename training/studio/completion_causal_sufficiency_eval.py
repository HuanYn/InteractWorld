"""Fixed TF length validation and real causal rollout, kept physically separate.

Validation alone may read all GT latents.  The generation path calls only the
old, audited z0-only extractor and never receives a validation tensor.  Neither
clean-history MSE nor completion is a visual/control or realtime certificate.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.studio import completion_causal_eval as engine
from training.studio import completion_causal_sufficiency_train as training

ROOT = training.ROOT
SOURCE_REVISION = training.SOURCE_REVISION
REVISION = 'tf20-40-80-160-fixed-supervised32-and-z0-causal-pairs-v2'
MODE = 'completion_causal_sufficiency_eval'
VALIDATE_MODE = 'completion_causal_sufficiency_validate'
MODES = (MODE, VALIDATE_MODE)
ENDPOINTS = (20, 40, 80, 160)
CASES = engine.CASES
VALIDATION_CASES = ('pair-j', 'pair-l', 'holdout-j', 'holdout-l')
FIXED = copy.deepcopy(engine.baseline.FIXED)
CONDITIONS = engine.CONDITIONS
SIGMAS = (.2, .4, .6, .8)
VALIDATION_NOISE_SEEDS = (170071, 170093)
VALIDATION_NAMESPACE = 'completion_causal_fixed_validation32_seed170071_170093_v2'
MAX_SECONDS = 1200
CAP, BYTES = training.CAP, training.BYTES


def job(mode, endpoint_step, case_id=None):
    require(type(endpoint_step) is int and endpoint_step in ENDPOINTS, 'fixed20/40/80/160 length comparison required')
    require(mode in MODES, 'known sufficiency validation/evaluation mode required')
    if mode == VALIDATE_MODE:
        require(case_id is None, 'supervised validation always includes all four fixed windows')
        return f'ccs-val-{endpoint_step}-v2'
    require(endpoint_step != 20 and case_id in CASES, 'TF20 generation is reused, not regenerated; fixed two cases only')
    return f'ccs-eval-{endpoint_step}-{case_id}-v2'


def source_binding():
    import train_completion_causal_tf_sufficiency as entry
    from training import causal_tf
    return {key: sha256(path) for key, path in dict(wrapper=__file__, entry=entry.__file__,
        v1_entry=entry.previous.__file__, generation_engine=engine.__file__,
        baseline_inputs=engine.baseline.__file__, clean_history_loss=causal_tf.__file__).items()}


def fixed_fields(mode, endpoint_step, case_id=None):
    identity = job(mode, endpoint_step, case_id)
    p = training.paths(endpoint_step)
    fields = dict(mode=mode, experiment_revision=REVISION, source_revision=SOURCE_REVISION,
        job_id=identity, endpoint_step=endpoint_step, checkpoint_step=endpoint_step,
        checkpoint=str(p['checkpoint']), training_config=str(p['config']), endpoint_job=str(p['endpoint']),
        max_seconds=MAX_SECONDS, cleanup_seconds=60, protocol_cap_gpu_seconds=CAP,
        protocol_incremental_bytes=BYTES, headroom_bytes=BYTES, parent_teacher_qualified=False,
        reference_enabled=False, repeated_development_qualification=True,
        independent_final_test=False, training_episode_exclusion=True)
    if mode == MODE:
        episode, start, target, role = FIXED[case_id]
        fields.update(case_id=case_id, episode_id=episode, source_start=start, target=target,
            candidate_role=role, conditions=list(CONDITIONS), seed=42, steps=40, cfg_scale=5,
            solver='flow_euler', timestep_shift=5., frames=97, future_frames=96, latent_blocks=8,
            latent_frames_per_block=3, fps=16, width=832, height=480, cfg_cache_offload=False,
            lora_inference_dtype='float32', lora_strength=1,
            phase_order='generate_two_then_release_model_then_decode_two',
            evaluation_view='original_unmirrored', evaluation_action_swap='JL_only',
            future_clean_latents_read=False, future_clean_latents_supplied=False)
    else:
        fields.update(validation_cases=list(VALIDATION_CASES), requested_sigmas=list(SIGMAS),
            validation_noise_seeds=list(VALIDATION_NOISE_SEEDS), validation_namespace=VALIDATION_NAMESPACE,
            forwards=32, objective='correct_action_future_flow_MSE_clean_completed_history',
            cfg_scale=None, generator_called=False, future_GT_used_only_for_supervised_metric=True,
            first_latent_excluded_from_MSE=True, no_parameter_update=True)
    return fields


def _digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_request(request):
    expected = fixed_fields(request.get('mode'), request.get('endpoint_step'), request.get('case_id'))
    extra = {'checkpoint_stat', 'training_config_sha256', 'endpoint_binding', 'stage_binding',
        'schema_bridge', 'source_binding', 'schedule_receipt', 'schedule_receipt_sha256',
        'remaining_planned_gpu_seconds', 'budget_contract', 'budget_preparation'}
    if request['mode'] == MODE:
        extra |= {'baseline_binding', 'baseline_request'}
    else:
        extra |= {'validation_binding'}
    require(set(request) == set(expected) | extra
        and all(request.get(k) == v and type(request.get(k)) is type(v) for k, v in expected.items()),
        'fixed sufficiency endpoint/solver/validation contract changed')
    require(request['source_binding'] == source_binding() and request['stage_binding'] == training.stage_binding(),
        'actual sufficiency evaluation source or stage binding changed')
    stat = request['checkpoint_stat']
    require(isinstance(stat, dict) and set(stat) == {'bytes', 'mtime_ns'}
        and all(type(v) is int and v > 0 for v in stat.values()), 'actual complete checkpoint stat required')
    require(_digest(request['training_config_sha256']) and _digest(request['schedule_receipt_sha256']),
        'actual config and finite schedule pins required')
    require(request['schema_bridge'] == training.read(ROOT/'schema-bridge.json'), 'explicit complete-state bridge changed')
    for key in ('endpoint_binding', *(['baseline_binding'] if request['mode'] == MODE else [])):
        require(isinstance(request[key], dict)
            and set(request[key]) == {n+'_sha256' for n in ('request', 'result', 'settlement')}
            and all(_digest(v) for v in request[key].values()), 'settled small-record SHA256 pins required')
    if request['mode'] == MODE:
        require(request['baseline_request'] == str(engine.baseline.ROOT /
            engine.baseline.JOBS['frozen', request['case_id']] / 'request.json'), 'reuse genuine baseline input record only')
    else:
        validate_validation_binding(request['validation_binding'])
        require(request['validation_binding']['training_data_binding'] ==
            request['schema_bridge']['qualified_data_binding'], 'validation training exclusion must bind this same TF trajectory')
        require(request['validation_binding'] == training.read(ROOT/'validation-contract.json'),
            'immutable validation GT/draw contract changed between endpoints')
    training.schedule_binding(request, ROOT/request['job_id'])


def generation_inputs(request, condition):
    """Only this z0-only loader can supply the generator; no supervised values."""
    require(request.get('mode') == MODE, 'supervised validation cannot supply generation inputs')
    require(condition in CONDITIONS, 'fixed original/JL-only condition required')
    original, rows, actual = engine.baseline.baseline_records(request['case_id'], request['baseline_binding'])
    require(actual == request['baseline_binding'], 'same-input baseline records changed')
    values = engine.baseline.load_inputs(original, condition)
    require(values[5] == rows[condition]['lineage']
        and values[5]['future_clean_latents_read'] is False
        and values[5]['future_clean_latents_supplied'] is False, 'generation must read strictly z0 only')
    return values


def validation_draws():
    """Stable exact32 forward identities; no endpoint-dependent random sampling."""
    import torch
    rows = []
    for case_id in VALIDATION_CASES:
        for sigma in SIGMAS:
            actual = float(torch.tensor(sigma, dtype=torch.bfloat16))
            for replicate, seed in enumerate(VALIDATION_NOISE_SEEDS):
                rows.append(dict(index=len(rows), case_id=case_id, sigma_requested=sigma,
                    sigma_bf16=actual, model_timestep=actual*1000,
                    noise_replicate=replicate, noise_seed=seed, condition='correct_original'))
    return rows


def validate_validation_binding(binding):
    from training.studio import balanced_turn_cache as cache
    helper = engine.baseline.inputs_helper
    require(isinstance(binding, dict) and set(binding) == {'schema', 'cache_binding', 'windows', 'draws',
        'draws_sha256', 'training_data_binding', 'training_excluded_episode_ids', 'repeated_development_qualification',
        'independent_final_test', 'future_GT_loader_only'}
        and binding['schema'] == 'completion_causal_fixed_supervised_validation32_v2'
        and binding['draws'] == validation_draws() and binding['draws_sha256'] == object_sha(binding['draws'])
        and binding['repeated_development_qualification'] is True
        and binding['independent_final_test'] is False and binding['future_GT_loader_only'] is True,
        'fixed32 correct validation draw and interpretation contract changed')
    require(set(binding['windows']) == set(VALIDATION_CASES), 'all four validation windows required')
    for case_id, row in binding['windows'].items():
        require(tuple(row[k] for k in ('episode_id', 'source_start', 'target', 'candidate_role')) == FIXED[case_id]
            and row['purpose'] == 'qualification' and row['split'] == 'train'
            and Path(row['shard_path']).is_relative_to(cache.CACHE_ROOT)
            and set(row['shard_stat']) == {'bytes', 'mtime_ns'}
            and _digest(row['supervised_clean_latents_sha256_float32']), 'fixed validation window/tensor pin changed')
    excluded = sorted({value[0] for value in FIXED.values()})
    require(binding['training_excluded_episode_ids'] == excluded, 'three qualification episodes must be excluded')
    # Verify the source/cache/data records at execution, not only caller pins.
    actual = cache.validate_completed_cache('qualification')
    require(helper.cache_binding(actual) == binding['cache_binding'], 'actual qualification cache binding changed')
    for case_id in VALIDATION_CASES:
        row = dict(binding['windows'][case_id])
        row.pop('supervised_clean_latents_sha256_float32')
        require(row == helper.select_window(actual, case_id), 'actual validation window changed')
    import train_completion_causal_tf as entry
    selection = entry.validate_data_binding(binding['training_data_binding'])
    require(not (set(excluded) & {r['episode_id'] for r in selection.rows}), 'validation episode leaked into TF60')


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def extract_supervised(payload, row):
    """Separate full-GT extractor; never called by generation_inputs/generate_clip."""
    import torch
    first, prompt, actions = engine.baseline.extract_condition(payload, row)
    clean = payload['clean_latents'][row['window_index']].detach().float().clone()
    require(tuple(clean.shape) == (25, 48, 30, 52) and bool(torch.isfinite(clean).all())
        and torch.equal(clean[:1], first[0].float()), 'finite exact97 supervised GT geometry required')
    for value, name in ((first, 'first_latent_sha256_float32'), (prompt, 'prompt_sha256_float32'),
        (actions, 'original_action_sha256_float32')):
        require(engine.baseline.matched.tensor_sha(value) == row[name], 'supervised source tensor changed: '+name)
    return clean, prompt, actions


def load_supervised_window(row):
    import torch
    from training.studio import balanced_turn_cache as cache
    shard = contained(row['shard_path'], cache.CACHE_ROOT)
    require(engine.baseline.stat_identity(shard) == row['shard_stat'], 'supervised shard changed')
    payload = torch.load(shard, map_location='cpu', weights_only=True, mmap=True)
    values = extract_supervised(payload, row)
    del payload
    require(engine.baseline.stat_identity(shard) == row['shard_stat'], 'supervised shard changed during read')
    return values


def make_validation_binding(data_binding):
    from training.studio import balanced_turn_cache as cache
    helper = engine.baseline.inputs_helper
    actual = cache.validate_completed_cache('qualification')
    rows = {}
    for case_id in VALIDATION_CASES:
        row = helper.select_window(actual, case_id)
        clean, _, _ = load_supervised_window(row)
        row['supervised_clean_latents_sha256_float32'] = engine.baseline.matched.tensor_sha(clean)
        rows[case_id] = row
    draws = validation_draws()
    result = dict(schema='completion_causal_fixed_supervised_validation32_v2',
        cache_binding=helper.cache_binding(actual), windows=rows, draws=draws, draws_sha256=object_sha(draws),
        training_data_binding=data_binding, training_excluded_episode_ids=sorted({value[0] for value in FIXED.values()}),
        repeated_development_qualification=True, independent_final_test=False, future_GT_loader_only=True)
    validate_validation_binding(result)
    return result


def prepare(*, output, endpoint_step, case_id=None, mode=MODE, max_seconds=MAX_SECONDS,
        schedule_receipt, source_revision=SOURCE_REVISION, **unused):
    require(not unused and source_revision == SOURCE_REVISION and type(max_seconds) is int
        and max_seconds == MAX_SECONDS, 'fixed new source/max1200; no unapproved override')
    request = fixed_fields(mode, endpoint_step, case_id)
    output = contained(output, ROOT)
    require(output == ROOT/request['job_id'] and not output.exists(), 'new explicit formal evaluation identity required')
    _, bridge = training.ensure_stage_records()
    endpoint_request, _, _ = training.completed_endpoint(endpoint_step)
    p = training.paths(endpoint_step)
    receipt = training.read(schedule_receipt)
    require(isinstance(receipt.get('jobs'), list) and receipt['jobs'], 'fixed remaining schedule required')
    remaining = sum(row['max_seconds']+row['cleanup_seconds'] for row in receipt['jobs'][1:])
    request.update(checkpoint_stat=engine.baseline.stat_identity(p['checkpoint']),
        training_config_sha256=endpoint_request['config_sha256'], endpoint_binding=training.record_binding(p['endpoint']),
        stage_binding=training.stage_binding(), schema_bridge=bridge, source_binding=source_binding(),
        schedule_receipt=str(schedule_receipt), schedule_receipt_sha256=sha256(schedule_receipt),
        remaining_planned_gpu_seconds=remaining)
    if mode == MODE:
        request.update(baseline_binding=engine.baseline.baseline_records(case_id)[2],
            baseline_request=str(engine.baseline.ROOT/engine.baseline.JOBS['frozen', case_id]/'request.json'))
        values = [generation_inputs(request, condition) for condition in CONDITIONS]
        engine.baseline.validate_pair_inputs(*values)
        preparation = dict(source_input_lineages={c:v[5] for c,v in zip(CONDITIONS, values)},
            future_clean_latents_read=False, future_clean_latents_supplied=False)
    else:
        request['validation_binding'] = make_validation_binding(bridge['qualified_data_binding'])
        fixed = ROOT/'validation-contract.json'
        if fixed.exists():
            require(training.read(fixed) == request['validation_binding'],
                'fixed validation GT/draw identity changed between training lengths')
        else:
            require(endpoint_step == 20, 'pin the fixed supervised contract at TF20 before later lengths')
            write_json(fixed, request['validation_binding'])
        preparation = dict(validation_binding=request['validation_binding'], future_GT_loader_only=True,
            generated_video=False, generator_called=False)
    contract, evidence = training.validate_combined_budget(request, output)
    request.update(budget_contract=contract, budget_preparation=evidence)
    validate_request(request)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'request.json', request)
    write_json(output/'preparation.json', dict(status='cpu_prepared', weights_loaded=False, gpu_launched=False,
        training_length_adequacy_proven=False, budget_evidence=evidence, **preparation))
    return request


def load_endpoint(request, endpoint_request, endpoint):
    import torch
    import train_completion_causal_tf_sufficiency as entry
    step = request['endpoint_step']
    p = training.paths(step)
    require(engine.baseline.stat_identity(p['checkpoint']) == request['checkpoint_stat']
        and sha256(p['config']) == request['training_config_sha256'], 'actual complete checkpoint/config changed')
    config = entry.previous.load_config(p['config']) if step == 20 else entry.load_config(p['config'])
    data = endpoint_request['data_binding']
    payload = torch.load(p['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    if step == 20:
        entry.previous.validate_checkpoint(payload, config=config,
            hashes=entry.previous.artifact_hashes(config, p['config'], data),
            data_binding=data, sampling=endpoint['sampling_contract'])
        require(payload['stage'] == entry.previous.STAGE and payload['checkpoint_schema'] == entry.previous.CHECKPOINT_SCHEMA
            and payload['step'] == 20 and payload['sampling_cursor'] == 160, 'real original TF20 checkpoint required')
    else:
        bridge = request['schema_bridge']
        entry.validate_checkpoint(payload, config=config,
            hashes=entry.artifact_hashes(config, p['config'], data, bridge), data_binding=data,
            sampling=endpoint['sampling_contract'], bridge_binding=bridge)
        require(all(payload.get(k) == v for k,v in entry.counters(step).items()), 'wrong actual formal step/cursor')
    entry.previous.validate_base_identity(payload, config)
    fields = ('parent_lineage', 'base_identity', 'model_signature', 'frozen_action_signature',
        'sampling_contract', 'method_contract')
    require(all(payload[k] == endpoint[k] for k in fields), 'actual checkpoint and settled endpoint differ')
    lineage = dict(stage=payload['stage'], checkpoint_schema=payload['checkpoint_schema'], step=step,
        checkpoint=str(p['checkpoint']), checkpoint_stat=request['checkpoint_stat'],
        sampling_cursor=payload['sampling_cursor'], micro_batches_consumed=payload['micro_batches_consumed'],
        **{k:payload[k] for k in fields}, schema_migration=None if step == 20 else payload['schema_migration'],
        inference_uses='frozen_parent_Adapter_gate_and_trained_FP32_LoRA', reference_enabled=False,
        parent_teacher_qualified=False, training_length_adequacy_proven=False, realtime=False)
    require(engine.baseline.stat_identity(p['checkpoint']) == request['checkpoint_stat'], 'checkpoint changed during read')
    return payload['frozen_action_model'], payload['trainable_model'], config, lineage


def make_validation_batch(clean, prompt, actions, draw):
    """Fresh local CPU RNG, same noise for a replicate across sigma/endpoints."""
    import torch
    require(draw in validation_draws(), 'only fixed correct validation samples')
    require(tuple(clean.shape) == (25, 48, 30, 52) and tuple(actions.shape) == (96, 8),
        'native97 supervised window required')
    generator = torch.Generator(device='cpu').manual_seed(draw['noise_seed'])
    noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
    sigma = draw['sigma_bf16']
    noisy, target = clean*(1-sigma)+noise*sigma, noise-clean
    noisy[0].copy_(clean[0]); target[0].zero_()
    # Match the FP32 timestep interface used by training. Only the chosen sigma
    # is quantized; rounding 1000*sigma again would mismatch the noising level.
    timestep = torch.full((25,), draw['model_timestep'], dtype=torch.float32)
    timestep[0] = 0
    batch = dict(clean_latents=clean[None], noisy_latents=noisy[None], target_flow=target[None],
        timesteps=timestep[None], prompt_embeds=prompt[None], actions=actions[None])
    return batch, dict(noise_sha256_float32=engine.baseline.matched.tensor_sha(noise),
        clean_latents_sha256_float32=engine.baseline.matched.tensor_sha(clean),
        rng='independent_local_CPU_torch_generator;no_training_rng_state_modified')


def validate_model(model, config, request, *, device, check, progress=None, loss_fn=None):
    """Exactly32 no-grad packed TF forwards; no KV generation or optimizer."""
    import torch
    from training.causal_tf import causal_teacher_forcing_loss
    require(request.get('mode') == VALIDATE_MODE, 'supervised validation request only')
    loss_fn = causal_teacher_forcing_loss if loss_fn is None else loss_fn
    rows = []
    for case_id in VALIDATION_CASES:
        row = request['validation_binding']['windows'][case_id]
        clean, prompt, actions = load_supervised_window(row)
        require(engine.baseline.matched.tensor_sha(clean) == row['supervised_clean_latents_sha256_float32'],
            'actual supervised future GT pin changed')
        for draw in (v for v in validation_draws() if v['case_id'] == case_id):
            check(); tick = time.monotonic()
            batch, identity = make_validation_batch(clean, prompt, actions, draw)
            with torch.no_grad(), engine._autocast(device):
                loss = loss_fn(model, batch, config, device)
            require(bool(torch.isfinite(loss)) and float(loss) >= 0, 'nonfinite supervised validation MSE')
            rows.append(dict(draw, **identity, future_flow_mse=float(loss), seconds=time.monotonic()-tick,
                future_GT_used_only_for_supervised_metric=True, no_parameter_update=True))
            del loss, batch
            if progress is not None: progress(copy.deepcopy(rows))
    require(len(rows) == 32 and [r['index'] for r in rows] == list(range(32)), 'all32 correct forwards required')
    return rows


def _build_model(request, endpoint_request, endpoint, report):
    import torch
    import train_completion_causal_tf as entry
    frozen, state, config, lineage = load_endpoint(request, endpoint_request, endpoint)
    contained(config.model.base_model_path, training.RUNTIME.parent)
    tick = time.monotonic()
    model, _ = entry.build_model(config, torch.device('cpu'), frozen_action_model=frozen, trainable_model=state)
    require(entry.model_signature(model) == lineage['model_signature'] and model.is_causal is True
        and model.uniform_timestep is False and model.model.controlled_action_enabled is True
        and model.model.controlled_action_ratio_cap == .1 and model.model.controlled_action_eps == 1e-6,
        'actual causal controlled FP32 LoRA model contract changed')
    del frozen, state
    gc.collect()
    model.requires_grad_(False).eval()
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device=device)
    lora = [p for n,p in model.named_parameters() if n.endswith(('.lora_a.weight', '.lora_b.weight'))]
    require(lora and all(p.dtype == torch.float32 for p in lora)
        and model.model.controlled_action_gate.dtype == torch.float32, 'actual trained FP32 low bits must be preserved')
    report.update(checkpoint_lineage=lineage, model_load_seconds=time.monotonic()-tick, model_load_count=1)
    return model, config, lineage, device


def run(request, directory, lease):
    import numpy as np
    import torch
    from PIL import Image
    from training.demo.backend import validate_active_authority, validate_lease
    from training.eval.rollout15s import ffmpeg_writer_factory
    from utils.wan_wrapper import WanVAEWrapper
    validate_request(request)
    directory = contained(directory, ROOT)
    require(directory == ROOT/request['job_id'] and training.read(directory/'request.json') == request
        and not any((directory/n).exists() for n in ('progress.json', 'validation.json', *CONDITIONS)),
        'new prepared formal endpoint job required')
    require(lease.get('gpu_index') == 2, 'formal chain fixed to original Creator GPU2')
    validate_active_authority(lease)
    validate_lease(lease, max_seconds=MAX_SECONDS, job_id=request['job_id'], request_sha256=sha256(directory/'request.json'))
    contract, budget = training.validate_combined_budget(request, directory, lease)
    require(contract == request['budget_contract'], 'actual family charges changed since preparation')
    write_json(directory/'budget-at-run.json', budget)
    require(os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'offline existing assets only')
    began, model, vae, generated, results = time.monotonic(), None, None, {}, []
    report = dict(status='running', mode=request['mode'], revision=REVISION, request=request,
        endpoint_step=request['endpoint_step'], cases=results, budget_evidence=budget,
        quality_evaluation='awaiting_actual_visual_review', quality_improvement_verified=False,
        action_correctness_verified=False, training_length_adequacy_proven=False,
        realtime=False, automatic_next_stage=False, future_GT_used_only_for_supervised_metric=request['mode']==VALIDATE_MODE)
    def check():
        validate_active_authority(lease)
        require(time.monotonic()-began < MAX_SECONDS-15, 'finite1200-second formal endpoint deadline reached')
    try:
        check()
        endpoint_request, endpoint, _ = training.completed_endpoint(request['endpoint_step'])
        require(training.record_binding(training.paths(request['endpoint_step'])['endpoint']) == request['endpoint_binding'],
            'settled formal endpoint records changed')
        if request['mode'] == MODE:
            values = {c:generation_inputs(request, c) for c in CONDITIONS}
            engine.baseline.validate_pair_inputs(values['original'], values['jl_swap'])
        else:
            validate_validation_binding(request['validation_binding'])
        model, config, lineage, device = _build_model(request, endpoint_request, endpoint, report)
        if request['mode'] == VALIDATE_MODE:
            def progress(rows):
                write_json(directory/'progress.json', dict(report, phase='supervised_validation', rows=rows,
                    completed_forwards=len(rows), elapsed_seconds=time.monotonic()-began))
            rows = validate_model(model, config, request, device=device, check=check, progress=progress)
            means = {case:sum(r['future_flow_mse'] for r in rows if r['case_id']==case)/8 for case in VALIDATION_CASES}
            per_sigma = {str(sigma):sum(r['future_flow_mse'] for r in rows if r['sigma_requested']==sigma)/8 for sigma in SIGMAS}
            report.update(status='completed', rows=rows, actual_model_forward_calls=32,
                future_flow_mse=sum(means.values())/4, per_window_mean=means, per_sigma_mean=per_sigma,
                supervised_scope='clean_history_correct_action_future_flow_MSE;not_self_rollout_or_counterfactual_GT',
                generator_called=False, optimizer_steps=0, generated_video=False,
                validation_binding=request['validation_binding'], no_parameter_update=True)
        else:
            require(Path(values['original'][5]['negative_source']['source_model']).name == Path(config.model.base_model_path).name,
                'text negative and causal model base differ')
            for condition in CONDITIONS:
                check()
                first, positive, actions, initial, negative, input_lineage = values[condition]
                target = directory/condition; target.mkdir(exist_ok=False)
                Image.fromarray(initial).save(target/'initial.png')
                np.save(target/'actions.npy', actions, allow_pickle=False)
                write_json(target/'inputs.json', dict(condition=condition, source_input_lineage=input_lineage,
                    checkpoint_lineage=lineage, actual_actions=actions.tolist(), seed=42,
                    future_clean_latents_read=False, future_clean_latents_supplied=False))
                def progress(rows):
                    write_json(directory/'progress.json', dict(report, phase='generating', active_condition=condition,
                        current_blocks=rows, elapsed_seconds=time.monotonic()-began))
                latents, generation = engine.generate_clip(model, config, first, positive, negative, actions,
                    device=device, check=check, progress=progress)
                generated[condition] = (latents, generation)
                write_json(target/'latent-generation.json', generation)
            report['generation_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            require(report['generation_peak_reserved_bytes'] < 23*1024**3, 'resident KV generation exceeded23GiB')
            model.to('cpu'); model = None; gc.collect(); torch.cuda.empty_cache(); check()
            tick = time.monotonic()
            vae = WanVAEWrapper(vae_type='Wan2.2_VAE', pretrained_path=str(Path(config.model.base_model_path)/'Wan2.2_VAE.pth'))
            vae.requires_grad_(False).eval().to(device=device, dtype=torch.bfloat16)
            report['vae_load_seconds'] = time.monotonic()-tick
            for condition in CONDITIONS:
                check(); latents, generation = generated.pop(condition)
                target = directory/condition
                writer = ffmpeg_writer_factory(target/'raw.mp4', 832, 480, 16)
                decoded = engine.decode_clip(vae, latents, values[condition][3], device=device, check=check, writer=writer)
                del latents
                row = dict(status='completed', mode=MODE, revision=REVISION, endpoint_step=request['endpoint_step'],
                    case_id=request['case_id'], condition=condition, lineage=values[condition][5],
                    checkpoint_lineage=lineage, generation=generation, **decoded,
                    actual_model_forward_calls=generation['actual_model_forward_calls'], raw_video=str(target/'raw.mp4'),
                    quality_evaluation='awaiting_actual_visual_review', quality_improvement_verified=False,
                    action_correctness_verified=False, future_clean_latents_read=False,
                    future_clean_latents_supplied=False, realtime=False)
                results.append(row); write_json(target/'result.json', row)
            require(len(results)==2 and sum(r['actual_model_forward_calls'] for r in results)==1316,
                'two original/JLswap real causal clips required')
            require(sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()) < 100_000_000,
                'bounded paired output exceeded100MB')
            report.update(status='completed', actual_model_forward_calls=1316, denoising_forward_calls=1280,
                context_forward_calls=36, reference_enabled=False, cfg_cache_offload=False,
                phase_order=request['phase_order'], generated_latents_saved=False,
                future_clean_latents_read=False, future_clean_latents_supplied=False,
                streaming_latency_measured=False, sustained_streaming_fps_measured=False)
        require(torch.cuda.max_memory_reserved() < 23*1024**3, 'formal validation/evaluation exceeded23GiB')
        check()
        return report
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', automatic_retry=False)
        write_json(directory/'failure.json', dict(report, elapsed_seconds=time.monotonic()-began))
        raise
    finally:
        generated.clear()
        for item in (model, vae):
            if item is not None:
                try: item.to('cpu')
                except Exception as error: report.setdefault('cleanup_errors', []).append(repr(error))
        if torch.cuda.is_initialized():
            report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            torch.cuda.empty_cache()
        report['elapsed_seconds'] = time.monotonic()-began
        write_json(directory/('validation.json' if request['mode']==VALIDATE_MODE else 'evaluation.json'), report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--endpoint-step', type=int, choices=ENDPOINTS, required=True)
    parser.add_argument('--case-id', choices=CASES)
    parser.add_argument('--mode', choices=MODES, default=MODE)
    parser.add_argument('--schedule-receipt', required=True)
    parser.add_argument('--source-revision', default=SOURCE_REVISION)
    parser.add_argument('--max-seconds', type=int, default=MAX_SECONDS)
    print(json.dumps(prepare(**vars(parser.parse_args()))))
