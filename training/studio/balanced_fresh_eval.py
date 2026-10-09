"""Fixed four-window round20/fresh60 qualification, two conditions per load.

Dataset split and evaluation purpose are deliberately separate. Both endpoints
are validated using their genuine training schema; no future clean latent is
materialized by this evaluator and no visual-success flag is set by generation.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import re
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.studio import balanced_turn_cache as cache
from training.studio import controlled_matched_window_eval as matched
from training.studio import controlled_lora_pair_eval as paired
from training.studio import master_writeback_eval as master
from training.studio.quality_probe import QualityRuntime, capture_lora_scales

ROOT = cache.BASE / 'teacher-balanced-fresh60-v1'
MODE = 'balanced_fresh_eval'
REVISION = 'balanced-four-window-round20-fresh60-original-jlswap-v1'
SOURCE_REVISION = 'creator-balanced-fresh60-v1'
STAGE = 'controlled_action_balanced_fresh60_native97_v1'
SCHEMA = 'controlled_action_balanced_fresh60_checkpoint_v1'
ARMS = ('round20', 'fresh60')
CONDITIONS = ('original', 'jl_swap')
# Lookup by immutable role/id, never cache row position (cache order is lexical).
FIXED = {
    'pair-j': ('063168aac541196a82d74a399c633dc6', 450, 'J', 'qualification_pair'),
    'pair-l': ('063168aac541196a82d74a399c633dc6', 1050, 'L', 'qualification_pair'),
    'holdout-j': ('00afe1074cb9437f5fc87870dec48263', 720, 'J', 'holdout_candidate'),
    'holdout-l': ('06737e69c9479b803087f53be0485f12', 90, 'L', 'holdout_candidate'),
}
JOBS = {(arm, case): f'cbf-{arm}-{case}-v1' for arm in ARMS for case in FIXED}
read = master.read


def paths(arm):
    require(arm in ARMS, 'fixed round20/fresh60 endpoint required')
    if arm == 'round20':
        return master.paths('round')
    profile = ROOT / 'cbf-profile2-v1'
    return dict(checkpoint=profile / 'training/checkpoints/step-0000060.pt',
                config=profile / 'training.yaml', endpoint=ROOT / 'cbf-to60-v1')


def completed_endpoint(arm):
    if arm == 'round20':
        request, result = master.completed_endpoint('round')
        return request, result['execution']
    require(arm == 'fresh60', 'fixed endpoint required')
    from training.studio import balanced_fresh_train as training
    request, execution, settlement = training.completed_endpoint(step=60)
    p = paths(arm)
    require(settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed'
            and execution.get('stage') == STAGE and execution.get('checkpoint_schema') == SCHEMA
            and execution.get('step') == execution.get('optimizer_internal_step') == 60
            and execution.get('micro_batches_consumed') == execution.get('new_stage_micro_batches_consumed')
            == execution.get('lifetime_micro_batches_consumed') == execution.get('sampling_cursor') == 480
            and execution.get('historical_micro_batches_before_stage') == 0
            and execution.get('checkpoint_path') == str(p['checkpoint'])
            and request.get('config') == str(p['config']) and request.get('stop_after_step') == 60
            and request.get('config_sha256') == sha256(p['config']) and p['checkpoint'].is_file(),
            'completed settled genuinely fresh60/new480 endpoint required')
    return request, execution


def record_binding(directory):
    return {name + '_sha256': sha256(Path(directory) / (name + '.json'))
            for name in ('request', 'result', 'settlement')}


def select_window(binding, case_id):
    require(case_id in FIXED and binding.get('schema') == cache.SCHEMA
            and binding.get('purpose') == 'qualification' and binding.get('original_split') == 'train'
            and binding.get('qualification_never_training') is True,
            'qualification purpose must not be confused with original dataset train split')
    rows = binding['windows']
    identities = [(r['episode_id'], r['source_start'], r['target'], r['candidate_role']) for r in rows]
    require(len(rows) == 4 and set(identities) == set(FIXED.values())
            and all(r['purpose'] == 'qualification' and r['split'] == 'train' for r in rows),
            'all four preselected windows/roles required; no replacements or training windows')
    row = copy.deepcopy(rows[identities.index(FIXED[case_id])])
    require(row['annotation_cache_actions_equal'] is True
            and row['view_id'] == cache.view_id(row), 'actual cached window identity/labels differ')
    return row


def cache_binding(binding):
    return {k: binding[k] for k in ('schema', 'purpose', 'binding', 'cache_job',
                                    'cache_result_sha256', 'original_split', 'qualification_never_training')}


def negative_binding():
    source = matched._source_evaluation()  # Genuine old input record, not a relabelled new request.
    path = contained(source['negative_cache'], cache.DATA_ROOT)
    receipt = path.with_suffix('.json')
    require(sha256(path) == source['negative_cache_sha256']
            and sha256(receipt) == source['negative_receipt_sha256'], 'unchanged original negative text required')
    return dict(source_request=str(matched.SOURCE_EVAL), source_request_sha256=sha256(matched.SOURCE_EVAL),
                path=str(path), sha256=source['negative_cache_sha256'],
                receipt=str(receipt), receipt_sha256=source['negative_receipt_sha256'])


def fixed_fields(arm, case_id):
    require((arm, case_id) in JOBS, 'only fixed two endpoints by four qualification identities')
    p = paths(arm)
    identity, start, direction, role = FIXED[case_id]
    return dict(mode=MODE, experiment_revision=REVISION, source_revision=SOURCE_REVISION,
        job_id=JOBS[arm, case_id], arm=arm, case_id=case_id, episode_id=identity, source_start=start,
        target=direction, candidate_role=role, purpose='qualification', original_split='train',
        conditions=list(CONDITIONS), checkpoint=str(p['checkpoint']), training_config=str(p['config']),
        checkpoint_stage=master.STAGE if arm == 'round20' else STAGE,
        checkpoint_schema=master.SCHEMA if arm == 'round20' else SCHEMA,
        checkpoint_step=20 if arm == 'round20' else 60, endpoint_job=str(p['endpoint']),
        seed=42, steps=40, cfg_scale=5, frames=97, future_frames=96, fps=16, width=832, height=480,
        solver='flow_euler', residency='phase', fastpath=False, cleanup_seconds=60,
        evaluation_view='original_unmirrored', evaluation_action_swap='JL_only')


def validate_request(request):
    expected = fixed_fields(request.get('arm'), request.get('case_id'))
    require(all(request.get(k) == v and type(request.get(k)) is type(v) for k, v in expected.items()),
            'fixed identity, genuine checkpoint schema or sampling contract changed')
    require(set(request) == set(expected) | {'max_seconds', 'checkpoint_stat', 'training_config_sha256',
            'endpoint_binding', 'cache_binding', 'negative_binding', 'baseline_binding'}, 'unapproved input override')
    require(type(request['max_seconds']) is int and 60 <= request['max_seconds'] <= 1200, 'finite two-case <=1200 seconds')
    stat = request['checkpoint_stat']
    require(set(stat) == {'bytes', 'mtime_ns'} and all(type(v) is int and v > 0 for v in stat.values()), 'checkpoint stat required')
    require(isinstance(request['training_config_sha256'], str)
            and re.fullmatch('[0-9a-f]{64}', request['training_config_sha256']), 'actual config binding required')
    for name in ('endpoint_binding', 'baseline_binding'):
        value = request[name]
        if name == 'baseline_binding' and request['arm'] == 'round20':
            require(value is None, 'baseline must not pretend to have a prior comparison result')
        else:
            require(set(value) == {n + '_sha256' for n in ('request', 'result', 'settlement')}
                    and all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) for v in value.values()),
                    'completed small record binding required')


def baseline_records(case_id, binding=None):
    directory = ROOT / JOBS['round20', case_id]
    actual = record_binding(directory)
    require(binding is None or actual == binding, 'paired round20 baseline records changed')
    request, result, settlement = (read(directory / (n + '.json')) for n in ('request', 'result', 'settlement'))
    validate_request(request)
    require(request['arm'] == 'round20' and request['case_id'] == case_id and result.get('request') == request
            and result.get('status') == 'completed' and result.get('mode') == MODE
            and result.get('revision') == REVISION and result.get('actual_model_forward_calls') == 160
            and result.get('model_load_count') == 1 and settlement.get('status') == 'settled'
            and settlement.get('outcome') == 'completed', 'actual settled new-input baseline pair required')
    cases = result['cases']
    require([r['condition'] for r in cases] == list(CONDITIONS)
            and all(r['status'] == 'completed' and r['frames_written'] == 97
                    and r['actual_model_forward_calls'] == 80 and r['case_id'] == case_id for r in cases),
            'both completed baseline conditions required')
    return request, {r['condition']: r for r in cases}, actual


def make_request(*, arm, case_id, source_revision=SOURCE_REVISION, max_seconds=1200):
    require(source_revision == SOURCE_REVISION, 'fixed evaluation source revision required')
    request = fixed_fields(arm, case_id)
    endpoint_request, _ = completed_endpoint(arm)
    binding = cache.validate_completed_cache('qualification')
    select_window(binding, case_id)
    p = paths(arm)
    request.update(max_seconds=max_seconds, checkpoint_stat=master.stat_identity(p['checkpoint']),
        training_config_sha256=endpoint_request['config_sha256'], endpoint_binding=record_binding(p['endpoint']),
        cache_binding=cache_binding(binding), negative_binding=negative_binding(),
        baseline_binding=baseline_records(case_id)[2] if arm == 'fresh60' else None)
    validate_request(request)
    return request


def extract_condition(payload, row):
    # The shared extractor indexes [window_index, :1] BEFORE clone/cast/finite checks.
    first, prompt, actions = matched._extract_condition(payload, row)
    require(actions.tolist() == row['future_actions_96x8'], 'actual annotation96 and cached actions differ')
    return first, prompt, actions


def load_inputs(request, condition):
    import numpy as np
    import torch
    validate_request(request)
    require(condition in CONDITIONS, 'fixed original/JL-only swapped condition')
    binding = cache.validate_completed_cache('qualification')
    require(cache_binding(binding) == request['cache_binding'], 'completed qualification cache binding changed')
    row = select_window(binding, request['case_id'])
    shard = contained(row['shard_path'], cache.CACHE_ROOT)
    require(master.stat_identity(shard) == row['shard_stat'], 'cache shard changed after verification')
    payload = torch.load(shard, map_location='cpu', weights_only=True, mmap=True)
    first, prompt, original = extract_condition(payload, row)
    del payload
    for value, name in ((first, 'first_latent_sha256_float32'), (prompt, 'prompt_sha256_float32'),
                        (original, 'original_action_sha256_float32')):
        require(matched.tensor_sha(value) == row[name], 'selected first/text/action changed: ' + name)
    require(master.stat_identity(shard) == row['shard_stat'], 'cache changed during z0 copy')
    actions = original.clone()
    if condition == 'jl_swap':
        actions[:, 5], actions[:, 7] = original[:, 7], original[:, 5]
    initial_path = contained(row['initial_path'], cache.ROOT / 'data/initial')
    require(sha256(initial_path) == row['initial_sha256'], 'display-only first RGB changed')
    initial = np.load(initial_path, allow_pickle=False)
    require(initial.shape == (480, 832, 3) and initial.dtype == np.uint8, 'original unannotated display RGB required')
    require(negative_binding() == request['negative_binding'], 'negative prompt source changed')
    base_name = read(request['negative_binding']['receipt'])['source_model']
    negative, negative_source = matched.load_negative(request['negative_binding']['path'], base_name)
    lineage = dict(case_id=request['case_id'], selection=row, purpose='qualification', original_split='train',
        cache_binding=request['cache_binding'], first_latent_sha256_float32=matched.tensor_sha(first),
        prompt_sha256_float32=matched.tensor_sha(prompt), prompt_text=row['prompt_text'],
        negative_source=negative_source, negative_sha256_float32=matched.tensor_sha(negative),
        original_action_sha256_float32=matched.tensor_sha(original), actual_action_sha256_float32=matched.tensor_sha(actions),
        original_actions=original.tolist(), actual_actions=actions.tolist(), actual_action_condition=condition,
        future_clean_latents_read=False, future_clean_latents_supplied=False, vae_encoding_used=False,
        original_rgb_frames_decoded_by_eval=0, display_rgb_reused_from_cache=True,
        candidate_training_episode_excluded=True, baseline_ancestor_exposure_not_recomputed=True,
        evaluation_view='original_unmirrored', evaluation_action_swap='JL_only',
        independent_official_dev_split=False, generalization_verified=False,
        method_scope='fixed_small_qualification_pool_not_data_only_causal_ablation')
    return first, prompt, actions.numpy(), initial, negative, lineage


def validate_pair_inputs(original, swapped):
    import numpy as np
    import torch
    for index in (0, 1, 4):
        require(torch.equal(original[index], swapped[index]), 'paired z0/static text/negative must be identical')
    require(np.array_equal(original[3], swapped[3]) and np.array_equal(
        original[2][:, [0, 1, 2, 3, 4, 7, 6, 5]], swapped[2]), 'only J/L may change; A/D and first RGB remain unchanged')


def _compare_baseline_inputs(lineage, before):
    require(lineage == before['lineage'], 'same original qualification input and only endpoint may differ')


def prepare(*, output, arm, case_id, source_revision=SOURCE_REVISION, max_seconds=1200):
    output = contained(output, ROOT)
    request = make_request(arm=arm, case_id=case_id, source_revision=source_revision, max_seconds=max_seconds)
    require(output == ROOT / request['job_id'] and not output.exists(), 'new fixed evaluation directory only')
    values = [load_inputs(request, condition) for condition in CONDITIONS]
    validate_pair_inputs(*values)
    if arm == 'fresh60':
        _, before, _ = baseline_records(case_id, request['baseline_binding'])
        for condition, value in zip(CONDITIONS, values):
            _compare_baseline_inputs(value[5], before[condition])
    inputs = {condition: value[5] for condition, value in zip(CONDITIONS, values)}
    del values
    gc.collect()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'request.json', request)
    write_json(output / 'preparation.json', dict(status='cpu_prepared', request=request, inputs=inputs,
        weights_loaded=False, future_clean_latents_read=False, gpu_launched=False, automatic_next_stage=False))
    return request


def _load_checkpoint(request, endpoint):
    import torch
    p = paths(request['arm'])
    require(master.stat_identity(p['checkpoint']) == request['checkpoint_stat']
            and sha256(p['config']) == request['training_config_sha256'], 'actual checkpoint/config changed')
    if request['arm'] == 'round20':
        # This is an internal descriptor of the REAL round checkpoint. We do not
        # send a forged fresh request/payload through an old-schema validator.
        descriptor = dict(arm='round', checkpoint=str(p['checkpoint']),
            checkpoint_stat=request['checkpoint_stat'], training_config_sha256=request['training_config_sha256'])
        return master._load_checkpoint(descriptor, dict(execution=endpoint))
    import train_balanced_fresh_teacher as entry
    config = entry.load_config(p['config'])
    payload = torch.load(p['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    entry.validate_checkpoint(payload, config=config, hashes=entry.artifact_hashes(config, p['config']))
    entry.validate_base_identity(payload, config)
    require(payload['stage'] == STAGE and payload['checkpoint_schema'] == SCHEMA
            and payload['step'] == payload['optimizer_internal_step'] == 60
            and payload['historical_micro_batches_before_stage'] == 0
            and payload['micro_batches_consumed'] == payload['new_stage_micro_batches_consumed']
            == payload['lifetime_micro_batches_consumed'] == payload['sampling_cursor'] == 480,
            'genuine fresh60 checkpoint schema/counters required')
    fields = ('initialization_contract', 'sampling_contract', 'method_contract', 'precision_contract',
              'base_identity', 'model_signature')
    for key in fields:
        require(payload[key] == endpoint[key], 'completed candidate execution metadata differs: ' + key)
    initialization = payload['initialization_contract']
    require(initialization['source_checkpoint'] is None and initialization['start_step'] == 0
            and initialization['historical_micro_batches_before_stage'] == 0
            and initialization['new_optimizer'] is True and initialization['optimizer_restored'] is False,
            'candidate must start fresh, not inherit a prior trained action teacher')
    state = payload['trainable_model']
    require(all(value.dtype == (torch.float32 if name.endswith('controlled_action_gate') else torch.bfloat16)
                for name, value in state.items()), 'candidate saved BF16 Adapter/FP32 gate export required')
    lineage = dict(stage=STAGE, checkpoint_schema=SCHEMA, step=60, checkpoint=str(p['checkpoint']),
        checkpoint_stat=request['checkpoint_stat'], **{key: payload[key] for key in fields},
        optimizer_internal_step=60, new_stage_micro_batches_consumed=480,
        historical_micro_batches_before_stage=0, lifetime_micro_batches_consumed=480, sampling_cursor=480,
        inference_uses='genuine_fresh60_validated_BF16_model_export', trained_lora=False,
        reference_memory_implemented=False, complete_abot_reproduction=False)
    require(master.stat_identity(p['checkpoint']) == request['checkpoint_stat'], 'candidate changed during load')
    return state, config, lineage


class FreshRuntime(QualityRuntime):
    """Use the genuine 60-step model constructor, not an old20 config projection."""
    def __init__(self, state, config, negative):
        import torch
        import train_balanced_fresh_teacher as entry
        from training.models.lora import load_trainable_state_dict
        from utils.controlled_action import controlled_action_contract
        from utils.wan_wrapper import WanVAEWrapper
        self.torch, self.config, self.residency = torch, copy.deepcopy(config.model), 'phase'
        self.model = self.vae = None
        self.lora_scales, self.negative, self.forward_calls = [], negative, 0
        try:
            self.model, summary = entry._build_model(config, torch.device('cpu'))
            require(self.model.is_causal is False and self.model.uniform_timestep is True, 'bidirectional fresh teacher required')
            expected = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
            require(bool(expected) and set(expected) == set(state), 'candidate export parameter names differ')
            for name, param in expected.items():
                require(name == 'model.controlled_action_gate' or name.startswith('model.act_control_adapter.'),
                        'candidate export may contain only Adapter/gate')
                value = state[name]
                require(value.shape == param.shape and value.dtype == param.dtype and bool(torch.isfinite(value).all()),
                        'candidate export tensor differs: ' + name)
            load_trainable_state_dict(self.model, state)
            lora = [(n, p) for n, p in self.model.named_parameters() if n.endswith(('.lora_a.weight', '.lora_b.weight'))]
            require(bool(lora) and all(p.dtype == torch.float32 and not p.requires_grad for _, p in lora)
                    and all(not bool(torch.count_nonzero(p)) for n, p in lora if n.endswith('.lora_b.weight')),
                    'fresh frozen FP32 LoRA must have zero B residual')
            require(getattr(self.model.model, 'controlled_action_enabled', False) is True
                    and self.model.model.controlled_action_ratio_cap == .1
                    and self.model.model.controlled_action_eps == 1e-6 and self.config.action_scale == 1.,
                    'training/inference controlled residual must be identical')
            require(not any(bool(getattr(self.model.model, flag, False)) for flag in
                    ('enable_teacache', 'use_teacache', 'enable_fbcache', 'use_fbcache')), 'approximate caches disabled')
            self.controlled_action_receipt = controlled_action_contract(self.model.model)
            self.model.requires_grad_(False).eval()
            self.trainable_parameters = summary.trainable_parameters
            self.vae = WanVAEWrapper(pretrained_path=str(Path(self.config.base_model_path) / 'Wan2.2_VAE.pth'))
            self.vae.requires_grad_(False).eval().to(dtype=torch.bfloat16)
            self.lora_scales = capture_lora_scales(self.model)
            self.configure(dict(cfg_scale=5, lora_strength=1, action_scale=1, solver='flow_euler'))
        except BaseException:
            self.close()
            raise


def validate_generation(generation, calls, lineage, previous=None):
    stages = generation['stage_totals_seconds']
    require(calls == 80 and generation['frames_written'] == 97
            and generation['first_latent_sha256_float32'] == lineage['first_latent_sha256_float32']
            and 'cached_z0_condition' in stages and 'encode_initial' not in stages and 'encode_endpoint' not in stages,
            'one cached z0/97 frames/40 CFG pairs without VAE encoding required')
    require(re.fullmatch('[0-9a-f]{64}', generation['noise_sha256_float32']), 'actual sampled noise identity required')
    if previous is not None:
        for key in ('first_latent_sha256_float32', 'noise_sha256_float32'):
            require(generation[key] == previous['generation'][key], 'paired actual noise/z0 differs: ' + key)


def run(request, directory, lease):
    import numpy as np
    import torch
    from PIL import Image
    from training.demo.backend import validate_active_authority, validate_lease
    from training.eval.rollout15s import ffmpeg_writer_factory
    validate_request(request)
    directory = contained(directory, ROOT)
    require(directory == ROOT / request['job_id'] and read(directory / 'request.json') == request
            and not any((directory / name).exists() for name in ('progress.json', *CONDITIONS)), 'new prepared job only')
    require(lease['gpu_index'] in (1, 2), 'only original GPU1/2 authority; GPU0 grant has ended')
    validate_lease(lease, max_seconds=request['max_seconds'], job_id=request['job_id'], request_sha256=sha256(directory / 'request.json'))
    require(os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'offline assets required')
    began, runtime, engine = time.monotonic(), None, None
    results, peak_reserved = [], 0

    def check(minimum_remaining=30):
        validate_active_authority(lease)
        require(request['max_seconds'] - (time.monotonic() - began) >= minimum_remaining, 'finite two-case deadline')
        require(matched.directory_bytes(directory) < 40_000_000, 'two-case40MB output limit')

    try:
        check()
        _, endpoint = completed_endpoint(request['arm'])
        require(record_binding(paths(request['arm'])['endpoint']) == request['endpoint_binding'], 'endpoint records changed')
        values = {condition: load_inputs(request, condition) for condition in CONDITIONS}
        validate_pair_inputs(values['original'], values['jl_swap'])
        before = baseline_records(request['case_id'], request['baseline_binding'])[1] if request['arm'] == 'fresh60' else None
        if before is not None:
            for condition in CONDITIONS:
                _compare_baseline_inputs(values[condition][5], before[condition])
        state, config, checkpoint_lineage = _load_checkpoint(request, endpoint)
        require(Path(values['original'][5]['negative_source']['source_model']).name == Path(config.model.base_model_path).name,
                'negative text base model must match actual endpoint')
        tick = time.monotonic()
        runtime = (paired.ControlledPairRuntime(state, config, values['original'][4], arm='frozen')
                   if request['arm'] == 'round20' else FreshRuntime(state, config, values['original'][4]))
        state.clear(); gc.collect()
        load_seconds = time.monotonic() - tick
        engine = matched.CachedInitialEngine({}, runtime.config, residency='phase', fastpath=False, _runtime=runtime,
            cached_first=values['original'][0],
            _writer_factory=lambda *args: matched.BoundedPreservingWriter(ffmpeg_writer_factory(*args)))
        engine.load_seconds = load_seconds
        for condition in CONDITIONS:
            check(270)
            first, prompt, actions, initial, negative, lineage = values[condition]
            target = directory / condition
            target.mkdir(exist_ok=False)
            inputs = dict(request=request, arm=request['arm'], case_id=request['case_id'], condition=condition,
                lineage=lineage, checkpoint_lineage=checkpoint_lineage, actual_actions=actions.tolist(), seed=42,
                future_clean_latents_read=False, future_clean_latents_supplied=False)
            Image.fromarray(initial).save(target / 'initial.png')
            np.save(target / 'actions.npy', actions, allow_pickle=False)
            write_json(target / 'inputs.json', inputs)
            calls_before = runtime.forward_calls
            generation, endpoint_rgb = engine.run_segment(initial=initial, prompt=prompt, actions=actions.copy(),
                seed=42, output_path=target / 'raw.mp4', steps=40)
            del endpoint_rgb
            calls = runtime.forward_calls - calls_before
            validate_generation(generation, calls, lineage, before[condition] if before else None)
            if results:
                validate_generation(generation, calls, lineage, results[0])
            generation.update(initial_condition='fixed_qualification_cached_z0;RGB0_display_only',
                continuation='independent_seed42_noise_and_z0;no_future_ground_truth', vae_encoding_used=False)
            generation['sampler'].update(cfg='text_classifier_free', cfg_scale=5, model_forwards_per_step=2)
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
            row = dict(status='completed', mode=MODE, revision=REVISION, arm=request['arm'], case_id=request['case_id'],
                target=request['target'], condition=condition, lineage=lineage, checkpoint_lineage=checkpoint_lineage,
                generation=generation, frames_written=97, raw_video=str(target / 'raw.mp4'),
                actual_model_forward_calls=calls, cumulative_model_forward_calls=runtime.forward_calls,
                quality_evaluation='awaiting_actual_visual_review', quality_improvement_verified=False,
                action_correctness_verified=False, generalization_verified=False, automatic_next_stage=False)
            results.append(row)
            write_json(target / 'result.json', row)
            write_json(directory / 'progress.json', dict(status='running', cases=results, elapsed_seconds=time.monotonic()-began))
        require(len(results) == 2 and runtime.forward_calls == 160 and peak_reserved < 23*1024**3,
                'two actual conditions/160 forwards/below23GiB required')
        check()
        return dict(status='completed', mode=MODE, revision=REVISION, request=request, arm=request['arm'], case_id=request['case_id'],
            cases=results, actual_model_forward_calls=160, model_load_count=1, load_seconds=load_seconds,
            checkpoint_lineage=checkpoint_lineage, elapsed_seconds=time.monotonic()-began, peak_reserved_bytes=peak_reserved,
            future_clean_latents_read=False, future_clean_latents_supplied=False, quality_improvement_verified=False,
            action_correctness_verified=False, generalization_verified=False, reference_memory_implemented=False,
            complete_abot_reproduction=False, data_only_causal_ablation=False, realtime=False, automatic_next_stage=False)
    except BaseException as error:
        write_json(directory / 'failure.json', dict(status='failed', cases=results, error=f'{type(error).__name__}: {error}',
            elapsed_seconds=time.monotonic()-began, automatic_retry=False,
            partial_generation=copy.deepcopy(engine.last_run_report) if engine is not None else None))
        raise
    finally:
        if engine is not None:
            engine.close()
        elif runtime is not None:
            runtime.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--case-id', choices=tuple(FIXED), required=True)
    parser.add_argument('--source-revision', default=SOURCE_REVISION)
    parser.add_argument('--max-seconds', type=int, default=1200)
    args = parser.parse_args()
    print(json.dumps(prepare(output=args.output, arm=args.arm, case_id=args.case_id,
        source_revision=args.source_revision, max_seconds=args.max_seconds), indent=2))
