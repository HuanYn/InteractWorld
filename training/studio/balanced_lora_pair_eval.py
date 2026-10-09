"""Genuine balanced LoRA24 endpoints; original/JL-only pairs on fixed dev windows.

Reuse only stateless input/engine helpers, never forge an old-stage request or
checkpoint schema. Trainable-arm FP32 LoRA is loaded in full and used at strength1.
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
from training.studio import balanced_fresh_eval as inputs_helper
from training.studio import controlled_matched_window_eval as matched
from training.studio.quality_probe import QualityRuntime, capture_lora_scales
from training.studio.balanced_lora_pair_train import profile_job

ROOT = cache.BASE / 'teacher-balanced-lora-pair-v1'
MODE = 'balanced_lora_pair_eval'
REVISION = 'balanced-lora24-fixed-four-window-original-jlswap-v1'
SOURCE_REVISION = 'creator-balanced-lora-pair-v1'
STAGE = 'controlled_action_balanced_lora_pair_native97_v1'
SCHEMA = 'controlled_action_balanced_lora_pair_checkpoint_v1'
ARMS, CONDITIONS = ('frozen', 'trainable'), ('original', 'jl_swap')
FIXED = copy.deepcopy(inputs_helper.FIXED)
JOBS = {(arm, case): f'cbl-{arm}-{case}-v1' for arm in ARMS for case in FIXED}
read = inputs_helper.read
record_binding = inputs_helper.record_binding
stat_identity = inputs_helper.master.stat_identity
extract_condition = inputs_helper.extract_condition
validate_pair_inputs = inputs_helper.validate_pair_inputs
validate_generation = inputs_helper.validate_generation


def paths(arm):
    require(arm in ARMS, 'explicit new frozen/trainable endpoint required')
    root = ROOT / profile_job(arm)
    return dict(checkpoint=root / 'training/checkpoints/step-0000024.pt',
                config=root / 'training.yaml', endpoint=ROOT / f'cbl-{arm}-to24-v1')


def completed_endpoint(arm):
    from training.studio import balanced_lora_pair_train as training
    request, execution, settlement = training.completed_endpoint(arm=arm, step=24)
    p = paths(arm)
    require(settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed'
            and execution.get('stage') == STAGE and execution.get('checkpoint_schema') == SCHEMA
            and execution.get('arm') == arm and execution.get('step') == execution.get('optimizer_internal_step') == 24
            and execution.get('micro_batches_consumed') == execution.get('new_stage_micro_batches_consumed')
            == execution.get('sampling_cursor') == 192 and execution.get('historical_micro_batches_before_stage') == 480
            and execution.get('lifetime_micro_batches_consumed') == 672
            and execution.get('checkpoint_path') == str(p['checkpoint']) and request.get('config') == str(p['config'])
            and request.get('stop_after_step') == 24 and request.get('config_sha256') == sha256(p['config'])
            and p['checkpoint'].is_file(), 'real settled LoRA24 endpoint required')
    return request, execution


def fixed_fields(arm, case_id):
    require((arm, case_id) in JOBS, 'only two new arms/four fixed qualification windows')
    p = paths(arm)
    identity, start, direction, role = FIXED[case_id]
    return dict(mode=MODE, experiment_revision=REVISION, source_revision=SOURCE_REVISION,
        job_id=JOBS[arm, case_id], arm=arm, case_id=case_id, episode_id=identity, source_start=start,
        target=direction, candidate_role=role, purpose='qualification', original_split='train',
        conditions=list(CONDITIONS), checkpoint=str(p['checkpoint']), training_config=str(p['config']),
        checkpoint_stage=STAGE, checkpoint_schema=SCHEMA, checkpoint_step=24, endpoint_job=str(p['endpoint']),
        seed=42, steps=40, cfg_scale=5, frames=97, future_frames=96, fps=16, width=832, height=480,
        solver='flow_euler', residency='phase', fastpath=False, cleanup_seconds=60,
        evaluation_view='original_unmirrored', evaluation_action_swap='JL_only', lora_strength=1,
        lora_inference_dtype='float32', repeated_development_qualification=True)


def validate_request(request):
    expected = fixed_fields(request.get('arm'), request.get('case_id'))
    require(all(request.get(k) == v and type(request.get(k)) is type(v) for k, v in expected.items()),
            'fixed new endpoint/schema/input contract changed')
    require(set(request) == set(expected) | {'max_seconds', 'checkpoint_stat', 'training_config_sha256',
            'endpoint_binding', 'cache_binding', 'negative_binding', 'baseline_binding'}, 'unapproved input override')
    require(type(request['max_seconds']) is int and 60 <= request['max_seconds'] <= 1200, 'finite <=1200s eval required')
    stat = request['checkpoint_stat']
    require(set(stat) == {'bytes', 'mtime_ns'} and all(type(v) is int and v > 0 for v in stat.values()), 'checkpoint stat required')
    require(isinstance(request['training_config_sha256'], str)
            and re.fullmatch('[0-9a-f]{64}', request['training_config_sha256']), 'actual training config identity required')
    for field in ('endpoint_binding', 'baseline_binding'):
        value = request[field]
        if field == 'baseline_binding' and request['arm'] == 'frozen':
            require(value is None, 'frozen arm is the new primary baseline, not historical fresh60 media')
        else:
            require(isinstance(value, dict) and set(value) == {n+'_sha256' for n in ('request', 'result', 'settlement')}
                    and all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) for v in value.values()),
                    'completed small record binding required')


def baseline_records(case_id, binding=None):
    directory = ROOT / JOBS['frozen', case_id]
    actual = record_binding(directory)
    require(binding is None or binding == actual, 'new same-input frozen baseline records changed')
    request, result, settlement = (read(directory/(n+'.json')) for n in ('request', 'result', 'settlement'))
    validate_request(request)
    require(request['arm'] == 'frozen' and request['case_id'] == case_id and result.get('request') == request
            and result.get('status') == 'completed' and result.get('mode') == MODE and result.get('revision') == REVISION
            and result.get('actual_model_forward_calls') == 160 and result.get('model_load_count') == 1
            and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed',
            'genuine new settled frozen two-case baseline required')
    rows = result['cases']
    require([r.get('condition') for r in rows] == list(CONDITIONS)
            and all(r.get('status') == 'completed' and r.get('frames_written') == 97
                    and r.get('actual_model_forward_calls') == 80 and r.get('case_id') == case_id for r in rows),
            'complete original/JL-only frozen pair required')
    return request, {r['condition']:r for r in rows}, actual


def make_request(*, arm, case_id, source_revision=SOURCE_REVISION, max_seconds=1200):
    require(source_revision == SOURCE_REVISION, 'fixed new source revision required')
    request = fixed_fields(arm, case_id)
    endpoint_request, _ = completed_endpoint(arm)
    binding = cache.validate_completed_cache('qualification')
    inputs_helper.select_window(binding, case_id)
    p = paths(arm)
    request.update(max_seconds=max_seconds, checkpoint_stat=stat_identity(p['checkpoint']),
        training_config_sha256=endpoint_request['config_sha256'], endpoint_binding=record_binding(p['endpoint']),
        cache_binding=inputs_helper.cache_binding(binding), negative_binding=inputs_helper.negative_binding(),
        baseline_binding=baseline_records(case_id)[2] if arm == 'trainable' else None)
    validate_request(request)
    return request


def load_inputs(request, condition):
    """Real cached source descriptor; never send new requests to old validators."""
    import numpy as np
    import torch
    validate_request(request)
    require(condition in CONDITIONS, 'fixed condition required')
    binding = cache.validate_completed_cache('qualification')
    require(inputs_helper.cache_binding(binding) == request['cache_binding'], 'qualification cache binding changed')
    row = inputs_helper.select_window(binding, request['case_id'])
    shard = contained(row['shard_path'], cache.CACHE_ROOT)
    require(stat_identity(shard) == row['shard_stat'], 'cache shard stat changed')
    payload = torch.load(shard, map_location='cpu', weights_only=True, mmap=True)
    first, prompt, original = extract_condition(payload, row)  # Index :1 BEFORE copying; future is never read.
    del payload
    for tensor, field in ((first, 'first_latent_sha256_float32'), (prompt, 'prompt_sha256_float32'),
                          (original, 'original_action_sha256_float32')):
        require(matched.tensor_sha(tensor) == row[field], 'actual source identity changed: '+field)
    require(stat_identity(shard) == row['shard_stat'], 'cache changed during z0 copy')
    actions = original.clone()
    if condition == 'jl_swap':
        actions[:, 5], actions[:, 7] = original[:, 7], original[:, 5]
    initial_path = contained(row['initial_path'], cache.ROOT / 'data/initial')
    require(sha256(initial_path) == row['initial_sha256'], 'display-only RGB changed')
    initial = np.load(initial_path, allow_pickle=False)
    require(initial.shape == (480, 832, 3) and initial.dtype == np.uint8, 'original RGB0 display geometry differs')
    require(inputs_helper.negative_binding() == request['negative_binding'], 'actual negative source changed')
    negative, negative_source = matched.load_negative(request['negative_binding']['path'],
        read(request['negative_binding']['receipt'])['source_model'])
    lineage = dict(case_id=request['case_id'], selection=row, purpose='qualification', original_split='train',
        cache_binding=request['cache_binding'], first_latent_sha256_float32=matched.tensor_sha(first),
        prompt_sha256_float32=matched.tensor_sha(prompt), prompt_text=row['prompt_text'],
        negative_source=negative_source, negative_sha256_float32=matched.tensor_sha(negative),
        original_action_sha256_float32=matched.tensor_sha(original), actual_action_sha256_float32=matched.tensor_sha(actions),
        original_actions=original.tolist(), actual_actions=actions.tolist(), actual_action_condition=condition,
        future_clean_latents_read=False, future_clean_latents_supplied=False, vae_encoding_used=False,
        original_rgb_frames_decoded_by_eval=0, display_rgb_reused_from_cache=True,
        candidate_training_episode_excluded=True, baseline_ancestor_exposure_not_recomputed=True,
        evaluation_view='original_unmirrored', evaluation_action_swap='JL_only', independent_official_dev_split=False,
        repeated_development_qualification=True, generalization_verified=False,
        method_scope='matched_new24_frozen_vs_trainable_LoRA;reused_small_development_pool')
    return first, prompt, actions.numpy(), initial, negative, lineage


def prepare(*, output, arm, case_id, source_revision=SOURCE_REVISION, max_seconds=1200):
    output = contained(output, ROOT)
    request = make_request(arm=arm, case_id=case_id, source_revision=source_revision, max_seconds=max_seconds)
    require(output == ROOT/request['job_id'] and not output.exists(), 'new fixed eval directory only')
    values = [load_inputs(request, condition) for condition in CONDITIONS]
    validate_pair_inputs(*values)
    if arm == 'trainable':
        _, baseline, _ = baseline_records(case_id, request['baseline_binding'])
        for condition, value in zip(CONDITIONS, values):
            require(value[5] == baseline[condition]['lineage'], 'new frozen/trainable actual inputs differ')
    lines = {condition:value[5] for condition, value in zip(CONDITIONS, values)}
    del values
    gc.collect()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'request.json', request)
    write_json(output/'preparation.json', dict(status='cpu_prepared', request=request, inputs=lines,
        weights_loaded=False, future_clean_latents_read=False, gpu_launched=False, automatic_next_stage=False))
    return request


def _load_checkpoint(request, endpoint):
    import torch
    import train_balanced_lora_pair as entry
    p, arm = paths(request['arm']), request['arm']
    require(stat_identity(p['checkpoint']) == request['checkpoint_stat']
            and sha256(p['config']) == request['training_config_sha256'], 'actual checkpoint/config changed')
    config = entry.load_config(p['config'])
    payload = torch.load(p['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    entry.validate_checkpoint(payload, config=config, hashes=entry.artifact_hashes(config, p['config']), arm=arm)
    entry.validate_base_identity(payload, config)
    require(payload['stage'] == STAGE and payload['checkpoint_schema'] == SCHEMA and payload['arm'] == arm
            and payload['step'] == payload['optimizer_internal_step'] == 24
            and payload['micro_batches_consumed'] == payload['new_stage_micro_batches_consumed'] == payload['sampling_cursor'] == 192
            and payload['historical_micro_batches_before_stage'] == 480 and payload['lifetime_micro_batches_consumed'] == 672,
            'genuine new24 weights-only-stage endpoint/counters required')
    fields = ('parent_lineage', 'sampling_contract', 'method_contract', 'precision_contract', 'model_signature', 'base_identity')
    require(all(payload[k] == endpoint[k] for k in fields), 'completed endpoint metadata differs from actual checkpoint')
    lineage = dict(stage=STAGE, checkpoint_schema=SCHEMA, arm=arm, step=24, checkpoint=str(p['checkpoint']),
        checkpoint_stat=request['checkpoint_stat'], **{k:payload[k] for k in fields}, optimizer_internal_step=24,
        new_stage_micro_batches_consumed=192, historical_micro_batches_before_stage=480,
        lifetime_micro_batches_consumed=672, sampling_cursor=192,
        inference_uses='validated_BF16_Adapter_FP32_gate_and_all_trainable_FP32_LoRA_export',
        trained_lora=arm == 'trainable', checkpoint_contains_lora=arm == 'trainable', lora_strength=1,
        lora_inference_dtype='float32', reference_memory_implemented=False, complete_abot_reproduction=False,
        parent_strict_resume=False, strict_resume_only_within_new_arm=True)
    require(stat_identity(p['checkpoint']) == request['checkpoint_stat'], 'checkpoint changed during load')
    return payload['trainable_model'], config, lineage


def load_exact_export(model, state, *, arm):
    """Small testable safety gate: exact keys/geometry/dtypes, including ALL LoRA."""
    import torch
    from training.models.lora import load_trainable_state_dict
    require(arm in ARMS, 'declared LoRA arm required')
    expected = {n:p for n, p in model.named_parameters() if p.requires_grad}
    require(bool(expected) and set(expected) == set(state), 'full actual trainable export required; no missing LoRA')
    for name, parameter in expected.items():
        is_lora = name.endswith(('.lora_a.weight', '.lora_b.weight'))
        is_gate = name == 'model.controlled_action_gate'
        require(is_gate or name.startswith('model.act_control_adapter.') or (arm == 'trainable' and is_lora),
                'unexpected trainable export '+name)
        wanted = torch.float32 if is_gate or is_lora else torch.bfloat16
        tensor = state[name]
        require(torch.is_tensor(tensor) and tensor.shape == parameter.shape and tensor.dtype == parameter.dtype == wanted
                and bool(torch.isfinite(tensor).all()), 'exact export precision/shape required '+name)
    lora = [(n,p) for n,p in model.named_parameters() if n.endswith(('.lora_a.weight', '.lora_b.weight'))]
    require(bool(lora) and all(p.dtype == torch.float32 and p.requires_grad == (arm == 'trainable') for _,p in lora),
            'constructor LoRA trainability/FP32 differs')
    load_trainable_state_dict(model, state)
    require(all(torch.equal(p.detach(), state[n]) for n,p in expected.items()), 'export load changed or omitted parameter values')
    b_nonzero = [bool(torch.count_nonzero(p)) for n,p in lora if n.endswith('.lora_b.weight')]
    require(bool(b_nonzero) and (any(b_nonzero) if arm == 'trainable' else not any(b_nonzero)),
            'trainable must load actual nonzero B; frozen must retain zero B')
    return dict(arm=arm, actual_export_tensors=len(state), loaded_lora_tensors=sum(n in state for n,_ in lora),
        lora_parameter_tensors=len(lora), lora_b_nonzero_tensors=sum(b_nonzero), lora_dtype='float32',
        all_export_values_exactly_loaded=True, lora_strength=1, lora_inference_cast_to_bf16=False)


class PairRuntime(QualityRuntime):
    def __init__(self, state, config, negative, *, arm, checkpoint_lineage):
        import torch
        import train_balanced_lora_pair as entry
        from utils.controlled_action import controlled_action_contract
        from utils.wan_wrapper import WanVAEWrapper
        self.torch, self.config, self.residency = torch, copy.deepcopy(config.model), 'phase'
        self.model = self.vae = None
        self.lora_scales, self.negative, self.forward_calls = [], negative, 0
        try:
            self.model, summary = entry._build_model(config, torch.device('cpu'), arm=arm)
            require(entry.model_signature(self.model, arm) == checkpoint_lineage['model_signature'], 'actual new model signature differs')
            require(self.model.is_causal is False and self.model.uniform_timestep is True, 'BID uniform-timestep teacher required')
            self.export_receipt = load_exact_export(self.model, state, arm=arm)
            require(getattr(self.model.model, 'controlled_action_enabled', False) is True
                    and self.model.model.controlled_action_ratio_cap == .1 and self.model.model.controlled_action_eps == 1e-6
                    and self.config.action_scale == 1., 'training/inference action injection mismatch')
            require(not any(bool(getattr(self.model.model, k, False)) for k in
                    ('enable_teacache', 'use_teacache', 'enable_fbcache', 'use_fbcache')), 'approximate caches disabled')
            self.controlled_action_receipt = controlled_action_contract(self.model.model)
            self.model.requires_grad_(False).eval()  # Do NOT .to(dtype=BF16): trained LoRA must remain FP32.
            self.trainable_parameters = summary.trainable_parameters
            self.vae = WanVAEWrapper(pretrained_path=str(Path(self.config.base_model_path)/'Wan2.2_VAE.pth'))
            self.vae.requires_grad_(False).eval().to(dtype=torch.bfloat16)
            self.lora_scales = capture_lora_scales(self.model)
            self.configure(dict(cfg_scale=5, lora_strength=1, action_scale=1, solver='flow_euler'))
        except BaseException:
            self.close()
            raise

    def generate(self, *args, **kwargs):
        require(all(m.lora_a.weight.dtype == self.torch.float32 and m.lora_b.weight.dtype == self.torch.float32
                    and m.scaling == scale for m,scale in self.lora_scales), 'FP32 LoRA/strength1 changed before inference')
        return super().generate(*args, **kwargs)


def run(request, directory, lease):
    import numpy as np
    import torch
    from PIL import Image
    from training.demo.backend import validate_active_authority, validate_lease
    from training.eval.rollout15s import ffmpeg_writer_factory
    validate_request(request)
    directory = contained(directory, ROOT)
    require(directory == ROOT/request['job_id'] and read(directory/'request.json') == request
            and not any((directory/name).exists() for name in ('progress.json', *CONDITIONS)), 'new prepared job required')
    require(lease['gpu_index'] in (1, 2), 'original GPU1/2 authority only')
    validate_lease(lease, max_seconds=request['max_seconds'], job_id=request['job_id'], request_sha256=sha256(directory/'request.json'))
    require(os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'offline assets required')
    began, runtime, engine, results, peak = time.monotonic(), None, None, [], 0
    def check(minimum=30):
        validate_active_authority(lease)
        require(request['max_seconds']-(time.monotonic()-began) >= minimum, 'finite two-case deadline')
        require(matched.directory_bytes(directory) < 40_000_000, 'two-case40MB job cap')
    try:
        check()
        _, endpoint = completed_endpoint(request['arm'])
        require(record_binding(paths(request['arm'])['endpoint']) == request['endpoint_binding'], 'endpoint records changed')
        values = {c:load_inputs(request,c) for c in CONDITIONS}
        validate_pair_inputs(values['original'], values['jl_swap'])
        before = baseline_records(request['case_id'], request['baseline_binding'])[1] if request['arm'] == 'trainable' else None
        if before is not None:
            require(all(values[c][5] == before[c]['lineage'] for c in CONDITIONS), 'same-input new frozen comparison differs')
        state, config, checkpoint_lineage = _load_checkpoint(request, endpoint)
        require(Path(values['original'][5]['negative_source']['source_model']).name == Path(config.model.base_model_path).name,
                'negative text base must match model')
        tick = time.monotonic()
        runtime = PairRuntime(state, config, values['original'][4], arm=request['arm'], checkpoint_lineage=checkpoint_lineage)
        state.clear(); gc.collect()
        load_seconds = time.monotonic()-tick
        engine = matched.CachedInitialEngine({}, runtime.config, residency='phase', fastpath=False, _runtime=runtime,
            cached_first=values['original'][0], _writer_factory=lambda *args: matched.BoundedPreservingWriter(ffmpeg_writer_factory(*args)))
        engine.load_seconds = load_seconds
        for condition in CONDITIONS:
            check(270)
            first, prompt, actions, initial, negative, lineage = values[condition]
            target = directory/condition
            target.mkdir(exist_ok=False)
            Image.fromarray(initial).save(target/'initial.png')
            np.save(target/'actions.npy', actions, allow_pickle=False)
            write_json(target/'inputs.json', dict(request=request, arm=request['arm'], case_id=request['case_id'], condition=condition,
                lineage=lineage, checkpoint_lineage=checkpoint_lineage, export_receipt=runtime.export_receipt,
                actual_actions=actions.tolist(), seed=42, future_clean_latents_read=False, future_clean_latents_supplied=False))
            calls_before = runtime.forward_calls
            generation, endpoint_rgb = engine.run_segment(initial=initial, prompt=prompt, actions=actions.copy(),
                seed=42, output_path=target/'raw.mp4', steps=40)
            del endpoint_rgb
            calls = runtime.forward_calls-calls_before
            validate_generation(generation, calls, lineage, before[condition] if before else None)
            if results:
                validate_generation(generation, calls, lineage, results[0])
            generation.update(initial_condition='fixed_qualification_cached_z0;RGB0_display_only',
                continuation='independent_seed42_noise_and_z0;no_future_ground_truth', vae_encoding_used=False)
            generation['sampler'].update(cfg='text_classifier_free', cfg_scale=5, model_forwards_per_step=2)
            peak = max(peak, torch.cuda.max_memory_reserved())
            row = dict(status='completed', mode=MODE, revision=REVISION, arm=request['arm'], case_id=request['case_id'],
                target=request['target'], condition=condition, lineage=lineage, checkpoint_lineage=checkpoint_lineage,
                export_receipt=runtime.export_receipt, generation=generation, frames_written=97, raw_video=str(target/'raw.mp4'),
                actual_model_forward_calls=calls, cumulative_model_forward_calls=runtime.forward_calls,
                quality_evaluation='awaiting_actual_visual_review', quality_improvement_verified=False,
                action_correctness_verified=False, generalization_verified=False, automatic_next_stage=False)
            results.append(row)
            write_json(target/'result.json', row)
            write_json(directory/'progress.json', dict(status='running', cases=results, elapsed_seconds=time.monotonic()-began))
        require(len(results) == 2 and runtime.forward_calls == 160 and peak < 23*1024**3, 'two native97 cases/160forwards/<23GiB required')
        check()
        return dict(status='completed', mode=MODE, revision=REVISION, request=request, arm=request['arm'], case_id=request['case_id'],
            cases=results, actual_model_forward_calls=160, model_load_count=1, load_seconds=load_seconds,
            checkpoint_lineage=checkpoint_lineage, export_receipt=runtime.export_receipt, elapsed_seconds=time.monotonic()-began,
            peak_reserved_bytes=peak, future_clean_latents_read=False, future_clean_latents_supplied=False,
            quality_improvement_verified=False, action_correctness_verified=False, generalization_verified=False,
            reference_memory_implemented=False, complete_abot_reproduction=False, repeated_development_qualification=True,
            realtime=False, automatic_next_stage=False)
    except BaseException as error:
        write_json(directory/'failure.json', dict(status='failed', cases=results, error=f'{type(error).__name__}: {error}',
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
