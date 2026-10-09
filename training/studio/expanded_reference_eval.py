"""Fixed new20 expanded60 reference pair; four reused development windows.

Only new arm checkpoints are evaluated. Historical frozen24 records bind inputs
and noise, not the equal-compute no-reference arm or a quality conclusion.
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
from training.studio import balanced_lora_pair_eval as baseline
from training.studio import controlled_matched_window_eval as matched
from training.studio import expanded_reference_train as training
from training.studio import teacher_reference_eval as reference_runtime
from training.studio.quality_probe import capture_lora_scales

ROOT, CAP, BYTES = training.ROOT, training.CAP, training.BYTES
MODE = 'expanded_reference_eval'
REVISION = 'expanded60-reference20-fixed-four-dev-original-jlswap-v1'
SOURCE_REVISION = training.SOURCE_REVISION
STAGE = 'controlled_action_expanded_reference_native97_v1'
SCHEMA = 'controlled_action_expanded_reference_checkpoint_v1'
ARMS, CONDITIONS = training.ARMS, ('original', 'jl_swap')
FIXED = copy.deepcopy(baseline.FIXED)
JOBS = {(arm, case): f'cer-{arm}-{case}-v1' for arm in ARMS for case in FIXED}
read, record_binding, stat_identity = training.read, training.record_binding, baseline.stat_identity
ReferenceCFGFlow = reference_runtime.ReferenceCFGFlow


def paths(arm):
    return training.paths(arm, 20)


def fixed_fields(arm, case_id):
    require((arm, case_id) in JOBS, 'only two new arms and four fixed development windows')
    p = paths(arm)
    episode, start, direction, role = FIXED[case_id]
    return dict(mode=MODE, experiment_revision=REVISION, source_revision=SOURCE_REVISION,
        job_id=JOBS[arm, case_id], arm=arm, case_id=case_id, episode_id=episode, source_start=start,
        target=direction, candidate_role=role, purpose='qualification', original_split='train',
        conditions=list(CONDITIONS), checkpoint=str(p['checkpoint']), training_config=str(p['config']),
        checkpoint_stage=STAGE, checkpoint_schema=SCHEMA, checkpoint_step=20, endpoint_job=str(p['endpoint']),
        seed=42, steps=40, cfg_scale=5, frames=97, future_frames=96, fps=16, width=832, height=480,
        solver='flow_euler', residency='phase', fastpath=False, max_seconds=1200, cleanup_seconds=60,
        reference_enabled=arm == 'reference', lora_strength=1, lora_inference_dtype='float32',
        evaluation_view='original_unmirrored', evaluation_action_swap='JL_only',
        repeated_development_qualification=True, protocol_cap_gpu_seconds=CAP,
        protocol_incremental_bytes=BYTES,
        baseline_request=str(baseline.ROOT/baseline.JOBS['frozen', case_id]/'request.json'))


def validate_request(request):
    expected = fixed_fields(request.get('arm'), request.get('case_id'))
    require(all(request.get(k) == v and type(request.get(k)) is type(v) for k, v in expected.items()),
            'fixed new20 endpoint/reference/input/bounded protocol changed')
    require(set(request) == set(expected) | {'checkpoint_stat', 'training_config_sha256',
        'endpoint_binding', 'baseline_binding', 'reference_cache_binding'}, 'unapproved input override')
    stat = request['checkpoint_stat']
    require(isinstance(stat, dict) and set(stat) == {'bytes', 'mtime_ns'}
            and all(type(v) is int and v > 0 for v in stat.values()), 'actual checkpoint stat required')
    digest = lambda v: isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) is not None
    require(digest(request['training_config_sha256']), 'actual new-arm config identity required')
    for field in ('endpoint_binding', 'baseline_binding'):
        value = request[field]
        require(isinstance(value, dict) and set(value) == {n+'_sha256' for n in ('request', 'result', 'settlement')}
                and all(digest(v) for v in value.values()), 'completed small-record binding required')
    binding = request['reference_cache_binding']
    require(isinstance(binding, dict) and set(binding) == {'path', 'sha256', 'receipt', 'receipt_sha256',
        'manifest', 'manifest_sha256', 'windows'} and type(binding['windows']) is int and binding['windows'] == 4,
        'expanded qualification4 reference binding required, not training60')
    require(binding['path'] == str(ROOT/'data/reference/qualification.references.jsonl')
        and binding['receipt'] == binding['path']+'.receipt.json'
        and binding['manifest'] == str(ROOT/'data/manifests/qualification.reference.jsonl')
        and all(digest(binding[k]) for k in ('sha256', 'receipt_sha256', 'manifest_sha256')),
        'fixed expanded qualification reference locations/pins differ')


def reference_binding():
    from training.studio import expanded_reference_cache as cache
    return cache.validate_completed_cache('qualification')['indexes']['qualification']


def load_reference(request):
    from training.expanded_reference_dataset import ExpandedReferenceStore
    from training.reference_dataset import reference_sample_id
    validate_request(request)
    binding = request['reference_cache_binding']
    require(reference_binding() == binding, 'expanded qualification reference cache changed')
    store = ExpandedReferenceStore(binding, purpose='qualification')
    sample = reference_sample_id(request['episode_id'], request['source_start'])
    refs = store.for_window(request['episode_id'], request['source_start'], batched=True)
    valid = bool(refs['ref_mask'].any())
    require(valid is (request['case_id'] != 'holdout-j'), 'original highway invalid-reference mask must remain false')
    lineage = dict(sample_id=sample, source_start=request['source_start'],
        source_kind='original_raw_initial_frame', cache_binding=binding, row=store.rows[sample],
        reference_enabled=request['reference_enabled'], valid_reference=valid,
        future_rgb_read=False, future_latents_read=False,
        actual_reference_supplied=request['reference_enabled'],
        effective_reference_supplied=request['reference_enabled'] and valid)
    return refs if request['reference_enabled'] else None, lineage


def load_inputs(request, condition):
    """Read the genuine old request, never reinterpret a new request as old schema."""
    validate_request(request)
    require(condition in CONDITIONS, 'fixed original/JL-only condition required')
    original, before, _ = baseline.baseline_records(request['case_id'], request['baseline_binding'])
    values = baseline.load_inputs(original, condition)
    require(values[5] == before[condition]['lineage'], 'original cached input lineage differs from actual frozen baseline')
    return values


def prepare(*, output, arm, case_id, source_revision=SOURCE_REVISION, max_seconds=1200):
    require(source_revision == SOURCE_REVISION and type(max_seconds) is int and max_seconds == 1200,
            'fixed source and1200+60 eval bound required')
    request = fixed_fields(arm, case_id)
    output = contained(output, ROOT)
    require(output == ROOT/request['job_id'] and not output.exists(), 'new fixed arm/case identity required')
    endpoint_request, _, _ = training.completed_endpoint(arm, 20)
    p = paths(arm)
    request.update(checkpoint_stat=stat_identity(p['checkpoint']),
        training_config_sha256=endpoint_request['config_sha256'], endpoint_binding=record_binding(p['endpoint']),
        baseline_binding=baseline.baseline_records(case_id)[2], reference_cache_binding=reference_binding())
    validate_request(request)
    values = [load_inputs(request, condition) for condition in CONDITIONS]
    baseline.validate_pair_inputs(*values)
    _, reference_lineage = load_reference(request)
    lines = {condition: value[5] for condition, value in zip(CONDITIONS, values)}
    del values
    gc.collect()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'request.json', request)
    write_json(output/'preparation.json', dict(status='cpu_prepared', request=request, inputs=lines,
        reference_lineage=reference_lineage, weights_loaded=False, future_clean_latents_read=False,
        gpu_launched=False, teacher_qualified=False, automatic_next_stage=False))
    return request


def load_endpoint(request, endpoint_request, endpoint):
    import torch
    entry = training.entry_module()
    p, arm = paths(request['arm']), request['arm']
    require(stat_identity(p['checkpoint']) == request['checkpoint_stat']
            and sha256(p['config']) == request['training_config_sha256'], 'actual new20 checkpoint/config changed')
    config = entry.load_config(p['config'])
    data, refs = endpoint_request['data_binding'], endpoint_request['reference_binding']
    payload = torch.load(p['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    entry.validate_checkpoint(payload, config=config, arm=arm, data_binding=data, reference_binding=refs,
        hashes=entry.artifact_hashes(config, p['config'], data, refs), sampling=endpoint['sampling_contract'])
    entry.validate_base_identity(payload, config)
    counters = dict(step=20, optimizer_internal_step=20, micro_batches_consumed=160,
        new_stage_micro_batches_consumed=160, sampling_cursor=160,
        historical_micro_batches_before_stage=1312, lifetime_micro_batches_consumed=1472)
    require(payload.get('stage') == STAGE and payload.get('checkpoint_schema') == SCHEMA
        and payload.get('arm') == arm and all(payload.get(k) == v for k, v in counters.items()),
        'genuine new20 same-arm endpoint/history required, not old expanded80 or pilot20')
    fields = ('parent_lineage', 'sampling_contract', 'method_contract', 'precision_contract', 'model_signature', 'base_identity')
    require(all(payload[k] == endpoint[k] for k in fields), 'completed endpoint metadata differs from actual checkpoint')
    lineage = dict(stage=STAGE, checkpoint_schema=SCHEMA, arm=arm, checkpoint=str(p['checkpoint']),
        checkpoint_stat=request['checkpoint_stat'], **counters, **{k: payload[k] for k in fields},
        inference_uses='exact_new20_BF16_Adapter_FP32_gate_frozen_FP32_LoRA_export',
        trained_lora=False, reference_enabled=arm == 'reference', parent_strict_resume=False,
        strict_resume_only_within_new_arm=True, complete_abot_reproduction=False)
    require(stat_identity(p['checkpoint']) == request['checkpoint_stat'], 'checkpoint changed during load')
    return payload['trainable_model'], config, lineage


class ExpandedReferenceRuntime(reference_runtime.PairRuntime):
    """Reuse exact CFG generation, replacing only construction with the new entry."""
    def __init__(self, state, config, negative, *, arm, checkpoint_lineage, references):
        import torch
        from utils.controlled_action import controlled_action_contract
        from utils.wan_wrapper import WanVAEWrapper
        entry = training.entry_module()
        require(arm in ARMS and (references is not None) is (arm == 'reference'), 'training/inference reference arm differs')
        self.torch, self.config, self.residency = torch, copy.deepcopy(config.model), 'phase'
        self.model = self.vae = None
        self.lora_scales, self.negative, self.forward_calls = [], negative, 0
        self.references = references
        try:
            self.model, summary = entry._build_model(config, torch.device('cpu'), arm=arm)
            require(entry.model_signature(self.model, arm) == checkpoint_lineage['model_signature'], 'actual new model signature differs')
            require(self.model.is_causal is False and self.model.uniform_timestep is True, 'BID uniform-timestep teacher required')
            self.export_receipt = reference_runtime.load_exact_export(self.model, state, arm=arm)
            require(getattr(self.model.model, 'controlled_action_enabled', False) is True
                and self.model.model.controlled_action_ratio_cap == .1 and self.model.model.controlled_action_eps == 1e-6
                and self.config.action_scale == 1., 'training/inference action injection differs')
            require(not any(bool(getattr(self.model.model, k, False)) for k in
                ('enable_teacache', 'use_teacache', 'enable_fbcache', 'use_fbcache')), 'approximate caches disabled')
            self.controlled_action_receipt = controlled_action_contract(self.model.model)
            self.model.requires_grad_(False).eval()  # Preserve FP32 frozen LoRA and trained gate.
            self.trainable_parameters = summary.trainable_parameters
            self.vae = WanVAEWrapper(pretrained_path=str(Path(self.config.base_model_path)/'Wan2.2_VAE.pth'))
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
    from training.demo.backend import validate_active_authority, validate_lease
    from training.eval.rollout15s import ffmpeg_writer_factory
    validate_request(request)
    directory = contained(directory, ROOT)
    require(directory == ROOT/request['job_id'] and read(directory/'request.json') == request
        and not any((directory/name).exists() for name in ('progress.json', *CONDITIONS)), 'new prepared job required')
    require(lease['gpu_index'] == 2, 'fixed stage GPU2 only')
    validate_lease(lease, max_seconds=1200, job_id=request['job_id'], request_sha256=sha256(directory/'request.json'))
    require(os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'offline assets required')
    began, runtime, engine, results, peak = time.monotonic(), None, None, [], 0
    def check(minimum=30):
        validate_active_authority(lease)
        require(1200-(time.monotonic()-began) >= minimum, 'finite two-clip deadline')
        require(matched.directory_bytes(directory) < 40_000_000, 'two-clip40MB job cap')
    try:
        check()
        endpoint_request, endpoint, _ = training.completed_endpoint(request['arm'], 20)
        require(record_binding(paths(request['arm'])['endpoint']) == request['endpoint_binding'], 'endpoint records changed')
        values = {c: load_inputs(request, c) for c in CONDITIONS}
        baseline.validate_pair_inputs(values['original'], values['jl_swap'])
        _, before, _ = baseline.baseline_records(request['case_id'], request['baseline_binding'])
        references, reference_lineage = load_reference(request)
        state, config, checkpoint_lineage = load_endpoint(request, endpoint_request, endpoint)
        require(Path(values['original'][5]['negative_source']['source_model']).name == Path(config.model.base_model_path).name,
                'negative text base must match model')
        tick = time.monotonic()
        runtime = ExpandedReferenceRuntime(state, config, values['original'][4], arm=request['arm'],
            checkpoint_lineage=checkpoint_lineage, references=references)
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
                lineage=lineage, reference_lineage=reference_lineage, checkpoint_lineage=checkpoint_lineage,
                export_receipt=runtime.export_receipt, actual_actions=actions.tolist(), seed=42,
                future_clean_latents_read=False, future_clean_latents_supplied=False))
            calls_before = runtime.forward_calls
            generation, endpoint_rgb = engine.run_segment(initial=initial, prompt=prompt, actions=actions.copy(),
                seed=42, output_path=target/'raw.mp4', steps=40)
            del endpoint_rgb
            calls = runtime.forward_calls-calls_before
            baseline.validate_generation(generation, calls, lineage, before[condition])
            if results:
                baseline.validate_generation(generation, calls, lineage, results[0])
            generation.update(initial_condition='fixed_qualification_cached_z0;RGB0_display_only',
                continuation='independent_seed42_noise_and_z0;no_future_ground_truth', vae_encoding_used=False)
            generation['sampler'].update(cfg='text_classifier_free', cfg_scale=5, model_forwards_per_step=2)
            peak = max(peak, torch.cuda.max_memory_reserved())
            row = dict(status='completed', mode=MODE, revision=REVISION, arm=request['arm'], case_id=request['case_id'],
                target=request['target'], condition=condition, lineage=lineage, reference_lineage=reference_lineage,
                checkpoint_lineage=checkpoint_lineage, export_receipt=runtime.export_receipt, generation=generation,
                frames_written=97, raw_video=str(target/'raw.mp4'), actual_model_forward_calls=calls,
                cumulative_model_forward_calls=runtime.forward_calls,
                quality_evaluation='awaiting_actual_visual_review', quality_improvement_verified=False,
                action_correctness_verified=False, generalization_verified=False, teacher_qualified=False, automatic_next_stage=False)
            results.append(row)
            write_json(target/'result.json', row)
            write_json(directory/'progress.json', dict(status='running', cases=results, elapsed_seconds=time.monotonic()-began))
        require(len(results) == 2 and runtime.forward_calls == 160 and peak < 23*1024**3, 'two native97 clips/160forwards/<23GiB required')
        check()
        return dict(status='completed', mode=MODE, revision=REVISION, request=request, arm=request['arm'], case_id=request['case_id'],
            cases=results, actual_model_forward_calls=160, model_load_count=1, load_seconds=load_seconds,
            checkpoint_lineage=checkpoint_lineage, reference_lineage=reference_lineage, export_receipt=runtime.export_receipt,
            elapsed_seconds=time.monotonic()-began, peak_reserved_bytes=peak,
            future_clean_latents_read=False, future_clean_latents_supplied=False,
            quality_evaluation='awaiting_actual_visual_review', quality_improvement_verified=False,
            action_correctness_verified=False, generalization_verified=False, teacher_qualified=False,
            reference_memory_implemented=True, complete_abot_reproduction=False, repeated_development_qualification=True,
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
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))
