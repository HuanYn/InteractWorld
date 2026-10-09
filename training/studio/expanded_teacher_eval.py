"""One real expanded80 endpoint, four fixed dev windows, original/JL pairs."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.studio import balanced_lora_pair_eval as baseline
from training.studio import controlled_matched_window_eval as matched
from training.studio import expanded_teacher_train as training
from training.studio.quality_probe import QualityRuntime, capture_lora_scales

ROOT, CAP, BYTES = training.ROOT, training.CAP, training.BYTES
MODE = 'expanded_teacher_eval'
REVISION = 'expanded80-fixed-four-dev-original-jlswap-v1'
SOURCE_REVISION = training.SOURCE_REVISION
CONDITIONS = ('original', 'jl_swap')
FIXED = copy.deepcopy(baseline.FIXED)
JOBS = {case: f'ced-eval-{case}-v1' for case in FIXED}
RECOVERY_JOB = 'ced-eval-pair-l-v2'


def pair_l_recovery_binding():
    """Only this recorded pre-reservation failure has an explicit v2 alias."""
    plan_path = ROOT / 'eval-recovery.json'
    plan = training.read(plan_path)
    deployment = training.read(ROOT / 'deployment-eval-recovery.json')
    require(plan.get('schema') == 'expanded_teacher_eval_recovery_v1'
        and plan.get('failed_job') == JOBS['pair-l'] and plan.get('replacement_job') == RECOVERY_JOB
        and plan.get('replacement_case') == 'pair-l' and plan.get('storage_scan_timeout_seconds') == 240
        and deployment.get('eval_recovery_sha256') == sha256(plan_path)
        and deployment.get('status') == 'deployed_not_launched'
        and deployment.get('new_eval_source_sha256') == sha256(__file__), 'explicit deployed pair-l-only recovery required')
    failed = ROOT / JOBS['pair-l']
    names = {'preparation.json', 'request.json', 'launcher.log', 'recovery-launch.log'}
    require(failed.is_dir() and {p.name for p in failed.iterdir()} == names
        and all(p.is_file() and not p.is_symlink() for p in failed.iterdir())
        and {n: sha256(failed / n) for n in names} == deployment['failed_file_sha256'],
        'old pre-reservation evidence changed or a lease/worker/result appeared')
    text = (failed / 'launcher.log').read_text(encoding='utf-8')
    require('TimeoutExpired' in text and "'du'" in text and '240 seconds' in text, 'original du240 reserve timeout required')
    ledger = training.read(training.RUNTIME / 'budget.json')
    require(not any(r.get('context', {}).get('job_id') == JOBS['pair-l']
        or r.get('context', {}).get('job_directory') == str(failed) for r in ledger['reservations'].values()),
        'old failed identity must have no reservation or charge')
    return dict(path=str(plan_path), sha256=sha256(plan_path))


def fixed_fields(case_id, *, eval_recovery=False):
    require(case_id in FIXED, 'only four fixed repeated-development cases')
    require(type(eval_recovery) is bool and (not eval_recovery or case_id == 'pair-l'), 'only pair-l has a recovery alias')
    identity, start, target, role = FIXED[case_id]
    p = training.paths(80)
    request = dict(mode=MODE, experiment_revision=REVISION, source_revision=SOURCE_REVISION,
        job_id=RECOVERY_JOB if eval_recovery else JOBS[case_id], case_id=case_id, episode_id=identity, source_start=start, target=target,
        candidate_role=role, purpose='qualification', original_split='train', conditions=list(CONDITIONS),
        checkpoint=str(p['checkpoint']), training_config=str(p['config']), endpoint_job=str(p['endpoint']),
        checkpoint_step=80, seed=42, steps=40, cfg_scale=5, frames=97, future_frames=96,
        fps=16, width=832, height=480, solver='flow_euler', residency='phase', fastpath=False,
        max_seconds=1200, cleanup_seconds=60, reference_enabled=False, lora_strength=1,
        lora_inference_dtype='float32', evaluation_view='original_unmirrored', evaluation_action_swap='JL_only',
        repeated_development_qualification=True, protocol_cap_gpu_seconds=CAP, protocol_incremental_bytes=BYTES)
    if eval_recovery:
        request['eval_recovery_binding'] = pair_l_recovery_binding()
    return request


def validate_request(request):
    expected = fixed_fields(request.get('case_id'), eval_recovery=request.get('job_id') == RECOVERY_JOB)
    require(all(request.get(k) == v and type(request.get(k)) is type(v) for k, v in expected.items())
            and set(request) == set(expected) | {'checkpoint_stat', 'training_config_sha256',
                'endpoint_binding', 'baseline_binding', 'baseline_request'}, 'fixed expanded80 evaluation contract changed')
    require(request['baseline_request'] == str(baseline.ROOT / baseline.JOBS['frozen', request['case_id']] / 'request.json'),
            'only genuine frozen24 baseline input record may be reused')


def baseline_inputs(request, condition):
    """Pass the genuine old request to its own input loader; never forge a schema."""
    validate_request(request)
    original_request, prior, binding = baseline.baseline_records(request['case_id'], request['baseline_binding'])
    require(binding == request['baseline_binding'], 'frozen24 input record changed')
    values = baseline.load_inputs(original_request, condition)
    require(values[5] == prior[condition]['lineage'], 'cached z0/text/actions differ from actual frozen24 inputs')
    return values


def prepare(*, output, case_id, source_revision=SOURCE_REVISION, max_seconds=1200, eval_recovery=False):
    require(source_revision == SOURCE_REVISION and max_seconds == 1200, 'fixed source revision/eval cap')
    request = fixed_fields(case_id, eval_recovery=eval_recovery)
    output = contained(output, ROOT)
    require(output == ROOT / request['job_id'] and not output.exists(), 'new fixed eval identity required')
    endpoint_request, _, _ = training.completed_endpoint(80)
    p = training.paths(80)
    request.update(checkpoint_stat=baseline.stat_identity(p['checkpoint']),
        training_config_sha256=endpoint_request['config_sha256'], endpoint_binding=training.record_binding(p['endpoint']),
        baseline_binding=baseline.baseline_records(case_id)[2],
        baseline_request=str(baseline.ROOT / baseline.JOBS['frozen', case_id] / 'request.json'))
    validate_request(request)
    values = [baseline_inputs(request, c) for c in CONDITIONS]
    baseline.validate_pair_inputs(*values)
    inputs = {c: v[5] for c, v in zip(CONDITIONS, values)}
    del values
    gc.collect()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'request.json', request)
    write_json(output / 'preparation.json', dict(status='cpu_prepared', request=request, source_input_lineages=inputs,
        inputs_are_actual_frozen24_descriptors=True, model_weights_loaded=False, gpu_launched=False,
        future_clean_latents_read=False, automatic_next_stage=False))
    return request


def load_endpoint(request, endpoint_request, endpoint):
    import torch
    import train_expanded_teacher as entry
    p = training.paths(80)
    require(baseline.stat_identity(p['checkpoint']) == request['checkpoint_stat']
            and sha256(p['config']) == request['training_config_sha256'], 'actual checkpoint/config changed')
    config = entry.load_config(p['config'])
    binding = endpoint_request['data_binding']
    payload = torch.load(p['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    entry.validate_checkpoint(payload, config=config, hashes=entry.artifact_hashes(config, p['config'], binding),
                              data_binding=binding, sampling=endpoint['sampling_contract'])
    entry.validate_base_identity(payload, config)
    require(all(payload.get(k) == v for k, v in entry.counters(80).items()), 'actual80/640/1312 counters required')
    fields = ('parent_lineage', 'base_identity', 'model_signature', 'sampling_contract', 'method_contract', 'precision_contract')
    require(payload['stage'] == entry.STAGE and payload['checkpoint_schema'] == entry.CHECKPOINT_SCHEMA
            and all(payload[k] == endpoint[k] for k in fields), 'actual new endpoint/schema/contracts differ')
    lineage = dict(stage=entry.STAGE, checkpoint_schema=entry.CHECKPOINT_SCHEMA,
        checkpoint=str(p['checkpoint']), checkpoint_stat=request['checkpoint_stat'], **entry.counters(80),
        **{k: payload[k] for k in fields}, reference_enabled=False, trained_lora=False,
        inference_uses='exact_expanded80_BF16_Adapter_FP32_gate_export_frozen_FP32_LoRA',
        parent_strict_resume=False, strict_resume_only_within_new_stage=True)
    require(baseline.stat_identity(p['checkpoint']) == request['checkpoint_stat'], 'checkpoint changed during read')
    return payload['trainable_model'], config, lineage


class ExpandedRuntime(QualityRuntime):
    def __init__(self, state, config, negative, lineage):
        import torch
        import train_expanded_teacher as entry
        from utils.controlled_action import controlled_action_contract
        from utils.wan_wrapper import WanVAEWrapper
        self.torch, self.config, self.residency = torch, copy.deepcopy(config.model), 'phase'
        self.model = self.vae = None
        self.lora_scales, self.negative, self.forward_calls = [], negative, 0
        try:
            self.model, summary = entry._build_model(config, torch.device('cpu'))
            require(entry.model_signature(self.model) == lineage['model_signature']
                    and self.model.is_causal is False and self.model.uniform_timestep is True, 'new bidirectional model signature differs')
            self.export_receipt = baseline.load_exact_export(self.model, state, arm='frozen')
            require(getattr(self.model.model, 'controlled_action_enabled', False) is True
                    and self.model.model.controlled_action_ratio_cap == .1
                    and self.model.model.controlled_action_eps == 1e-6 and self.config.action_scale == 1,
                    'training/inference action residual differs')
            require(not any(bool(getattr(self.model.model, k, False)) for k in
                ('enable_teacache', 'use_teacache', 'enable_fbcache', 'use_fbcache')), 'approximate caches forbidden')
            self.controlled_action_receipt = controlled_action_contract(self.model.model)
            self.model.requires_grad_(False).eval()  # Preserve FP32 gate and frozen LoRA.
            self.trainable_parameters = summary.trainable_parameters
            self.vae = WanVAEWrapper(pretrained_path=str(Path(self.config.base_model_path) / 'Wan2.2_VAE.pth'))
            self.vae.requires_grad_(False).eval().to(dtype=torch.bfloat16)
            self.lora_scales = capture_lora_scales(self.model)
            self.configure(dict(cfg_scale=5, lora_strength=1, action_scale=1, solver='flow_euler'))
        except BaseException:
            self.close()
            raise

    def generate(self, *args, **kwargs):
        require(all(m.lora_a.weight.dtype == self.torch.float32 and m.lora_b.weight.dtype == self.torch.float32
                    and m.scaling == scale for m, scale in self.lora_scales), 'FP32 frozen LoRA strength changed')
        return super().generate(*args, **kwargs)


def run(request, directory, lease):
    import numpy as np
    import torch
    from PIL import Image
    from training.demo.backend import validate_active_authority, validate_lease
    from training.eval.rollout15s import ffmpeg_writer_factory
    validate_request(request)
    directory = contained(directory, ROOT)
    require(directory == ROOT / request['job_id'] and training.read(directory / 'request.json') == request
            and not any((directory / n).exists() for n in ('progress.json', *CONDITIONS)), 'new prepared eval only')
    require(lease.get('gpu_index') == 2, 'only original Creator GPU2')
    validate_lease(lease, max_seconds=1200, job_id=request['job_id'], request_sha256=sha256(directory / 'request.json'))
    require(os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'offline model assets required')
    began, runtime, engine, results, peak = time.monotonic(), None, None, [], 0
    def check(minimum=30):
        validate_active_authority(lease)
        require(1200 - (time.monotonic() - began) >= minimum and matched.directory_bytes(directory) < 40_000_000,
                'finite paired eval time/storage boundary')
    try:
        check()
        endpoint_request, endpoint, _ = training.completed_endpoint(80)
        require(training.record_binding(training.paths(80)['endpoint']) == request['endpoint_binding'], 'endpoint records changed')
        values = {c: baseline_inputs(request, c) for c in CONDITIONS}
        baseline.validate_pair_inputs(values['original'], values['jl_swap'])
        _, before, _ = baseline.baseline_records(request['case_id'], request['baseline_binding'])
        state, config, lineage = load_endpoint(request, endpoint_request, endpoint)
        require(Path(values['original'][5]['negative_source']['source_model']).name == Path(config.model.base_model_path).name,
                'negative text base differs')
        tick = time.monotonic()
        runtime = ExpandedRuntime(state, config, values['original'][4], lineage)
        state.clear()
        gc.collect()
        load_seconds = time.monotonic() - tick
        engine = matched.CachedInitialEngine({}, runtime.config, residency='phase', fastpath=False, _runtime=runtime,
            cached_first=values['original'][0], _writer_factory=lambda *args: matched.BoundedPreservingWriter(ffmpeg_writer_factory(*args)))
        engine.load_seconds = load_seconds
        for condition in CONDITIONS:
            check(270)
            first, prompt, actions, initial, negative, input_lineage = values[condition]
            target = directory / condition
            target.mkdir(exist_ok=False)
            Image.fromarray(initial).save(target / 'initial.png')
            np.save(target / 'actions.npy', actions, allow_pickle=False)
            write_json(target / 'inputs.json', dict(request=request, condition=condition, source_input_lineage=input_lineage,
                checkpoint_lineage=lineage, export_receipt=runtime.export_receipt, actual_actions=actions.tolist(),
                seed=42, reference_enabled=False, future_clean_latents_read=False))
            calls_before = runtime.forward_calls
            generation, last_rgb = engine.run_segment(initial=initial, prompt=prompt, actions=actions.copy(),
                seed=42, output_path=target / 'raw.mp4', steps=40)
            del last_rgb
            calls = runtime.forward_calls - calls_before
            baseline.validate_generation(generation, calls, input_lineage, before[condition])
            if results:
                baseline.validate_generation(generation, calls, input_lineage, results[0])
            generation.update(initial_condition='fixed_qualification_cached_z0;RGB0_display_only',
                continuation='independent_seed42_noise_and_z0;no_future_ground_truth', vae_encoding_used=False)
            generation['sampler'].update(cfg='text_classifier_free', cfg_scale=5, model_forwards_per_step=2)
            peak = max(peak, torch.cuda.max_memory_reserved())
            row = dict(status='completed', mode=MODE, revision=REVISION, case_id=request['case_id'], condition=condition,
                target=request['target'], source_input_lineage=input_lineage, checkpoint_lineage=lineage,
                generation=generation, frames_written=97, raw_video=str(target / 'raw.mp4'),
                actual_model_forward_calls=calls, quality_evaluation='awaiting_actual_visual_review',
                quality_improvement_verified=False, action_correctness_verified=False, automatic_next_stage=False)
            results.append(row)
            write_json(target / 'result.json', row)
            write_json(directory / 'progress.json', dict(status='running', cases=results, elapsed_seconds=time.monotonic() - began))
        require(len(results) == 2 and runtime.forward_calls == 160 and peak < 23 * 1024**3, 'complete paired eval resource gate')
        check()
        return dict(status='completed', mode=MODE, revision=REVISION, request=request, case_id=request['case_id'],
            cases=results, actual_model_forward_calls=160, model_load_count=1, load_seconds=load_seconds,
            elapsed_seconds=time.monotonic() - began, peak_reserved_bytes=peak,
            checkpoint_lineage=lineage, export_receipt=runtime.export_receipt,
            reference_enabled=False, future_clean_latents_read=False, quality_improvement_verified=False,
            repeated_development_qualification=True, complete_abot_reproduction=False, realtime=False, automatic_next_stage=False)
    except BaseException as error:
        write_json(directory / 'failure.json', dict(status='failed', error=f'{type(error).__name__}: {error}', cases=results,
            elapsed_seconds=time.monotonic() - began, automatic_retry=False,
            partial_generation=copy.deepcopy(engine.last_run_report) if engine is not None else None))
        raise
    finally:
        if engine is not None:
            engine.close()
        elif runtime is not None:
            runtime.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--case-id', choices=tuple(FIXED), required=True)
    parser.add_argument('--source-revision', default=SOURCE_REVISION)
    parser.add_argument('--max-seconds', type=int, default=1200)
    parser.add_argument('--eval-recovery', action='store_true')
    print(json.dumps(prepare(**vars(parser.parse_args()))))
