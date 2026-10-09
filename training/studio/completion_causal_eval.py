"""Bounded new-stage causal TF20 evaluation on two real original/JL pairs.

Input loading reuses genuine baseline requests, not the old R005 evaluator's
checkpoint contract. Each clip owns two independent resident GPU KV histories.
The model generates both clips before a separately loaded VAE decodes them;
therefore block timings are not input-to-visible-frame streaming latency.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import re
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.studio import balanced_lora_pair_eval as baseline
from training.studio.causal_native_eval import FullHistoryCFG, close_video_preserving, normalize_initial_latent
from training.studio.controlled_action_train import RUNTIME

ROOT = RUNTIME / 'abot-completion-20261007/causal-tf-v1'
MODE = 'completion_causal_eval'
REVISION = 'completion-tf20-native97-resident-separate-cfg-kv-pairs-v1'
SOURCE_REVISION = 'creator-abot-completion-tf-v1'
CAP, BYTES = 21600, 8_000_000_000
CONDITIONS = ('original', 'jl_swap')
CASES = ('pair-j', 'holdout-l')
FIXED = {key: copy.deepcopy(baseline.FIXED[key]) for key in CASES}
JOBS = {key: f'ccf-eval-{key}-v1' for key in CASES}
BLOCKS, BLOCK_LATENTS, STEPS = 8, 3, 40
FORWARDS_PER_CLIP = 2 + BLOCKS * (2 * STEPS + 2)


def paths():
    profile = ROOT / 'ccf-profile2-v1'
    return dict(config=profile / 'training.yaml',
                checkpoint=profile / 'training/checkpoints/step-0000020.pt',
                endpoint=ROOT / 'ccf-to20-v1')


def fixed_fields(case_id):
    require(case_id in CASES, 'only fixed pair-j and holdout-l evaluations')
    episode, start, target, role = FIXED[case_id]
    p = paths()
    return dict(mode=MODE, experiment_revision=REVISION, source_revision=SOURCE_REVISION,
        job_id=JOBS[case_id], case_id=case_id, episode_id=episode, source_start=start,
        target=target, candidate_role=role, conditions=list(CONDITIONS),
        checkpoint=str(p['checkpoint']), training_config=str(p['config']), endpoint_job=str(p['endpoint']),
        checkpoint_step=20, max_seconds=1800, cleanup_seconds=60,
        seed=42, steps=STEPS, cfg_scale=5, solver='flow_euler', timestep_shift=5.,
        frames=97, future_frames=96, latent_blocks=BLOCKS, latent_frames_per_block=BLOCK_LATENTS,
        fps=16, width=832, height=480, cfg_cache_offload=False,
        reference_enabled=False, lora_inference_dtype='float32', lora_strength=1,
        phase_order='generate_two_then_release_model_then_decode_two',
        evaluation_view='original_unmirrored', evaluation_action_swap='JL_only',
        repeated_development_qualification=True,
        protocol_cap_gpu_seconds=CAP, protocol_incremental_bytes=BYTES)


def validate_request(request):
    expected = fixed_fields(request.get('case_id'))
    extra = {'checkpoint_stat', 'training_config_sha256', 'endpoint_binding', 'baseline_binding', 'baseline_request'}
    require(set(request) == set(expected) | extra
        and all(request.get(k) == v and type(request.get(k)) is type(v) for k, v in expected.items()),
        'fixed new causal evaluation identity, solver or context changed')
    require(request['baseline_request'] == str(baseline.ROOT / baseline.JOBS['frozen', request['case_id']] / 'request.json'),
        'reuse the genuine frozen24 input request only')
    digest = lambda value: isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None
    require(digest(request['training_config_sha256']), 'actual configuration binding required')
    for key in ('endpoint_binding', 'baseline_binding'):
        value = request[key]
        require(isinstance(value, dict) and set(value) == {n + '_sha256' for n in ('request', 'result', 'settlement')}
            and all(digest(item) for item in value.values()), 'completed small-record binding required')
    stat = request['checkpoint_stat']
    require(isinstance(stat, dict) and set(stat) == {'bytes', 'mtime_ns'}
        and all(type(v) is int and v > 0 for v in stat.values()), 'checkpoint stat identity required')


def baseline_inputs(request, condition):
    """The shared loader mmap-slices only z0 before cloning or validating it."""
    validate_request(request)
    require(condition in CONDITIONS, 'fixed original/JL-only condition required')
    original_request, prior, binding = baseline.baseline_records(request['case_id'], request['baseline_binding'])
    require(binding == request['baseline_binding'], 'actual baseline records changed')
    values = baseline.load_inputs(original_request, condition)
    require(values[5] == prior[condition]['lineage'], 'cached z0/text/action input differs from actual baseline')
    require(values[5]['future_clean_latents_read'] is False
        and values[5]['future_clean_latents_supplied'] is False, 'future clean latents forbidden')
    return values


def prepare(*, output, case_id, max_seconds=1800, source_revision=SOURCE_REVISION):
    from training.studio import completion_causal_train as training
    require(source_revision == SOURCE_REVISION and type(max_seconds) is int and max_seconds == 1800,
        'fixed revision and bounded 1800-second evaluation required')
    request = fixed_fields(case_id)
    output = contained(output, ROOT)
    require(output == ROOT / request['job_id'] and not output.exists(), 'new explicit evaluation identity required')
    endpoint_request, _, _ = training.completed_endpoint(20)
    p = paths()
    require(all(training.paths(20)[k] == p[k] for k in p), 'training and evaluation endpoint locations differ')
    request.update(checkpoint_stat=baseline.stat_identity(p['checkpoint']),
        training_config_sha256=endpoint_request['config_sha256'], endpoint_binding=training.record_binding(p['endpoint']),
        baseline_binding=baseline.baseline_records(case_id)[2],
        baseline_request=str(baseline.ROOT / baseline.JOBS['frozen', case_id] / 'request.json'))
    validate_request(request)
    values = [baseline_inputs(request, condition) for condition in CONDITIONS]
    baseline.validate_pair_inputs(*values)
    lineages = {condition: value[5] for condition, value in zip(CONDITIONS, values)}
    del values
    gc.collect()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'request.json', request)
    write_json(output / 'preparation.json', dict(status='cpu_prepared', source_input_lineages=lineages,
        weights_loaded=False, gpu_launched=False, future_clean_latents_read=False,
        checkpoint_source='new_completion_causal_tf20_not_old_R005', automatic_next_stage=False))
    return request


def load_endpoint(request, endpoint_request, endpoint):
    import torch
    import train_completion_causal_tf as entry
    p = paths()
    require(baseline.stat_identity(p['checkpoint']) == request['checkpoint_stat']
        and sha256(p['config']) == request['training_config_sha256'], 'actual new checkpoint/config changed')
    config = entry.load_config(p['config'])
    binding = endpoint_request['data_binding']
    payload = torch.load(p['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    entry.validate_checkpoint(payload, config=config, hashes=entry.artifact_hashes(config, p['config'], binding),
        data_binding=binding, sampling=endpoint['sampling_contract'])
    entry.validate_base_identity(payload, config)
    require(payload['stage'] == entry.STAGE and payload['checkpoint_schema'] == entry.CHECKPOINT_SCHEMA
        and payload['step'] == 20 and payload['micro_batches_consumed'] == 160,
        'actual new causal20 checkpoint required; no stage coercion')
    fields = ('parent_lineage', 'base_identity', 'model_signature', 'frozen_action_signature',
              'sampling_contract', 'method_contract')
    require(all(payload[k] == endpoint[k] for k in fields), 'actual checkpoint and settled endpoint differ')
    lineage = dict(stage=payload['stage'], checkpoint_schema=payload['checkpoint_schema'], step=20,
        checkpoint=str(p['checkpoint']), checkpoint_stat=request['checkpoint_stat'],
        **{k: payload[k] for k in fields}, inference_uses='frozen_parent_Adapter_gate_and_trained_FP32_LoRA',
        reference_enabled=False, parent_teacher_qualified=False, realtime=False)
    require(baseline.stat_identity(p['checkpoint']) == request['checkpoint_stat'], 'checkpoint changed during read')
    return payload['frozen_action_model'], payload['trainable_model'], config, lineage


def _autocast(device):
    import torch
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


def generate_clip(model, config, first, positive, negative, actions, *, device, check,
                  flow_factory=FullHistoryCFG, solver=None, synchronize=None, progress=None):
    """Generate exactly one clip; fresh RNG and KV on every call, CPU latents only.

    CPU dependency injection exists only for focused contract tests. Production
    run requires a validated CUDA lease and always uses the real implementations.
    """
    import numpy as np
    import torch
    from training.longforcing_lite import _solve_block
    solver = _solve_block if solver is None else solver
    device = torch.device(device)
    synchronize = (lambda: torch.cuda.synchronize(device)) if synchronize is None and device.type == 'cuda' else (synchronize or (lambda: None))
    require(isinstance(actions, np.ndarray) and actions.shape == (96, 8) and actions.dtype == np.float32
        and np.isfinite(actions).all() and np.isin(actions, (0., 1.)).all(), 'exact finite binary96x8 action input required')
    first = normalize_initial_latent(first).to(device=device)
    rng = torch.Generator(device='cpu').manual_seed(42)
    history, flow, rows = first, None, []
    began = time.monotonic()
    try:
        with torch.no_grad(), _autocast(device):
            flow = flow_factory(model, config, first, positive, negative, capacity=25, offload=False, check=check)
            for index in range(BLOCKS):
                check()
                tick = time.monotonic()
                noise = torch.randn((1, 3, 48, 30, 52), generator=rng, dtype=torch.float32).to(device, torch.bfloat16)
                action = torch.from_numpy(actions[index * 12:(index + 1) * 12].copy())[None].to(device, torch.bfloat16)
                calls_before = flow.stats['forward_calls']
                endpoint = solver(flow.velocity, initial_noise=noise, history=history, action_block=action,
                    block_index=index, conditions={}, steps=STEPS, timestep_shift=5.)
                require(tuple(endpoint.shape) == (1, 3, 48, 30, 52)
                    and bool(torch.isfinite(endpoint).all()), 'nonfinite or wrong-shaped generated causal block')
                flow.commit(endpoint, index)
                history = torch.cat((history, endpoint.detach()), dim=1)
                synchronize()
                row = dict(index=index, action_start=index * 12, action_stop=(index + 1) * 12,
                    latent_start=1 + index * 3, latent_stop=4 + index * 3,
                    seconds=time.monotonic() - tick, forward_calls=flow.stats['forward_calls'] - calls_before,
                    prediction='flow_velocity', solver='flow_euler', steps=STEPS,
                    clean_context_commit=True, visible_frame_latency_measured=False)
                require(row['forward_calls'] == 82, '40x2 denoising plus two clean commits required')
                rows.append(row)
                if progress is not None:
                    progress(copy.deepcopy(rows))
            require(flow.position == 25 and flow.stats['forward_calls'] == FORWARDS_PER_CLIP
                and flow.stats['evictions'] == 0 and flow.stats['host_transfer_bytes'] == 0,
                'complete full GPU KV history and exact forward count required')
            return history.detach().cpu(), dict(blocks=rows, kv=copy.deepcopy(flow.stats),
                latent_generation_seconds=time.monotonic() - began,
                actual_model_forward_calls=flow.stats['forward_calls'],
                denoising_forward_calls=640, context_forward_calls=18,
                rng='fresh_CPU_generator_seed42_eight_sequential_noise_blocks',
                model_input_future_clean=False, visible_frame_latency_measured=False)
    finally:
        if flow is not None:
            flow.close()


def decode_clip(vae, latents, initial, *, device, check, writer, finalize=close_video_preserving):
    """One continuous VAE decode cache per clip; initial RGB is display-only."""
    import numpy as np
    import torch
    device = torch.device(device)
    require(tuple(latents.shape) == (1, 25, 48, 30, 52), 'generated native97 latent sequence required')
    require(initial.shape == (480, 832, 3) and initial.dtype == np.uint8, 'exact initial display RGB required')
    vae.model.clear_cache()
    written, finalized, began = 0, False, time.monotonic()
    try:
        for start, stop, count in [(0, 1, 1), *[(1 + i * 3, 4 + i * 3, 12) for i in range(BLOCKS)]]:
            check()
            with torch.no_grad(), _autocast(device):
                decoded = vae.decode_to_pixel(latents[:, start:stop].to(device, torch.bfloat16), use_cache=True, return_in_cpu=True)
            require(tuple(decoded.shape) == (1, count, 3, 480, 832) and bool(torch.isfinite(decoded).all()),
                'continuous VAE decoded shape/finite contract failed')
            rgb = decoded[0].permute(0, 2, 3, 1).add(1).mul(127.5).round().clamp(0, 255).byte().numpy()
            if start == 0:
                rgb[0] = initial
            writer.write(rgb)
            written += count
        finalize(writer)
        finalized = True
        require(written == 97, 'exact97 real decoded frames required')
        return dict(frames_written=written, decode_seconds=time.monotonic() - began,
            vae_cache_reset=True, vae_continuous_decode=True, vae_encoding_used=False,
            exact_initial_rgb_preserved=True)
    finally:
        vae.model.clear_cache()
        if not finalized:
            try:
                finalize(writer)  # Preserve partial media; never call failure-time unlink.
            except Exception:
                pass


def run(request, directory, lease):
    import numpy as np
    import torch
    from PIL import Image
    import train_completion_causal_tf as entry
    from training.demo.backend import validate_active_authority, validate_lease
    from training.eval.rollout15s import ffmpeg_writer_factory
    from training.studio import completion_causal_train as training
    from utils.wan_wrapper import WanVAEWrapper
    validate_request(request)
    directory = contained(directory, ROOT)
    require(directory == ROOT / request['job_id'] and training.read(directory / 'request.json') == request
        and not any((directory / n).exists() for n in ('progress.json', *CONDITIONS)), 'new prepared evaluation only')
    require(lease.get('gpu_index') in (1, 2), 'original Creator GPU1/2 authority only')
    validate_lease(lease, max_seconds=1800, job_id=request['job_id'], request_sha256=sha256(directory / 'request.json'))
    require(os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'offline existing assets required')
    began, model, vae, generated, results = time.monotonic(), None, None, {}, []
    report = dict(status='running', mode=MODE, revision=REVISION, request=request, case_id=request['case_id'],
        cases=results, quality_evaluation='awaiting_actual_visual_review', quality_improvement_verified=False,
        action_correctness_verified=False, realtime=False, automatic_next_stage=False,
        future_clean_latents_read=False, future_clean_latents_supplied=False)
    cuda_started = False
    def check():
        validate_active_authority(lease)
        require(time.monotonic() - began < 1785, 'finite causal evaluation deadline reached')
    try:
        check()
        endpoint_request, endpoint, _ = training.completed_endpoint(20)
        require(training.record_binding(paths()['endpoint']) == request['endpoint_binding'], 'new endpoint records changed')
        values = {condition: baseline_inputs(request, condition) for condition in CONDITIONS}
        baseline.validate_pair_inputs(values['original'], values['jl_swap'])
        frozen, state, config, lineage = load_endpoint(request, endpoint_request, endpoint)
        require(Path(values['original'][5]['negative_source']['source_model']).name == Path(config.model.base_model_path).name,
            'text negative and new model base differ')
        contained(config.model.base_model_path, RUNTIME.parent)
        report['checkpoint_lineage'] = lineage
        tick = time.monotonic()
        model, _ = entry.build_model(config, torch.device('cpu'), frozen_action_model=frozen, trainable_model=state)
        require(entry.model_signature(model) == lineage['model_signature'], 'loaded causal LoRA signature differs')
        require(model.is_causal is True and model.uniform_timestep is False
            and model.model.controlled_action_enabled is True
            and model.model.controlled_action_ratio_cap == .1 and model.model.controlled_action_eps == 1e-6
            and config.model.action_scale == 1., 'new controlled causal model required')
        del frozen, state
        gc.collect()
        model.requires_grad_(False).eval()
        device = torch.device('cuda')
        torch.cuda.reset_peak_memory_stats(device)
        cuda_started = True
        model.to(device=device)  # Deliberately no global dtype cast: preserve FP32 LoRA/gate.
        lora = [p for n, p in model.named_parameters() if n.endswith(('.lora_a.weight', '.lora_b.weight'))]
        require(bool(lora) and all(p.dtype == torch.float32 for p in lora)
            and model.model.controlled_action_gate.dtype == torch.float32, 'trained FP32 low bits must not be rounded')
        del lora
        report['model_load_seconds'] = time.monotonic() - tick
        report['model_load_count'] = 1
        for condition in CONDITIONS:
            check()
            first, positive, actions, initial, negative, input_lineage = values[condition]
            target = directory / condition
            target.mkdir(exist_ok=False)
            Image.fromarray(initial).save(target / 'initial.png')
            np.save(target / 'actions.npy', actions, allow_pickle=False)
            write_json(target / 'inputs.json', dict(condition=condition, source_input_lineage=input_lineage,
                checkpoint_lineage=lineage, actual_actions=actions.tolist(), seed=42,
                future_clean_latents_read=False, future_clean_latents_supplied=False))
            def progress(rows):
                write_json(directory / 'progress.json', dict(report, phase='generating', active_condition=condition,
                    current_blocks=rows, elapsed_seconds=time.monotonic() - began))
            latents, generated_report = generate_clip(model, config, first, positive, negative, actions,
                device=device, check=check, progress=progress)
            generated[condition] = (latents, generated_report)
            write_json(target / 'latent-generation.json', generated_report)
        report['generation_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        require(report['generation_peak_reserved_bytes'] < 23 * 1024**3, 'resident KV evaluation exceeded23GiB')
        model.to('cpu')
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        check()
        tick = time.monotonic()
        vae = WanVAEWrapper(vae_type='Wan2.2_VAE', pretrained_path=str(Path(config.model.base_model_path) / 'Wan2.2_VAE.pth'))
        vae.requires_grad_(False).eval().to(device=device, dtype=torch.bfloat16)
        report['vae_load_seconds'] = time.monotonic() - tick
        for condition in CONDITIONS:
            check()
            latents, generation = generated.pop(condition)
            target = directory / condition
            writer = ffmpeg_writer_factory(target / 'raw.mp4', 832, 480, 16)
            decoded = decode_clip(vae, latents, values[condition][3], device=device, check=check, writer=writer)
            del latents
            row = dict(status='completed', mode=MODE, revision=REVISION, case_id=request['case_id'], condition=condition,
                lineage=values[condition][5], checkpoint_lineage=lineage, generation=generation,
                **decoded, actual_model_forward_calls=generation['actual_model_forward_calls'],
                raw_video=str(target / 'raw.mp4'), quality_evaluation='awaiting_actual_visual_review',
                quality_improvement_verified=False, action_correctness_verified=False,
                visible_frame_latency_measured=False, realtime=False)
            results.append(row)
            write_json(target / 'result.json', row)
            write_json(directory / 'progress.json', dict(report, phase='decoding', elapsed_seconds=time.monotonic() - began))
        require(len(results) == 2 and sum(row['actual_model_forward_calls'] for row in results) == 1316
            and torch.cuda.max_memory_reserved() < 23 * 1024**3,
            'two real causal clips required')
        require(sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()) < 100_000_000,
            'bounded paired output exceeded100MB')
        report.update(status='completed', actual_model_forward_calls=1316, denoising_forward_calls=1280,
            context_forward_calls=36, reference_enabled=False, cfg_cache_offload=False,
            phase_order=request['phase_order'], generated_latents_saved=False,
            streaming_latency_measured=False, sustained_streaming_fps_measured=False,
            parent_teacher_qualified=False, repeated_development_qualification=True,
            complete_abot_reproduction=False)
        check()
        return report
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', automatic_retry=False)
        write_json(directory / 'failure.json', dict(report, elapsed_seconds=time.monotonic() - began))
        raise
    finally:
        generated.clear()
        for item in (model, vae):
            if item is not None:
                try:
                    item.to('cpu')
                except Exception as error:
                    report.setdefault('cleanup_errors', []).append(repr(error))
        if cuda_started:
            report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            torch.cuda.empty_cache()
        report['elapsed_seconds'] = time.monotonic() - began
        write_json(directory / 'evaluation.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--case-id', choices=CASES, required=True)
    parser.add_argument('--source-revision', default=SOURCE_REVISION)
    parser.add_argument('--max-seconds', type=int, default=1800)
    print(json.dumps(prepare(**vars(parser.parse_args()))))
