"""Eight first-step B20 CFG forwards; no training, VAE, rollout or saved tensors."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.studio import paired_turn_b_eval as b

MODE = 'paired_turn_cfg_signal'
REVISION = 'b20-first-euler-cfg-action-signal-eight-forwards-v1'
SOURCE_REVISION = 'creator-b20-cfg-first8-v1'
ROOT = b.a.matched.BASE / 'teacher-b20-cfg-signal-v1'
JOB_ID = 'cbcs-first8-v1'
REVIEW = b.ROOT / 'b/review-result.json'
MAX_SECONDS, CLEANUP_SECONDS, STAGE_BYTES = 900, 60, 100_000_000
TARGETS, CONDITIONS = ('J', 'L'), ('original', 'jl_swap')
read = b.read


def protocol():
    return dict(checkpoint=str(b.CHECKPOINT), checkpoint_stage=b.STAGE, targets=list(TARGETS),
        conditions=list(CONDITIONS), text_branches=['positive', 'original_negative'],
        model_forward_calls=8, model_loads=1, sigma=1., seed=42, cfg_scale=5,
        source_sampler_steps=40, source_revision=SOURCE_REVISION,
        noise='existing_segment6_five_CPU_FP32_draws_first25_then_BF16;verified_existing_noise_sha',
        actual_cfg='TextCFGFlow: negative+5*(positive-negative);BF16_prediction_arithmetic',
        statistics='future24_only;FP32_differences_and_metrics;no_saved_tensors',
        measurable_threshold='RMS(delta_pos)>1e-7*max(RMS(pos_original),RMS(pos_swap),1)',
        window_support='measurable AND (RMS(delta_cfg)/RMS(delta_pos)<=0.25 OR dot(delta_cfg,delta_pos)<0)',
        overall_support='both_fixed_windows_support', future_clean_latents_read=False,
        training=False, backward=False, vae=False, video_generation=False,
        quality_verified=False, automatic_next_stage=False)


def source_binding():
    """Actual remote small-file evidence, not a generated fixture or a visual judge."""
    review = read(REVIEW)
    expected = dict(schema='paired-turn-b-visual-review-v1', arm='B_new_pool', reviewed_by='main_assistant',
        review_scope='fixed_0_to_6_seconds_and_prespecified_turn_timestamps', execution_contract_valid=True,
        all_four_completed_and_settled=True, all_four_decoded_97=True, visually_reviewed=True,
        paired_pass_count=0, teacher_ready=False, protocol_closed=True, automatic_further_training=False)
    require(all((review.get(key) is value) if type(value) is bool else review.get(key) == value
                for key, value in expected.items()), 'actual closed negative B visual review required')
    rows = review.get('cases', [])
    require(len(rows) == 4 and {row.get('job_id') for row in rows} == set(b.CASES.values()),
            'B review must bind all four fixed cases')
    files, cases, model_identity = {str(REVIEW):sha256(REVIEW)}, {}, None
    for (target, condition), job in b.CASES.items():
        directory = b.ROOT / job
        row = next(value for value in rows if value['job_id'] == job)
        require(row.get('target') == target and row.get('condition') == condition
                and row.get('decoded_frames') == 97 and row.get('visually_reviewed') is True,
                'per-case actual B visual review missing')
        documents = {}
        for name in ('request', 'result', 'settlement'):
            path = directory / (name + '.json')
            digest = sha256(path)
            require(row.get(name + '_path') == str(path) and row.get(name + '_sha256') == digest,
                    'B review source binding changed: ' + job + '/' + name)
            files[str(path)] = digest
            documents[name] = read(path)
        request, result, settlement = (documents[name] for name in ('request', 'result', 'settlement'))
        b.validate_request(request)
        require(request['job_id'] == job and result.get('request') == request
                and result.get('mode') == b.MODE and result.get('revision') == b.REVISION
                and result.get('status') == 'completed' and result.get('frames_written') == 97
                and result.get('actual_model_forward_calls') == 80
                and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed',
                'four real completed/settled B evaluations required')
        lineage, generation = result['lineage'], result['generation']
        require(lineage.get('future_clean_latents_read') is False
                and lineage.get('future_clean_latents_supplied') is False
                and generation['first_latent_sha256_float32'] == lineage['first_latent_sha256_float32'],
                'B source must be z0-only')
        identity = {key:request[key] for key in ('checkpoint', 'checkpoint_sha256', 'training_config', 'training_config_sha256')}
        require(model_identity is None or model_identity == identity, 'B source endpoints differ')
        model_identity = identity
        cases[job] = dict(evaluation_request=str(directory / 'request.json'), target=target, condition=condition,
            first_latent_sha256_float32=lineage['first_latent_sha256_float32'],
            prompt_sha256_float32=lineage['prompt_sha256_float32'], negative_source=lineage['negative_source'],
            actual_action_sha256_float32=lineage['actual_action_sha256_float32'],
            noise_sha256_float32=generation['noise_sha256_float32'])
    require(sha256(b.TRAINING_CONFIG) == model_identity['training_config_sha256'], 'B config changed')
    files[str(b.TRAINING_CONFIG)] = model_identity['training_config_sha256']
    return dict(review_path=str(REVIEW), review_sha256=files[str(REVIEW)], files=files,
                cases=cases, model=model_identity, B_paired_pass_count=0, previous_protocol_closed=True)


def validate_request(request, directory=None):
    expected = dict(mode=MODE, experiment_revision=REVISION, source_revision=SOURCE_REVISION,
        job_id=JOB_ID, max_seconds=MAX_SECONDS, cleanup_seconds=CLEANUP_SECONDS,
        stage_bytes_limit=STAGE_BYTES, protocol=protocol())
    require(set(request) == set(expected) | {'binding'} and all(request.get(k) == v for k, v in expected.items()),
            'only the independent fixed eight-forward protocol is allowed')
    require(isinstance(request.get('binding'), dict), 'actual B evidence binding required')
    if directory is not None:
        require(contained(directory, ROOT) == ROOT / JOB_ID, 'fixed new diagnostic directory required')


def segment_noise(seed=42):
    """Reuse the existing five-draw mapping; no future clean latent is involved."""
    import torch
    from training.demo.action_backend import joint_noise_from_chunks
    require(seed == 42, 'source seed cannot change')
    rng = torch.Generator(device='cpu').manual_seed(seed)
    draws = torch.stack([torch.randn((1, 13, 48, 30, 52), generator=rng, dtype=torch.float32)
                         for _ in range(5)])
    return joint_noise_from_chunks(draws)[:, :25].clone()


def load_cases(binding):
    """Call the exact B input loader; it copies z0 only, never future clean values."""
    import torch
    loaded = {}
    for target in TARGETS:
        pair = {}
        for condition in CONDITIONS:
            job = b.CASES[target, condition]
            request = read(binding['cases'][job]['evaluation_request'])
            values = b.load_inputs(request)
            first, prompt, actions, _, negative, lineage, _, endpoint_result = values
            actual = read(b.ROOT / job / 'result.json')
            for key in ('selection', 'first_latent_sha256_float32', 'prompt_sha256_float32', 'negative_source',
                        'actual_actions', 'actual_action_sha256_float32', 'cache_binding_sha256'):
                require(lineage[key] == actual['lineage'][key], 'actual B input changed: ' + key)
            pair[condition] = dict(request=request, first=first, prompt=prompt, actions=actions,
                negative=negative, lineage=lineage, endpoint_result=endpoint_result)
        original, swapped = pair['original'], pair['jl_swap']
        require(all(torch.equal(original[key], swapped[key]) for key in ('first', 'prompt', 'negative')),
                'only J/L actions may differ inside a window')
        expected = b.a.action_condition(torch.from_numpy(original['actions']), 'jl_swap')
        require(torch.equal(expected, torch.from_numpy(swapped['actions'])), 'source swap changed other keys')
        loaded[target] = pair
    noise = segment_noise()
    digest = b.a.matched.tensor_sha(noise)
    require(all(row['noise_sha256_float32'] == digest for row in binding['cases'].values()),
            'reconstructed first-step noise differs from actual B generations')
    return loaded, noise


def prepare(*, output, source_revision=SOURCE_REVISION, max_seconds=MAX_SECONDS):
    output = contained(output, ROOT)
    require(output == ROOT / JOB_ID and not output.exists(), 'new fixed diagnostic output required')
    request = dict(mode=MODE, experiment_revision=REVISION, source_revision=source_revision, job_id=JOB_ID,
        max_seconds=max_seconds, cleanup_seconds=CLEANUP_SECONDS, stage_bytes_limit=STAGE_BYTES,
        protocol=protocol(), binding=source_binding())
    validate_request(request, output)
    loaded, noise = load_cases(request['binding'])
    del loaded, noise
    gc.collect()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'request.json', request)
    write_json(output / 'preparation.json', dict(status='cpu_prepared', request=request,
        gpu_launched=False, weights_loaded=False, future_clean_latents_read=False, tensors_saved=False,
        existing_eval_noise_verified=True, previous_protocol_closed=True))
    return request


def signal_statistics(original, swapped):
    """Actual BF16 CFG comes from TextCFGFlow; all statistics exclude latent0."""
    import torch
    keys = ('positive', 'negative', 'cfg_actual')
    tensors = [row[key] for row in (original, swapped) for key in keys]
    require(all(torch.is_tensor(value) and value.dtype == torch.bfloat16 and value.ndim == 5
                and value.shape == tensors[0].shape and value.shape[1] >= 2
                and bool(torch.isfinite(value).all()) for value in tensors), 'finite matched BF16 predictions required')
    left = {key:original[key][:, 1:].float() for key in keys}
    right = {key:swapped[key][:, 1:].float() for key in keys}
    rms = lambda value:float(value.square().mean().sqrt())
    pos, neg = left['positive'] - right['positive'], left['negative'] - right['negative']
    cfg = left['cfg_actual'] - right['cfg_actual']
    ideal = 5 * pos - 4 * neg
    rp, rn, rc, ri = (rms(value) for value in (pos, neg, cfg, ideal))
    threshold = 1e-7 * max(rms(left['positive']), rms(right['positive']), 1.)
    measurable = rp > threshold

    def alignment(x, y):
        dot = float((x * y).sum())
        denom = rms(x) * rms(y)
        return dict(dot_sum=dot, mean_dot=float((x * y).mean()),
                    cosine=None if denom == 0 else float((x * y).mean()) / denom)

    cfg_pos = alignment(cfg, pos)
    ratio = rc / rp if measurable else None
    supported = measurable and (ratio <= .25 or cfg_pos['dot_sum'] < 0)
    return dict(first_latent_excluded=True, compared_future_values=pos.numel(),
        positive_original_rms=rms(left['positive']), positive_swap_rms=rms(right['positive']),
        delta_positive_rms=rp, delta_negative_rms=rn, delta_cfg_actual_rms=rc, delta_cfg_linear_fp32_rms=ri,
        measurable_threshold=threshold, measurable=measurable,
        cfg_to_positive_ratio=ratio, linear_cfg_to_positive_ratio=ri / rp if measurable else None,
        cfg_vs_positive=cfg_pos, negative_vs_positive=alignment(neg, pos), cfg_actual_vs_linear=alignment(cfg, ideal),
        actual_minus_linear_delta_rms=rms(cfg - ideal),
        original_cfg_rounding_rms=rms(left['cfg_actual'] - (left['negative'] + 5 * (left['positive'] - left['negative']))),
        swap_cfg_rounding_rms=rms(right['cfg_actual'] - (right['negative'] + 5 * (right['positive'] - right['negative']))),
        window_first_step_hypothesis_supported=supported, visual_quality_verified=False)


def run(request, directory, lease):
    import torch
    from training.demo.backend import validate_active_authority
    from training.longforcing_lite import euler_sigmas
    from training.models.action_adapter import build_action_context
    from training.runtime import append_jsonl
    from training.studio.quality_probe import TextCFGFlow
    validate_request(request, directory)
    directory = Path(directory)
    require(not (directory / 'diagnostic-result.json').exists(), 'no repeated execution of this identity')
    began, model, calls = time.monotonic(), None, 0
    report = dict(status='running', mode=MODE, revision=REVISION, request=request, windows={},
        actual_model_forward_calls=0, model_loads=0, future_clean_latents_read=False, tensors_saved=False,
        vae_constructed=False, video_generated=False, training=False, automatic_next_stage=False,
        first_step_hypothesis_supported=False, quality_improvement_verified=False, action_correctness_verified=False)

    def phase(name, **extra):
        append_jsonl(directory / 'phases.jsonl', dict(phase=name, elapsed_seconds=time.monotonic()-began, **extra))

    def check():
        require(time.monotonic() - began < MAX_SECONDS - 15, 'finite CFG diagnostic deadline reached')
        require(b.a.matched.directory_bytes(ROOT) < STAGE_BYTES - 1_000_000, 'independent100MB stage cap reached')
        validate_active_authority(lease)

    try:
        check()
        require(os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'offline assets required')
        require(source_binding() == request['binding'], 'actual reviewed B sources changed after preparation')
        phase('inputs_begin')
        cases, noise = load_cases(request['binding'])
        phase('inputs_ready')
        source = cases['J']['original']
        phase('checkpoint_validation_begin')
        state, config, lineage = b._load_checkpoint(source['request'], source['endpoint_result'])
        report['checkpoint_lineage'] = lineage
        phase('checkpoint_validation_complete')
        check()
        require(torch.cuda.is_available(), 'guarded diagnostic requires CUDA')
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        phase('model_load_begin')
        # This is exactly ControlledPairRuntime's model constructor and weight loader,
        # without its unused VAE construction. No model/global monkey-patching.
        model, _ = b.a.paired.build_inference_model(config, state, arm='frozen')
        state.clear()
        del state
        model.to('cuda')
        torch.cuda.synchronize()
        report['model_loads'] = 1
        phase('model_load_complete')
        check()
        sigmas = euler_sigmas(40, shift=5., device=torch.device('cuda'), dtype=torch.bfloat16)
        require(float(sigmas[0]) == 1., 'must inspect actual first Euler sigma1')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for target in TARGETS:
                pair = cases[target]
                first = pair['original']['first'].to('cuda', dtype=torch.bfloat16)
                current = noise.to('cuda', dtype=torch.bfloat16).clone()
                current[:, :1] = first
                timestep = (sigmas[0] * 1000).expand(current.shape[:2]).clone()
                timestep[:, 0] = 0
                predictions = {}
                for condition in CONDITIONS:
                    source = pair[condition]
                    prompt = source['prompt'].to('cuda', dtype=torch.bfloat16)
                    negative = source['negative'].to('cuda', dtype=torch.bfloat16)
                    context = build_action_context(torch.from_numpy(source['actions']).unsqueeze(0),
                        height=480, width=832, device='cuda', dtype=torch.bfloat16)
                    conditions = dict(prompt_embeds=[prompt], act_context=context, act_context_scale=1.)
                    captured = {}

                    def record_forward(*args, conditional_dict, **kwargs):
                        nonlocal calls
                        check()
                        branch = 'positive' if not captured else 'negative'
                        require(len(captured) < 2 and calls < 8 and conditional_dict['act_context'] is context
                                and conditional_dict['act_context_scale'] == 1., 'fixed branch actions/forward count changed')
                        expected_text = prompt if branch == 'positive' else negative
                        require(conditional_dict['prompt_embeds'][0] is expected_text, 'text branch differs from source')
                        phase('forward_begin', target=target, condition=condition, branch=branch, call=calls + 1)
                        calls += 1
                        report['actual_model_forward_calls'] = calls
                        output = model(*args, conditional_dict=conditional_dict, **kwargs)
                        require(output[0].dtype == torch.bfloat16 and tuple(output[0].shape) == (1,25,48,30,52),
                                'actual source BF16 velocity shape differs')
                        captured[branch] = output[0].detach().cpu()
                        phase('forward_complete', target=target, condition=condition, branch=branch, call=calls)
                        return output

                    flow = TextCFGFlow(record_forward, 5, negative)
                    guided, _ = flow(current, conditional_dict=conditions, timestep=timestep,
                                     replace_first_timestep_and_noise_latents=True)
                    require(flow.forward_calls == 2 and len(captured) == 2 and guided.dtype == torch.bfloat16,
                            'exact positive+negative source CFG arithmetic required')
                    captured['cfg_actual'] = guided.detach().cpu()
                    predictions[condition] = captured
                    del guided, flow, captured, context, conditions, prompt, negative
                report['windows'][target] = signal_statistics(predictions['original'], predictions['jl_swap'])
                phase('window_statistics_complete', target=target, statistics=report['windows'][target])
                del predictions, current, first, timestep
        require(calls == 8 and report['model_loads'] == 1, 'must complete exactly eight forwards and one model load')
        report.update(status='completed', first_step_hypothesis_supported=all(
            report['windows'][target]['window_first_step_hypothesis_supported'] for target in TARGETS),
            both_windows_measurable=all(report['windows'][target]['measurable'] for target in TARGETS),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            interpretation='fixed_first_step_only;not_video_quality_control_success_or_permission_for_more_training')
        check()
    except BaseException as error:
        report.update(status='failed', failure_type=type(error).__name__, error=str(error))
        phase('failed', failure_type=type(error).__name__, error=str(error))
        raise
    finally:
        phase('cleanup_begin')
        model = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        report['elapsed_seconds'] = time.monotonic() - began
        write_json(directory / 'diagnostic-result.json', report)
        phase('cleanup_complete')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-revision', default=SOURCE_REVISION)
    parser.add_argument('--max-seconds', type=int, default=MAX_SECONDS)
    args = vars(parser.parse_args())
    print(json.dumps(prepare(**args)))


if __name__ == '__main__':
    main()
