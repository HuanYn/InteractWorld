"""Fixed round20/L fresh-original versus neutral text; no training or automatic J.

Preparation is CPU-only. Encoding and video generation require separate existing
Creator GPU1/2 leases. Source master requests are validated as themselves before
an explicit new text condition is applied; old prompt lineage is never relabeled.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import os
from pathlib import Path
import re
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.studio import master_writeback_eval as master
from training.studio import prompt_clause_probe as text_probe

ROOT = master.ROOT.parent / 'teacher-neutral-caption-v1'
ENCODE_MODE, EVAL_MODE = 'neutral_caption_encode', 'neutral_caption_eval'
REVISION = 'round20-L-fresh-original-versus-neutral-v1'
SOURCE_REVISION = 'creator-neutral-caption-v1'
ENCODE_JOB = 'cncap-encode-v1'
EVAL_JOBS = {'original': 'cncap-l-original-v1', 'jl_swap': 'cncap-l-swap-v1'}
CASE_NAMES = ('fresh_original', 'neutral')
EDITS = (
    ('stands centered in the road', 'is centered in the road'),
    ('with short brown hair and a neutral posture; his presence is static and integrated into the setting.',
     'with short brown hair.'),
)
SCHEMA = 'neutral_caption_fresh_pair_cache_v1'
STAGE_BYTES = 100_000_000
CACHE_BYTES = 10_000_000
VIDEO_JOB_BYTES = 40_000_000
read = master.read


def edit_prompt(original):
    require(isinstance(original, str) and all(original.count(old) == 1 for old, _ in EDITS),
            'both fixed static clauses must each occur exactly once')
    value = original
    for old, new in EDITS:
        value = value.replace(old, new, 1)
    return value


def source_path(condition):
    require(condition in EVAL_JOBS, 'only original or JL-only swap on L allowed')
    return master.ROOT / master.CASES['round', 'L', condition]


def source_records(condition, binding=None):
    path = source_path(condition)
    actual = dict(job=str(path), **{name + '_sha256': sha256(path / (name + '.json'))
                                  for name in ('request', 'result', 'settlement')})
    require(binding is None or binding == actual, 'fixed completed round source record changed')
    req, result, settlement = (read(path / (name + '.json')) for name in ('request', 'result', 'settlement'))
    master.validate_request(req)
    require(req['arm'] == 'round' and req['target'] == 'L' and req['condition'] == condition
            and result.get('mode') == master.MODE and result.get('revision') == master.REVISION
            and result.get('request') == req and result.get('status') == 'completed'
            and result.get('frames_written') == 97 and result.get('actual_model_forward_calls') == 80
            and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed',
            'completed and settled actual round20 L source required')
    return req, result, actual


def load_source(condition, binding):
    req, result, _ = source_records(condition, binding)
    values = master.load_inputs(req)  # Actual old request, not a fabricated new mode.
    lineage = values[5]
    for key in ('selection', 'cache_binding_sha256', 'first_latent_sha256_float32', 'prompt_sha256_float32',
                'prompt_text', 'negative_source', 'original_action_sha256_float32', 'actual_action_sha256_float32',
                'original_actions', 'actual_actions', 'actual_action_condition'):
        require(lineage[key] == result['lineage'][key], 'validated master input changed: ' + key)
    require(lineage['selection']['episode_id'] == '0328fbd39403bc7955f73eca6a9524f2'
            and lineage['selection']['source_start'] == 210
            and lineage['future_clean_latents_read'] is False and lineage['future_clean_latents_supplied'] is False,
            'only exact L/source210 cached z0 input allowed')
    return req, result, values


def fixed_fields(mode, condition):
    require(mode in (ENCODE_MODE, EVAL_MODE), 'explicit neutral caption mode required')
    require(condition is None if mode == ENCODE_MODE else condition in EVAL_JOBS, 'wrong job condition')
    return dict(mode=mode, experiment_revision=REVISION, source_revision=SOURCE_REVISION,
        job_id=ENCODE_JOB if mode == ENCODE_MODE else EVAL_JOBS[condition], condition=condition,
        arm='round', target='L', encoder_root=str(text_probe.ENCODER_ROOT),
        cases=list(CASE_NAMES), seed=42, steps=40, cfg_scale=5, future_frames=96, solver='flow_euler',
        residency='phase', fastpath=False, cleanup_seconds=60)


def validate_request(request):
    expected = fixed_fields(request.get('mode'), request.get('condition'))
    require(all(request.get(k) == v and type(request.get(k)) is type(v) for k, v in expected.items()),
            'fixed round20/L/case-order/generation settings changed')
    extra = {'max_seconds', 'source_bindings', 'prompts', 'encoder_receipt_sha256'}
    if request['mode'] == EVAL_MODE:
        extra.add('encoding_binding')
    require(set(request) == set(expected) | extra, 'unapproved caption override')
    cap = 900 if request['mode'] == ENCODE_MODE else 1200
    require(type(request['max_seconds']) is int and 60 <= request['max_seconds'] <= cap,
            'fixed finite per-job deadline required')
    require(set(request['source_bindings']) == set(EVAL_JOBS), 'both fixed source records required')
    for condition, row in request['source_bindings'].items():
        require(set(row) == {'job', 'request_sha256', 'result_sha256', 'settlement_sha256'}
                and row['job'] == str(source_path(condition)), 'source record path differs')
        require(all(isinstance(row[k], str) and re.fullmatch('[0-9a-f]{64}', row[k])
                    for k in ('request_sha256', 'result_sha256', 'settlement_sha256')), 'small record hashes required')
    prompts = request['prompts']
    require(set(prompts) == set(CASE_NAMES) and prompts['neutral'] == edit_prompt(prompts['fresh_original']),
            'only exact two static-clause edits allowed')
    require(isinstance(request['encoder_receipt_sha256'], str)
            and re.fullmatch('[0-9a-f]{64}', request['encoder_receipt_sha256']), 'encoder receipt binding required')
    if request['mode'] == EVAL_MODE:
        binding = request['encoding_binding']
        require(set(binding) == {'job', 'request_sha256', 'result_sha256', 'settlement_sha256',
                                 'encoding_sha256', 'cache_sha256', 'cache_stat'}
                and binding['job'] == str(ROOT / ENCODE_JOB), 'only completed fixed text cache permitted')
        require(all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v)
                    for k, v in binding.items() if k.endswith('_sha256')), 'text cache hashes required')
        require(set(binding['cache_stat']) == {'bytes', 'mtime_ns'}
                and all(type(v) is int and v > 0 for v in binding['cache_stat'].values())
                and binding['cache_stat']['bytes'] <= CACHE_BYTES, 'small text-only cache required')


def completed_encoding():
    directory = ROOT / ENCODE_JOB
    req, result, settlement, encoding = (read(directory / name) for name in
        ('request.json', 'result.json', 'settlement.json', 'encoding.json'))
    validate_request(req)
    require(req['mode'] == ENCODE_MODE and result.get('status') == 'completed'
            and result.get('mode') == ENCODE_MODE and result.get('request') == req
            and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed'
            and encoding.get('status') == 'completed' and encoding.get('schema') == SCHEMA
            and encoding.get('request') == req and result.get('encoding') == encoding,
            'fresh encoding must complete and settle before video preparation')
    path = directory / 'embeddings.pt'
    identity = master.stat_identity(path)
    require(identity == encoding['cache_stat'] and 0 < identity['bytes'] <= CACHE_BYTES,
            'small fresh text cache stat changed')
    binding = dict(job=str(directory), **{name + '_sha256': sha256(directory / (name + '.json'))
        for name in ('request', 'result', 'settlement', 'encoding')},
        cache_sha256=encoding['cache_sha256'], cache_stat=identity)
    return req, encoding, binding


def prepare(*, output, source_revision, mode, condition=None, max_seconds):
    output = contained(output, ROOT)
    request = fixed_fields(mode, condition)
    require(source_revision == SOURCE_REVISION and output == ROOT / request['job_id'] and not output.exists(),
            'new fixed job path and source revision required; no overwrite/retry')
    records = {key: source_records(key) for key in EVAL_JOBS}
    text = records['original'][1]['lineage']['prompt_text']
    require(text == records['jl_swap'][1]['lineage']['prompt_text'], 'both action cases must share original text')
    assets = text_probe.validate_encoder_assets(text_probe.ENCODER_ROOT)
    request.update(max_seconds=max_seconds, source_bindings={k: v[2] for k, v in records.items()},
        prompts=dict(fresh_original=text, neutral=edit_prompt(text)), encoder_receipt_sha256=sha256(assets['receipt']))
    if mode == EVAL_MODE:
        encoded_request, _, binding = completed_encoding()
        require(all(request[k] == encoded_request[k] for k in ('prompts', 'source_bindings', 'encoder_receipt_sha256')),
                'encoding/source contract differs from requested evaluation')
        request['encoding_binding'] = binding
    validate_request(request)
    # Existing loader validates actions/z0/text/cache lineage, but loads no checkpoint.
    for key in (EVAL_JOBS if mode == ENCODE_MODE else (condition,)):
        _, _, values = load_source(key, request['source_bindings'][key])
        require(values[5]['prompt_text'] == text, 'source prompt changed during CPU preparation')
        del values
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'request.json', request)
    write_json(output / 'preparation.json', dict(status='cpu_prepared', request=request,
        weights_loaded=False, encoder_loaded=False, gpu_launched=False, automatic_next_stage=False))
    return request


def prepare_encode(*, output, source_revision, max_seconds=900):
    return prepare(output=output, source_revision=source_revision, mode=ENCODE_MODE, max_seconds=max_seconds)


def prepare_eval(*, output, source_revision, condition, max_seconds=1200):
    return prepare(output=output, source_revision=source_revision, mode=EVAL_MODE,
                   condition=condition, max_seconds=max_seconds)


def fresh_embeddings(prompts, assets, directory, check):
    """Original WanTextEncoder numerical path, without modifying old CASE_NAMES."""
    import torch
    from utils.wan_wrapper import WanTextEncoder
    encoder = output = None
    embeddings, rows = {}, []
    began = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        check()
        encoder = WanTextEncoder(tokenizer_path=str(Path(assets['root']) / 'google/umt5-xxl'),
            encoder_pth_path=str(Path(assets['root']) / 'models_t5_umt5-xxl-enc-bf16.pth'))
        encoder.eval().requires_grad_(False).to(device='cuda', dtype=torch.bfloat16)
        torch.cuda.synchronize()
        load_seconds = time.monotonic() - began
        for name in CASE_NAMES:
            check()
            ids, mask = encoder.tokenizer([prompts[name]], return_mask=True, add_special_tokens=True)
            require(tuple(ids.shape) == tuple(mask.shape) == (1, 512), 'fixed512 tokenization required')
            tokens = int(mask.gt(0).sum())
            require(1 <= tokens <= 512 and bool((mask[0, :tokens] > 0).all())
                    and not bool(mask[0, tokens:].count_nonzero()), 'valid padded prefix mask required')
            tick = time.monotonic()
            with torch.inference_mode():
                output = encoder([prompts[name]])['prompt_embeds']
            torch.cuda.synchronize()
            require(tuple(output.shape) == (1, 512, 4096), 'fixed encoder output shape required')
            embeddings[name] = output[0].detach().cpu().clone()
            output = None
            rows.append(dict(name=name, prompt=prompts[name], encoding_seconds=time.monotonic() - tick,
                **text_probe.embedding_metadata(embeddings[name], tokens)))
            write_json(directory / 'progress.json', dict(status='running', phase='encoding', cases=rows))
        peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        require(reserved < 23 * 1024**3, 'text encoding reserved memory must remain below23GiB')
    finally:
        output = encoder = None
        gc.collect()
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return embeddings, dict(cases=rows, model_load_seconds=load_seconds,
        elapsed_seconds=time.monotonic() - began, peak_allocated_bytes=peak, peak_reserved_bytes=reserved,
        allocated_after_release_bytes=torch.cuda.memory_allocated(), t5_released_before_video_runtime=True,
        encoding='WanTextEncoder.eval/frozen/CUDA-BF16;inference_mode;separate batch1;no additional autocast')


def run_encode(request, directory, check):
    import torch
    assets = text_probe.validate_encoder_assets(request['encoder_root'])
    require(sha256(assets['receipt']) == request['encoder_receipt_sha256'], 'encoder receipt changed')
    originals = []
    for key in EVAL_JOBS:
        _, _, values = load_source(key, request['source_bindings'][key])
        require(values[5]['prompt_text'] == request['prompts']['fresh_original'], 'original text changed')
        originals.append(values[1])
    require(torch.equal(originals[0], originals[1]), 'both L cases must share the same cached text')
    embeddings, measurement = fresh_embeddings(request['prompts'], assets, directory, check)
    comparison = text_probe.velocity_difference(originals[0].float(), embeddings['fresh_original'].float())
    comparison.update(scope='embedding numeric difference only;main comparison uses both fresh texts',
                      existing_cache_dtype=str(originals[0].dtype), fresh_dtype=str(embeddings['fresh_original'].dtype))
    payload = dict(schema=SCHEMA, prompts=request['prompts'], embeddings=embeddings,
                   source_bindings=request['source_bindings'])
    path = directory / 'embeddings.pt'
    require(not path.exists(), 'new cache only; no overwrite')
    torch.save(payload, path)
    require(path.stat().st_size <= CACHE_BYTES, 'text cache output limit exceeded')
    encoding = dict(status='completed', schema=SCHEMA, request=request, encoder_assets=assets,
        **measurement, fresh_original_vs_existing_cache=comparison, cache_path=str(path),
        cache_sha256=sha256(path), cache_stat=master.stat_identity(path), old_cache_modified=False,
        effective_comparison='fresh_original versus neutral;old cached videos are context only')
    check()
    write_json(directory / 'encoding.json', encoding)
    return dict(status='completed', mode=ENCODE_MODE, request=request, encoding=encoding,
                quality_improvement_verified=False, automatic_next_stage=False)


def load_embeddings(request):
    import torch
    encoded_request, encoding, binding = completed_encoding()
    require(binding == request['encoding_binding'] and all(request[k] == encoded_request[k]
        for k in ('source_bindings', 'prompts', 'encoder_receipt_sha256')), 'completed cache binding changed')
    path = ROOT / ENCODE_JOB / 'embeddings.pt'
    require(sha256(path) == binding['cache_sha256'], 'small text cache bytes changed')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    require(set(payload) == {'schema', 'prompts', 'embeddings', 'source_bindings'} and payload['schema'] == SCHEMA
            and payload['prompts'] == request['prompts'] and payload['source_bindings'] == request['source_bindings']
            and set(payload['embeddings']) == set(CASE_NAMES), 'fresh pair cache contract differs')
    rows = {row['name']: row for row in encoding['cases']}
    require(set(rows) == set(CASE_NAMES), 'both freshly encoded conditions required')
    for name in CASE_NAMES:
        observed = text_probe.embedding_metadata(payload['embeddings'][name], rows[name]['token_count'])
        require(rows[name]['prompt'] == request['prompts'][name] and all(rows[name][k] == v for k, v in observed.items()),
                'fresh prompt embedding/padding metadata changed')
    return payload['embeddings'], rows


def validate_case(generation, before_calls, after_calls, baseline):
    require(type(before_calls) is int and type(after_calls) is int and before_calls in (0, 80),
            'expected cumulative case start0 or80')
    # Preserve the old validator's exact80 requirement; provide this case's delta.
    master.b.validate_generation(generation, after_calls - before_calls, baseline)
    require(after_calls == before_calls + 80, 'each text case must execute80 actual model forwards')
    return dict(actual_model_forward_calls=after_calls - before_calls,
                cumulative_model_forward_calls=after_calls)


def run_eval(request, directory, check):
    import numpy as np
    import torch
    from PIL import Image
    from training.eval.rollout15s import ffmpeg_writer_factory
    condition = request['condition']
    source_req, source_result, values = load_source(condition, request['source_bindings'][condition])
    first, old_prompt, actions, initial, negative, old_lineage, _, endpoint = values
    require(old_lineage['prompt_text'] == request['prompts']['fresh_original'], 'source text changed')
    embeddings, encoded_rows = load_embeddings(request)
    state, config, checkpoint_lineage = master._load_checkpoint(source_req, endpoint)
    runtime = engine = None
    results = []
    try:
        check()
        tick = time.monotonic()
        runtime = master.b.a.paired.ControlledPairRuntime(state, config, negative, arm='frozen')
        state.clear()
        gc.collect()
        load_seconds = time.monotonic() - tick
        engine = master.b.a.matched.CachedInitialEngine({}, runtime.config, residency='phase', fastpath=False,
            _runtime=runtime, cached_first=first,
            _writer_factory=lambda *args: master.b.a.matched.BoundedPreservingWriter(ffmpeg_writer_factory(*args)))
        engine.load_seconds = load_seconds
        for name in CASE_NAMES:
            check(minimum_remaining=270)
            target = directory / name
            target.mkdir(exist_ok=False)
            actual_prompt = dict(encoded_rows[name], kind='explicit_fresh_caption_override', text=request['prompts'][name],
                text_sha256=hashlib.sha256(request['prompts'][name].encode('utf-8')).hexdigest(),
                encoding_binding=request['encoding_binding'])
            # Do not rewrite old prompt_text/hash: source and actual condition remain separate.
            inputs = dict(request=request, source_request=source_req, source_lineage=copy.deepcopy(old_lineage),
                checkpoint_lineage=checkpoint_lineage, actual_prompt_condition=actual_prompt,
                actual_actions=actions.tolist(), future_clean_latents_supplied=False, old_cache_modified=False,
                negative_source=old_lineage['negative_source'], seed=42)
            Image.fromarray(initial).save(target / 'initial.png')
            np.save(target / 'actions.npy', actions, allow_pickle=False)
            write_json(target / 'inputs.json', inputs)
            start_calls = runtime.forward_calls
            tick = time.monotonic()
            generation, endpoint_rgb = engine.run_segment(initial=initial, prompt=embeddings[name],
                actions=actions.copy(), seed=42, output_path=target / 'raw.mp4', steps=40)
            del endpoint_rgb
            calls = validate_case(generation, start_calls, runtime.forward_calls, source_result)
            generation.update(**calls, initial_condition='same_round_L_cached_z0;RGB0_display_only',
                continuation='independent_cached_z0_condition;no_future_ground_truth', vae_encoding_used=False,
                actual_prompt_condition=actual_prompt, original_prompt_lineage_unchanged=True)
            generation['sampler'].update(cfg='text_classifier_free', cfg_scale=5, model_forwards_per_step=2)
            require(torch.cuda.max_memory_reserved() < 23 * 1024**3, 'video reserved memory must remain below23GiB')
            row = dict(status='completed', mode=EVAL_MODE, revision=REVISION, case=name, condition=condition,
                generation=generation, raw_video=str(target / 'raw.mp4'), actual_prompt_condition=actual_prompt,
                elapsed_seconds=time.monotonic() - tick, peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                **calls, quality_evaluation='awaiting_actual_visual_review', quality_improvement_verified=False)
            results.append(row)
            write_json(target / 'result.json', row)
            write_json(directory / 'progress.json', dict(status='running', phase='generated', cases=results))
        require(runtime.forward_calls == 160 and len(results) == 2, 'two text conditions/160 actual forwards required')
        return dict(status='completed', mode=EVAL_MODE, revision=REVISION, request=request,
            source_request=source_req, source_lineage=old_lineage, checkpoint_lineage=checkpoint_lineage,
            cases=results, actual_model_forward_calls=160, model_load_count=1, load_seconds=load_seconds,
            encoding_binding=request['encoding_binding'], old_cache_modified=False,
            effective_comparison='fresh_original versus neutral;old cached video is context only',
            generation_stage_t5_loaded=False, future_clean_latents_supplied=False,
            quality_improvement_verified=False, action_correctness_verified=False, generalization_verified=False,
            realtime=False, automatic_next_stage=False)
    except BaseException as error:
        write_json(directory / 'failure.json', dict(status='failed', cases=results,
            failure_type=type(error).__name__, error=str(error),
            partial_generation=copy.deepcopy(engine.last_run_report) if engine is not None else None))
        raise
    finally:
        if engine is not None:
            engine.close()
        elif runtime is not None:
            runtime.close()


def run(request, directory, lease):
    from training.demo.backend import validate_active_authority
    import torch
    validate_request(request)
    require(type(lease.get('gpu_index')) is int and lease['gpu_index'] in (1, 2),
            'only original Creator GPU1/2 authority applies; neverGPU0')
    require(os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1',
            'offline existing assets only')
    directory = contained(directory, ROOT)
    require(directory == ROOT / request['job_id'] and directory.is_dir()
            and not any((directory / name).exists() for name in ('progress.json', 'encoding.json', 'embeddings.pt', *CASE_NAMES)),
            'new prepared job required; no automatic retry')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one leased GPU required')
    began = time.monotonic()
    def check(*, minimum_remaining=30):
        require(request['max_seconds'] - (time.monotonic() - began) >= minimum_remaining, 'finite caption deadline')
        require(master.b.a.matched.directory_bytes(ROOT) < STAGE_BYTES - 100_000, 'stage100MB cap exceeded')
        require(master.b.a.matched.directory_bytes(directory) < (
            CACHE_BYTES + 1_000_000 if request['mode'] == ENCODE_MODE else VIDEO_JOB_BYTES), 'job output cap exceeded')
        validate_active_authority(lease)
    check()
    write_json(directory / 'progress.json', dict(status='running', phase='starting', mode=request['mode']))
    try:
        result = run_encode(request, directory, check) if request['mode'] == ENCODE_MODE else run_eval(request, directory, check)
        check()
        result['elapsed_seconds'] = time.monotonic() - began
        write_json(directory / 'progress.json', dict(status='completed', mode=request['mode'], elapsed_seconds=result['elapsed_seconds']))
        return result
    except BaseException as error:
        write_json(directory / 'progress.json', dict(status='failed', mode=request['mode'],
            elapsed_seconds=time.monotonic() - began, failure_type=type(error).__name__, error=str(error),
            automatic_retry=False))
        raise
