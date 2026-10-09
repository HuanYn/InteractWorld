"""Two fresh-text six-second R005 clips differing by one exact static clause.

Old prompt caches, checkpoint lineage and default inference are never modified.
This runner is launched only through the existing finite Creator lease/watchdog.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import re
import time

from training.demo.contracts import contained, require, write_json
from training.studio.batched_cfg_probe import velocity_difference
from training.studio.overlap_probe import DATA_ROOT, HEIGHT, R005_SHA, R005_SOURCE_JOB, SETTINGS, WIDTH
from training.studio.quality_probe import DEFAULT_NEGATIVE_CACHE, QualityRuntime, annotate_report, load_negative

OLD_CLAUSE = 'positioned as a static element within the vast, somber landscape'
NEW_CLAUSE = 'positioned within the vast, somber landscape'
ENCODER_REVISION = '921dbaf3f1674a56f47e83fb80a34bac8a8f203e'
ENCODER_ROOT = DATA_ROOT / 'agent-runtime/event-condition-ablation-20260928-v1/text-encoder'
ENCODER_FILES = {
    'google/umt5-xxl/special_tokens_map.json': 6623,
    'google/umt5-xxl/spiece.model': 4548313,
    'google/umt5-xxl/tokenizer.json': 16837417,
    'google/umt5-xxl/tokenizer_config.json': 61728,
    'models_t5_umt5-xxl-enc-bf16.pth': 11361920418,
}
CASE_NAMES = ('original_fresh', 'without_static_clause')
REVISION = 'R005-static-clause-fresh-text-pair6s-v1'
FIXED = dict(mode='prompt_clause_probe', max_seconds=1000, source_job=str(R005_SOURCE_JOB),
    checkpoint_sha256=R005_SHA, encoder_root=str(ENCODER_ROOT), negative_cache=DEFAULT_NEGATIVE_CACHE,
    residency='co_resident', fastpath=False, width=WIDTH, height=HEIGHT, seed=42, future_frames=96,
    cases=list(CASE_NAMES), minimum_case_remaining_seconds=270, **SETTINGS)


def validate_request(request):
    require(set(request) == set(FIXED) | {'job_id', 'source_revision', 'experiment_revision'}, 'unknown prompt probe fields')
    require(all(request[key] == value and type(request[key]) is type(value) for key, value in FIXED.items()),
            'prompt probe fixes two6s/R005/singleclause/seed42/40EulerCFG5/1000seconds')
    require(isinstance(request['job_id'], str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', request['job_id'])
            and isinstance(request['source_revision'], str) and bool(request['source_revision'])
            and request['experiment_revision'] == REVISION, 'invalid prompt probe identity')


def edit_static_clause(original):
    require(isinstance(original, str) and original.count(OLD_CLAUSE) == 1, 'exact static clause must occur once')
    return original.replace(OLD_CLAUSE, NEW_CLAUSE, 1)


def validate_encoder_assets(root, *, expected_root=ENCODER_ROOT, data_root=DATA_ROOT, expected_files=None):
    """Reuse the existing receipt and exact file sizes, never rehash large weights.

    Expected-root/files injection is for small CPU fixtures only. Production run
    uses the fixed request root and all five source-pinned real sizes above.
    """
    expected_files = ENCODER_FILES if expected_files is None else expected_files
    root = contained(root, data_root)
    require(root == Path(expected_root).resolve() and root.is_dir(), 'encoder root differs from the fixed data path')
    receipt_path = contained(root.parent / 'encoder-fetch.json', data_root)
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    require(receipt.get('status') == 'completed' and receipt.get('repo') == 'Wan-AI/Wan2.2-TI2V-5B'
            and receipt.get('revision') == ENCODER_REVISION and receipt.get('output') == str(expected_root),
            'encoder completed source/revision/output receipt differs')
    records = receipt.get('completed')
    require(isinstance(records, list) and len(records) == len(expected_files)
            and all(isinstance(row, dict) for row in records)
            and {row.get('name') for row in records} == set(expected_files), 'encoder must contain exactly five bound file records')
    files = []
    for row in records:
        name, size = row['name'], row.get('bytes')
        require(type(size) is int and size == expected_files[name], 'encoder receipt file size differs')
        path = contained(root / name, root)
        require(path.is_file() and path.stat().st_size == size, 'encoder asset missing or size changed')
        require(isinstance(row.get('sha256'), str) and re.fullmatch(r'[0-9a-f]{64}', row['sha256']), 'encoder receipt digest missing')
        files.append(dict(name=name, bytes=size, prior_download_sha256=row['sha256']))
    return dict(root=str(root), receipt=str(receipt_path), repo=receipt['repo'], revision=receipt['revision'],
        files=files, verification='existing completed download receipt plus exact paths/file sizes;no full rehash',
        full_weight_rehash=False, newly_downloaded=False)


def embedding_metadata(embedding, tokens):
    import torch
    from training.demo.action_backend import tensor_sha
    require(torch.is_tensor(embedding) and embedding.device.type == 'cpu' and embedding.dtype == torch.bfloat16
            and tuple(embedding.shape) == (512, 4096) and bool(torch.isfinite(embedding).all()),
            'fresh text embedding must be finite CPU BF16[512,4096]')
    require(type(tokens) is int and 1 <= tokens <= 512 and not bool(torch.count_nonzero(embedding[tokens:])),
            'invalid text token count or nonzero padding')
    return dict(shape=list(embedding.shape), dtype=str(embedding.dtype), token_count=tokens, padding_rows=512 - tokens,
                finite=True, padding_zero=True, embedding_sha256_float32=tensor_sha(embedding))


def encode_pair(prompts, assets, directory, check):
    """Match the original cache's separate B1 inference_mode calls, without autocast."""
    import torch
    from utils.wan_wrapper import WanTextEncoder
    encoder, output = None, None
    began, embeddings, rows = time.monotonic(), {}, []
    check()
    torch.cuda.reset_peak_memory_stats()
    try:
        encoder = WanTextEncoder(tokenizer_path=str(Path(assets['root']) / 'google/umt5-xxl'),
            encoder_pth_path=str(Path(assets['root']) / 'models_t5_umt5-xxl-enc-bf16.pth'))
        encoder.eval().requires_grad_(False).to(device='cuda', dtype=torch.bfloat16)
        torch.cuda.synchronize()
        load_seconds = time.monotonic() - began
        for name in CASE_NAMES:
            check()
            ids, mask = encoder.tokenizer([prompts[name]], return_mask=True, add_special_tokens=True)
            require(tuple(ids.shape) == tuple(mask.shape) == (1, 512), 'fixed tokenizer shape differs')
            count = int(mask.gt(0).sum())
            require(1 <= count <= 512 and bool((mask[0, :count] > 0).all())
                    and not bool(mask[0, count:].count_nonzero()), 'token mask must be a valid padded prefix')
            tick = time.monotonic()
            with torch.inference_mode():
                output = encoder([prompts[name]])['prompt_embeds']
            torch.cuda.synchronize()
            seconds = time.monotonic() - tick
            require(tuple(output.shape) == (1, 512, 4096), 'encoder output batch/shape differs')
            embeddings[name] = output[0].detach().cpu().clone()
            output = None
            rows.append(dict(name=name, prompt=prompts[name], encoding_seconds=seconds,
                             **embedding_metadata(embeddings[name], count)))
            write_json(directory / 'progress.json', dict(status='running', phase='encoding_text', encoded=rows))
        peak = int(torch.cuda.max_memory_allocated())
    finally:
        # No encoder module survives into video-runtime construction, including on error.
        output = encoder = None
        gc.collect()
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    report = dict(status='completed', cases=rows, model_load_seconds=load_seconds,
        elapsed_seconds=time.monotonic() - began, peak_vram_bytes=peak,
        t5_loaded_for_prompt_preparation=True, t5_released_before_video_runtime=True,
        allocated_after_release_bytes=int(torch.cuda.memory_allocated()),
        encoding='WanTextEncoder.eval/frozen/CUDA-BF16;inference_mode;separate batch1;no additional autocast',
        original_cache_modified=False, encoder_assets=assets)
    return embeddings, report


def run(request, directory, lease):
    import hashlib
    import numpy as np
    import torch
    from PIL import Image
    from training.demo.backend import validate_active_authority
    from training.demo.latency_engine import ReusableActionEngine
    from training.studio.worker import read_source
    validate_request(request)
    directory = contained(directory, DATA_ROOT)
    require(directory.name == request['job_id'] and directory.is_dir()
            and not any((directory / name).exists() for name in ('inputs.json', 'embeddings.pt', 'progress.json')),
            'new prompt probe directory required;no overwriting or retries')
    began, runtime, engine, results = time.monotonic(), None, None, []

    def check(*, next_case=False):
        remaining = request['max_seconds'] - (time.monotonic() - began)
        require(remaining >= (270 if next_case else 30), 'prompt probe deadline;no next case or retry')
        validate_active_authority(lease)

    check()
    assets = validate_encoder_assets(request['encoder_root'])
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one leased CUDA GPU required')
    state, config, cached_prompt, lineage, initial, actions, scene = read_source(request)
    require(lineage['sha256'] == R005_SHA and lineage['step'] == 1040 and scene['seed'] == 42
            and initial.shape == (HEIGHT, WIDTH, 3) and initial.dtype == np.uint8, 'actual originalR005 source differs')
    require(Path(config.base_model_path).resolve().is_relative_to(DATA_ROOT.resolve()), 'model assets must remain on data disk')
    original = scene['prompt']
    prompts = dict(zip(CASE_NAMES, [original, edit_static_clause(original)]))
    selected_actions = actions[:96].copy()
    expected = np.zeros((96, 8), np.float32)
    expected[:, [0, 5]] = 1
    require(selected_actions.dtype == np.float32 and np.array_equal(selected_actions, expected), 'source first96 must remain WJ')
    negative, negative_source = load_negative(request['negative_cache'], config.base_model_path)
    inputs = dict(mode='prompt_clause_probe', source_job=request['source_job'], checkpoint=lineage['path'],
        checkpoint_sha256=R005_SHA, checkpoint_step=1040, scene_id=scene['scene_id'], original_prompt=original,
        original_prompt_condition=copy.deepcopy(lineage.get('prompt_condition')),
        actual_prompts=prompts, runtime_text_override=True, old_prompt_cache_modified=False,
        prompt_change=dict(old_clause=OLD_CLAUSE, new_clause=NEW_CLAUSE, exact_occurrences=1, no_other_text_change=True),
        encoder_assets=assets, negative_prompt=negative_source, seed=42, width=WIDTH, height=HEIGHT,
        settings=SETTINGS, residency='co_resident', fastpath=False, expected_frames_per_case=97,
        actual_actions=[dict(frames=96, keys=['W', 'J'])], source_revision=request['source_revision'],
        experiment_revision=REVISION, effective_comparison='fresh_original versus fresh_edited;old_cached baseline is context only')
    write_json(directory / 'inputs.json', inputs)
    Image.fromarray(initial).save(directory / 'initial.png')
    np.save(directory / 'actions.npy', selected_actions, allow_pickle=False)
    embeddings, encoding = encode_pair(prompts, assets, directory, check)
    require(torch.is_tensor(cached_prompt) and cached_prompt.device.type == 'cpu'
            and tuple(cached_prompt.shape) == (512, 4096), 'original cached prompt shape/device differs')
    encoding['fresh_original_vs_existing_cache'] = dict(
        velocity_difference(cached_prompt.float(), embeddings[CASE_NAMES[0]].float()),
        scope='numeric encoding difference only;not a visual-quality gate;exact equality is not required',
        original_cache_dtype=str(cached_prompt.dtype), fresh_dtype=str(embeddings[CASE_NAMES[0]].dtype))
    write_json(directory / 'encoding.json', encoding)
    torch.save(dict(prompts=prompts, embeddings=embeddings, source_model=config.base_model_path,
        encoder_assets=assets, original_cache_modified=False), directory / 'embeddings.pt')
    inputs['encoding_receipt'] = str(directory / 'encoding.json')
    write_json(directory / 'inputs.json', inputs)
    try:
        check(next_case=True)
        tick = time.monotonic()
        runtime = QualityRuntime(state, config, 'co_resident', negative)
        runtime.synchronize()
        cold = time.monotonic() - tick
        engine = ReusableActionEngine(state, config, residency='co_resident', fastpath=False, _runtime=runtime)
        engine.load_seconds = cold
        for name in CASE_NAMES:
            check(next_case=True)
            target = contained(directory / name, directory)
            target.mkdir(exist_ok=False)
            case = dict(name=name, mode='segment6', **SETTINGS)
            runtime.configure(case)
            engine.action_scale = .03
            condition = dict(kind='fresh_reencoded_runtime_text_override', text=prompts[name],
                text_sha256=hashlib.sha256(prompts[name].encode('utf-8')).hexdigest(), encoder_assets=assets,
                **next(row for row in encoding['cases'] if row['name'] == name))
            case_inputs = dict(case=case, actual_prompt_condition=condition,
                original_prompt_condition=inputs['original_prompt_condition'], training_prompt_lineage_unchanged=True,
                initial=str(directory / 'initial.png'), actions=str(directory / 'actions.npy'), seed=42,
                checkpoint_sha256=R005_SHA, t5_loaded_for_prompt_preparation=True, generation_stage_t5_loaded=False,
                negative_prompt=negative_source, runtime_text_override=True)
            write_json(target / 'inputs.json', case_inputs)
            write_json(directory / 'progress.json', dict(status='running', phase='generating', current_case=name, cases=results))
            started = time.monotonic()
            try:
                report, endpoint = engine.run_segment(initial=initial, prompt=embeddings[name],
                    actions=selected_actions.copy(), seed=42, output_path=target / 'raw.mp4', steps=40)
                del endpoint  # No float RGB endpoint artifact is needed for this independent pair.
                annotate_report(report, case, runtime, negative_source, cold)
                require(runtime.forward_calls == 80 and report.get('frames_written') == 97, '6s CFG forward/frame count differs')
                if results:
                    for field in ('noise_sha256_float32', 'first_latent_sha256_float32'):
                        require(report[field] == results[0]['generation'][field], 'paired image/noise changed')
                report.update(t5_loaded_for_prompt_preparation=True, generation_stage_t5_loaded=False,
                              runtime_text_override=True, actual_prompt=prompts[name])
                entry = dict(**case, status='completed', prompt=prompts[name], actual_prompt_condition=condition,
                    seed=42, generation=report, raw_video=str(target / 'raw.mp4'), elapsed_seconds=time.monotonic() - started,
                    elapsed_seconds_scope='generation_case_excludes_shared_encoder_and_video_model_load',
                    quality_evaluation='pending_visual_comparison', quality_improvement_verified=False, realtime=False)
                results.append(entry)
                write_json(target / 'result.json', entry)
                write_json(directory / 'progress.json', dict(status='running', phase='generated', cases=results))
            except Exception as error:
                failed_report = copy.deepcopy(engine.last_run_report or {})
                annotate_report(failed_report, case, runtime, negative_source, cold)
                failed_report.update(t5_loaded_for_prompt_preparation=True, generation_stage_t5_loaded=False,
                                     runtime_text_override=True, actual_prompt=prompts[name])
                write_json(target / 'result.json', dict(**case, status='failed', prompt=prompts[name],
                    error=f'{type(error).__name__}: {error}', generation=failed_report,
                    elapsed_seconds=time.monotonic() - started, runtime_text_override=True, automatic_retry=False))
                raise
    finally:
        if engine is not None:
            engine.close()
        elif runtime is not None:
            runtime.close()
    return dict(status='completed', mode='prompt_clause_probe', cases=results, encoding=encoding,
        load_seconds=cold, elapsed_seconds=time.monotonic() - began, same_noise_and_initial_verified=True,
        actual_model_forward_calls=sum(row['generation']['actual_model_forward_calls'] for row in results),
        t5_loaded_for_prompt_preparation=True, generation_stage_t5_loaded=False, old_cache_modified=False,
        original_demo_changed=False, effective_comparison=inputs['effective_comparison'],
        quality_evaluation='pending_visual_comparison', quality_improvement_verified=False, automatic_retry=False, realtime=False)


def prepare(*, output, source_revision):
    directory = contained(output, DATA_ROOT)
    request = dict(copy.deepcopy(FIXED), job_id=directory.name, source_revision=source_revision, experiment_revision=REVISION)
    validate_request(request)
    require(directory != DATA_ROOT.resolve() and not directory.exists(), 'new output directory required')
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'request.json', request)
    return dict(status='cpu_request_prepared', request=str(directory / 'request.json'), max_seconds=1000,
                gpu_launched=False, encoder_loaded=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), ensure_ascii=False))


if __name__ == '__main__':
    main()
