"""Four fixed training-window action qualifications, never a dev benchmark.

Only cached z0, static text and recorded actions are read. Future clean latent
values cannot enter the rollout: the initial-latent engine explicitly bypasses
RGB re-encoding while retaining the existing noise/CFG/Euler/decoding path.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import re
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.demo.latency_engine import ReusableActionEngine
from training.studio import controlled_lora_pair_eval as paired
from training.studio.abot_fresh_eval import PreservingWriter
from training.studio.causal_native_eval import tensor_sha
from training.studio.quality_probe import load_negative

MODE = 'controlled_matched_window_eval'
REVISION = 'controlled-frozen20-train-top1-original-vs-jlswap-z0only-v1'
DATA_ROOT = Path('/DATA_DISK/InteractWorld')
BASE = DATA_ROOT / 'creator-runtime/abot-retrain-20261003'
ROOT = BASE / 'teacher-matched-window-v1'
SELECTION = BASE / 'teacher-supervision-check-v1/selection.json'
SOURCE_EVAL = BASE / 'teacher-controlled-lora-pair-v1/clp-frozen-forward-v1/request.json'
FIXED = {'WJ': ('046156eeb391248540b55d758033e5bb', 0, 64),
         'WL': ('0497066dcee6e8e6a21375c8951dd802', 1, 345)}
CONDITIONS = ('original', 'jl_swap')
KEYS = 'WASDIJKL'
JOB_BYTES_LIMIT = 20_000_000
VIDEO_BYTES_LIMIT = 16_000_000


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _stat(path):
    value = Path(path).stat()
    return dict(bytes=value.st_size, mtime_ns=value.st_mtime_ns)


def directory_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def training_exposure(config, row):
    """Resolve real absolute indices from small metadata; never call __getitem__."""
    from training.data.action_dataset import cache_index_path, _read_jsonl, _seed64
    from training.data.action_resampled import ResampledActionDataset, SAMPLING_NAMESPACE
    index = cache_index_path(config.data.manifest_path)
    receipt = _read(index.with_suffix(index.suffix+'.receipt.json'))
    require(receipt['index_sha256'] == sha256(index)
            and receipt['manifest_sha256'] == config.data.manifest_sha256,
            'sampling feature index differs from training config')
    episodes = sorted([r for r in _read_jsonl(index) if r['split'] == 'train'],key=lambda r:str(r['episode_id']))
    require(len(episodes) == 64 and len({r['episode_id'] for r in episodes}) == 64
            and all(r['num_windows'] == 4 for r in episodes), 'fixed64 episodes/256 windows required')
    # The production resolver needs only these metadata fields. Bypass its
    # tensor-loading constructor, not its actual sampling_spec/window policy.
    resolver = ResampledActionDataset.__new__(ResampledActionDataset)
    resolver.num_samples, resolver.episodes = 320, episodes
    resolver.data_seed = _seed64(int(config.training.seed),SAMPLING_NAMESPACE)
    seen = [i for i in range(320) if (lambda spec:
        (spec['episode_id'],spec['window_index']) == (row['episode_id'],row['window_index']))(resolver.sampling_spec(i))]
    return dict(policy='absolute_index_episode_round_robin_window_permutation_v1',
        seed=int(config.training.seed), feature_index=str(index), feature_index_sha256=receipt['index_sha256'],
        parent_absolute_indices=[i for i in seen if i < 160],
        paired_stage_absolute_indices=[i for i in seen if i >= 160],
        parent_range=[0,159],paired_stage_range=[160,319],
        seen_by_endpoint=bool(seen),future_latents_read=False)


def _selected(target):
    require(target in FIXED, 'fixed WJ/WL target required')
    selection = _read(SELECTION)
    require(selection['status'] == 'fixed_before_any_video_decode', 'original pre-visual selection required')
    row = selection['targets'][target]['windows'][0]
    require((row['episode_id'], row['window_index'], row['source_start']) == FIXED[target]
            and row['annotation_cache_actions_equal'] is True, 'fixed top1 supervision window changed')
    for key in ('shard_path', 'annotations_path', 'video_path'):
        contained(row[key], DATA_ROOT)
    return row


def _source_evaluation():
    source = _read(SOURCE_EVAL)
    paired.validate_request(source)
    require(source['arm'] == 'frozen' and source['checkpoint_step'] == 20
            and source['job_id'] == 'clp-frozen-forward-v1', 'only completed frozen+20 endpoint is permitted')
    return source


def validate_request(request):
    require(request.get('mode') == MODE and request.get('experiment_revision') == REVISION,
            'explicit fixed matched-window route required')
    require(request.get('target') in FIXED and request.get('condition') in CONDITIONS,
            'only fixed top1 original/JL-swap cases permitted')
    require(request.get('selection') == str(SELECTION) and request.get('source_evaluation') == str(SOURCE_EVAL),
            'fixed source records required')
    require(request.get('seed') == 42 and request.get('steps') == 40 and request.get('cfg_scale') == 5
            and request.get('future_frames') == 96 and request.get('solver') == 'flow_euler'
            and request.get('fastpath') is False and request.get('arm') == 'frozen', 'fixed generation contract required')
    require(type(request.get('max_seconds')) is int and 60 <= request['max_seconds'] <= 1200,
            'finite <=1200-second watchdog required')
    require(isinstance(request.get('job_id'), str)
            and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}', request['job_id']), 'safe unique identity required')
    require(isinstance(request.get('source_revision'), str) and bool(request['source_revision']), 'source revision required')
    for name in ('selection_sha256','source_evaluation_sha256'):
        require(isinstance(request.get(name),str) and re.fullmatch(r'[0-9a-f]{64}',request[name]), 'source digest required')
    require(not any(name in request for name in ('checkpoint','action_scale','ratio_cap','gate_scale',
        'lora_strength','initial_latent','future_clean_latents','prompt','actions','window_index','episode_id')),
        'no checkpoint/input/scale override accepted')


def make_request(*, job_id, source_revision, target, condition, max_seconds=1200):
    _selected(target)
    _source_evaluation()
    request = dict(mode=MODE, experiment_revision=REVISION, job_id=job_id, source_revision=source_revision,
        target=target, condition=condition, arm='frozen', selection=str(SELECTION), selection_sha256=sha256(SELECTION),
        source_evaluation=str(SOURCE_EVAL), source_evaluation_sha256=sha256(SOURCE_EVAL),
        seed=42, steps=40, cfg_scale=5, future_frames=96, solver='flow_euler', fastpath=False,
        max_seconds=max_seconds)
    validate_request(request)
    return request


def actions_from_runs(row):
    import torch
    actions = torch.zeros((96,8),dtype=torch.float32)
    cursor = 0
    for run in row['actions']['runs']:
        start, end = run['start_future_index'], run['end_future_index_exclusive']
        require(start == cursor and type(end) is int and start < end <= 96, 'non-contiguous frozen action runs')
        keys = run['keys']
        require(keys == 'NONE' or (len(set(keys)) == len(keys) and all(k in KEYS for k in keys)), 'invalid frozen key run')
        if keys != 'NONE':
            actions[start:end,[KEYS.index(k) for k in keys]] = 1
        cursor = end
    require(cursor == 96, 'frozen action runs must cover96 frames')
    return actions


def _extract_condition(payload, row):
    """Slice z0 before cloning/casting; never materialize future latent values."""
    import torch
    window = row['window_index']
    require(payload['episode_id'] == row['episode_id']
            and payload['kind'] == 'action_teacher_window97_features_v1'
            and payload['prompt_policy'] == 'scene_static_only_v1'
            and int(payload['source_starts'][window]) == row['source_start'], 'cached source window differs')
    first = payload['clean_latents'][window, :1].detach().clone().unsqueeze(0)
    prompt = payload['prompt_embeds'].detach().clone()
    actions = payload['actions'][window].detach().float().clone()
    require(tuple(first.shape) == (1,1,48,30,52) and first.dtype == torch.bfloat16
            and bool(torch.isfinite(first).all()), 'cached z0 must be finite BF16[1,1,48,30,52]')
    require(prompt.ndim == 2 and 1 <= prompt.shape[0] <= 512 and prompt.shape[1] == 4096
            and prompt.dtype == torch.bfloat16 and bool(torch.isfinite(prompt).all()), 'cached static text differs')
    require(torch.equal(actions,actions_from_runs(row)), 'recorded actual96 actions changed')
    return first, prompt, actions


def _original_rgb0(row):
    """Decode exactly one source RGB for display; never condition on future RGB."""
    import torch
    from scripts.cache_abot_features import decode_video_window, _window_start_seconds
    from training.data.safe_annotations import read_annotation_bundle
    from training.data.action_schema import parse_action_document
    sequence = parse_action_document(read_annotation_bundle(row['annotations_path']).action)
    require(sequence.fps == 30 and sequence.frames[row['source_start']].frame_id == row['source_frame_start'],
            'original annotation start differs')
    pixels, backend = decode_video_window(row['video_path'],
        start_seconds=_window_start_seconds(sequence,row['source_start']), num_frames=1,
        target_fps=16,source_fps=30,height=480,width=832,backend='cpu')
    require(backend == 'ffmpeg_cpu' and tuple(pixels.shape) == (3,1,480,832)
            and pixels.dtype == torch.uint8, 'only one original RGB frame may be decoded')
    return pixels[:,0].permute(1,2,0).contiguous().numpy()


def load_inputs(request):
    import torch
    import train_controlled_lora_pair as entry
    validate_request(request)
    require(sha256(SELECTION) == request['selection_sha256']
            and sha256(SOURCE_EVAL) == request['source_evaluation_sha256'], 'fixed source records changed')
    row, source = _selected(request['target']), _source_evaluation()
    require(_stat(row['shard_path']) == row['shard_stat'], 'verified cache stat changed')
    require(sha256(source['training_config']) == source['training_config_sha256'], 'frozen endpoint config changed')
    config = entry.load_config(source['training_config'])
    manifest = [json.loads(line) for line in Path(config.data.manifest_path).read_text(encoding='utf-8').splitlines() if line.strip()]
    matches = [r for r in manifest if r['episode_id'] == row['episode_id'] and r['split'] == 'train']
    require(len(matches) == 1 and row['source_start'] in matches[0]['selected_source_starts'], 'selected training binding differs')
    for key in ('annotations_path','video_path'):
        require(matches[0][key] == row[key], 'source path differs from training manifest')
    payload = torch.load(row['shard_path'], map_location='cpu', weights_only=True, mmap=True)
    first, prompt, original = _extract_condition(payload,row)
    del payload
    actions = original.clone()
    if request['condition'] == 'jl_swap':
        actions[:,5],actions[:,7] = original[:,7],original[:,5]
    initial = _original_rgb0(row)
    require(sha256(source['negative_cache']) == source['negative_cache_sha256']
            and sha256(Path(source['negative_cache']).with_suffix('.json')) == source['negative_receipt_sha256'],
            'fixed negative text source changed')
    negative, negative_source = load_negative(source['negative_cache'], config.model.base_model_path)
    receipt = _read(Path(row['shard_path']).with_name('window97-receipt.json'))
    exposure = training_exposure(config,row)
    lineage = dict(training_window=True, independent_dev_evaluation=False, generalization_verified=False,
        selection=row, source_evaluation=str(SOURCE_EVAL), source_evaluation_sha256=request['source_evaluation_sha256'],
        checkpoint=source['checkpoint'], checkpoint_sha256=source['checkpoint_sha256'], arm='frozen',
        conditioning='cached_training_z0_and_static_text_only', future_clean_latents_read=False,
        future_clean_latents_supplied=False, original_rgb_frames_decoded=1, vae_encoding_used=False,
        first_latent_sha256_float32=tensor_sha(first), prompt_sha256_float32=tensor_sha(prompt),
        prompt_text=receipt['cache_binding']['prompt'], negative_source=negative_source,
        original_action_sha256_float32=tensor_sha(original), actual_action_sha256_float32=tensor_sha(actions),
        actual_action_condition=request['condition'], original_actions=original.tolist(), actual_actions=actions.tolist(),
        training_exposure=exposure,actually_seen_by_endpoint=exposure['seen_by_endpoint'],
        method_scope=('training_seen_window_qualification_not_teacher_readiness_or_generalization'
            if exposure['seen_by_endpoint'] else 'training_manifest_window_not_seen_by_endpoint'))
    return first,prompt,actions.numpy(),initial,negative,config,lineage,source


def prepare(*, output, **options):
    output = contained(output,ROOT)
    require(not output.exists() and output.name == options['job_id'], 'new matching output required')
    request = make_request(**options)
    inputs = load_inputs(request)
    lineage = inputs[-2]
    del inputs
    gc.collect()
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'request.json',request)
    write_json(output/'preparation.json',dict(status='cpu_prepared',request=request,lineage=lineage,
        gpu_launched=False,weights_loaded=False,video_generated=False))
    return request


class CachedInitialEngine(ReusableActionEngine):
    """Explicit opt-in initial condition, no changes to existing RGB engines."""
    def __init__(self, *args, cached_first, **kwargs):
        require(tuple(cached_first.shape) == (1,1,48,30,52), 'only one cached latent condition allowed')
        self._cached_first = cached_first.detach().cpu().clone()
        super().__init__(*args, **kwargs)

    def _encode(self, pixel, index):
        require(index == -1 and tuple(pixel.shape) == (1,3,1,480,832), 'no later frame or latent encoding allowed')
        return self._timed('cached_z0_condition',lambda:self._cached_first.clone())


class BoundedPreservingWriter(PreservingWriter):
    """Keep failed media, bound files; chain also enforces RLIMIT_FSIZE16MB."""
    def _check_size(self):
        for name in ('path','partial'):
            path = getattr(self.writer,name,None)
            if path is not None and Path(path).exists():
                require(Path(path).stat().st_size <= VIDEO_BYTES_LIMIT, 'matched video exceeds16MB limit')

    def write(self,frames):
        self._check_size()
        super().write(frames)
        self._check_size()

    def close(self):
        super().close()
        self._check_size()


def run(request,directory,lease):
    import numpy as np
    import torch
    from PIL import Image
    from training.demo.backend import validate_active_authority
    from training.eval.rollout15s import ffmpeg_writer_factory
    validate_request(request)
    directory = contained(directory,ROOT)
    require(directory.name == request['job_id'],'matched job identity differs')
    target = directory/'matched-evaluation'
    target.mkdir(exist_ok=False)
    began = time.monotonic()
    report = dict(status='running',mode=MODE,revision=REVISION,request=request,
        future_clean_latents_supplied=False,training_window=True,independent_dev_evaluation=False,
        quality_improvement_verified=False,action_correctness_verified=False,realtime=False)
    runtime = engine = None
    def check():
        require(time.monotonic()-began < request['max_seconds']-20,'finite matched evaluation deadline reached')
        require(directory_bytes(directory) < JOB_BYTES_LIMIT-100_000,'matched job reached20MB reserved-output limit')
        validate_active_authority(lease)
    try:
        check()
        require(os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1','offline assets required')
        first,prompt,actions,initial,negative,config,lineage,source = load_inputs(request)
        state,weight_config,checkpoint_lineage = paired._load_weights(source)
        require(weight_config.to_dict() == config.to_dict(),'source training config differs')
        lineage['checkpoint_lineage'] = checkpoint_lineage
        report['lineage'] = lineage
        Image.fromarray(initial).save(target/'initial.png')
        np.save(target/'actions.npy',actions,allow_pickle=False)
        write_json(target/'inputs.json',dict(request=request,lineage=lineage,actual_actions=actions.tolist()))
        check()
        tick = time.monotonic()
        runtime = paired.ControlledPairRuntime(state,config,negative,arm='frozen')
        state.clear()
        gc.collect()
        report['load_seconds'] = time.monotonic()-tick
        engine = CachedInitialEngine({},runtime.config,residency='phase',fastpath=False,_runtime=runtime,
            cached_first=first,_writer_factory=lambda *args:BoundedPreservingWriter(ffmpeg_writer_factory(*args)))
        engine.load_seconds = report['load_seconds']
        generation,_ = engine.run_segment(initial=initial,prompt=prompt,actions=actions,
            seed=42,output_path=target/'raw.mp4',steps=40)
        require(generation['frames_written'] == 97 and runtime.forward_calls == 80
                and generation['first_latent_sha256_float32'] == lineage['first_latent_sha256_float32'],
                'must preserve cached z0 and produce97 frames with40 CFG pairs')
        stages = generation['stage_totals_seconds']
        require('encode_initial' not in stages and 'encode_endpoint' not in stages
                and 'cached_z0_condition' in stages, 'must use cached z0 without any VAE encoding')
        generation.update(initial_condition='cached_training_z0;RGB0_display_only',
            continuation='independent_cached_z0_condition;no_future_ground_truth',vae_encoding_used=False)
        generation['sampler'].update(cfg='text_classifier_free',cfg_scale=5,model_forwards_per_step=2)
        report.update(status='completed',frames_written=97,generation=generation,raw_video=str(target/'raw.mp4'),
            actual_model_forward_calls=runtime.forward_calls,peak_allocated_bytes=generation['peak_vram_bytes'],
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),quality_evaluation='awaiting_visual_review')
        check()
    except BaseException as error:
        report.update(status='failed',failure_type=type(error).__name__,error=str(error))
        raise
    finally:
        try:
            if engine is not None: engine.close()
            elif runtime is not None: runtime.close()
        finally:
            report['elapsed_seconds'] = time.monotonic()-began
            report['output_bytes_before_result'] = directory_bytes(directory)
            report['output_bytes_limit'] = JOB_BYTES_LIMIT
            report['video_bytes_limit'] = VIDEO_BYTES_LIMIT
            write_json(target/'result.json',report)
    return report
