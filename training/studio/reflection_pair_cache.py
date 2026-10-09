"""Two exact RGB reflections; CPU semantic gate precedes leased VAE encoding.

This is an independent augmented-data schema, not a relabeled native cache.
The preview MP4 is display-only: encoding always decodes the original source.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.demo.relocation import ArtifactHashes
from training.studio import paired_turn_cache as parent

ROOT = parent.BASE / 'teacher-reflection-pair-v1'
MODE = 'reflection_pair_cache'
REVISION = 'raw-rgb97-horizontal-reflection-ad-jl-v1'
SOURCE_REVISION = 'creator-reflection-pair-v1'
SCHEMA = 'reflection_four_views_cache_v1'
SHARD_KIND = 'reflection_native97_tensor_shard_v1'
JOB_ID = 'crf-cache2-v1'
VIEW_IDS = ('J_original', 'L_original', 'J_reflect', 'L_reflect')
MIRROR_PERMUTATION = (0, 3, 2, 1, 4, 7, 6, 5)  # A/D columns1/3; J/L columns5/7; time unchanged.
CACHE_ROOT = ROOT / 'data/features'
MANIFEST = ROOT / 'data/manifests/views.jsonl'
INDEX = CACHE_ROOT / 'views.index.jsonl'
OUTPUT_CAP = 100_000_000
read, stat_identity = parent.read, parent._stat


def mirror_actions(actions):
    import torch
    from training.data.action_schema import ACTION_KEYS
    require(tuple(ACTION_KEYS) == tuple('WASDIJKL'), 'canonical key order changed')
    value = torch.as_tensor(actions)
    require(tuple(value.shape) == (96, 8) and bool(((value == 0) | (value == 1)).all()),
            'exact binary96x8 actions required')
    return value[:, MIRROR_PERMUTATION].clone()


def mirror_rgb(rgb):
    import torch
    require(tuple(rgb.shape) == (3, 97, 480, 832) and rgb.dtype == torch.uint8,
            'flip only complete post-resize RGB uint8[3,97,480,832], never latent')
    return rgb.flip(-1).contiguous()


def rgb_sha(rgb):
    return hashlib.sha256(rgb.contiguous().numpy().tobytes()).hexdigest()


def source_binding():
    result = parent.validate_completed_cache()
    require(list(result['windows']) == ['J', 'L'] or set(result['windows']) == {'J', 'L'},
            'exact paired original sources required')
    binding = dict(job=str(parent.ROOT / parent.JOB_ID), **{
        name + '_sha256': sha256(parent.ROOT / parent.JOB_ID / (name + '.json'))
        for name in ('request', 'result', 'settlement')})
    return result, binding


def decode_original(target, timeout=180):
    require(target in parent.FIXED_STARTS, 'fixed J/L source only')
    result, _ = source_binding()
    row = result['windows'][target]
    rgb, backend = parent.native.decode_window97(row['video_path'],
        start_seconds=row['source_start_seconds'], num_frames=97, target_fps=16,
        source_fps=30, height=480, width=832, backend='auto', timeout=timeout)
    return rgb, backend


def prepare_preview(*, output=None, source_revision=SOURCE_REVISION, max_seconds=600):
    """CPU-only fixed RGB previews; does not prepare a GPU request or load weights."""
    import torch
    from PIL import Image, ImageDraw
    from training.eval.rollout15s import ffmpeg_writer_factory
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'preview must explicitly hide CUDA')
    require(source_revision == SOURCE_REVISION and type(max_seconds) is int and 60 <= max_seconds <= 600,
            'fixed bounded CPU preview required')
    output = contained(output or ROOT / 'preview', ROOT)
    require(output == ROOT / 'preview' and not output.exists(), 'new preview only, no overwrite')
    torch.set_num_threads(min(8, torch.get_num_threads()))
    began = time.monotonic()
    original, binding = source_binding()
    output.mkdir(parents=True, exist_ok=False)
    cases = {}
    try:
        for target in ('J', 'L'):
            require(time.monotonic() - began < max_seconds - 30, 'finite preview deadline')
            raw, backend = decode_original(target, min(180, max(1, int(max_seconds - (time.monotonic()-began)-30))))
            rgb = mirror_rgb(raw)
            require(torch.equal(mirror_rgb(rgb), raw), 'RGB reflection involution failed')
            actions = mirror_actions(original['windows'][target]['future_actions_96x8'])
            require(torch.equal(mirror_actions(actions), torch.as_tensor(original['windows'][target]['future_actions_96x8'])),
                    'action reflection involution failed')
            directory = output / target
            directory.mkdir()
            frames = rgb.permute(1, 2, 3, 0).contiguous().numpy()
            path = directory / 'raw.mp4'
            writer = ffmpeg_writer_factory(path, 832, 480, 16)
            writer.write(frames)
            writer.close()
            # Same existing ffmpeg-only complete-decode check as the raw-turn preview.
            # Geometry/rate were explicitly supplied to the RGB writer above.
            # This display MP4 is never loaded by the VAE job.
            probe = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '2', '-i', str(path),
                '-map', '0:v:0', '-progress', 'pipe:1', '-nostats', '-f', 'null', '-'],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120, check=True)
            counts = re.findall(r'^frame=(\d+)$',probe.stdout,flags=re.MULTILINE)
            require(bool(counts) and counts[-1] == '97' and 'progress=end' in probe.stdout,
                    'preview must fully decode97 frames')
            selected = list(range(0, 97, 16)) + ([52, 68, 84] if target == 'L' else [])
            selected.sort()
            sheet = Image.new('RGB', (832, 262 * ((len(selected)+1)//2)), 'white')
            draw = ImageDraw.Draw(sheet)
            (directory / 'frames').mkdir()
            for i, index in enumerate(selected):
                image = Image.fromarray(frames[index])
                image.save(directory / 'frames' / f'f{index:03d}.png')
                x,y=(i%2)*416,(i//2)*262
                sheet.paste(image.resize((416,240)), (x,y+22))
                draw.text((x+5,y+4), f'{target} reflected | {index/16:g}s', fill='black')
            sheet.save(directory / 'contact.jpg', quality=90)
            row = original['windows'][target]
            cases[target] = dict(target=target, parent_source_start=row['source_start'],
                episode_id=parent.FIXED_EPISODE, transform='horizontal_reflection_v1',
                key_permutation=list(MIRROR_PERMUTATION), original_actions=row['future_actions_96x8'],
                actual_actions=actions.tolist(), rgb_sha256_uint8=rgb_sha(rgb), decode_backend=backend,
                sampled_shape=list(rgb.shape), raw_video=str(path), video_stat=stat_identity(path),
                frames=97, fully_decoded=True, selected_frame_indices=selected,
                selected_seconds=[i/16 for i in selected], preview_not_training_input=True)
            write_json(directory / 'input.json', cases[target])
            del raw, rgb, frames
        result = dict(status='completed', schema='reflection_semantic_preview_v1', revision=REVISION,
            source_revision=source_revision, source_binding=binding, cases=cases,
            gpu_used=False, weights_loaded=False, latent_read=False, elapsed_seconds=time.monotonic()-began,
            training_input='future cache will re-decode original RGB, never these preview MP4s')
        require(parent.native._stored_bytes(output) < 40_000_000, 'preview40MB cap')
        write_json(output / 'result.json', result)
        return result
    except BaseException as error:
        write_json(output / 'failure.json', dict(status='failed', error=f'{type(error).__name__}: {error}',
            cases=cases, elapsed_seconds=time.monotonic()-began, automatic_retry=False))
        raise


def visual_gate():
    path = ROOT / 'preview/review-result.json'
    review = read(path)
    preview_path = ROOT / 'preview/result.json'
    preview = read(preview_path)
    _, binding = source_binding()
    require(preview.get('status') == 'completed' and preview.get('schema') == 'reflection_semantic_preview_v1'
            and preview.get('source_binding') == binding, 'fixed actual preview/source required')
    require(review.get('status') == 'actually_reviewed_passed' and review.get('actual_visual_review') is True
            and review.get('preview_result_sha256') == sha256(preview_path)
            and set(review.get('cases', {})) == {'J', 'L'}, 'actual bounded mirror semantic review required')
    for target in ('J', 'L'):
        row, judgement = preview['cases'][target], review['cases'][target]
        require(row['video_stat'] == stat_identity(ROOT / f'preview/{target}/raw.mp4')
                and row['frames'] == 97 and row['fully_decoded'] is True
                and row['parent_source_start'] == parent.FIXED_STARTS[target]
                and judgement.get('direction_compatible') is True and judgement.get('character_identifiable') is True
                and judgement.get('appearance_pass') is True
                and judgement.get('reviewed_seconds') == row['selected_seconds'],
                'both fixed reflected sources must actually pass every selected timepoint')
    return dict(review_path=str(path), review_sha256=sha256(path), preview_result_sha256=sha256(preview_path),
                source_binding=binding, semantic_pass=True)


def validate_request(request, directory=None):
    expected = dict(mode=MODE, experiment_revision=REVISION, source_revision=SOURCE_REVISION, job_id=JOB_ID,
        cleanup_seconds=60, new_windows=2, reuse_original_windows=2, preview_mp4_training_input=False,
        latent_flip=False, output_cap_bytes=OUTPUT_CAP)
    require(all(request.get(k) == v for k,v in expected.items())
            and set(request) == set(expected) | {'max_seconds','visual_gate'}, 'fixed reflection cache request required')
    require(type(request['max_seconds']) is int and 60 <= request['max_seconds'] <= 900, 'cache<=900s required')
    if directory is not None:
        require(Path(directory) == ROOT / JOB_ID, 'fixed reflection cache job directory')


def prepare(*, output=None, source_revision, max_seconds=900):
    directory = contained(output or ROOT / JOB_ID, ROOT)
    request = dict(mode=MODE, experiment_revision=REVISION, source_revision=source_revision, job_id=JOB_ID,
        cleanup_seconds=60, new_windows=2, reuse_original_windows=2, preview_mp4_training_input=False,
        latent_flip=False, output_cap_bytes=OUTPUT_CAP, max_seconds=max_seconds, visual_gate=visual_gate())
    validate_request(request, directory)
    require(not directory.exists() and not (ROOT/'data').exists(), 'fresh reflection cache only')
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory/'request.json', request)
    return request


def _view(original, target, transform):
    row = copy.deepcopy(original['windows'][target])
    row.update(view_id=target+('_original' if transform == 'identity' else '_reflect'), target=target,
        parent_episode_id=row['episode_id'], parent_source_start=row['source_start'], transform=transform,
        prompt_sha256_float32=original['prompt_sha256_float32'])
    return row


def run(request, directory, lease):
    import numpy as np
    import torch
    from training.demo.backend import validate_lease, validate_active_authority
    directory = contained(directory, ROOT)
    validate_request(request, directory)
    require(read(directory/'request.json') == request and lease['gpu_index'] in (1,2), 'original GPU1/2 lease only')
    validate_lease(lease,max_seconds=request['max_seconds'],job_id=JOB_ID,request_sha256=sha256(directory/'request.json'))
    require(not (ROOT/'data').exists() and not (directory/'progress.json').exists(), 'no overwrite or automatic retry')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one leased visible GPU required')
    began, vae, encoded = time.monotonic(), None, 0
    def check(phase):
        validate_active_authority(lease)
        require(time.monotonic()-began < request['max_seconds']-20, 'finite reflection cache deadline')
        require(parent.native._stored_bytes(ROOT/'data') < OUTPUT_CAP if (ROOT/'data').exists() else True, 'cache100MB cap')
        write_json(directory/'progress.json', dict(status='running',phase=phase,encoded_windows=encoded,
            elapsed_seconds=time.monotonic()-began))
    try:
        check('binding')
        require(visual_gate() == request['visual_gate'], 'actual visual gate changed')
        original,_ = source_binding()
        prepared = read(parent.ROOT/'data/prepare.json')
        binding = prepared['binding']
        verifier = ArtifactHashes(parent.VERIFICATION_CACHE,protected_paths=[binding['vae_path'],prepared['prompts']])
        require(verifier(binding['vae_path']) == parent.VAE_SHA256
                and stat_identity(binding['vae_path']) == binding['vae_stat']
                and verifier(prepared['prompts']) == prepared['prompts_sha256'], 'trusted VAE/static text mismatch')
        text = torch.load(prepared['prompts'],map_location='cpu',weights_only=True,mmap=True)['prompt_embeds'][parent.FIXED_EPISODE].clone()
        require(text.dtype == torch.bfloat16 and tuple(text.shape) == (512,4096)
                and parent._tensor_sha(text) == original['prompt_sha256_float32'], 'unchanged original static text required')
        preview = read(ROOT/'preview/result.json')
        from utils.wan_wrapper import WanVAEWrapper
        torch.cuda.reset_peak_memory_stats()
        check('vae_load')
        vae = WanVAEWrapper(pretrained_path=binding['vae_path']).requires_grad_(False).eval().to('cuda',torch.bfloat16)
        latents, actions, mirrors = [], [], {}
        for target in ('J','L'):
            check('decode_original_'+target)
            rgb,backend = decode_original(target,min(180,max(1,int(request['max_seconds']-(time.monotonic()-began)-20))))
            rgb = mirror_rgb(rgb)
            require(rgb_sha(rgb) == preview['cases'][target]['rgb_sha256_uint8'], 'mirror RGB differs from actually reviewed preview')
            action = mirror_actions(original['windows'][target]['future_actions_96x8']).float()
            require(action.tolist() == preview['cases'][target]['actual_actions'], 'mirrored actions differ from preview')
            initial = ROOT/f'data/initial/{target}_reflect.npy'
            initial.parent.mkdir(parents=True,exist_ok=True)
            with initial.open('xb') as stream:
                np.save(stream,rgb[:,0].permute(1,2,0).contiguous().numpy(),allow_pickle=False)
            pixels = rgb.unsqueeze(0).float().div_(127.5).sub_(1.0)
            check('encode_'+target)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                latent = vae.encode_to_latent(pixels.to('cuda',torch.bfloat16)).detach().cpu()
            if latent.ndim == 5 and latent.shape[0] == 1:
                latent = latent[0]
            require(tuple(latent.shape) == (25,48,30,52) and bool(torch.isfinite(latent).all()), 'finite native97 latent required')
            latents.append(latent.to(torch.bfloat16)); actions.append(action)
            row = _view(original,target,'horizontal_reflection_v1')
            row.update(future_actions_96x8=action.tolist(),annotation_cache_actions_equal=False,
                actions_derived_by='AD_and_JL_swap_after_exact_96frame_sampling',
                parent_original_action_sha256_float32=row['original_action_sha256_float32'],
                original_action_sha256_float32=parent._tensor_sha(action),
                first_latent_sha256_float32=parent._tensor_sha(latents[-1][:1].unsqueeze(0)),
                initial_path=str(initial),initial_sha256=sha256(initial),
                rgb_sha256_uint8=preview['cases'][target]['rgb_sha256_uint8'],decode_backend=backend)
            # Original runs are provenance only; never let them describe transformed labels.
            row.pop('actions',None); row.pop('direction_statistics',None)
            mirrors[target] = row
            encoded += 1
            del pixels,rgb,latent
        CACHE_ROOT.mkdir(parents=True,exist_ok=False)
        shard = CACHE_ROOT/'reflected-two-windows.pt'
        with shard.open('xb') as stream:
            torch.save(dict(schema_version=1,kind=SHARD_KIND,episode_id=parent.FIXED_EPISODE,
                clean_latents=torch.stack(latents),actions=torch.stack(actions),prompt_embeds=text,
                source_starts=torch.tensor([0,210]),transform='horizontal_reflection_v1',
                key_permutation=list(MIRROR_PERMUTATION)),stream)
        views = {}
        for target in ('J','L'):
            row = _view(original,target,'identity'); views[row['view_id']] = row
        for i,target in enumerate(('J','L')):
            row = mirrors[target]
            row.update(shard_path=str(shard),shard_stat=stat_identity(shard),window_index=i)
            views[row['view_id']] = row
        parent.native._jsonl(MANIFEST,[dict(schema=SCHEMA,**views[key]) for key in VIEW_IDS])
        parent.native._jsonl(INDEX,[dict(schema=SCHEMA,**views[key]) for key in VIEW_IDS])
        receipt = dict(status='completed',schema=SCHEMA,revision=REVISION,source_binding=request['visual_gate']['source_binding'],
            visual_gate=request['visual_gate'],manifest=str(MANIFEST),manifest_sha256=sha256(MANIFEST),
            cache_index=str(INDEX),cache_index_sha256=sha256(INDEX),views=views,
            newly_encoded_windows=2,reused_original_windows=2,latent_flip=False,preview_mp4_training_input=False,
            text_encoder_loaded=False,prompt_text=original['prompt_text'],prompt_sha256_float32=original['prompt_sha256_float32'])
        receipt_path = Path(str(INDEX)+'.receipt.json')
        write_json(receipt_path,receipt)
        check('complete')
        result = dict(receipt,mode=MODE,request=request,receipt=str(receipt_path),receipt_sha256=sha256(receipt_path),
            elapsed_seconds=time.monotonic()-began,peak_vram_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(),output_bytes=parent.native._stored_bytes(ROOT/'data'),
            training_started=False,automatic_next_stage=False)
        require(encoded == 2 and result['peak_reserved_vram_bytes'] < 23*1024**3, 'exact two finite new VAE windows required')
        write_json(directory/'cache-result.json',result)
        return result
    except BaseException as error:
        write_json(directory/'failure.json',dict(status='failed',error=f'{type(error).__name__}: {error}',
            elapsed_seconds=time.monotonic()-began,encoded_windows=encoded,automatic_retry=False))
        raise
    finally:
        del vae
        if torch.cuda.is_initialized(): torch.cuda.empty_cache()


def validate_completed_cache():
    directory = ROOT/JOB_ID
    request,result,settlement = (read(directory/(name+'.json')) for name in ('request','result','settlement'))
    validate_request(request,directory)
    require(result.get('status') == 'completed' and result.get('mode') == MODE
            and result.get('request') == request and result.get('schema') == SCHEMA
            and result == read(directory/'cache-result.json')
            and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed',
            'actual completed settled reflection cache required')
    require(result['visual_gate'] == request['visual_gate'] == visual_gate(), 'semantic review changed')
    require(result['newly_encoded_windows'] == 2 and result['reused_original_windows'] == 2
            and result['latent_flip'] is False and result['preview_mp4_training_input'] is False,
            'only two genuinely reencoded mirrored windows')
    for key in ('manifest','cache_index','receipt'):
        require(sha256(result[key]) == result[key+'_sha256'], 'new small cache record changed')
    require(set(result['views']) == set(VIEW_IDS), 'exact four views required')
    for name,row in result['views'].items():
        require(row['view_id'] == name and row['shard_stat'] == stat_identity(row['shard_path'])
                and row['initial_sha256'] == sha256(row['initial_path']), 'view source/stat changed')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--source-revision',default=SOURCE_REVISION)
    args = parser.parse_args()
    if args.preview:
        prepare_preview(output=args.output,source_revision=args.source_revision)
    else:
        prepare(output=args.output,source_revision=args.source_revision)
