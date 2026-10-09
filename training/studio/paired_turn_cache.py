"""Exactly two new raw offsets, cached by the existing leased GPU worker only."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.demo.relocation import ArtifactHashes
from training.studio import expand_window97 as native

MODE = 'paired_turn_cache'
REVISION = 'same-episode-fixed-raw-turn-offsets-vae97-static-text-v1'
DATA_ROOT = native.DATA_ROOT
BASE = DATA_ROOT/'creator-runtime/abot-retrain-20261003'
ROOT = BASE/'teacher-paired-turn-v1'
CACHE_ROOT = ROOT/'data/features'
SOURCE_ROOT = DATA_ROOT/'creator-runtime/native97-20260928'
SELECTION_ROOT = BASE/'teacher-raw-turn-pairs-v1'
FIXED_EPISODE = '0328fbd39403bc7955f73eca6a9524f2'
FIXED_STARTS = {'J':0, 'L':210}
JOB_ID = 'cpt-cache2-v1'
VERIFICATION_CACHE = DATA_ROOT/'creator-runtime/verified-artifacts.json'
VAE_SHA256 = '20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36'
OUTPUT_CAP = 100_000_000


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _stat(path):
    value = Path(path).stat()
    return dict(bytes=value.st_size,mtime_ns=value.st_mtime_ns)


def _tensor_sha(value):
    return hashlib.sha256(value.detach().float().contiguous().numpy().tobytes()).hexdigest()


def _validate_windows(candidate, previews):
    """No resampling/reselection beyond the two frozen annotation selections."""
    require(candidate['episode_id'] == FIXED_EPISODE, 'fixed raw-turn episode changed')
    windows = {}
    for target,start in FIXED_STARTS.items():
        row = candidate[target+'_window']
        require(row['episode_id'] == FIXED_EPISODE and row['split'] == 'train'
                and row['source_start'] == start, 'fixed raw-turn offset changed')
        offsets = [start+(2*i*30+16)//32 for i in range(97)]
        require(row['output97_source_indices'] == offsets
                and row['output97_source_frame_ids'] == [v+1 for v in offsets]
                and row['source_frame_start'] == start+1
                and row['source_start_seconds'] == start/30, 'raw source frame/time contract changed')
        actions = row['future_actions_96x8']
        require(len(actions) == 96 and all(len(frame) == 8 and all(v in (0,1) for v in frame)
                for frame in actions), 'expected96 canonical binary eight-key actions')
        for key in ('episode_id','source_start','output97_source_indices','future_actions_96x8'):
            require(previews[target][key] == row[key], 'preview differs from fixed annotation selection')
        windows[target] = copy.deepcopy(row)
    return windows


def _source_binding():
    from training.data.safe_annotations import read_annotation_bundle
    from training.data.action_schema import parse_action_document
    from scripts.cache_abot_window97_features import sampled_actions
    from scripts.cache_scene_static_prompts import static_caption
    selection_path = SELECTION_ROOT/'result.json'
    preview_path = SELECTION_ROOT/'preview/result.json'
    visual_review = SELECTION_ROOT/'visual-review.md'
    source_prepare = SOURCE_ROOT/'data/prepare.json'
    source_manifest = SOURCE_ROOT/'data/manifests/train.jsonl'
    source_cache_request = SOURCE_ROOT/'native97-cache/request.json'
    selection, preview, prepared = read(selection_path), read(preview_path), read(source_prepare)
    require(selection['status'] == preview['status'] == prepared['status'] == 'completed'
            and len(selection['candidates']) == 1, 'completed unique raw-turn selection required')
    previews = {key:read(SELECTION_ROOT/f'preview/{key}/input.json') for key in FIXED_STARTS}
    windows = _validate_windows(selection['candidates'][0],previews)
    require(visual_review.is_file(), 'preserved actual visual review record required')
    require(prepared['kind'] == 'expanded_train_window97_prepare_v1'
            and prepared['manifest'] == str(source_manifest)
            and sha256(source_manifest) == prepared['manifest_sha256'] == selection['manifest']['sha256']
            and selection['manifest']['path'] == str(source_manifest), 'trusted source manifest changed')
    records = [json.loads(line) for line in source_manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    selected = [row for row in records if row['episode_id'] == FIXED_EPISODE]
    require(len(selected) == 1 and selected[0]['split'] == 'train', 'unique original train row required')
    row = selected[0]
    source_text = SOURCE_ROOT/'data/text/train-prompts.pt'
    require(prepared['prompts'] == str(source_text), 'only the existing native97 static text is allowed')
    receipt_path = native.SOURCE_PROMPTS.with_suffix('.pt.receipt.json')
    text_binding = prepared['episodes'][FIXED_EPISODE]
    receipt = read(receipt_path)
    require(text_binding == receipt['episodes'][FIXED_EPISODE]
            and prepared['original_prompt_cache_sha256'] == receipt['cache_sha256']
            and receipt['kind'] == 'scene_static_prompt_cache'
            and receipt['prompt_policy'] == 'scene_static_only_v1', 'static text provenance changed')
    for key in ('video_path','annotations_path'):
        contained(row[key],DATA_ROOT)
        require(all(w[key] == row[key] for w in windows.values()), 'selected raw payload paths differ')
    require(_stat(row['video_path']) == dict(bytes=row['video_probe']['file_bytes'],mtime_ns=row['video_mtime_ns']),
            'previously verified source video stat changed')
    require(sha256(row['annotations_path']) == text_binding['annotations_sha256'], 'annotation identity changed')
    bundle = read_annotation_bundle(row['annotations_path'])
    require(static_caption(bundle.caption) == text_binding['prompt']
            and hashlib.sha256(text_binding['prompt'].encode()).hexdigest() == text_binding['prompt_sha256'],
            'only identical explicit scene_static text is permitted')
    sequence = parse_action_document(bundle.action)
    require(sequence.fps == 30, 'original source FPS changed')
    for target,start in FIXED_STARTS.items():
        require(start in sequence.eligible_resampled_window_starts(97,output_fps=16)
                and sampled_actions(sequence,start).tolist() == windows[target]['future_actions_96x8'],
                'fixed sampled actions no longer match raw annotations')
        case = preview['cases'][target]
        require(case['frames'] == 97 and case['fully_decoded'] is True
                and case['source_start'] == start and case['actual_actions'] == windows[target]['future_actions_96x8'],
                'completed original-video preview contract changed')
    old_cache = read(source_cache_request)
    vae_path = contained(old_cache['vae_path'],DATA_ROOT)
    require(old_cache['mode'] == 'window97_cache' and old_cache['vae_sha256'] == VAE_SHA256
            and vae_path.name == 'Wan2.2_VAE.pth'
            and vae_path.parent.name == Path(prepared['original_encoder']['base_model']).name,
            'trusted original VAE identity changed')
    files = [selection_path,preview_path,visual_review,source_prepare,source_manifest,receipt_path,source_cache_request]
    files.extend(SELECTION_ROOT/f'preview/{key}/input.json' for key in FIXED_STARTS)
    return dict(files={str(path):sha256(path) for path in files},windows=windows,original_row=row,
        source_manifest=str(source_manifest),source_text=str(source_text),source_text_sha256=prepared['prompts_sha256'],
        source_text_stat=_stat(source_text),text_binding=text_binding,
        original_prompt_cache_sha256=prepared['original_prompt_cache_sha256'],
        vae_path=str(vae_path),vae_sha256=VAE_SHA256,vae_stat=_stat(vae_path))


def validate_request(request,directory=None):
    require(request.get('mode') == MODE and request.get('experiment_revision') == REVISION
            and request.get('job_id') == JOB_ID, 'fixed paired-turn cache identity required')
    if directory is not None:
        require(Path(directory).resolve() == (ROOT/JOB_ID).resolve(), 'fixed cache job directory required')
    require(type(request.get('max_seconds')) is int and 60 <= request['max_seconds'] <= 900,
            'cache watchdog must be <=900 seconds')
    require(request.get('prepared_path') == str(ROOT/'data/prepare.json')
            and request.get('cache_root') == str(CACHE_ROOT)
            and request.get('verification_cache') == str(VERIFICATION_CACHE)
            and request.get('reuse_completed') is False, 'cache location/reuse policy changed')
    require(request.get('fixed_starts') == FIXED_STARTS and request.get('episode_id') == FIXED_EPISODE
            and request.get('windows') == 2 and request.get('output_cap_bytes') == OUTPUT_CAP,
            'only two exact new windows are allowed')
    require(isinstance(request.get('source_revision'),str) and bool(request['source_revision']), 'source revision required')


def prepare(*,output=None,source_revision,max_seconds=900):
    """CPU only: bind raw metadata and copy one existing static text embedding."""
    import torch
    began = time.monotonic()
    output = contained(output or ROOT/JOB_ID,ROOT)
    require(output == (ROOT/JOB_ID).resolve() and not output.exists()
            and not (ROOT/'data').exists(), 'new cache identity/data directory required; no overwrite')
    require(type(max_seconds) is int and 60 <= max_seconds <= 900
            and isinstance(source_revision,str) and bool(source_revision), 'bounded CPU-prepared cache request required')
    binding = _source_binding()
    verifier = ArtifactHashes(VERIFICATION_CACHE,protected_paths=[binding['source_text'],binding['vae_path']])
    require(verifier(binding['source_text']) == binding['source_text_sha256'], 'original static text integrity mismatch')
    # Preparation never constructs a VAE, reads VAE weights or opens any latent shard.
    source = torch.load(binding['source_text'],map_location='cpu',weights_only=True,mmap=True)
    require(source['schema_version'] == 1 and source['kind'] == 'scene_static_prompt_cache', 'static text payload differs')
    embeddings = {FIXED_EPISODE:source['prompt_embeds'][FIXED_EPISODE].detach().clone()}
    require(tuple(embeddings[FIXED_EPISODE].shape) == (512,4096)
            and embeddings[FIXED_EPISODE].dtype == torch.bfloat16, 'original static text must remain BF16[512,4096]')
    native.CachedTextVAEEncoder(embeddings,{FIXED_EPISODE:binding['text_binding']},None,{})
    require(time.monotonic()-began < 600, 'CPU preparation600s deadline reached')
    row = copy.deepcopy(binding['original_row'])
    row['selected_source_starts'] = list(FIXED_STARTS.values())
    manifest = ROOT/'data/manifests/train.jsonl'
    text_path = ROOT/'data/text/train-prompts.pt'
    native._jsonl(manifest,[row])
    native._save_tensor(text_path,dict(schema_version=1,kind='scene_static_prompt_cache',prompt_embeds=embeddings),ROOT,OUTPUT_CAP)
    prepared = dict(status='cpu_prepared',kind=REVISION,binding=binding,
        manifest=str(manifest),manifest_sha256=sha256(manifest),prompts=str(text_path),prompts_sha256=sha256(text_path),
        prompt_sha256_float32=_tensor_sha(embeddings[FIXED_EPISODE]),weights_loaded=False,old_latents_read=False,
        text_encoder_loaded=False,new_windows_encoded=0,elapsed_seconds=time.monotonic()-began)
    write_json(ROOT/'data/prepare.json',prepared)
    request = dict(mode=MODE,experiment_revision=REVISION,job_id=JOB_ID,source_revision=source_revision,
        max_seconds=max_seconds,prepared_path=str(ROOT/'data/prepare.json'),prepared_sha256=sha256(ROOT/'data/prepare.json'),
        cache_root=str(CACHE_ROOT),verification_cache=str(VERIFICATION_CACHE),reuse_completed=False,
        fixed_starts=FIXED_STARTS,episode_id=FIXED_EPISODE,windows=2,output_cap_bytes=OUTPUT_CAP)
    validate_request(request,output)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'request.json',request)
    return request


def _validate_prepared(request):
    require(sha256(request['prepared_path']) == request['prepared_sha256'], 'new cache preparation changed')
    prepared = read(request['prepared_path'])
    require(prepared['status'] == 'cpu_prepared' and prepared['kind'] == REVISION
            and prepared['binding'] == _source_binding(), 'fixed source/text/selection metadata changed')
    require(prepared['manifest'] == str(ROOT/'data/manifests/train.jsonl')
            and prepared['prompts'] == str(ROOT/'data/text/train-prompts.pt'), 'prepared output paths differ')
    for key in ('manifest','prompts'):
        require(sha256(prepared[key]) == prepared[key+'_sha256'], 'new '+key+' integrity mismatch')
    return prepared


def run(request,directory,lease):
    import numpy as np
    import torch
    from training.demo.backend import validate_lease,validate_active_authority
    from scripts.cache_abot_window97_features import cache_window97_manifest
    directory = contained(directory,ROOT)
    validate_request(request,directory)
    require(read(directory/'request.json') == request, 'leased request differs')
    validate_lease(lease,max_seconds=request['max_seconds'],job_id=JOB_ID,request_sha256=sha256(directory/'request.json'))
    require(not CACHE_ROOT.exists() and not (directory/'cache-result.json').exists()
            and not (directory/'cache-progress.json').exists(), 'no overwrite or automatic retry')
    began,vae = time.monotonic(),None
    encoded, decoded, initials = 0,[],{}

    def progress(phase):
        validate_active_authority(lease)
        require(time.monotonic()-began < request['max_seconds']-20, 'finite cache deadline reached')
        require(native._stored_bytes(ROOT) < OUTPUT_CAP-20_000_000, 'paired cache100MB output cap reached')
        write_json(directory/'cache-progress.json',dict(status='running',phase=phase,decoded_targets=decoded,
            encoded_windows=encoded,elapsed_seconds=time.monotonic()-began))
        print(f'PAIRED_CACHE {phase} decoded={len(decoded)} encoded={encoded}',flush=True)

    try:
        progress('binding')
        prepared = _validate_prepared(request)
        binding = prepared['binding']
        verifier = ArtifactHashes(VERIFICATION_CACHE,protected_paths=[binding['vae_path'],prepared['prompts']])
        require(verifier(binding['vae_path']) == VAE_SHA256 and _stat(binding['vae_path']) == binding['vae_stat'],
                'trusted VAE integrity/stat mismatch')
        embeddings = torch.load(prepared['prompts'],map_location='cpu',weights_only=True)['prompt_embeds']
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one leased visible device required')
        torch.cuda.reset_peak_memory_stats()
        progress('vae_load_started')
        from utils.wan_wrapper import WanVAEWrapper
        vae = WanVAEWrapper(pretrained_path=binding['vae_path']).requires_grad_(False).eval().to('cuda',torch.bfloat16)
        progress('vae_load_completed')

        def encode(pixels):
            nonlocal encoded
            progress('vae_encode_started')
            require(encoded < 2, 'exactly two VAE windows permitted')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                result = vae.encode_to_latent(pixels.to('cuda',torch.bfloat16)).cpu()
            encoded += 1
            progress('vae_encode_completed')
            return result

        encoder = native.CachedTextVAEEncoder(embeddings,{FIXED_EPISODE:binding['text_binding']},encode,
            dict(kind='wan_vae_with_original_static_text_cache',text_encoder_loaded=False,
                vae_path=binding['vae_path'],vae_sha256=VAE_SHA256,
                original_prompt_cache_sha256=binding['original_prompt_cache_sha256'],
                selection_revision=REVISION))

        def starts(sequence,identity,count,seed):
            require(identity == FIXED_EPISODE and count == 2 and seed == 42, 'cache selection contract differs')
            return list(FIXED_STARTS.values())

        def probe(path):
            progress('raw_payload_check')
            row = binding['original_row']
            require(str(path) == row['video_path']
                    and _stat(path) == dict(bytes=row['video_probe']['file_bytes'],mtime_ns=row['video_mtime_ns']),
                    'verified raw video changed')
            return dict(row['video_probe'],video_sha256=row['video_sha256'])

        def decode(path,**kwargs):
            progress('raw_decode_started')
            target = next((key for key,start in FIXED_STARTS.items() if kwargs['start_seconds'] == start/30),None)
            require(target is not None and target not in decoded and str(path) == binding['original_row']['video_path'],
                    'unexpected or repeated raw source decode')
            rgb,backend = native.decode_window97(path,timeout=min(180,max(1,int(request['max_seconds']-(time.monotonic()-began)-20))),**kwargs)
            initial = ROOT/f'data/initial/{target}.npy'
            initial.parent.mkdir(parents=True,exist_ok=True)
            with initial.open('xb') as stream:
                np.save(stream,rgb[:,0].permute(1,2,0).contiguous().numpy(),allow_pickle=False)
            initials[target] = dict(initial_path=str(initial),initial_sha256=sha256(initial))
            decoded.append(target)
            progress('raw_decode_completed')
            return rgb,backend

        receipt = cache_window97_manifest(prepared['manifest'],CACHE_ROOT,encoder=encoder,
            source_manifest_path=binding['source_manifest'],max_windows_per_episode=2,max_total_windows=2,
            shard_size=2,seed=42,decoder=decode,probe_fn=probe,starts_fn=starts,reuse_completed=False,
            execution_metadata=dict(mode=MODE,job_id=JOB_ID,prepare_path=request['prepared_path'],text_encoder_loaded=False))
        require(receipt['episodes'] == 1 and receipt['windows'] == encoded == 2 and decoded == ['J','L']
                and not receipt['excluded_windows'] and not receipt['reused_episodes'], 'exactly two new windows must complete')
        index = Path(receipt['index'])
        rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
        require(len(rows) == 1 and len(rows[0]['shards']) == 1, 'one episode/two-window shard required')
        shard = CACHE_ROOT/rows[0]['shards'][0]['path']
        payload = torch.load(shard,map_location='cpu',weights_only=True,mmap=True)
        require(payload['source_starts'].tolist() == [0,210], 'new cache source offsets differ')
        windows = {}
        for local_index,target in enumerate(FIXED_STARTS):
            row = copy.deepcopy(binding['windows'][target])
            actions = payload['actions'][local_index].float()
            require(actions.tolist() == row['future_actions_96x8'], 'new cached actions differ from frozen source')
            row.pop('original_video_not_read',None)  # That field belongs to the older label-only scan.
            row.update(window_index=local_index,shard_path=str(shard),shard_stat=_stat(shard),
                annotation_cache_actions_equal=True,original_action_sha256_float32=_tensor_sha(actions),
                first_latent_sha256_float32=_tensor_sha(payload['clean_latents'][local_index,:1].unsqueeze(0)),
                **initials[target])
            windows[target] = row
        require(_tensor_sha(payload['prompt_embeds']) == prepared['prompt_sha256_float32'], 'new cache static text differs')
        result = dict(status='completed',mode=MODE,request=request,cache_index=str(index),cache_index_sha256=sha256(index),
            cache_receipt=str(index)+'.receipt.json',cache_receipt_sha256=sha256(str(index)+'.receipt.json'),
            windows=windows,prompt_text=binding['text_binding']['prompt'],prompt_sha256_float32=prepared['prompt_sha256_float32'],
            elapsed_seconds=time.monotonic()-began,peak_vram_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(),newly_encoded_windows=encoded,
            old_latents_read=False,text_encoder_loaded=False,training_started=False,visual_quality_verified=False,
            output_bytes=native._stored_bytes(ROOT))
        require(result['output_bytes'] < OUTPUT_CAP-100_000, 'cache output cap exceeded')
        write_json(directory/'cache-result.json',result)
        return result
    except BaseException as error:
        write_json(directory/'cache-failed.json',dict(status='failed',error=f'{type(error).__name__}: {error}',
            elapsed_seconds=time.monotonic()-began,encoded_windows=encoded,decoded_targets=decoded))
        raise
    finally:
        del vae
        if torch.cuda.is_initialized(): torch.cuda.empty_cache()


def validate_completed_cache():
    """Read-only binding for A evaluation; never builds or repairs a cache."""
    result_path = ROOT/JOB_ID/'result.json'
    result = read(result_path)
    validate_request(result['request'],ROOT/JOB_ID)
    require(result['status'] == 'completed' and result['mode'] == MODE and result['newly_encoded_windows'] == 2
            and result['old_latents_read'] is False, 'completed new paired-turn cache required')
    require(read(ROOT/JOB_ID/'cache-result.json') == result, 'worker result differs from cache body')
    require(read(ROOT/JOB_ID/'settlement.json')['outcome'] == 'completed', 'cache GPU job must be settled completed')
    _validate_prepared(result['request'])
    index = CACHE_ROOT/'train.features.jsonl'
    require(result['cache_index'] == str(index) and result['cache_index_sha256'] == sha256(index)
            and result['cache_receipt'] == str(index)+'.receipt.json'
            and result['cache_receipt_sha256'] == sha256(str(index)+'.receipt.json'), 'new feature index binding changed')
    from training.data.action_dataset import validate_feature_cache_binding,validate_window97_contract
    receipt = validate_feature_cache_binding(index,ROOT/'data/manifests/train.jsonl')
    validate_window97_contract(receipt)
    require(receipt['episodes'] == 1 and receipt['windows'] == 2
            and receipt['config']['max_total_windows'] == receipt['config']['max_windows_per_episode'] == 2,
            'completed receipt is not the exact two-window cache')
    for target,start in FIXED_STARTS.items():
        row = result['windows'][target]
        require(row['episode_id'] == FIXED_EPISODE and row['source_start'] == start
                and row['window_index'] == list(FIXED_STARTS).index(target)
                and row['shard_stat'] == _stat(contained(row['shard_path'],CACHE_ROOT))
                and row['initial_sha256'] == sha256(contained(row['initial_path'],ROOT/'data/initial')),
                'new window/first-image identity changed')
    return result
