"""One bounded VAE-only cache of ten reviewed raw windows; no training launch.

Six train windows and four qualification windows have separate manifests/indexes.
All retain their original dataset split=train; purpose is a separate restriction.
Existing static text is copied by CPU mmap, never re-encoded or inferred from video.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.demo.relocation import ArtifactHashes
from training.studio import expand_window97 as native

MODE = 'balanced_turn_cache'
REVISION = SOURCE_REVISION = 'creator-balanced-cache-v1'
SCHEMA = 'balanced_native97_separated_cache_v1'
DATA_ROOT = native.DATA_ROOT
BASE = DATA_ROOT / 'creator-runtime/abot-retrain-20261003'
ROOT = BASE / 'teacher-balanced-cache-v1'
SOURCE = BASE / 'teacher-balanced-turns-v1'
POOL_SOURCE = BASE / 'teacher-multiepisode-turns-v1'
JOB_ID = 'cbtc-cache10-v1'
CACHE_ROOT = ROOT / 'data/features'
SOURCE_MANIFEST = native.SOURCE_MANIFEST
SOURCE_TEXT = native.SOURCE_PROMPTS
TEXT_RECEIPT = SOURCE_TEXT.with_suffix('.pt.receipt.json')
VERIFICATION_CACHE = DATA_ROOT / 'creator-runtime/verified-artifacts.json'
VAE_REQUEST = DATA_ROOT / 'creator-runtime/native97-20260928/native97-cache/request.json'
VAE_SHA256 = '20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36'
SELECTION_SHA = '76632d040369673bef81807a47bb57f5152a5127b3396d28d7e67e20d1b0755d'
IDENTITIES_SHA = 'cf1e14017aac2bf572de69db6af8970e026c284e63fe868c64f9aa7329dc57c3'
WINDOWS_SHA = 'c8bd2652e475792f05563ad4bf18425d65ac076ccae0474221ed5ab5c2988d98'
PAIR = '063168aac541196a82d74a399c633dc6'
OUTPUT_CAP, PROTOCOL_GPU_SECONDS = 1_000_000_000, 1800
PURPOSE_COUNTS = {'train': (6, 6), 'qualification': (3, 4)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stat_identity(path):
    value = Path(path).stat()
    return dict(bytes=value.st_size, mtime_ns=value.st_mtime_ns)


def tensor_sha(value):
    return hashlib.sha256(value.detach().float().contiguous().numpy().tobytes()).hexdigest()


def view_id(row):
    return row['episode_id'] + '-s' + str(row['source_start'])


def partition_windows(identities, selected, a_result, b_result, a_review, b_review):
    """Exact fixed identity/role and actual sampled-review binding, CPU records only."""
    roles, windows = identities['roles'], copy.deepcopy(selected['windows'])
    require(len(roles) == 9 and len({r['episode_id'] for r in roles}) == 9 and len(windows) == 10
            and selected['roles'] == roles and identities['replacements_allowed'] is False,
            'fixed9 original identities/10 windows required')
    require(roles[0] == dict(episode_id=PAIR, target='both', candidate_role='qualification_pair', original_split='train')
            and [r['target'] for r in roles[1:]] == ['J'] * 4 + ['L'] * 4
            and [r['candidate_role'] for r in roles[1:]] == (['train_candidate'] * 3 + ['holdout_candidate']) * 2,
            'frozen three-train/one-holdout per direction and isolated pair required')
    require(a_result.get('status') == b_result.get('status') == 'completed_awaiting_visual_review'
            and a_result.get('downloaded_video_count') == 1 and b_result.get('downloaded_video_count') == 8
            and a_result.get('gpu_seconds') == b_result.get('gpu_seconds') == 0,
            'completed two CPU raw-data stages required')
    require(a_review.get('schema') == 'balanced_turns_stage_a_root_visual_review_v1'
            and a_review.get('source_episode') == PAIR
            and a_review.get('sampled_raw_pair_qualification_pass') is True
            and a_review.get('gate_decision') == 'allow_only_preselected_stage_B_data_preparation',
            'actual Stage A sampled visual qualification required')
    require(b_review.get('schema') == 'balanced_turns_stage_b_root_visual_review_v1'
            and b_review.get('status') == 'reviewed_sampled_raw_dataset_qualified'
            and b_review.get('sampled_raw_dataset_qualification_pass') is True
            and b_review.get('sampled_raw_stage_b_qualification_pass') is True
            and b_review.get('gate_decision') == 'allow_only_separately_fixed_bounded_cache_and_training_protocol'
            and b_review.get('automatic_cache_or_train') is False,
            'actual Stage B sampled visual qualification required; not an automatic-training gate')
    cases = a_result['preview_cases'] + b_result['preview_cases']
    require(len(cases) == 10 and len(b_review.get('windows', [])) == 8, 'ten decoded/raw-reviewed cases required')
    parts = {'train': [], 'qualification': []}
    for index, (row, case) in enumerate(zip(windows, cases)):
        identity, start = row['episode_id'], row['source_start']
        require(row['split'] == 'train' and len(row['future_actions_96x8']) == 96
                and all(len(r) == 8 and all(v in (0, 1) for v in r) for r in row['future_actions_96x8']),
                'native97 requires97 RGB but96 future canonical8-key actions')
        require((case['episode_id'], case['target'], case['source_start'], case['candidate_role'])
                == (identity, row['target'], start, row['candidate_role'])
                and case['frames'] == 97 and case['fully_decoded'] is True
                and case['original_video'] is True and case['model_generated'] is False
                and case['actual_actions'] == row['future_actions_96x8']
                and case['frame_indices'] == row['fixed_preview_frame_indices'], 'actual raw preview differs from fixed source window')
        if index < 2:
            direction = ('J', 'L')[index]
            require(identity == PAIR and start == (450, 1050)[index] and row['target'] == direction
                    and row['candidate_role'] == 'qualification_pair'
                    and a_review[direction]['turn_compatible'] is True
                    and a_review[direction]['source_start'] == start
                    and a_review[direction]['frame_indices'] == case['frame_indices'], 'pair role/review changed')
        else:
            role, review = roles[index - 1], b_review['windows'][index - 2]
            for key in ('episode_id', 'target', 'candidate_role'):
                require(row[key] == role[key] == review[key], 'fixed role/review identity differs')
            require(review['source_start'] == start and all(review.get(k) is True for k in
                ('sampled_raw_window_qualification_pass', 'turn_compatible', 'person_identifiable_at_selected_times')),
                'one raw source window lacks actual positive sampled review')
            require(review.get('sampled_shot_cut_observed') is False
                    and review.get('sampled_severe_structure_failure') is False, 'raw structure gate failed')
        purpose = 'train' if row['candidate_role'] == 'train_candidate' else 'qualification'
        row.update(view_id=view_id(row), purpose=purpose, source_kind='original_dataset_rgb97')
        parts[purpose].append(row)
    for purpose, (episodes, count) in PURPOSE_COUNTS.items():
        require(len(parts[purpose]) == count and len({r['episode_id'] for r in parts[purpose]}) == episodes,
                'purpose episode/window counts changed')
    require({r['episode_id'] for r in parts['train']}.isdisjoint(r['episode_id'] for r in parts['qualification']),
            'qualification episode may never enter the training index')
    return parts


def source_binding():
    from training.data.action_schema import parse_action_document
    from training.data.safe_annotations import read_annotation_bundle
    from scripts.cache_scene_static_prompts import static_caption
    from scripts.cache_abot_features import _window_start_seconds
    from scripts.cache_abot_window97_features import sampled_actions
    paths = dict(identities=SOURCE / 'identity-selection.json', selected=SOURCE / 'window-selection.json',
        a_result=SOURCE / 'result.json', b_result=SOURCE / 'stage-b/result.json',
        a_review=SOURCE / 'review/review-result.json', b_review=SOURCE / 'stage-b/review/review-result.json',
        source_selection=POOL_SOURCE / 'selection.json', annotations=POOL_SOURCE / 'recovery-v2/annotation-downloads.json',
        a_check=SOURCE / 'source-check.json', b_checks=SOURCE / 'stage-b/source-checks.json',
        a_download=SOURCE / 'video-download.json', b_downloads=SOURCE / 'stage-b/video-downloads.json',
        text_receipt=TEXT_RECEIPT, source_manifest=SOURCE_MANIFEST, vae_request=VAE_REQUEST)
    files = {str(p): sha256(p) for p in paths.values()}
    require(files[str(paths['identities'])] == IDENTITIES_SHA and files[str(paths['selected'])] == WINDOWS_SHA
            and files[str(paths['source_selection'])] == SELECTION_SHA, 'frozen identity/window/old256 selection changed')
    docs = {k: read(p) for k, p in paths.items() if k != 'source_manifest'}
    parts = partition_windows(*(docs[k] for k in ('identities', 'selected', 'a_result', 'b_result', 'a_review', 'b_review')))
    selection, receipt = docs['source_selection'], docs['text_receipt']
    require(selection['source_manifest'] == str(SOURCE_MANIFEST)
            and selection['source_manifest_sha256'] == files[str(SOURCE_MANIFEST)] == receipt['manifest_sha256']
            and selection['source_text_receipt'] == str(TEXT_RECEIPT)
            and selection['source_text_receipt_sha256'] == files[str(TEXT_RECEIPT)]
            and receipt['schema_version'] == 1 and receipt['kind'] == 'scene_static_prompt_cache'
            and receipt['prompt_policy'] == 'scene_static_only_v1', 'trusted original static text/manifest receipt changed')
    original = {r['episode_id']: r for r in native._read_rows(SOURCE_MANIFEST)}
    selected_rows = {r['episode_id']: r for r in selection['selected_rows']}
    annotation_receipts = {r['episode_id']: r for r in docs['annotations']}
    probes = {r['episode_id']: r for r in [docs['a_check'], *docs['b_checks']]}
    videos = {r['episode_id']: r for r in [docs['a_download'], *docs['b_downloads']]}
    windows = [r for rows in parts.values() for r in rows]
    ids = sorted({r['episode_id'] for r in windows})
    require(len(ids) == 9 and set(probes) == set(videos) == set(ids), 'exact nine raw asset receipts required')
    annotations = [POOL_SOURCE / 'payload' / identity / 'annotations.tar' for identity in ids]
    verifier = ArtifactHashes(VERIFICATION_CACHE, protected_paths=annotations)
    rows, bindings, stats = {}, {}, {}
    for identity, archive in zip(ids, annotations):
        source = original[identity]
        require(source == selected_rows[identity] and source['split'] == 'train', 'original train row/split changed')
        annotation, text = annotation_receipts[identity], receipt['episodes'][identity]
        require(annotation['local_path'] == str(archive) and annotation['status'] in ('reused', 'downloaded')
                and archive.stat().st_size == annotation['file_bytes']
                and verifier(archive) == annotation['sha256'] == text['annotations_sha256'], 'annotation/text integrity differs')
        bundle = read_annotation_bundle(archive)
        require(text['split'] == 'train' and static_caption(bundle.caption) == text['prompt']
                and hashlib.sha256(text['prompt'].encode()).hexdigest() == text['prompt_sha256'], 'scene_static caption binding differs')
        sequence = parse_action_document(bundle.action)
        require(sequence.fps == 30, 'raw30fps source contract required')
        selected = [w for w in windows if w['episode_id'] == identity]
        for row in selected:
            start = row['source_start']; offsets = sequence.resampled_offsets(97, output_fps=16)
            require(start in sequence.eligible_resampled_window_starts(97, output_fps=16)
                    and row['future_actions_96x8'] == sampled_actions(sequence, start).tolist()
                    and row['output97_source_indices'] == [start + x for x in offsets]
                    and row['output97_source_frame_ids'] == [sequence.frames[start + x].frame_id for x in offsets]
                    and row['source_start_seconds'] == _window_start_seconds(sequence, start), 'exact97 source time/96 action mismatch')
        video = SOURCE / 'payload' / identity / 'video.mp4'
        probe, video_receipt = probes[identity], videos[identity]
        require(video_receipt['local_path'] == str(video) and probe['video_source_time_alignment_verified'] is True
                and stat_identity(video) == dict(bytes=video_receipt['file_bytes'], mtime_ns=probe['video_mtime_ns'])
                and video_receipt['sha256'] == probe['video_sha256'], 'already decoded raw video receipt/stat differs')
        require(all(w['video_path'] == str(video) and w['annotations_path'] == str(archive) for w in selected), 'raw asset path changed')
        rows[identity] = dict(source, video_path=str(video), annotations_path=str(archive),
            video_probe=probe['video_probe'], video_sha256=video_receipt['sha256'], video_mtime_ns=probe['video_mtime_ns'],
            selected_source_starts=sorted(w['source_start'] for w in selected),
            cache_purpose=selected[0]['purpose'], candidate_roles=sorted({w['candidate_role'] for w in selected}))
        bindings[identity] = text
        stats[str(archive)] = stat_identity(archive); stats[str(video)] = stat_identity(video)
    vae = docs['vae_request']; vae_path = contained(vae['vae_path'], DATA_ROOT)
    require(vae.get('mode') == 'window97_cache' and vae.get('vae_sha256') == VAE_SHA256
            and vae_path.name == 'Wan2.2_VAE.pth'
            and vae_path.parent.name == Path(receipt['encoder']['base_model']).name, 'same official VAE/text base required')
    stats[str(SOURCE_TEXT)] = stat_identity(SOURCE_TEXT); stats[str(vae_path)] = stat_identity(vae_path)
    return dict(files=files, source_stats=stats, purpose_windows=parts, source_rows=rows, text_bindings=bindings,
        source_text=str(SOURCE_TEXT), source_text_sha256=receipt['cache_sha256'],
        original_encoder=receipt['encoder'], vae_path=str(vae_path), vae_sha256=VAE_SHA256,
        source_manifest=str(SOURCE_MANIFEST), original_split='train',
        raw_video_hashes_reused=True, preview_mp4_training_input=False, old_latents_read=False)


def validate_request(request, directory=None):
    require(request.get('mode') == MODE and request.get('experiment_revision') == REVISION
            and request.get('source_revision') == SOURCE_REVISION and request.get('job_id') == JOB_ID,
            'fixed balanced VAE-only request required')
    if directory is not None:
        require(Path(directory).resolve() == (ROOT / JOB_ID).resolve(), 'unique new cache job required')
    require(request.get('max_seconds') == 900 and request.get('cleanup_seconds') == 60
            and request.get('protocol_cap_gpu_seconds') == PROTOCOL_GPU_SECONDS
            and request.get('output_cap_bytes') == OUTPUT_CAP and request.get('new_windows') == 10
            and request.get('train_windows') == 6 and request.get('qualification_windows') == 4
            and request.get('prepared_path') == str(ROOT / 'data/prepare.json')
            and request.get('cache_root') == str(CACHE_ROOT), 'fixed900+60 seconds/1GB/6+4 windows required')
    require(request.get('reuse_completed') is False and request.get('automatic_training') is False,
            'no retry/reuse or automatic training')


def prepare(*, output=None, source_revision=SOURCE_REVISION, max_seconds=900):
    """CPU: validate9 captions/10 actions, mmap-clone9 text tensors, no model load."""
    import torch
    began = time.monotonic()
    output = contained(output or ROOT / JOB_ID, ROOT)
    require(output == (ROOT / JOB_ID).resolve() and not output.exists() and not (ROOT / 'data').exists(),
            'fresh independent cache/data directory required; no overwrite')
    require(source_revision == SOURCE_REVISION and max_seconds == 900, 'fixed cache revision/deadline required')
    binding = source_binding()
    verifier = ArtifactHashes(VERIFICATION_CACHE, protected_paths=[SOURCE_TEXT, binding['vae_path']])
    require(verifier(SOURCE_TEXT) == binding['source_text_sha256'], 'trusted complete r004 text integrity mismatch')
    payload = torch.load(SOURCE_TEXT, map_location='cpu', weights_only=True, mmap=True)
    require(payload.get('schema_version') == 1 and payload.get('kind') == 'scene_static_prompt_cache', 'r004 text schema mismatch')
    embeddings = {identity: payload['prompt_embeds'][identity].detach().clone() for identity in binding['text_bindings']}
    require(len(embeddings) == 9 and all(tuple(t.shape) == (512, 4096) and t.dtype == torch.bfloat16
            and t.device.type == 'cpu' and bool(torch.isfinite(t).all()) for t in embeddings.values()), 'nine finite BF16[512,4096] texts required')
    del payload
    native.CachedTextVAEEncoder(embeddings, binding['text_bindings'], None, {})
    manifests = {}
    for purpose, windows in binding['purpose_windows'].items():
        identities = sorted({r['episode_id'] for r in windows})
        path = ROOT / f'data/manifests/{purpose}.jsonl'
        native._jsonl(path, [binding['source_rows'][identity] for identity in identities])
        manifests[purpose] = dict(path=str(path), sha256=sha256(path), episodes=len(identities), windows=len(windows))
    relocated = ROOT / 'data/manifests/source-relocated.jsonl'
    native._jsonl(relocated, [binding['source_rows'][identity] for identity in sorted(binding['source_rows'])])
    text_path = ROOT / 'data/text/selected-prompts.pt'
    native._save_tensor(text_path, dict(schema_version=1, kind='scene_static_prompt_cache', prompt_embeds=embeddings), ROOT, OUTPUT_CAP)
    prepared = dict(status='cpu_prepared', schema=SCHEMA, revision=REVISION, binding=binding, manifests=manifests,
        source_relocated_manifest=str(relocated), source_relocated_manifest_sha256=sha256(relocated),
        prompts=str(text_path), prompts_sha256=sha256(text_path),
        prompt_sha256_float32={identity: tensor_sha(t) for identity, t in embeddings.items()},
        model_weights_loaded=False, text_encoder_loaded=False, old_latents_read=False, new_windows_encoded=0,
        elapsed_seconds=time.monotonic() - began)
    require(time.monotonic() - began < 600, 'CPU prepare600-second bound')
    write_json(ROOT / 'data/prepare.json', prepared)
    request = dict(mode=MODE, experiment_revision=REVISION, source_revision=SOURCE_REVISION, job_id=JOB_ID,
        max_seconds=900, cleanup_seconds=60, protocol_cap_gpu_seconds=PROTOCOL_GPU_SECONDS, output_cap_bytes=OUTPUT_CAP,
        new_windows=10, train_windows=6, qualification_windows=4, cache_root=str(CACHE_ROOT),
        prepared_path=str(ROOT / 'data/prepare.json'), prepared_sha256=sha256(ROOT / 'data/prepare.json'),
        reuse_completed=False, automatic_training=False)
    validate_request(request, output)
    output.mkdir(parents=True, exist_ok=False); write_json(output / 'request.json', request)
    return request


def validate_prepared(request):
    require(sha256(request['prepared_path']) == request['prepared_sha256'], 'prepared cache input changed')
    prepared = read(request['prepared_path'])
    require(prepared['schema'] == SCHEMA and prepared['status'] == 'cpu_prepared', 'CPU preparation missing')
    for path, digest in prepared['binding']['files'].items():
        require(sha256(path) == digest, 'source/review small record changed: ' + path)
    for path, expected in prepared['binding']['source_stats'].items():
        require(stat_identity(contained(path, DATA_ROOT)) == expected, 'source asset stat changed: ' + path)
    require(prepared['prompts'] == str(ROOT / 'data/text/selected-prompts.pt')
            and sha256(prepared['prompts']) == prepared['prompts_sha256'], 'new selected text cache changed')
    for purpose in PURPOSE_COUNTS:
        info = prepared['manifests'][purpose]
        require(info['path'] == str(ROOT / f'data/manifests/{purpose}.jsonl') and sha256(info['path']) == info['sha256'],
                'purpose manifest differs')
    require(prepared['source_relocated_manifest'] == str(ROOT / 'data/manifests/source-relocated.jsonl')
            and sha256(prepared['source_relocated_manifest']) == prepared['source_relocated_manifest_sha256'],
            'source relocation manifest changed')
    return prepared


def run(request, directory, lease):
    import numpy as np
    import torch
    from training.demo.backend import validate_active_authority, validate_lease
    from scripts.cache_abot_window97_features import cache_window97_manifest
    directory = contained(directory, ROOT); validate_request(request, directory)
    require(read(directory / 'request.json') == request and lease['gpu_index'] in (1, 2), 'unchanged request/original GPU1/2 lease required')
    validate_lease(lease, max_seconds=900, job_id=JOB_ID, request_sha256=sha256(directory / 'request.json'))
    require(not CACHE_ROOT.exists() and not (directory / 'progress.json').exists(), 'no cache overwrite or automatic retry')
    began, vae = time.monotonic(), None
    decoded, encoded, initials, current = [], [], {}, None

    def check(phase):
        validate_active_authority(lease)
        require(time.monotonic() - began < 880, '900-second finite cache completion margin reached')
        require(native._stored_bytes(ROOT) < OUTPUT_CAP - 25_000_000, '1GB stage cap')
        write_json(directory / 'progress.json', dict(phase=phase, decoded_windows=len(decoded), encoded_windows=len(encoded),
            elapsed_seconds=time.monotonic() - began))
        print('BALANCED_CACHE ' + phase + f' decoded={len(decoded)} encoded={len(encoded)}', flush=True)

    try:
        check('binding')
        prepared = validate_prepared(request); binding = prepared['binding']
        verifier = ArtifactHashes(VERIFICATION_CACHE, protected_paths=[binding['vae_path']])
        require(verifier(binding['vae_path']) == VAE_SHA256, 'trusted official VAE mismatch')
        text = torch.load(prepared['prompts'], map_location='cpu', weights_only=True, mmap=True)['prompt_embeds']
        require(set(text) == set(binding['text_bindings']) and all(tensor_sha(v) == prepared['prompt_sha256_float32'][k]
                for k, v in text.items()), 'selected static text differs')
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one leased visible GPU required')
        torch.cuda.reset_peak_memory_stats()
        from utils.wan_wrapper import WanVAEWrapper
        check('vae_load_started')
        vae = WanVAEWrapper(pretrained_path=binding['vae_path']).requires_grad_(False).eval().to('cuda', torch.bfloat16)
        check('vae_load_completed')
        windows = [r for group in binding['purpose_windows'].values() for r in group]

        def encode(pixels):
            require(current is not None and current not in encoded and len(encoded) < 10, 'only ten independent raw VAE encodes')
            check('encode_' + current)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                latent = vae.encode_to_latent(pixels.to('cuda', torch.bfloat16)).detach().cpu()
            encoded.append(current); check('encoded_' + current)
            return latent

        encoder = native.CachedTextVAEEncoder(text, binding['text_bindings'], encode,
            dict(kind='wan_vae_with_original_static_text_cache', text_encoder_loaded=False,
                vae_path=binding['vae_path'], vae_sha256=VAE_SHA256,
                original_prompt_cache_sha256=binding['source_text_sha256'], selection_revision=REVISION))

        def probe(path):
            row = next(r for r in binding['source_rows'].values() if r['video_path'] == str(path))
            require(stat_identity(path) == dict(bytes=row['video_probe']['file_bytes'], mtime_ns=row['video_mtime_ns']),
                    'verified raw video changed')
            return dict(row['video_probe'], video_sha256=row['video_sha256'])

        def decode(path, **kwargs):
            nonlocal current
            matches = [r for r in windows if r['video_path'] == str(path) and r['source_start_seconds'] == kwargs['start_seconds']]
            require(len(matches) == 1, 'decode must match exactly one frozen raw offset')
            row = matches[0]; current = row['view_id']
            require(current not in decoded, 'duplicate source decode')
            check('decode_' + current)
            rgb, backend = native.decode_window97(path, timeout=min(180, max(1, int(880 - (time.monotonic() - began)))), **kwargs)
            initial = ROOT / 'data/initial' / (current + '.npy')
            initial.parent.mkdir(parents=True, exist_ok=True)
            with initial.open('xb') as stream:
                np.save(stream, rgb[:, 0].permute(1, 2, 0).contiguous().numpy(), allow_pickle=False)
            initials[current] = dict(initial_path=str(initial), initial_sha256=sha256(initial))
            decoded.append(current)
            return rgb, backend

        indexes, cached = {}, []
        for purpose, (episode_count, window_count) in PURPOSE_COUNTS.items():
            wanted = binding['purpose_windows'][purpose]
            mapping = {identity: sorted(r['source_start'] for r in wanted if r['episode_id'] == identity)
                       for identity in {r['episode_id'] for r in wanted}}

            def starts(sequence, identity, count, seed):
                require(seed == 42 and identity in mapping and len(mapping[identity]) <= count, 'purpose/source selection changed')
                return mapping[identity]

            receipt = cache_window97_manifest(prepared['manifests'][purpose]['path'], CACHE_ROOT, encoder=encoder,
                source_manifest_path=prepared['source_relocated_manifest'], max_windows_per_episode=2,
                max_total_windows=window_count, shard_size=2, seed=42, decoder=decode, probe_fn=probe,
                starts_fn=starts, reuse_completed=False, execution_metadata=dict(mode=MODE, job_id=JOB_ID,
                    purpose=purpose, original_split='train', training_pool=(purpose == 'train'), text_encoder_loaded=False))
            require(receipt['episodes'] == episode_count and receipt['windows'] == window_count
                    and not receipt['excluded_windows'] and not receipt['reused_episodes'], 'all fixed purpose windows must complete')
            index = Path(receipt['index']); entries = native._read_rows(index)
            indexes[purpose] = dict(path=str(index), sha256=sha256(index), receipt=str(index) + '.receipt.json',
                receipt_sha256=sha256(str(index) + '.receipt.json'), manifest=prepared['manifests'][purpose]['path'],
                manifest_sha256=prepared['manifests'][purpose]['sha256'], episodes=episode_count, windows=window_count)
            for entry in entries:
                identity = entry['episode_id']; require(len(entry['shards']) == 1, 'one small shard per selected episode')
                shard_row = entry['shards'][0]; path = CACHE_ROOT / shard_row['path']
                payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
                require(payload['source_starts'].tolist() == mapping[identity], 'actual cache starts differ')
                for local, start in enumerate(mapping[identity]):
                    row = next(copy.deepcopy(r) for r in wanted if r['episode_id'] == identity and r['source_start'] == start)
                    actions = payload['actions'][local].float()
                    require(actions.tolist() == row['future_actions_96x8'] and tuple(payload['clean_latents'][local].shape) == (25, 48, 30, 52)
                            and tensor_sha(payload['prompt_embeds']) == prepared['prompt_sha256_float32'][identity], 'actual cached labels/shape/text differ')
                    row.update(window_index=local, shard_path=str(path), shard_stat=stat_identity(path),
                        recorded_shard_sha256=shard_row['sha256'], annotation_cache_actions_equal=True,
                        original_action_sha256_float32=tensor_sha(actions),
                        first_latent_sha256_float32=tensor_sha(payload['clean_latents'][local, :1].unsqueeze(0)),
                        prompt_sha256_float32=prepared['prompt_sha256_float32'][identity],
                        prompt_text=binding['text_bindings'][identity]['prompt'],
                        prompt_text_sha256=binding['text_bindings'][identity]['prompt_sha256'], **initials[row['view_id']])
                    row.pop('original_video_not_read', None); cached.append(row)
                del payload
        require(len(encoded) == len(decoded) == len(set(encoded)) == len(cached) == 10 and set(encoded) == set(decoded), 'exactly ten real VAE windows required')
        require(torch.cuda.max_memory_reserved() < 23 * 1024 ** 3, 'cache peak reserve exceeded23GiB')
        check('complete')
        result = dict(status='completed', mode=MODE, schema=SCHEMA, revision=REVISION, request=request,
            indexes=indexes, windows=cached, manifests=prepared['manifests'], original_split='train',
            newly_encoded_windows=10, train_windows=6, qualification_windows=4, distinct_episodes=9,
            vae_model_load_count=1, text_encoder_loaded=False, old_latents_read=False, preview_mp4_training_input=False,
            latent_shape=[25, 48, 30, 52], action_shape=[96, 8], conditioning_z0_shape=[1, 1, 48, 30, 52],
            elapsed_seconds=time.monotonic() - began, output_bytes=native._stored_bytes(ROOT),
            peak_vram_bytes=torch.cuda.max_memory_allocated(), peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(),
            training_started=False, automatic_next_stage=False, model_quality_improved=False)
        write_json(directory / 'cache-result.json', result)
        return result
    except BaseException as error:
        write_json(directory / 'failure.json', dict(status='failed', error=f'{type(error).__name__}: {error}',
            elapsed_seconds=time.monotonic() - began, decoded_windows=decoded, encoded_windows=encoded,
            partial_outputs_preserved=True, automatic_retry=False, automatic_training=False))
        raise
    finally:
        del vae
        if torch.cuda.is_initialized(): torch.cuda.empty_cache()


def validate_completed_cache(purpose=None):
    require(purpose is None or purpose in PURPOSE_COUNTS, 'explicit train or qualification purpose only')
    directory = ROOT / JOB_ID
    request, result, settlement = (read(directory / (name + '.json')) for name in ('request', 'result', 'settlement'))
    validate_request(request, directory); validate_prepared(request)
    require(result.get('status') == 'completed' and result.get('mode') == MODE and result.get('schema') == SCHEMA
            and result.get('request') == request and result == read(directory / 'cache-result.json')
            and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed'
            and result.get('newly_encoded_windows') == 10 and result.get('vae_model_load_count') == 1,
            'real completed and settled ten-window cache required')
    from training.data.action_dataset import validate_feature_cache_binding, validate_window97_contract
    for name, (episodes, count) in PURPOSE_COUNTS.items():
        info = result['indexes'][name]
        require(info['path'] == str(CACHE_ROOT / f'{name}.features.jsonl') and sha256(info['path']) == info['sha256']
                and sha256(info['receipt']) == info['receipt_sha256'], 'separate purpose index/receipt differs')
        receipt = validate_feature_cache_binding(info['path'], info['manifest']); validate_window97_contract(receipt)
        rows = [r for r in result['windows'] if r['purpose'] == name]
        require(len(rows) == count and len({r['episode_id'] for r in rows}) == episodes
                and receipt['windows'] == count and receipt['episodes'] == episodes, 'purpose count mismatch')
        for row in rows:
            require(stat_identity(contained(row['shard_path'], CACHE_ROOT)) == row['shard_stat']
                    and sha256(contained(row['initial_path'], ROOT / 'data/initial')) == row['initial_sha256'], 'cache/initial image changed')
    if purpose is None:
        return result
    return dict(schema=SCHEMA, purpose=purpose, binding=result['indexes'][purpose],
        windows=[r for r in result['windows'] if r['purpose'] == purpose],
        cache_job=str(directory), cache_result_sha256=sha256(directory / 'result.json'),
        original_split='train', qualification_never_training=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / JOB_ID)
    parser.add_argument('--source-revision', default=SOURCE_REVISION)
    args = parser.parse_args()
    print(json.dumps(prepare(output=args.output, source_revision=args.source_revision), indent=2))
