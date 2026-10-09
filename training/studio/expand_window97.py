"""Bounded train-only RGB97 preparation and VAE-only cache body.

CPU prepare is explicit; importing this module downloads nothing. GPU cache()
must be called by the existing leased worker/watchdog, never directly launched
by this module. Existing payloads, model assets and all prior results stay intact.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from training.demo.contracts import require, write_json
from training.demo.relocation import ArtifactHashes
from training.data.action_dataset import SCENE_STATIC_PROMPT_CACHE_KIND, SCENE_STATIC_PROMPT_POLICY, sha256_file
from training.data.action_schema import ACTION_KEYS, parse_action_document
from training.data.safe_annotations import read_annotation_bundle
from training.data.abot_remote import (
    OFFICIAL_REPO_ID, OFFICIAL_REVISION, HuggingFaceClient, download_remote_file,
    _path_from_hf_uri, _path_from_annotation_uri,
)
from scripts.cache_scene_static_prompts import static_caption
from scripts.cache_abot_features import _ffmpeg_command, validate_action_video_alignment
from training.studio.rebuild_subset import (
    DATA_ROOT, SOURCE_MANIFEST, SOURCE_PROMPTS, DEV_TEXT_IDS,
    _json, _jsonl, _data_path, _save_tensor, _stored_bytes, _BoundedClient,
    load_selected_prompts, ffmpeg_probe,
)

REUSE_ROOT = DATA_ROOT / 'creator-runtime/subset-20260928/data'
DOWNLOAD_CAP = 8_000_000_000
FEATURE_CAP = 2_000_000_000
SELECTION_POLICY = 'train64_seed42_control_coverage_nonoverlap_rgb97_v1'


def select_rows(rows, receipt, retained_ids, count=64):
    require(type(count) is int and 8 <= count <= 64, 'select 8–64 original train episodes')
    require(len(retained_ids) == 8 and len(set(retained_ids)) == 8, 'retain exactly the eight existing train episodes')
    require(len({r['episode_id'] for r in rows}) == len(rows), 'duplicate source episode')
    eligible = {}
    for row in rows:
        identity = row['episode_id']
        binding = receipt['episodes'].get(identity, {})
        if row.get('split') != 'train' or binding.get('split') != 'train':
            continue
        require(re.fullmatch(r'[0-9a-f]{32}', identity) is not None, 'unsafe original episode ID')
        require(binding.get('prompt') and hashlib.sha256(binding['prompt'].encode()).hexdigest() == binding['prompt_sha256'],
                'original static prompt binding changed')
        if row.get('source_fps') == 30 and row.get('output_fps') == 16 and row.get('total_action_frames', 0) >= 181:
            eligible[identity] = row
    require(set(retained_ids).issubset(eligible), 'retained data are not original text-covered training episodes')
    remaining = sorted((r for k, r in eligible.items() if k not in retained_ids),
                       key=lambda r: hashlib.sha256(f"42\0{r['episode_id']}".encode()).hexdigest())
    selected = [eligible[k] for k in retained_ids] + remaining[:count - 8]
    require(len(selected) == count, 'insufficient original training episodes')
    return [dict(r) for r in sorted(selected, key=lambda r: r['episode_id'])]


def action_counts(actions):
    import numpy as np
    values = np.asarray(actions, dtype=bool)
    require(values.ndim == 2 and values.shape[1] == 8 and len(values), 'expected nonempty canonical action rows')
    return dict(frames=len(values), keys=dict(zip(ACTION_KEYS, values.sum(0).astype(int).tolist())),
                zero=int((~values.any(1)).sum()),
                combinations={k: int(values[:, [ACTION_KEYS.index(c) for c in k]].all(1).sum())
                              for k in ('WJ', 'WL', 'WI', 'WK')})


def select_starts(sequence, identity, count=4):
    """Prefer observed combinations, then pitch/movement; never invent labels.

    At most four non-overlapping six-second source spans. Rare WI/WK are
    preferred only when actually present for >=16 sampled action frames.
    Per-episode quotas do not imply globally uniform action coverage.
    """
    import numpy as np
    require(type(count) is int and 1 <= count <= 4, 'at most four RGB97 windows')
    starts = np.asarray(sequence.eligible_resampled_window_starts(97, output_fps=16), dtype=np.int64)
    require(len(starts) > 0, 'episode has no complete continuous RGB97 window')
    raw = np.asarray([frame.keys for frame in sequence.frames], dtype=bool)
    offsets = np.asarray(sequence.resampled_offsets(97, output_fps=16)[1:])
    clips = raw[starts[:, None] + offsets]
    active = clips.any(2).sum(1)
    tie = [hashlib.sha256(f'42\0{identity}\0{int(s)}'.encode()).digest() for s in starts]
    available = np.ones(len(starts), dtype=bool)
    picked = []
    for target in ('WI', 'WK', 'WJ', 'WL', 'I', 'K', 'W', 'S', 'A', 'D', 'J', 'L', None):
        if len(picked) == count or not available.any():
            break
        hits = (clips[:, :, [ACTION_KEYS.index(k) for k in target]].all(2).sum(1)
                if target else active)
        candidates = np.flatnonzero(available & (hits >= (16 if target else 0)))
        while len(candidates) and len(picked) < count:
            winner = min(candidates.tolist(), key=lambda i: (-int(hits[i]), -int(active[i]), tie[i]))
            picked.append(int(starts[winner]))
            available &= abs(starts - starts[winner]) >= 181
            if target:
                break
            candidates = np.flatnonzero(available)
    require(picked, 'no valid selected window')
    return sorted(picked)


def _read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def _retained_rows(source_sha, prompt_sha):
    prepared = _json(REUSE_ROOT / 'prepare.json')
    manifest = REUSE_ROOT / 'manifests/train.jsonl'
    require(prepared.get('status') == 'completed' and sha256_file(manifest) == prepared['manifest_sha256']
            and prepared['source_manifest_sha256'] == source_sha
            and prepared['original_prompt_cache_sha256'] == prompt_sha, 'retained subset source binding changed')
    rows = {r['episode_id']: r for r in _read_rows(manifest) if r['split'] == 'train'}
    require(len(rows) == 8, 'expected the existing eight verified train payloads')
    return rows


def _validate_payload(row, root, bindings):
    identity = row['episode_id']
    require(row.get('payload_origin') in ('retained_subset', 'expanded_download'), 'unknown payload origin')
    owner = REUSE_ROOT if row.get('payload_origin') == 'retained_subset' else root
    for key, filename in (('video_path', 'video.mp4'), ('annotations_path', 'annotations.tar')):
        require(Path(row[key]).resolve() == owner.resolve() / 'payload' / identity / filename,
                'payload is outside its bound episode directory')
    stat = Path(row['video_path']).stat()
    require(stat.st_size == row['video_probe']['file_bytes'] and stat.st_mtime_ns == row['video_mtime_ns'],
            'verified video identity changed')
    require(sha256_file(row['annotations_path']) == bindings[identity]['annotations_sha256'],
            'annotation differs from the original static text binding')


def write_eval_sidecar(root, prompts, receipt, source_sha, train_manifest):
    """Dev text only, explicitly not a training-cache feature index."""
    output = Path(root) / 'eval/prompts.pt'
    values = {identity: prompts[identity] for identity in DEV_TEXT_IDS}
    require(all(receipt['episodes'][key]['split'] == 'dev' for key in values), 'evaluation episode split changed')
    _save_tensor(output, dict(schema_version=1, kind=SCENE_STATIC_PROMPT_CACHE_KIND, prompt_embeds=values),
                 Path(root) / 'eval', FEATURE_CAP)
    result = dict(schema_version=1, kind='window97_eval_text_reference_v1',
        prompt_policy=SCENE_STATIC_PROMPT_POLICY, scope='dev_text_only_no_video_features',
        cache_path=str(output), cache_sha256=sha256_file(output),
        source_manifest=str(SOURCE_MANIFEST), source_manifest_sha256=source_sha,
        source_prompt_cache=str(SOURCE_PROMPTS), source_prompt_cache_sha256=receipt['cache_sha256'],
        source_prompt_receipt=str(SOURCE_PROMPTS.with_suffix('.pt.receipt.json')),
        source_prompt_receipt_sha256=sha256_file(SOURCE_PROMPTS.with_suffix('.pt.receipt.json')),
        source_feature_index_sha256=receipt['feature_index_sha256'],
        source_feature_receipt_sha256=receipt['feature_receipt_sha256'], encoder=receipt['encoder'],
        episodes={key: receipt['episodes'][key] for key in values},
        selected_embedding_sha256_float32={key: hashlib.sha256(value.detach().float().contiguous().numpy().tobytes()).hexdigest()
                                          for key, value in values.items()},
        training_manifest=str(train_manifest), training_manifest_sha256=sha256_file(train_manifest))
    write_json(output.with_suffix('.pt.receipt.json'), result)
    return str(output)


def prepare(output, *, episodes=64, max_seconds=1800, verification_cache=None):
    """Explicit CPU download phase; annotations for every selection precede videos."""
    began = time.monotonic()
    require(type(max_seconds) is int and 60 <= max_seconds <= 1800, 'CPU prepare requires a <=1800s deadline')
    root = _data_path(output)
    require(not root.exists(), 'use a new output directory; previous data are preserved')
    receipt = _json(SOURCE_PROMPTS.with_suffix('.pt.receipt.json'))
    source_sha = sha256_file(SOURCE_MANIFEST)
    require(source_sha == receipt['manifest_sha256'], 'trusted original manifest differs from text cache')
    originals = _read_rows(SOURCE_MANIFEST)
    retained = _retained_rows(source_sha, receipt['cache_sha256'])
    selected = select_rows(originals, receipt, set(retained), episodes)
    by_id = {r['episode_id']: r for r in originals}
    require(all(by_id[k]['split'] == 'dev' for k in DEV_TEXT_IDS), 'dev text references changed split')
    digest = ArtifactHashes(_data_path(verification_cache) if verification_cache else None,
                            protected_paths=[SOURCE_MANIFEST, SOURCE_PROMPTS])
    prompts = load_selected_prompts(SOURCE_PROMPTS, receipt, selected + [by_id[k] for k in DEV_TEXT_IDS], verifier=digest)
    client = HuggingFaceClient(repo_id=OFFICIAL_REPO_ID, revision=OFFICIAL_REVISION, timeout_seconds=30.)
    paths = []
    for row in selected:
        row['remote_video'] = _path_from_hf_uri(row['video_uri'], repo_id=OFFICIAL_REPO_ID, revision=OFFICIAL_REVISION)
        row['remote_annotations'] = _path_from_annotation_uri(row['annotations_uri'], repo_id=OFFICIAL_REPO_ID, revision=OFFICIAL_REVISION)
        if row['episode_id'] not in retained:
            paths.extend((row['remote_annotations'], row['remote_video']))
    catalog = {item.path: item for item in client.get_paths_info(paths)} if paths else {}
    require(set(catalog) == set(paths), 'pinned remote metadata are incomplete')
    for row in selected:
        if row['episode_id'] in retained:
            continue
        catalog[row['remote_video']].validate(require_sha256=True)
        item = catalog[row['remote_annotations']]
        expected = receipt['episodes'][row['episode_id']]['annotations_sha256']
        require(item.sha256 in (None, expected), 'remote annotation differs from original text provenance')
        catalog[item.path] = replace(item, sha256=expected)
    expected_new_bytes = sum(item.size for item in catalog.values())
    require(expected_new_bytes <= DOWNLOAD_CAP, 'selected downloads exceed the fixed 8GB cap')
    require(shutil.disk_usage(DATA_ROOT).free > expected_new_bytes + FEATURE_CAP,
            'insufficient physical headroom for selected payload and bounded features')
    root.mkdir(parents=True, exist_ok=False)
    bounded = _BoundedClient(client, began + max_seconds, DOWNLOAD_CAP)
    downloads, sequences, raw_actions, selected_actions = [], {}, [], []

    def check():
        require(time.monotonic() - began < max_seconds, 'CPU preparation deadline reached')

    try:
        # Parse all small annotations before transferring any new video payload.
        for row in selected:
            check()
            identity = row['episode_id']
            if identity in retained:
                old = retained[identity]
                row.update({key: old[key] for key in ('video_path', 'annotations_path', 'video_probe',
                                                     'video_sha256', 'video_mtime_ns')})
                row['payload_origin'] = 'retained_subset'
                _validate_payload(row, root, receipt['episodes'])
            else:
                row['payload_origin'] = 'expanded_download'
                row['annotations_path'] = str(root / 'payload' / identity / 'annotations.tar')
                downloads.append(asdict(download_remote_file(bounded, catalog[row['remote_annotations']],
                    row['annotations_path'], min_free_bytes=FEATURE_CAP)))
            bundle = read_annotation_bundle(row['annotations_path'])
            require(static_caption(bundle.caption) == receipt['episodes'][identity]['prompt'],
                    'static caption differs from its retained text embedding')
            sequence = parse_action_document(bundle.action)
            require(sequence.fps == 30, 'source actions must remain 30fps')
            row['selected_source_starts'] = select_starts(sequence, identity)
            raw = [frame.keys for frame in sequence.frames]
            sampled = [sequence.frames[start + offset].keys for start in row['selected_source_starts']
                       for offset in sequence.resampled_offsets(97, output_fps=16)[1:]]
            row['action_statistics'] = dict(raw=action_counts(raw), selected=action_counts(sampled))
            sequences[identity] = sequence
            raw_actions.extend(raw)
            selected_actions.extend(sampled)
            write_json(root / 'prepare-progress.json', dict(status='annotations', parsed=len(sequences),
                selected_episodes=len(selected), transferred_bytes=bounded.transferred))
        statistics = dict(policy=SELECTION_POLICY, raw=action_counts(raw_actions), selected=action_counts(selected_actions),
            per_episode={r['episode_id']: dict(starts=r['selected_source_starts'], **r['action_statistics']) for r in selected},
            caveat='observed labels only; WI/WK coverage and visual response are not assumed')
        write_json(root / 'action-statistics.json', statistics)
        for row in selected:
            check()
            if row['payload_origin'] != 'retained_subset':
                row['video_path'] = str(root / 'payload' / row['episode_id'] / 'video.mp4')
                downloads.append(asdict(download_remote_file(bounded, catalog[row['remote_video']],
                    row['video_path'], min_free_bytes=FEATURE_CAP)))
                row['video_probe'] = ffmpeg_probe(row['video_path'], timeout=min(180, max(1, int(max_seconds - (time.monotonic() - began)))))
                row['video_sha256'] = catalog[row['remote_video']].sha256
                row['video_mtime_ns'] = Path(row['video_path']).stat().st_mtime_ns
            validate_action_video_alignment(sequences[row['episode_id']], row['video_probe'])
            write_json(root / 'prepare-progress.json', dict(status='videos', episode_id=row['episode_id'],
                downloads=downloads, transferred_bytes=bounded.transferred))
        check()
        manifest = root / 'manifests/train.jsonl'
        relocated = root / 'manifests/source-relocated.jsonl'
        current = {r['episode_id']: r for r in selected}
        source_rows = [dict(r, **({k: current[r['episode_id']][k] for k in ('video_path', 'annotations_path')}
                                  if r['episode_id'] in current else {})) for r in originals]
        _jsonl(manifest, selected)
        _jsonl(relocated, source_rows)
        text = root / 'text/train-prompts.pt'
        _save_tensor(text, dict(schema_version=1, kind=SCENE_STATIC_PROMPT_CACHE_KIND,
            prompt_embeds={r['episode_id']: prompts[r['episode_id']] for r in selected}), root / 'text', FEATURE_CAP)
        eval_cache = write_eval_sidecar(root, prompts, receipt, source_sha, manifest)
        result = dict(status='completed', kind='expanded_train_window97_prepare_v1',
            manifest=str(manifest), manifest_sha256=sha256_file(manifest),
            source_manifest=str(SOURCE_MANIFEST), source_manifest_sha256=source_sha,
            relocated_source_manifest=str(relocated), relocated_source_manifest_sha256=sha256_file(relocated),
            original_prompt_cache=str(SOURCE_PROMPTS), original_prompt_cache_sha256=receipt['cache_sha256'],
            original_encoder=receipt['encoder'], prompts=str(text), prompts_sha256=sha256_file(text),
            episodes={r['episode_id']: receipt['episodes'][r['episode_id']] for r in selected},
            eval_prompt_cache=eval_cache, eval_prompt_receipt_sha256=sha256_file(Path(eval_cache).with_suffix('.pt.receipt.json')),
            retained_subset=str(REUSE_ROOT), retained_prepare_sha256=sha256_file(REUSE_ROOT / 'prepare.json'),
            downloads=downloads, download_bytes=bounded.transferred, expected_new_payload_bytes=expected_new_bytes,
            download_cap_bytes=DOWNLOAD_CAP, feature_cap_bytes=FEATURE_CAP, source_revision=OFFICIAL_REVISION,
            selection_policy=SELECTION_POLICY, raw_episodes=len(selected), max_windows_per_episode=4,
            windows=sum(len(r['selected_source_starts']) for r in selected), text_encoder_loaded=False,
            elapsed_seconds=time.monotonic() - began)
        write_json(root / 'prepare.json', result)
        return result
    except Exception as error:
        write_json(root / 'prepare-failed.json', dict(status='failed', error=f'{type(error).__name__}: {error}',
            downloads=downloads, transferred_bytes=bounded.transferred))
        raise


def decode_window97(video_path, *, start_seconds, num_frames=97, target_fps=16, source_fps=30,
                    height=480, width=832, backend='auto', timeout=90):
    import torch
    require((num_frames, target_fps, source_fps, height, width) == (97, 16, 30, 480, 832),
            'expanded cache decoder only supports the fixed RGB97 contract')
    offsets = [(2 * i * source_fps + target_fps) // (2 * target_fps) for i in range(num_frames)]
    command = _ffmpeg_command(Path(video_path), start_seconds=start_seconds, num_frames=181,
                             decode_fps=30, height=height, width=width, nvdec=False)
    command[1:1] = ['-threads', '2', '-filter_threads', '2']
    decoded = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
    require(decoded.returncode == 0 and len(decoded.stdout) == 181 * height * width * 3,
            f'incomplete CPU RGB97 decode: {decoded.stderr[-400:].decode(errors="replace")}')
    pixels = torch.frombuffer(bytearray(decoded.stdout), dtype=torch.uint8).reshape(181, height, width, 3)
    return pixels[offsets].permute(3, 0, 1, 2).contiguous(), 'cpu_ffmpeg_nearest_30_to_16'


class CachedTextVAEEncoder:
    """Original text lookup plus an injected VAE callable; no T5 constructor."""
    def __init__(self, embeddings, bindings, encode_video, provenance):
        import torch
        require(set(embeddings) == set(bindings), 'cached text episode set differs from training bindings')
        self.lookup = {}
        for identity, binding in bindings.items():
            require(binding['split'] == 'train', 'VAE cache encoder must not receive dev embeddings')
            prompt, tensor = binding['prompt'], embeddings[identity]
            require(hashlib.sha256(prompt.encode()).hexdigest() == binding['prompt_sha256'], 'cached prompt text changed')
            require(tensor.ndim == 2 and tensor.shape[1] == 4096 and 1 <= len(tensor) <= 512
                    and tensor.device.type == 'cpu' and bool(torch.isfinite(tensor).all()), 'invalid cached CPU text')
            if prompt in self.lookup:
                require(torch.equal(self.lookup[prompt], tensor), 'same text has conflicting retained embeddings')
            self.lookup[prompt] = tensor
        self.video_encoder, self.provenance = encode_video, provenance

    def encode_text(self, prompts):
        require(isinstance(prompts, list) and len(prompts) == 1 and prompts[0] in self.lookup,
                'unbound prompt cannot trigger text re-encoding')
        return self.lookup[prompts[0]]

    def encode_video(self, pixels):
        return self.video_encoder(pixels)


def validate_prepared(root):
    """Bind new paths to immutable original splits/text and stat-verified payloads."""
    root = _data_path(root)
    prepared = _json(root / 'prepare.json')
    require(prepared.get('status') == 'completed' and prepared.get('kind') == 'expanded_train_window97_prepare_v1',
            'expanded RGB97 preparation is incomplete')
    for key, expected in (('manifest', root / 'manifests/train.jsonl'),
                          ('relocated_source_manifest', root / 'manifests/source-relocated.jsonl'),
                          ('prompts', root / 'text/train-prompts.pt')):
        require(prepared[key] == str(expected) and sha256_file(expected) == prepared[key + '_sha256'],
                f'prepared {key} binding changed')
    require(prepared['source_manifest'] == str(SOURCE_MANIFEST)
            and sha256_file(SOURCE_MANIFEST) == prepared['source_manifest_sha256'], 'original source manifest changed')
    original = {r['episode_id']: r for r in _read_rows(SOURCE_MANIFEST)}
    receipt = _json(SOURCE_PROMPTS.with_suffix('.pt.receipt.json'))
    require(prepared['original_prompt_cache_sha256'] == receipt['cache_sha256'], 'original cached text binding changed')
    rows = _read_rows(prepared['manifest'])
    require(8 <= len(rows) <= 64 and len({r['episode_id'] for r in rows}) == len(rows), 'invalid train episode count')
    require(set(prepared['episodes']) == {r['episode_id'] for r in rows}, 'prepared episode bindings differ')
    require(prepared['retained_subset'] == str(REUSE_ROOT)
            and sha256_file(REUSE_ROOT / 'prepare.json') == prepared['retained_prepare_sha256'], 'retained source changed')
    selected = {r['episode_id']: r for r in rows}
    relocated = _read_rows(prepared['relocated_source_manifest'])
    expected_relocated = [dict(r, **({k: selected[r['episode_id']][k] for k in ('video_path', 'annotations_path')}
                                    if r['episode_id'] in selected else {})) for r in original.values()]
    require(relocated == expected_relocated, 'relocated source changed original rows/splits beyond path relocation')
    for row in rows:
        identity = row['episode_id']
        require(row['split'] == original[identity]['split'] == 'train'
                and prepared['episodes'][identity] == receipt['episodes'][identity], 'train/static binding changed')
        _validate_payload(row, root, prepared['episodes'])
        starts = row['selected_source_starts']
        require(1 <= len(starts) <= 4 and starts == sorted(set(starts))
                and all(b - a >= 181 for a, b in zip(starts, starts[1:])), 'overlapping or excessive selected windows')
    return prepared, rows


def cache(request, directory, lease):
    """Worker body only; caller owns finite GPU lease, accounting and watchdog."""
    import torch
    from training.demo.backend import validate_active_authority
    from scripts.cache_abot_window97_features import cache_window97_manifest
    from utils.wan_wrapper import WanVAEWrapper
    began = time.monotonic()
    root, directory = _data_path(request['subset_dir']), _data_path(directory)
    maximum = request.get('max_seconds')
    require(type(maximum) is int and 60 <= maximum <= 3600, 'cache requires a finite <=3600s deadline')
    require(not (root / 'features').exists() or request.get('reuse_completed') is True,
            'new cache directory required unless explicit verified resume')

    def check():
        require(time.monotonic() - began < maximum - 20, 'RGB97 cache deadline reached')
        require(sum(_stored_bytes(root / name) for name in ('text', 'eval', 'features')) + 25_000_000 < FEATURE_CAP,
                'expanded feature cap reached')
        validate_active_authority(lease)

    check()
    prepared, rows = validate_prepared(root)
    vae_path = _data_path(request['vae_path'])
    require(vae_path.name == 'Wan2.2_VAE.pth' and vae_path.parent.name == Path(prepared['original_encoder']['base_model']).name,
            'VAE identity differs from original cached text model')
    digest = ArtifactHashes(_data_path(request['verification_cache']) if request.get('verification_cache') else None,
                            protected_paths=[vae_path, root / 'prepare.json', Path(prepared['prompts'])])
    require(re.fullmatch(r'[0-9a-f]{64}', request.get('vae_sha256', '')) is not None
            and digest(vae_path) == request['vae_sha256'], 'trusted VAE integrity mismatch')
    embeddings = torch.load(prepared['prompts'], map_location='cpu', weights_only=True)['prompt_embeds']
    by_video, by_id = {r['video_path']: r for r in rows}, {r['episode_id']: r for r in rows}
    vae = encoder = None
    write_json(directory / 'cache-inputs.json', dict(subset_dir=str(root), manifest_sha256=prepared['manifest_sha256'],
        num_frames=97, max_windows_per_episode=4, max_total_windows=256, text_encoder_loaded=False,
        vae=dict(path=str(vae_path), sha256=request['vae_sha256']), max_seconds=maximum))
    try:
        check()
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one leased visible CUDA device required')
        torch.cuda.reset_peak_memory_stats()
        vae = WanVAEWrapper(pretrained_path=str(vae_path)).requires_grad_(False).eval().to(device='cuda', dtype=torch.bfloat16)

        encoded_this_job = 0

        def encode(pixels):
            nonlocal encoded_this_job
            check()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                latent = vae.encode_to_latent(pixels.to(device='cuda', dtype=torch.bfloat16)).cpu()
            encoded_this_job += 1
            elapsed = time.monotonic() - began
            write_json(directory / 'cache-progress.json', dict(status='running',
                newly_encoded_windows=encoded_this_job, planned_total_windows=prepared['windows'],
                elapsed_seconds=elapsed, peak_vram_bytes=torch.cuda.max_memory_allocated(),
                note='new encodes in this job; complete saved episodes may be reused separately'))
            if encoded_this_job == 1 or encoded_this_job % 8 == 0:
                print(f'RGB97 cache: {encoded_this_job} new windows encoded; {elapsed:.1f}s elapsed', flush=True)
            return latent

        encoder = CachedTextVAEEncoder(embeddings, prepared['episodes'], encode,
            dict(kind='wan_vae_with_original_static_text_cache', text_encoder_loaded=False,
                 vae_path=str(vae_path), vae_sha256=request['vae_sha256'],
                 original_prompt_cache_sha256=prepared['original_prompt_cache_sha256']))

        def probe(path):
            check()
            row = by_video[str(path)]
            _validate_payload(row, root, prepared['episodes'])
            return dict(row['video_probe'], video_sha256=row['video_sha256'])

        def starts(sequence, identity, count, seed):
            require(seed == 42, 'prepared sampling seed changed')
            return by_id[identity]['selected_source_starts'][:count]

        def decoder(path, **kwargs):
            check()
            return decode_window97(path, timeout=min(90, max(1, int(maximum - (time.monotonic() - began) - 20))), **kwargs)

        result = cache_window97_manifest(prepared['manifest'], root / 'features', encoder=encoder,
            source_manifest_path=prepared['relocated_source_manifest'], max_windows_per_episode=4,
            max_total_windows=256, shard_size=4, seed=42, decoder=decoder, probe_fn=probe, starts_fn=starts,
            reuse_completed=request.get('reuse_completed') is True,
            execution_metadata=dict(mode='creator_leased_vae_only_window97', selection_policy=SELECTION_POLICY,
                                    prepare_path=str(root / 'prepare.json'), text_encoder_loaded=False))
        require(result['episodes'] == len(rows) and result['windows'] == prepared['windows'], 'cached selection count differs')
        result.update(status='completed', mode='window97_cache', text_encoder_loaded=False,
            eval_prompt_cache=prepared['eval_prompt_cache'], elapsed_seconds=time.monotonic() - began,
            peak_vram_bytes=torch.cuda.max_memory_allocated(), training_started=False)
        write_json(directory / 'cache-result.json', result)
        return result
    except Exception as error:
        write_json(directory / 'cache-failed.json', dict(status='failed', error=f'{type(error).__name__}: {error}',
                                                      elapsed_seconds=time.monotonic() - began))
        raise
    finally:
        del encoder, vae, embeddings
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='phase', required=True)
    cpu = commands.add_parser('prepare')
    cpu.add_argument('--output', required=True)
    cpu.add_argument('--episodes', type=int, default=64)
    cpu.add_argument('--max-seconds', type=int, default=1800)
    cpu.add_argument('--verification-cache')
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, episodes=args.episodes, max_seconds=args.max_seconds,
                             verification_cache=args.verification_cache), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
