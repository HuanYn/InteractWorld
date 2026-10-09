"""Bounded original-train subset + VAE-only feature rebuild; never starts training.

``prepare`` downloads at most 2 decimal GB on CPU. ``cache`` is a worker body
for an external CreatorGuard lease/watchdog, not an independent GPU launcher.
Original manifests, text embeddings and model assets are never overwritten.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from training.demo.contracts import require, write_json
from training.data.action_dataset import (
    CACHE_SCHEMA_VERSION, FROZEN_DEMO_EPISODE, SCENE_STATIC_PROMPT_CACHE_KIND,
    SCENE_STATIC_PROMPT_POLICY, cache_index_path, sha256_file,
)
from training.data.abot_remote import (
    OFFICIAL_REPO_ID, OFFICIAL_REVISION, HuggingFaceClient, download_remote_file,
    _path_from_hf_uri, _path_from_annotation_uri,
)
from training.data.action_schema import parse_action_document
from training.data.safe_annotations import read_annotation_bundle
from scripts.cache_scene_static_prompts import static_caption
from scripts.cache_abot_features import (
    _selected_starts, _sampled_actions, _window_start_seconds, _ffmpeg_command,
    validate_action_video_alignment,
)

DATA_ROOT = Path('/DATA_DISK/InteractWorld')
OLD_DATA = DATA_ROOT / 'legacy-assets/data'
SOURCE_MANIFEST = OLD_DATA / 'manifests/train.jsonl'
SOURCE_PROMPTS = OLD_DATA / 'features/scene-static-prompts-r004.pt'
BYTE_CAP = 2_000_000_000
LATENT_SHAPE = (13, 48, 30, 52)
DEV_TEXT_IDS = ('019285161503b1d49c72e81863eb17ef', '009b81104345761d84612406c8f74c91')


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def _data_path(path):
    value = Path(path).resolve()
    require(value.is_relative_to(DATA_ROOT.resolve()) and value != DATA_ROOT.resolve(),
            'all new subset assets must remain under the project data root')
    return value


def _stored_bytes(directory):
    return sum(path.stat().st_size for path in Path(directory).rglob('*') if path.is_file())


def _save_tensor(path, payload, budget_root, cap=BYTE_CAP):
    import torch
    path = Path(path)
    require(not path.exists(), 'refusing to overwrite an existing tensor artifact')
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    require(_stored_bytes(budget_root) + buffer.tell() <= cap, 'feature byte cap exceeded')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('xb') as stream:
        stream.write(buffer.getbuffer())
    temporary.replace(path)


def select_rows(rows, receipt, count):
    require(type(count) is int and 4 <= count <= 8, 'select exactly 4–8 original train episodes')
    require(len({row['episode_id'] for row in rows}) == len(rows), 'duplicate source episode')
    selected = []
    for row in sorted(rows, key=lambda item: item['episode_id']):
        identity = row['episode_id']
        binding = receipt['episodes'].get(identity, {})
        if (row.get('split') != 'train' or identity == FROZEN_DEMO_EPISODE
                or binding.get('split') != 'train' or not binding.get('prompt')):
            continue
        require(re.fullmatch(r'[0-9a-f]{32}', identity) is not None, 'unsafe source episode ID')
        require(row.get('source_fps') == 30 and row.get('output_fps') == 16, 'source FPS contract differs')
        require(hashlib.sha256(binding['prompt'].encode()).hexdigest() == binding['prompt_sha256'],
                'source prompt receipt text changed')
        if int(row.get('eligible_resampled_windows', {}).get('49', 0)) > 0:
            selected.append(dict(row))
        if len(selected) == count:
            return selected
    raise ValueError('insufficient original-train episodes covered by the static text cache')


def load_selected_prompts(source_path, receipt, rows, *, verifier=sha256_file):
    import torch
    require(receipt.get('schema_version') == 1 and receipt.get('kind') == SCENE_STATIC_PROMPT_CACHE_KIND
            and receipt.get('prompt_policy') == SCENE_STATIC_PROMPT_POLICY, 'unexpected original prompt cache')
    # Once during prepare, not once per window or GPU job. Rebuilt small tensors
    # receive a new receipt; the original migrated receipt is not rewritten.
    require(verifier(source_path) == receipt['cache_sha256'], 'original prompt cache integrity mismatch')
    payload = torch.load(source_path, map_location='cpu', weights_only=True, mmap=True)
    require(payload.get('schema_version') == 1 and payload.get('kind') == SCENE_STATIC_PROMPT_CACHE_KIND,
            'original prompt payload schema mismatch')
    selected = {}
    for row in rows:
        tensor = payload['prompt_embeds'][row['episode_id']]
        require(torch.is_tensor(tensor) and tensor.ndim == 2 and 1 <= tensor.shape[0] <= 512
                and tensor.shape[1] == 4096 and tensor.is_floating_point()
                and bool(torch.isfinite(tensor).all()), 'invalid cached text embedding')
        selected[row['episode_id']] = tensor.detach().to(dtype=torch.bfloat16).clone()
    return selected


class _BoundedClient:
    """Preserve the existing size/hash downloader with a whole-stage byte/time cap."""
    def __init__(self, client, deadline, cap):
        self.client, self.deadline, self.remaining = client, deadline, cap
        self.transferred = 0

    def open_file(self, remote, *, offset):
        require(time.monotonic() < self.deadline and offset == 0, 'prepare deadline/retry boundary reached')
        response = self.client.open_file(remote, offset=offset)
        owner = self

        class Response:
            status, headers = response.status, response.headers
            remaining_file = remote.size

            def read(self, amount=-1):
                require(time.monotonic() < owner.deadline, 'prepare download deadline reached')
                amount = min(amount if amount >= 0 else 4 * 1024 * 1024,
                             owner.remaining, self.remaining_file)
                if amount == 0:
                    return b''
                value = response.read(amount)
                require(len(value) <= amount, 'download response exceeded requested byte bound')
                owner.remaining -= len(value)
                owner.transferred += len(value)
                self.remaining_file -= len(value)
                return value

            def close(self):
                response.close()

        return Response()


def ffmpeg_probe(path, timeout=180):
    """Full CPU decode/count, using existing ffmpeg; ffprobe is not required."""
    command = ['ffmpeg', '-hide_banner', '-nostdin', '-v', 'info', '-xerror', '-threads', '2',
               '-i', str(path), '-map', '0:v:0', '-an', '-sn', '-dn', '-vsync', '0',
               '-progress', 'pipe:1', '-nostats', '-f', 'null', '-']
    result = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
    text = result.stderr.decode(errors='replace')
    require(result.returncode == 0, f'CPU video decode failed: {text[-600:]}')
    line = next((line for line in text.splitlines() if 'Stream #' in line and 'Video:' in line), '')
    fps = re.search(r'([0-9.]+) fps\b', line)
    size = re.search(r'\b([0-9]{2,5})x([0-9]{2,5})\b', line)
    counts = re.findall(rb'^frame=([0-9]+)\s*$', result.stdout, re.MULTILINE)
    require(fps is not None and abs(float(fps.group(1)) - 30) < .05 and size is not None and counts,
            'video metadata does not prove the expected 30fps/count/size contract')
    frames = int(counts[-1])
    return dict(codec='decoded_by_ffmpeg', fps=30., frames=frames,
                width=int(size.group(1)), height=int(size.group(2)), duration_seconds=frames / 30.,
                frame_count_source='full_ffmpeg_CPU_decode', file_bytes=Path(path).stat().st_size)


def prepare(output, *, episodes=8, max_windows=2, max_seconds=1800, verification_cache=None):
    from training.demo.relocation import ArtifactHashes
    began = time.monotonic()
    require(type(max_windows) is int and 1 <= max_windows <= 2, 'at most two windows per episode')
    require(type(max_seconds) is int and 60 <= max_seconds <= 1800, 'finite CPU prepare duration required')
    root = _data_path(output)
    require(not root.exists(), 'prepare requires a new subset directory; old/failed artifacts are retained')
    receipt = _json(SOURCE_PROMPTS.with_suffix('.pt.receipt.json'))
    source_hash = sha256_file(SOURCE_MANIFEST)
    require(source_hash == receipt['manifest_sha256'], 'trusted original training manifest differs from text receipt')
    rows = [json.loads(line) for line in SOURCE_MANIFEST.read_text(encoding='utf-8').splitlines() if line.strip()]
    selected = select_rows(rows, receipt, episodes)
    paths = []
    for row in selected:
        row['remote_video'] = _path_from_hf_uri(row['video_uri'], repo_id=OFFICIAL_REPO_ID, revision=OFFICIAL_REVISION)
        row['remote_annotations'] = _path_from_annotation_uri(row['annotations_uri'], repo_id=OFFICIAL_REPO_ID, revision=OFFICIAL_REVISION)
        paths.extend([row['remote_video'], row['remote_annotations']])
    client = HuggingFaceClient(repo_id=OFFICIAL_REPO_ID, revision=OFFICIAL_REVISION, timeout_seconds=30.)
    catalog = {item.path: item for item in client.get_paths_info(paths)}
    require(set(catalog) == set(paths), 'pinned remote metadata missing files')
    for row in selected:
        video = catalog[row['remote_video']]
        video.validate(require_sha256=True)
        annotation = catalog[row['remote_annotations']]
        expected = receipt['episodes'][row['episode_id']]['annotations_sha256']
        require(annotation.sha256 in (None, expected), 'remote annotation differs from trusted cached text')
        catalog[annotation.path] = replace(annotation, sha256=expected)
    # Prefer eight episodes, reduce to six/four only on this explicit byte cap.
    candidate_counts = [episodes] + [count for count in (6, 4) if count < episodes]
    fits = [(count, sum(catalog[row['remote_video']].size + catalog[row['remote_annotations']].size
                       for row in selected[:count])) for count in candidate_counts]
    fits = [(count, size) for count, size in fits if size <= BYTE_CAP]
    require(fits, 'even four selected full episodes exceed the 2GB download cap')
    count, total = fits[0]
    selected = selected[:count]
    require(total <= BYTE_CAP, 'selected full videos + annotations exceed the 2GB download cap')
    require(shutil.disk_usage(DATA_ROOT).free > total + BYTE_CAP, 'insufficient space for bounded payload/features')
    by_id = {row['episode_id']: row for row in rows}
    references = []
    for identity in DEV_TEXT_IDS:
        row, binding = by_id[identity], receipt['episodes'][identity]
        require(row['split'] == binding['split'] == 'dev', 'fixed comparison image must stay dev-only')
        references.append(dict(row, cache_state='text_only_reference_no_video_features',
            source_reference=dict(manifest=str(SOURCE_MANIFEST), manifest_sha256=source_hash,
                                  feature_index_sha256=receipt['feature_index_sha256'])))
    digest = ArtifactHashes(_data_path(verification_cache) if verification_cache else None,
                            protected_paths=[SOURCE_MANIFEST, SOURCE_PROMPTS, SOURCE_PROMPTS.with_suffix('.pt.receipt.json')])
    prompts = load_selected_prompts(SOURCE_PROMPTS, receipt, selected + references, verifier=digest)
    root.mkdir(parents=True, exist_ok=False)
    bounded = _BoundedClient(client, began + max_seconds, BYTE_CAP)
    download_records, bindings = [], {}
    try:
        for row in selected:
            identity = row['episode_id']
            for remote_key, local_key, filename in [('remote_annotations', 'annotations_path', 'annotations.tar'),
                                                     ('remote_video', 'video_path', 'video.mp4')]:
                target = root / 'payload' / identity / filename
                downloaded = download_remote_file(bounded, catalog[row[remote_key]], target,
                                                   min_free_bytes=BYTE_CAP)
                row[local_key] = str(target)
                download_records.append(asdict(downloaded))
            bundle = read_annotation_bundle(row['annotations_path'])
            binding = receipt['episodes'][identity]
            require(static_caption(bundle.caption) == binding['prompt'], 'downloaded static text differs from original embedding')
            sequence = parse_action_document(bundle.action)
            probe = ffmpeg_probe(row['video_path'], min(180, max(1, int(began + max_seconds - time.monotonic()))))
            validate_action_video_alignment(sequence, probe)
            starts = _selected_starts(sequence, identity, max_windows, 42)
            require(starts, 'downloaded episode has no aligned 49-frame windows')
            row['selected_source_starts'] = starts
            row['video_probe'] = probe
            row['video_sha256'] = catalog[row['remote_video']].sha256
            row['video_mtime_ns'] = Path(row['video_path']).stat().st_mtime_ns
            bindings[identity] = dict(binding)
            write_json(root / 'prepare-progress.json', dict(status='downloading', downloads=download_records))
        manifest = root / 'manifests/train.jsonl'
        for row in references:
            bindings[row['episode_id']] = dict(receipt['episodes'][row['episode_id']])
        _jsonl(manifest, selected + references)
        prompts_path = root / 'text/selected-prompts.pt'
        _save_tensor(prompts_path, dict(schema_version=1, kind=SCENE_STATIC_PROMPT_CACHE_KIND,
                                      prompt_embeds=prompts), root / 'text')
        result = dict(status='completed', mode='rebuild_subset_prepare', manifest=str(manifest),
            manifest_sha256=sha256_file(manifest), source_manifest=str(SOURCE_MANIFEST),
            source_manifest_sha256=source_hash, original_prompt_cache=str(SOURCE_PROMPTS),
            original_prompt_cache_sha256=receipt['cache_sha256'], original_encoder=receipt['encoder'],
            prompt_policy=SCENE_STATIC_PROMPT_POLICY, prompts=str(prompts_path),
            prompts_sha256=sha256_file(prompts_path), episodes=bindings, downloads=download_records,
            download_bytes=bounded.transferred, download_cap_bytes=BYTE_CAP, feature_cap_bytes=BYTE_CAP,
            max_windows_per_episode=max_windows, elapsed_seconds=time.monotonic() - began,
            source_revision=OFFICIAL_REVISION, raw_episodes=len(selected),
            dev_text_only_ids=list(DEV_TEXT_IDS),
            windows=sum(len(row['selected_source_starts']) for row in selected))
        write_json(root / 'prepare.json', result)
        return result
    except Exception as error:
        write_json(root / 'prepare-failed.json', dict(status='failed', error=f'{type(error).__name__}: {error}',
            downloads=download_records, transferred_bytes=bounded.transferred))
        raise


def decode_window(path, start, sequence, timeout):
    """Exact existing nearest-source-index policy with finite CPU ffmpeg timeout."""
    import torch
    offsets = sequence.resampled_offsets(49, output_fps=16)
    count = offsets[-1] + 1
    command = _ffmpeg_command(Path(path), start_seconds=_window_start_seconds(sequence, start),
        num_frames=count, decode_fps=30, height=480, width=832, nvdec=False)
    command[1:1] = ['-threads', '2', '-filter_threads', '2']
    result = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
    require(result.returncode == 0 and len(result.stdout) == count * 480 * 832 * 3,
            f'incomplete CPU window decode: {result.stderr[-400:].decode(errors="replace")}')
    frames = torch.frombuffer(bytearray(result.stdout), dtype=torch.uint8).reshape(count, 480, 832, 3)
    return frames[list(offsets)].permute(3, 0, 1, 2).contiguous()


def write_features(root, prepared, rows, prompts, encode, *, check=lambda: None):
    """CPU-injectable writer; actual cache() supplies the leased VAE encoder."""
    import torch
    root = Path(root)
    feature_root = root / 'features'
    feature_root.mkdir(exist_ok=False)
    manifest = root / 'manifests/train.jsonl'
    index = cache_index_path(manifest)
    entries = []
    for row in rows:
        check()
        identity = row['episode_id']
        if row['split'] == 'dev':
            require(identity in DEV_TEXT_IDS and row.get('cache_state') == 'text_only_reference_no_video_features',
                    'only fixed dev text references may omit video features')
            entries.append(dict(episode_id=identity, split='dev', num_windows=0, shards=[],
                cache_state=row['cache_state'], source_reference=row['source_reference']))
            continue
        require(row['split'] == 'train' and identity in prompts and identity in prepared['episodes'],
                'unbound/non-training cache episode')
        sequence = parse_action_document(read_annotation_bundle(row['annotations_path']).action)
        require(1 <= len(row['selected_source_starts']) <= 2, 'window count outside bounded subset')
        latents, actions = [], []
        for start in row['selected_source_starts']:
            check()
            latent = encode(row, start, sequence).detach().cpu()
            if latent.ndim == 5 and latent.shape[0] == 1:
                latent = latent[0]
            require(tuple(latent.shape) == LATENT_SHAPE and bool(torch.isfinite(latent).all()), 'invalid VAE49 latent')
            latents.append(latent.to(torch.bfloat16))
            actions.append(_sampled_actions(sequence, start))
        shard = feature_root / 'episodes' / identity / 'shard-00000.pt'
        _save_tensor(shard, dict(schema_version=1, episode_id=identity,
            clean_latents=torch.stack(latents), actions=torch.stack(actions),
            prompt_embeds=prompts[identity], source_starts=torch.tensor(row['selected_source_starts'])), feature_root)
        shards = [dict(path=shard.relative_to(feature_root).as_posix(), sha256=sha256_file(shard), samples=len(latents))]
        source = dict(video=row['video_probe'], annotations_sha256=prepared['episodes'][identity]['annotations_sha256'])
        source['video']['video_sha256'] = row['video_sha256']
        bound = dict(schema_version=1, episode_id=identity, split='train', num_windows=len(latents),
            shards=shards, source=source, prompt_policy=SCENE_STATIC_PROMPT_POLICY,
            selected_source_starts=row['selected_source_starts'])
        receipt_path = shard.parent / 'receipt.json'
        write_json(receipt_path, bound)
        entries.append(dict(bound, episode_receipt=dict(path=receipt_path.relative_to(feature_root).as_posix(),
                                                        sha256=sha256_file(receipt_path))))
    _jsonl(index, entries)
    feature_receipt = dict(schema_version=1, manifest=str(manifest), manifest_sha256=sha256_file(manifest),
        index=str(index), index_sha256=sha256_file(index), episodes=len(entries),
        windows=sum(item['num_windows'] for item in entries), prompt_policy=SCENE_STATIC_PROMPT_POLICY,
        kind='action49_rebuilt_static_subset_v1', config=dict(num_frames=49, target_fps=16, height=480, width=832),
        source_prepare=str(root / 'prepare.json'), vae=prepared.get('vae'),
        text_encoder_loaded=False, original_prompt_cache_sha256=prepared['original_prompt_cache_sha256'])
    write_json(index.with_suffix(index.suffix + '.receipt.json'), feature_receipt)
    sidecar = feature_root / 'scene-static-subset.pt'
    _save_tensor(sidecar, dict(schema_version=1, kind=SCENE_STATIC_PROMPT_CACHE_KIND, prompt_embeds=prompts), feature_root)
    sidecar_receipt = dict(schema_version=1, kind=SCENE_STATIC_PROMPT_CACHE_KIND,
        prompt_policy=SCENE_STATIC_PROMPT_POLICY, cache_path=str(sidecar), cache_sha256=sha256_file(sidecar),
        manifest_sha256=feature_receipt['manifest_sha256'], feature_index_sha256=feature_receipt['index_sha256'],
        feature_receipt_sha256=sha256_file(index.with_suffix(index.suffix + '.receipt.json')),
        encoder=prepared['original_encoder'], episodes=prepared['episodes'],
        reuse_policy='exact original episode/static-text embedding; new subset binding; no T5 encoding')
    write_json(sidecar.with_suffix('.pt.receipt.json'), sidecar_receipt)
    require(_stored_bytes(feature_root) <= BYTE_CAP, 'feature output exceeded 2GB')
    total_features = _stored_bytes(feature_root) + _stored_bytes(root / 'text')
    require(total_features <= BYTE_CAP, 'combined text/video feature outputs exceeded 2GB')
    return dict(manifest=str(manifest), feature_index=str(index), prompt_cache_path=str(sidecar),
                episodes=sum(row['split'] == 'train' for row in rows),
                dev_text_only_ids=[row['episode_id'] for row in rows if row['split'] == 'dev'],
                windows=feature_receipt['windows'], feature_bytes=total_features)


def cache(request, directory, lease):
    import torch
    from training.demo.backend import validate_active_authority
    from training.demo.relocation import ArtifactHashes
    from utils.wan_wrapper import WanVAEWrapper

    began = time.monotonic()
    root, directory = _data_path(request['subset_dir']), _data_path(directory)
    require(not (root / 'features').exists(), 'cache requires a new feature directory; failed outputs are retained')
    maximum = request.get('max_seconds')
    require(type(maximum) is int and 60 <= maximum <= 3600, 'finite cache watchdog duration required')

    def check():
        require(time.monotonic() - began < maximum - 15, 'cache deadline reached')
        validate_active_authority(lease)

    check()
    prepared = _json(root / 'prepare.json')
    manifest, prompt_path = root / 'manifests/train.jsonl', root / 'text/selected-prompts.pt'
    require(prepared['status'] == 'completed' and prepared['manifest'] == str(manifest)
            and sha256_file(manifest) == prepared['manifest_sha256']
            and sha256_file(prompt_path) == prepared['prompts_sha256'], 'prepared subset changed')
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    training_rows = [row for row in rows if row['split'] == 'train']
    require(4 <= len(training_rows) <= 8 and sum(len(row['selected_source_starts']) for row in training_rows) <= 16,
            'prepared subset exceeds episode/window cap')
    if 'payload_reference_root' in prepared:
        from training.studio.reselect_subset import validate_payload_reference
        validate_payload_reference(root, prepared, rows)
    else:
        for row in training_rows:
            for field, filename in [('video_path', 'video.mp4'), ('annotations_path', 'annotations.tar')]:
                require(Path(row[field]).resolve() == root / 'payload' / row['episode_id'] / filename,
                        'prepared payload path escaped subset')
            require(Path(row['video_path']).stat().st_size == row['video_probe']['file_bytes']
                    and Path(row['video_path']).stat().st_mtime_ns == row['video_mtime_ns']
                    and sha256_file(row['annotations_path']) == prepared['episodes'][row['episode_id']]['annotations_sha256'],
                    'verified prepared payload changed')
    prompts = torch.load(prompt_path, map_location='cpu', weights_only=True)['prompt_embeds']
    vae_path = _data_path(request['vae_path'])
    require(vae_path.name == 'Wan2.2_VAE.pth' and vae_path.parent.name == Path(prepared['original_encoder']['base_model']).name,
            'VAE model identity differs from original text features')
    digest = ArtifactHashes(_data_path(request['verification_cache']) if request.get('verification_cache') else None,
                            protected_paths=[vae_path, manifest, prompt_path, root / 'prepare.json'])
    require(re.fullmatch(r'[0-9a-f]{64}', request.get('vae_sha256', '')) is not None
            and digest(vae_path) == request['vae_sha256'], 'trusted VAE integrity mismatch')
    prepared['vae'] = dict(path=str(vae_path), sha256=request['vae_sha256'])
    write_json(directory / 'cache-inputs.json', dict(subset_dir=str(root), manifest_sha256=prepared['manifest_sha256'],
        vae=prepared['vae'], text_encoder_loaded=False, max_seconds=maximum))
    vae = None
    try:
        check()
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one leased visible CUDA device required')
        torch.cuda.reset_peak_memory_stats()
        vae = WanVAEWrapper(pretrained_path=str(vae_path)).requires_grad_(False).eval().to(device='cuda', dtype=torch.bfloat16)

        def encode(row, start, sequence):
            check()
            pixels = decode_window(row['video_path'], start, sequence,
                                   min(90, max(1, int(maximum - (time.monotonic() - began) - 15))))
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                normalized = pixels.unsqueeze(0).float().div(127.5).sub(1.)
                return vae.encode_to_latent(normalized.to(device='cuda', dtype=torch.bfloat16)).cpu()

        result = write_features(root, prepared, rows, prompts, encode, check=check)
        result.update(status='completed', mode='rebuild_subset_cache', elapsed_seconds=time.monotonic() - began,
                      peak_vram_bytes=torch.cuda.max_memory_allocated(), text_encoder_loaded=False, training_started=False)
        write_json(directory / 'cache-result.json', result)
        return result
    except Exception as error:
        write_json(directory / 'cache-failed.json', dict(status='failed', error=f'{type(error).__name__}: {error}',
                                                       elapsed_seconds=time.monotonic() - began))
        raise
    finally:
        del vae
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='phase', required=True)
    cpu = commands.add_parser('prepare')
    cpu.add_argument('--output', required=True)
    cpu.add_argument('--episodes', type=int, default=8)
    cpu.add_argument('--max-windows', type=int, default=2)
    cpu.add_argument('--max-seconds', type=int, default=1800)
    cpu.add_argument('--verification-cache')
    gpu = commands.add_parser('cache')
    gpu.add_argument('--request', required=True)
    gpu.add_argument('--directory', required=True)
    gpu.add_argument('--lease', required=True)
    args = parser.parse_args()
    result = (prepare(args.output, episodes=args.episodes, max_windows=args.max_windows, max_seconds=args.max_seconds,
                      verification_cache=args.verification_cache)
              if args.phase == 'prepare' else cache(_json(args.request), args.directory, _json(args.lease)))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
