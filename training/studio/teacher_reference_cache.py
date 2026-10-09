"""Bounded original-frame character-reference cache for teacher-reference-pilot-v1.

CLI only prepares CPU records. ``run(request, directory, lease)`` is a worker
body: an external leased worker must provide the existing accounting/watchdog.
It never launches training, substitutes preview JPEGs, or reuses video latents.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

from training.demo.contracts import contained, require, sha256, write_json
from training.demo.relocation import ArtifactHashes
from training.reference_crops import SOURCE_KIND, crop_reference
from training.reference_dataset import (PAYLOAD_KIND, PURPOSE_COUNTS, RECEIPT_SCHEMA,
    REFERENCE_SHAPE, STAGE, ReferenceFeatureStore, reference_sample_id, tensor_sha256)
from training.studio import balanced_turn_cache as balanced

MODE = 'teacher_reference_cache'
REVISION = STAGE
SOURCE_REVISION = 'creator-reference-pilot-v1'
SCHEMA = 'teacher_reference_cache_result_v1'
DATA_ROOT = balanced.DATA_ROOT
ROOT = balanced.BASE / STAGE
JOB_ID = 'ref-cache10-v1'
CACHE_ROOT = ROOT / 'data/reference'
ANNOTATIONS = ROOT / 'reference-boxes.json'
ANNOTATION_SHA256 = 'fd7ab7365ee5bdaa15d787056d2c62558569442dea1974f15a573da2198400cc'
MAX_SECONDS, CLEANUP_SECONDS, OUTPUT_CAP = 900, 60, 100_000_000
VAE_SHA256 = balanced.VAE_SHA256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def _exclusive_jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n')


def _require_data_root():
    require(DATA_ROOT.is_absolute() and DATA_ROOT.as_posix().startswith('/data/'),
            'cache prepare/run must operate on the remote /data filesystem')
    contained(ROOT, DATA_ROOT)


def bind_annotations(document, original):
    """Join the ten prior raw windows and the initial-frame-only annotation set."""
    import torch
    require(document.get('schema_version') == 1 and document.get('stage') == 'teacher-reference-integration-v1'
            and document.get('future_frames_used_for_reference_selection') is False
            and document.get('detector_used') is False and document.get('not_user_annotation') is True,
            'existing assistant initial-frame annotation provenance required')
    require(document.get('valid_count') == 9 and document.get('invalid_count') == 1
            and document.get('coordinate_system') == dict(order='left,top,right,bottom', right_bottom='exclusive',
                image_width=832, image_height=480), 'fixed9 valid/1invalid raw832x480 coordinates required')
    annotations = document.get('rows', [])
    windows = original.get('windows', [])
    require(len(annotations) == len(windows) == 10, 'exact original10 windows and10 annotations required')
    by_id = {row['sample_id']: row for row in annotations}
    require(len(by_id) == 10, 'duplicate annotation sample_id')
    rows = []
    dummy = torch.zeros(3, 480, 832, dtype=torch.uint8)
    for window in windows:
        identity = reference_sample_id(window['episode_id'], window['source_start'])
        require(identity in by_id, 'original source window lacks a matching annotation')
        annotated = by_id[identity]
        purpose = window['purpose']
        require(purpose in PURPOSE_COUNTS and window['split'] == 'train'
                and window['candidate_role'] == ('train_candidate' if purpose == 'train' else annotated['role'])
                and (purpose == 'train' or window['candidate_role'] in ('qualification_pair', 'holdout_candidate')),
                'original training/qualification role differs')
        require(annotated['episode'] == window['episode_id'] and annotated['source_start'] == window['source_start']
                and annotated['video_path'] == window['video_path'] and annotated['role'] == window['candidate_role']
                and annotated['target'] == window['target'] and annotated['purpose'] in (None, purpose),
                'annotation must match original episode/source_start/video/role/target')
        exact_seek = (window['source_start'] * 1_000_000 // 30) / 1_000_000
        require(window['source_start_seconds'] == exact_seek and window['output97_source_indices'][0] == window['source_start'],
                'raw30fps exact initial-frame offset differs')
        crop_reference(dummy, annotated, sample_id=identity, source_start=window['source_start'], source_kind=SOURCE_KIND)
        row = copy.deepcopy(annotated)
        row.pop('first_frame', None)  # Annotation JPEG is never an input path.
        row.update(episode_id=window['episode_id'], purpose=purpose, original_split='train',
            source_start_seconds=exact_seek, source_fps=30, initial_path=window['initial_path'],
            initial_sha256=window['initial_sha256'])
        rows.append(row)
    require(len({r['sample_id'] for r in rows}) == 10 and sum(r['valid'] for r in rows) == 9,
            'duplicate window or validity count changed')
    for purpose, count in PURPOSE_COUNTS.items():
        require(sum(row['purpose'] == purpose for row in rows) == count, 'fixed6train/4qualification partition changed')
    require({r['episode_id'] for r in rows if r['purpose'] == 'train'}.isdisjoint(
            r['episode_id'] for r in rows if r['purpose'] == 'qualification'), 'qualification episode leaked into train')
    require(all(row['valid'] for row in rows if row['purpose'] == 'train')
            and all(row['purpose'] == 'qualification' and row['role'] == 'holdout_candidate'
                    and row['target'] == 'J' for row in rows if not row['valid']), 'invalid slot must remain held-out J')
    return sorted(rows, key=lambda row: (row['purpose'] != 'train', row['sample_id']))


def source_binding(annotations=ANNOTATIONS):
    annotations = contained(annotations, ROOT)
    require(sha256(annotations) == ANNOTATION_SHA256, 'frozen initial-frame annotation hash differs')
    original = balanced.validate_completed_cache()
    prepared_path = balanced.ROOT / 'data/prepare.json'
    original_binding = read(prepared_path)['binding']
    rows = bind_annotations(read(annotations), original)
    for row in rows:
        source = original_binding['source_rows'][row['episode_id']]
        require(source['video_path'] == row['video_path'], 'original raw-video path differs')
        row.update(video_sha256=source['video_sha256'],
                   video_stat=original_binding['source_stats'][row['video_path']])
        contained(row['video_path'], DATA_ROOT)
        contained(row['initial_path'], balanced.ROOT / 'data/initial')
    directory = balanced.ROOT / balanced.JOB_ID
    files = {str(path): sha256(path) for path in [annotations, prepared_path,
        *(directory / (name + '.json') for name in ('request', 'result', 'settlement'))]}
    vae_path = contained(original_binding['vae_path'], DATA_ROOT)
    require(original_binding['vae_sha256'] == VAE_SHA256, 'same original Wan2.2 VAE required')
    return dict(rows=rows, files=files, annotations=str(annotations), annotation_sha256=ANNOTATION_SHA256,
        source_cache_result=str(directory / 'result.json'), source_cache_result_sha256=sha256(directory / 'result.json'),
        vae_path=str(vae_path), vae_sha256=VAE_SHA256, vae_stat=balanced.stat_identity(vae_path))


def validate_request(request, directory=None):
    fixed = dict(mode=MODE, experiment_revision=REVISION, source_revision=SOURCE_REVISION, job_id=JOB_ID,
        max_seconds=MAX_SECONDS, cleanup_seconds=CLEANUP_SECONDS, output_cap_bytes=OUTPUT_CAP,
        train_windows=6, qualification_windows=4, new_reference_images=9, invalid_reference_slots=1,
        prepared_path=str(ROOT / 'data/prepare.json'), cache_root=str(CACHE_ROOT),
        automatic_training=False, reuse_completed=False)
    require(all(request.get(k) == v and type(request.get(k)) is type(v) for k, v in fixed.items())
            and set(request) == set(fixed) | {'prepared_sha256'}, 'fixed bounded reference cache request required')
    require(isinstance(request['prepared_sha256'], str) and len(request['prepared_sha256']) == 64,
            'prepared reference hash required')
    if directory is not None:
        require(Path(directory).resolve() == (ROOT / JOB_ID).resolve(), 'unique reference cache job directory required')


def prepare(*, output=None, annotations=ANNOTATIONS, source_revision=SOURCE_REVISION, max_seconds=MAX_SECONDS):
    """Remote CPU only: bind settled source records and write new small manifests."""
    _require_data_root()
    output = contained(output or ROOT / JOB_ID, ROOT)
    require(output == (ROOT / JOB_ID).resolve() and not output.exists() and not (ROOT / 'data').exists(),
            'fresh reference data/job directory required; no overwrite')
    require(source_revision == SOURCE_REVISION and max_seconds == MAX_SECONDS, 'fixed revision and900-second cache job required')
    binding = source_binding(annotations)
    manifests = {}
    for purpose, count in PURPOSE_COUNTS.items():
        path = ROOT / f'data/manifests/{purpose}.reference.jsonl'
        _exclusive_jsonl(path, [row for row in binding['rows'] if row['purpose'] == purpose])
        manifests[purpose] = dict(path=str(path), sha256=sha256(path), windows=count)
    prepared = dict(schema=SCHEMA, stage=STAGE, status='cpu_prepared', binding=binding, manifests=manifests,
                    model_weights_loaded=False, old_latents_read=False, reference_images_encoded=0)
    _exclusive_json(ROOT / 'data/prepare.json', prepared)
    request = dict(mode=MODE, experiment_revision=REVISION, source_revision=SOURCE_REVISION, job_id=JOB_ID,
        max_seconds=MAX_SECONDS, cleanup_seconds=CLEANUP_SECONDS, output_cap_bytes=OUTPUT_CAP,
        train_windows=6, qualification_windows=4, new_reference_images=9, invalid_reference_slots=1,
        prepared_path=str(ROOT / 'data/prepare.json'), prepared_sha256=sha256(ROOT / 'data/prepare.json'),
        cache_root=str(CACHE_ROOT), automatic_training=False, reuse_completed=False)
    validate_request(request, output)
    _exclusive_json(output / 'request.json', request)
    return request


def validate_prepared(request):
    require(sha256(request['prepared_path']) == request['prepared_sha256'], 'prepared reference records changed')
    prepared = read(request['prepared_path'])
    require(prepared.get('schema') == SCHEMA and prepared.get('stage') == STAGE
            and prepared.get('status') == 'cpu_prepared', 'reference preparation schema differs')
    for path, digest in prepared['binding']['files'].items():
        require(sha256(path) == digest, 'bound source/annotation record changed')
    for purpose in PURPOSE_COUNTS:
        info = prepared['manifests'][purpose]
        require(info['path'] == str(ROOT / f'data/manifests/{purpose}.reference.jsonl')
                and sha256(info['path']) == info['sha256'], 'reference purpose manifest changed')
    for row in prepared['binding']['rows']:
        require(balanced.stat_identity(contained(row['video_path'], DATA_ROOT)) == row['video_stat'], 'raw-video stat changed')
        require(sha256(row['initial_path']) == row['initial_sha256'], 'original unoverlaid RGB witness changed')
    return prepared


def decode_initial(row, *, timeout):
    """Same exact seek/scale/crop as the bound native97 cache, one RGB frame."""
    import torch
    from scripts.cache_abot_features import _ffmpeg_command
    require(type(row['source_start']) is int and row['source_start'] >= 0
            and row['source_start_seconds'] == (row['source_start'] * 1_000_000 // 30) / 1_000_000,
            'exact raw30fps initial frame required')
    command = _ffmpeg_command(Path(row['video_path']), start_seconds=row['source_start_seconds'],
        num_frames=1, decode_fps=30, height=480, width=832, nvdec=False)
    command[1:1] = ['-nostdin', '-threads', '2']
    process = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
    require(process.returncode == 0 and len(process.stdout) == 480 * 832 * 3,
            'incomplete original initial-frame decode: ' + process.stderr[-400:].decode(errors='replace'))
    return torch.frombuffer(bytearray(process.stdout), dtype=torch.uint8).reshape(480, 832, 3).permute(2, 0, 1).contiguous()


def _pixel_hash(rgb):
    return hashlib.sha256(rgb.contiguous().numpy().tobytes()).hexdigest()


def write_reference_cache(prepared, *, encode, decode, check):
    """Materialize this fixed cache using injected I/O; test doubles prove no GPU quality.

    Production encode receives normalized [1,3,1,256,256] RGB, and must actually
    return WanVAEWrapper's [1,1,48,16,16]. Invalid slots never invoke encode.
    """
    import numpy as np
    import torch
    require(not CACHE_ROOT.exists(), 'reference cache already exists; no overwrite/retry')
    binding = prepared['binding']
    cached, encoded_shapes = [], []
    for row in binding['rows']:
        check('decode_' + row['sample_id'])
        raw = decode(row)
        witness = np.load(row['initial_path'], allow_pickle=False)
        require(witness.dtype == np.uint8 and witness.shape == (480, 832, 3)
                and torch.equal(raw, torch.from_numpy(witness.copy()).permute(2, 0, 1)),
                'new raw decode does not match the prior exact unoverlaid initial RGB; no fallback')
        crop = crop_reference(raw, row, sample_id=row['sample_id'], source_start=row['source_start'], source_kind=SOURCE_KIND)
        crop_digest = None
        if crop.valid:
            check('encode_' + row['sample_id'])
            pixels = crop.crop.float().div(127.5).sub(1).unsqueeze(0).unsqueeze(2)
            actual = encode(pixels)
            require(torch.is_tensor(actual) and tuple(actual.shape) == (1, 1, 48, 16, 16)
                    and actual.device.type == 'cpu' and actual.is_floating_point()
                    and bool(torch.isfinite(actual).all()), 'actual VAE must confirm finite CPU[1,1,48,16,16]')
            encoded_shapes.append(list(actual.shape))
            latent = actual.permute(0, 2, 1, 3, 4).to(torch.bfloat16).contiguous()
            crop_digest = _pixel_hash(crop.crop)
        else:
            latent = torch.zeros(REFERENCE_SHAPE, dtype=torch.bfloat16)
        mask = torch.tensor([crop.valid], dtype=torch.bool)
        pixel_digest = _pixel_hash(raw)
        path = CACHE_ROOT / 'tensors' / (row['sample_id'] + '.pt')
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(kind=PAYLOAD_KIND, stage=STAGE, sample_id=row['sample_id'], episode_id=row['episode_id'],
            source_start=row['source_start'], purpose=row['purpose'], ref_latents=latent, ref_mask=mask,
            annotation_sha256=binding['annotation_sha256'], vae_sha256=binding['vae_sha256'],
            raw_rgb_sha256=pixel_digest, crop_rgb_sha256=crop_digest)
        with path.open('xb') as stream:
            torch.save(payload, stream)
        cached.append(dict(row, payload_path=str(path), payload_sha256=sha256(path),
            raw_rgb_sha256=pixel_digest, crop_rgb_sha256=crop_digest, latent_sha256_float32=tensor_sha256(latent)))
        check('saved_' + row['sample_id'])
    require(len(cached) == 10 and len(encoded_shapes) == 9, 'exactly ten slots and nine actual VAE encodes required')
    indexes = {}
    for purpose, count in PURPOSE_COUNTS.items():
        rows = [row for row in cached if row['purpose'] == purpose]
        index = CACHE_ROOT / f'{purpose}.references.jsonl'
        manifest = prepared['manifests'][purpose]
        _exclusive_jsonl(index, rows)
        receipt = dict(schema=RECEIPT_SCHEMA, stage=STAGE, status='completed', purpose=purpose,
            index=str(index), index_sha256=sha256(index), manifest=manifest['path'], manifest_sha256=manifest['sha256'],
            windows=count, valid_count=sum(row['valid'] for row in rows), reference_shape=list(REFERENCE_SHAPE),
            observed_vae_output_shape=encoded_shapes[0], source_kind=SOURCE_KIND, original_split='train',
            annotation_sha256=binding['annotation_sha256'], source_cache_result_sha256=binding['source_cache_result_sha256'],
            vae_sha256=binding['vae_sha256'], actual_vae_encodes=sum(row['valid'] for row in rows),
            preview_jpeg_input=False, future_reference_frames_used=False, old_video_latents_read=False)
        receipt_path = Path(str(index) + '.receipt.json')
        _exclusive_json(receipt_path, receipt)
        indexes[purpose] = dict(path=str(index), sha256=sha256(index), receipt=str(receipt_path),
            receipt_sha256=sha256(receipt_path), manifest=manifest['path'], manifest_sha256=manifest['sha256'], windows=count)
        store = ReferenceFeatureStore(index, purpose=purpose, expected_index_sha256=indexes[purpose]['sha256'])
        for row in rows:
            store.load(row['sample_id'])
    return dict(indexes=indexes, windows=cached, actual_vae_encodes=9, decoded_initial_frames=10,
                observed_vae_output_shape=encoded_shapes[0], reference_shape=list(REFERENCE_SHAPE))


def run(request, directory, lease):
    """Only the existing finite leased worker may invoke this GPU body."""
    import torch
    from training.demo.backend import validate_active_authority, validate_lease
    _require_data_root()
    directory = contained(directory, ROOT)
    validate_request(request, directory)
    require(read(directory / 'request.json') == request and lease['gpu_index'] in (1, 2),
            'unchanged request and existing Creator GPU1/2 lease required')
    validate_lease(lease, max_seconds=MAX_SECONDS, job_id=JOB_ID, request_sha256=sha256(directory / 'request.json'))
    require(not CACHE_ROOT.exists() and not (directory / 'progress.json').exists(), 'no reference cache overwrite or retry')
    began, vae = time.monotonic(), None
    def check(phase):
        validate_active_authority(lease)
        require(time.monotonic() - began < MAX_SECONDS - 20, 'reference cache finite completion margin reached')
        require(balanced.native._stored_bytes(ROOT) < OUTPUT_CAP - 5_000_000, 'reference cache100MB output cap')
        write_json(directory / 'progress.json', dict(phase=phase, elapsed_seconds=time.monotonic() - began))
    try:
        check('validate_inputs')
        prepared = validate_prepared(request)
        binding = prepared['binding']
        require(balanced.stat_identity(binding['vae_path']) == binding['vae_stat'], 'original VAE stat changed')
        verifier = ArtifactHashes(balanced.VERIFICATION_CACHE, protected_paths=[binding['vae_path']])
        require(verifier(binding['vae_path']) == VAE_SHA256, 'original VAE hash changed')
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one leased visible GPU required')
        torch.cuda.reset_peak_memory_stats()
        from utils.wan_wrapper import WanVAEWrapper
        check('vae_load')
        vae = WanVAEWrapper(pretrained_path=binding['vae_path']).requires_grad_(False).eval().to('cuda', torch.bfloat16)
        def encode(pixels):
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                return vae.encode_to_latent(pixels.to('cuda', torch.bfloat16)).detach().cpu()
        def decode(row):
            require(balanced.stat_identity(row['video_path']) == row['video_stat'], 'raw-video changed before decode')
            return decode_initial(row, timeout=min(120, max(1, int(MAX_SECONDS - 20 - (time.monotonic() - began)))))
        cached = write_reference_cache(prepared, encode=encode, decode=decode, check=check)
        require(torch.cuda.max_memory_reserved() < 23 * 1024**3, 'reference cache exceeds23GiB')
        check('completed')
        result = dict(status='completed', mode=MODE, schema=SCHEMA, stage=STAGE, request=request, **cached,
            vae_model_load_count=1, peak_vram_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(), elapsed_seconds=time.monotonic() - began,
            output_bytes=balanced.native._stored_bytes(ROOT), training_started=False, automatic_next_stage=False,
            old_video_latents_read=False, future_reference_frames_used=False, quality_improved=False)
        _exclusive_json(directory / 'cache-result.json', result)
        return result
    except BaseException as error:
        write_json(directory / 'failure.json', dict(status='failed', error=f'{type(error).__name__}: {error}',
            elapsed_seconds=time.monotonic() - began, partial_outputs_preserved=True, automatic_retry=False))
        raise
    finally:
        del vae
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def validate_completed_cache(purpose=None):
    require(purpose is None or purpose in PURPOSE_COUNTS, 'explicit train/qualification purpose required')
    directory = ROOT / JOB_ID
    request, result, settlement = (read(directory / (name + '.json')) for name in ('request', 'result', 'settlement'))
    validate_request(request, directory)
    validate_prepared(request)
    require(result.get('status') == 'completed' and result.get('mode') == MODE and result.get('schema') == SCHEMA
            and result.get('stage') == STAGE and result.get('request') == request
            and result == read(directory / 'cache-result.json') and result.get('actual_vae_encodes') == 9
            and result.get('decoded_initial_frames') == 10 and result.get('vae_model_load_count') == 1
            and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed',
            'actual completed and settled reference cache required')
    require(set(result.get('indexes', {})) == set(PURPOSE_COUNTS)
            and len(result.get('windows', [])) == 10
            and len({row['sample_id'] for row in result['windows']}) == 10,
            'completed cache must bind both purpose indexes and exactly ten unique windows')
    for name, info in result['indexes'].items():
        require(info['path'] == str(CACHE_ROOT / f'{name}.references.jsonl')
                and sha256(info['receipt']) == info['receipt_sha256'], 'completed reference index binding differs')
        store = ReferenceFeatureStore(info['path'], purpose=name, expected_index_sha256=info['sha256'])
        require({row['sample_id']: row for row in result['windows'] if row['purpose'] == name} == store.rows,
                'completed cache window rows differ from the immutable reference index')
        for sample_id in store.rows:
            store.load(sample_id)
    return result if purpose is None else dict(result, windows=[r for r in result['windows'] if r['purpose'] == purpose])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / JOB_ID)
    parser.add_argument('--annotations', type=Path, default=ANNOTATIONS)
    parser.add_argument('--source-revision', default=SOURCE_REVISION)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(output=args.output, annotations=args.annotations, source_revision=args.source_revision), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
