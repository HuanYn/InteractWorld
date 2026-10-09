"""One leased41-image VAE cache, reusing immutable old6train+4qualification.

Preparation and validation touch only metadata and small original PNG assets.
There is no video decode, future-frame selection, diffusion/text model, launch,
training, weight hashing fallback, or old-reference re-encoding in this module.
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
from training.expanded_reference_dataset import (STAGE, PAYLOAD_KIND, RECEIPT_SCHEMA,
    PURPOSE_COUNTS, VALID_COUNTS, LEGACY_INDEX_SHA256, ExpandedReferenceStore, check_index_binding)
from training.reference_crops import SOURCE_KIND, crop_reference
from training.reference_dataset import REFERENCE_SHAPE, ReferenceFeatureStore, reference_sample_id, tensor_sha256
from training.studio import balanced_turn_cache as balanced
from training.studio import teacher_reference_cache as legacy

MODE = 'expanded_reference_cache'
REVISION = STAGE
SOURCE_REVISION = 'creator-expanded-reference-v1'
SCHEMA = 'expanded_reference_cache_result_v1'
DATA_ROOT = balanced.DATA_ROOT
ROOT = balanced.BASE / STAGE
JOB_ID = 'cer-cache54-v1'
CACHE_ROOT = ROOT / 'data/reference'
ANNOTATIONS = ROOT / 'reference-boxes54-reviewed.json'
INITIAL_RESULT = ROOT / 'initial-rgb54-v1/result.json'
SOURCE_ROOT = balanced.BASE / 'teacher-expanded-data-v1'
ANNOTATION_SHA256 = '89d28efbffa453ed3404323f5e8c83d216c424ac4f93b8e6aef809002211dd2d'
INITIAL_RESULT_SHA256 = '2db9c0015a9995c8d93c034755569b7927f2b963e9d7a181aea6f9d9b09086c1'
SOURCE_PINS = {
    'candidate-manifest64.json': 'bee08b8ff64f2f20b1debdbc9ed075897890aff3a28559b5b1a507899b0d096e',
    'qualified-subset.json': 'd0db64665a88fce15f7471158825b43c882ec07b7b100e6533b4c9ab3a41678a',
    'visual-review.json': '56cb40775edcde58b34965fe9175c01feb4c3aa9bb72b598b2337fe988b8e4f2',
    'preview/source-preflight.json': '8b3b4dac88c5ce6dc3aefad923dfadc3dfe7af8fe75e2adc57a80846f8a72944',
}
MAX_SECONDS, CLEANUP_SECONDS, OUTPUT_CAP = 900, 60, 100_000_000
VAE_SHA256 = balanced.VAE_SHA256
read, _exclusive_json, _exclusive_jsonl = legacy.read, legacy._exclusive_json, legacy._exclusive_jsonl


def _require_data_root():
    require(DATA_ROOT.is_absolute() and DATA_ROOT.as_posix().startswith('/data/'), 'remote /data filesystem required')
    contained(ROOT, DATA_ROOT)


def legacy_lineage():
    """Validate settled old cache records and fixed indexes without reading tensors."""
    directory = legacy.ROOT / legacy.JOB_ID
    request, result, settlement = (read(directory / (name + '.json')) for name in ('request', 'result', 'settlement'))
    legacy.validate_request(request, directory)
    prepared_path = Path(request['prepared_path'])
    require(sha256(prepared_path) == request['prepared_sha256'], 'old reference preparation pin changed')
    prepared = read(prepared_path)
    require(result.get('status') == 'completed' and result.get('mode') == legacy.MODE
            and result.get('schema') == legacy.SCHEMA and result.get('stage') == legacy.STAGE
            and result.get('request') == request and result == read(directory / 'cache-result.json')
            and result.get('actual_vae_encodes') == 9 and result.get('decoded_initial_frames') == 10
            and result.get('vae_model_load_count') == 1 and settlement.get('status') == 'settled'
            and settlement.get('outcome') == 'completed', 'original reference cache must remain completed and settled')
    require(set(result['indexes']) == set(PURPOSE_COUNTS), 'old purpose indexes missing')
    rows = []
    for purpose, info in result['indexes'].items():
        check_index_binding(info)
        require(info['path'] == str(legacy.CACHE_ROOT / f'{purpose}.references.jsonl')
                and info['sha256'] == LEGACY_INDEX_SHA256[purpose], 'original fixed reference index differs')
        store = ReferenceFeatureStore(info['path'], purpose=purpose, expected_index_sha256=info['sha256'])
        require(store.binding['vae_sha256'] == VAE_SHA256
                and store.rows == {r['sample_id']: r for r in result['windows'] if r['purpose'] == purpose},
                'old reference result/index rows differ')
        rows.extend(copy.deepcopy(list(store.rows.values())))
    files = {str(path): sha256(path) for path in [prepared_path,
        *(directory / (name + '.json') for name in ('request', 'result', 'settlement', 'cache-result'))]}
    files.update(prepared['binding']['files'])
    for path, digest in files.items():
        require(sha256(path) == digest, 'old source record changed')
    old = prepared['binding']
    require(old['vae_sha256'] == VAE_SHA256, 'same original VAE required')
    return dict(rows=rows, indexes=result['indexes'], files=files,
                vae_path=old['vae_path'], vae_stat=old['vae_stat'])


def bind_rows(document, initial, selected_rows, old_rows):
    """Pure exact54+old10 join; validity never changes the60-window selection."""
    import torch
    require(document.get('schema') == 'expanded_reference_initial_annotations54_v1'
            and document.get('stage') == STAGE and document.get('not_user_annotation') is True
            and document.get('detector_used') is False and document.get('future_frames_used') is False
            and document.get('valid_count') == 41 and document.get('invalid_count') == 13
            and document.get('coordinate_system') == dict(order='left,top,right,bottom', right_bottom='exclusive', width=832, height=480),
            'final41/13 original-initial-frame assistant annotation contract required')
    require(initial.get('schema') == 'expanded_reference_exact_initial_rgb54_v1'
            and initial.get('status') == 'completed_awaiting_initial_frame_crop_review'
            and initial.get('cpu_only') is True and initial.get('gpu_seconds') == 0
            and initial.get('model_loaded') is False and initial.get('vae_encoded') is False,
            'completed CPU exact initial extraction required')
    selected = {reference_sample_id(r['episode_id'], r['source_start']): r for r in selected_rows}
    new = {r['sample_id']: r for r in document['rows']}
    frames = {r['sample_id']: r for r in initial['frames']}
    old = {r['sample_id']: r for r in old_rows}
    require(len(selected_rows) == len(selected) == 60 and len(document['rows']) == len(new) == len(initial['frames']) == len(frames) == 54
            and len(old_rows) == len(old) == 10 and set(new) == set(frames) and not set(new) & set(old),
            'exact unique60 selected,54 new,10 legacy references required')
    require(set(new) == {k for k, r in selected.items() if r['cache_origin'] == 'native97_pool256'}
            and {k for k, r in old.items() if r['purpose'] == 'train'}
                == {k for k, r in selected.items() if r['cache_origin'] == 'balanced_train6'},
            'reference IDs must exactly cover54 native +6 old train windows')
    dummy = torch.zeros(3, 480, 832, dtype=torch.uint8)
    output = []
    for sample, row in new.items():
        frame, source = frames[sample], selected[sample]
        require(row['sample_id'] == reference_sample_id(row['episode_id'], row['source_start'])
                and all(row[k] == frame[k] == source[k] for k in ('episode_id', 'window_index', 'source_start', 'target'))
                and row['purpose'] == 'train' and frame['cache_origin'] == source['cache_origin'] == 'native97_pool256'
                and row['source_video'] == frame['source_video'] == source['video_path']
                and row['source_png'] == frame['png'] and row['png_sha256'] == frame['png_sha256']
                and row['rgb_sha256'] == frame['rgb_sha256'] and row['source_video_stat'] == frame['source_video_stat']
                and row['source_start_seconds'] == frame['source_start_seconds']
                == (row['source_start'] * 1_000_000 // 30) / 1_000_000,
                'annotation/source/frame/full window identity differs')
        require(frame['source_kind'] == SOURCE_KIND and frame['output_shape'] == [3, 480, 832]
                and frame['decoded_output_frames'] == 1 and frame['future_rgb_retained'] is False
                and frame['preview_pixels_used'] is False and frame['latent_pixels_used'] is False
                and frame['image_overlay'] is False and frame['source_frame_id'] == row['source_start'] + 1
                and frame['seek_policy'] == 'round(annotation_timestamp*30)_then_floor_microseconds;equals_exact_source_start'
                and frame['spatial_policy'] == 'original_native97_fps30_scale832x480_increase_lanczos_center_crop_rgb24',
                'original exact single-frame seek/resize provenance differs')
        crop_reference(dummy, row, sample_id=sample, source_start=row['source_start'], source_kind=SOURCE_KIND)
        output.append(dict(copy.deepcopy(row), cache_source='new54', cache_origin='native97_pool256',
            original_split='train', source_shape=frame['source_shape'], seek_policy=frame['seek_policy'],
            spatial_policy=frame['spatial_policy']))
    require(sum(r['valid'] for r in output) == 41, 'actual new validity count differs')
    for sample, row in old.items():
        alias = dict(copy.deepcopy(row), cache_source='legacy10')
        if row['purpose'] == 'train':
            source = selected[sample]
            require(row['target'] == source['target'] and row['valid'] is True, 'old anchor target/validity differs')
            alias.update(cache_origin=source['cache_origin'], window_index=source['window_index'])
        output.append(alias)
    require({r['episode_id'] for r in output if r['purpose'] == 'train'}.isdisjoint(
            r['episode_id'] for r in output if r['purpose'] == 'qualification'), 'qualification episode leaked into train')
    return sorted(output, key=lambda r: (r['purpose'] != 'train', r['sample_id']))


def source_binding(annotations=ANNOTATIONS):
    from training.data.exact_cached_dataset import ExactWindowSelection
    annotations = contained(annotations, ROOT)
    require(annotations == ANNOTATIONS.resolve() and sha256(annotations) == ANNOTATION_SHA256
            and sha256(INITIAL_RESULT) == INITIAL_RESULT_SHA256, 'frozen final annotations/extraction pins differ')
    initial = read(INITIAL_RESULT)
    files = {str(annotations): ANNOTATION_SHA256, str(INITIAL_RESULT): INITIAL_RESULT_SHA256}
    require(set(initial['bindings']) == set(SOURCE_PINS), 'exact extraction source bindings required')
    for name, digest in SOURCE_PINS.items():
        path = SOURCE_ROOT / name
        require(initial['bindings'][name] == dict(path=str(path), sha256=digest) and sha256(path) == digest,
                'qualified expanded60 source record changed')
        files[str(path)] = digest
    subset = read(SOURCE_ROOT / 'qualified-subset.json')
    require(subset['candidate_manifest_sha256'] == SOURCE_PINS['candidate-manifest64.json']
            and subset['review_sha256'] == SOURCE_PINS['visual-review.json'], 'qualified subset review lineage differs')
    selected = ExactWindowSelection(SOURCE_ROOT / 'candidate-manifest64.json', subset=subset['selected'])
    old = legacy_lineage()
    rows = bind_rows(read(annotations), initial, selected.rows, old['rows'])
    files.update(old['files'])
    return dict(rows=rows, lineage=dict(files=files, annotation_sha256=ANNOTATION_SHA256,
        initial_result_sha256=INITIAL_RESULT_SHA256, selection_sha256=selected.contract['selection_sha256'],
        legacy_indexes=old['indexes'], vae_sha256=VAE_SHA256),
        vae_path=str(contained(old['vae_path'], DATA_ROOT)), vae_stat=old['vae_stat'])


def request_fields():
    return dict(mode=MODE, experiment_revision=REVISION, source_revision=SOURCE_REVISION, job_id=JOB_ID,
        max_seconds=MAX_SECONDS, cleanup_seconds=CLEANUP_SECONDS, output_cap_bytes=OUTPUT_CAP,
        train_windows=60, qualification_windows=4, new_reference_images=41, new_invalid_slots=13,
        reused_reference_slots=10, old_reference_reencodes=0,
        prepared_path=str(ROOT / 'data/prepare.json'), cache_root=str(CACHE_ROOT),
        automatic_training=False, reuse_completed=False)


def validate_request(request, directory=None):
    fixed = request_fields()
    require(set(request) == set(fixed) | {'prepared_sha256'}
            and all(request.get(k) == v and type(request.get(k)) is type(v) for k, v in fixed.items())
            and isinstance(request['prepared_sha256'], str) and len(request['prepared_sha256']) == 64,
            'fixed bounded expanded reference cache request required')
    if directory is not None:
        require(Path(directory).resolve() == (ROOT / JOB_ID).resolve(), 'unique new cache job directory required')


def prepare(*, output=None, annotations=ANNOTATIONS, source_revision=SOURCE_REVISION, max_seconds=MAX_SECONDS):
    _require_data_root()
    output = contained(output or ROOT / JOB_ID, ROOT)
    require(output == (ROOT / JOB_ID).resolve() and not output.exists() and not (ROOT / 'data').exists(),
            'fresh new cache data/job directory required; existing CPU outputs stay untouched')
    require(source_revision == SOURCE_REVISION and max_seconds == MAX_SECONDS, 'fixed source revision and900 seconds required')
    binding = source_binding(annotations)
    manifests = {}
    for purpose, count in PURPOSE_COUNTS.items():
        path = ROOT / f'data/manifests/{purpose}.reference.jsonl'
        _exclusive_jsonl(path, [r for r in binding['rows'] if r['purpose'] == purpose])
        manifests[purpose] = dict(path=str(path), sha256=sha256(path), windows=count)
    prepared = dict(schema=SCHEMA, stage=STAGE, status='cpu_prepared', binding=binding, manifests=manifests,
                    model_weights_loaded=False, reference_images_encoded=0)
    _exclusive_json(ROOT / 'data/prepare.json', prepared)
    request = dict(request_fields(), prepared_sha256=sha256(ROOT / 'data/prepare.json'))
    validate_request(request, output)
    _exclusive_json(output / 'request.json', request)
    return request


def validate_prepared(request):
    require(sha256(request['prepared_path']) == request['prepared_sha256'], 'prepared expanded reference records changed')
    prepared = read(request['prepared_path'])
    require(prepared.get('schema') == SCHEMA and prepared.get('stage') == STAGE
            and prepared.get('status') == 'cpu_prepared', 'expanded preparation schema differs')
    for path, digest in prepared['binding']['lineage']['files'].items():
        require(sha256(path) == digest, 'bound source record changed')
    for purpose in PURPOSE_COUNTS:
        info = prepared['manifests'][purpose]
        require(info['path'] == str(ROOT / f'data/manifests/{purpose}.reference.jsonl')
                and sha256(info['path']) == info['sha256'], 'expanded reference manifest changed')
    return prepared


def read_initial_png(row):
    """Only lossless retained source RGB, not a contact/JPEG or latent decode."""
    import numpy as np
    import torch
    from PIL import Image
    path = contained(row['source_png'], ROOT / 'initial-rgb54-v1/frames')
    require(path.suffix == '.png' and sha256(path) == row['png_sha256'], 'original PNG pin differs')
    with Image.open(path) as image:
        require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (832, 480), 'lossless original832x480 RGB PNG required')
        array = np.array(image, dtype=np.uint8, copy=True)
    require(hashlib.sha256(array.tobytes()).hexdigest() == row['rgb_sha256'], 'original HWC pixel hash differs')
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def verified_vae_stat(binding):
    """Require a current recorded full SHA; never hash/load a weight to verify it."""
    path = contained(binding['vae_path'], DATA_ROOT)
    require(balanced.stat_identity(path) == binding['vae_stat'], 'original VAE stat changed')
    verifier = ArtifactHashes(balanced.VERIFICATION_CACHE, protected_paths=[path])
    before = verifier._identity(path)
    require(verifier.values.get(path) == (before, VAE_SHA256)
            and verifier._identity(path) == before, 'no current trusted VAE SHA/stat cache hit; stop without full weight hash')


def write_reference_cache(prepared, *, encode, load_rgb=read_initial_png, check=lambda phase: None):
    """Injectable CPU contract: only41 valid new images invoke the actual VAE."""
    import torch
    require(not CACHE_ROOT.exists(), 'new reference cache exists; no overwrite/retry')
    binding, cached, observed = prepared['binding'], [], []
    for source in binding['rows']:
        row = copy.deepcopy(source)
        if row['cache_source'] == 'legacy10':
            cached.append(row)
            continue
        check('read_' + row['sample_id'])
        raw = load_rgb(row)
        crop = crop_reference(raw, row, sample_id=row['sample_id'], source_start=row['source_start'], source_kind=SOURCE_KIND)
        crop_digest = None
        if crop.valid:
            check('encode_' + row['sample_id'])
            actual = encode(crop.crop.float().div(127.5).sub(1).unsqueeze(0).unsqueeze(2))
            require(torch.is_tensor(actual) and actual.device.type == 'cpu' and actual.is_floating_point()
                    and tuple(actual.shape) == (1, 1, 48, 16, 16) and bool(torch.isfinite(actual).all()),
                    'actual VAE must confirm finite CPU[1,1,48,16,16]')
            observed.append(list(actual.shape))
            latent = actual.permute(0, 2, 1, 3, 4).to(torch.bfloat16).contiguous()
            crop_digest = hashlib.sha256(crop.crop.contiguous().numpy().tobytes()).hexdigest()
        else:
            latent = torch.zeros(REFERENCE_SHAPE, dtype=torch.bfloat16)
        payload = dict(kind=PAYLOAD_KIND, stage=STAGE, sample_id=row['sample_id'], episode_id=row['episode_id'],
            source_start=row['source_start'], purpose='train', ref_latents=latent,
            ref_mask=torch.tensor([crop.valid], dtype=torch.bool), annotation_sha256=binding['lineage']['annotation_sha256'],
            vae_sha256=VAE_SHA256, raw_rgb_sha256=row['rgb_sha256'], png_sha256=row['png_sha256'], crop_rgb_sha256=crop_digest)
        path = CACHE_ROOT / 'tensors' / (row['sample_id'] + '.pt')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            torch.save(payload, stream)
        cached.append(dict(row, payload_path=str(path), payload_sha256=sha256(path),
            raw_rgb_sha256=row['rgb_sha256'], crop_rgb_sha256=crop_digest, latent_sha256_float32=tensor_sha256(latent)))
        check('saved_' + row['sample_id'])
    require(len(cached) == 64 and len(observed) == 41, 'exact64 slots and41 new encodes required')
    indexes = {}
    for purpose, count in PURPOSE_COUNTS.items():
        rows = [r for r in cached if r['purpose'] == purpose]
        require(len(rows) == count and sum(r['valid'] for r in rows) == VALID_COUNTS[purpose], 'purpose/valid count differs')
        manifest = prepared['manifests'][purpose]
        index = CACHE_ROOT / f'{purpose}.references.jsonl'
        _exclusive_jsonl(index, rows)
        receipt_path = Path(str(index) + '.receipt.json')
        receipt = dict(schema=RECEIPT_SCHEMA, stage=STAGE, status='completed', purpose=purpose,
            index=str(index), index_sha256=sha256(index), manifest=manifest['path'], manifest_sha256=manifest['sha256'],
            windows=count, valid_count=VALID_COUNTS[purpose], reference_shape=list(REFERENCE_SHAPE),
            observed_vae_output_shape=observed[0], source_kind=SOURCE_KIND, lineage=binding['lineage'],
            actual_new_vae_encodes=41 if purpose == 'train' else 0, old_reference_reencodes=0,
            future_reference_frames_used=False, preview_jpeg_input=False, old_video_latents_read=False)
        _exclusive_json(receipt_path, receipt)
        indexes[purpose] = dict(path=str(index), sha256=sha256(index), receipt=str(receipt_path),
            receipt_sha256=sha256(receipt_path), manifest=manifest['path'], manifest_sha256=manifest['sha256'], windows=count)
    return dict(indexes=indexes, windows=cached, actual_vae_encodes=41, initial_pngs_read=54,
        reused_reference_slots=10, old_reference_reencodes=0, observed_vae_output_shape=observed[0], reference_shape=list(REFERENCE_SHAPE))


def run(request, directory, lease):
    import torch
    from training.demo.backend import validate_active_authority, validate_lease
    _require_data_root()
    directory = contained(directory, ROOT)
    validate_request(request, directory)
    require(read(directory / 'request.json') == request and lease['gpu_index'] == 2, 'unchanged request and leased protocol GPU2 required')
    validate_lease(lease, max_seconds=MAX_SECONDS, job_id=JOB_ID, request_sha256=sha256(directory / 'request.json'))
    require(not CACHE_ROOT.exists() and not (directory / 'progress.json').exists(), 'no cache overwrite or retry')
    began, vae = time.monotonic(), None
    def check(phase):
        validate_active_authority(lease)
        require(time.monotonic() - began < MAX_SECONDS - 20, 'cache finite completion margin reached')
        require(balanced.native._stored_bytes(ROOT / 'data') < OUTPUT_CAP - 5_000_000, 'new cache100MB cap reached')
        write_json(directory / 'progress.json', dict(phase=phase, elapsed_seconds=time.monotonic() - began))
    try:
        check('validate_inputs')
        prepared = validate_prepared(request)
        verified_vae_stat(prepared['binding'])
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one leased visible GPU required')
        torch.cuda.reset_peak_memory_stats()
        from utils.wan_wrapper import WanVAEWrapper
        check('vae_load')
        vae = WanVAEWrapper(pretrained_path=prepared['binding']['vae_path']).requires_grad_(False).eval().to('cuda', torch.bfloat16)
        def encode(pixels):
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                return vae.encode_to_latent(pixels.to('cuda', torch.bfloat16)).detach().cpu()
        cached = write_reference_cache(prepared, encode=encode, check=check)
        require(torch.cuda.max_memory_reserved() < 23 * 1024**3, 'cache exceeds23GiB')
        check('completed')
        result = dict(status='completed', mode=MODE, schema=SCHEMA, stage=STAGE, request=request, **cached,
            vae_model_load_count=1, peak_vram_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(), elapsed_seconds=time.monotonic() - began,
            output_bytes=balanced.native._stored_bytes(ROOT / 'data'), training_started=False, automatic_next_stage=False,
            future_reference_frames_used=False, old_video_latents_read=False, quality_improved=False)
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
            and result == read(directory / 'cache-result.json') and result.get('actual_vae_encodes') == 41
            and result.get('initial_pngs_read') == 54 and result.get('reused_reference_slots') == 10
            and result.get('old_reference_reencodes') == 0 and result.get('vae_model_load_count') == 1
            and settlement.get('status') == 'settled' and settlement.get('outcome') == 'completed',
            'actual completed and settled expanded cache required')
    require(set(result['indexes']) == set(PURPOSE_COUNTS) and len(result['windows']) == 64
            and len({r['sample_id'] for r in result['windows']}) == 64, 'exact64 distinct purpose references required')
    for role, info in result['indexes'].items():
        require(info['path'] == str(CACHE_ROOT / f'{role}.references.jsonl'), 'expanded reference index location differs')
        store = ExpandedReferenceStore(info, purpose=role)
        require(store.rows == {r['sample_id']: r for r in result['windows'] if r['purpose'] == role},
                'expanded cache result/index rows differ')
    return result if purpose is None else dict(result, windows=[r for r in result['windows'] if r['purpose'] == purpose])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / JOB_ID)
    parser.add_argument('--source-revision', default=SOURCE_REVISION)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(output=args.output, source_revision=args.source_revision), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
