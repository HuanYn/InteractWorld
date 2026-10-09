"""CPU-only re-selection of two49-frame windows in each of eight retained videos.

Video/annotation payloads are read-only references to one bound original subset.
Only a new manifest, preparation receipt and exact selected-text copy are written.
VAE feature generation remains in rebuild_subset.cache under its existing guard.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import shutil
import time

from training.demo.contracts import require, write_json
from training.studio import rebuild_subset as original


def _read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def _load_original(source):
    source = original._data_path(source)
    prepared = original._json(source / 'prepare.json')
    require(prepared.get('status') == 'completed' and prepared.get('mode') == 'rebuild_subset_prepare'
            and 'payload_reference_root' not in prepared, 'reference must be one original completed subset, not a reference chain')
    manifest, prompts = source / 'manifests/train.jsonl', source / 'text/selected-prompts.pt'
    require(prepared.get('manifest') == str(manifest) and prepared.get('prompts') == str(prompts)
            and original.sha256_file(manifest) == prepared.get('manifest_sha256')
            and original.sha256_file(prompts) == prepared.get('prompts_sha256'), 'original manifest/prompts binding changed')
    rows = _read_rows(manifest)
    ids = [row.get('episode_id') for row in rows]
    require(len(ids) == 10 and len(set(ids)) == 10
            and all(isinstance(identity, str) and re.fullmatch('[0-9a-f]{32}', identity) for identity in ids),
            'reference must contain eight train episodes and two fixed dev text references')
    training = [row for row in rows if row.get('split') == 'train']
    dev = [row for row in rows if row.get('split') == 'dev']
    require(len(training) == 8 and {row['episode_id'] for row in dev} == set(original.DEV_TEXT_IDS)
            and all(row.get('cache_state') == 'text_only_reference_no_video_features' for row in dev),
            'original train/dev membership differs from the eight-video subset')
    require(prepared.get('raw_episodes') == 8 and prepared.get('windows') == 16
            and prepared.get('max_windows_per_episode') == 2
            and set(prepared.get('dev_text_only_ids', [])) == set(original.DEV_TEXT_IDS)
            and set(prepared.get('episodes', {})) == set(ids)
            and all(prepared['episodes'][row['episode_id']].get('split') == row['split'] for row in rows),
            'original preparation episode bindings/counts changed')
    require(all(len(row.get('selected_source_starts', [])) == 2 for row in training),
            'original subset must have exactly two windows per episode')
    return source, prepared, rows


def _validate_payloads(source, prepared, rows):
    """Verify unchanged original payload bindings and actual temporal eligibility."""
    for row in rows:
        if row['split'] != 'train':
            continue
        identity = row['episode_id']
        for field, filename in [('video_path', 'video.mp4'), ('annotations_path', 'annotations.tar')]:
            require(original._data_path(row[field]) == source / 'payload' / identity / filename,
                    'referenced payload path escaped its original subset')
        video_stat = Path(row['video_path']).stat()
        require(video_stat.st_size == row['video_probe']['file_bytes']
                and video_stat.st_mtime_ns == row['video_mtime_ns']
                and original.sha256_file(row['annotations_path']) == prepared['episodes'][identity]['annotations_sha256'],
                'verified original video stat/annotation hash changed')
        require(isinstance(row.get('video_sha256'), str) and re.fullmatch('[0-9a-f]{64}', row['video_sha256']),
                'trusted original video digest is missing')
        sequence = original.parse_action_document(original.read_annotation_bundle(row['annotations_path']).action)
        original.validate_action_video_alignment(sequence, row['video_probe'])
        starts = row.get('selected_source_starts')
        require(isinstance(starts, list) and len(starts) == 2
                and all(type(start) is int and start >= 0 for start in starts) and len(set(starts)) == 2,
                'each episode requires two distinct integer source starts')
        eligible = set(sequence.eligible_resampled_window_starts(49, output_fps=16))
        require(all(start in eligible for start in starts), 'selected source start is not eligible for49 frames at16fps')


def validate_payload_reference(root, prepared, rows):
    """Cache-time gate; no arbitrary external paths or changed episode contracts."""
    root = original._data_path(root)
    source, source_prepared, source_rows = _load_original(prepared['payload_reference_root'])
    require(source != root and not root.is_relative_to(source) and not source.is_relative_to(root),
            'new and original data roots must be disjoint')
    require(prepared.get('mode') == 'reselect_subset_prepare'
            and prepared.get('original_prepare') == str(source / 'prepare.json')
            and original.sha256_file(source / 'prepare.json') == prepared.get('original_prepare_sha256')
            and prepared.get('original_manifest') == source_prepared['manifest']
            and prepared.get('original_manifest_sha256') == source_prepared['manifest_sha256'],
            'original preparation/manifest reference binding changed')
    for key in ('episodes', 'original_encoder', 'original_prompt_cache', 'original_prompt_cache_sha256',
                'source_manifest', 'source_manifest_sha256', 'prompt_policy', 'dev_text_only_ids',
                'raw_episodes', 'windows', 'max_windows_per_episode'):
        require(prepared.get(key) == source_prepared.get(key), f'reference changed original {key}')
    require(prepared.get('prompts_sha256') == source_prepared['prompts_sha256'],
            'reselection changed the actual cached text embeddings')
    require(len(rows) == len(source_rows)
            and [row.get('episode_id') for row in rows] == [row['episode_id'] for row in source_rows],
            'reselection must preserve original episode order and membership')
    for row, before in zip(rows, source_rows):
        if before['split'] == 'train':
            require({key: value for key, value in row.items() if key != 'selected_source_starts'}
                    == {key: value for key, value in before.items() if key != 'selected_source_starts'},
                    'reselection changed payload, split, text or metadata beyond window starts')
        else:
            require(row == before, 'dev text references must remain unchanged')
    require(prepared.get('selected_source_starts') == {
        row['episode_id']: row['selected_source_starts'] for row in rows if row['split'] == 'train'
    }, 'selected-window receipt differs from manifest')
    _validate_payloads(source, source_prepared, rows)
    return source


def prepare(*, source, selection, output):
    began = time.monotonic()
    source, source_prepared, source_rows = _load_original(source)
    root = original._data_path(output)
    require(not root.exists() and source != root and not root.is_relative_to(source)
            and not source.is_relative_to(root), 'output must be a new data root disjoint from the original')
    selection_path = Path(selection).resolve(strict=True)
    selected = original._json(selection_path)
    train_ids = {row['episode_id'] for row in source_rows if row['split'] == 'train'}
    require(isinstance(selected, dict) and set(selected) == train_ids,
            'selection JSON must map exactly the original eight train episode IDs')
    rows = copy.deepcopy(source_rows)
    for row in rows:
        if row['split'] == 'train':
            row['selected_source_starts'] = copy.deepcopy(selected[row['episode_id']])
    prepared = copy.deepcopy(source_prepared)
    prepared.update(mode='reselect_subset_prepare', payload_reference_root=str(source),
        original_prepare=str(source / 'prepare.json'), original_prepare_sha256=original.sha256_file(source / 'prepare.json'),
        original_manifest=source_prepared['manifest'], original_manifest_sha256=source_prepared['manifest_sha256'],
        selection_json=str(selection_path), selection_json_sha256=original.sha256_file(selection_path),
        selected_source_starts=selected, manifest=str(root / 'manifests/train.jsonl'),
        prompts=str(root / 'text/selected-prompts.pt'), downloads=[], download_bytes=0,
        payload_reference_only=True, gpu_launched=False, features_generated=False,
        video_integrity='original recorded SHA retained; unchanged size/mtime checked; video bytes not rehashed',
        note='Same eight train videos and fixed dev text references; only two eligible starts per train episode changed.')
    # Also exercise the cache-time reference contract before publishing any files.
    validate_payload_reference(root, prepared, rows)
    root.mkdir(parents=True, exist_ok=False)
    original._jsonl(root / 'manifests/train.jsonl', rows)
    (root / 'text').mkdir()
    shutil.copyfile(source / 'text/selected-prompts.pt', root / 'text/selected-prompts.pt')
    prepared.update(manifest_sha256=original.sha256_file(root / 'manifests/train.jsonl'),
                    prompts_sha256=original.sha256_file(root / 'text/selected-prompts.pt'),
                    elapsed_seconds=time.monotonic() - began)
    require(prepared['prompts_sha256'] == source_prepared['prompts_sha256'], 'copied prompt bytes changed')
    write_json(root / 'prepare.json', prepared)
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='original verified eight-video subset data root')
    parser.add_argument('--selection', required=True, type=Path, help='JSON episode_id -> two eligible source sequence indices')
    parser.add_argument('--output', required=True, type=Path, help='new data root under the project data disk')
    print(json.dumps(prepare(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
