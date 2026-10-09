"""Opt-in editorial junction gate; never a video/navigation success evaluator.

The legacy v3 builder is deliberately untouched. This module needs actual,
bounded frame files and hash-bound production records for every adjacency.
An authored reason, a matching hash, or an AI approval alone is insufficient.
"""
from collections.abc import Mapping
import copy
import hashlib
import json
from pathlib import Path
import re

from .shot_continuity import validate_shot_transition


SCHEMA = 'interactworld.editorial_junction_gate.v1'
RELATIONS = ('strict_continuation', 'approved_creative_cut', 'independent_route_boundary')
KNOWN_MAIN_RESET_SHAS = frozenset({
    'f072f3402a94538e0600c6cbcc8df41dafa8787bad8649049af5f92a338af34d',
})
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_VIDEO_BYTES = 256 * 1024 * 1024


def _sha_file(path, expected, limit):
    if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('missing_or_invalid_sha256')
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError('missing_file_path')
    source = Path(path).resolve(strict=True)
    size = source.stat().st_size
    if not source.is_file() or not 0 < size <= limit:
        raise ValueError('file_empty_or_over_bound')
    digest = hashlib.sha256()
    count = 0
    with source.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            count += len(block)
            if count > limit:
                raise ValueError('file_grew_over_bound')
            digest.update(block)
    if count != size or source.stat().st_size != size or digest.hexdigest() != expected:
        raise ValueError('actual_file_sha256_mismatch_or_changed')
    return source


def _record(reference):
    if not isinstance(reference, Mapping):
        raise ValueError('hash_bound_source_record_required')
    path = _sha_file(reference.get('path'), reference.get('sha256'), MAX_JSON_BYTES)
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, Mapping):
        raise ValueError('source_record_must_be_an_object')
    return value


def _frame_binding(frame, clip, *, role):
    """Validate known worker/decode fields, never a caller-selected goal field.

    native_rgb_precompression: actual worker last PNG, explicitly before MP4.
    mp4_decoded_rgb: actual decoded source MP4 final PNG and zero-based index.
    generation_input_rgb: exact input SHA from that next clip's worker record.
    Native vs compressed frames are reported separately, never conflated.
    """
    if not isinstance(frame, Mapping):
        raise ValueError(role + '_frame_evidence_missing')
    _sha_file(frame.get('path'), frame.get('sha256'), 16 * 1024 * 1024)
    source = _record(frame.get('source_record'))
    video_sha = source.get('video_sha256', source.get('raw_video_sha256'))
    representation = frame.get('representation')
    if role == 'previous_last' and representation == 'native_rgb_precompression':
        if ('before MP4 compression' not in str(source.get('native_frame_provenance', ''))
                or source.get('last_frame_sha256') != frame['sha256']
                or source.get('actual_generated_frames') != int(clip['actual_media']['nb_frames'])):
            raise ValueError('native_last_requires_actual_worker_precompression_record')
    elif role == 'previous_last' and representation == 'mp4_decoded_rgb':
        video_sha = source.get('source_video_sha256')
        if (source.get('selected_frame_is_actual_mp4_decode') is not True
                or source.get('source_frame_sha256', source.get('decoded_frame_png_sha256')) != frame['sha256']
                or source.get('source_frame_is_precompression_native', False) is not False
                or source.get('selected_frame_index') != int(clip['actual_media']['nb_frames']) - 1):
            raise ValueError('decoded_last_requires_actual_mp4_final_frame_record')
    elif role == 'next_input' and representation == 'generation_input_rgb':
        # Explicit input aliases only. A supplied_goal SHA cannot qualify.
        recorded = [source.get(key) for key in ('initial_sha256', 'input_frame_sha256', 'input_copy_sha256')]
        if frame['sha256'] not in recorded:
            raise ValueError('next_input_requires_actual_generation_input_record_not_goal')
    else:
        raise ValueError(role + '_representation_unknown_or_wrong_role')
    if video_sha != clip['source_sha256']:
        raise ValueError(role + '_production_record_not_bound_to_this_video')
    return {'path': str(frame['path']), 'sha256': frame['sha256'],
            'representation': representation, 'source_record': dict(frame['source_record']),
            'production_record_fields_verified': True,
            'independent_frame_decode_performed': False}


def _spatial_declaration(value):
    if (not isinstance(value, Mapping)
            or any(not isinstance(value.get(key), str) or not value[key].strip()
                   for key in ('from_place', 'to_place', 'relation'))):
        raise ValueError('explicit_from_to_spatial_relation_required')
    if any(value.get(key, False) is not False for key in
           ('continuous_navigation_claimed', 'spatial_return_verified', 'threshold_crossing_claimed')):
        raise ValueError('editorial_spatial_declaration_cannot_claim_navigation')
    return dict(value)


def validate_timeline_junctions(manifest, declarations, *, known_reset_shas=KNOWN_MAIN_RESET_SHAS):
    """Return editorial eligibility only; missing/conflicting evidence fails closed.

    declarations.junctions has exactly one record per consecutive clip pair:
      previous_clip_id, next_clip_id, relation, previous_last, next_input.
    Every frame is {path, sha256, representation, source_record:{path,sha256}}.
    Cuts/boundaries also require reason, source:{path,sha256}, spatial_relation.
    Creative cuts additionally require review:{path,sha256}; that JSON must bind
    both clip IDs, both video hashes, both frame hashes and the exact spatial
    declaration. This remains a recorded editorial judgment, not pixel truth.
    """
    report = dict(schema=SCHEMA, status='rejected', edit_eligible=False, junctions=[], errors=[],
        visual_continuity_verified=False, spatial_return_supported=False,
        visual_success_verified=False, training_eligible=False, continuous_navigation_claimed=False,
        model_calls=0, gpu_seconds=0, world_commits=0, media_rendered=False,
        source_files_modified=False, candidate_manifest=None)
    clips = manifest.get('clips') if isinstance(manifest, Mapping) else None
    if not isinstance(clips, list) or not 2 <= len(clips) <= 64:
        report['errors'].append('bounded_timeline_with_at_least_two_clips_required')
        return report
    ids = [clip.get('id') for clip in clips if isinstance(clip, Mapping)]
    if (len(ids) != len(clips) or any(not isinstance(ident, str) or not ident for ident in ids)
            or len(set(ids)) != len(ids)):
        report['errors'].append('unique_explicit_clip_ids_required')
        return report
    clip_errors = {}
    verified_videos = set()
    for clip in clips:
        try:
            if clip.get('route_id') not in ('A', 'B', 'C') or not clip['id'].startswith(clip['route_id'] + '-'):
                raise ValueError('explicit_ABC_route_and_clip_identity_required')
            frames = int(clip['actual_media']['nb_frames'])
            if frames <= 0 or clip.get('renderable') is not True:
                raise ValueError('actual_completed_source_required')
            key = (str(clip['raw_video']), clip['source_sha256'])
            if key not in verified_videos:
                _sha_file(*key, MAX_VIDEO_BYTES)
                verified_videos.add(key)
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            clip_errors[clip['id']] = str(exc)
    rows = declarations.get('junctions') if isinstance(declarations, Mapping) else None
    if not isinstance(rows, list):
        rows = []
        report['errors'].append('explicit_junction_declarations_missing')
    index = {}
    actual_pairs = set(zip(ids, ids[1:]))
    for row in rows:
        if not isinstance(row, Mapping):
            report['errors'].append('junction_record_must_be_an_object')
            continue
        key = (row.get('previous_clip_id'), row.get('next_clip_id'))
        if not all(isinstance(ident, str) and ident for ident in key):
            report['errors'].append('junction_clip_ids_must_be_strings')
            continue
        if key in index or key not in actual_pairs:
            report['errors'].append('duplicate_or_nonadjacent_junction:' + str(key))
        index[key] = row
    for previous, following in zip(clips, clips[1:]):
        key = (previous['id'], following['id'])
        result = dict(previous_clip_id=key[0], next_clip_id=key[1], status='rejected',
            edit_eligible=False, input_lineage_match=None, visual_continuity_verified=False,
            spatial_return_supported=False, errors=[])
        report['junctions'].append(result)
        try:
            for ident in key:
                if ident in clip_errors:
                    raise ValueError('source_media_unverified:' + ident + ':' + clip_errors[ident])
            row = index.get(key)
            if row is None:
                raise ValueError('junction_declaration_missing')
            relation = row.get('relation')
            result['relation'] = relation
            if relation not in RELATIONS:
                raise ValueError('unknown_relation_no_implicit_cut_approval')
            same_route = previous['route_id'] == following['route_id']
            initial_source = row.get('initial_source')
            candidate_frame = row.get('next_input')
            previous_frame = row.get('previous_last')
            # The original opening's own last RGB is a valid next input. A
            # reset means returning to it from a *different* actual last RGB;
            # both file/production-record bindings are still verified below.
            if same_route and ((isinstance(candidate_frame, Mapping)
                    and candidate_frame.get('sha256') in known_reset_shas
                    and (not isinstance(previous_frame, Mapping)
                         or previous_frame.get('sha256') != candidate_frame.get('sha256')))
                    or initial_source in ('opening_fallback', 'main_view_reset', 'independent_route_opening')):
                raise ValueError('known_main_reset_forbidden_inside_route_even_if_cut_approved')
            if (relation == 'independent_route_boundary') == same_route:
                raise ValueError('route_boundary_only_between_distinct_ABC_routes')
            before = _frame_binding(row.get('previous_last'), previous, role='previous_last')
            after = _frame_binding(candidate_frame, following, role='next_input')
            result.update(previous_last=before, next_input=after)
            lineage = validate_shot_transition(
                dict(clip_id=previous['id'], route_id=previous['route_id'],
                     actual_last_path=before['path'], actual_last_sha256=before['sha256']),
                dict(clip_id=following['id'], route_id=following['route_id'],
                     next_initial_path=after['path'], next_initial_sha256=after['sha256'],
                     initial_source=initial_source),
                relation={'approved_creative_cut': 'intentional_cut',
                          'independent_route_boundary': 'route_boundary'}.get(relation, relation),
                reason=row.get('reason'))
            result['input_lineage_match'] = lineage['input_lineage_match']
            result['input_lineage_gate'] = lineage
            if lineage['errors']:
                raise ValueError('input_lineage_rejected:' + ','.join(lineage['errors']))
            if relation == 'strict_continuation':
                if lineage['strict_continuation_input_eligible'] is not True:
                    raise ValueError('strict_input_not_eligible')
                result.update(status='strict_input_editorially_eligible_visual_unverified', edit_eligible=True)
                continue
            if not isinstance(row.get('reason'), str) or not row['reason'].strip():
                raise ValueError('explicit_cut_or_boundary_reason_required')
            source = _record(row.get('source'))
            source_bindings = dict(previous_clip_id=key[0], next_clip_id=key[1],
                previous_video_sha256=previous['source_sha256'], next_video_sha256=following['source_sha256'])
            if any(source.get(field) != value for field, value in source_bindings.items()):
                raise ValueError('cut_or_boundary_source_must_bind_exact_clip_ids_and_both_videos')
            spatial = _spatial_declaration(row.get('spatial_relation'))
            result.update(reason=row['reason'], source=dict(row['source']), spatial_relation=spatial)
            if relation == 'independent_route_boundary':
                if initial_source != 'independent_route_opening':
                    raise ValueError('route_boundary_requires_explicit_independent_opening')
                result.update(status='independent_route_boundary_editorially_eligible', edit_eligible=True)
                continue
            review = _record(row.get('review'))
            bindings = dict(previous_clip_id=key[0], next_clip_id=key[1],
                previous_video_sha256=previous['source_sha256'], next_video_sha256=following['source_sha256'],
                previous_last_sha256=before['sha256'], next_input_sha256=after['sha256'],
                spatial_relation=spatial)
            if (review.get('schema') != 'interactworld.creative_junction_review.v1'
                    or review.get('decision') != 'approved_creative_edit'
                    or review.get('reviewer_type') not in ('ai', 'human')
                    or not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip()
                    or any(review.get(field) != value for field, value in bindings.items())
                    or review.get('spatial_relation_supported_for_edit') is not True
                    or review.get('visual_continuity_verified') is not False
                    or review.get('spatial_return_supported') is not False):
                raise ValueError('creative_cut_review_missing_exact_endpoints_or_spatial_editorial_judgment')
            result.update(status='approved_creative_cut_editorially_eligible_visual_unverified',
                          edit_eligible=True, review=dict(row['review']), reviewer_type=review['reviewer_type'],
                          review_is_ground_truth=False)
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            result['errors'].append(str(exc))
    report['junction_count'] = len(report['junctions'])
    report['eligible_junction_count'] = sum(row['edit_eligible'] for row in report['junctions'])
    report['edit_eligible'] = not report['errors'] and all(row['edit_eligible'] for row in report['junctions'])
    if report['edit_eligible']:
        report['status'] = 'editorial_junctions_eligible_visual_unverified_not_render_authorization'
        candidate = copy.deepcopy(manifest)
        candidate.update(schema='interactworld.creative_lineage_game_film.v4',
            status='junctions_editorially_eligible_no_capture_no_render',
            allow_capture=False, allow_render=False, renderable=False,
            visual_continuity_verified=False, spatial_return_supported=False,
            visual_success_verified=False, training_eligible=False,
            junction_gate_contract=SCHEMA, junctions=copy.deepcopy(report['junctions']))
        report['candidate_manifest'] = candidate
    return report
