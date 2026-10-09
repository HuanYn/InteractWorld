"""Read-only frame lineage gate, not a visual/spatial continuity evaluator.

Only the two declared frame files are hashed. Equal bytes establish input
lineage, not a seamless generated transition, camera pose or return to a door.
"""
from collections.abc import Mapping
import hashlib
from pathlib import Path
import re


CONTRACT_ID = 'shot_input_lineage_v1'
RELATIONS = ('strict_continuation', 'intentional_cut', 'route_boundary', 'creative_semantic_cut')
MAX_FRAME_BYTES = 16 * 1024 * 1024


def _frame_evidence(record, prefix):
    """Inspect exactly one bounded file; never create/copy/change source assets."""
    declared = record.get(prefix + '_sha256')
    path = record.get(prefix + '_path')
    evidence = {'declared_path': str(path) if isinstance(path, (str, Path)) else None,
                'declared_sha256': declared, 'actual_sha256': None,
                'file_verified': False, 'error': None}
    if not isinstance(declared, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', declared):
        evidence['error'] = prefix + '_sha256_missing_or_invalid'
        return evidence
    if not isinstance(path, (str, Path)) or not str(path).strip():
        evidence['error'] = prefix + '_path_missing_or_invalid'
        return evidence
    try:
        source = Path(path).resolve(strict=True)
        if not source.is_file() or not 0 < source.stat().st_size <= MAX_FRAME_BYTES:
            evidence['error'] = prefix + '_file_empty_or_over_bound'
            return evidence
        size = source.stat().st_size
        digest = hashlib.sha256()
        read_bytes = 0
        with source.open('rb') as stream:
            while block := stream.read(1024 * 1024):
                read_bytes += len(block)
                if read_bytes > MAX_FRAME_BYTES:
                    evidence['error'] = prefix + '_file_grew_over_bound'
                    return evidence
                digest.update(block)
        actual = digest.hexdigest()
        evidence.update(resolved_path=str(source), actual_sha256=actual, bytes=read_bytes)
        if read_bytes != size or source.stat().st_size != size:
            evidence['error'] = prefix + '_file_changed_during_read'
        elif actual != declared.lower():
            evidence['error'] = prefix + '_declared_sha256_mismatch'
        else:
            evidence['file_verified'] = True
    except (OSError, RuntimeError, ValueError) as exc:
        evidence['error'] = prefix + '_file_unavailable:' + type(exc).__name__
    return evidence


def validate_shot_transition(previous, next_source, *, relation=None, reason=None):
    """Validate source lineage without approving pixels, navigation or execution.

Flat records:
      previous: clip_id (or clipid), actual_last_path, actual_last_sha256
      next: clip_id, next_initial_path, next_initial_sha256, initial_source
    ``relation_category``/``reason`` may be stored on next_source or passed as
    keyword arguments. Strict input provenance must be previous_actual_last.
    Route-boundary records additionally need distinct, explicit route_id values.
    """
    result = {
        'contract_id': CONTRACT_ID, 'relation_category': relation,
        'previous_clip_id': None, 'next_clip_id': None, 'reason': reason,
        'status': 'rejected', 'input_lineage_match': None,
        'strict_continuation_input_eligible': False,
        'visual_continuity_verified': False, 'spatial_return_supported': False,
        'semantic_cut_correctness_verified': False,
        'source_files_modified': False, 'model_calls': 0, 'gpu_seconds': 0,
        'world_commits': 0, 'errors': [],
        'scope': 'Input frame lineage only; equal SHA is not proof of seamless video or physical return.',
    }
    if not isinstance(previous, Mapping) or not isinstance(next_source, Mapping):
        result['errors'].append('two_source_records_required')
        return result
    relation = relation if relation is not None else next_source.get('relation_category')
    # Both spellings enforce the same strict contract, never a weaker fallback.
    if relation == 'continuation':
        relation = 'strict_continuation'
    reason = reason if reason is not None else next_source.get('reason')
    result.update(relation_category=relation, reason=reason,
        previous_clip_id=previous.get('clip_id', previous.get('clipid')),
        next_clip_id=next_source.get('clip_id', next_source.get('clipid')),
        next_initial_source=next_source.get('initial_source'))
    if relation not in RELATIONS:
        result['errors'].append('unknown_relation_category')
    if not all(isinstance(result[key], str) and result[key].strip()
               for key in ('previous_clip_id', 'next_clip_id')):
        result['errors'].append('clip_ids_required')
    before = _frame_evidence(previous, 'actual_last')
    after = _frame_evidence(next_source, 'next_initial')
    result.update(previous_last_evidence=before, next_initial_evidence=after)
    result['errors'].extend(row['error'] for row in (before, after) if row['error'])
    if before['file_verified'] and after['file_verified']:
        result['input_lineage_match'] = before['actual_sha256'] == after['actual_sha256']
    if result['errors']:
        return result
    if relation == 'strict_continuation':
        # A supplied future goal or a convenient opening is not the actual last
        # frame, even if a caller gives the same declared hash to both records.
        if next_source.get('initial_source') != 'previous_actual_last':
            result['errors'].append('strict_continuation_requires_previous_actual_last_not_goal_or_opening')
        if result['input_lineage_match'] is not True:
            result['errors'].append('strict_continuation_initial_does_not_match_actual_last')
        if not result['errors']:
            result.update(status='input_lineage_accepted_visual_unverified',
                          strict_continuation_input_eligible=True)
    elif relation in ('intentional_cut', 'creative_semantic_cut'):
        if not isinstance(reason, str) or not reason.strip():
            result['errors'].append('explicit_cut_reason_required')
        elif relation == 'intentional_cut':
            result['status'] = 'declared_intentional_cut_not_continuity_verified'
        else:
            result['status'] = 'creative_semantic_cut_needs_review_not_assumed_correct'
    elif relation == 'route_boundary':
        old_route, new_route = previous.get('route_id'), next_source.get('route_id')
        result.update(previous_route_id=old_route, next_route_id=new_route)
        if (not isinstance(old_route, str) or not old_route.strip()
                or not isinstance(new_route, str) or not new_route.strip() or old_route == new_route):
            result['errors'].append('route_boundary_requires_distinct_explicit_route_ids')
        else:
            result['status'] = 'declared_route_boundary_independent_opening_not_continuation'
    return result
