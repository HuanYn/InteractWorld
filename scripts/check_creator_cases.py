"""Evaluate small CPU rule/contract cases; never call models or generate video."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.creator.planner import plan_request

KEYS = frozenset('WASDIJKL')
FIELDS = {'status', 'explanation', 'action_segments', 'goals', 'edit_scope', 'preserved', 'planner_kind'}
STATUSES = {'ready', 'clarify', 'unsupported'}


def timeline(segments):
    """Independent small contract check, not a call back into planner validation."""
    if not isinstance(segments, list) or not 1 <= len(segments) <= 240:
        raise ValueError('expected 1..240 action segments')
    rows = []
    for segment in segments:
        if not isinstance(segment, dict) or set(segment) != {'frames', 'keys'}:
            raise ValueError('unknown/missing segment fields')
        frames, keys = segment['frames'], segment['keys']
        if type(frames) is not int or not 1 <= frames <= 240:
            raise ValueError('frames must be integers in 1..240')
        if (not isinstance(keys, list) or any(type(key) is not str or key not in KEYS for key in keys)
                or len(set(keys)) != len(keys)):
            raise ValueError('invalid/duplicate control keys')
        if any(a in keys and b in keys for a, b in [('W', 'S'), ('A', 'D'), ('I', 'K'), ('J', 'L')]):
            raise ValueError('opposing simultaneous controls')
        rows.extend([frozenset(keys)] * frames)
        if len(rows) > 240:
            raise ValueError('timeline exceeds 240 frames')
    if len(rows) != 240:
        raise ValueError('timeline must contain exactly 240 frames')
    return rows


def preserved(before, after, keys):
    keys = frozenset(keys)
    return all(a & keys == b & keys for a, b in zip(before, after))


def validate_result(result, previous):
    errors, rows = [], None
    if not isinstance(result, dict) or set(result) != FIELDS:
        return ['result schema differs from contract'], None
    if result['status'] not in STATUSES:
        errors.append('unknown status')
    if result['edit_scope'] not in {'all', 'camera', 'movement'}:
        errors.append('unknown edit_scope')
    if type(result['explanation']) is not str or not result['explanation']:
        errors.append('missing explanation')
    if not isinstance(result['goals'], list) or any(type(x) is not str for x in result['goals']):
        errors.append('invalid goals')
    if type(result['preserved']) is not bool:
        errors.append('preserved must be boolean')
    if result['planner_kind'] not in {'rule_fallback', 'external_proposal_validated'}:
        errors.append('unexpected planner provenance for this CPU-only run')
    if result['status'] == 'ready':
        try:
            rows = timeline(result['action_segments'])
        except ValueError as error:
            errors.append(str(error))
        if rows is not None:
            scoped = previous is not None and result['edit_scope'] in {'camera', 'movement'}
            keep = 'WASD' if result['edit_scope'] == 'camera' else 'IJKL'
            actual_preservation = bool(scoped and preserved(timeline(previous['action_segments']), rows, keep))
            if scoped and not actual_preservation:
                errors.append('controls outside actual edit_scope changed')
            if result['preserved'] != actual_preservation:
                errors.append('preserved flag differs from per-frame verification')
    elif result['action_segments'] != [] or result['preserved'] is not False:
        errors.append('non-ready result must be non-executable and not claim preservation')
    return errors, rows


def check_case(case, previous_plans):
    previous = copy.deepcopy(previous_plans.get(case.get('previous_plan')))
    if 'previous_plan' in case and previous is None:
        raise ValueError(f"unknown previous_plan for {case['id']}")
    expected = case['expected']
    if expected['status'] not in STATUSES or not set(expected['preserve_keys']).issubset(KEYS):
        raise ValueError(f"invalid expected label/keys for {case['id']}")
    expected_rows = timeline(expected['action_segments']) if 'action_segments' in expected else None
    if expected_rows is not None and expected['status'] != 'ready':
        raise ValueError('non-ready expectation cannot include executable actions')
    before, proposal = copy.deepcopy(previous), copy.deepcopy(case.get('proposal'))
    proposal_before = copy.deepcopy(proposal)
    errors, mismatches, preservation_errors = [], [], []
    try:
        result = plan_request(case['text'], previous=previous, proposal=proposal)
        errors, rows = validate_result(result, before)
        if previous != before or proposal != proposal_before:
            errors.append('planner mutated caller inputs')
        if result.get('status') != expected['status']:
            mismatches.append(f"status expected {expected['status']}, got {result.get('status')}")
        if 'edit_scope' in expected and result.get('edit_scope') != expected['edit_scope']:
            mismatches.append(f"edit_scope expected {expected['edit_scope']}, got {result.get('edit_scope')}")
        if expected_rows is not None and rows != expected_rows:
            mismatches.append('timeline does not match expected per-frame controls')
        preservation_status = 'not_applicable'
        if expected['preserve_keys']:
            if before is None:
                raise ValueError('preserve_keys requires a previous_plan')
            if rows is None:
                preservation_status = 'not_executed'
            elif not preserved(timeline(before['action_segments']), rows, expected['preserve_keys']):
                preservation_status = 'failed'
                preservation_errors.append('expected protected controls changed on one or more frames')
            else:
                preservation_status = 'passed'
    except Exception as error:
        result = {'exception': type(error).__name__, 'message': str(error)}
        errors.append(f'{type(error).__name__}: {error}')
        preservation_status = 'not_executed'
    return dict(id=case['id'], category=case['category'], text=case['text'],
                previous_plan=case.get('previous_plan'), expected=expected, actual=result,
                contract_errors=errors, expectation_errors=mismatches,
                preservation=preservation_status, preservation_errors=preservation_errors,
                passed=not (errors or mismatches or preservation_errors))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=ROOT / 'examples/creator_instruction_cases.json')
    parser.add_argument('--output', type=Path, help='Optional full JSON report; refuses to overwrite an existing report.')
    args = parser.parse_args(argv)
    source = args.cases.read_bytes()
    dataset = json.loads(source)
    if dataset.get('schema_version') != 1 or dataset.get('total_frames') != 240 or dataset.get('fps') != 16:
        parser.error('cases must use schema_version=1, total_frames=240, fps=16')
    cases = dataset['cases']
    if not cases or len({case['id'] for case in cases}) != len(cases):
        parser.error('case IDs must be unique and cases must be nonempty')
    for previous in dataset['previous_plans'].values():
        timeline(previous['action_segments'])
    results = [check_case(case, dataset['previous_plans']) for case in cases]
    summary = dict(measurement='rule_and_contract_cases_only', models_invoked=False, video_generated=False,
                   cases_sha256=hashlib.sha256(source).hexdigest(), cases=len(results),
                   contract_pass=sum(not r['contract_errors'] for r in results),
                   expectation_match=sum(not r['expectation_errors'] and not r['contract_errors'] for r in results),
                   preservation_checked=sum(r['preservation'] in {'passed', 'failed'} for r in results),
                   preservation_pass=sum(r['preservation'] == 'passed' for r in results),
                   passed=sum(r['passed'] for r in results), failed=sum(not r['passed'] for r in results),
                   actual_status_counts=dict(Counter(r['actual'].get('status', 'exception') for r in results)))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(dict(summary=summary, cases=results), stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('\n| case | expected | actual | contract | preservation | result |')
    print('| --- | --- | --- | --- | --- | --- |')
    for result in results:
        actual = result['actual'].get('status', 'exception')
        contract = 'FAIL' if result['contract_errors'] else 'pass'
        outcome = 'pass' if result['passed'] else 'FAIL'
        print(f"| {result['id']} | {result['expected']['status']} | {actual} | {contract} | {result['preservation']} | {outcome} |")
    for result in results:
        if not result['passed']:
            detail = result['contract_errors'] + result['expectation_errors'] + result['preservation_errors']
            print(f"\n{result['id']}: {'; '.join(detail)}")
    return 1 if summary['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
