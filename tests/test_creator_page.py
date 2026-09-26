"""CPU-only tests of the browser's input comparison and evidence attribution.

The DOM/fetch stubs below only isolate pure JS helpers. They neither run a model
nor create video evidence. Full playback is checked separately in a browser.
"""
import json
import shutil
import subprocess

import pytest

from training.creator.page import JS


def browser_helper(expression):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is required to check the embedded browser helpers')
    harness = r"""
const vm = require('node:vm');
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => input += chunk);
process.stdin.on('end', () => {
  const payload = JSON.parse(input);
  const context = vm.createContext({
    document: {getElementById: () => ({}), addEventListener: () => {}},
    fetch: () => new Promise(() => {}),
  });
  vm.runInContext(payload.source, context, {timeout: 1000});
  process.stdout.write(JSON.stringify(vm.runInContext(payload.expression, context)));
});
"""
    result = subprocess.run(
        [node, '-e', harness],
        input=json.dumps(dict(source=JS, expression=expression)),
        text=True, encoding='utf-8', capture_output=True, timeout=10, check=True,
    )
    return json.loads(result.stdout)


def plan(*segments):
    return dict(status='ready', action_segments=[
        dict(frames=frames, keys=keys) for frames, keys in segments
    ])


def test_frame_exact_diff_preserves_movement_and_detects_shifted_timing():
    original = plan((120, ['W']), (120, ['W', 'I']))
    shorter = plan((180, ['W']), (60, ['W', 'I']))
    shifted = plan((120, ['W', 'I']), (120, ['W']))
    comparisons = browser_helper(
        f'[{json.dumps(shorter)},{json.dumps(shifted)}]'
        f'.map(after => compareInputs({json.dumps(original)}, after))'
    )
    short = comparisons[0]
    assert short['changed'] == 60
    movement = next(item for item in short['keys'] if item['key'] == 'W')
    camera = next(item for item in short['keys'] if item['key'] == 'I')
    assert movement == dict(key='W', before=240, after=240, added=0, removed=0)
    assert camera == dict(key='I', before=120, after=60, added=0, removed=60)
    shift = comparisons[1]
    assert shift['changed'] == 240
    camera = next(item for item in shift['keys'] if item['key'] == 'I')
    assert camera == dict(key='I', before=120, after=120, added=120, removed=120)


def test_invalid_plan_has_no_diff_and_review_evidence_keeps_video_identity():
    ready = plan((240, ['W']))
    invalid = [None, dict(status='clarify', action_segments=[]),
               plan((239, ['W'])), plan((240, ['UNKNOWN'])),
               plan((240, ['W', 'W']))]
    assert browser_helper(
        f'{json.dumps(invalid)}.map(before => compareInputs(before, {json.dumps(ready)}))'
    ) == [None] * len(invalid)
    evidence = [dict(video='reference', time_seconds=4, observation='reference only'),
                dict(video='current', time_seconds=6, observation='current only'),
                dict(time_seconds=8, observation='legacy current')]
    assert browser_helper(f'reviewEvidence({json.dumps(evidence)})') == (
        '参考视频 · 4 秒 · reference only\n'
        '当前视频 · 6 秒 · current only\n'
        '当前视频 · 8 秒 · legacy current'
    )


def test_review_records_keep_human_and_model_separate_without_duplicates():
    model = dict(source='model_assessment', verdict='uncertain', evidence='model observation', created_at=1)
    first = dict(source='human', verdict='unsatisfied', evidence='human first',
                 criteria=dict(camera_response='unsatisfied'), created_at=2,
                 previous_assessment=model)
    second = dict(source='human', verdict='satisfied', evidence='human second', created_at=3,
                  previous_assessment=model)
    version = dict(review_history=[model, first], review=second)
    records = browser_helper(f'reviewRecords({json.dumps(version)})')
    assert [item['source'] for item in records] == ['model_assessment', 'human', 'human']
    assert [item['evidence'] for item in records] == ['model observation', 'human first', 'human second']
    assert all('previous_assessment' not in item for item in records)


def test_review_edit_warns_when_viewed_version_is_not_last_ready_baseline():
    versions = [dict(version_id='old', plan=plan((240, ['W']))),
                dict(version_id='ready', plan=plan((240, ['W', 'I']))),
                dict(version_id='clarify', plan=dict(status='clarify'))]
    result = browser_helper(
        'selectedId="test"; sessions=[{session_id:"test",versions:' + json.dumps(versions) + '}];'
        '[reviewEditWarning("old"), reviewEditWarning("ready")]'
    )
    assert '版本 2' in result[0]
    assert result[1] == ''
