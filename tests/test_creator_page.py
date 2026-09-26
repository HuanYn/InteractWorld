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
  const nodes = new Map(), storage = new Map();
  const node = () => ({value:'', textContent:'', dataset:{}, childNodes:[],
    append(...children) { this.childNodes.push(...children); },
    replaceChildren(...children) { this.childNodes = [...children]; },
    setAttribute() {}, focus() {}, scrollIntoView() {},
    querySelector() { return node(); }, querySelectorAll() { return []; },
  });
  const context = vm.createContext({
    document: {getElementById: id => {if (!nodes.has(id)) nodes.set(id, node()); return nodes.get(id);},
      createElement: () => node(), addEventListener: () => {}},
    localStorage: {getItem: key => storage.get(key) || null, setItem: (key, value) => storage.set(key, value)},
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


def version_records():
    return [dict(version_id='old', plan=plan((240, ['W']))),
            dict(version_id='ready', plan=plan((240, ['W', 'I']))),
            dict(version_id='clarify', plan=dict(status='clarify'))]


def test_edit_baseline_defaults_dynamic_but_fixed_selection_survives_refresh():
    result = browser_helper(
        'selectedId="test"; sessions=[{session_id:"test",versions:' + json.dumps(version_records()) + '}];'
        'const observed=[editBaseVersion().version_id];'
        'setEditBase("old"); sessions[0].versions.push({version_id:"new",plan:{status:"ready"}});'
        'observed.push(editBaseVersion().version_id); editBases.clear();'
        'observed.push(editBaseVersion().version_id); setEditBase("");'
        'observed.push(editBaseVersion().version_id); observed'
    )
    assert result == ['ready', 'old', 'old', 'new']


def test_baseline_select_only_ready_versions_and_does_not_leak_between_sessions():
    result = browser_helper(
        'selectedId="test"; sessions=[{session_id:"test",versions:' + json.dumps(version_records()) + '},'
        '{session_id:"other",versions:[{version_id:"other-ready",plan:{status:"ready"}}]}];'
        'setEditBase("old");renderEditBase();'
        'const observed=[$("edit-base-version").childNodes.map(option=>option.value),'
        'setEditBase("clarify"),setEditBase("other-ready"),editBaseVersion().version_id];'
        'selectedId="other";observed.push(editBaseVersion().version_id,editBaseSelection());'
        'selectedId="";observed.push(editBaseVersion());'
        'selectedId="test";observed.push(editBaseVersion().version_id);observed'
    )
    assert result == [['', 'old', 'ready'], False, False, 'old', 'other-ready', '', None, 'old']


def test_plan_payload_pins_displayed_baseline_and_does_not_infer_id_from_text():
    result = browser_helper(
        'selectedId="test"; sessions=[{session_id:"test",versions:' + json.dumps(version_records()) + '}];'
        'config={rule_planner_available:true};$("scene").value="mountain";$("seed").value="42";'
        '$("planner-mode").value="local_model";$("instruction").value="基于版本B，保持前进不变，只在第8到10秒抬头";'
        'const dynamic=planPayload();setEditBase("old");[dynamic,planPayload()]'
    )
    assert [item['base_version_id'] for item in result] == ['ready', 'old']
    assert all(item['planner'] == 'local_model' and item['seed'] == 42 for item in result)
    assert all('版本B' in item['text'] for item in result)


def test_review_edit_selects_its_own_baseline_without_model_or_generation_call():
    result = browser_helper(
        'selectedId="test"; sessions=[{session_id:"test",versions:' + json.dumps(version_records()) + '}];'
        'config={};updateControls=()=>{};let calls=0;api=()=>{calls++;};'
        'useReviewEdit("old","保持前进不变，只在第8到10秒抬头");'
        '[editBaseVersion().version_id,$("edit-base-version").value,$("instruction").value,calls,$("notice").textContent]'
    )
    assert result[:4] == ['old', 'old', '保持前进不变，只在第8到10秒抬头', 0]
    assert '固定为此历史版本' in result[4]


def test_comparison_and_acceptance_do_not_change_edit_baseline():
    result = browser_helper(
        'selectedId="test"; sessions=[{session_id:"test",accepted_version:"ready",versions:' + json.dumps(version_records()) + '}];'
        'setEditBase("old");renderVersions=()=>{};chooseComparison("left","ready");'
        'chooseComparison("right","clarify");[editBaseVersion().version_id,comparison().left,comparison().right]'
    )
    assert result == ['old', 'ready', 'clarify']


def test_planned_version_response_controls_preview_and_survives_polling():
    records = version_records()
    records[0]['base_version_id'] = 'actual-parent'
    records[0]['plan']['edit_patch'] = dict(schema_version=1, fps=16, total_frames=240,
        edits=[dict(op='replace_intervals', key='I', intervals=[dict(start_seconds=8, end_seconds=10)])],
        protected_keys=['W'])
    saved = dict(session_id='test', planned_version_id='old', versions=records)
    result = browser_helper(
        'config={};upsertSession(' + json.dumps(saved) + ');renderPlan();'
        'const walk=node=>[node.textContent,...node.childNodes.flatMap(walk)];'
        'const observed=[plannedVersion().version_id,walk($("plan-details")).join(" ")];'
        'sessions[0].versions.push({version_id:"later",plan:{status:"ready"}});'
        'observed.push(plannedVersion().version_id);plannedVersions.clear();'
        'observed.push(plannedVersion().version_id);observed'
    )
    assert result[0] == result[2] == result[3] == 'old'
    assert 'actual-parent' in result[1]
    assert 'replace_intervals' in result[1]
    assert 'start_seconds' in result[1]
