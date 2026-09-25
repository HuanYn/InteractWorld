"""CPU-only checks at the model-output / deterministic-planner boundary."""
import copy
from types import SimpleNamespace

import pytest

from training.creator.model_worker import _render_prompt, validate_plan
from training.creator.planner import expand_segments, plan_request


def _proposal(segments, scope='all'):
    return dict(status='ready', explanation='按输入提议动作。', action_segments=segments,
                goals=['人物持续向前移动', '后半段可见镜头向上变化'], edit_scope=scope)


def test_model_proposals_keep_movement_and_original_camera_start_on_second_turn():
    first = _proposal([{'frames': 120, 'keys': ['W']}, {'frames': 120, 'keys': ['W', 'I']}])
    previous = plan_request('一直前进，后半段抬头', proposal=validate_plan(first))
    assert previous['status'] == 'ready'
    second = _proposal([{'frames': 120, 'keys': ['W']}, {'frames': 60, 'keys': ['W', 'I']},
                        {'frames': 60, 'keys': ['W']}], scope='camera')
    edited = plan_request('保留前进，只缩短抬头', previous, validate_plan(second))
    assert edited['status'] == 'ready' and edited['preserved'] is True
    rows = expand_segments(edited['action_segments'])
    assert all('W' in row for row in rows)
    assert [index for index, row in enumerate(rows) if 'I' in row] == list(range(120, 180))


@pytest.mark.parametrize('field,value', [('explanation', 'x' * 2001), ('goals', ['x' * 201]),
                                       ('goals', ['goal'] * 17)])
def test_model_output_bounds_match_downstream_planner(field, value):
    proposal = _proposal([{'frames': 240, 'keys': ['W']}])
    proposal[field] = value
    with pytest.raises(ValueError):
        validate_plan(proposal)


def test_downstream_still_rejects_proposal_that_drops_protected_movement():
    previous = plan_request('一直前进，后半段抬头')
    proposal = _proposal([{'frames': 120, 'keys': []}, {'frames': 60, 'keys': ['I']},
                          {'frames': 60, 'keys': []}], scope='camera')
    original = copy.deepcopy(previous)
    result = plan_request('保留前进，只缩短抬头', previous, validate_plan(proposal))
    assert result['status'] == 'clarify' and previous == original


@pytest.mark.parametrize('template,expected', [
    ('{{ messages }}', {}),
    ('{% if enable_thinking %}<think>{% endif %}', {'enable_thinking': False}),
    ({'default': '{% if enable_thinking %}<think>{% endif %}'}, {'enable_thinking': False}),
])
def test_thinking_switch_only_reaches_templates_that_support_it(template, expected):
    seen = {}
    def render(messages, **kwargs):
        seen.update(kwargs)
        return 'rendered'
    processor = SimpleNamespace(chat_template=template, apply_chat_template=render)
    assert _render_prompt(processor, [{'role': 'user', 'content': 'plan'}]) == 'rendered'
    assert seen == dict(tokenize=False, add_generation_prompt=True, **expected)


def test_dynamic_camera_scope_is_prompted_without_repairing_model_output(tmp_path, monkeypatch):
    from training.creator import model_worker as worker
    (tmp_path / 'config.json').write_text('{}')
    previous = plan_request('一直前进，后半段抬头')
    seen = []
    monkeypatch.setattr(worker, '_generate', lambda p, s, c: seen.append((s, c)) or _proposal(previous['action_segments']))
    result = worker.execute_request(tmp_path, dict(kind='plan', text='保留前进，只缩短抬头', previous_plan=previous, capabilities={}), tmp_path)
    assert worker._read_json(seen[0][1])['requested_edit_scope'] == 'camera' and 'MUST be "camera"' in seen[0][0] and result['edit_scope'] == 'all'
