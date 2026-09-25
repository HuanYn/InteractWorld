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


def _judgment(observations, *, verdict='satisfied'):
    return dict(verdict=verdict, evidence=observations,
                decision='accept' if verdict == 'satisfied' else 'revise',
                revision_text='' if verdict == 'satisfied' else '调整镜头输入。',
                confidence_note='Only sparse sampled frames were inspected; not calibrated.')


def _inspection_files(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    current, reference = tmp_path / 'current.mp4', tmp_path / 'reference.mp4'
    # Explicit unit fixtures, not real videos or claims of visual model output.
    current.write_bytes(b'current test fixture')
    reference.write_bytes(b'reference test fixture')
    return current, reference


def test_goal_semantics_separate_camera_movement_relative_and_dense_time():
    from training.creator.model_worker import goal_semantics
    goals = ['镜头向上俯仰', '人物全程向前移动', '镜头比之前更高', '缩短镜头抬头时长', '停止移动']
    semantics = goal_semantics(goals)
    assert [item['categories'] for item in semantics] == [
        ['camera'], ['movement'], ['camera', 'relative'], ['camera', 'relative'], ['movement']]
    assert [item['requires_dense_temporal_evidence'] for item in semantics] == [False, True, False, True, True]


def test_relative_goal_without_reference_skips_model_and_frame_extraction(tmp_path, monkeypatch):
    from training.creator import model_worker as worker
    current, _ = _inspection_files(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('missing-reference rule must run before sampling or model inference')
    monkeypatch.setattr(worker, '_generate', forbidden)
    monkeypatch.setattr(worker, 'sample_frames', forbidden)
    result = worker.execute_request(tmp_path, dict(kind='inspect', video_path=str(current),
                                                  goals=['缩短镜头抬头时长']), tmp_path)
    assert (result['verdict'], result['decision'], result['reference_used']) == ('uncertain', 'ask_user', False)
    assert 'reference_required' in result['rule_reasons']
    assert result['original_model_judgment'] is None and not result['rule_downgraded']


def test_reference_inputs_are_labelled_bounded_and_never_include_actions(tmp_path, monkeypatch):
    from training.creator import model_worker as worker
    current, reference = _inspection_files(tmp_path)
    def sample(path, directory):
        offset = 0.25 if str(path) == str(reference) else 0.0
        return [(directory / f'{index}.png', index + offset) for index in range(8)], ''
    def generate(model_path, system, text, samples, reference_samples=()):
        content = worker._read_json(text)
        assert len(samples) == len(reference_samples) == 8
        assert content['sample_times_seconds'] == dict(current=list(range(8)), reference=[x + 0.25 for x in range(8)])
        assert 'action_segments' not in text and 'FORBIDDEN_ACTION_EVIDENCE' not in text
        assert 'CURRENT' in system and 'REFERENCE' in system and 'never character head pose' in system
        return _judgment([dict(video='current', time_seconds=0, observation='Horizon is lower within this image.'),
                          dict(video='reference', time_seconds=0.25, observation='Horizon is higher within this image.')])
    monkeypatch.setattr(worker, 'sample_frames', sample)
    monkeypatch.setattr(worker, '_generate', generate)
    result = worker.execute_request(tmp_path, dict(kind='inspect', video_path=str(current), reference_video_path=str(reference),
        goals=['镜头比之前更高'], action_segments='FORBIDDEN_ACTION_EVIDENCE'), tmp_path)
    assert result['reference_used'] is True and result['verdict'] == 'satisfied'
    assert not result['rule_downgraded'] and result['original_model_judgment'] is None
    assert '未校准' in result['confidence_note']


@pytest.mark.parametrize('evidence,reference_times', [
    ([dict(time_seconds=0, observation='Missing source label.')], [0.25]),
    ([dict(video='current', time_seconds=0.25, observation='Wrong video timestamp.')], [0.25]),
    ([dict(video='reference', time_seconds=0, observation='Wrong video timestamp.')], [0.25]),
    ([dict(video='reference', time_seconds=0, observation='No reference was supplied.')], None),
])
def test_comparison_evidence_validates_each_videos_actual_timestamps(evidence, reference_times):
    from training.creator.model_worker import validate_inspection
    with pytest.raises(ValueError, match='video|unsampled'):
        validate_inspection(_judgment(evidence), [0.0], reference_times)


def test_legacy_single_video_evidence_still_validates():
    from training.creator.model_worker import validate_inspection
    judgment = _judgment([dict(time_seconds=0, observation='The horizon is near the image centre.')])
    assert validate_inspection(judgment, [0.0]) == judgment


@pytest.mark.parametrize('goal,observation,reason', [
    ('镜头向上俯仰', '人物头部没有抬起。', 'character_pose_is_not_camera_evidence'),
    ('人物向前移动', '人物持续前进，随后停止移动。', 'sparse_frames_do_not_establish_continuity_or_stopping'),
    ('人物全程向前移动', '人物在图像中央。', 'dense_temporal_evidence_required'),
    ('缩短镜头抬头时长', '天空在当前图像中占比更大。', 'dense_temporal_evidence_required'),
])
def test_sparse_or_character_pose_claims_are_downgraded_with_original_preserved(goal, observation, reason):
    from training.creator.model_worker import apply_observation_rules, goal_semantics
    original = _judgment([dict(video='current', time_seconds=0, observation=observation),
                          dict(video='reference', time_seconds=0, observation='山脊位于图像中部。')], verdict='unsatisfied')
    before = copy.deepcopy(original)
    result = apply_observation_rules(original, goal_semantics([goal]), reference_used=True)
    assert (result['verdict'], result['decision'], result['revision_text']) == ('uncertain', 'ask_user', '')
    assert reason in result['rule_reasons'] and result['rule_downgraded']
    assert result['original_model_judgment'] == before and original == before
    if reason != 'dense_temporal_evidence_required':
        assert all(item['observation'] != observation for item in result['evidence'])


def test_relative_comparison_needs_evidence_from_both_supplied_videos():
    from training.creator.model_worker import apply_observation_rules, goal_semantics
    original = _judgment([dict(video='current', time_seconds=0, observation='山脊位于图像下方。')])
    result = apply_observation_rules(original, goal_semantics(['镜头比之前更高']), reference_used=True)
    assert result['verdict'] == 'uncertain'
    assert 'relative_comparison_needs_both_video_sources' in result['rule_reasons']
    assert result['original_model_judgment'] == original


def test_more_than_eight_frames_per_video_rejected_before_model_import():
    from training.creator.model_worker import _generate
    with pytest.raises(ValueError, match='eight frames per video'):
        _generate(None, '', '', [(None, 0)] * 8, reference_samples=[(None, 0)] * 9)


def test_camera_half_duration_synonyms_remain_relative_not_character_movement():
    from training.creator.model_worker import goal_semantics
    goals = ['镜头俯仰减半', '减少镜头转向', '增加镜头俯仰时间一半']
    for semantics in goal_semantics(goals):
        assert semantics['categories'] == ['camera', 'relative']
        assert semantics['requires_dense_temporal_evidence'] is True


def test_explicit_half_and_half_more_proposals_follow_extension_policy():
    previous = plan_request('一直前进，后半段抬头')
    for instruction, start, duration in [('保留前进，只把抬头减半', 120, 60),
                                         ('保留前进，抬头时间增加一半', 120, 90),
                                         ('保留前进，抬头时间增加一半', 105, 135)]:
        segments = [dict(frames=start, keys=['W']), dict(frames=duration, keys=['W', 'I'])]
        if start + duration < 240:
            segments.append(dict(frames=240-start-duration, keys=['W']))
        proposal = _proposal(segments, scope='camera')
        result = plan_request(instruction, previous, validate_plan(proposal))
        assert result['status'] == 'ready' and result['preserved'] is True
        assert [i for i, row in enumerate(expand_segments(result['action_segments'])) if 'I' in row] == list(range(start, start+duration))
        previous = result
