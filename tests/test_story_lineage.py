"""CPU input contracts only; nonplayable fixtures do not validate any video."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from interactive.agent import SkillSelector
from interactive.models import Plan
from interactive.renderers import LaboratoryWanRenderer, RenderResult
from interactive.story_lineage import request_evidence, verify_result
from interactive.live_continuation import rgb_sha256
from interactive.live_ui import start_ui_world, require_ui_frame


SESSION = 'd'*32


@pytest.fixture
def initialized(tmp_path):
    source = tmp_path/'source.png'; Image.new('RGB', (832, 480), '#123456').save(source)
    folder = tmp_path/SESSION; folder.mkdir()
    current = folder/'current.png'; Image.open(source).save(current)
    renderer = LaboratoryWanRenderer('http://never-called.invalid', source,
                                    seconds=6, story_lineage_guard=True)
    renderer.reset_story_session(SESSION)
    renderer.establish_story_input(SESSION, current, folder)
    return renderer, folder, current, source


def response_fixture(initialized, plan):
    renderer, folder, current, _ = initialized
    renderer.bind_story_frame(SESSION, current, session_dir=folder)
    skills = [getattr(s, 'value', s) for s in SkillSelector().select(plan)]
    out = folder/'turn-0001'; out.mkdir()
    Image.open(current).save(out/'input.png'); Image.open(current).save(out/'last.png')
    (out/'video.mp4').write_bytes(b'CPU fixture, not a playable model output')
    actions = np.zeros((96, 8), np.float32); cursor = 0
    for segment in SkillSelector().action_segments(skills, frames=96):
        end = cursor+segment['frames']
        for key in segment['keys']:
            actions[cursor:end, ['W', 'A', 'S', 'D', 'I', 'J', 'K', 'L'].index(key)] = 1
        cursor = end
    np.save(out/'actions.npy', actions)
    evidence = request_evidence(renderer, current, out, plan, skills)
    response = {'video_path': str(out/'video.mp4'), 'last_frame': str(out/'last.png'),
                'metadata': {'kind': 'wan_laboratory', 'checkpoint': 'R005-step1040',
                             'conditioning_image': str(out/'input.png'),
                             'frames': 97, 'fps': 16, 'prompt_key': renderer.prompt_for(plan)}}
    return renderer, current, out, skills, evidence, response


def test_guard_optin_is_disjoint_and_old_renderer_is_unchanged(initialized):
    _, _, _, source = initialized
    legacy = LaboratoryWanRenderer('http://never-called.invalid', source)
    assert not legacy.story_lineage_guard and not legacy.live_current_frame
    assert legacy.bind_story_frame('not-read', None) is None
    for settings in ({'story_lineage_guard': 1},
                     {'story_lineage_guard': True, 'npc_framing': True, 'seconds': 6},
                     {'story_lineage_guard': True, 'live_current_frame': True, 'live_prompt_key': 'neutral'}):
        with pytest.raises(ValueError, match='Story lineage'):
            LaboratoryWanRenderer('http://never-called.invalid', source, **settings)


def test_story_event_prompt_and_original_skill_contract_are_preserved(initialized):
    plan = Plan(player_intent='CPU fixture', semantic_action='search', target='desk')
    _, _, _, _, evidence, response = response_fixture(initialized, plan)
    verified = verify_result(response, evidence, plan)
    assert verified['expected_prompt_key'] == 'key_reveal'
    assert response['metadata']['prompt_key'] == 'key_reveal'
    assert 'live_current_frame_evidence' not in response['metadata']
    assert verified['mode'] == 'story_lineage_guard'
    assert verified['lineage_check_passed'] and verified['conditioning_rgb_match']
    assert not verified['visual_success_verified'] and not verified['same_door_position_verified']
    assert not verified['world_navigation_verified']


def test_missing_foreign_tampered_inputs_and_unowned_output_reject(initialized):
    renderer, folder, current, source = initialized
    renderer.bind_story_frame(SESSION, current, session_dir=folder)
    for frame in (None, source):
        with pytest.raises(ValueError, match='bound current'):
            request_evidence(renderer, frame, folder/'turn', None, ['static'])
    with pytest.raises(ValueError, match='belong to this engine'):
        renderer.bind_story_frame(SESSION, source, session_dir=folder)
    with pytest.raises(ValueError, match='another or uninitialized'):
        renderer.bind_story_frame('e'*32, current, session_dir=folder)
    with pytest.raises(ValueError, match='output must belong'):
        request_evidence(renderer, current, folder.parent/'outside', None, ['static'])
    Image.new('RGB', (832, 480), 'red').save(current)
    with pytest.raises(ValueError, match='pixels changed'):
        request_evidence(renderer, current, folder/'turn', None, ['static'])


@pytest.mark.parametrize('forbidden', ['pre_generated', 'explicit_goal_input', 'framing_constraint',
                                      'recovered_existing_initial', 'existing_video_reused'])
def test_prebuilt_goal_and_recovered_visual_results_reject(initialized, forbidden):
    plan = Plan(player_intent='CPU fixture', semantic_action='open', target='door_a')
    _, _, _, _, evidence, result = response_fixture(initialized, plan)
    result['metadata'][forbidden] = True
    with pytest.raises(ValueError, match='Story input lineage rejected'):
        verify_result(result, evidence, plan)
    assert 'story_lineage_evidence' not in result['metadata']


def test_actual_input_prompt_and_action_schedule_checked(initialized):
    plan = Plan(player_intent='CPU fixture', semantic_action='inspect', target='right')
    _, current, out, _, evidence, result = response_fixture(initialized, plan)
    result['metadata']['prompt_key'] = 'another-key'
    with pytest.raises(ValueError, match='prompt key'):
        verify_result(result, evidence, plan)
    result['metadata']['prompt_key'] = evidence['expected_prompt_key']
    Image.new('RGB', (832, 480), 'red').save(out/'input.png')
    with pytest.raises(ValueError, match='Actual model input differs'):
        verify_result(result, evidence, plan)
    Image.open(current).save(out/'input.png')
    np.save(out/'actions.npy', np.zeros((96, 8), np.float32))
    with pytest.raises(ValueError, match='action schedule differs'):
        verify_result(result, evidence, plan)


def test_single_story_submission_consumes_binding_and_retains_event_prompt(initialized, monkeypatch):
    plan = Plan(player_intent='CPU fixture', semantic_action='open', target='door_a')
    renderer, current, out, skills, _, result = response_fixture(initialized, plan)
    calls = []
    health = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'busy': False})
    def post(url, **kwargs):
        assert renderer._story_request_active
        calls.append(kwargs['json'])
        return SimpleNamespace(ok=True, json=lambda: result)
    monkeypatch.setattr('interactive.renderers.requests.get', lambda *a, **kw: health)
    monkeypatch.setattr('interactive.renderers.requests.post', post)
    monkeypatch.setattr(renderer, 'finish_render', lambda value, *args: RenderResult(
        value['video_path'], value['last_frame'], value['metadata']))
    actual = renderer.generate_plan(current, skills, '', None, out, plan)
    assert len(calls) == 1 and calls[0]['prompt_key'] == 'door_open'
    assert calls[0]['initial'] == str(current) and 'framing_condition' not in calls[0]
    assert actual.metadata['story_lineage_evidence']['lineage_check_passed']
    assert renderer._story_binding is None and not renderer._story_request_active
    with pytest.raises(ValueError, match='before every story render'):
        renderer.generate_plan(current, skills, '', None, out, plan)
    assert len(calls) == 1


def test_explicit_reset_cannot_replace_inflight_binding(initialized):
    renderer, folder, current, _ = initialized
    renderer._story_request_active = True
    with pytest.raises(ValueError, match='active render'):
        renderer.reset_story_session('e'*32)
    with pytest.raises(ValueError, match='in-flight'):
        renderer.bind_story_frame(SESSION, current, session_dir=folder)
    renderer._story_request_active = False
    renderer.reset_story_session('e'*32)
    assert renderer._story_binding is None and not renderer._story_input_initialized
    with pytest.raises(ValueError, match='another or uninitialized'):
        renderer.bind_story_frame(SESSION, current, session_dir=folder)


def test_existing_ui_helper_routes_story_rgb_without_changing_legacy(initialized):
    _, _, current, _ = initialized
    calls = []
    engine = SimpleNamespace(frame=None, start_from_current_frame=lambda p: calls.append(('rgb', p)),
                             start_world=lambda p=None: calls.append(('legacy', p)))
    start_ui_world(engine, str(current), story_lineage_guard=True)
    assert calls == [('rgb', str(current))]
    with pytest.raises(ValueError, match='不会自动生成'):
        require_ui_frame(engine, story_lineage_guard=True)
    start_ui_world(engine, str(current))
    assert calls[-1] == ('legacy', str(current))
