"""Bounded Chinese control phrasing, not model or visual quality evidence."""
import copy

import pytest

from training.creator.planner import diagnose_proposal, expand_segments, plan_request


@pytest.mark.parametrize('text',[
    '先把视角左转再向右转',
    '先镜头向左转，然后右转',
    '把视角先左转再右转',
])
def test_explicit_camera_subject_carries_across_turns(text):
    result=plan_request(text)
    assert result['status']=='ready'
    assert result['action_segments']==[{'frames':120,'keys':['J']},{'frames':120,'keys':['L']}]
    assert '0–7.5 秒' in result['explanation'] and '7.5–15 秒' in result['explanation']


@pytest.mark.parametrize('text',[
    '先左转再右转', '先向左转，然后向右转',
    '先镜头左转，然后向前走，再右转',
])
def test_missing_or_changed_subject_is_not_guessed(text):
    assert plan_request(text)['status']=='clarify'


def test_camera_ellipsis_keeps_history_movement_frame_for_frame():
    previous={'action_segments':[{'frames':64,'keys':['W']},{'frames':176,'keys':['D','I']}]}
    original=copy.deepcopy(previous)
    result=plan_request('只改镜头，先左转再右转',previous)
    assert result['status']=='ready' and result['edit_scope']=='camera' and result['preserved']
    assert previous==original
    for before,after in zip(expand_segments(previous['action_segments']),expand_segments(result['action_segments'])):
        assert set(before)&set('WASD')==set(after)&set('WASD')
    assert plan_request('保持镜头不变，先左转再右转',previous)['status']=='clarify'


def test_numeric_camera_ellipsis_uses_exact_existing_time_contract():
    result=plan_request('先镜头左转3秒再向右转2秒')
    assert result['status']=='ready'
    assert result['action_segments']==[
        {'frames':48,'keys':['J']},{'frames':32,'keys':['L']},{'frames':160,'keys':[]}]
    assert '0–7.5' not in result['explanation']


def test_small_amount_is_not_silently_converted_to_distance_or_new_seconds():
    result=plan_request('再向前走一点，然后抬头')
    assert result['status']=='ready'
    assert result['action_segments']==[{'frames':120,'keys':['W']},{'frames':120,'keys':['I']}]
    assert '0–7.5 秒' in result['explanation']
    assert '没有被换算为实际距离或角度' in result['explanation']


def test_external_proposal_still_checks_order_and_reports_default():
    good={'edit_scope':'all','action_segments':[{'frames':120,'keys':['J']},{'frames':120,'keys':['L']}]}
    text='先把视角左转再向右转'
    assert '0–7.5 秒' in plan_request(text,proposal=good)['explanation']
    reversed_plan={**good,'action_segments':list(reversed(good['action_segments']))}
    rejected=diagnose_proposal(text,proposal=reversed_plan)
    assert rejected['plan']['status']=='clarify'
    assert rejected['feedback']['code']=='timing_mismatch'


@pytest.mark.parametrize('text',['向山走','朝山移动，然后抬头','往远处的山峰方向走','向着那座山走一点'])
def test_mountain_reference_requests_explicit_user_rewrite_not_navigation(text):
    proposal={'edit_scope':'all','action_segments':[{'frames':240,'keys':['W']}]}
    result=plan_request(text,proposal=proposal)
    assert result['status']=='clarify' and result['action_segments']==[]
    assert '自行确认并改写' in result['explanation']
    assert '不是向山导航' in result['explanation']


def test_movement_and_camera_are_not_confused():
    result=plan_request('先向左移动，再向右移动')
    assert result['action_segments']==[{'frames':120,'keys':['A']},{'frames':120,'keys':['D']}]
    assert plan_request('人物先左转再右转')['status']=='unsupported'
