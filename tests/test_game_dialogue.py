"""Software-only fixture checks; no model/video/dialogue quality claim."""
from copy import deepcopy
import json
import pytest

from interactive.game_dialogue import build_game_dialogue, render_dialogue_html


def turn(index=1, **updates):
    value = {'turn_id': index, 'player_text': '你好 Anna', 'valid': True,
        'rule_committed': True, 'reason': '你见到了紧张的研究员 Anna。',
        'npc_response': 'Anna：你是谁？这里发生了事故。',
        'plan': {'target': 'anna', 'player_intent': 'THIS IS NOT NPC DIALOGUE'},
        'state': {'story': {'route': None, 'node': 'meet_anna', 'terminal': False}}}
    value.update(updates)
    return value


def session(*turns):
    return {'session_id': 'software-fixture', 'turns': list(turns),
        'state': {'story': {'route': 'B', 'node': 'ending_key', 'terminal': True}}}


def test_exact_rule_npc_line_is_separate_from_player_narrator_and_intent():
    payload = build_game_dialogue(session(turn()))
    assert [m['speaker'] for m in payload['messages']] == ['玩家', '旁白', 'Anna']
    npc = payload['messages'][-1]
    assert npc['text'] == '你是谁？这里发生了事故。'
    assert npc['raw_text'] == 'Anna：你是谁？这里发生了事故。'
    assert npc['source'] == 'authored_rule_dialogue'
    assert npc['source_reference'] == 'turns[0].npc_response'
    assert 'THIS IS NOT NPC DIALOGUE' not in json.dumps(payload)
    assert not npc['audio_generated'] and not npc['lip_sync_verified']
    assert not npc['visual_success_verified']


def test_goal_step_instruction_is_not_recast_as_player_quotation():
    payload = build_game_dialogue(session(turn(player_text='规则技能建议：talk:anna:distrust',
        plan_source='reviewed_goal_plan')))
    assert payload['messages'][0]['speaker'] == '行动'
    assert payload['messages'][0]['source'] == 'goal_tool_instruction'
    assert payload['messages'][0]['role'] == 'action'


def test_rejected_or_uncommitted_npc_reply_does_not_become_dialogue():
    payload = build_game_dialogue(session(turn(valid=False, rule_committed=False,
        reason='Anna 信任不足，不会透露密码。')))
    assert not [m for m in payload['messages'] if m['role'] == 'npc']
    assert payload['messages'][-1]['source'] == 'rule_rejection'


def test_selecting_historical_turn_excludes_future_route_and_dialogue():
    second = turn(2, npc_response='Leo：我们现在有足够证据了。',
        plan={'target': 'leo'}, state={'story': {'route': 'C', 'terminal': True}})
    payload = build_game_dialogue(session(turn(), second), turn_id=1)
    assert len(payload['groups']) == 1 and not payload['terminal']
    assert payload['route'] is None
    assert 'Leo' not in json.dumps(payload)


def test_opening_and_terminal_without_turns_are_labelled_authored_narration():
    start = build_game_dialogue({'state': {'story': {'node': 'enter_lab'}}})
    assert start['latest']['turn_id'] == 0
    assert start['messages'][0]['source'] == 'authored_scene_intro'
    assert start['messages'][0]['speaker'] == '旁白'
    ending = build_game_dialogue(session())
    assert ending['terminal']
    assert ending['messages'][0]['source'] == 'authored_ending_label'


def test_archive_opening_does_not_reveal_later_ending_or_invent_missing_turn():
    payload = build_game_dialogue(session(turn()), turn_id=0)
    assert not payload['terminal'] and payload['route'] is None
    assert payload['latest']['turn_id'] == 0
    assert payload['messages'][0]['source'] == 'authored_scene_intro'
    with pytest.raises(ValueError, match='not present'):
        build_game_dialogue(session(turn()), turn_id=7)
    with pytest.raises(ValueError, match='nonnegative'):
        build_game_dialogue(session(turn()), turn_id='1')


def test_legacy_memory_keeps_unknown_turn_identity_and_does_not_duplicate_turns():
    data = {'memory': {'recent': [turn(turn_id=None)]}}
    payload = build_game_dialogue(data)
    assert payload['history_source'] == 'short_term_memory'
    assert payload['latest']['turn_id'] is None
    assert payload['messages'][-1]['source_reference'] == 'memory.recent[0].npc_response'
    data['turns'] = [turn()]
    payload = build_game_dialogue(data)
    assert len(payload['groups']) == 1 and payload['history_source'] == 'session_turns'


def test_html_escapes_all_archive_content_without_scripts_or_unsafe_attributes():
    payload = build_game_dialogue(session(turn(player_text='<img src=x onerror=alert(1)>',
        npc_response='Anna：<script>alert("x")</script>')))
    payload['route_title'] = '<svg onload="boom">'
    payload['messages'][0]['role'] = 'npc" onclick="evil'
    html = render_dialogue_html(payload)
    assert '<script>' not in html and '<img ' not in html and '<svg ' not in html
    assert '&lt;script&gt;' in html and '&lt;img ' in html and '&lt;svg ' in html
    assert 'data-role="npc" onclick=' not in html


def test_history_and_latest_are_json_safe_and_readonly():
    data = session(turn(), turn(2, npc_response='Leo: 最终证据已保存。',
        plan={'target': 'leo'}))
    original = deepcopy(data)
    payload = build_game_dialogue(data)
    html = render_dialogue_html(payload)
    json.dumps(payload, ensure_ascii=False)
    assert data == original
    assert payload['latest']['turn_id'] == 2
    assert '对话历史' in html and '最终证据已保存。' in html
    assert '对话历史' not in render_dialogue_html(payload, show_history=False)


def test_model_source_must_be_explicit_not_inferred_from_agent_kind():
    authored = build_game_dialogue(session(turn(agent_kind='qwen_real')))
    assert authored['messages'][-1]['source'] == 'authored_rule_dialogue'
    declared = build_game_dialogue(session(turn(npc_response_source='language_model')))
    assert declared['messages'][-1]['source'] == 'model_npc_response'
    unknown = build_game_dialogue(session(turn(npc_response_source='some_unknown_provider')))
    assert unknown['messages'][-1]['source'] == 'unverified_npc_response'
