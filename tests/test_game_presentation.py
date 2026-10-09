"""CPU presentation contracts, never evidence of model or visual success."""
import json
from pathlib import Path

import pytest
from interactive.game_presentation import compact_dialogue_html,ending_text,GameArchive
from interactive.game_dialogue import build_game_dialogue,render_dialogue_html
from interactive.engine import InteractionEngine
from interactive.route_showcase import route_goal
from interactive.tasks import ToolGoalDraft


def test_compact_npc_is_exact_not_reason_or_tool_instruction():
    data={'state':{'story':{'route':'B'}},'turns':[
        {'turn_id':3,'valid':True,'rule_committed':True,'plan_source':'reviewed_goal_plan',
         'player_text':'approach:anna','reason':'已经靠近 Anna',
         'plan':{'target':'anna'},'npc_response':'Anna：你是谁？这里发生了事故。'}]}
    html=compact_dialogue_html(data)
    assert '你是谁？这里发生了事故。' in html
    assert '已经靠近 Anna' not in html and 'approach:anna' not in html
    assert 'iw-dialogue-header' not in html and 'iw-dialogue-source' not in html
    assert '已编写剧情台词' not in html and '第 3 轮' not in html
    # Presentation is minimal; the unmodified full record remains traceable.
    full=build_game_dialogue(data)
    assert full['latest']['messages'][-1]['source']=='authored_rule_dialogue'
    assert '已编写剧情台词' in render_dialogue_html(full)


def test_compact_rejection_remains_visible_and_escaped():
    data={'turns':[{'turn_id':2,'valid':False,'rule_committed':False,
        'player_text':'开门','reason':'缺少钥匙。<img src=x onerror=alert(1)>',
        'npc_response':'Anna：不能把未提交的台词显示为成功。'}]}
    html=compact_dialogue_html(data)
    assert '系统' in html and '缺少钥匙。' in html
    assert 'data-role="system"' in html and '&lt;img' in html and '<img' not in html
    assert 'Anna' not in html and '不能把未提交' not in html


def test_compact_narration_keeps_exact_text_without_route_debug():
    text='你找到了一把红钥匙。请先确认门是否能打开。'
    data={'state':{'story':{'route':'B'}},'turns':[{'turn_id':6,
        'valid':True,'rule_committed':True,'player_text':'搜索桌子',
        'reason':text,'npc_response':''}]}
    html=compact_dialogue_html(data)
    assert '旁白' in html and text in html
    assert '第 6 轮' not in html and '规则旁白' not in html


def test_task_only_block_is_visible_without_rewriting_archived_turn():
    data={'task':{'status':'needs_clarification','message':'请先说明要打开哪一扇门。'},
        'turns':[{'turn_id':1,'valid':True,'rule_committed':True,
                  'reason':'已经靠近 Anna','npc_response':'Anna：你是谁？'}]}
    current=compact_dialogue_html(data)
    assert '系统' in current and data['task']['message'] in current
    assert 'data-source="task_feedback"' in current and 'Anna' not in current
    historical=compact_dialogue_html(data,1)
    assert 'Anna' in historical and '你是谁？' in historical
    assert data['task']['message'] not in historical


def test_endings_are_route_specific():
    assert '核心区域' in ending_text('A') and '储藏室' not in ending_text('A')
    assert '储藏室' in ending_text('B')
    assert 'Leo' in ending_text('C') and '红钥匙' not in ending_text('C')


def test_read_only_archive_never_imports_state(tmp_path):
    session=tmp_path/'session.json'
    data={'session_id':'old','state':{'story':{'route':'C','terminal':True}},'turns':[
        {'turn_id':1,'valid':True,'reason':'检查终端','state':{'story':{'route':'C','terminal':False}},'npc_response':''}]}
    session.write_text(json.dumps(data),encoding='utf-8')
    video=tmp_path/'original.mp4'; video.write_bytes(b'placeholder-not-visual-evidence')
    manifest=tmp_path/'index.json'
    manifest.write_text(json.dumps({'routes':[{'route_id':r,'title':r,'session_export':str(session),
        'videos':{'1':str(video)}} for r in 'ABC']}),encoding='utf-8')
    before=session.read_bytes()
    archive=GameArchive(manifest)
    result=archive.view('C','1')
    assert result[2]['terminal'] is False
    assert result[3]['active_world_mutated'] is False
    assert result[3]['live_generation'] is False
    assert session.read_bytes()==before
    with pytest.raises(ValueError): archive.view('C','99')


class RefusalAgent:
    def plan_goal(self,*args,**kwargs):
        return ToolGoalDraft(status='needs_clarification',clarification='Fixture refusal, not a model result')


@pytest.mark.parametrize('route', ['A','C'])
def test_three_route_fallback_opt_in_keeps_failure_and_state(tmp_path,route,monkeypatch):
    import interactive.tasks as tasks
    def fixture_proposal(engine,text):
        return ToolGoalDraft(status='needs_clarification',clarification='Fixture refusal').compile()
    monkeypatch.setattr(tasks,'propose_goal',fixture_proposal)
    engine=InteractionEngine(tmp_path)
    engine.authored_route_fallback_enabled=True
    before=engine.get_state()
    task=engine.replace_goal(route_goal(route))
    assert task.status=='ready'
    assert task.planning_source=='authored_skill_fallback'
    assert task.model_failure
    assert engine.history==[] and engine.state.story==type(engine.state.story).model_validate(before['story'])
    assert engine.state.inventory.items==[]


def test_three_route_fallback_default_off(tmp_path,monkeypatch):
    import interactive.tasks as tasks
    monkeypatch.setattr(tasks,'propose_goal',lambda e,t:ToolGoalDraft(
        status='needs_clarification',clarification='Fixture refusal').compile())
    engine=InteractionEngine(tmp_path)
    task=engine.replace_goal(route_goal('C'))
    assert task.status=='needs_clarification'
    assert task.planning_source=='language_model'
