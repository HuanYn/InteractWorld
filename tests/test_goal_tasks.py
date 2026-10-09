import json
import pytest
from interactive.engine import InteractionEngine
from interactive.models import Plan,Observation
from interactive.renderers import RenderResult
from interactive.tasks import GoalProposal,GoalStep,GoalTask,propose_goal
from interactive.agent import LLMAgent


class Renderer:
    def generate(self,*args): return RenderResult(None,None,{'kind':'test_double'})


def step(action,target,instruction='',dialogue=None):
    return GoalStep(instruction=instruction or f'{action} {target}',plan=Plan(
        player_intent='fixture',semantic_action=action,target=target,dialogue_action=dialogue))


ROUTES={
 'B':([step('approach','anna'),step('talk','anna',dialogue='distrust'),step('search','desk'),
       step('pick_up','red_key'),step('open','door_a'),step('move_to','storage_room')],'ending_key'),
 'A':([step('approach','anna'),step('interact','anna',dialogue='help'),step('ask','anna',dialogue='password'),
       step('inspect','terminal'),step('move_to','core_area')],'ending_anna'),
 'C':([step('inspect','terminal'),step('inspect','terminal'),step('approach','leo'),
       step('ask','leo',dialogue='logs')],'ending_terminal')}


def task_for(route):
    steps,ending=ROUTES[route]
    return GoalTask('CPU fixture, not model output',GoalProposal(status='ready',steps=steps,success_checks=[ending]))


@pytest.mark.parametrize('route',['A','B','C'])
def test_bounded_task_routes(tmp_path,route):
    e=InteractionEngine(tmp_path,renderer=Renderer()); t=e.task=task_for(route)
    before=e.get_state(); t.preflight(e)
    assert e.get_state()==before and not e.history
    for _ in t.proposal.steps:
        row=t.advance(e); assert row['valid']
    assert t.status=='completed_rule_goal' and e.state.story.node==ROUTES[route][1]
    assert not t.as_dict()['visual_goal_verified']
    assert all(r['plan_source']=='reviewed_goal_plan' for r in e.history)
    assert json.loads(open(e.export_session(),encoding='utf-8').read())['task']['status']==t.status


def test_preflight_rejects_missing_prerequisites(tmp_path):
    e=InteractionEngine(tmp_path,renderer=Renderer())
    t=GoalTask('open without key',GoalProposal(status='ready',steps=[step('open','door_a')],success_checks=['door_open']))
    with pytest.raises(ValueError,match='不可执行'): t.preflight(e)
    assert not e.history and e.state.objects['door_a'].state=='locked'
    evidence=t.as_dict()['precondition_failure']
    assert evidence['state_source']=='dry_run_simulated_not_committed'
    assert evidence['not_execution_authority'] is True
    evidence['reason']='external mutation'
    assert t.as_dict()['precondition_failure']['reason']!='external mutation'


def test_failed_step_and_manual_interruption(tmp_path):
    e=InteractionEngine(tmp_path,renderer=Renderer()); t=e.task=task_for('B'); t.preflight(e)
    class Broken:
        def generate(self,*args): raise RuntimeError('fixture')
    e.renderer=Broken(); assert not t.advance(e)['valid']
    assert t.status=='paused' and t.index==0
    e.renderer=Renderer(); t.resume(); assert t.advance(e)['valid']
    assert t.index==1
    e.interact('观察房间'); assert t.status=='paused'
    e.reset(); assert e.task is None


def test_stop_and_retry_cap(tmp_path):
    e=InteractionEngine(tmp_path,renderer=Renderer()); t=e.task=task_for('B')
    t.pause_requested.set(); assert t.advance(e) is None and not e.history
    t.resume(); t.attempts=12
    assert t.advance(e) is None and t.status=='paused'
    with pytest.raises(ValueError): t.resume()


def test_visual_contradiction_pauses_without_replaying_commit(tmp_path):
    class Observer:
        def observe(self,*args): return Observation(source='unavailable')
        def assess_goal(self,*args): return {'status':'contradicted','ground_truth':False}
    e=InteractionEngine(tmp_path,renderer=Renderer(),observer=Observer()); t=e.task=task_for('B')
    assert t.advance(e)['valid']
    assert t.status=='paused' and t.index==1 and e.state.story.flags['met_anna']


def test_goal_parse_clarification_and_bad_success_claim(tmp_path):
    class Provider:
        def complete(self,messages,**kwargs):
            assert 'supported_calls' in messages[-1]['content']
            return '{"status":"needs_clarification","clarification":"当前没有桥梁场景","steps":[],"success_checks":[]}'
    e=InteractionEngine(tmp_path,agent=LLMAgent(Provider()),renderer=Renderer())
    t=GoalTask('go to bridge',propose_goal(e,'走到桥边')); t.preflight(e)
    assert t.status=='needs_clarification' and not e.history
    bad=GoalTask('bad check',GoalProposal(status='ready',steps=[step('observe','room')],success_checks=['key_owned']))
    with pytest.raises(ValueError,match='规则目标'): bad.preflight(e)
    with pytest.raises(ValueError): GoalProposal(status='ready',steps=[])
