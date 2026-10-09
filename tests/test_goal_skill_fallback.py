"""Authored fallback contracts, not evidence that a real planner succeeded."""
import pytest
from interactive.engine import InteractionEngine
from interactive.agent import LLMAgent
from interactive.goal_fallback import key_request,key_skill_proposal
from interactive.tasks import plan_signature,GoalTask


FULL_ROUTE='我不相信 Anna，选择独自调查。搜索桌子，拿起红钥匙，用钥匙打开门，再进入储藏室，完成钥匙路线。已经完成的动作不要重复。'

def test_full_route_fallback_is_explicit_bounded_and_does_not_commit(tmp_path):
    from interactive.goal_fallback import key_route_skill_proposal,key_route_request
    class Refuser:
        def complete(self,*args,**kwargs):
            return '{"status":"needs_clarification","clarification":"直接动作的前置条件不满足，不能擅自改写或补全动作序列。"}'
    e=InteractionEngine(tmp_path,agent=LLMAgent(Refuser()))
    e.state=e.state.apply_delta({'story':{'node':'meet_anna','flags':{'met_anna':True}}})
    before=e.get_state(); t=e.replace_goal(FULL_ROUTE)
    assert t.status=='ready' and t.planning_source=='authored_skill_fallback'
    assert t.model_failure['status']=='needs_clarification' and t.proposal.success_checks==['ending_key']
    assert plan_signature(t.proposal.steps)==[['talk','anna','distrust'],['search','desk',None],
        ['pick_up','red_key',None],['open','door_a',None],['move_to','storage_room',None]]
    assert not e.history and not e.state.inventory.items
    assert e.state.story.model_dump()==before['story']
    for suffix in ('不要开门。','不要接近Anna。','去桥边。','打开门后杀死Anna。'):
        assert not key_route_request(FULL_ROUTE+suffix)
    assert not key_route_request('拿钥匙，不要开门')
    e.state.story.route='A'
    assert key_route_skill_proposal(e,FULL_ROUTE) is None


def test_skill_is_limited_and_never_executes(tmp_path):
    engine=InteractionEngine(tmp_path)
    before=engine.get_state()
    proposal=key_skill_proposal(engine,'我不信任 Anna，帮我找到钥匙，但不要开门，也不要进入储藏室。')
    assert plan_signature(proposal.steps)==[['approach','anna',None],['talk','anna','distrust'],['search','desk',None],['pick_up','red_key',None]]
    GoalTask('找到钥匙，不要开门',proposal).preflight(engine)
    assert before==engine.get_state() and not engine.history
    for text in ('拿钥匙再开门','去桥边拿钥匙','找钥匙，不要搜索桌子','只拾取钥匙','不要拿钥匙，只看看左边'):
        assert not key_request(text)


def test_fallback_declines_forbidden_approach_and_skips_owned_key(tmp_path):
    engine=InteractionEngine(tmp_path)
    proposal=key_skill_proposal(engine,'找到钥匙，不要接近 Anna。')
    assert proposal.status=='needs_clarification'
    engine.state=engine.state.apply_delta({'story':{'route':'B'},'inventory':{'items':['red_key']}})
    proposal=key_skill_proposal(engine,'找到钥匙，不要开门')
    assert plan_signature(proposal.steps)==[['observe','room',None]]


def test_engine_exposes_fallback_and_original_model_failure(tmp_path):
    class BadProvider:
        def complete(self,*args,**kwargs): return '{"invalid":true}'
    engine=InteractionEngine(tmp_path,agent=LLMAgent(BadProvider()))
    task=engine.replace_goal('找到钥匙，不要开门')
    assert task.status=='ready' and task.planning_source=='authored_skill_fallback'
    assert task.as_dict()['model_failure']['status']=='invalid_model_plan'
    assert not engine.history and not engine.state.inventory.items
    with pytest.raises(ValueError): engine.replace_goal('去桥边')


def test_goal_adapter_is_separate_from_single_action_provider(tmp_path):
    class SingleProvider:
        def complete(self,*args,**kwargs): raise AssertionError('Goal must not route to single-action provider')
    class GoalProvider:
        def complete(self,*args,**kwargs):
            return '{"status":"ready","steps":[{"instruction":"看看左边","tool_id":"inspect:left"}],"success_checks":[],"completion_mode":"action_execution"}'
    engine=InteractionEngine(tmp_path,agent=LLMAgent(SingleProvider(),goal_provider=GoalProvider()))
    task=engine.replace_goal('只看看左边')
    assert task.planning_source=='language_model' and task.status=='ready'
