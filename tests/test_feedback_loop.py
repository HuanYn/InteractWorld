"""CPU contract tests only: no assertions about real model/video accuracy."""
import json
import pytest
from interactive.engine import InteractionEngine
from interactive.models import Observation,Plan
from interactive.renderers import RenderResult
from interactive.tasks import GoalProposal,GoalTask,GoalStep
from interactive.agent import LLMAgent


class Renderer:
    kind='fixture'
    def __init__(self): self.calls=0
    def generate(self,*args):
        self.calls+=1
        return RenderResult(None,None,{'kind':'fixture'})


def step(action,target,dialogue=None):
    return GoalStep(instruction=action+' '+target,plan=Plan(player_intent='fixture',
        semantic_action=action,target=target,dialogue_action=dialogue))


def picked_engine(tmp_path):
    e=InteractionEngine(tmp_path,renderer=Renderer())
    for s in [step('approach','anna'),step('talk','anna','distrust'),step('search','desk'),step('pick_up','red_key')]:
        assert e.interact(s.instruction,task_plan=s.plan)['valid']
    return e


class Observer:
    def __init__(self,status='contradicted'): self.status=status
    def observe(self,*args): return Observation(source='unavailable')
    def assess_goal(self,*args): return {'status':self.status,'ground_truth':False}


def test_committed_operation_id_is_idempotent(tmp_path):
    e=picked_engine(tmp_path); p=step('open','door_a')
    a=e.interact(p.instruction,task_plan=p.plan,operation_id='open-once')
    count=e.renderer.calls; history=len(e.history)
    b=e.interact(p.instruction,task_plan=p.plan,operation_id='open-once')
    assert b['idempotent_replay'] and a['turn_id']==b['turn_id']
    assert e.renderer.calls==count and len(e.history)==history
    with pytest.raises(ValueError): e.interact('different',task_plan=p.plan,operation_id='open-once')


def test_last_step_visual_failure_does_not_finish_or_repeat(tmp_path):
    e=picked_engine(tmp_path); e.observer=Observer()
    t=e.task=GoalTask('open',GoalProposal(status='ready',steps=[step('open','door_a')],success_checks=['door_open']))
    assert t.advance(e)['rule_committed']
    assert t.status=='paused' and t.index==1 and e.pending_visual
    assert e.state.objects['door_a'].state=='open'
    blocked=e.interact('enter',task_plan=step('move_to','storage_room').plan)
    assert not blocked['valid'] and blocked['failure_type']=='visual_pending'
    assert e.state.player.location!='storage_room'
    e.resolve_visual('Human fixture acknowledgment, not model evaluation')
    t.resume(); t.advance(e)
    assert t.status=='completed_rule_goal'


def test_visual_retry_uses_pre_frame_without_reapplying_rule(tmp_path):
    e=picked_engine(tmp_path); e.observer=Observer()
    e.interact('open',task_plan=step('open','door_a').plan)
    before=e.get_state(); turns=len(e.history)
    result=e.retry_visual()
    assert result['rule_effect_reapplied'] is False and e.get_state()==before and len(e.history)==turns
    with pytest.raises(ValueError,match='一次'): e.retry_visual()


def test_critical_unknown_is_gated_but_mock_is_not_claimed_real(tmp_path):
    e=picked_engine(tmp_path); e.observer=Observer('uncertain'); e.renderer.kind='wan_fixture'
    result=e.interact('open',task_plan=step('open','door_a').plan)
    assert result['failure_type']=='visual_uncertain' and e.pending_visual


def test_replanning_reads_latest_state_and_keeps_goal(tmp_path):
    e=picked_engine(tmp_path)
    # Simulate a new goal whose stale next step is invalid, without mutating the state illegally.
    proposal=GoalProposal(status='ready',steps=[step('move_to','storage_room')],success_checks=['door_open'])
    repaired=GoalProposal(status='ready',steps=[step('open','door_a')],success_checks=['door_open'])
    class Provider:
        def complete(self,messages,**kwargs):
            context=json.loads(messages[-1]['content'])
            assert context['feedback']['required_success_checks']==['door_open']
            assert context['state']['inventory']['items']==['red_key']
            return repaired.model_dump_json()
    e.agent=LLMAgent(Provider()); t=e.task=GoalTask('open the door',proposal)
    assert not t.advance(e)['valid']
    assert t.replans==1 and t.status=='ready'
    assert t.advance(e)['valid'] and t.status=='completed_rule_goal'


def test_replan_cannot_change_goal_or_retry_forever(tmp_path):
    e=picked_engine(tmp_path)
    class Provider:
        def complete(self,*args,**kwargs):
            return GoalProposal(status='ready',steps=[step('observe','room')],success_checks=['key_owned']).model_dump_json()
    e.agent=LLMAgent(Provider())
    t=GoalTask('open',GoalProposal(status='ready',steps=[step('move_to','storage_room')],success_checks=['door_open']))
    for _ in range(2):
        t.advance(e); assert t.status=='paused'; t.resume()
    t.advance(e)
    assert t.replans==2 and t.status=='paused' and e.state.objects['door_a'].state=='locked'


def test_replace_goal_preserves_inventory_and_archives_old_plan(tmp_path):
    e=picked_engine(tmp_path)
    e.task=GoalTask('enter',GoalProposal(status='ready',steps=[step('open','door_a'),step('move_to','storage_room')],success_checks=['at_storage']))
    class Provider:
        def complete(self,*args,**kwargs):
            return GoalProposal(status='ready',steps=[step('inspect','left')],completion_mode='action_execution').model_dump_json()
    e.agent=LLMAgent(Provider()); t=e.replace_goal('别进去，看看左边')
    assert e.state.inventory.items==['red_key'] and len(e.task_history)==1
    t.advance(e)
    assert t.status=='completed_action_goal' and e.state.objects['door_a'].state=='locked'


def test_look_request_does_not_erase_previous_pending_effect(tmp_path):
    e=picked_engine(tmp_path); e.observer=Observer()
    e.interact('open',task_plan=step('open','door_a').plan)
    pending=e.pending_visual.copy(); e.observer=Observer('supported')
    t=GoalTask('look left',GoalProposal(status='ready',steps=[step('inspect','left')],completion_mode='action_execution'))
    t.advance(e)
    assert t.status=='completed_action_goal' and e.pending_visual==pending


def test_schema_repair_is_one_attempt_and_raw_errors_are_retained(tmp_path):
    from interactive.tasks import propose_goal
    e=picked_engine(tmp_path)
    class Provider:
        calls=0
        def complete(self,*args,**kwargs):
            self.calls+=1
            return '{"status":"ready","success_checks":["at_laboratory"]}'
    provider=Provider(); e.agent=LLMAgent(provider)
    with pytest.raises(ValueError): propose_goal(e,'find the key')
    assert provider.calls==2 and len(e.goal_planner_calls)==2
    assert not any(r['parsed'] for r in e.goal_planner_calls)


def test_preflight_failure_is_replanned_without_committing_actions(tmp_path):
    e=picked_engine(tmp_path); count=len(e.history)
    class Provider:
        def complete(self,*args,**kwargs):
            return GoalProposal(status='ready',steps=[step('open','door_a')],success_checks=['door_open']).model_dump_json()
    e.agent=LLMAgent(Provider())
    t=GoalTask('open',GoalProposal(status='ready',steps=[step('move_to','storage_room')],success_checks=['door_open']))
    t.prepare(e)
    assert t.status=='ready' and t.replans==1 and len(e.history)==count
    assert e.state.objects['door_a'].state=='locked'


def test_explicit_no_open_constraint_is_not_just_prompt_text(tmp_path):
    e=picked_engine(tmp_path)
    task=GoalTask('拿到钥匙，不要开门，不要进入储藏室',GoalProposal(status='ready',
        steps=[step('open','door_a')],success_checks=['key_owned']))
    with pytest.raises(ValueError,match='明确禁止'): task.preflight(e)
    assert task.advance(e) is None and task.failure_type=='constraint_violation'
    assert e.state.objects['door_a'].state=='locked'


def test_structured_repair_identifies_contact_and_duplicate_plan(tmp_path):
    e=InteractionEngine(tmp_path,renderer=Renderer())
    bad=GoalProposal(status='ready',steps=[step('talk','anna','distrust'),
        step('search','desk'),step('pick_up','red_key')],success_checks=['key_owned'])
    fixed=bad.model_copy(update={'steps':[step('approach','anna')]+bad.steps})
    class Provider:
        calls=0
        def complete(self,messages,**kwargs):
            self.calls+=1
            feedback=json.loads(messages[-1]['content'])['feedback']
            assert 'approach:anna' in feedback['failed_precondition']['one_step_repair_suggestions']
            assert not feedback['failed_precondition']['state_before_failed_step']['story']['flags'].get('met_anna')
            return (bad if self.calls==1 else fixed).model_dump_json()
    e.agent=LLMAgent(Provider()); task=GoalTask('找钥匙，不要开门',bad)
    before=e.get_state(); task.prepare(e)
    assert task.status=='ready' and task.replans==2
    assert task.replan_history[0]['repeated_failed_plan']
    assert e.get_state()==before and not e.history


def test_forbidden_contact_has_no_repair_candidate_or_greeting_bypass(tmp_path):
    e=InteractionEngine(tmp_path,renderer=Renderer())
    for s in (step('approach','anna'),step('talk','anna'),step('talk','anna','distrust')):
        task=GoalTask('找到钥匙，但不要接近 Anna',GoalProposal(status='ready',steps=[s],success_checks=['key_owned']))
        with pytest.raises(ValueError,match='明确禁止'): task.preflight(e)


def test_replacement_prompt_separates_old_goal_without_mutating_state(tmp_path):
    from interactive.tasks import goal_messages
    e=InteractionEngine(tmp_path,renderer=Renderer())
    e.state=e.state.apply_delta({'player':{'goal':'只看看左边'}})
    context=json.loads(goal_messages(e,'找到钥匙，不要开门')[1]['content'])
    assert context['previous_goal_history_only']=='只看看左边'
    assert context['state']['player']['goal']==context['user_goal']=='找到钥匙，不要开门'
    assert list(context)[-1]=='user_goal'
    assert e.state.player.goal=='只看看左边'


def test_explicit_direction_cannot_pass_as_generic_observation(tmp_path):
    e=InteractionEngine(tmp_path,renderer=Renderer())
    bad=GoalProposal(status='ready',steps=[step('observe','room')],completion_mode='action_execution')
    fixed=GoalProposal(status='ready',steps=[step('inspect','left')],completion_mode='action_execution')
    class Provider:
        def complete(self,messages,**kwargs):
            assert json.loads(messages[-1]['content'])['feedback']['required_observation_target']=='left'
            return fixed.model_dump_json()
    e.agent=LLMAgent(Provider())
    task=GoalTask('别继续之前的计划，只看看左边。',bad)
    with pytest.raises(ValueError,match='inspect:left'): task.preflight(e)
    task.prepare(e)
    assert task.status=='ready' and task.proposal.steps[0].plan.target=='left'
    assert not e.history
