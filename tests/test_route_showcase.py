"""CPU-only rule previews; none of these tests produce a model/video success."""
from types import SimpleNamespace
import pytest
from interactive.models import WorldState
from interactive.rules import RuleEngine
from interactive.story import StoryGraph
from interactive.tasks import GoalProposal, GoalTask, check_goal
from interactive.route_showcase import route_catalog, route_goal, preview_route, authored_route_fallback


def engine(state=None):
    return SimpleNamespace(state=state or WorldState(), rules=RuleEngine(), graph=StoryGraph())


def simulate(e, steps):
    """Explicit rule simulation in test, not a model, renderer or committed run."""
    state = e.state.model_copy(deep=True)
    for step in steps:
        result = e.rules.validate(state, step.plan)
        assert result.valid
        state = e.graph.transition(state.apply_delta(result.state_delta), result.story_event)
    return state


@pytest.mark.parametrize('route,count,ending', [('A', 5, 'ending_anna'), ('B', 6, 'ending_key'), ('C', 4, 'ending_terminal')])
def test_fresh_route_preflight_is_read_only(route, count, ending):
    e = engine()
    before = e.state.serialize()
    result = preview_route(e, route)
    assert result['status'] == 'rule_plan_ready'
    proposal = GoalProposal.model_validate(result['proposal'])
    assert len(proposal.steps) == count <= 8
    assert proposal.success_checks == [ending]
    assert check_goal(simulate(e, proposal.steps), [ending])
    assert e.state.serialize() == before
    assert not result['execution_authorized'] and not result['visual_success_verified']


def test_catalog_is_independent_and_ui_is_fill_only():
    cards = route_catalog()
    assert [x['route_id'] for x in cards] == ['A', 'B', 'C']
    assert all(x['selection_effect'] == 'fill_goal_only' for x in cards)
    cards[0]['goal'] = 'changed'
    assert route_goal('A') != 'changed'
    with pytest.raises(ValueError):
        route_goal('unknown')


def test_c_continuation_keeps_required_second_inspection_only():
    e = engine()
    full = GoalProposal.model_validate(preview_route(e, 'C')['proposal'])
    e.state = simulate(e, full.steps[:1])
    left = GoalProposal.model_validate(preview_route(e, 'C')['proposal'])
    assert [(s.plan.semantic_action, s.plan.target) for s in left.steps] == [
        ('inspect', 'terminal'), ('approach', 'leo'), ('ask', 'leo')]
    e.state = simulate(e, left.steps[:1])
    assert len(preview_route(e, 'C')['proposal']['steps']) == 2


def test_remaining_a_b_skip_committed_effects():
    for route, prefix in [('A', 3), ('B', 4)]:
        e = engine()
        full = GoalProposal.model_validate(preview_route(e, route)['proposal'])
        e.state = simulate(e, full.steps[:prefix])
        remaining = preview_route(e, route)
        assert len(remaining['proposal']['steps']) == len(full.steps) - prefix
        assert remaining['proposal']['steps'][0]['plan']['semantic_action'] in ('use', 'open')


def test_route_conflict_and_terminal_do_not_switch_state():
    e = engine(WorldState().apply_delta({'story': {'route': 'B'}}))
    assert preview_route(e, 'A')['reason'] == 'incompatible_route_reset_required'
    e = engine()
    e.state = simulate(e, GoalProposal.model_validate(preview_route(e, 'A')['proposal']).steps)
    assert preview_route(e, 'A')['status'] == 'completed_rule_only'
    assert preview_route(e, 'C')['reason'] == 'story_finished_other_route'


def test_unusual_user_constraints_do_not_enter_authored_fallback():
    e = engine()
    assert authored_route_fallback(e, route_goal('A'), None) is None
    assert authored_route_fallback(e, route_goal('A'), {}) is None
    assert authored_route_fallback(e, route_goal('A') + '不要接近 Anna。', {'error': 'model failed'}) is None
    assert preview_route(e, 'B', goal='拿到钥匙，不要开门')['status'] == 'unsupported'


def test_fallback_retains_model_failure_and_no_execution_authority():
    e = engine()
    failure = {'status': 'needs_clarification', 'raw_response': 'original model refusal'}
    result = authored_route_fallback(e, route_goal('C'), failure)
    assert result['planning_source'] == 'authored_skill_fallback'
    assert result['model_failure'] == failure
    assert not result['execution_authorized']
    GoalTask(route_goal('C'), GoalProposal.model_validate(result['proposal'])).preflight(e)
    result['model_failure']['raw_response'] = 'edited copy'
    assert failure['raw_response'] == 'original model refusal'


def test_inconsistent_snapshot_is_not_fixed_or_injected():
    e = engine(WorldState().apply_delta({'story': {'route': 'A', 'node': 'enter_lab',
                                               'flags': {'met_anna': True}}}))
    before = e.state.serialize()
    assert preview_route(e, 'A')['status'] == 'blocked'
    assert e.state.serialize() == before
