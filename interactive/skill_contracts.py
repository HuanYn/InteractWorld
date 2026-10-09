"""Authored skill compilation and read-only preflight; never model/visual evidence.

Kept outside the frozen Planner prompt. A preview is not execution permission:
the normal task engine must revalidate current state, constraints and visual gates.
"""
from .tasks import GoalTask, ToolGoalDraft, ToolGoalStep, check_goal
from .tools import available_tools


CONTRACTS = {
    'retrieve_key': ('B', 'key_owned'),
    'negotiate_password': ('A', 'password_known'),
    'unlock_terminal': ('A', 'terminal_unlocked'),
}


def preview_skill(engine, name, *, goal='', allow_route_selection=False):
    """Compile remaining actions on the supplied snapshot without committing it.

    Callers must pass the actual user goal to preserve supported prohibitions.
    An empty goal permits inspection only, not handing the proposal to execution.
    Route selection requires an explicit caller decision, never inferred here.
    """
    result = dict(skill=name, source='authored_contract_not_learned',
                  status='unsupported', reason='', proposal=None,
                  rule_goal_complete=False, simulated_goal_reachable=False,
                  visual_goal_verified=False, execution_authorized=False,
                  state_source='current_rule_snapshot_not_visual_truth')
    if name not in CONTRACTS:
        result['reason'] = 'unknown_skill'
        return result
    route, check = CONTRACTS[name]
    state = engine.state
    if check_goal(state, [check]):
        result.update(status='completed_rule_only', rule_goal_complete=True)
        return result
    if state.story.terminal:
        result.update(status='blocked', reason='story_finished')
        return result
    if state.story.route not in (None, route):
        result.update(status='blocked', reason='incompatible_route')
        return result
    if state.story.route is None and not allow_route_selection:
        result.update(status='needs_clarification', reason='route_selection_not_authorized')
        return result
    flags = state.story.flags
    calls = []
    if name == 'unlock_terminal':
        # This skill does not silently expand into negotiating with Anna.
        if state.story.route != 'A' or not flags.get('password_known'):
            result.update(status='blocked', reason='missing_password_or_route_A')
            return result
        calls = ['use:terminal']
    else:
        if not flags.get('met_anna'):
            calls.append('approach:anna')
        if name == 'retrieve_key':
            if state.story.route is None:
                calls.append('talk:anna:distrust')
            if not flags.get('found_key'):
                calls.append('search:desk')
            calls.append('pick_up:red_key')
        else:
            if not flags.get('helped_anna'):
                calls.append('interact:anna:help')
            calls.append('ask:anna:password')
    proposal = ToolGoalDraft(status='ready', success_checks=[check], steps=[
        ToolGoalStep(instruction=call, tool_id=call) for call in calls]).compile()
    task = GoalTask(goal or 'Read-only authored skill preview', proposal)
    try:
        task.preflight(engine)
    except ValueError as exc:
        result.update(status='blocked', reason=str(exc))
        return result
    result.update(status='rule_plan_ready', simulated_goal_reachable=True,
                  proposal=proposal.model_dump(mode='json'),
                  requires_current_goal_and_execution_revalidation=True)
    return result


def immediate_affordances(state):
    """Reuse production rule+story checks, not a second hand-written rule set."""
    return dict(scope='immediate_rule_legal_not_future_plan_or_visual_capability',
                tools=available_tools(state), visual_goal_verified=False,
                execution_authorized=False)
