"""Three authored story routes, presented without fabricating execution evidence.

UI selection only supplies a goal. Previews run the production rules on a copy;
they neither call the planner/renderer nor commit world, memory or inventory.
The optional fallback is exact-goal-only and requires a recorded model failure.
"""
from copy import deepcopy
import re

from .tasks import GoalTask, ToolGoalDraft, ToolGoalStep, check_goal


ROUTES = {
    'A': {
        'route_id': 'A', 'title': '相信 Anna · 合作调查',
        'goal': '走近 Anna，帮助 Anna，询问 Anna 密码，使用密码解锁终端，再进入核心区域，完成 Anna 路线。已经完成的动作不要重复。',
        'npc': ['Anna'], 'ending': 'ending_anna',
        'summary': '建立信任 → 得到密码 → 解锁核心区域 → 与 Anna 合作调查。',
        'full_tool_path': ['approach:anna', 'interact:anna:help', 'ask:anna:password',
                           'use:terminal', 'move_to:core_area'],
        'prerequisites': ['帮助前先见到 Anna', '信任达到 0.6 才能取得密码',
                          '知道密码才能解锁终端', '终端解锁后才能进入核心区域'],
        'key_events': ['meet_anna', 'trust_anna', 'get_password', 'core_area', 'ending_anna'],
        'completion_description': '规则节点 ending_anna 且 terminal=true；不等于已验证画面进入核心区域。',
    },
    'B': {
        'route_id': 'B', 'title': '不相信 Anna · 红钥匙调查',
        'goal': '我不相信 Anna，选择独自调查。搜索桌子，拿起红钥匙，用钥匙打开门，再进入储藏室，完成钥匙路线。已经完成的动作不要重复。',
        'npc': ['Anna'], 'ending': 'ending_key',
        'summary': '独自调查 → 搜索抽屉 → 拾取红钥匙 → 开门 → 储藏室线索。',
        'full_tool_path': ['approach:anna', 'talk:anna:distrust', 'search:desk',
                           'pick_up:red_key', 'open:door_a', 'move_to:storage_room'],
        'prerequisites': ['选择独自调查前先见到 Anna', 'B 路线搜索桌子才能在规则中发现钥匙',
                          '发现后才能拾取', '背包持有红钥匙才能开门', '门已打开才能进入储藏室'],
        'key_events': ['meet_anna', 'distrust_anna', 'find_red_key', 'key_picked_up',
                       'open_door', 'ending_key'],
        'completion_description': '规则节点 ending_key 且 terminal=true；拾取、开门和穿门画面仍需单独验证。',
    },
    'C': {
        'route_id': 'C', 'title': '公开日志 · Leo 证据调查',
        'goal': '检查终端，恢复公开事故日志，找到 Leo，向 Leo 询问安全日志，完成终端路线。已经完成的动作不要重复。',
        'npc': ['Leo'], 'ending': 'ending_terminal',
        'summary': '检查锁定终端 → 恢复公开日志 → 找到 Leo → 核对事故证据。',
        'full_tool_path': ['inspect:terminal', 'inspect:terminal', 'approach:leo', 'ask:leo:logs'],
        'prerequisites': ['第一次检查选择 C 路线', '第二次检查才恢复公开日志',
                          '公开日志提供 Leo 位置', '见到 Leo 后才能核对安全日志'],
        'key_events': ['inspect_terminal', 'retrieve_log', 'meet_leo', 'ending_terminal'],
        'completion_description': '规则节点 ending_terminal 且 terminal=true；不等于视觉身份或日志内容已被模型读出。',
    },
}


def route_catalog():
    """JSON-safe cards for UI; clicking a card must not start execution."""
    cards = deepcopy(list(ROUTES.values()))
    for card in cards:
        card.update(selection_effect='fill_goal_only', source='authored_story_configuration',
                    execution_authorized=False, visual_success_verified=False)
    return cards


def route_goal(route_id):
    if route_id not in ROUTES:
        raise ValueError('Unknown route; select A, B or C.')
    return ROUTES[route_id]['goal']


def _goal_key(text):
    # Whitespace/punctuation normalization only; not general intent recognition.
    return re.sub(r'[\s，,。；;.!！]+', '', text).casefold()


def _remaining_calls(state, route_id):
    flags = state.story.flags
    calls = []
    if route_id in ('A', 'B'):
        if not flags.get('met_anna'):
            calls.append('approach:anna')
        if route_id == 'A':
            if not flags.get('helped_anna'):
                calls.append('interact:anna:help')
            if not flags.get('password_known'):
                calls.append('ask:anna:password')
            if not flags.get('terminal_accessed'):
                calls.append('use:terminal')
            calls.append('move_to:core_area')
        else:
            if not flags.get('distrusted_anna'):
                calls.append('talk:anna:distrust')
            if not flags.get('found_key'):
                calls.append('search:desk')
            if 'red_key' not in state.inventory.items:
                calls.append('pick_up:red_key')
            if state.objects['door_a'].state != 'open':
                calls.append('open:door_a')
            calls.append('move_to:storage_room')
    else:
        if not flags.get('log_retrieved'):
            if state.story.node != 'inspect_terminal':
                calls.append('inspect:terminal')
            calls.append('inspect:terminal')
        if not flags.get('met_leo'):
            calls.append('approach:leo')
        calls.append('ask:leo:logs')
    return calls


def preview_route(engine, route_id, *, goal=None):
    """Return a current-state authored proposal after a strictly read-only preflight.

    Only the displayed canonical complete goal is accepted here. Arbitrary user
    requests and constraints must remain the real planner's responsibility.
    Caller must still explicitly authorize route selection and each execution.
    """
    expected = route_goal(route_id)
    text = expected if goal is None else goal
    result = dict(route_id=route_id, source='authored_contract_not_learned',
                  status='unsupported', reason='', proposal=None,
                  execution_authorized=False, visual_success_verified=False,
                  simulated_rule_ending=False, state_mutated=False,
                  requires_real_planner_first=True,
                  state_source='current_rule_snapshot_not_visual_truth')
    if not isinstance(text, str) or _goal_key(text) != _goal_key(expected):
        result['reason'] = 'not_exact_displayed_route_goal'
        return result
    state = engine.state
    ending = ROUTES[route_id]['ending']
    if check_goal(state, [ending]):
        result.update(status='completed_rule_only', rule_goal_complete=True)
        return result
    if state.story.terminal:
        result.update(status='blocked', reason='story_finished_other_route')
        return result
    if state.story.route not in (None, route_id):
        result.update(status='blocked', reason='incompatible_route_reset_required')
        return result
    calls = _remaining_calls(state, route_id)
    draft = ToolGoalDraft(status='ready', success_checks=[ending], steps=[
        ToolGoalStep(instruction='显式编写的路线技能：' + tool_id, tool_id=tool_id)
        for tool_id in calls])
    proposal = draft.compile()
    try:
        GoalTask(text, proposal).preflight(engine)
    except ValueError as exc:
        result.update(status='blocked', reason=str(exc))
        return result
    result.update(status='rule_plan_ready', simulated_rule_ending=True,
                  proposal=proposal.model_dump(mode='json'),
                  requires_execution_revalidation=True)
    return result


def authored_route_fallback(engine, text, model_failure):
    """Exact three-goal fallback, never a silent substitute for model planning.

    Returns provenance + JSON proposal, not a running task. The caller must keep
    the original failed model record, label planning_source and use normal task
    execution. No match, no real recorded failure, or blocked route => None.
    """
    if not isinstance(model_failure, dict) or not model_failure:
        return None
    if not isinstance(text, str):
        return None
    route_id = next((r for r, c in ROUTES.items() if _goal_key(c['goal']) == _goal_key(text)), None)
    if route_id is None:
        return None
    result = preview_route(engine, route_id, goal=text)
    if result['status'] != 'rule_plan_ready':
        return None
    result.update(planning_source='authored_skill_fallback',
                  model_failure=deepcopy(model_failure),
                  fallback_notice='模型计划未通过；使用显式编写的路线技能，非模型自主修复。')
    return result
