"""Opt-in spatial planner schema and separately labelled authored contracts.

The real planner still supplies its own proposal. The authored exact-goal
fallback requires a recorded failure and never executes or changes state.
"""
from copy import deepcopy
import json
import re
from typing import Literal

from pydantic import Field

from .models import Plan
from .spatial_interactions import SPATIAL_PROFILE_ID, SpatialRuleEngine, SpatialStoryGraph
from .tasks import GoalProposal, GoalStep, GoalTask, ToolGoalDraft, ToolGoalStep, check_goal
from .tools import TOOL_CALLS


# Independent copy: the old typed planner schema and its scores are unchanged.
SPATIAL_TOOL_CALLS = tuple(TOOL_CALLS) + (
    ('approach', 'terminal', None), ('approach', 'door_a', None), ('use', 'door_a', None))
SPATIAL_TOOL_IDS = {':'.join(part for part in call if part): call for call in SPATIAL_TOOL_CALLS}
SpatialToolId = Literal[tuple(SPATIAL_TOOL_IDS)]


class SpatialToolGoalStep(ToolGoalStep):
    tool_id: SpatialToolId


class SpatialToolGoalDraft(ToolGoalDraft):
    steps: list[SpatialToolGoalStep] = Field(default_factory=list, max_length=8)

    def compile(self):
        steps = []
        for step in self.steps:
            action, target, dialogue = SPATIAL_TOOL_IDS[step.tool_id]
            steps.append(GoalStep(instruction=step.instruction,
                plan=Plan(player_intent=step.instruction, semantic_action=action,
                          target=target, dialogue_action=dialogue)))
        return GoalProposal(status=self.status, clarification=self.clarification,
            steps=steps, success_checks=self.success_checks, completion_mode=self.completion_mode)

SPATIAL_ROUTES = {
    'A': {
        'title': '信任 Anna：密码与核心区域',
        'source': 'authored_spatial_contract_not_learned',
        'checks': ['ending_anna', 'at_core', 'terminal_unlocked'],
        'steps': [
            ('走近 Anna', 'approach', 'anna', None),
            ('帮助 Anna', 'interact', 'anna', 'help'),
            ('询问终端密码', 'ask', 'anna', 'password'),
            ('走近终端', 'approach', 'terminal', None),
            ('在终端输入密码解锁原门', 'use', 'terminal', None),
            ('走到原门前', 'approach', 'door_a', None),
            ('打开已解锁的原门', 'open', 'door_a', None),
            ('进入核心区域', 'move_to', 'core_area', None),
        ],
    },
    'B': {
        'title': '独自调查：钥匙与储藏室',
        'source': 'authored_spatial_contract_not_learned',
        'checks': ['ending_key', 'at_storage', 'key_owned'],
        'steps': [
            ('走近 Anna', 'approach', 'anna', None),
            ('告诉 Anna 我要独自调查', 'talk', 'anna', 'distrust'),
            ('搜索桌子找到红钥匙', 'search', 'desk', None),
            ('拾取红钥匙', 'pick_up', 'red_key', None),
            ('走到原门前', 'approach', 'door_a', None),
            ('使用红钥匙解锁原门', 'use', 'door_a', None),
            ('打开已解锁的原门', 'open', 'door_a', None),
            ('进入储藏室', 'move_to', 'storage_room', None),
        ],
    },
    'C': {
        'title': '公开日志：通行权限与 Leo',
        'source': 'authored_spatial_contract_not_learned',
        'checks': ['ending_terminal', 'at_storage', 'public_logs', 'met_leo'],
        'steps': [
            ('走近终端', 'approach', 'terminal', None),
            ('在终端旁检查公开日志', 'inspect', 'terminal', None),
            ('恢复公开日志及剧情通行权限', 'inspect', 'terminal', None),
            ('走到原门前', 'approach', 'door_a', None),
            ('按日志通行权限打开原门', 'open', 'door_a', None),
            ('进入储藏室寻找 Leo', 'move_to', 'storage_room', None),
            ('在储藏室内走近 Leo', 'approach', 'leo', None),
            ('与 Leo 核对事故日志', 'ask', 'leo', 'logs'),
        ],
    },
}

_ROUTE_GOALS = {
    'A': '走近 Anna，帮助 Anna，询问终端密码，走近终端并输入密码，走到原门前，打开门，再进入核心区域，完成 Anna 路线。已经完成的动作不要重复。',
    'B': '我不相信 Anna，选择独自调查。搜索桌子，拿起红钥匙，走到原门前，用钥匙解锁原门，再开门进入储藏室，完成钥匙路线。已经完成的动作不要重复。',
    'C': '走近终端，检查并恢复公开事故日志，取得日志剧情通行权限，走到原门前，开门进入储藏室，在室内找到 Leo 并核对安全日志，完成终端路线。已经完成的动作不要重复。',
}
_ENDINGS = {'A': 'ending_anna', 'B': 'ending_key', 'C': 'ending_terminal'}


def spatial_route_goal(route_id):
    if route_id not in _ROUTE_GOALS:
        raise ValueError('Unknown spatial route; select A, B or C.')
    return _ROUTE_GOALS[route_id]


def spatial_route_catalog():
    """Selection fills a new-profile goal only, never runs a route/model."""
    cards = []
    for route_id, contract in SPATIAL_ROUTES.items():
        calls = [':'.join(part for part in (action, target, dialogue) if part)
                 for _, action, target, dialogue in contract['steps']]
        cards.append(dict(route_id=route_id, title=contract['title'],
            goal=spatial_route_goal(route_id), npc=['Leo'] if route_id == 'C' else ['Anna'],
            ending=_ENDINGS[route_id], summary=' → '.join(row[0] for row in contract['steps']),
            full_tool_path=calls, profile_id=SPATIAL_PROFILE_ID,
            source='authored_spatial_story_configuration', selection_effect='fill_goal_only',
            prerequisites=['接近终端才能检查或输入密码', '解锁、开门、跨门分开',
                           '进入储藏室之后才能遇见 Leo'],
            completion_description='规则结局，不等于物理接近、相同门开合或人物身份已视觉验证。',
            execution_authorized=False, visual_success_verified=False))
    return cards


def spatial_route_proposal(route):
    """Full new-session route, explicit authored fallback, not an LLM response.

It intentionally does not silently switch a legacy session to the new profile
or rewrite a historical plan. Preflight with the active spatial rules/graph.
"""
    contract = SPATIAL_ROUTES[route]
    return GoalProposal(status='ready', steps=[
        GoalStep(instruction=text, plan=Plan(player_intent=text, semantic_action=action,
            target=target, dialogue_action=dialogue,
            reason_summary='显式编写的空间路线合同；不是模型生成的规划。'))
        for text, action, target, dialogue in contract['steps']],
        success_checks=contract['checks'])


def spatial_tool_context(state):
    """Read-only affordances for the new profile; cannot execute world effects."""
    rules, graph = SpatialRuleEngine(), SpatialStoryGraph()
    rows = []
    for action, target, dialogue in SPATIAL_TOOL_CALLS:
        result = rules.validate(state, Plan(player_intent='spatial_precondition_query',
            semantic_action=action, target=target, dialogue_action=dialogue))
        allowed, reason = result.valid, result.reason
        if allowed:
            try:
                graph.transition(state.apply_delta(result.state_delta), result.story_event)
            except ValueError as exc:
                allowed, reason = False, str(exc)
        rows.append({'call': [action, target, dialogue], 'allowed_now': allowed,
                     'rule_reason': reason, 'story_event': result.story_event,
                     'profile_id': SPATIAL_PROFILE_ID, 'visual_success_verified': False})
    return rows


def spatial_goal_messages(engine, text, feedback=None):
    """Independent model prompt/schema; never append to the legacy route prompt."""
    if not text.strip() or len(text) > 2000:
        raise ValueError('请输入不超过 2000 字符的目标。')
    prompt = (
        'You plan remaining actions in the opt-in laboratory_spatial_v1 authored world. '
        'Return ONLY one JSON SpatialToolGoalDraft using enumerated tool_id values. '
        'The language model proposes actions; it cannot change state or declare visual success. '
        'Plan the COMPLETE remaining sequence toward the actual user goal, at most 8 steps. '
        'Current rule state is authoritative for prerequisites, never pixels or the previous goal. '
        'player.interaction_target is rule proximity, NOT a verified camera pose. '
        'Only approach:terminal establishes proximity to the terminal; inspect/use require it. '
        'Only approach:door_a establishes proximity to the original door; use/open and entry require it. '
        'Approaching a different object replaces the interaction target. Inspecting the door does NOT approach it. '
        'Unlocking does NOT open the door, and opening does NOT enter its room. '
        'Route A: approach:anna if not met, interact:anna:help if not helped, ask:anna:password if not known, '
        'approach:terminal, use:terminal to authorize/unlock door_a, approach:door_a, open:door_a, '
        'move_to:core_area. This is exactly 8 steps from a fresh state; no extra explore is required. '
        'inspect:terminal on route A only examines the interface and does NOT enter a password. '
        'Route B: approach:anna if not met, talk:anna:distrust, search:desk, pick_up:red_key, '
        'approach:door_a, use:door_a to unlock with the key, open:door_a, move_to:storage_room. '
        'A locked door cannot be opened directly even when the key is in inventory. '
        'Route C: approach:terminal, inspect:terminal twice to retrieve public logs and the explicitly '
        'AUTHORED log_passage_authorized permission, approach:door_a, open:door_a, '
        'move_to:storage_room, approach:leo, ask:leo:logs. Do NOT switch C into the key route. '
        'C does not require or create a red_key, and public logs do NOT unlock the core terminal. '
        'Leo is inside storage_room. Knowing his location or opening a door does not meet him. '
        'Route C enters storage_room without ending; only the later Leo dialogue reaches ending_terminal. '
        'state.story.route=null means no route has been selected; seeing or meeting Anna alone does not choose A/B. '
        'Honor explicit route preferences and prohibitions. Never change an established route. '
        'Already committed effects must not be repeated. Keep any still-needed approach step from the actual '
        'current interaction_target; do not assume an old completed approach still locates the player there. '
        'Only broader user goals authorize missing prerequisites. If the user forbids necessary steps or '
        'requests unsupported worlds/objects, return needs_clarification without executable steps. '
        'Use only enumerated success_checks, which are rule-state checks, not visual evidence. '
        'Observation-only inspect/wait requests may use action_execution with no success_checks, '
        'without claiming the requested viewing angle was visually verified. '
        'When feedback is supplied, preserve required_success_checks and required_completion_mode exactly; '
        'repair only the remaining sequence toward ORIGINAL user_goal. Never repeat rejected_plan or '
        'previous_rejected_plans unchanged, replay committed effects, or treat repair suggestions as authorization. '
        'No long reasoning text. Schema: ' + json.dumps(SpatialToolGoalDraft.model_json_schema()))
    state = engine.get_state()
    previous_goal = state['player']['goal']
    state['player']['goal'] = text
    context = {
        'profile_id': SPATIAL_PROFILE_ID,
        'previous_goal_history_only': previous_goal, 'state': state,
        'observation': engine.observation.model_dump(), 'memory': engine.memory.retrieve(),
        'supported_calls': SPATIAL_TOOL_CALLS,
        'skills': spatial_route_catalog(),
        'feedback': feedback, 'pending_visual': getattr(engine, 'pending_visual', None),
        'current_tool_preconditions': spatial_tool_context(engine.state),
        'request_priority': 'Only user_goal is the current request; history, cards and observations do not authorize other actions.',
        'user_goal': text,
    }
    return [{'role': 'system', 'content': prompt},
            {'role': 'user', 'content': json.dumps(context, ensure_ascii=False)}]


def _goal_key(text):
    return re.sub(r'[\s，,。；;.!！]+', '', text).casefold()


def _step_done(state, action, target, dialogue, route_id):
    """Committed effects/proximity only, not unsupported natural-language parsing."""
    flags = state.story.flags
    if target == 'anna':
        return bool((action == 'approach' and flags.get('met_anna'))
            or (dialogue == 'help' and flags.get('helped_anna'))
            or (dialogue == 'password' and flags.get('password_known'))
            or (dialogue == 'distrust' and flags.get('distrusted_anna')))
    if action == 'search':
        return bool(flags.get('found_key'))
    if action == 'pick_up':
        return 'red_key' in state.inventory.items
    terminal_done = ((route_id == 'A' and flags.get('terminal_accessed')
                      and flags.get('door_authorized_by_password'))
                     or (route_id == 'C' and flags.get('log_retrieved')
                         and flags.get('log_passage_authorized')))
    if target == 'terminal':
        return bool(terminal_done or (action == 'approach'
            and state.player.location == 'laboratory' and state.player.interaction_target == target))
    if target == 'door_a':
        if route_id == 'C' and state.player.location == 'storage_room':
            return True
        if action == 'approach':
            return state.player.location == 'laboratory' and state.player.interaction_target == target
        if action == 'use':
            return state.objects[target].state in ('unlocked', 'open')
        return state.objects[target].state == 'open' and bool(flags.get('door_opened'))
    if target in ('storage_room', 'core_area'):
        return state.player.location == target
    if target == 'leo' and action == 'approach':
        return bool(flags.get('met_leo'))
    return False


def spatial_preview_route(engine, route_id, *, goal=None):
    expected = spatial_route_goal(route_id)
    text = expected if goal is None else goal
    result = dict(route_id=route_id, profile_id=SPATIAL_PROFILE_ID,
        source='authored_spatial_contract_not_learned', status='unsupported', reason='',
        proposal=None, execution_authorized=False, visual_success_verified=False,
        simulated_rule_ending=False, state_mutated=False, requires_real_planner_first=True,
        state_source='current_rule_snapshot_not_visual_truth')
    if not isinstance(text, str) or _goal_key(text) != _goal_key(expected):
        result['reason'] = 'not_exact_displayed_spatial_route_goal'
        return result
    if not isinstance(engine.rules, SpatialRuleEngine) or not isinstance(engine.graph, SpatialStoryGraph):
        result.update(status='blocked', reason='spatial_profile_not_installed')
        return result
    state = engine.state
    if check_goal(state, [_ENDINGS[route_id]]):
        result.update(status='completed_rule_only', rule_goal_complete=True)
        return result
    if state.story.terminal or state.story.route not in (None, route_id):
        result.update(status='blocked', reason='incompatible_route_reset_required')
        return result
    candidate = state.model_copy(deep=True)
    steps = []
    for text_step, action, target, dialogue in SPATIAL_ROUTES[route_id]['steps']:
        if _step_done(candidate, action, target, dialogue, route_id):
            continue
        item = GoalStep(instruction='显式编写的空间路线技能：' + text_step,
            plan=Plan(player_intent=text_step, semantic_action=action,
                      target=target, dialogue_action=dialogue))
        validity = engine.rules.validate(candidate, item.plan)
        if not validity.valid:
            result.update(status='blocked', reason=validity.reason)
            return result
        try:
            candidate = engine.graph.transition(candidate.apply_delta(validity.state_delta), validity.story_event)
        except ValueError as exc:
            result.update(status='blocked', reason=str(exc))
            return result
        steps.append(item)
    proposal = GoalProposal(status='ready', steps=steps, success_checks=[_ENDINGS[route_id]])
    try:
        GoalTask(text, proposal).preflight(engine)
    except ValueError as exc:
        result.update(status='blocked', reason=str(exc))
        return result
    result.update(status='rule_plan_ready', simulated_rule_ending=True,
                  proposal=proposal.model_dump(mode='json'), requires_execution_revalidation=True)
    return result


def spatial_authored_route_fallback(engine, text, model_failure):
    if not isinstance(model_failure, dict) or not model_failure or not isinstance(text, str):
        return None
    route_id = next((r for r in SPATIAL_ROUTES
                     if _goal_key(spatial_route_goal(r)) == _goal_key(text)), None)
    if route_id is None:
        return None
    result = spatial_preview_route(engine, route_id, goal=text)
    if result['status'] != 'rule_plan_ready':
        return None
    result.update(planning_source='authored_skill_fallback', model_failure=deepcopy(model_failure),
                  fallback_notice='模型计划未通过；使用显式编写的空间路线技能，非模型自主修复。')
    return result
