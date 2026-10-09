"""Opt-in authored spatial story rules; no renderer or visual success claims.

Legacy RuleEngine/StoryGraph remain unchanged. Install both classes together
only for a new spatial-profile session, not by rewriting a historical archive.
"""
from pathlib import Path

from .models import Plan, RuleResult, WorldState
from .rules import RuleEngine
from .story import StoryGraph


SPATIAL_PROFILE_ID = 'laboratory_spatial_v1'
ROOT = Path(__file__).resolve().parents[1]


class SpatialStoryGraph(StoryGraph):
    def __init__(self):
        super().__init__(ROOT / 'configs/story-spatial.yaml')


class SpatialRuleEngine(RuleEngine):
    """Finite authored proximity/door states, never inferred from pixels."""

    profile_id = SPATIAL_PROFILE_ID

    @staticmethod
    def _ok(reason, delta=None, event=None, npc=''):
        return RuleResult(valid=True, reason=reason, state_delta=delta or {},
                          story_event=event, npc_response=npc)

    @staticmethod
    def _no(reason, suggestion='先观察当前状态，再执行缺少的空间前置动作。'):
        return RuleResult(valid=False, reason=reason, fallback_suggestion=suggestion)

    @staticmethod
    def _near(state, target):
        return (state.player.location == 'laboratory'
                and state.player.interaction_target == target)

    def validate(self, state: WorldState, plan: Plan):
        if plan.needs_clarification or state.story.terminal:
            return super().validate(state, plan)
        action, target = plan.semantic_action, plan.target
        flags, route = state.story.flags, state.story.route
        in_lab = state.player.location in ('entrance', 'laboratory')

        if action == 'approach' and target in ('terminal', 'door_a'):
            if not in_lab:
                return self._no('目标位于外侧实验室，不能从房间内部远程接近。')
            name = '终端' if target == 'terminal' else '原门'
            return self._ok(f'规则位置已接近{name}；镜头是否到位须另行检查。',
                            {'player': {'location': 'laboratory',
                                        'interaction_target': target}})

        if action in ('explore', 'move_to') and target in ('room', 'laboratory', 'entrance'):
            if not in_lab:
                return self._no('当前空间路线不支持从内部直接跳回外侧实验室。')
            result = super().validate(state, plan)
            if result.valid:
                result.state_delta.setdefault('player', {})['interaction_target'] = None
            return result

        if target == 'anna' and action in ('approach', 'talk', 'ask', 'interact', 'follow'):
            if not in_lab:
                return self._no('Anna 在外侧实验室，不能隔着已进入的房间远程交谈。')
            result = super().validate(state, plan)
            if result.valid and action in ('approach', 'follow'):
                result.state_delta.setdefault('player', {}).update(
                    location='laboratory', interaction_target='anna')
            return result

        if target == 'terminal' and action in ('inspect', 'use', 'interact'):
            if not self._near(state, 'terminal'):
                return self._no('尚未接近终端，不能远程检查或输入密码。', '先走近终端。')
            if route == 'A':
                if action == 'inspect':
                    return self._ok('你在终端旁查看输入界面；检查本身不会输入密码或开门。')
                if not flags.get('password_known'):
                    return self._no('尚未获得终端密码。', '先向 Anna 询问密码。')
                if flags.get('terminal_accessed') and flags.get('door_authorized_by_password'):
                    return self._ok('密码已验证，原门已授权；不会重复提交授权。')
                door_state = 'open' if state.objects['door_a'].state == 'open' else 'unlocked'
                return self._ok('在终端输入密码，授权并解锁原门；尚未开门或进入核心区域。',
                    {'objects': {'terminal': {'state': 'unlocked'},
                                 'door_a': {'state': door_state}},
                     'story': {'flags': {'terminal_accessed': True,
                                         'door_authorized_by_password': True}}},
                    'terminal_unlocked')
            if route == 'B':
                return self._no('钥匙路线不使用终端权限。', '继续寻找钥匙，再接近原门。')
            if flags.get('log_retrieved'):
                return self._ok('公开日志已经恢复；已有通行权限以当前规则记录为准。')
            if state.story.node == 'inspect_terminal':
                # New authored C-route rule, not a model-discovered permission.
                return self._ok('恢复公开日志，并获得剧情编写的日志通行权限；Leo 位于门内储藏室。',
                    {'objects': {'door_a': {'state': 'unlocked'}},
                     'story': {'route': 'C', 'flags': {'log_retrieved': True,
                                                       'log_passage_authorized': True}}},
                    'retrieve_log')
            return self._ok('终端仍锁定，但可读取公开事故日志；再次检查以恢复日志。',
                            {'story': {'route': 'C'}}, 'inspect_terminal')

        if target == 'door_a' and action in ('use', 'open'):
            if not self._near(state, 'door_a'):
                return self._no('尚未接近原门，不能远程解锁或开门。', '先走到原门前。')
            door = state.objects['door_a'].state
            if route == 'A':
                if not (flags.get('terminal_accessed') and flags.get('door_authorized_by_password')):
                    return self._no('原门尚未获得密码路线授权。', '先接近终端并输入密码。')
                open_event = 'open_core_door'
            elif route == 'B':
                if 'red_key' not in state.inventory.items:
                    return self._no('没有红钥匙，不能解锁原门。', '搜索桌子并拾取红钥匙。')
                if action == 'use' and door == 'locked':
                    return self._ok('在原门前用红钥匙解锁；门仍未打开。',
                        {'objects': {'door_a': {'state': 'unlocked'}},
                         'story': {'flags': {'door_unlocked_by_key': True}}}, 'unlock_key_door')
                open_event = 'open_door'
            elif route == 'C':
                if not (flags.get('log_retrieved') and flags.get('log_passage_authorized')):
                    return self._no('尚未获得日志路线的剧情通行权限。', '先接近终端并恢复公开日志。')
                open_event = 'open_log_door'
            else:
                return self._no('尚未选择密码、钥匙或日志路线，不能凭空打开原门。')
            if door == 'open':
                return self._ok('原门已经打开；不会重复提交开门状态。')
            if door != 'unlocked':
                return self._no('原门仍锁定；解锁与开门是两个动作。',
                                '钥匙路线先使用红钥匙解锁；其他路线先取得终端授权。')
            if action == 'use':
                return self._ok('原门已经解锁；使用权限不会替代开门动作。')
            return self._ok('在原门前打开已解锁的门；尚未跨过门口。',
                {'objects': {'door_a': {'state': 'open'}},
                 'story': {'flags': {'door_opened': True}}}, open_event)

        if target == 'storage_room' and action == 'open':
            return self._no('开门对象是原门，不是储藏室区域；开门和进入必须分开。',
                            '先接近原门并开门，再进入储藏室。')

        if target in ('core_area', 'storage_room') and action in ('move_to', 'approach', 'follow'):
            if state.player.location == target and target == 'storage_room' and route == 'C':
                return self._ok('已经位于储藏室内；没有再次跨门或提交进入状态。')
            if not self._near(state, 'door_a'):
                return self._no('必须先走到原门前，不能直接跳入门内区域。', '先接近原门。')
            if state.objects['door_a'].state != 'open' or not flags.get('door_opened'):
                return self._no('原门还未打开，不能进入。', '先解锁并打开原门。')
            if target == 'core_area':
                if route != 'A' or not flags.get('door_authorized_by_password'):
                    return self._no('核心区域仅属于已获终端授权的密码路线。')
                event, reason = 'ending_anna', '跨过原门，进入核心区域，完成 Anna 路线。'
            elif route == 'B' and 'red_key' in state.inventory.items:
                event, reason = 'ending_key', '跨过原门，进入储藏室，完成钥匙路线。'
            elif route == 'C' and flags.get('log_passage_authorized'):
                event, reason = 'enter_storage_for_leo', '依据日志通行权限进入储藏室；现在才可以接近 Leo。'
            else:
                return self._no('该区域不属于当前已授权路线，不能改变剧情分支。')
            return self._ok(reason, {'player': {'location': target, 'interaction_target': None}}, event)

        if target == 'leo' and action in ('approach', 'talk', 'ask', 'interact'):
            if route != 'C' or state.player.location != 'storage_room':
                return self._no('Leo 在门内储藏室；未进入之前不能遇见或远程交谈。',
                                '按日志路线接近原门、开门并进入储藏室。')
            if not (flags.get('log_retrieved') and flags.get('log_passage_authorized')):
                return self._no('尚未恢复日志及其剧情通行权限。')
            if not flags.get('met_leo') and action != 'approach':
                return self._no('已进入储藏室，但尚未走近 Leo。', '先在室内接近 Leo。')
            if plan.dialogue_action not in (None, 'greet', 'logs'):
                return self._no('此路线的 Leo 对话只支持问候与事故日志。')
            result = super().validate(state, plan)
            if result.valid and action == 'approach':
                result.state_delta.setdefault('player', {})['interaction_target'] = 'leo'
            return result

        return super().validate(state, plan)
