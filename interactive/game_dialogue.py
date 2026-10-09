"""Read-only game dialogue presentation, never a planner or world-state writer.

The current engine's ``npc_response`` is authored by RuleEngine.  An action
intent, a rule reason, or VLM prose must never be presented as an NPC quotation.
This JSON view is also suitable for offline captions; it supplies no audio,
lip-sync, or frame-level timing claim.
"""
from __future__ import annotations

from html import escape
import re


ROUTE_TITLES = {'A': '相信 Anna', 'B': '独自调查 · 红钥匙', 'C': '终端日志 · Leo'}
SOURCE_LABELS = {
    'player_input': '玩家输入',
    'goal_tool_instruction': '计划动作 · 非玩家原话',
    'authored_rule_dialogue': '已编写剧情台词',
    'model_npc_response': '模型回复 · 未验证',
    'unverified_npc_response': '回复来源未验证',
    'rule_narration': '规则旁白 · 非画面认证',
    'rule_rejection': '规则拒绝 · 状态未推进',
    'authored_scene_intro': '开场说明',
    'authored_ending_label': '规则结局 · 非视觉认证',
}

# Scoped styles can be appended to the existing Gradio Blocks CSS.  No script
# or external resources are required; all dynamic content is escaped below.
GAME_DIALOGUE_CSS = """
.iw-dialogue-stage{color:#eef5ff;background:linear-gradient(145deg,#111e30,#18283e);
border:1px solid #748baa;border-radius:12px;padding:14px 18px;box-shadow:0 8px 24px #0004}
.iw-dialogue-stage .iw-dialogue-header{display:flex;gap:10px;justify-content:space-between;
font-size:12px;color:#aebed4;margin-bottom:10px;flex-wrap:wrap}
.iw-dialogue-stage .iw-dialogue-row{display:flex;gap:10px;padding:7px 0;align-items:flex-start}
.iw-dialogue-stage .iw-dialogue-speaker{font-weight:700;min-width:52px;color:#b7d8ff}
.iw-dialogue-stage .iw-dialogue-row[data-role="npc"] .iw-dialogue-speaker{color:#ffdf9c}
.iw-dialogue-stage .iw-dialogue-row[data-role="narrator"] .iw-dialogue-speaker{color:#b2c0d4}
.iw-dialogue-stage .iw-dialogue-text{white-space:pre-wrap;line-height:1.65;font-size:16px;overflow-wrap:anywhere}
.iw-dialogue-stage .iw-dialogue-source{display:block;font-size:10px;color:#97a9c2;margin-top:2px}
.iw-dialogue-stage .iw-dialogue-footer{color:#91a5bf;font-size:11px;margin-top:10px;border-top:1px solid #ffffff17;padding-top:8px}
.iw-dialogue-stage details{margin-top:10px;border-top:1px solid #ffffff17;padding-top:8px}
.iw-dialogue-stage summary{cursor:pointer;color:#c3d4eb;font-size:12px}
.iw-dialogue-stage .iw-dialogue-history-group{border-top:1px solid #ffffff13;margin-top:10px;padding-top:7px}
.iw-dialogue-stage .iw-dialogue-history-group .iw-dialogue-text{font-size:13px}
"""


def _mapping(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    return {}


def _session_view(session):
    if isinstance(session, dict):
        return session
    # Do not call export_session(): rendering UI must not write files or change
    # the history.  All fields below are already held by InteractionEngine.
    return {
        'session_id': getattr(session, 'session_id', None),
        'state': session.get_state() if hasattr(session, 'get_state') else {},
        'turns': session.get_history() if hasattr(session, 'get_history') else [],
        'memory': session.memory.retrieve() if hasattr(session, 'memory') else {},
    }


def _message(turn_id, speaker, text, source, reference, *, role='narrator',
             route=None, timestamp=None, raw_text=None, event=None, committed=False):
    return {
        'turn_id': turn_id, 'speaker': speaker, 'role': role, 'text': text,
        'source': source, 'source_reference': reference,
        'source_label': SOURCE_LABELS[source], 'route': route,
        'timestamp': timestamp, 'time_basis': 'turn_boundary',
        'video_time_seconds': None,
        'raw_text': raw_text if raw_text is not None else text,
        'story_event': event, 'rule_committed': committed,
        'visual_success_verified': False, 'audio_generated': False,
        'lip_sync_verified': False,
    }


def _npc_line(row):
    raw = row.get('npc_response')
    if not isinstance(raw, str) or not raw.strip():
        return None
    raw = raw.strip()
    match = re.match(r'^(Anna|Leo)\s*[:：]\s*(.*)$', raw, flags=re.I | re.S)
    target = str(_mapping(row.get('plan')).get('target', '')).lower()
    speaker = (match.group(1).title() if match else
               {'anna': 'Anna', 'leo': 'Leo'}.get(target, 'NPC'))
    text = match.group(2).strip() if match else raw
    # Current engine has no LLM-generated NPC response path.  Preserve an
    # explicit future source declaration, but do not infer one from agent_kind.
    declared = row.get('npc_response_source')
    source = ('model_npc_response' if declared in ('language_model', 'model_npc_response')
              else 'authored_rule_dialogue' if declared in (None, 'rule', 'authored_rule_dialogue')
              else 'unverified_npc_response')
    return speaker, text, raw, source


def build_game_dialogue(session, turn_id=None):
    """Extract exact committed dialogue plus separately labelled narration.

    ``session`` may be an InteractionEngine or its existing JSON export.  Pass
    ``turn_id`` when showing an archived turn/video; later dialogue is then
    excluded.  Missing turn IDs in short-term memory remain ``None`` instead of
    inventing a video correspondence.  The returned structure is JSON-safe.
    """
    if turn_id is not None and (type(turn_id) is not int or turn_id < 0):
        raise ValueError('turn_id must be a nonnegative integer')
    data = _session_view(session)
    state = _mapping(data.get('state'))
    story = _mapping(state.get('story'))
    route = story.get('route')
    rows = data.get('turns') or []
    from_memory = False
    if not rows:
        rows = _mapping(data.get('memory')).get('recent') or []
        from_memory = bool(rows)
    groups = []
    for index, value in enumerate(rows):
        row = _mapping(value)
        current_id = row.get('turn_id')
        if turn_id is not None and (current_id is None or current_id > turn_id):
            continue
        prefix = f'memory.recent[{index}]' if from_memory else f'turns[{index}]'
        row_story = _mapping(_mapping(row.get('state')).get('story'))
        row_route = row_story.get('route')
        committed = bool(row.get('rule_committed', row.get('valid', False)))
        timestamp = row.get('timestamp') or row.get('created_at')
        common = {'route': row_route, 'timestamp': timestamp,
                  'event': row.get('story_event'), 'committed': committed}
        messages = []
        player_text = row.get('player_text')
        if isinstance(player_text, str) and player_text.strip():
            tool_instruction = row.get('plan_source') == 'reviewed_goal_plan'
            source = 'goal_tool_instruction' if tool_instruction else 'player_input'
            messages.append(_message(current_id, '行动' if tool_instruction else '玩家',
                player_text.strip(), source, prefix + '.player_text',
                role='action' if tool_instruction else 'player', **common))
        reason = row.get('reason')
        if isinstance(reason, str) and reason.strip():
            messages.append(_message(current_id, '旁白' if committed else '系统',
                reason.strip(), 'rule_narration' if committed else 'rule_rejection',
                prefix + '.reason', role='narrator' if committed else 'system', **common))
        npc = _npc_line(row) if committed else None
        if npc:
            speaker, text, raw, source = npc
            messages.append(_message(current_id, speaker, text, source,
                prefix + '.npc_response', role='npc', raw_text=raw, **common))
        if messages:
            groups.append({'turn_id': current_id, 'route': row_route,
                'timestamp': timestamp, 'story_event': row.get('story_event'),
                'rule_committed': committed, 'messages': messages})
    if turn_id is not None:
        # Selecting a historical video must use that turn's story, not the
        # final session route/terminal that may not have existed at that time.
        selected = next((r for r in rows if _mapping(r).get('turn_id') == turn_id), None)
        if turn_id == 0:
            before = _mapping(_mapping(rows[0]).get('previous_state')) if rows else {}
            selected_story = _mapping(before.get('story')) or {'node': 'enter_lab', 'terminal': False, 'route': None}
        elif selected is None:
            raise ValueError('Requested turn is not present in the dialogue archive')
        else:
            selected_story = _mapping(_mapping(_mapping(selected).get('state')).get('story'))
        story = selected_story
        route = selected_story.get('route')
    terminal = bool(story.get('terminal'))
    if not groups:
        text = ('本路线的规则结局已完成；视频事件仍需单独核实。' if terminal
                else '你来到一间废弃实验室。请用文字描述你的行动。')
        source = 'authored_ending_label' if terminal else 'authored_scene_intro'
        initial = _message(0, '旁白', text, source, 'state.story', route=route,
                           event=story.get('node'))
        groups = [{'turn_id': 0, 'route': route, 'timestamp': None,
                   'story_event': story.get('node'), 'rule_committed': False,
                   'messages': [initial]}]
    messages = [message for group in groups for message in group['messages']]
    return {
        'schema': 'game-dialogue/v1', 'session_id': data.get('session_id'),
        'route': route, 'route_title': ROUTE_TITLES.get(route, '路线未选择'),
        'story_node': story.get('node', 'enter_lab'), 'terminal': terminal,
        'groups': groups, 'messages': messages, 'latest': groups[-1],
        'history_source': 'short_term_memory' if from_memory else 'session_turns',
        'npc_dialogue_origin': 'current RuleEngine-authored dialogue; not inferred from video',
        'visual_success_verified': False, 'audio_generated': False,
        'lip_sync_verified': False,
    }


def render_dialogue_html(payload, show_history=True, max_history_groups=30):
    """Safe, script-free game dialogue box for Gradio HTML or offline reports."""
    def safe(value):
        return escape(str(value if value is not None else ''), quote=True)

    def render_group(group):
        rows = []
        for message in group.get('messages', []):
            role = message.get('role')
            # Do not inject an arbitrary attribute/CSS selector from archive.
            role = role if role in ('player', 'npc', 'action', 'narrator', 'system') else 'system'
            rows.append(
                f'<div class="iw-dialogue-row" data-role="{role}">'
                f'<span class="iw-dialogue-speaker">{safe(message.get("speaker"))}</span>'
                f'<div><div class="iw-dialogue-text">{safe(message.get("text"))}</div>'
                f'<span class="iw-dialogue-source" title="{safe(message.get("source_reference"))}">'
                f'{safe(message.get("source_label"))}</span></div></div>')
        return ''.join(rows)

    latest = payload.get('latest') or {}
    current_id = latest.get('turn_id')
    turn_label = '开场' if current_id == 0 else f'第 {current_id} 轮' if current_id is not None else '记忆回看'
    terminal = ' · 规则结局' if payload.get('terminal') else ''
    result = [
        '<section class="iw-dialogue-stage" aria-label="游戏对话框">',
        '<div class="iw-dialogue-header">',
        f'<span>{safe(payload.get("route_title"))}{terminal}</span><span>{safe(turn_label)}</span></div>',
        render_group(latest),
        '<div class="iw-dialogue-footer">玩家输入、台词来源与规则旁白分别标注；不代表画面动作验证、语音或口型同步。</div>',
    ]
    history = payload.get('groups', [])[:-1]
    if show_history and history:
        count = max(0, min(int(max_history_groups), 100))
        visible = history[-count:] if count else []
        result.append(f'<details><summary>对话历史 · {len(history)} 轮</summary>')
        for group in visible:
            label = group.get('turn_id')
            label = f'第 {label} 轮' if label is not None else '记忆条目'
            result.append('<div class="iw-dialogue-history-group">'
                f'<div class="iw-dialogue-source">{safe(label)}</div>{render_group(group)}</div>')
        result.append('</details>')
    result.append('</section>')
    return ''.join(result)
