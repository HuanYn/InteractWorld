"""Presentation only: overlay exact dialogue, never replace a live world."""
from html import escape
import json
from pathlib import Path

from .game_dialogue import build_game_dialogue, render_dialogue_html, GAME_DIALOGUE_CSS


GAME_STAGE_CSS = GAME_DIALOGUE_CSS + """
#iw-game-stage,#iw-game-screen,#iw-archive-stage{position:relative;isolation:isolate;gap:0!important}
#iw-game-stage video,#iw-archive-stage video{max-height:62vh;object-fit:contain;background:#07101a}
#iw-game-dialogue,#iw-archive-dialogue{position:absolute!important;bottom:45px;left:9%;right:9%;
width:82%!important;z-index:8;background:transparent!important;border:none!important;
padding:0!important;pointer-events:none;max-height:43%;overflow:hidden}
#iw-game-dialogue .iw-dialogue-stage,#iw-archive-dialogue .iw-dialogue-stage{
background:linear-gradient(110deg,#07101aef,#12243ae8);border:1px solid #ad915c;
border-left:4px solid #e0ba72;border-radius:5px;padding:9px 16px;box-shadow:0 3px 12px #0008}
#iw-game-dialogue .iw-dialogue-header,#iw-archive-dialogue .iw-dialogue-header{margin-bottom:2px;font-size:10px}
#iw-game-dialogue .iw-dialogue-row,#iw-archive-dialogue .iw-dialogue-row{padding:2px 0}
#iw-game-dialogue .iw-dialogue-text,#iw-archive-dialogue .iw-dialogue-text{font-size:17px;line-height:1.45;color:#f4f7ff!important}
#iw-game-dialogue .iw-dialogue-header,#iw-archive-dialogue .iw-dialogue-header{color:#b5c6db!important}
#iw-game-dialogue .iw-dialogue-source,#iw-archive-dialogue .iw-dialogue-source{color:#b5c6db!important}
#iw-game-dialogue .iw-dialogue-footer,#iw-archive-dialogue .iw-dialogue-footer{display:none}
#iw-route-cards{gap:12px}#iw-route-cards button{white-space:normal;min-height:58px}
#iw-game-layout{flex-direction:column!important;gap:10px!important}
#iw-game-layout>div{width:100%!important;min-width:0!important;flex-basis:auto!important}
#iw-game-stage{width:100%;max-width:1120px!important;margin:0 auto!important}
#iw-game-screen video{max-height:none;width:100%;aspect-ratio:832/480;object-fit:contain}
#iw-game-summary{max-height:560px;overflow:auto;padding-right:12px}
#iw-game-stage{--iw-prompt-height:96px;min-width:0!important;max-width:100%;overflow:hidden;
border-radius:10px;background:#07101a}
#iw-game-screen{aspect-ratio:832/480;min-height:360px!important;min-width:0!important;
max-width:100%;overflow:hidden;background:#07101a}
#iw-game-screen>:first-child{flex:1 1 auto!important;min-height:inherit;width:100%}
#iw-game-stage #iw-game-dialogue{bottom:calc(48px + var(--iw-prompt-height) + 10px);max-height:30%}
#iw-game-prompt{position:absolute!important;bottom:46px;left:5%;right:5%;width:90%!important;
min-height:var(--iw-prompt-height);box-sizing:border-box;z-index:12;padding:8px 14px!important;
gap:4px!important;border:none!important;border-radius:8px!important;
background:#101823c9!important;backdrop-filter:none;box-shadow:none!important}
#iw-prompt-heading,#iw-prompt-compose{gap:12px!important;align-items:center!important;flex-wrap:nowrap!important;
min-width:0!important;max-width:100%;overflow:hidden}
#iw-prompt-heading>div{min-width:0!important}
#iw-prompt-heading #iw-prompt-label{color:#f5f9ff!important;flex:0 0 112px!important;
width:112px!important;min-width:112px!important;max-width:112px!important;
font-size:12px;font-weight:700;letter-spacing:.1em}
#iw-prompt-heading #iw-input-mode{flex:0 0 160px!important;width:160px!important;
min-width:160px!important;max-width:160px!important;margin-left:auto!important}
#iw-prompt-label .html-container,#iw-game-input-status .html-container{padding:0!important}
#iw-prompt-label .prose,#iw-game-input-status .prose{margin:0!important;max-width:none!important}
#iw-game-prompt #iw-prompt-label span{white-space:nowrap;font-size:12px!important;letter-spacing:.06em!important}
#iw-game-prompt #iw-prompt-label .prose,#iw-game-prompt #iw-prompt-label .prose *{color:#edf3fa!important}
#iw-input-mode fieldset{border:none!important;background:transparent!important;padding:0!important;gap:4px}
#iw-input-mode label{background:transparent!important;color:#c9d5e6!important;border:1px solid transparent!important;
border-radius:6px!important;padding:4px 10px!important;font-size:12px!important;box-shadow:none!important}
#iw-input-mode label:has(input:checked){background:#86d9ec!important;color:#06232d!important;border-color:#86d9ec!important}
#iw-input-mode input{position:absolute;opacity:0;pointer-events:none}
#iw-game-text{flex:1 1 auto!important;min-width:0!important;background:transparent!important;border:none!important;box-shadow:none!important}
#iw-prompt-compose::before{content:">";position:absolute;left:0;top:50%;transform:translateY(-50%);
color:#8cddef;font-size:18px;line-height:1;pointer-events:none;z-index:2}
#iw-game-text textarea{background:transparent!important;color:#f5f9ff!important;border:none!important;
font-size:17px!important;line-height:1.2!important;box-shadow:none!important;resize:none;
min-height:36px!important;max-height:44px!important;padding:2px 2px 2px 18px!important;overflow-y:auto!important}
#iw-game-text textarea::placeholder{color:#b0bfd0!important;opacity:.9}
#iw-game-send{position:relative;flex:0 0 38px!important;width:38px!important;min-width:38px!important;
height:38px!important;min-height:38px!important;padding:0!important;border-radius:50%!important;
background:#8cddef!important;color:transparent!important;border:none!important;font-size:0!important;box-shadow:none!important}
#iw-game-send::after{content:"✓" / "";position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
font-size:23px;line-height:1;color:#06232d;pointer-events:none}
#iw-game-send:disabled{opacity:.48!important}
#iw-game-input-status{min-width:0!important;width:0!important;flex:1 1 0!important;
color:#b4c4d8!important;padding:0!important;overflow:hidden!important}
.iw-input-status{display:flex;align-items:center;gap:6px;font-size:11px;line-height:1.4;min-width:0}
.iw-input-status strong{white-space:nowrap;font-weight:600}.iw-input-detail{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.iw-input-dot{width:6px;height:6px;border-radius:50%;background:#94a9bb;flex-shrink:0}
.iw-input-status[data-phase=completed]{color:#a9e8cf}.iw-input-status[data-phase=completed] .iw-input-dot{background:#86d9b8}
.iw-input-status[data-phase=sending]{color:#a9e9ff}.iw-input-status[data-phase=sending] .iw-input-dot{background:#83d7f1}
.iw-input-status[data-phase=failed]{color:#ffb6af}.iw-input-status[data-phase=failed] .iw-input-dot{background:#f39b94}
#iw-game-prompt #iw-game-input-status .prose,#iw-game-prompt .iw-input-status{color:#cdd9e7!important}
#iw-game-prompt .iw-input-status strong,#iw-game-prompt .iw-input-status span:not(.iw-input-dot){color:inherit!important}
#iw-game-prompt .iw-input-status[data-phase=completed]{color:#b5efd7!important}
#iw-game-prompt .iw-input-status[data-phase=sending]{color:#b9edff!important}
#iw-game-prompt .iw-input-status[data-phase=failed]{color:#ffc3bd!important}
#iw-game-prompt .iw-input-dot,#iw-game-prompt .iw-input-detail,#iw-input-mode-hint{display:none!important}
#iw-game-prompt .iw-input-status[data-phase=failed] .iw-input-detail{display:inline!important}
@media(max-width:650px){#iw-game-dialogue,#iw-archive-dialogue{left:3%;right:3%;width:94%!important;bottom:36px}
#iw-game-dialogue .iw-dialogue-text,#iw-archive-dialogue .iw-dialogue-text{font-size:13px}
#iw-game-screen{min-height:220px!important}
#iw-game-stage #iw-game-dialogue{bottom:40px;max-height:45%}
#iw-game-prompt{position:relative!important;left:auto;right:auto;bottom:auto;width:100%!important;
min-height:96px;margin-top:0!important;padding:8px 10px!important;border-radius:0 0 8px 8px!important;gap:4px!important}
#iw-prompt-heading{gap:6px!important;flex-wrap:nowrap!important}
#iw-prompt-heading #iw-prompt-label{width:104px!important;min-width:104px!important;
max-width:104px!important;flex-basis:104px!important}
#iw-game-input-status{order:0;width:0!important;flex:1 1 0!important}
#iw-prompt-heading #iw-input-mode{width:128px!important;min-width:128px!important;
max-width:128px!important;flex-basis:128px!important}
#iw-input-mode label{padding:4px 8px!important}
#iw-game-text textarea{font-size:16px!important;min-height:36px!important}}
"""


def compact_dialogue_html(session, turn_id=None):
    """Only speaker + one exact line; full provenance stays in the history payload."""
    payload = build_game_dialogue(session, turn_id)
    messages = payload['latest']['messages']
    npc = [m for m in messages if m['role'] == 'npc']
    narration = [m for m in messages if m['role'] in ('narrator', 'system')]
    selected = (npc or narration or messages)[-1:]
    # New entry only: use a short first-person action intent, not production
    # terminology from the rule reason. Full original reasons/source/warnings
    # remain in history/export. Never rewrite an NPC line or a refusal as success.
    story_guard=(session.get('story_lineage_guard') is True if isinstance(session,dict)
        else getattr(getattr(session,'renderer',None),'story_lineage_guard',False))
    if story_guard and not npc:
        history=session.get('turns',[]) if isinstance(session,dict) else getattr(session,'history',[])
        row=next((r for r in reversed(history) if turn_id is None or r.get('turn_id')==turn_id),None)
        if row and row.get('rule_committed'):
            intent=action_intent_caption(row.get('plan',{}))
            if intent:
                selected=[dict(role='narrator',speaker='旁白',text=intent,
                    source='player_action_intent_presentation_not_visual_success',turn_id=row.get('turn_id'))]
    # Task-only refusals can occur without a new turn. Keep their actual message
    # on screen even with the state panel folded. Never apply final task feedback
    # to a selected historical turn/video.
    task = session.get('task') if isinstance(session, dict) else getattr(session, 'task', None)
    status = task.get('status') if isinstance(task, dict) else getattr(task, 'status', None)
    message = task.get('message') if isinstance(task, dict) else getattr(task, 'message', None)
    resume = session.get('session_resume') if isinstance(session,dict) else getattr(session,'session_resume',None)
    continued = isinstance(resume,dict) and resume.get('kind') in ('pinned_story_contract_continuation_v1','pinned_story_oom_continuation_v1')
    if turn_id is None and status in ('needs_clarification', 'paused') and isinstance(message, str) and message.strip():
        selected = [{'role':'system', 'speaker':'系统', 'text':'这一步还没完成，先停在这里。' if continued else message,
                     'source':'task_feedback', 'turn_id':None}]
    elif turn_id is None and continued and status=='ready' and message=='接着向右看。':
        selected=[{'role':'narrator','speaker':'旁白','text':message,
                   'source':'continuation_action_intent_not_visual_success','turn_id':None}]
    rows = []
    for message in selected:
        role = message['role'] if message['role'] in ('npc', 'narrator', 'system', 'player', 'action') else 'system'
        safe = lambda value: escape(str(value if value is not None else ''), quote=True)
        rows.append(f'<div class="iw-dialogue-row" data-role="{role}" '
                    f'data-turn-id="{safe(message.get("turn_id"))}" '
                    f'data-source="{safe(message.get("source"))}">'
                    f'<span class="iw-dialogue-speaker">{safe(message.get("speaker"))}</span>'
                    f'<div class="iw-dialogue-text">{safe(message.get("text"))}</div></div>')
    return '<section class="iw-dialogue-stage" aria-label="游戏对话框">' + ''.join(rows) + '</section>'


def action_intent_caption(plan):
    """Display the attempted action only, never certify its generated result."""
    action,target=plan.get('semantic_action'),plan.get('target')
    names={'anna':'Anna','leo':'Leo','terminal':'终端','door_a':'那扇门','desk':'桌子',
           'storage_room':'储藏室','core_area':'核心区域','red_key':'红钥匙'}
    if action=='inspect' and target in ('right','left','room'):
        return {'right':'我向右看看。','left':'我向左看看。','room':'我环顾四周。'}[target]
    if action=='approach' and target in names:return '我朝'+names[target]+'走去。'
    if action=='search' and target=='desk':return '我查看桌面和抽屉，寻找钥匙。'
    if action=='pick_up' and target=='red_key':return '我试着拿起红钥匙。'
    if action=='use' and target=='door_a':return '我用钥匙试着解锁这扇门。'
    if action=='open' and target=='door_a':return '我试着推开这扇门。'
    if action=='move_to' and target in names:return '我朝'+names[target]+'里面走去。'
    if action=='use' and target=='terminal':return '我在终端输入密码。'
    if action=='inspect' and target=='terminal':return '我查看终端里的公开日志。'
    return None


ENDING_TEXT = {
    'A': '你与 Anna 合作取得密码，按剧情规则解锁终端并进入核心区域。',
    'B': '你选择独自调查，按剧情规则取得红钥匙并进入储藏室。',
    'C': '你恢复公开日志，并与 Leo 核对事故记录，完成证据调查。',
}


def ending_text(route):
    return ENDING_TEXT.get(route, '本路线的规则结局已完成。') + ' 规则结局不代表画面事件已验证。'


class GameArchive:
    """Only operator-listed saved sessions; never an active-engine import."""
    def __init__(self, manifest):
        self.manifest = Path(manifest)
        data = json.loads(self.manifest.read_text(encoding='utf-8'))
        self.items = {item['route_id']: item for item in data['routes']}
        if set(self.items) != {'A', 'B', 'C'}:
            raise ValueError('Archive must explicitly list A/B/C')

    def choices(self):
        return [(item['title'], key) for key, item in self.items.items()]

    def load(self, route):
        item = self.items[route]
        data = json.loads(Path(item['session_export']).read_text(encoding='utf-8'))
        return item, data

    def turns(self, route):
        item, data = self.load(route)
        choices = []
        for row in data.get('turns', []):
            tid = row['turn_id']
            if str(tid) in item.get('videos', {}):
                choices.append((f'第 {tid} 轮 · {row.get("reason", "")[:45]}', str(tid)))
        return choices

    def view(self, route, turn):
        item, data = self.load(route)
        if str(turn) not in item['videos']:
            raise ValueError('Turn not listed in this read-only archive')
        video = Path(item['videos'][str(turn)])
        if not video.is_file():
            raise FileNotFoundError('Archived video unavailable')
        payload = build_game_dialogue(data, int(turn))
        metadata = dict(item, mode='read_only_historical_replay',
                        selected_turn=int(turn), live_generation=False,
                        active_world_mutated=False, visual_success_verified=False)
        return str(video), compact_dialogue_html(data, int(turn)), payload, metadata
