"""Text-input presentation and real submission status, with no world effects."""
from dataclasses import dataclass
from html import escape


INPUT_MODES = [('行动', 'action'), ('交谈', 'talk')]
INPUT_HINTS = {
    'action': '输入行动，例如：回头看右边。',
    'talk': '输入对 Anna 或 Leo 说的话…',
}


def input_hint(mode):
    if mode not in INPUT_HINTS:
        raise ValueError('不支持此输入类型。仅支持文字行动或文字交谈。')
    return INPUT_HINTS[mode]


def status_html(phase='idle', detail='输入文字后发送；不会自动启动模型。'):
    labels = {'idle': '待发送', 'queued': '待发送', 'sending': '发送中',
              'completed': '已完成', 'failed': '失败'}
    if phase not in labels:
        raise ValueError('Unknown submission phase')
    safe_detail = escape(str(detail), quote=True)
    return (f'<div class="iw-input-status" data-phase="{phase}" role="status" aria-live="polite" '
            f'title="{safe_detail}" aria-label="{labels[phase]}：{safe_detail}">'
            f'<span class="iw-input-dot"></span><strong>{labels[phase]}</strong>'
            f'<span class="iw-input-detail">{safe_detail}</span></div>')


def queue_text_input(text, engine, mode='action'):
    """Request the existing task's boundary pause; do not submit or edit state."""
    input_hint(mode)
    if not isinstance(text, str) or not text.strip():
        return status_html('failed', '请输入文字；尚未提交。')
    if len(text) > 2000:
        return status_html('failed', '输入超过 2000 字符；尚未提交。')
    task = getattr(engine, 'task', None)
    if task is not None and task.status in ('ready', 'running'):
        task.pause_requested.set()
        return status_html('queued', '已请求暂停旧计划，等待当前片段结束后处理新输入。')
    return status_html('queued', '等待处理；尚未宣告动作或视频成功。')


def submitted_result_status(engine, previous_turns):
    """Read an actual handler result; never equate HTTP completion and success."""
    history = getattr(engine, 'history', [])
    if len(history) <= previous_turns:
        return 'failed', '处理结束，但未产生新的动作记录；请查看服务和会话状态。'
    row = history[-1]
    if not row.get('valid') or not row.get('rule_committed'):
        return 'failed', str(row.get('reason') or '规则未提交本次动作。')[:300]
    if row.get('renderer_status') == 'failed':
        return 'failed', str(row.get('reason') or '视频生成失败。')[:300]
    turn = row.get('turn_id', len(history))
    if row.get('renderer_status') == 'dialogue_only':
        return 'completed', f'第 {turn} 轮已记录剧情对话，保留原画面；没有新视频或语音。'
    if row.get('visual_advisory_only') or row.get('failure_type'):
        return 'completed', f'第 {turn} 轮规则已提交，画面仍有警告或待检查；不代表视觉成功。'
    return 'completed', f'第 {turn} 轮已处理；规则提交与视觉成功分别判断。'


@dataclass(frozen=True)
class InputUpdate:
    phase: str
    detail: str
    result: tuple | None = None
    text: str = ''


def submission_updates(text, engine, perform, mode='action'):
    """Wrap the original interaction callback, without rewriting user text.

    ``perform`` remains the existing ``interact(text, engine)`` callback and its
    return tuple remains untouched. Mode is an input aid, not a forced action,
    rule bypass, dialogue generator or state injection.
    """
    original = text if isinstance(text, str) else ''
    previous_turns = len(getattr(engine, 'history', []))
    try:
        input_hint(mode)
        if not original.strip():
            raise ValueError('请输入文字；尚未提交。')
        if len(original) > 2000:
            raise ValueError('输入超过 2000 字符；尚未提交。')
        yield InputUpdate('sending', '正在处理真实输入，异步生成期间请等待。', text=original)
        result = tuple(perform(original, engine))
        phase, detail = submitted_result_status(result[0], previous_turns)
        yield InputUpdate(phase, detail, result, '' if phase == 'completed' else original)
    except Exception as exc:
        # Gradio/API/service errors are shown as failure, never a false "Sent".
        yield InputUpdate('failed', str(exc)[:300] or '处理失败；未确认动作提交。', text=original)
