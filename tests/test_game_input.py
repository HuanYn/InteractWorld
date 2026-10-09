"""CPU-only input contracts; fixtures do not demonstrate model/visual success."""
import ast
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from interactive.game_input import (input_hint, queue_text_input, status_html,
                                    submitted_result_status, submission_updates)
from interactive.game_presentation import GAME_STAGE_CSS


def engine_with(*rows, task=None):
    return SimpleNamespace(history=list(rows), task=task)


def committed_row(**extra):
    return dict(turn_id=1, valid=True, rule_committed=True,
                renderer_status='completed', **extra)


def test_mode_only_changes_hint_not_submitted_text():
    assert 'Anna' in input_hint('talk')
    with pytest.raises(ValueError):
        input_hint('voice')
    original = '  我想问 Anna 终端的事。  '
    engine = engine_with()
    seen = []
    def perform(text, current):
        seen.append((text, current))
        current.history.append(committed_row())
        return current, 'unchanged-view', ''
    updates = list(submission_updates(original, engine, perform, 'talk'))
    assert seen == [(original, engine)]
    assert [item.phase for item in updates] == ['sending', 'completed']
    assert updates[-1].result == (engine, 'unchanged-view', '')
    assert updates[-1].text == ''


def test_queue_pause_is_boundary_request_only():
    task = SimpleNamespace(status='running', pause_requested=Event())
    engine = engine_with(task=task)
    before = vars(engine).copy()
    html = queue_text_input('改为观察右边', engine)
    assert task.pause_requested.is_set()
    assert task.status == 'running' and engine.history == []
    assert vars(engine) == before
    assert 'data-phase="queued"' in html and '等待当前片段结束' in html


@pytest.mark.parametrize('text', ['', '  ', 'x' * 2001, None])
def test_invalid_input_neither_calls_handler_nor_pauses_task(text):
    task = SimpleNamespace(status='running', pause_requested=Event())
    engine = engine_with(task=task)
    calls = []
    assert 'data-phase="failed"' in queue_text_input(text, engine)
    assert not task.pause_requested.is_set()
    updates = list(submission_updates(text, engine, lambda *args: calls.append(args)))
    assert len(updates) == 1 and updates[0].phase == 'failed'
    assert calls == []


def test_offline_failure_keeps_text_never_reports_completed():
    def offline(*args):
        raise RuntimeError('模型服务离线，没有提交。')
    updates = list(submission_updates('开门', engine_with(), offline))
    assert [item.phase for item in updates] == ['sending', 'failed']
    assert updates[-1].text == '开门' and updates[-1].result is None
    assert '离线' in updates[-1].detail


def test_rule_rejection_is_failure_despite_callback_returning_normally():
    engine = engine_with()
    def reject(text, current):
        current.history.append(dict(turn_id=1, valid=False, rule_committed=False,
                                    renderer_status='not_requested', reason='缺少钥匙。'))
        return current, 'rule-rejection-view', ''
    final = list(submission_updates('开门', engine, reject))[-1]
    assert final.phase == 'failed' and final.text == '开门'
    assert final.detail == '缺少钥匙。' and final.result[1] == 'rule-rejection-view'


def test_advisory_and_dialogue_results_do_not_claim_visual_or_new_video():
    engine = engine_with(committed_row(visual_advisory_only=True, failure_type='visual_uncertain'))
    phase, detail = submitted_result_status(engine, 0)
    assert phase == 'completed' and '不代表视觉成功' in detail
    engine.history[-1]['renderer_status'] = 'dialogue_only'
    phase, detail = submitted_result_status(engine, 0)
    assert phase == 'completed' and '没有新视频或语音' in detail


def test_no_new_record_is_not_success():
    phase, detail = submitted_result_status(engine_with(committed_row()), 1)
    assert phase == 'failed' and '未产生新的动作记录' in detail


def test_status_html_escapes_errors_and_has_live_status():
    html = status_html('failed', '<img src=x onerror="alert(1)">')
    assert '<img' not in html and '&lt;img' in html
    assert 'aria-live="polite"' in html
    assert 'title="&lt;img' in html and 'aria-label="失败：&lt;img' in html
    assert '&quot;alert(1)&quot;' in html
    with pytest.raises(ValueError):
        status_html('sent')


def test_stage_contains_real_text_and_send_components_and_keeps_original_api():
    source = (Path(__file__).parents[1] / 'interactive/ui.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    stage = [node for node in ast.walk(tree) if isinstance(node, ast.With) and
             any(isinstance(item.context_expr, ast.Call) and
                 any(k.arg == 'elem_id' and isinstance(k.value, ast.Constant) and
                     k.value.value == 'iw-game-stage' for k in item.context_expr.keywords)
                 for item in node.items)]
    assert len(stage) == 1
    stage_source = ast.get_source_segment(source, stage[0])
    assert 'gr.Textbox' in stage_source and "elem_id='iw-game-text'" in stage_source
    assert 'gr.Button' in stage_source and "elem_id='iw-game-send'" in stage_source
    assert "api_name='game_interact'" in source
    assert "interact,[text,session],outputs+[text],api_name='interact'" in source
    assert 'submission_updates(text,engine,interact,input_mode)' in source
    assert "os.environ.get('IW_VISUAL_REVIEW_POLICY')=='demo_advisory'" in source
    assert 'gr.Audio' not in stage_source and 'WASD' not in stage_source


def test_prompt_and_dialogue_avoid_controls_and_mobile_stacking_is_explicit():
    assert '#iw-game-prompt{position:absolute' in GAME_STAGE_CSS
    assert 'bottom:46px' in GAME_STAGE_CSS
    assert 'bottom:calc(48px + var(--iw-prompt-height) + 10px)' in GAME_STAGE_CSS
    assert '@media(max-width:650px)' in GAME_STAGE_CSS
    assert '#iw-game-prompt{position:relative' in GAME_STAGE_CSS


def test_minimal_game_layout_folds_details_without_autostart_or_api_changes():
    source = (Path(__file__).parents[1] / 'interactive/ui.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    accordion_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute) and node.func.attr == 'Accordion']
    expected = {'开始与路线', '设置与记录', '状态与执行详情', '起始画面设置', '路线与游玩说明',
                '对话历史与来源', '三路线真实记录回看（只读，与当前游玩分开）'}
    for label in expected:
        call = next(node for node in accordion_calls if node.args and
                    isinstance(node.args[0], ast.Constant) and node.args[0].value == label)
        assert any(k.arg == 'open' and isinstance(k.value, ast.Constant) and
                   k.value.value is False for k in call.keywords)
    assert '#iw-game-layout{flex-direction:column' in GAME_STAGE_CSS
    assert '#iw-game-prompt .iw-input-status[data-phase=failed] .iw-input-detail{display:inline!important}' in GAME_STAGE_CSS
    assert 'app.load(check_service,[],[service,submit]' in source
    assert 'app.load(start_model' not in source
    # Existing explicit planning/execution/export endpoints and strict-page opt-out survive.
    for name in ('plan_goal', 'goal_next', 'goal_run', 'goal_pause', 'export_session'):
        assert "api_name='" + name + "'" in source
    assert "and (os.environ.get('IW_VISUAL_REVIEW_POLICY')=='demo_advisory' or live_current_frame)" in source


def test_all_non_stage_controls_have_one_real_collapsed_parent():
    source = (Path(__file__).parents[1] / 'interactive/ui.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    containers = [node for node in ast.walk(tree) if isinstance(node, ast.With) and
                  any(isinstance(item.context_expr, ast.IfExp) and
                      isinstance(item.context_expr.body, ast.Call) and
                      item.context_expr.body.args and
                      isinstance(item.context_expr.body.args[0], ast.Constant) and
                      item.context_expr.body.args[0].value == '设置与记录'
                      for item in node.items)]
    assert len(containers) == 1
    details = ast.get_source_segment(source, containers[0])
    for text in ('对话历史与来源', 'world_state_components()', 'Reset / Start Demo',
                 '目标任务：拆解', '导出本次会话记录', '三路线真实记录回看',
                 'add_data_workbench', 'add_run_panel', "api_name='import_initial'"):
        assert text in details
    assert "elem_id='iw-game-stage'" not in details
    # The game-only wrapper is a no-op for the original strict two-column entry.
    assert 'if game_enabled else nullcontext()' in details
    assert 'if not game_enabled:\n                with gr.Column(scale=2):\n                    summary,state=world_state_components()' in source
