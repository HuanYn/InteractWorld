"""Explicitly labelled, read-only browser evidence. Does not start models."""
from pathlib import Path


def recording_paths(root):
    root=Path(root).resolve()
    return [p for name in ('browser-workflow.webm','export-recovery.webm')
            if (p:=root/name).is_file() and p.resolve().is_relative_to(root)]


def add_delivery_recordings(gr,root):
    notes=Path(root)/'interview-handoff.md'
    if notes.is_file():
        with gr.Accordion('面试讲解与证据索引',open=False):
            gr.Markdown(notes.read_text(encoding='utf-8'))
    paths=recording_paths(root)
    if not paths: return
    with gr.Accordion('真实浏览器演示录像（含失败与恢复）',open=False):
        gr.Markdown('第一段是完整原速操作与等待：初始6秒生成 → 多步规划 → 生成中改为看左边 → 执行新观察。'
                    '原录制最后导出失败，记录未删改。第二段是修复后只读下载同一历史会话，未重新生成视频。'
                    '\n\n两段不是一次无故障验收，也不是一分钟成功剧情。状态检查通过不等于画面动作成功。')
        for path in paths:
            label=('① 原始流程（末尾导出失败，保留完整等待）' if path.name=='browser-workflow.webm'
                   else '② 单独补录：历史会话下载恢复（不使用GPU）')
            gr.Video(value=str(path),label=label,interactive=False)
        gr.Markdown('会话核对：2条提交动作（接近Anna、观察左侧）；旧计划暂停/被替换；未获得钥匙，门未打开。'
                    '向左工具参数正确，但画面方向尚无独立成功认证。可在下方“已保存会话下载”读取9127fafc1a94开头的会话。')
