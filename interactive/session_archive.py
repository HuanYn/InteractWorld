"""Read-only access to operator-owned session exports; never resumes execution."""
import json
from pathlib import Path
import re

def archived_sessions(root):
    root=Path(root).resolve()
    entries=[]
    for file in root.glob('*/session-export.json'):
        if not file.resolve().is_relative_to(root) or not re.fullmatch(r'[a-f0-9]{32}',file.parent.name): continue
        entries.append((file.stat().st_mtime,file.parent.name))
    return [(f'已保存会话 · {name[:12]}',name) for _,name in sorted(entries,reverse=True)[:30]]

def archived_export(root,session_id,evidence_reader=None):
    if not isinstance(session_id,str) or not re.fullmatch(r'[a-f0-9]{32}',session_id):
        raise ValueError('请选择已有会话。')
    root=Path(root).resolve(); file=(root/session_id/'session-export.json').resolve()
    if not file.is_relative_to(root) or not file.is_file(): raise ValueError('会话文件不可用。')
    value=json.loads(file.read_text(encoding='utf-8'))
    if value.get('session_id')!=session_id: raise ValueError('会话ID不匹配。')
    details={'source':'saved_session_read_only','session_id':session_id,
        'turn_count':len(value.get('turns',[])), 'task':value.get('task'),
        'previous_goals':value.get('task_history',[]),'world_state':value.get('state'),
        'visual_success_verified':False,'restores_execution':False}
    if evidence_reader is not None:
        details['current_renderer_evidence_not_session_result']=evidence_reader()
    return str(file),details

def add_session_archive(gr,root,evidence_reader=None):
    with gr.Accordion('已保存会话下载（无需模型在线）',open=False):
        gr.Markdown('只读查看并下载服务器已保存的真实会话，不恢复任务、不再次生成、不改变当前世界。页面重开后也可使用。')
        choice=gr.Dropdown(choices=archived_sessions(root),label='已保存会话')
        refresh=gr.Button('刷新已保存会话')
        download=gr.Button('读取并导出已保存会话')
        file=gr.File(label='已保存会话 JSON',interactive=False)
        details=gr.JSON(label='归档来源与状态')
        refresh.click(lambda:gr.update(choices=archived_sessions(root)),[],choice,api_name='archive_list')
        download.click(lambda selected:archived_export(root,selected,evidence_reader),choice,[file,details],api_name='archive_export')
