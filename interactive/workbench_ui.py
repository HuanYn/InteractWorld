"""Read-only interview evidence view; never invent missing human scores."""
import json
from pathlib import Path
from .visual_data import save_review


def review_history(folder,candidate_id):
    """Read only real submitted labels; do not auto-fill an unreviewed sample."""
    path=Path(folder)/'visual-reviews.jsonl'
    if not path.is_file(): return []
    records=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        try: row=json.loads(line)
        except ValueError: continue
        if isinstance(row,dict) and row.get('candidate_id')==candidate_id: records.append(row)
    return records


def add_data_workbench(gr,root):
    with gr.Accordion('数据工作台：原始输入 → 可达状态 → 处理标签',open=False):
        if not root or not (Path(root)/'summary.json').is_file():
            gr.Markdown('数据目录尚未配置；不会用示例数值冒充数据成果。')
            return
        folder=Path(root)
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        rows=[json.loads(line) for line in (folder/'records.jsonl').read_text(encoding='utf-8').splitlines() if line]
        gr.Markdown('当前为本项目规则可达状态构建的任务参考，不是真人采集，也不是模型成功轨迹。'
                    '人工语义审核尚未完成；同一世界的不同状态组合不等于独立世界。')
        gr.JSON(value=summary,label='实际数据统计与限制')
        selected=gr.Dropdown(choices=[r['id'] for r in rows],value=rows[0]['id'] if rows else None,label='查看一个实际样本')
        with gr.Row():
            raw=gr.JSON(label='原始请求与到达该状态的动作')
            processed=gr.JSON(label='处理后的标签、划分与审核状态')
        lookup={r['id']:r for r in rows}
        def show(key):
            row=lookup.get(key,{})
            return ({k:row.get(k) for k in ('player_input','world_state','reachability_path')},
                    {k:row.get(k) for k in ('expected_output','rule_reason','seed_group','split','source','human_review','visual_ground_truth')})
        selected.change(show,[selected],[raw,processed],api_name=False)
        if rows:
            first=show(rows[0]['id']); raw.value=first[0]; processed.value=first[1]
        goal_path=folder/'goal-records.jsonl'
        if goal_path.is_file():
            goals=[json.loads(line) for line in goal_path.read_text(encoding='utf-8').splitlines() if line]
            goal_lookup={r['id']:r for r in goals}
            gr.Markdown('多步目标参考由有限规则搜索构造，含路线冲突、禁止开门及中途改为观察；不是模型实际成功轨迹。')
            goal_choice=gr.Dropdown(choices=list(goal_lookup),label='多步／修改目标参考')
            goal_record=gr.JSON(label='目标、可达状态、参考工具序列及来源')
            goal_choice.change(lambda key:goal_lookup.get(key,{}),[goal_choice],[goal_record],api_name=False)
        visual_path=folder/'visual-candidates.jsonl'
        videos=[json.loads(x) for x in visual_path.read_text(encoding='utf-8').splitlines() if x] if visual_path.is_file() else []
        gr.Markdown(f'已索引 {len(videos)} 个真实生成视频候选。未审核的事件标签留空，不自动作为成功视频训练。')
        if videos:
            choice=gr.Dropdown(choices=[r['id'] for r in videos],label='视觉事件候选')
            with gr.Row():
                clip=gr.Video(label='原始生成片段（非当前请求）')
                annotation=gr.JSON(label='请求、模型判断与待填写的人工标签')
            index={r['id']:r for r in videos}
            def show_video(key):
                row=dict(index.get(key,{}))
                row['saved_human_reviews']=review_history(folder,key)
                path=row.get('raw_video')
                return (path if path and Path(path).is_file() else None,row)
            choice.change(show_video,[choice],[clip,annotation],api_name=False)
            gr.Markdown('按实际视频标注；“请求开门”但门没有动时应标不符合。保存记录不会直接把视频送入训练。')
            verdict=gr.Radio(choices=[('符合','supported'),('不符合','contradicted'),('不确定','uncertain')],
                label='人工检查结论')
            event=gr.Textbox(label='实际发生的事件',placeholder='例如：镜头靠近门，门仍然关闭。')
            with gr.Row():
                start=gr.Number(value=None,label='事件开始秒数（可留空）',minimum=0)
                end=gr.Number(value=None,label='事件结束秒数（可留空）',minimum=0)
            note=gr.Textbox(label='可见证据与缺陷',placeholder='描述物体状态变化、接触是否可见、形变或遮挡等。')
            save=gr.Button('保存本条视觉数据审核（不覆盖原视频）')
            receipt=gr.JSON(label='保存记录；仍需质量和来源检查才能用于训练')
            export_reviews=gr.Button('导出已保存人工审核',size='sm')
            review_file=gr.File(label='人工记录 JSONL（不含视频和权重）',interactive=False)
            def annotate(key,verdict,event,start,end,note):
                if key not in index: raise gr.Error('请先选择一个真实视频候选。')
                try: return save_review(folder,index[key],verdict,event,start,end,note)
                except ValueError as exc: raise gr.Error(str(exc))
            save.click(annotate,[choice,verdict,event,start,end,note],[receipt],api_name=False)
            def export():
                path=folder/'visual-reviews.jsonl'
                if not path.is_file(): raise gr.Error('还没有保存任何人工审核记录。')
                return str(path)
            export_reviews.click(export,[],[review_file],api_name=False)
