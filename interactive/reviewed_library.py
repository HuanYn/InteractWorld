"""Read-only six-second candidates and their existing AI reviews, not live actions."""
import json
from pathlib import Path
from .feedback import assess_prebuilt

LABELS={'supported':'局部支持（非成功认证）','mismatch':'画面不符',
        'uncertain':'不确定','not_visually_testable':'无法视觉验证'}


def load_candidates(root):
    root=Path(root).resolve()
    rows=[]
    for relative in ('ai-reviews.jsonl','targeted-v1/ai-reviews.jsonl'):
        file=root/relative
        if not file.is_file(): continue
        for line in file.read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            row=json.loads(line)
            video=Path(row['raw_video']).resolve()
            if not video.is_relative_to(root) or video.suffix.lower()!='.mp4' or not video.is_file():
                raise ValueError('Reviewed media must be an existing MP4 inside the data directory')
            row=dict(row,raw_video=str(video))
            generation=row.get('generation') or row.get('metadata') or {}
            candidate=Path(generation.get('conditioning_image') or video.parent/'input.png').resolve()
            row['display_input']=str(candidate) if candidate.is_relative_to(root) and candidate.is_file() and candidate.suffix.lower()=='.png' else None
            rows.append(row)
    if len({r['id'] for r in rows})!=len(rows): raise ValueError('Duplicate candidate IDs')
    return rows


def media_paths(root):
    return [r[k] for r in load_candidates(root) for k in ('raw_video','display_input') if r.get(k)]


def candidate_view(row):
    review={k:row.get(k) for k in ('ai_verdict','actual_event','quality_exclusion','quality_notes','review_method')}
    assessment=assess_prebuilt({'pre_generated':True,'asset_id':row['id'],'asset_review':review},None)
    blocked=assessment['status']=='contradicted'
    status=('**暂停：已有AI复核指出画面不符或质量问题。** 不应把本片当成功事件继续执行。' if blocked else
            '**待审核：离线局部支持不代表本次动作完成。** 关键事件不能仅凭本片自动放行。')
    status+='\n\n这里只展示生产反馈策略对已存标签的处理；没有执行任务、修改背包或调用VLM/GPU。'
    details={k:row.get(k) for k in ('id','title','source_type','input_group','seed','review_status',
        'ai_verdict','actual_event','event_interval_seconds','quality_exclusion','quality_notes','review_method',
        'training_exclusion_reason')}
    details.update(prompt=row.get('prompt') or row.get('exact_prompt'),
        generation=row.get('generation') or row.get('metadata'),human_review=False,ground_truth=False,
        training_eligible=False,current_request_visual_success=False,
        scope='read_only_saved_candidate_policy_preview_not_execution',feedback_assessment=assessment)
    return row.get('display_input'),row['raw_video'],status,details


def add_reviewed_library(gr,root):
    rows=load_candidates(root)
    with gr.Accordion('六秒素材与失败边界（32条AI复核，只读）',open=False):
        gr.Markdown(f'已接入{len(rows)}条实际生成片段，原始输入、真实提示词、来源和AI复核一同展示。'
                    '这是已有独立分镜，不是当前指令生成，不替换当前世界。AI复核不是人工真值；没有样本因此自动获得训练资格。')
        if not rows: return
        lookup={r['id']:r for r in rows}
        first=next((r['id'] for r in rows if r['id']=='key_reveal-s106'),rows[0]['id'])
        choice=gr.Dropdown(choices=[(f"{r['id']} · {('质量排除' if r.get('quality_exclusion') else LABELS.get(r['ai_verdict'],'未复核'))}",r['id']) for r in rows],
                           value=first,label='六秒复核素材')
        initial=candidate_view(lookup[first])
        status=gr.Markdown(initial[2],elem_id='iw-reviewed-status')
        with gr.Row():
            before=gr.Image(value=initial[0],label='当时模型输入（原样，不是新生成）',interactive=False)
            video=gr.Video(value=initial[1],label='六秒原片（预生成候选）',interactive=False,elem_id='iw-reviewed-clip')
        details=gr.JSON(value=initial[3],label='来源、AI复核、反馈策略与训练资格',elem_id='iw-reviewed-details')
        def show(key):
            if key not in lookup: raise ValueError('请选择已有素材')
            return candidate_view(lookup[key])
        choice.change(show,[choice],[before,video,status,details],api_name='reviewed_candidate')
