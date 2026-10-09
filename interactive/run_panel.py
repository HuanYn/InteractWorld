"""Read saved jobs, never launch work or infer success from process exit status."""
import json
from pathlib import Path


def read_json(path):
    try: return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError): return None


def _session_records(engine):
    """Only actual completed renders qualify as selectable video versions."""
    if engine is None: return {}
    records={}
    for row in engine.history:
        if not row.get('rule_committed') or row.get('renderer_status')!='completed': continue
        path=row.get('video')
        if not path or not Path(path).is_file(): continue
        version_id=f'turn:{row["turn_id"]}'
        renderer=row.get('renderer',{})
        records[version_id]={'id':version_id,'video':path,'turn_id':row['turn_id'],
            'source':'pre_generated' if renderer.get('pre_generated') else getattr(engine.renderer,'kind','unknown'),
            'player_text':row.get('player_text'),'plan':row.get('plan'),
            'visual_assessment':row.get('visual_assessment'),
            'human_reviews':[r for r in engine.visual_reviews if r.get('operation_id')==row.get('operation_id') and r.get('source') in ('human_override','human_review')],
            'human_review_scope':'operation including its retries; not an independent per-version score',
            'visual_goal_verified':False,'latency_seconds':row.get('latency_seconds'),
            'world_state_rolled_back':False}
    for index,review in enumerate(engine.visual_reviews):
        path=review.get('video')
        if review.get('source')!='visual_retry' or review.get('status')!='completed' or not path or not Path(path).is_file(): continue
        version_id=f'retry:{index}'
        records[version_id]={'id':version_id,'video':path,'turn_id':review.get('turn_id'),
            'source':'visual_retry','renderer':review.get('renderer'),
            'visual_assessment':review.get('assessment'),'rule_effect_reapplied':False,
            'visual_goal_verified':False,'latency_seconds':review.get('latency_seconds'),
            'world_state_rolled_back':False}
    return records


def session_versions(engine):
    return [(f'{key} · '+('视觉重试' if row['source']=='visual_retry' else
        '预生成分镜' if row['source']=='pre_generated' else '本会话已完成片段'),key)
        for key,row in _session_records(engine).items()]


def version_view(engine,key):
    row=_session_records(engine).get(key)
    return (row['video'],row) if row else (None,{'status':'no_available_version','visual_goal_verified':False})


def session_details(engine):
    if engine is None: return {'status':'not_started'}
    return {'session_id':engine.session_id,'current_task':engine.task.as_dict() if engine.task else None,
        'previous_goals':engine.task_history,'pending_visual':engine.pending_visual,
        'pending_visual_queue':getattr(engine,'pending_visual_queue',[]),
        'visual_review_policy':getattr(engine,'visual_review_policy','model_gate'),
        'visual_reviews':engine.visual_reviews,'selected_output':getattr(engine,'selected_output',None),
        'visual_judgments_are_ground_truth':False}


def saved_run_videos(folder):
    """Enumerate completed clips from operator-owned reports, including retries."""
    root=Path(folder); records={}
    for path in sorted(root.glob('*/result.json'))[:40]:
        run=read_json(path)
        if not isinstance(run,dict): continue
        name=path.parent.name
        tasks=[run.get('task')]+[event.get('task') for event in run.get('events',[]) if isinstance(event,dict)]
        sources={task['task_id']:task.get('planning_source','unknown') for task in tasks
                 if isinstance(task,dict) and task.get('task_id')}
        for index,row in enumerate(run.get('turns') or []):
            video=row.get('video')
            if not (row.get('rule_committed') and row.get('renderer_status')=='completed' and video and Path(video).is_file()): continue
            key=f'{name} / turn {row.get("turn_id",index)}'
            records[key]={'video':video,'request':row.get('player_text'),'plan':row.get('plan'),
                'visual_assessment':row.get('visual_assessment'),'process_status':run.get('status'),
                'planning_source':sources.get(str(row.get('operation_id','')).split(':')[0],'see_original_report'),
                'task_status':(run.get('task') or {}).get('status'),
                'final_run_task_planning_source':(run.get('task') or {}).get('planning_source','see_original_report'),
                'planning_source_scope':'Final run task only; may differ from this earlier or imported turn. See report events.',
                'report':str(path),'visual_ground_truth':False,'playback_only':True}
        for index,row in enumerate(run.get('visual_reviews') or []):
            video=row.get('video')
            if not (row.get('source')=='visual_retry' and row.get('status')=='completed' and video and Path(video).is_file()): continue
            key=f'{name} / retry {index}'
            records[key]={'video':video,'assessment':row.get('assessment'),
                'rule_effect_reapplied':False,'visual_ground_truth':False,'playback_only':True,'report':str(path)}
    return records


def snapshot(folder):
    root=Path(folder); result={}; latest_video=None
    latest_mtime=-1
    def consider_video(path):
        nonlocal latest_video,latest_mtime
        if path and Path(path).is_file():
            modified=Path(path).stat().st_mtime
            if modified>latest_mtime: latest_video,latest_mtime=path,modified
    # Read bounded operator-owned run directories, never paths supplied by a user.
    candidates=sorted(root.glob('*/result.json'))[:40]
    if (root/'result.json').is_file(): candidates.insert(0,root/'result.json')
    for path in candidates:
        name=path.parent.name
        run=read_json(path)
        if not isinstance(run,dict): continue
        turns=run.get('turns') or []
        task=run.get('task') or {}
        result[name]={'request':run.get('request',run.get('goal')),'process_status':run.get('status'),
            'task_status':task.get('status'),'error':run.get('error'),
            'actual_turns':len(turns),'rule_goal_completed':run.get('rule_goal_completed'),
            'human_visual_review':run.get('human_review','pending'),
            'message':task.get('message'),
            'visual_goal_verified':False,'report_path':str(path)}
        correction=read_json(root.parent/'continuation-20260927'/f'{name}-semantic-correction.json')
        if correction:
            result[name]['semantic_validation_correction']=correction
        for turn in turns:
            video=turn.get('video')
            if (turn.get('renderer_status')=='completed' and turn.get('rule_committed') and
                    video and Path(video).is_file()): consider_video(video)
    stage=read_json(root/'planner-sft/stage.json')
    if stage:
        profiles=sorted((root/'planner-sft/adapter').glob('profile-step-*.json'),key=lambda p:int(p.stem.split('-')[-1]))
        result['planner_training']={'status':stage.get('status'),'target':stage.get('training_target'),
            'current_step':stage.get('current_step'),'error':stage.get('error'),
            'latest_saved_profile':read_json(profiles[-1]) if profiles else None,
            'maximum_optimizer_steps':stage.get('max_optimizer_steps'),'auto_deploy':False}
        comparison=read_json(root/'planner-sft/comparison.json')
        if comparison: result['planner_development_comparison']=comparison
    continuation=root.parent/'continuation-20260927'
    goal=read_json(continuation/'goal-sft-v1/stage.json')
    if goal:
        profiles=sorted((continuation/'goal-sft-v1/training').glob('profile-step-*.json'),key=lambda p:int(p.stem.split('-')[-1]))
        result['current_multi_step_training']={'status':goal.get('status'),'current_step':goal.get('current_step'),
            'data_summary':goal.get('data_summary'),'error':goal.get('error'),
            'latest_saved_profile':read_json(profiles[-1]) if profiles else None,
            'candidate_for_live_validation':goal.get('candidate_for_live_validation'),'auto_deploy':False}
        paired=continuation/'goal-sft-v1/paired-recheck'
        qualification=read_json(paired/'status.json') or {}
        if qualification.get('status')=='completed':
            comparison=read_json(paired/'comparison.json')
            if comparison:
                result['multi_step_development_comparison']=comparison
                result['comparison_scope']={'source':str(paired/'comparison.json'),
                    'same_source_version_recheck':True,'test_split_used':False,
                    'note':'15条单世界开发样本，含4条多步；规则计划评分，不是视频成功率。'}
        else:
            result['multi_step_comparison_pending']={
                'status':qualification.get('status','awaiting_same_version_baseline'),
                'note':'首轮存在代码版本变更；同版本基线补齐前不展示训练收益。'}
    interrupt=read_json(continuation/'interrupt-v1/result.json')
    if interrupt:
        result['real_model_interrupt']={'process_status':interrupt.get('status'),
            'task_status':interrupt.get('task',{}).get('status'),'error':interrupt.get('error'),
            'actual_turns':len(interrupt.get('turns',[])),'interruption':interrupt.get('interruption'),
            'note':'中断消息由测试程序定时发出；规划和视频为真实模型，不是人工用户研究。'}
        for turn in interrupt.get('turns',[]):
            video=turn.get('video')
            if video and Path(video).is_file(): consider_video(video)
    for name in ('summary.json','qualification.json','completion.json','delivery.json'):
        value=read_json(root/name)
        if value is not None: result[name]=value
    return result,latest_video


def add_run_panel(gr,folder,label='本轮真实运行结果（不把退出成功当任务成功）'):
    if not folder: return
    with gr.Accordion(label,open=False):
        gr.Markdown('宽泛目标与更明确的分步输入分别列出，不混为同输入对照。过程完成不等于任务完成，规则完成也不等于画面正确。训练数据与样本量见数据工作台。')
        initial,video=snapshot(folder)
        result=gr.JSON(value=initial,label='实际模型调用、失败原因及训练进度')
        latest=gr.Video(value=video,label='本轮最近已完成步骤（若有），不是整条任务成功证明')
        refresh=gr.Button('刷新保存的结果（不会启动生成）',size='sm')
        refresh.click(lambda:snapshot(folder),[],[result,latest],api_name=False,queue=False)
        timer=gr.Timer(value=15)
        timer.tick(lambda:snapshot(folder),[],[result,latest],api_name=False,queue=False)
        clips=gr.Dropdown(choices=list(saved_run_videos(folder)),label='选择已保存运行的原片或重试片（仅回放，不启动GPU）')
        selected_clip=gr.Video(label='所选真实生成片段')
        clip_details=gr.JSON(label='该片段的请求、判断和来源；不是视觉成功证明')
        def choose_clip(key):
            row=saved_run_videos(folder).get(key,{})
            return row.get('video'),row
        clips.change(choose_clip,[clips],[selected_clip,clip_details],api_name=False,queue=False)
        refresh_clips=gr.Button('刷新已保存片段列表',size='sm')
        refresh_clips.click(lambda:gr.update(choices=list(saved_run_videos(folder)),value=None),[],[clips],api_name=False,queue=False)
