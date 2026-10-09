import os
import json
from contextlib import nullcontext
from pathlib import Path
import gradio as gr
from .engine import InteractionEngine
from .agent import LLMAgent,OpenAIProvider
from .observer import VLMObserver
from .renderers import LaboratoryWanRenderer
from .cinematic import CinematicRenderer
from .presentation import service_status,next_actions,turn_summary,task_summary
from .workbench_ui import add_data_workbench
from .run_panel import add_run_panel,session_versions,version_view,session_details
from .game_dialogue import build_game_dialogue,render_dialogue_html
from .game_presentation import GAME_STAGE_CSS,compact_dialogue_html,ending_text,GameArchive
from .game_input import INPUT_MODES,input_hint,status_html,queue_text_input,submission_updates
from .route_showcase import route_catalog,route_goal
from .live_ui import LIVE_CURRENT_FRAME_CSS,start_ui_world,require_ui_frame,live_media_state
from .closed_handoff import closed_reason,verify_closed_handoff
from .runtime_control import validate_public_environment


def _configured_visual_observer(provider):
    return VLMObserver(provider,contract_version=os.environ.get('IW_VISUAL_CONTRACT_VERSION','v1'))


def build_ui(output_root,mode='mock',evidence_dir=None,visual_observer=False):
    validate_public_environment()
    handoff=verify_closed_handoff()
    closed=handoff is not None
    def require_open_handoff():
        if closed_reason(): raise gr.Error(closed_reason())

    live_current_frame=os.environ.get('IW_LIVE_CURRENT_FRAME','0')=='1'
    story_lineage_guard=os.environ.get('IW_STORY_LINEAGE_GUARD','0')=='1'
    if live_current_frame and (mode!='wan' or os.environ.get('IW_VISUAL_REVIEW_POLICY','model_gate') not in ('model_gate','manual_critical')):
        raise ValueError('Live current-frame entry requires Wan and a strict visual review policy')
    if story_lineage_guard and (live_current_frame or mode!='wan'
            or os.environ.get('IW_GAME_PRESENTATION')!='1'
            or os.environ.get('IW_VISUAL_REVIEW_POLICY')!='demo_advisory'):
        raise ValueError('Story lineage UI requires the opt-in Wan advisory game entry, not strict live-camera mode')
    current_rgb_entry=live_current_frame or story_lineage_guard
    room_cut_manifest=os.environ.get('IW_DECLARED_ROOM_MANIFEST') if story_lineage_guard else None
    # Never opt the preserved strict8893 page into the advisory game controls.
    game_enabled=(os.environ.get('IW_GAME_PRESENTATION','0')=='1'
                  and (os.environ.get('IW_VISUAL_REVIEW_POLICY')=='demo_advisory' or live_current_frame))
    story_profile=os.environ.get('IW_STORY_PROFILE','legacy')
    if story_profile!='legacy' and (not game_enabled or story_profile!='spatial_v1'):
        raise ValueError('Spatial story profile requires the explicitly opted-in game entry')
    if story_profile=='spatial_v1':
        from .spatial_skills import spatial_route_catalog as active_route_catalog,spatial_route_goal as active_route_goal
    else:
        active_route_catalog,active_route_goal=route_catalog,route_goal
    evidence_reader=None
    if os.environ.get('IW_RENDERER_EVIDENCE_PROFILE') and os.environ.get('IW_RENDERER_ACCEPTANCE_PACK'):
        from .renderer_evidence import renderer_evidence
        def evidence_reader():
            return renderer_evidence(os.environ['IW_RENDERER_EVIDENCE_PROFILE'],
                                     os.environ['IW_RENDERER_ACCEPTANCE_PACK'])
    visual_labels={'supported':'VLM认为可能符合（未验证）','contradicted':'VLM认为不符合',
        'uncertain':'画面证据不足','not_visually_testable':'无法仅由画面验证'}
    def factory():
        goal_model=os.environ.get('IW_GOAL_MODEL')
        agent=LLMAgent(OpenAIProvider(),goal_provider=OpenAIProvider(model=goal_model) if goal_model else None) if mode in ('llm','wan','cinematic') else None
        observer=_configured_visual_observer(OpenAIProvider()) if visual_observer or mode in ('wan','cinematic') else None
        renderer=LaboratoryWanRenderer(os.environ.get('IW_WAN_BASE_URL','http://127.0.0.1:8878'),
            os.environ['IW_LAB_INITIAL_IMAGE'],cfg=float(os.environ.get('IW_LAB_CFG','3')),
            seconds=int(os.environ.get('IW_LAB_SECONDS','3')),
            lora_strength=float(os.environ.get('IW_LAB_LORA_STRENGTH','1')),
            npc_framing=False if current_rgb_entry else os.environ.get('IW_LAB_NPC_FRAMING','0')=='1',
            live_current_frame=live_current_frame,live_prompt_key=os.environ.get('IW_LIVE_PROMPT_KEY'),
            live_full_window_pan=live_current_frame and os.environ.get('IW_LIVE_FULL_WINDOW_PAN','0')=='1',
            story_lineage_guard=story_lineage_guard) if mode=='wan' else None
        if mode=='cinematic': renderer=CinematicRenderer(os.environ['IW_CINEMATIC_MANIFEST'])
        engine=InteractionEngine(output_root,agent=agent,observer=observer,renderer=renderer,
            visual_review_policy=os.environ.get('IW_VISUAL_REVIEW_POLICY','model_gate'),story_profile=story_profile)
        engine.goal_contract_normalization=os.environ.get('IW_GOAL_CONTRACT_NORMALIZATION','0')=='1'
        engine.direction_completion_compiler=os.environ.get('IW_DIRECTION_COMPLETION_COMPILER')=='pure_direction_v1'
        engine.authored_route_fallback_enabled=game_enabled
        engine.retain_dialogue_frame=game_enabled
        engine.live_current_frame=live_current_frame
        return engine

    def view(engine,row=None):
        state=engine.state
        summary=(f'### {engine.graph.title(state)}\n\n目标：{state.player.goal}  \n位置：{state.player.location}  '
            f'\n路线：{state.story.route or "未选择"}  \n背包：{", ".join(state.inventory.items) or "空"}  '
            f'\nAnna 信任：{state.characters["anna"].trust:.1f} · Leo 信任：{state.characters["leo"].trust:.1f}')
        if mode=='cinematic':
            flags=state.story.flags
            summary+='\n\n### 可见事件 / 剧情状态\n'
            summary+='\n- '+('🗝 规则背包已持有红钥匙' if 'red_key' in state.inventory.items else '🔎 规则状态：已发现红钥匙' if flags.get('found_key') else '❔ 尚未发现钥匙')
            summary+='\n- '+('🚪 规则状态：门已打开（画面另行检查）' if state.objects['door_a'].state=='open' else '🔒 储藏室门锁定')
            summary+='\n- '+('🖥 核心区域已解锁' if flags.get('terminal_accessed') else '📄 公开日志已恢复' if flags.get('log_retrieved') else '🖥 终端等待调查')
            if row:
                summary+='\n\n'+('**本轮事件：** '+row['reason'] if row['valid'] else '**动作被阻止：** '+row['reason'])
                if row.get('npc_response'): summary+='\n\n> '+row['npc_response']
            if state.story.terminal: summary+='\n\n## 本路线结束\n点击重新开始可体验另一条分支。'
        summary+='\n\n### 本轮执行\n\n'+turn_summary(row)
        if (engine.initial_recovery or {}).get('type')=='pinned_initial_media_import':
            summary+='\n\n**开场来源**：复用已生成的真实 R005 开场，新故事没有继承任何旧动作或背包；后续每步重新请求模型生成。'
        if getattr(engine.renderer,'npc_framing',False):
            summary+='\n\n**候选取景控制**：NPC 接近镜头使用当前图派生的二维推近条件，仍由 R005 采样；不代表玩家实际走近或三维导航。左右观察仍使用原动作按键路径。'
        summary+='\n\n### 下一步\n\n'+next_actions(state,engine.story_profile)
        summary+=task_summary(engine.task)
        if engine.visual_review_policy=='demo_advisory':
            summary+='\n\n**完整演示模式**：剧情按已校验规则推进；画面不符或证据不足仅提示，不代表视觉验证通过。'
            if engine.visual_warnings:
                summary+=f'  \n已保留 {len(engine.visual_warnings)} 项画面提示；未作人工通过。'
            if row and row.get('npc_response'): summary+='\n\n> '+row['npc_response']
            if state.story.terminal:
                summary+='\n\n## 本路线结束\n'+ending_text(state.story.route)+' 可以导出会话，或重新开始体验其他分支。'
        if engine.pending_visual:
            count=len(engine.unresolved_visuals())
            summary+=(f'\n\n### 视觉待处理 · {count} 项\n\n画面尚未通过当前审核门槛，剧情不按视觉失败自动推进。新动作只从当前画面继续，不回放旧动作。'
                if live_current_frame else f'\n\n### 视觉待处理 · {count} 项\n\n规则动作已提交，但画面尚未通过当前审核门槛。后续依赖动作已拦截；可以观察、重试本步画面或人工审核。')
            if getattr(engine,'visual_review_policy','model_gate')=='manual_critical':
                summary+='  \n当前是人工审核降级模式：关键动作即使被 VLM 判为符合，也需要你观看后明确决定。'
        if engine.visual_reviews:
            review=engine.visual_reviews[-1]
            summary+='\n\n最近视觉处理：'+str(review.get('source'))+' · '+str(review.get('status',review.get('note','')))
        debug={k:row.get(k) for k in ('agent_kind','plan','valid','reason','camera_skills','story_event','renderer_status','renderer','model_error','planner_fallback','visual_assessment')} if row else {}
        messages=[]
        for turn in engine.history:
            visual=turn.get('visual_assessment',{})
            prebuilt=visual.get('source')=='prebuilt_asset_ai_review'
            label=({'contradicted':'素材已知不符','uncertain':'不能证明本次动作视觉完成'} if prebuilt else visual_labels).get(visual.get('status'),'未验证')
            note=('\n\n画面检查（'+('预生成素材 AI 复核，非本次生成/真人真值' if prebuilt else 'VLM 判断，非真值')+'）：'+label+' — '+visual.get('evidence','')) if visual else ''
            messages.extend([{'role':'user','content':turn['player_text']},{'role':'assistant','content':turn['reason']+
                ('\n\n'+turn['npc_response'] if turn['npc_response'] else '')+note}])
        media=live_media_state(engine)
        video_value=(gr.update(value=media['video'],visible=media['video_visible'])
                     if current_rgb_entry else engine.video)
        result=(engine,video_value,summary,engine.get_state(),debug,engine.memory.retrieve(),messages)
        if game_enabled:
            payload=build_game_dialogue(engine)
            result+= (compact_dialogue_html(engine),render_dialogue_html(payload),payload)
        return result

    def start(engine,image=None):
        require_ready()
        if engine is not None: engine.export_session()
        replacement=factory()
        try: start_ui_world(replacement,image,live_current_frame=live_current_frame,story_lineage_guard=story_lineage_guard,
                            configured_image=os.environ.get('IW_LAB_INITIAL_IMAGE'))
        except Exception as exc: raise gr.Error('初始化失败，请检查服务或起始素材：'+str(exc)[:250])
        replacement.export_session()
        return view(replacement)

    def interact(text,engine):
        require_open_handoff()
        if not isinstance(text,str) or not text.strip(): raise gr.Error('请输入一句行动。')
        ready,message=service_status(mode)
        if not ready: raise gr.Error(message)
        engine=engine or factory()
        try: require_ui_frame(engine,live_current_frame=live_current_frame,story_lineage_guard=story_lineage_guard)
        except ValueError as exc: raise gr.Error(str(exc))
        try: row=engine.interact(text)
        except ValueError as exc: raise gr.Error(str(exc))
        engine.export_session()
        return (*view(engine,row),'')

    def queue_game_input(text,engine,input_mode):
        require_open_handoff()
        return queue_text_input(text,engine,input_mode)

    def game_interact(text,engine,input_mode):
        require_open_handoff()
        # Keep view()/all old API output tuples unchanged. The game-only event
        # adds its status field and calls the very same interaction callback.
        for update in submission_updates(text,engine,interact,input_mode):
            values=update.result[:-1] if update.result is not None else tuple(gr.skip() for _ in outputs)
            yield (*values,update.text,status_html(update.phase,update.detail))

    def game_input_mode(mode):
        hint=input_hint(mode)
        return gr.update(placeholder=hint),hint

    def recover_initial(engine):
        require_open_handoff()
        if current_rgb_entry:raise gr.Error('当前帧模式不复用其他会话的开场。')
        require_ready()
        if engine is not None and (engine.history or engine.frame is not None):
            raise gr.Error('已有活动会话，不覆盖。')
        replacement=factory()
        replacement.recover_initial(os.environ['IW_RECOVER_INITIAL_SESSION'])
        return view(replacement)

    def check_service():
        ready,message=service_status(mode)
        return message,gr.update(interactive=ready)

    def current_image_preview(engine):
        media=live_media_state(engine)
        return gr.update(value=media['image'],visible=media['image_visible'])

    def room_scene_cut(engine):
        require_open_handoff()
        # An explicit CPU-only authored cut, not another interact/model handler.
        if not story_lineage_guard or not room_cut_manifest:
            raise gr.Error('此入口未启用明确声明的门内场景切换。')
        if engine is None or not engine.history or not engine.frame:
            raise gr.Error('请先在当前会话完成进入房间的规则动作。')
        from .live_continuation import rgb_sha256
        operation='declared-room-cut:'+engine.session_id+':'+str(engine.history[-1]['turn_id'])
        saved=engine.declared_scene_cut_operations.get(operation)
        before_sha=(saved['signature']['expected_current_rgb_sha256'] if saved else rgb_sha256(engine.frame))
        try:
            engine.declare_room_scene_cut(engine.state.player.location,operation_id=operation,
                expected_current_rgb_sha256=before_sha,manifest_path=room_cut_manifest,enabled=True)
        except (ValueError,OSError) as exc:
            raise gr.Error('门内场景未切换，当前画面与剧情保留：'+str(exc)[:250])
        engine.export_session()
        return view(engine,engine.history[-1])

    def import_opening(engine):
        require_open_handoff()
        if current_rgb_entry:raise gr.Error('当前帧模式不导入预生成开场。')
        require_ready()
        if engine is not None and (engine.history or engine.frame is not None):
            raise gr.Error('已有活动会话，不覆盖。')
        replacement=factory()
        try: replacement.import_initial_archive(os.environ['IW_INITIAL_IMPORT_MANIFEST'])
        except Exception as exc: raise gr.Error('开场复用中止，未恢复旧剧情或生成新视频：'+str(exc)[:250])
        return view(replacement)

    def resume_saved(engine):
        require_open_handoff()
        if current_rgb_entry:raise gr.Error('当前帧模式不替换为历史会话。')
        from .session_resume import resume_archive
        if engine is not None and (engine.history or engine.frame is not None):
            raise gr.Error('已有活动会话，不覆盖。')
        replacement=factory()
        try:
            resume_archive(replacement,os.environ['IW_RESUME_SESSION'],os.environ['IW_RESUME_ARCHIVE_SHA256'],
                allow_demo_advisory=os.environ.get('IW_ALLOW_DEMO_ADVISORY_MIGRATION')=='1')
        except Exception as exc: raise gr.Error('恢复中止，未重新生成或提交动作：'+str(exc)[:250])
        return view(replacement,replacement.history[-1])

    def resume_story(engine):
        raise gr.Error('公开版不包含私有部署会话恢复；未生成或重新提交动作。')

    def export(engine):
        if engine is None: raise gr.Error('请先开始一个会话。')
        return engine.export_session()

    def require_ready():
        require_open_handoff()
        ready,message=service_status(mode)
        if not ready: raise gr.Error(message)

    def plan_goal(goal,engine):
        require_ready()
        if engine is None or engine.frame is None: raise gr.Error('请先点击 Start Demo，再拆解目标。')
        try:
            engine.replace_goal(goal)
        except Exception as exc: raise gr.Error('目标拆解未完成，世界未改变：'+str(exc)[:300])
        engine.export_session()
        return view(engine,engine.history[-1] if engine.history else None)

    def execute_goal(engine,all_steps=False):
        require_open_handoff()
        if engine is None or engine.task is None: raise gr.Error('请先拆解并检查目标计划。')
        if engine.task.status not in ('ready','running'): raise gr.Error('任务已暂停、结束或需要澄清，请先查看右侧原因。')
        require_ready()
        for _ in range(12 if all_steps else 1):
            if engine.task.status not in ('ready','running'): break
            try: row=engine.task.advance(engine)
            except Exception as exc:
                engine.task.status='paused'; engine.task.message='执行异常，任务暂停：'+str(exc)[:250]
                engine.export_session()
                yield view(engine); return
            engine.export_session()
            yield view(engine,row)
            if engine.task.status not in ('ready','running'): break

    def execute_all(engine):
        yield from execute_goal(engine,all_steps=True)

    def pause_goal(engine):
        require_open_handoff()
        if engine is not None and engine.task is not None:
            engine.task.pause_requested.set()
        gr.Info('已请求暂停：正在执行的步骤完成后停下，不会强行中断模型或回滚已提交动作。')

    def pause_for_manual_input(engine):
        require_open_handoff()
        if engine is not None and engine.task is not None and engine.task.status in ('ready','running'):
            engine.task.pause_requested.set()

    def resume_goal(engine):
        require_open_handoff()
        if engine is None or engine.task is None: raise gr.Error('当前没有任务。')
        # The task itself rejects dependent actions while visual review is pending.
        # Pure observation goals remain usable to inspect an unresolved scene.
        try: engine.task.resume()
        except ValueError as exc: raise gr.Error(str(exc))
        engine.export_session()
        return view(engine,engine.history[-1] if engine.history else None)

    def retry_visual(engine):
        require_open_handoff()
        if engine is None: raise gr.Error('当前没有会话。')
        require_ready()
        try: engine.retry_visual()
        except ValueError as exc: raise gr.Error(str(exc))
        return view(engine,engine.history[-1] if engine.history else None)

    def accept_visual(note,engine):
        require_open_handoff()
        if engine is None: raise gr.Error('当前没有会话。')
        try: engine.resolve_visual(note)
        except ValueError as exc: raise gr.Error(str(exc))
        return view(engine,engine.history[-1] if engine.history else None)

    def review_visual(decision,note,engine):
        require_open_handoff()
        if engine is None: raise gr.Error('当前没有会话。')
        try: engine.resolve_visual(note,decision=decision)
        except ValueError as exc: raise gr.Error(str(exc))
        return view(engine,engine.history[-1] if engine.history else None)

    def refresh_versions(engine):
        choices=session_versions(engine)
        ids=[item[1] for item in choices]
        return (gr.update(choices=choices,value=ids[0] if ids else None),
            gr.update(choices=choices,value=ids[-1] if ids else None),session_details(engine),
            getattr(engine,'selected_output',None) if engine else None)

    def choose_delivery(key,engine):
        require_open_handoff()
        if engine is None: raise gr.Error('当前没有会话。')
        try: selection=engine.select_output(key)
        except ValueError as exc: raise gr.Error(str(exc))
        return selection,engine.export_session()

    def start_model_services():
        from .runtime_control import start_models
        try: return start_models()
        except Exception as exc: return '模型服务未启动：'+str(exc)[:300]+'。没有提交任何视频生成。'

    def world_state_components():
        # Same components/outputs for both entries. Only the advisory game's
        # parent changes; the preserved strict UI keeps its original side column.
        with (gr.Accordion('状态与执行详情',open=False,elem_id='iw-game-details') if game_enabled else nullcontext()):
            summary=gr.Markdown(closed_reason() if closed else '点击 Start Demo 开始。',elem_id='iw-game-summary' if game_enabled else None)
            with gr.Accordion('World State（规则真值，不是视觉评测）',open=mode!='cinematic' and not game_enabled):
                state=gr.JSON(label='World State')
                if evidence_reader is not None:
                    gr.Markdown('**生成能力证据（只读）**：分开记录接口、方向、结构和目标可见性。当前抽屉与俯仰未联合达标；不是运行权重认证，也未启用新技能。')
                    capability=gr.JSON(value=evidence_reader(),label='能力版本、未通过原因与固定输入引用')
                    capability_refresh=gr.Button('刷新能力证据（不调用模型）',size='sm')
                    capability_refresh.click(evidence_reader,[],capability,api_name='renderer_evidence')
        return summary,state

    stage_css=(GAME_STAGE_CSS+(LIVE_CURRENT_FRAME_CSS if current_rgb_entry else '')
               if game_enabled else None)
    with gr.Blocks(title='InteractWorld-Agent',theme=gr.themes.Soft(),css=stage_css) as app:
        session=gr.State(value=None)
        clip_seconds=int(os.environ.get('IW_LAB_SECONDS','3'))
        visual_banner=(f'**真实 Wan 实验室视频模式**：每轮新生成约 {clip_seconds} 秒，以上一轮末帧续写；异步等待，非实时。'
            '\n\n起始图是生成式素材，不是 Wan 输出。模型没有实验室专用训练，物品交互、人物与空间一致性仍可能失败。'
            '\n\n剧情由规则驱动；画面由 VLM 单独检查，画面失败不会被包装成视觉成功。' if mode=='wan' else
            '**当前视觉模式：MOCK 示意动画，不是 Wan 生成。剧情状态不是视觉验证。**')
        if mode=='cinematic':
            visual_banner=('**互动影游 · 实验室事故**：输入你的行动，Qwen 实时理解，规则校验后播放对应事件分镜。'
                '\n\n**视频为预生成素材，不是本次输入后实时生成。** 由外部设计关键帧、Wan 动画及少量明确标注的剧情画面组成；分镜切换不代表连续三维导航。'
                '\n\n钥匙、开门、终端和房间变化已接入实际游玩；[逐轮生成研究版](http://127.0.0.1:8879/)单独保留。')
        if game_enabled:
            gr.Markdown('# InterActWorld · 当前帧相机验证' if live_current_frame else '# InterActWorld · 实验室事故')
        else:
            gr.Markdown('# InteractWorld-Agent\n文字交互 · 状态约束 · 剧情分支 · 记忆\n\n'+visual_banner
                +'\nAgent：'+('真实 Qwen LLM' if mode in ('llm','wan','cinematic') else '离线规则解析（不是 LLM）'))
        if closed:
            gr.Markdown(closed_reason())
        if (os.environ.get('IW_GOAL_CONTRACT_NORMALIZATION','0')=='1'
                and os.environ.get('IW_GOAL_CONTRACT_VALIDATION')=='npc-observation-cancel-v1' and not game_enabled):
            gr.Markdown('**独立功能回归验证 · npc-observation-cancel-v1**：窄 typed-tool 契约编译显式开启；'
                '保留真实模型原始步骤，不是模型自主修复。仅验证已知场景两步计划的边界中断；不是新产品页面、泛化或视觉成功。')
        if os.environ.get('IW_VISUAL_REVIEW_POLICY')=='demo_advisory' and not game_enabled:
            gr.Markdown('## 完整影游演示 · 实验室事故\n你来到一间废弃实验室，Anna 建议你跟随她调查。你可以相信她，也可以选择独自寻找线索。'
                '\n\n**本入口优先跑通故事**：R005 每步现场异步生成约六秒；规则前置条件、中途改口与已完成状态保留。画面问题仅作提示，取钥匙、开门细节不保证正确。'
                '\n\n可输入：我不相信 Anna，选择独自调查。搜索桌子，拿起红钥匙，用钥匙打开门，再进入储藏室，完成钥匙路线。已经完成的动作不要重复。'
                '\n\n部分完整路线可能使用已编写的技能兜底，界面会标注；这不是实时视频或纯模型自主完成。严格审核入口继续保留在 8893。')
        elif os.environ.get('IW_DELIVERY_VERSION')=='postdata6-v1' and not game_enabled and not live_current_frame:
            gr.Markdown('**影游交付固定使用 R005 step1040 动作模型，不等待新视频训练。** '
                '每段约六秒异步生成；实验性教师、因果和蒸馏分支不替换本入口。'
                '\n\n快速演示：开始场景 → 输入“走近 Anna，然后观察左边” → 执行中改为“别继续之前的计划，只观察右边” → 查看旧任务暂停与已提交状态 → 导出会话。'
                '这展示真实交互控制流程，不预先保证画面方向正确。取钥匙/开门保留原流程与视觉审核，不用规则通过冒充画面成功。')
            gr.Markdown('**六秒互动候选版**：多步目标使用新目标 LoRA；单步决策与视觉观察使用 Base。'
                '\n\n专项6例系统通过6例，其中模型4例、规则技能兜底2例，不是纯模型或视频成功率。'
                '\n\n[1336条任务数据与真实规划对照](http://127.0.0.1:8891/) · [六秒数据与一分钟诊断预览](http://127.0.0.1:8892/) · [保留的旧Demo](http://127.0.0.1:8886/)'
                '\n\n发现钥匙与穿门仍未视觉达标；一分钟视频是独立分镜诊断预览，不是完整成功故事。')
        if os.environ.get('IW_VISUAL_REVIEW_POLICY')=='manual_critical' and not live_current_frame:
            gr.Markdown('**人工审核降级版**：VLM 提供观察建议，不认证视觉成功。拿钥匙、开门、终端等关键步骤需要人工观看并确认后才能继续；不符合可以重试一次或改目标。')
        if os.environ.get('IW_DELIVERY_RECORDINGS') and not game_enabled:
            from .delivery_recordings import add_delivery_recordings
            add_delivery_recordings(gr,os.environ['IW_DELIVERY_RECORDINGS'])
        if mode in ('llm','wan','cinematic') and not game_enabled:
            gr.Markdown('自然语言由本地 Qwen 处理；右侧状态由规则校验后更新。若模型服务离线，会明确回退，不把回退算作模型成功。')
            if os.environ.get('IW_CPU_DEMO')=='1':
                gr.Markdown('**当前为 CPU 常驻模式，速度较慢：一条输入实测约 107 秒，实际随输入变化。请等待本次提交完成。** GPU 验证阶段的单条规划约 4–10 秒。')
        if visual_observer and mode not in ('wan','cinematic') and not game_enabled:
            gr.Markdown('VLM 观察已开启：当前读取的是示意图，只记录视觉证据，不改变背包、信任或剧情。')
        route_buttons=[]
        room_cut_button=None
        with (gr.Accordion('开始与路线',open=False,elem_id='iw-game-settings') if game_enabled else nullcontext()):
            if game_enabled:
                gr.Markdown('本会话当前帧逐段续写，纯交谈保留画面。三条剧情线保留；画面警告不代表视觉成功，异步生成，非实时。'
                    if story_lineage_guard else '当前帧交互验证：只从本会话当前画面生成左右观察。纯交谈不新增视频；取物、开门与空间移动尚未验收，不宣称完整故事可玩通。异步生成，非实时。'
                    if live_current_frame else '本地 Qwen 规划，R005 异步生成约六秒，非实时。纯交谈保留当前画面，只更新剧情台词。画面问题保留警告，规则结局不代表视觉成功。')
            service=gr.Markdown(closed_reason() if closed else '正在检查模型接口；历史回放不依赖模型在线。')
            refresh=gr.Button('刷新服务状态',size='sm')
            if os.environ.get('IW_MODEL_START_ENABLED')=='1':
                gr.Markdown(closed_reason() if closed else '模型默认按需启动，空闲后释放 GPU。先核验数据盘空间（共享盘繁忙时最多等 10 分钟，不占 GPU），再经授权、空闲卡及租约检查加载模型；通常加载需 1–2 分钟。超时明确提示，不自动反复重试。')
                model_start=gr.Button('本轮模型服务已结束' if closed else '按需启动模型服务（有限时长）',size='sm',interactive=not closed)
                model_start_status=gr.Markdown(closed_reason() if closed else '未请求启动。')
            else: model_start=None
            if game_enabled and not live_current_frame:
                with gr.Row(elem_id='iw-route-cards'):
                    for card in active_route_catalog():
                        with gr.Column():
                            button=gr.Button(card['route_id']+' · '+card['title'])
                            gr.Markdown(card['summary'])
                            route_buttons.append((card['route_id'],button))
                gr.Markdown('选路线只填目标，不自动执行或改状态。不同分支请开始新故事。'+
                    ('下方“历史记录回看”可不占 GPU 查看三线已存片段。' if os.environ.get('IW_GAME_ARCHIVE_MANIFEST') else ''))
            if story_lineage_guard and room_cut_manifest:
                room_cut_button=gr.Button('切换到门内场景',size='sm')
                gr.Markdown('需先完成进入房间的规则动作并结束当前任务；此按钮使用已声明美术图切镜，不生成视频、不认证跨门。')
        with gr.Row(elem_id='iw-game-layout' if game_enabled else None):
            with gr.Column(scale=3):
                if game_enabled:
                    with gr.Column(elem_id='iw-game-stage'):
                        with gr.Column(elem_id='iw-game-screen'):
                            if current_rgb_entry:
                                current_image=gr.Image(label='当前输入 RGB（不是生成视频）',type='filepath',
                                    show_label=False,interactive=False,show_download_button=False,
                                    visible=False,elem_id='iw-live-current-frame')
                            video=gr.Video(label='Storyboard / Generated Video',show_label=False,autoplay=False,
                                visible=not current_rgb_entry,elem_id='iw-live-video' if current_rgb_entry else None)
                            dialogue=gr.HTML(compact_dialogue_html({}),elem_id='iw-game-dialogue')
                        with gr.Column(elem_id='iw-game-prompt'):
                            with gr.Row(elem_id='iw-prompt-heading'):
                                gr.HTML('<span>USER PROMPT</span>',elem_id='iw-prompt-label')
                                input_status=gr.HTML(status_html(detail=closed_reason()) if closed else status_html(),elem_id='iw-game-input-status')
                                input_mode=gr.Radio(INPUT_MODES,value='action',show_label=False,
                                    container=False,elem_id='iw-input-mode')
                            with gr.Row(elem_id='iw-prompt-compose'):
                                text=gr.Textbox(label='Player Text',show_label=False,interactive=not closed,
                                    placeholder=input_hint('action'),lines=1,max_lines=2,
                                    container=False,elem_id='iw-game-text')
                                submit=gr.Button('发送',variant='primary',interactive=mode=='mock',
                                    elem_id='iw-game-send',scale=0)
                            input_mode_hint=gr.Markdown(input_hint('action'),elem_id='iw-input-mode-hint')
                        input_mode.change(game_input_mode,[input_mode],[text,input_mode_hint],
                            queue=False,api_name=False)
                else:
                    video=gr.Video(label='当前事件分镜（预生成）' if mode=='cinematic' else 'Storyboard / Generated Video',autoplay=mode=='cinematic')
                    chat=gr.Chatbot(label='Interaction',type='messages',height=240)
            if not game_enabled:
                with gr.Column(scale=2):
                    summary,state=world_state_components()
        with (gr.Accordion('设置与记录',open=False,elem_id='iw-game-records') if game_enabled else nullcontext()):
            if game_enabled:
                with gr.Accordion('对话历史与来源',open=False):
                    gr.Markdown('屏幕对话来自已提交的剧情台词，非语音或口型生成。纯交谈不重绘人物；移动、观察和物体动作仍请求 R005。未用静帧凑生成时长。')
                    dialogue_history=gr.HTML(render_dialogue_html(build_game_dialogue({})))
                    dialogue_data=gr.JSON(label='台词、旁白与玩家输入的实际来源')
                    chat=gr.Chatbot(label='Interaction',type='messages',height=240)
                summary,state=world_state_components()
                if (os.environ.get('IW_GOAL_CONTRACT_NORMALIZATION','0')=='1'
                        and os.environ.get('IW_GOAL_CONTRACT_VALIDATION')=='npc-observation-cancel-v1'):
                    with gr.Accordion('功能验证说明',open=False):
                        gr.Markdown('**独立功能回归验证 · npc-observation-cancel-v1**：窄 typed-tool 契约编译显式开启；保留真实模型原始步骤，不是模型自主修复。仅验证已知场景两步计划的边界中断；不是泛化或视觉成功。')
                if visual_observer and mode not in ('wan','cinematic'):
                    gr.Markdown('VLM 观察已开启：当前读取的是示意图，只记录视觉证据，不改变背包、信任或剧情。')
            if not game_enabled:
                text=gr.Textbox(label='Player Text',placeholder='例如：进入实验室，然后走近 Anna',max_lines=3)
            with (gr.Accordion('起始画面设置',open=False) if game_enabled else nullcontext()):
                image=gr.Image(type='filepath',label=('当前 RGB（精确832×480；不会重生成开场）' if current_rgb_entry
                    else '可选起始图（不改变实验室剧情设定；留空使用默认图）'),visible=mode=='wan')
            with gr.Row():
                if not game_enabled:
                    submit=gr.Button('Submit Action',variant='primary',interactive=mode=='mock')
                reset=gr.Button('从当前画面开始 / 重置' if current_rgb_entry else 'Reset / Start Demo')
                recover_start=(gr.Button('恢复已生成开场（不重新生成）')
                               if mode=='wan' and os.environ.get('IW_RECOVER_INITIAL_SESSION') and not current_rgb_entry else None)
            with gr.Accordion('目标任务：拆解 → 执行 → 反馈修正',open=not game_enabled):
                gr.Markdown('真实模型拆解最多 8 步。计划先由规则模拟检查，再由你选择执行。'
                    '规则失败最多自动重规划两次；视觉不符或关键动作证据不足会暂停。最多 12 次执行尝试。'
                    '仅支持当前实验室，不支持桥梁/山地等新世界；这不是 LongCoT 或 RL。')
                goal_text=gr.Textbox(label='本次目标',placeholder='例如：我不信任 Anna，帮我找到钥匙，打开门并进入储藏室。')
                if game_enabled:
                    for route_id,button in route_buttons:
                        button.click(lambda r=route_id:active_route_goal(r),[],goal_text,queue=False,api_name='select_route_'+route_id.lower())
                with gr.Row():
                    goal_plan=gr.Button('拆解／替换目标（不执行）')
                    goal_next=gr.Button('执行下一步')
                    goal_all=gr.Button('执行剩余计划（有界反馈闭环）')
                with gr.Row():
                    goal_pause=gr.Button('当前步骤结束后暂停')
                    goal_resume=gr.Button('确认检查后恢复任务')
                gr.Markdown('生成期间可提交新目标，当前短片段结束后取消旧计划。已经拿到的物品不会回退或再次领取。')
            with gr.Accordion('视觉失败处理：不重复提交剧情',open=False):
                gr.Markdown('视觉重试从该动作前的画面重新生成，最多一次，不重复改背包或剧情。若已产生后续画面，则拒绝覆盖。人工确认单独记录，不改写原模型判断。')
                visual_retry=gr.Button('重试本步画面（最多一次）')
                review_note=gr.Textbox(label='人工检查说明',placeholder='说明实际看见了什么，或为何接受这个表现。')
                review_decision=gr.Radio(choices=[('接受当前呈现并允许继续（不等于视觉真值）','accept'),
                    ('不符合，保持暂停','reject'),('看不清，保持暂停','uncertain')],
                    value='uncertain',label='审核决定（默认不放行）')
                visual_review=gr.Button('保存人工审核决定')
            with gr.Accordion('Agent 决策、执行和记忆',open=mode!='cinematic' and not game_enabled):
                with gr.Row():
                    debug=gr.JSON(label='Agent Action / Validation')
                    memory=gr.JSON(label='Memory / Event Timeline')
                export_button=gr.Button('导出本次会话记录',size='sm')
                export_file=gr.File(label='会话 JSON（不包含模型权重）',interactive=False)
            with gr.Accordion('版本比较与最终选择（不回滚世界）',open=False):
                gr.Markdown('按本会话真实产生的片段与视觉重试比较。失败步骤沿用的旧视频不会冒充新版本；选择交付片段只保存偏好，不回滚背包、剧情或当前续写帧。')
                versions_refresh=gr.Button('刷新本会话版本／任务记录',size='sm')
                with gr.Row():
                    with gr.Column():
                        version_a=gr.Dropdown(choices=[],label='原版本／参考版本')
                        compare_a=gr.Video(label='参考片段')
                        metadata_a=gr.JSON(label='参考版本来源与判断')
                    with gr.Column():
                        version_b=gr.Dropdown(choices=[],label='待选版本')
                        compare_b=gr.Video(label='待选片段')
                        metadata_b=gr.JSON(label='待选版本来源与判断')
                choose_version=gr.Button('选右侧版本作为交付片段（不改当前世界）')
                chosen=gr.JSON(label='已选择的结果')
                selected_export=gr.File(label='带结果选择的会话记录',interactive=False)
                task_details=gr.JSON(label='当前计划、重规划、旧目标与待审核事项')
            if mode=='cinematic':
                gr.Examples(examples=[['走近 Anna'],['我不相信 Anna，想独自调查'],['搜索桌子'],['拿起红钥匙'],['用钥匙打开门'],['进入储藏室'],['帮助 Anna'],['询问终端密码'],['检查终端'],['读取公开日志'],['找到 Leo']],inputs=text,
                    label='也可点击一句，再提交；仍由实时语言模型理解并校验，不能跳过前置条件')
                replay=Path(os.environ['IW_CINEMATIC_MANIFEST']).parent/'walkthroughs/index.json'
                if replay.is_file():
                    with gr.Accordion('直接观看三条完整路线回放',open=True):
                        gr.Markdown('根据真实 Qwen 网页游玩的决策记录，使用预生成分镜合成；不是实时生成或浏览器录屏。字幕区分玩家输入与规则事件。')
                        with gr.Row():
                            for item in json.loads(replay.read_text()):
                                gr.Video(value=item['path'],label='路线 '+item['route']+f' · {item["seconds"]:.1f} 秒',autoplay=False)
            with (gr.Accordion('路线与游玩说明',open=False) if game_enabled else nullcontext()):
                gr.Markdown('路线 A：走近 Anna → 帮助 Anna → 询问密码 → 检查终端 → 进入核心区域。  '
                    '\n路线 B：走近 Anna → 我不相信 Anna → 搜索桌子 → 拿起钥匙 → 打开门 → 进入储藏室。  '
                    '\n路线 C：检查终端 → 读取终端日志 → 找到 Leo → 询问 Leo。  '
                    '\n普通输入一次执行一步；多步骤目标请展开“目标任务”。“计划有效”不等于真实视频动作有效。')
            if game_enabled and os.environ.get('IW_GAME_ARCHIVE_MANIFEST'):
                archive=GameArchive(os.environ['IW_GAME_ARCHIVE_MANIFEST'])
                with gr.Accordion('三路线真实记录回看（只读，与当前游玩分开）',open=False):
                    gr.Markdown('A/C 是此前真实 Qwen/Wan 实验，B 是 10 月 7 日实际浏览器流程。不是这次点击后生成；各片时长、来源和视觉限制单独保留。回看不恢复或修改当前故事。')
                    archived_route=gr.Dropdown(choices=archive.choices(),value='B',label='回看路线')
                    archive_choices=archive.turns('B')
                    archived_turn=gr.Dropdown(choices=archive_choices,value=archive_choices[0][1] if archive_choices else None,label='实际片段')
                    initial_archive=archive.view('B',archive_choices[0][1]) if archive_choices else (None,'',{}, {})
                    with gr.Column(elem_id='iw-archive-stage'):
                        archived_video=gr.Video(value=initial_archive[0],label='历史真实生成片段（不是现场生成）')
                        archived_dialogue=gr.HTML(value=initial_archive[1],elem_id='iw-archive-dialogue')
                    with gr.Accordion('该片对话与来源记录',open=False):
                        archived_lines=gr.JSON(label='本轮实际台词')
                        archived_meta=gr.JSON(label='历史来源/模型/时长/限制')
                    def archive_route_changed(route):
                        choices=archive.turns(route)
                        return gr.update(choices=choices,value=choices[0][1] if choices else None)
                    archived_route.change(archive_route_changed,[archived_route],archived_turn,api_name='archive_route',queue=False)
                    archived_turn.change(archive.view,[archived_route,archived_turn],
                        [archived_video,archived_dialogue,archived_lines,archived_meta],api_name='archive_dialogue',queue=False)
            add_data_workbench(gr,os.environ.get('IW_FEEDBACK_DATA'))
            if os.environ.get('IW_REVIEWED_LIBRARY'):
                from .reviewed_library import add_reviewed_library
                add_reviewed_library(gr,os.environ['IW_REVIEWED_LIBRARY'])
            if os.environ.get('IW_DELIVERY_VERSION')=='postdata6-v1':
                from .session_archive import add_session_archive
                add_session_archive(gr,output_root,evidence_reader=evidence_reader)
            if os.environ.get('IW_DISABLE_LEGACY_RUN_PANELS')!='1':
                add_run_panel(gr,os.environ.get('IW_FEEDBACK_RUNS'))
                add_run_panel(gr,os.environ.get('IW_FEEDBACK_PREVIOUS_RUNS'),label='上一轮保存结果（历史证据，不是本次执行）')
            if evidence_dir:
                evidence=Path(evidence_dir)
                if mode=='wan':
                    with gr.Accordion('已保存的真实实验室路线（规则结果与画面判断分开）',open=not game_enabled):
                        found=False
                        for report_path in sorted(evidence.glob('loop-*.json')):
                            route=json.loads(report_path.read_text())
                            if 'session' not in route: continue
                            presentation=Path(route['session'])/'route-presentation.mp4'
                            if not route.get('complete') or not presentation.is_file(): continue
                            found=True
                            gr.Video(value=str(presentation),label='真实续写 · Route '+route['route'],autoplay=False)
                            review_file=report_path.with_name(report_path.stem+'-visual-review.json')
                            reviews={v['turn_id']:v['assessment'] for v in json.loads(review_file.read_text())['reviews']} if review_file.exists() else {}
                            rows=[[t['player_text'],t['state']['story']['node'],visual_labels.get(reviews.get(t['turn_id'],t.get('visual_assessment',{})).get('status'),'未检查')]
                                for t in route['turns']]
                            gr.Dataframe(value=rows,headers=['实际输入','规则剧情节点','VLM 判断（不是人工真值）'],interactive=False)
                        if not found: gr.Markdown('路线仍在生成；不会把旧视频或 Mock 当作本轮结果。')
                    if (evidence/'door-framing-six/result.json').is_file():
                        with gr.Accordion('开门失败的构图/时长对照（探索样例，不是成功率）',open=False):
                            gr.Markdown('同一初始素材与门提示词、seed 73、CFG 5、关闭 R005 动作适配器。'
                                '“目标镜头”是手工选择的二维裁剪，不是 Agent 自动定位或三维导航。'
                                '6 秒版本只延长时长；请直接观察门有没有真实打开，不以剧情标记代替。')
                            with gr.Row():
                                for label,relative in [('全景 · 3 秒','door-framing-pilot/full/video.mp4'),
                                    ('目标镜头 · 3 秒','door-framing-pilot/target/video.mp4'),
                                    ('目标镜头 · 6 秒','door-framing-six/output/video.mp4')]:
                                    gr.Video(value=str(evidence/relative),label=label,autoplay=False)
                with gr.Accordion('真实模型实验与视频（单独展示，不冒充实验室生成）',open=False):
                    report_path=evidence/('base-vs-sft-live.json' if mode=='wan' else 'base-vs-sft.json')
                    if report_path.is_file():
                        comparison=json.loads(report_path.read_text())
                        labels={'json_valid':'JSON 格式正确','action_correct':'动作严格匹配','target_correct':'目标严格匹配',
                            'state_consistency':'动作/目标/前置条件同时正确','invalid_action_detection':'非法请求识别召回',
                            'memory_consistency':'记忆样例动作/目标正确'}
                        rows=[]
                        for key,label in labels.items():
                            pair=[comparison[k]['metrics'][key] for k in ('base','sft')]
                            rows.append([label]+[f'{m["correct"]}/{m["total"]}'+(f' ({m["rate"]:.1%})' if m['rate'] is not None else '') for m in pair])
                        gr.Dataframe(value=rows,headers=['指标','Base','SFT'],interactive=False)
                        gr.Markdown(('新规划器实际训练 159 步；同契约、同批量的 153 条开发集对照。联合状态和显式非法判断改善，但动作、目标、记忆回退，因此仍使用 Base。' if mode=='wan' else
                            '固定种子家族划分的 153 条单世界改写样例，对应提示词 v1；当前工具契约提示词 v2 另做路线测试。')+
                            '不是开放世界/视频质量基准；记忆项不是因果消融。显式非法判断统计布尔字段，不等于实际越权次数。')
                    videos=[p.parent for p in sorted(evidence.glob('wan-check-*/report.json')) if (p.parent/'video.mp4').is_file()]
                    if videos:
                        wan=videos[-1]
                        display=wan/'inputs.mp4' if (wan/'inputs.mp4').is_file() else wan/'video.mp4'
                        gr.Video(value=str(display),label='真实 Wan R005 · 山地场景 · 15s',autoplay=False)
                        gr.Markdown('已保存的真实 LLM 规划 → 方向键编译 → 新生成视频。不是实时控制，也不是实验室/NPC 视觉完成证据。')
                        gr.JSON(value=json.loads((wan/'report.json').read_text()),label='真实生成记录')
            outputs=[session,video,summary,state,debug,memory,chat]
            if game_enabled: outputs += [dialogue,dialogue_history,dialogue_data]
            # A queued manual action must request interruption before waiting for the runner.
            if game_enabled:
                manual_click=submit.click(queue_game_input,[text,session,input_mode],[input_status],queue=False,api_name=False)
                manual_enter=text.submit(queue_game_input,[text,session,input_mode],[input_status],queue=False,api_name=False)
            else:
                manual_click=submit.click(pause_for_manual_input,[session],[],queue=False,api_name=False)
                manual_enter=text.submit(pause_for_manual_input,[session],[],queue=False,api_name=False)
            events=[]
            if room_cut_button is not None:
                events.append(room_cut_button.click(room_scene_cut,[session],outputs,
                    api_name='room_scene_cut',concurrency_limit=1,concurrency_id='interaction'))
            events.append(goal_plan.click(pause_for_manual_input,[session],[],queue=False,api_name=False).then(
                plan_goal,[goal_text,session],outputs,api_name='plan_goal',concurrency_limit=1,concurrency_id='interaction'))
            events.append(visual_retry.click(retry_visual,[session],outputs,api_name='visual_retry',concurrency_limit=1,concurrency_id='interaction'))
            events.append(visual_review.click(review_visual,[review_decision,review_note,session],outputs,api_name='visual_review',concurrency_limit=1,concurrency_id='interaction'))
            # Keep the old explicit approval API for existing clients. It still requires a note.
            legacy_accept=gr.Button(visible=False)
            events.append(legacy_accept.click(accept_visual,[review_note,session],outputs,api_name='visual_accept',concurrency_limit=1,concurrency_id='interaction'))
            events.append(goal_next.click(execute_goal,[session],outputs,api_name='goal_next',concurrency_limit=1,concurrency_id='interaction'))
            events.append(goal_all.click(execute_all,[session],outputs,api_name='goal_run',concurrency_limit=1,concurrency_id='interaction'))
            goal_pause.click(pause_goal,[session],[],queue=False,api_name='goal_pause')
            events.append(goal_resume.click(resume_goal,[session],outputs,api_name='goal_resume',concurrency_limit=1,concurrency_id='interaction'))
            app.load(check_service,[],[service,submit],queue=False,api_name=False)
            refresh.click(check_service,[],[service,submit],queue=False,api_name='service_status')
            if model_start is not None:
                model_start.click(start_model_services,[],[model_start_status],api_name='start_model_services',
                    concurrency_limit=1,concurrency_id='model_start').then(check_service,[],[service,submit],api_name=False)
            export_button.click(export,[session],[export_file],api_name='export_session',
                concurrency_limit=1,concurrency_id='interaction')
            events.append(reset.click(pause_for_manual_input,[session],[],queue=False,api_name=False).then(
                start,[session,image],outputs,api_name='start',concurrency_limit=1,concurrency_id='interaction'))
            if os.environ.get('IW_RESUME_SESSION') and mode=='wan' and not current_rgb_entry:
                resume_button=gr.Button('恢复已保存进度（不重新生成）')
                events.append(resume_button.click(resume_saved,[session],outputs,
                    api_name='resume_saved',concurrency_limit=1,concurrency_id='interaction'))
            if recover_start is not None:
                events.append(recover_start.click(recover_initial,[session],outputs,
                    api_name='recover_initial',concurrency_limit=1,concurrency_id='interaction'))
            if os.environ.get('IW_INITIAL_IMPORT_MANIFEST') and mode=='wan' and not current_rgb_entry:
                gr.Markdown('**开场只读复用**：复用已生成的真实 R005 开场建立空状态新故事；后续片段现场异步生成。开场不是本次新生成。')
                import_start=gr.Button('复用已生成开场，开始新故事')
                events.append(import_start.click(import_opening,[session],outputs,
                    api_name='import_initial',concurrency_limit=1,concurrency_id='interaction'))
            if game_enabled:
                events.append(manual_click.then(game_interact,[text,session,input_mode],outputs+[text,input_status],
                    api_name='game_interact',concurrency_limit=1,concurrency_id='interaction'))
                events.append(manual_enter.then(game_interact,[text,session,input_mode],outputs+[text,input_status],
                    api_name=False,concurrency_limit=1,concurrency_id='interaction'))
                # Preserve the existing /interact API schema and behavior, without
                # registering a second action handler on the visible send button.
                legacy_submit=gr.Button(visible=False)
                events.append(legacy_submit.click(pause_for_manual_input,[session],[],queue=False,api_name=False).then(
                    interact,[text,session],outputs+[text],api_name='interact',
                    concurrency_limit=1,concurrency_id='interaction'))
            else:
                events.append(manual_click.then(interact,[text,session],outputs+[text],api_name='interact',concurrency_limit=1,concurrency_id='interaction'))
                events.append(manual_enter.then(interact,[text,session],outputs+[text],api_name=False,concurrency_limit=1,concurrency_id='interaction'))
            version_outputs=[version_a,version_b,task_details,chosen]
            versions_refresh.click(refresh_versions,[session],version_outputs,api_name='session_versions',concurrency_limit=1,concurrency_id='interaction')
            version_a.change(version_view,[session,version_a],[compare_a,metadata_a],api_name=False,concurrency_limit=1,concurrency_id='interaction')
            version_b.change(version_view,[session,version_b],[compare_b,metadata_b],api_name=False,concurrency_limit=1,concurrency_id='interaction')
            choose_version.click(choose_delivery,[version_b,session],[chosen,selected_export],api_name='select_output',concurrency_limit=1,concurrency_id='interaction')
            for event in events:
                event.then(refresh_versions,[session],version_outputs,api_name=False,concurrency_limit=1,concurrency_id='interaction')
                if game_enabled and current_rgb_entry:
                    # Separate presentation update preserves every existing API
                    # output count. A real video covers the input immediately;
                    # this image is never encoded into a dummy opening clip.
                    event.then(current_image_preview,[session],[current_image],api_name=False,
                        concurrency_limit=1,concurrency_id='interaction')
        if closed:
            # Disable only action/state-changing controls. Existing archive,
            # status, download, version viewing and export callbacks stay intact.
            for control in (submit,reset,recover_start,goal_plan,goal_next,goal_all,goal_pause,
                            goal_resume,visual_retry,visual_review,legacy_accept,room_cut_button,
                            choose_version,*(button for _,button in route_buttons)):
                if control is not None: control.interactive=False
            for name in ('story_resume_button','resume_button','import_start'):
                if name in locals(): locals()[name].interactive=False
    return app
