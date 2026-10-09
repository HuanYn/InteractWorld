from pathlib import Path
import json
import threading
import time
import uuid
from .feedback import visual_problem, may_act_while_pending, assess_prebuilt
from .models import WorldState, Observation
from .rules import RuleEngine
from .story import StoryGraph
from .memory import Memory
from .agent import MockAgent, SkillSelector
from .renderers import MockRenderer
from .observer import MockObserver, Reconciler


class InteractionEngine:
    def __init__(self,output_root,agent=None,renderer=None,observer=None,visual_review_policy='model_gate',story_profile='legacy'):
        if visual_review_policy not in ('model_gate','manual_critical','demo_advisory'):
            raise ValueError('Unknown visual review policy')
        self.visual_review_policy=visual_review_policy
        if story_profile not in ('legacy','spatial_v1'):
            raise ValueError('Unknown story profile')
        self.story_profile=story_profile
        self.root=Path(output_root).resolve(); self.root.mkdir(parents=True,exist_ok=True)
        self.agent=agent or MockAgent(); self.renderer=renderer or MockRenderer()
        if getattr(self.renderer,'live_current_frame',False) and visual_review_policy=='demo_advisory':
            raise ValueError('Current-frame interaction cannot bypass visual pauses with demo_advisory')
        if hasattr(self.agent,'story_profile'):
            self.agent.story_profile=story_profile
        self.observer=observer or MockObserver()
        if story_profile=='spatial_v1':
            from .spatial_interactions import SpatialRuleEngine,SpatialStoryGraph
            self.rules=SpatialRuleEngine(); self.graph=SpatialStoryGraph()
        else:
            self.rules=RuleEngine(); self.graph=StoryGraph()
        self.selector=SkillSelector(); self.reconciler=Reconciler(); self.lock=threading.RLock()
        self.reset()

    def reset(self):
        with self.lock:
            if getattr(self.renderer,'story_lineage_guard',False) and getattr(self.renderer,'_story_request_active',False):
                raise ValueError('Cannot reset a story during an active render request')
            self.state=WorldState(); self.memory=Memory(); self.history=[]
            self.task=None
            self.desk_inspection=None
            self.completed_operations={}; self.render_contexts={}; self.pending_visual=None
            self.visual_reviews=[]; self.pending_visual_queue=[]; self.selected_output=None
            self.visual_warnings=[]
            self.task_history=[]
            self.goal_planner_calls=[]
            self.initial_recovery=None
            self.initial_current_frame=None
            self.session_resume=None
            self.declared_scene_cuts=[]; self.declared_scene_cut_operations={}
            self.observation=Observation(); self.frame=None; self.video=None
            self.session_id=uuid.uuid4().hex; self.folder=self.root/self.session_id; self.folder.mkdir()
            if getattr(self.renderer,'live_current_frame',False):
                self.renderer.reset_live_session(self.session_id)
            if getattr(self.renderer,'story_lineage_guard',False):
                self.renderer.reset_story_session(self.session_id)
            return self.get_state()

    def _bind_live_frame(self,frame,*,initial=False):
        if getattr(self.renderer,'story_lineage_guard',False):
            if initial:raise ValueError('Story lineage starts from explicit exact RGB, not a regenerated opening')
            return self.renderer.bind_story_frame(self.session_id,frame,session_dir=self.folder)
        if not getattr(self.renderer,'live_current_frame',False):return None
        return self.renderer.bind_current_frame(self.session_id,frame,
            initial=initial,session_dir=self.folder)

    def _check_live_result(self,result,binding):
        if binding is None:return
        key='story_lineage_evidence' if getattr(self.renderer,'story_lineage_guard',False) else 'live_current_frame_evidence'
        evidence=result.metadata.get(key,{})
        if (evidence.get('session_id')!=self.session_id or not evidence.get('lineage_check_passed')
                or not evidence.get('conditioning_rgb_match') or evidence.get('goal_condition_used')
                or evidence.get('pre_generated_used')):
            raise ValueError('Current-frame renderer did not verify this session input/actions')
        if (binding.frame_rgb_sha256 is not None and
                evidence.get('actual_previous_last_rgb_sha256')!=binding.frame_rgb_sha256):
            raise ValueError('Returned evidence belongs to a different current frame')
        if (not result.last_frame or not Path(result.last_frame).resolve().is_relative_to(self.folder)):
            raise ValueError('Returned current frame is outside this interaction session')

    def _live_action_rejection(self,plan):
        if not getattr(self.renderer,'live_current_frame',False):return None
        if plan.semantic_action in ('inspect','observe') and plan.target in ('left','right'):
            return None
        if plan.semantic_action in ('talk','ask','interact') and plan.target in ('anna','leo'):
            return None
        if plan.semantic_action in ('observe','wait') and plan.target in ('room','laboratory'):
            return None
        return ('当前严格交互入口尚未验证这个动作的画面能力；未生成替代片段、未提交剧情。'
                '查看物体需先明确从当前画面向左或向右转；拿取、开门、接近及进入仍待真实视觉验收。')

    def start_world(self,image=None):
        with self.lock:
            if getattr(self.renderer,'story_lineage_guard',False):
                raise ValueError('Story lineage requires start_from_current_frame; no automatic opening generation')
            self.reset()
            if image and not Path(image).is_file(): raise FileNotFoundError('Input image not found')
            binding=self._bind_live_frame(image,initial=True)
            result=self.renderer.generate(image,['static'],'abandoned laboratory',self.state,self.folder/'initial')
            self._check_live_result(result,binding)
            self.frame=result.last_frame; self.video=result.video_path
            self.observation=self.observer.observe(self.frame)
            self.state=self.reconciler.reconcile(self.state,self.observation)
            return {'state':self.get_state(),'video':self.video,'frame':self.frame,'renderer':result.metadata}

    def get_state(self): return self.state.model_dump(mode='json')

    def declare_room_scene_cut(self,room,*,operation_id,expected_current_rgb_sha256,manifest_path=None,enabled=False):
        from .declared_room_cut import declare_room_scene_cut
        return declare_room_scene_cut(self,room,operation_id=operation_id,
            expected_current_rgb_sha256=expected_current_rgb_sha256,manifest_path=manifest_path,enabled=enabled)

    def start_from_current_frame(self,image):
        """Opt-in fresh world from declared current pixels, no initial video.

        Only the existing observer/reconciler run. No saved state, goal,
        inventory, task or historical session is imported with the image.
        """
        story_guard=getattr(self.renderer,'story_lineage_guard',False)
        if not (getattr(self.renderer,'live_current_frame',False) or story_guard):
            raise ValueError('Starting from current RGB requires live current-frame mode or story lineage opt-in')
        from PIL import Image
        from .live_continuation import rgb_sha256
        with self.lock:
            source=Path(image).resolve(strict=True)
            with Image.open(source) as original:
                if original.mode!='RGB' or original.size!=(832,480):
                    raise ValueError('Current input must already be exact832x480 RGB; no resizing or crop')
                current=original.copy()
            source_rgb=rgb_sha256(source)
            self.reset()
            folder=self.folder/'initial-input';folder.mkdir()
            owned=folder/'current.png';current.save(owned)
            if rgb_sha256(owned)!=source_rgb:
                raise ValueError('Current RGB copy differs from the declared input pixels')
            if story_guard:self.renderer.establish_story_input(self.session_id,owned,self.folder)
            else:self.renderer.establish_live_input(self.session_id,owned,self.folder)
            self.frame=str(owned);self.video=None
            self.initial_current_frame={
                'kind':'explicit_current_rgb_input_no_initial_generation',
                'source_image':str(source),'owned_current_frame':str(owned),
                'rgb_sha256':source_rgb,'session_id':self.session_id,
                'pixel_copy_verified':True,'video_generation_calls':0,
                'story_lineage_guard':story_guard,
                'new_video_generated':False,'historical_session_restored':False,
                'story_or_inventory_imported':False,'goal_condition_used':False,
                'visual_success_verified':False,'navigation_verified':False}
            self.observation=self.observer.observe(self.frame)
            self.state=self.reconciler.reconcile(self.state,self.observation)
            self.export_session()
            return {'state':self.get_state(),'video':None,'frame':self.frame,
                    'renderer':dict(self.initial_current_frame)}

    def recover_initial(self,session_id):
        if getattr(self.renderer,'story_lineage_guard',False):
            raise ValueError('Story lineage forbids historical or orphan initial recovery')
        import re
        if not re.fullmatch(r'[a-f0-9]{32}',session_id): raise ValueError('Invalid initial session')
        with self.lock:
            if self.history or self.task is not None or self.frame is not None:
                raise ValueError('Cannot replace an active world during initial recovery')
            folder=(self.root/session_id).resolve()
            if folder.parent!=self.root: raise ValueError('Initial session escapes root')
            result=self.renderer.recover_initial(folder)
            self.session_id=session_id; self.folder=folder
            self.frame=result.last_frame; self.video=result.video_path
            self.observation=self.observer.observe(self.frame)
            self.state=self.reconciler.reconcile(self.state,self.observation)
            self.initial_recovery=result.metadata
            self.export_session()
            return result.metadata
    def get_history(self): return list(self.history)

    def import_initial_archive(self,manifest_path):
        """Read-only pinned opening reuse; keep this new session and empty story."""
        if getattr(self.renderer,'live_current_frame',False):
            raise ValueError('Live interaction cannot import or substitute a historical opening')
        if getattr(self.renderer,'story_lineage_guard',False):
            raise ValueError('Story lineage cannot import or substitute a historical opening')
        from .initial_import import import_initial_archive
        with self.lock:
            return import_initial_archive(self,manifest_path)

    def start_desk_inspection(self, render, assess, capabilities=None):
        """Opt-in visual diagnostic, not search:desk or inventory mutation."""
        from .desk_inspection import DeskInspection
        active=self.task
        if active is not None: active.pause_requested.set()
        with self.lock:
            if self.desk_inspection is not None:
                raise ValueError('Keep the existing inspection evidence; use a new session for a new attempt')
            if self.task is not None:
                self.task.pause_requested.set(); self.task.status='paused'
            self.desk_inspection=DeskInspection(self.frame,self.folder/'desk-inspection',render,assess,capabilities)
            self.export_session()
            return self.desk_inspection.as_dict()

    def advance_desk_inspection(self, operation_id):
        with self.lock:
            if self.desk_inspection is None: raise ValueError('No active desk inspection')
            skill=self.desk_inspection
            current_matches=self.frame==skill.frame
            if not current_matches: skill.cancel_requested.set()
            row=skill.advance(operation_id)
            if current_matches and skill.video and not row.get('idempotent_replay'):
                self.frame,self.video=skill.frame,skill.video
                self.observation=Observation(source='unavailable',
                    evidence='Frame updated by desk inspection; use its recorded criteria, not stale scene observations. Full observation needs refresh.')
            self.export_session()
            return row

    def replace_goal(self,text):
        """Called at a step boundary; never undo already committed actions."""
        from .tasks import GoalTask,propose_goal
        if not isinstance(text,str) or not text.strip() or len(text)>2000:
            raise ValueError('请输入不超过 2000 字符的目标。')
        # Set before waiting for an in-flight render to release the lock. Its
        # boundary check must see the interruption before another step starts.
        active=self.task
        if active is not None: active.pause_requested.set()
        if self.desk_inspection is not None: self.desk_inspection.cancel_requested.set()
        with self.lock:
            if self.task is not None:
                self.task.pause_requested.set(); self.task.status='paused'
                self.task.message='收到新的目标，旧计划未执行部分已取消。'
                self.task_history.append(self.task.as_dict())
            from .goal_fallback import key_request,key_skill_proposal,key_route_skill_proposal
            task=None; failure=None
            try:
                proposal=propose_goal(self,text)
                if (key_request(text) and proposal.status=='ready' and
                    (set(proposal.success_checks)!={'key_owned'} or proposal.completion_mode!='rule_state')):
                    raise ValueError('当前明确取钥匙目标与模型声明的完成条件不符。')
                task=GoalTask(text,proposal); task.planning_source='language_model'; task.prepare(self)
                if task.status!='ready': failure=task.as_dict()
            except ValueError as exc:
                failure={'status':'invalid_model_plan','error':str(exc)}
            if failure:
                # Legacy skills omit approach-to-terminal/door prerequisites.
                # Never silently offer them under the new spatial contract.
                fallback=(key_skill_proposal(self,text) or key_route_skill_proposal(self,text)) if self.story_profile=='legacy' else None
                if fallback is None and getattr(self,'authored_route_fallback_enabled',False):
                    if self.story_profile=='spatial_v1':
                        from .spatial_skills import spatial_authored_route_fallback as authored_route_fallback
                    else:
                        from .route_showcase import authored_route_fallback
                    from .tasks import GoalProposal
                    route_fallback=authored_route_fallback(self,text,failure)
                    if route_fallback is not None:
                        fallback=GoalProposal.model_validate(route_fallback['proposal'])
                if fallback is not None:
                    candidate=GoalTask(text,fallback)
                    candidate.planning_source='authored_skill_fallback'
                    candidate.model_failure=failure
                    if candidate.status=='ready':
                        candidate.preflight(self)
                        candidate.message='模型计划未通过，当前为规则技能兜底建议（不是模型自主修复）；执行前仍逐步校验，画面另行审核。'
                    task=candidate
                elif task is None:
                    raise ValueError(failure['error'])
            self.task=task
            self.state=self.state.apply_delta({'player':{'goal':text}})
            self.export_session()
            return task

    def interact(self,player_text,task_plan=None,operation_id=None):
        if not isinstance(player_text,str) or not player_text.strip(): raise ValueError('请输入文字。')
        if len(player_text)>2000: raise ValueError('输入过长，请限制在 2000 字符内。')
        if self.desk_inspection is not None: self.desk_inspection.cancel_requested.set()
        if operation_id is not None and (not isinstance(operation_id,str) or not operation_id or len(operation_id)>200):
            raise ValueError('操作ID必须是1到200字符的字符串。')
        if (task_plan is None and operation_id not in self.completed_operations and
                self.task is not None and self.task.status in ('ready','running')):
            self.task.pause_requested.set()
        with self.lock:
            signature={'text':player_text,'plan':task_plan.model_dump(mode='json') if task_plan is not None else None}
            if operation_id in self.completed_operations:
                saved=self.completed_operations[operation_id]
                if saved['signature']!=signature: raise ValueError('同一操作ID不能用于不同动作。')
                return dict(saved['row'],idempotent_replay=True)
            began=time.monotonic(); previous=self.state.model_copy(deep=True)
            if task_plan is None:
                if self.task is not None and self.task.status in ('ready','running'):
                    self.task.status='paused'; self.task.message='你手动输入了新动作，旧任务已暂停，避免继续过期计划。'
                from .declared_room_cut import ending_observation_context
                self.agent.story_observation_context=None
                self.agent.story_observation_context=ending_observation_context(self)
                try:
                    plan=self.agent.plan(player_text,previous,self.observation,self.memory)
                finally:
                    self.agent.story_observation_context=None
            else:
                from .models import Plan
                plan=Plan.model_validate(task_plan.model_dump())
            proposal=self.rules.validate(previous,plan)
            from .declared_room_cut import ending_observation_proposal
            ending_proposal=ending_observation_proposal(self,plan)
            post_ending_observation=bool(ending_proposal is not None and ending_proposal.valid)
            if ending_proposal is not None: proposal=ending_proposal
            if self.pending_visual and not may_act_while_pending(plan):
                proposal=proposal.model_copy(update={'valid':False,'reason':'上一动作的视觉结果尚未解决；请检查、重试画面或人工确认后继续。',
                    'state_delta':{},'story_event':None,'npc_response':''})
            if plan.player_intent=='safe_fallback':
                proposal=proposal.model_copy(update={'valid':False,
                    'reason':'模型输出或服务失败；保留当前画面和剧情，未执行本次请求。',
                    'state_delta':{},'story_event':None,'npc_response':''})
            live_rejection=self._live_action_rejection(plan)
            if live_rejection is None:
                from .declared_room_cut import declared_room_action_rejection
                live_rejection=declared_room_action_rejection(self,plan)
            if live_rejection is not None:
                proposal=proposal.model_copy(update={'valid':False,'reason':live_rejection,
                    'state_delta':{},'story_event':None,'npc_response':''})
            candidate=previous
            if proposal.valid:
                candidate=previous.apply_delta(proposal.state_delta)
                try:
                    candidate=self.graph.transition(candidate,proposal.story_event)
                except ValueError as exc:
                    proposal=proposal.model_copy(update={'valid':False,'reason':str(exc),'state_delta':{},'story_event':None,'npc_response':''})
                    candidate=previous
            skills=self.selector.select(plan) if proposal.valid else ['static']
            post_ending_observation=post_ending_observation and proposal.valid
            row={'turn_id':len(self.history)+1,'player_text':player_text,'plan':plan.model_dump(mode='json'),
                'agent_kind':self.agent.kind,'valid':proposal.valid,'reason':proposal.reason,
                'model_error':getattr(self.agent,'last_error',None) if task_plan is None else None,
                'plan_source':'single_turn' if task_plan is None else 'reviewed_goal_plan',
                'planner_fallback':plan.player_intent=='safe_fallback',
                'npc_response':proposal.npc_response,'story_event':proposal.story_event,'state_delta':proposal.state_delta,
                'camera_skills':[getattr(s,'value',s) for s in skills],'renderer_status':'not_requested',
                'fallback_suggestion':proposal.fallback_suggestion,'visual_goal_verified':False}
            if post_ending_observation:
                row['post_ending_observation']=True
                row['post_ending_state_effects_allowed']=False
            row.update(operation_id=operation_id or uuid.uuid4().hex,rule_committed=False,
                previous_state=previous.model_dump(mode='json'),previous_frame=self.frame,
                failure_type='unsupported' if live_rejection else
                    'visual_pending' if self.pending_visual and not may_act_while_pending(plan) else None)
            if proposal.valid:
                render_snapshot=self.renderer.snapshot() if hasattr(self.renderer,'snapshot') else None
                try:
                    from .dialogue_rendering import retained_dialogue_result
                    retained=retained_dialogue_result(plan,self.frame,self.video,
                        enabled=getattr(self,'retain_dialogue_frame',False),
                        allow_image_only=getattr(self.renderer,'story_lineage_guard',False))
                    render_args=(self.frame,skills,'abandoned laboratory',candidate,self.folder/f'turn-{row["turn_id"]:04d}')
                    if retained is not None:
                        if getattr(self.renderer,'story_lineage_guard',False):
                            from .story_lineage import verify_retained_dialogue
                            binding=self._bind_live_frame(self.frame)
                            verify_retained_dialogue(self.renderer,binding,retained,self.video)
                        result=retained
                    else:
                        binding=self._bind_live_frame(self.frame)
                        if hasattr(self.renderer,'generate_rule_plan'):
                            result=self.renderer.generate_rule_plan(*render_args,plan,proposal)
                        elif hasattr(self.renderer,'generate_plan'):
                            result=self.renderer.generate_plan(*render_args,plan)
                        else:
                            result=self.renderer.generate(*render_args)
                        self._check_live_result(result,binding)
                    observation=self.observation if retained is not None else self.observer.observe(result.last_frame,self.frame)
                    # This separate story-only observation may update perceived
                    # pixels, never terminal/location/inventory/environment state.
                    if not post_ending_observation:
                        candidate=self.reconciler.reconcile(candidate,observation)
                    if retained is not None:
                        row['visual_assessment']={'status':'not_visually_testable',
                            'source':'dialogue_presentation_policy','confidence':0,
                            'evidence':'Only authored dialogue/rule text changed; no new video, voice or lip-sync requested.',
                            'ground_truth':False,'authoritative_for_story':False}
                        row['camera_skills']=[]
                    elif hasattr(self.observer,'assess_video'):
                        row['visual_assessment']=self.observer.assess_video(plan,self.frame,result.last_frame,
                            result.metadata.get('raw_video_path',result.video_path))
                    elif hasattr(self.observer,'assess_goal'):
                        row['visual_assessment']=self.observer.assess_goal(plan,self.frame,result.last_frame)
                    row['visual_assessment']=assess_prebuilt(result.metadata,row.get('visual_assessment',{}))
                    self.frame=result.last_frame; self.video=result.video_path; self.observation=observation
                    self.state=candidate
                    row.update(renderer_status='dialogue_only' if retained is not None else 'completed',
                        renderer=result.metadata,rule_committed=True,
                        new_video_generated=retained is None and not result.metadata.get('pre_generated',False))
                    if retained is None:
                        self.render_contexts[row['operation_id']]={'frame':row['previous_frame'],'plan':plan,
                            'state':candidate.model_copy(deep=True),'skills':skills,'retries':0,
                            'renderer_snapshot':render_snapshot,'rule_result':proposal.model_copy(deep=True)}
                except Exception as exc:
                    if render_snapshot is not None:
                        self.renderer.restore(render_snapshot)
                    # All-or-nothing logical commit; failed rendering cannot advance inventory/story.
                    row.update(valid=False,reason='生成失败，未提交剧情变化：'+str(exc),
                        renderer_status='failed',state_delta={},story_event=None,npc_response='')
            row.update(latency_seconds=round(time.monotonic()-began,3),video=self.video,frame=self.frame,
                       state=self.get_state(),observation=self.observation.model_dump())
            row['failure_type']=row['failure_type'] or visual_problem(row,getattr(self.renderer,'kind','unknown'),self.visual_review_policy)
            if row['rule_committed'] and row['failure_type'] in ('visual_mismatch','visual_uncertain','manual_review_required'):
                pending={'operation_id':row['operation_id'],'turn_id':row['turn_id'],
                    'failure_type':row['failure_type'],'assessment':row.get('visual_assessment',{})}
                if self.visual_review_policy=='demo_advisory':
                    self.visual_warnings.append(dict(pending,source='demo_advisory',
                        visual_success_verified=False,blocks_story=False))
                    row['visual_advisory_only']=True
                elif self.pending_visual: self.pending_visual_queue.append(pending)
                else: self.pending_visual=pending
            self.memory.update(row,self.state)
            row['memory']=self.memory.retrieve(); self.history.append(row)
            with (self.folder/'turns.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            (self.folder/'state.json').write_text(self.state.serialize(),encoding='utf-8')
            # Network replay of a failed request must not silently run the GPU
            # again. An intentional retry gets a new operation ID.
            if operation_id:
                self.completed_operations[operation_id]={'signature':signature,'row':row}
            return row

    def unresolved_visuals(self):
        return ([self.pending_visual] if self.pending_visual else [])+list(self.pending_visual_queue)

    def _remove_current_visual(self):
        self.pending_visual=self.pending_visual_queue.pop(0) if self.pending_visual_queue else None

    def resolve_visual(self,note,decision='accept'):
        """Explicit human approval; never a planner tool or automatic VLM upgrade."""
        with self.lock:
            if not self.pending_visual: raise ValueError('没有待确认的视觉结果。')
            if not isinstance(note,str) or not note.strip() or len(note)>500:
                raise ValueError('请填写你检查画面后的说明（最多500字符）。')
            if decision not in ('accept','reject','uncertain'):
                raise ValueError('人工决定必须是 accept、reject 或 uncertain。')
            review=dict(self.pending_visual,source='human_override' if decision=='accept' else 'human_review',decision=decision,
                        note=note.strip(),visual_ground_truth=False,clears_pending=decision=='accept')
            self.visual_reviews.append(review)
            if decision=='accept': self._remove_current_visual()
            else: self.pending_visual.update(human_decision=decision,human_note=note.strip())
            self.export_session()
            return review

    def retry_visual(self):
        """Regenerate from the pre-action frame once, without applying rule effects again."""
        with self.lock:
            if getattr(self.renderer,'live_current_frame',False):
                raise ValueError('当前交互不得回到旧输入帧重拍；请从当前画面明确提出下一动作。')
            if getattr(self.renderer,'story_lineage_guard',False):
                raise ValueError('故事当前帧模式不回到旧输入帧重拍；请从当前画面提出下一动作。')
            if not self.pending_visual: raise ValueError('没有待修复的视觉动作。')
            pending=dict(self.pending_visual); context=self.render_contexts[pending['operation_id']]
            if context['retries']>=1: raise ValueError('本动作已用完一次视觉重试，请人工检查或修改目标。')
            # Do not overwrite a later camera move or a newly changed user state.
            if self.history[-1]['operation_id']!=pending['operation_id']:
                raise ValueError('已有后续操作，不能用旧帧覆盖当前画面。请人工检查。')
            context['retries']+=1
            started=time.monotonic()
            review=dict(pending,source='visual_retry',rule_effect_reapplied=False,status='failed')
            snapshot=self.renderer.snapshot() if hasattr(self.renderer,'snapshot') else None
            try:
                if context['renderer_snapshot'] is not None:
                    self.renderer.restore(context['renderer_snapshot'])
                args=(context['frame'],context['skills'],'abandoned laboratory',context['state'],
                      self.folder/f'visual-retry-{pending["turn_id"]:04d}')
                if hasattr(self.renderer,'generate_rule_plan') and 'rule_result' in context:
                    result=self.renderer.generate_rule_plan(*args,context['plan'],context['rule_result'])
                else:
                    result=(self.renderer.generate_plan(*args,context['plan']) if hasattr(self.renderer,'generate_plan')
                            else self.renderer.generate(*args))
                observation=self.observer.observe(result.last_frame,context['frame'])
                if hasattr(self.observer,'assess_video'):
                    assessment=self.observer.assess_video(context['plan'],context['frame'],result.last_frame,
                        result.metadata.get('raw_video_path',result.video_path))
                elif hasattr(self.observer,'assess_goal'):
                    assessment=self.observer.assess_goal(context['plan'],context['frame'],result.last_frame)
                else: assessment={'status':'uncertain','source':'unavailable'}
                assessment=assess_prebuilt(result.metadata,assessment)
                self.frame=result.last_frame; self.video=result.video_path; self.observation=observation
                self.state=self.reconciler.reconcile(self.state,observation)
                review.update(status='completed',video=self.video,frame=self.frame,renderer=result.metadata,
                              assessment=assessment)
                problem=visual_problem({'valid':True,'plan':context['plan'].model_dump(mode='json'),
                    'visual_assessment':assessment},getattr(self.renderer,'kind','unknown'),self.visual_review_policy)
                if problem is None and assessment.get('status')=='supported': self._remove_current_visual()
                else: self.pending_visual.update(assessment=assessment,failure_type=problem or 'visual_uncertain')
            except Exception as exc:
                if snapshot is not None: self.renderer.restore(snapshot)
                review['error']=type(exc).__name__+': '+str(exc)[:300]
            review['latency_seconds']=round(time.monotonic()-started,3)
            self.visual_reviews.append(review); self.export_session()
            return review

    def select_output(self,version_id):
        """Choose a delivery artifact, never rewind the simulation or reapply effects."""
        with self.lock:
            if not isinstance(version_id,str): raise ValueError('无效的视频版本。')
            kind,sep,number=version_id.partition(':')
            if not sep or not number.isdigit(): raise ValueError('无效的视频版本。')
            if kind=='turn':
                row=next((r for r in self.history if r['turn_id']==int(number)),None)
                eligible=bool(row and row.get('rule_committed') and row.get('renderer_status')=='completed')
            elif kind=='retry':
                index=int(number)
                row=self.visual_reviews[index] if index<len(self.visual_reviews) else None
                eligible=bool(row and row.get('source')=='visual_retry' and row.get('status')=='completed')
            else: row=None; eligible=False
            if not eligible or not row.get('video') or not Path(row['video']).is_file():
                raise ValueError('该版本没有可交付的视频文件。')
            self.selected_output={'version_id':version_id,'video':row['video'],'frame':row.get('frame'),
                'turn_id':row.get('turn_id'),'operation_id':row.get('operation_id'),
                'visual_ground_truth':False,'world_state_rolled_back':False}
            self.export_session()
            return dict(self.selected_output)

    def export_session(self):
        """Export this session only; no model weights or shared sessions."""
        with self.lock:
            payload={'session_id':self.session_id,'agent_kind':self.agent.kind,
                'goal_planner_model':getattr(getattr(self.agent,'goal_provider',None),'model',None),
                'goal_planner_provenance':getattr(self,'goal_planner_provenance',None),
                'story_profile':self.story_profile,
                'renderer_kind':getattr(self.renderer,'kind','unknown'),
                'live_current_frame':getattr(self.renderer,'live_current_frame',False),
                'story_lineage_guard':getattr(self.renderer,'story_lineage_guard',False),
                'state':self.get_state(),'memory':self.memory.retrieve(),
                'observation':self.observation.model_dump(mode='json'),
                'current_frame':self.frame,'current_video':self.video,
                'turns':self.history,'task':self.task.as_dict() if self.task else None,
                'pending_visual':self.pending_visual,'pending_visual_queue':self.pending_visual_queue,
                'visual_review_policy':self.visual_review_policy,'selected_output':self.selected_output,
                'visual_reviews':self.visual_reviews,'visual_warnings':self.visual_warnings,
                'visual_success_verified':False,'task_history':self.task_history,
                'goal_planner_calls':self.goal_planner_calls,
                'initial_recovery':self.initial_recovery,
                'initial_current_frame':self.initial_current_frame,
                'session_resume':self.session_resume,
                'declared_scene_cuts':self.declared_scene_cuts,
                'dialogue_render_policy':'retain_current_view_no_new_video' if getattr(self,'retain_dialogue_frame',False) else 'original_render_each_turn',
                'desk_inspection':self.desk_inspection.as_dict() if self.desk_inspection else None,
                'visual_judgments_are_ground_truth':False}
            path=self.folder/'session-export.json'
            path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
            return str(path)
