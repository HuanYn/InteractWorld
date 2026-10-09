"""Bounded goal planning and rule-validated execution, not LongCoT or RL."""
import json
import re
import threading
import uuid
from copy import deepcopy
from typing import Literal
from pydantic import Field,model_validator
from .models import StrictModel,Plan
from .tools import TOOL_CALLS
from .feedback import effect_already_done, may_act_while_pending
from .skills import skill_context

Check=Literal['key_owned','door_open','password_known','terminal_unlocked','public_logs',
    'at_core','at_storage','met_anna','met_leo','ending_anna','ending_key','ending_terminal']
TOOL_IDS={':'.join(part for part in call if part):call for call in TOOL_CALLS}
ToolId=Literal[tuple(TOOL_IDS)]


class PlanPreconditionError(ValueError):
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


class GoalCoverageError(ValueError):
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


def plan_signature(steps):
    return [[s.plan.semantic_action,s.plan.target,s.plan.dialogue_action] for s in steps]


class ToolGoalStep(StrictModel):
    instruction:str=Field(min_length=1,max_length=300)
    tool_id:ToolId


class ToolGoalDraft(StrictModel):
    status:Literal['ready','needs_clarification']
    clarification:str=Field(default='',max_length=400)
    steps:list[ToolGoalStep]=Field(default_factory=list,max_length=8)
    success_checks:list[Check]=Field(default_factory=list,max_length=6)
    completion_mode:Literal['rule_state','action_execution']='rule_state'

    def compile(self):
        steps=[]
        for step in self.steps:
            action,target,dialogue=TOOL_IDS[step.tool_id]
            steps.append(GoalStep(instruction=step.instruction,plan=Plan(player_intent=step.instruction,
                semantic_action=action,target=target,dialogue_action=dialogue)))
        return GoalProposal(status=self.status,clarification=self.clarification,steps=steps,
            success_checks=self.success_checks,completion_mode=self.completion_mode)


class GoalStep(StrictModel):
    instruction: str=Field(min_length=1,max_length=300)
    plan: Plan


class GoalProposal(StrictModel):
    status: Literal['ready','needs_clarification']
    clarification: str=Field(default='',max_length=400)
    steps: list[GoalStep]=Field(default_factory=list,max_length=8)
    success_checks: list[Check]=Field(default_factory=list,max_length=6)
    completion_mode: Literal['rule_state','action_execution']='rule_state'

    @model_validator(mode='after')
    def meaningful(self):
        if self.status=='ready' and (not self.steps or (self.completion_mode=='rule_state' and not self.success_checks)):
            raise ValueError('Ready plans need steps and explicit rule success checks')
        if self.status=='needs_clarification' and (not self.clarification or self.steps):
            raise ValueError('Clarification must explain the limit without executable steps')
        if self.completion_mode=='action_execution' and (self.success_checks or any(
                step.plan.semantic_action not in ('observe','wait','inspect') or
                step.plan.target not in ('left','right','room','desk','door_a') for step in self.steps)):
            raise ValueError('action_execution is only for supported non-mutating observation tools')
        return self


def check_goal(state,checks):
    flags=state.story.flags
    values={'key_owned':'red_key' in state.inventory.items,
        'door_open':state.objects['door_a'].state=='open','password_known':bool(flags.get('password_known')),
        'terminal_unlocked':state.objects['terminal'].state=='unlocked','public_logs':bool(flags.get('log_retrieved')),
        'at_core':state.player.location=='core_area','at_storage':state.player.location=='storage_room',
        'met_anna':bool(flags.get('met_anna')),'met_leo':bool(flags.get('met_leo'))}
    for ending in ('ending_anna','ending_key','ending_terminal'):
        values[ending]=state.story.terminal and state.story.node==ending
    return bool(checks) and all(values.get(c,False) for c in checks)


def goal_messages(engine,text,feedback=None):
    if getattr(engine,'story_profile','legacy')=='spatial_v1':
        from .spatial_skills import spatial_goal_messages
        return spatial_goal_messages(engine,text,feedback)
    if not text.strip() or len(text)>2000: raise ValueError('请输入不超过 2000 字符的目标。')
    prompt=('You are the multi-step goal planner for an authored laboratory world. '
        'Return ONLY one JSON ToolGoalDraft. Select tool_id values from the schema; the engine compiles them. '
        'Return the COMPLETE remaining sequence toward the final user goal, not just one next action. '
        'The user request does not itself change state. No long reasoning text. '
        'Seeing Anna in the image or visible_characters does NOT mean met_anna=true. '
        'Only state.story.flags.met_anna establishes prior contact. If false, dialogue with distrust/help/password '
        'needs approach:anna first, even when Anna is clearly visible. '
        'To choose route B, use approach:anna if not met, then talk:anna:distrust; search:desk reveals the key, '
        'pick_up:red_key puts it in inventory, open:door_a requires owning the key, move_to:storage_room requires an open door. '
        'avoid:anna chooses route C, NOT independent key investigation. Route A: approach:anna, '
        'interact:anna:help, ask:anna:password, use:terminal, move_to:core_area. Route C: '
        'inspect:terminal twice, approach:leo, ask:leo:logs. Never switch an established route. '
        'Only state.story.route establishes a selected route. null means NO route is selected; '
        'meeting Anna, camera observations and previous player goals do not select a route. '
        'Do not reject a goal as an incompatible selected route when that field is null. '
        'Plan at most 8 steps from the supplied CURRENT state. Include needed prerequisites only when '
        'the user authorizes achieving a broader goal. Preserve explicit route preference. If the user '
        'forbids prerequisites or asks for an unavailable bridge, mountain, object, arbitrary world or '
        'unverifiable visual goal, return needs_clarification. Never switch an established route. '
        'Do not introduce another branch to satisfy an incompatible goal. success_checks refer ONLY '
        'to rule state, not actual pixels. No long chain of thought. Every step is one supported tool call. '
        'Pure supported look/inspect/wait requests may use completion_mode=action_execution with no '
        'success_checks; this only means tool execution, not a verified viewing angle. '
        'When feedback is present, repair ONLY remaining steps toward the ORIGINAL goal, preserve '
        'completion_mode and success_checks exactly, do not replay already committed effects, '
        'and honor new user constraints. Skills are authored suggestions, not execution evidence. '
        'Use ONLY the enumerated success_checks; do NOT invent at_laboratory, door_closed or '
        'key_found. A goal to obtain a key without opening the door needs ["key_owned"] only; '
        'negative constraints restrict steps, not new checks. '
        'For replanning, copy required_success_checks and required_completion_mode exactly; '
        'a prerequisite such as meeting Anna is not a replacement for obtaining the key. '
        'Feedback.failed_precondition identifies the failing step and verified one-step repair suggestions. '
        'Suggestions are not authorization: honor user prohibitions and validate the COMPLETE remaining plan. '
        'Never repeat feedback.rejected_plan or previous_rejected_plans unchanged. '
        'Schema: '+json.dumps(ToolGoalDraft.model_json_schema()))
    planning_state=engine.get_state()
    previous_goal=planning_state['player']['goal']
    # A replacement has not committed yet: expose the current request to the
    # planner without mutating the engine, and label the old goal as history.
    planning_state['player']['goal']=text
    context={'previous_goal_history_only':previous_goal,
        'state':planning_state,'observation':engine.observation.model_dump(),
        'memory':engine.memory.retrieve(),'supported_calls':TOOL_CALLS,
        'skills':skill_context(engine.state),'feedback':feedback,
        'pending_visual':getattr(engine,'pending_visual',None)}
    context['current_tool_preconditions']=[]
    for action,target,dialogue in TOOL_CALLS:
        validity=engine.rules.validate(engine.state,Plan(player_intent='precondition_query',semantic_action=action,
          target=target,dialogue_action=dialogue))
        context['current_tool_preconditions'].append({'call':[action,target,dialogue],
            'allowed_now':validity.valid,'rule_reason':validity.reason})
    context['request_priority']='Only user_goal below is the current request. Previous goals, memory and observations do not authorize other actions.'
    context['user_goal']=text
    return [{'role':'system','content':prompt},{'role':'user','content':json.dumps(context,ensure_ascii=False)}]


def unsupported_route_refusal(engine,proposal):
    """Detect an explicit false factual premise, not all clarification requests."""
    return (engine.state.story.route is None and proposal.status=='needs_clarification'
        and bool(re.search(r'(?:已选|已经选择|当前|现有).{0,12}(?:路线|分支).{0,12}(?:不兼容|冲突)|'
                           r'(?:already.selected|current|existing).{0,20}(?:route|branch).{0,20}(?:incompatible|conflict)',
                           proposal.clarification,re.I)))


def propose_goal(engine,text,feedback=None):
    provider=getattr(engine.agent,'goal_provider',None) or getattr(engine.agent,'provider',None)
    if provider is None: raise ValueError('目标拆解需要真实语言模型；不会用预写路线冒充模型规划。')
    messages=goal_messages(engine,text,feedback)
    if not hasattr(engine,'goal_planner_calls'): engine.goal_planner_calls=[]
    schema_repairs=0; state_repairs=0
    # At most one JSON repair and one verified-state correction, three calls total.
    for attempt in range(3):
        raw=provider.complete(messages,max_tokens=1800).strip()
        record={'goal':text,'feedback':feedback,'schema_repair_attempt':schema_repairs,
                'state_consistency_attempt':state_repairs,'raw_response':raw,
                'messages':list(messages)}
        engine.goal_planner_calls.append(record)
        if raw.startswith('```') and raw.endswith('```'): raw='\n'.join(raw.splitlines()[1:-1])
        try:
            parsed=json.loads(raw)
            if not isinstance(parsed,dict): raise ValueError('Goal proposal must be one JSON object')
            # Retain the full-plan contract for existing clients, but ask the model
            # for compact typed tool IDs to eliminate invalid action/target triples.
            steps=parsed.get('steps',[])
            if not isinstance(steps,list): raise ValueError('steps must be a JSON array')
            compact=any(isinstance(step,dict) and 'tool_id' in step for step in steps) or not steps
            if compact:
                draft_class=ToolGoalDraft
                if getattr(engine,'story_profile','legacy')=='spatial_v1':
                    from .spatial_skills import SpatialToolGoalDraft
                    draft_class=SpatialToolGoalDraft
                    if getattr(engine,'direction_completion_compiler',False) is True:
                        from .direction_completion import compile_direction_completion
                        parsed,compilation=compile_direction_completion(text,parsed)
                        if compilation is not None:
                            record['direction_completion_compilation']=compilation
                draft=draft_class.model_validate(parsed)
                # Disabled by default, including every existing UI. An explicitly
                # opted-in engine may compile only this narrow typed-tool contract;
                # raw_response above remains the untouched model output.
                if getattr(engine,'goal_contract_normalization',False) is True and getattr(engine,'story_profile','legacy')=='legacy':
                    from .goal_contract_normalization import normalize_npc_observation_contract
                    draft,normalization=normalize_npc_observation_contract(draft)
                    if normalization is not None:
                        record['goal_contract_normalization']=normalization
                result=draft.compile()
            else:
                result=GoalProposal.model_validate(parsed)
            record['parsed']=True
            record['contract']=('spatial_tool_ids_v1' if getattr(engine,'story_profile','legacy')=='spatial_v1' else 'tool_ids_v1') if compact else 'legacy_full_plan'
        except ValueError as exc:
            record.update(parsed=False,error=str(exc)[:1000])
            if schema_repairs or attempt==2: raise
            schema_repairs+=1
            messages.extend([{'role':'assistant','content':raw},{'role':'user','content':
                'Repair this JSON once. Keep the user goal and explicit prohibitions. Do not invent schema fields or success checks. Validation error: '+str(exc)[:1000]}])
            continue
        if unsupported_route_refusal(engine,result):
            record['state_consistency_error']='claimed_selected_route_but_actual_route_null'
            if state_repairs or attempt==2:
                # Remain blocked; never turn a model refusal into an executable rule plan.
                return result.model_copy(update={'clarification':
                    '规划器重复误判路线：实际尚未选择路线。自动纠正已达上限，未执行动作；请修改目标或检查模型。'})
            state_repairs+=1
            messages.extend([{'role':'assistant','content':raw},{'role':'user','content':json.dumps({
                'verified_state_correction':{'field':'state.story.route','actual_value':None,
                    'source':'current_rule_state','error':'No route has been selected. Your selected-route conflict premise is false.'},
                'instruction':'Reconsider the ORIGINAL user_goal against the supplied current state. Preserve all user constraints and other capability limits. Return a complete supported plan or a different, factually supported clarification. This feedback does not authorize execution or change state.'},ensure_ascii=False)}])
            continue
        return result


class GoalTask:
    def __init__(self,goal,proposal):
        self.goal=goal; self.proposal=proposal; self.index=0; self.attempts=0
        self.status='ready' if proposal.status=='ready' else 'needs_clarification'
        self.message=proposal.clarification; self.trace=[]
        self.pause_requested=threading.Event()
        self.task_id=uuid.uuid4().hex; self.replans=0; self.revision=0
        self.replan_history=[]; self.failure_type=None
        self.precondition_failure=None
        self.rejected_plans=[]
        self.planning_source='provided_plan'
        self.model_failure=None
        from .direction_completion import direction_observation_target
        self.required_observation_target=direction_observation_target(goal)
        from .goal_coverage import explicit_ordered_calls
        self.required_ordered_calls=explicit_ordered_calls(goal)
        self.goal_coverage_failure=None
        self.goal_coverage_enabled=False
        # Conservative guards for explicit prohibitions in the supported lab UI.
        # This is not a general natural-language constraint parser.
        self.forbidden_actions=set()
        if re.search(r'(不要|别)(打开|开)(储藏室)?门',goal): self.forbidden_actions.add('open')
        if re.search(r'(不要|别)(进入|进去)',goal): self.forbidden_actions.add('move_to')
        self.forbid_anna_approach=bool(re.search(r'(不要|别|不允许)(接近|靠近|走近)\s*anna',goal,re.I))

    def _forbidden(self,plan,state=None):
        # Alias tools may have the same effect; wording restrictions cannot be
        # bypassed by changing open:door_a into use:door_a.
        if self.required_observation_target and (plan.semantic_action!='inspect' or plan.target!=self.required_observation_target):
            return True
        if plan.semantic_action in self.forbidden_actions: return True
        if self.forbid_anna_approach and plan.target=='anna':
            if plan.semantic_action in ('approach','follow'): return True
            if state is not None and not state.story.flags.get('met_anna') and plan.semantic_action in ('talk','ask','interact'):
                return True  # greeting aliases may otherwise establish the forbidden contact
        if 'open' in self.forbidden_actions and plan.target=='door_a' and plan.semantic_action=='use':
            return True
        return ('move_to' in self.forbidden_actions and plan.target in ('storage_room','core_area')
                and plan.semantic_action in ('move_to','approach','follow','open'))

    def _completed_goal_calls(self,engine):
        committed={row.get('operation_id') for row in self.trace if row.get('committed')}
        return [[row['plan'].get('semantic_action'),row['plan'].get('target'),row['plan'].get('dialogue_action')]
                for row in engine.history if row.get('rule_committed') and row.get('operation_id') in committed]

    def preflight(self,engine,*,completed_goal_calls=None):
        # Simulate only for executability. No renderer, memory mutation or success claim.
        self.goal_coverage_failure=None
        self.goal_coverage_enabled=bool(getattr(getattr(engine,'renderer',None),'story_lineage_guard',False))
        if self.goal_coverage_enabled and self.required_ordered_calls:
            from .goal_coverage import ordered_coverage
            completed=(self._completed_goal_calls(engine) if completed_goal_calls is None else completed_goal_calls)
            observed=completed+plan_signature(self.proposal.steps[self.index:])
            missing=ordered_coverage(self.required_ordered_calls,observed)
            if missing:
                self.goal_coverage_failure={
                    'source':'explicit_user_goal_narrow_sequence_not_a_generated_plan',
                    'original_goal':self.goal,'required_ordered_calls':deepcopy(self.required_ordered_calls),
                    'committed_goal_calls':completed,'observed_calls':observed,'missing_ordered_suffix':missing,
                    'world_state_changed':False,'visual_success_verified':False}
                raise GoalCoverageError('计划遗漏或颠倒了用户明确要求的后续动作；请保留原目标和完成条件，'
                    '按顺序覆盖：'+json.dumps(self.required_ordered_calls,ensure_ascii=False),self.goal_coverage_failure)
        if self.required_observation_target:
            if len(self.proposal.steps)!=1 or self._forbidden(self.proposal.steps[0].plan):
                raise ValueError('本次明确要求只观察 '+self.required_observation_target+'，必须且只能用 inspect:'+self.required_observation_target+'；不能添加动作或改变目标。')
            if self.proposal.completion_mode!='action_execution' or self.proposal.success_checks:
                raise ValueError('纯方向观察的工具步骤正确，但完成条件错误：必须是 completion_mode=action_execution、success_checks=[]；不能用规则事件充当观察完成或视觉成功。')
        candidate=engine.state.model_copy(deep=True)
        for number,step in enumerate(self.proposal.steps[self.index:],self.index+1):
            if self._forbidden(step.plan,candidate):
                raise ValueError(f'计划第 {number} 步违反用户明确禁止的动作：{step.plan.semantic_action}')
            if effect_already_done(candidate,step.plan): continue
            result=engine.rules.validate(candidate,step.plan)
            if not result.valid:
                suggestions=[]
                tool_ids=TOOL_IDS
                if getattr(engine,'story_profile','legacy')=='spatial_v1':
                    from .spatial_skills import SPATIAL_TOOL_IDS
                    tool_ids=SPATIAL_TOOL_IDS
                for tool_id,(action,target,dialogue) in tool_ids.items():
                    repair=Plan(player_intent='rule_repair_suggestion',semantic_action=action,target=target,dialogue_action=dialogue)
                    if self._forbidden(repair,candidate) or effect_already_done(candidate,repair): continue
                    valid=engine.rules.validate(candidate,repair)
                    if not valid.valid: continue
                    try: after=engine.graph.transition(candidate.apply_delta(valid.state_delta),valid.story_event)
                    except ValueError: continue
                    if after==candidate or (candidate.story.route and after.story.route!=candidate.story.route): continue
                    if engine.rules.validate(after,step.plan).valid: suggestions.append(tool_id)
                self.precondition_failure={'step_number':number,
                    'failed_call':plan_signature([step])[0],'reason':result.reason,
                    'fallback_suggestion':result.fallback_suggestion,
                    'state_before_failed_step':candidate.model_dump(mode='json'),
                    'state_source':'dry_run_simulated_not_committed',
                    'one_step_repair_suggestions':suggestions[:8],
                    'suggestions_source':'rule_simulation_not_model','not_execution_authority':True}
                raise PlanPreconditionError(f'计划第 {number} 步不可执行：{result.reason}',self.precondition_failure)
            candidate=engine.graph.transition(candidate.apply_delta(result.state_delta),result.story_event)
        if self.proposal.completion_mode=='rule_state' and self.status=='ready' and not check_goal(candidate,self.proposal.success_checks):
            raise ValueError('计划无法满足其声明的规则目标，请修改目标后重新拆解。')

    def prepare(self,engine):
        """Feed a rejected dry plan back to the model; share the same two-replan cap."""
        if self.status!='ready': return
        try:
            self.preflight(engine)
            return
        except ValueError as exc:
            reason=str(exc)
        while self.replans<2 and not self.pause_requested.is_set():
            self._recover(engine,{'failure_type':'decision_failure','reason':reason,'turn_id':None})
            if self.status in ('ready','needs_clarification'): return
            reason=self.message
        self.status='paused'
        self.message='规划未通过前置条件校验，未执行任何本任务动作：'+reason

    def _recover(self,engine,row):
        self.failure_type=row.get('failure_type') or 'decision_failure'
        if self.failure_type!='decision_failure' or self.pause_requested.is_set():
            self.status='paused'; self.message=row['reason']; return
        if self.replans>=2:
            self.status='paused'; self.message='已达到两次自动重规划上限：'+row['reason']; return
        if row.get('turn_id') is not None:
            # A real failed attempt may follow successful commits: do not carry
            # a rejected-plan fingerprint across different world states.
            self.rejected_plans=[]; self.precondition_failure=None
            try: self.preflight(engine)
            except ValueError: pass
        self.replans+=1
        rejected=plan_signature(self.proposal.steps[self.index:])
        if not self.rejected_plans: self.rejected_plans.append(rejected)
        feedback={'failure':row['reason'],'latest_turn':row['turn_id'],
                  'committed_steps':[t for t in self.trace if t.get('committed')],
                  'required_success_checks':self.proposal.success_checks,
                  'required_completion_mode':self.proposal.completion_mode,'remaining_only':True,
                  'failed_precondition':self.precondition_failure,
                  'rejected_plan':self.rejected_plans[-1],
                  'previous_rejected_plans':list(self.rejected_plans),
                  'explicit_forbidden_actions':sorted(self.forbidden_actions),
                  'forbid_anna_approach':self.forbid_anna_approach}
        feedback['required_observation_target']=self.required_observation_target
        feedback['goal_coverage_failure']=deepcopy(self.goal_coverage_failure)
        record={'attempt':self.replans,'feedback':feedback,'accepted':False}
        try:
            proposal=propose_goal(engine,self.goal,feedback=feedback)
            record['proposal']=proposal.model_dump(mode='json')
            if proposal.status=='needs_clarification':
                self.status='needs_clarification'; self.message=proposal.clarification; return
            if (set(proposal.success_checks)!=set(self.proposal.success_checks) or
                    proposal.completion_mode!=self.proposal.completion_mode):
                raise ValueError('修订不得悄悄改变原目标完成条件。')
            signature=plan_signature(proposal.steps)
            repeated=signature in self.rejected_plans
            if repeated:
                record['repeated_failed_plan']=True
            else:
                self.rejected_plans.append(signature)
            candidate=GoalTask(self.goal,proposal)
            # Recheck repeated proposals without execution so the next feedback
            # retains their CURRENT coverage/precondition error, not stale details
            # or the empty detail from a generic duplicate-signature ValueError.
            candidate.preflight(engine,completed_goal_calls=self._completed_goal_calls(engine))
            if repeated:
                raise ValueError('同一状态下重复了已拒绝计划，缺失前置步骤仍未修复；需要改变步骤或说明无法执行。')
            self.proposal=proposal; self.index=0; self.revision+=1
            self.status='ready'; self.message='已根据最新状态修订剩余步骤；下一步仍会校验。'
            record['accepted']=True; self.failure_type=None; self.precondition_failure=None
            self.goal_coverage_failure=None
        except Exception as exc:
            # Feedback must describe the latest rejected proposal, not a failed
            # step from an earlier plan that the model may already have removed.
            self.precondition_failure=exc.details if isinstance(exc,PlanPreconditionError) else None
            self.goal_coverage_failure=exc.details if isinstance(exc,GoalCoverageError) else None
            self.status='paused'; self.message='重规划未完成，世界状态保留：'+str(exc)[:250]
            record['error']=type(exc).__name__+': '+str(exc)[:250]
        finally:
            self.replan_history.append(record)
            if self.pause_requested.is_set():
                self.status='paused'; self.message='已按用户要求暂停，修订计划尚未执行。'

    def advance(self,engine):
        with engine.lock:
            return self._advance(engine)

    def _finish(self,engine):
        pending=engine.unresolved_visuals()
        own_turns={row.get('turn_id') for row in self.trace}
        blocking=next((p for p in pending if self.proposal.completion_mode!='action_execution'
                       or p['turn_id'] in own_turns),None)
        if blocking:
            self.status='paused'; self.failure_type=blocking['failure_type']
            self.message='最后一步视觉结果仍待处理，不能直接结束任务。'
            return
        self.failure_type=None
        if self.proposal.completion_mode=='action_execution':
            self.status='completed_action_goal'; self.message='观察工具已执行；不等于视角变化已视觉验证。'
        else:
            self.status='completed_rule_goal' if check_goal(engine.state,self.proposal.success_checks) else 'paused'
            self.message='规则目标已完成；不代表全部画面动作已验证。' if self.status=='completed_rule_goal' else '步骤已用尽，但规则目标未满足。'

    def _advance(self,engine):
        if self.pause_requested.is_set():
            self.status='paused'; self.message='已在步骤边界暂停；不撤销已完成动作。'; return None
        if self.status not in ('ready','running'): raise ValueError('任务未就绪或已暂停；请先检查原因再恢复。')
        if self.index>=len(self.proposal.steps):
            self._finish(engine)
            return None
        if self.attempts>=12:
            self.status='paused'; self.message='达到单任务 12 次执行尝试上限。'; return None
        while self.index<len(self.proposal.steps) and effect_already_done(engine.state,self.proposal.steps[self.index].plan):
            self.trace.append({'revision':self.revision,'step':self.index+1,'skipped_existing_effect':True,
                               'committed':False,'plan':self.proposal.steps[self.index].plan.model_dump(mode='json')})
            self.index+=1
        if self.index>=len(self.proposal.steps):
            self._finish(engine); return None
        if engine.pending_visual and not may_act_while_pending(self.proposal.steps[self.index].plan):
            self.status='paused'; self.failure_type='visual_pending'
            self.message='请先处理上一动作的视觉问题，规则不会重复提交。'; return None
        step=self.proposal.steps[self.index]; self.attempts+=1
        if self._forbidden(step.plan,engine.state):
            self.status='paused'; self.failure_type='constraint_violation'
            self.message='剩余计划违反用户明确限制，未执行该动作。'; return None
        row=engine.interact(step.instruction,task_plan=step.plan,
                            operation_id=f'{self.task_id}:{self.revision}:{self.index}:{self.attempts}')
        self.trace.append({'step':self.index+1,'turn_id':row['turn_id'],'valid':row['valid'],
            'revision':self.revision,'operation_id':row['operation_id'],'committed':row.get('rule_committed',False),
            'reason':row['reason'],'visual_assessment':row.get('visual_assessment',{})})
        if not row['valid']:
            self._recover(engine,row); return row
        self.index+=1
        # An old unresolved effect must not prevent a user from looking elsewhere.
        # A NEW visual failure in this observation still pauses normally.
        pending_is_current=any(p.get('turn_id')==row['turn_id'] for p in engine.unresolved_visuals())
        if engine.pending_visual and (self.proposal.completion_mode!='action_execution' or pending_is_current):
            self.status='paused'; self.failure_type=engine.pending_visual['failure_type']
            self.message='规则动作已提交，但视觉结果需检查或修复；不会重复执行该动作。'
        elif self.pause_requested.is_set():
            self.status='paused'; self.message='本步已结束，后续步骤暂停。'
        elif self.index==len(self.proposal.steps): self._finish(engine)
        else:
            self.status='running'; self.message='下一步执行前仍会重新校验当前规则状态。'
        return row

    def resume(self):
        if self.status!='paused' or (self.attempts>=12 and self.index<len(self.proposal.steps)):
            raise ValueError('此任务不能直接恢复，请重新拆解目标。')
        self.status='ready'; self.message='已恢复；不撤销已经提交的步骤。'
        self.pause_requested.clear()

    def as_dict(self):
        return {'goal':self.goal,'status':self.status,'message':self.message,'next_index':self.index,
            'task_id':self.task_id,'replans':self.replans,'revision':self.revision,'failure_type':self.failure_type,
            'replan_history':self.replan_history,
            'precondition_failure':deepcopy(self.precondition_failure),
            'rejected_plan_count':len(self.rejected_plans),
            'planning_source':self.planning_source,'model_failure':self.model_failure,
            'explicit_forbidden_actions':sorted(self.forbidden_actions),
            'goal_coverage_enabled':self.goal_coverage_enabled,
            'required_ordered_calls':deepcopy(self.required_ordered_calls),
            'goal_coverage_failure':deepcopy(self.goal_coverage_failure),
            'attempts':self.attempts,'proposal':self.proposal.model_dump(mode='json'),'trace':self.trace,
            'visual_goal_verified':False}
