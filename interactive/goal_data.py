"""Small authored goal curriculum solved by actual rules, not model rollouts."""
import hashlib
import json
from collections import deque
from .data_workbench import reachable_states
from .models import Plan
from .rules import RuleEngine
from .story import StoryGraph
from .tasks import ToolGoalDraft,TOOL_IDS,check_goal
from .feedback import effect_already_done

GOALS=[
    ('key','选择独自调查，拿到红钥匙；不要开门，也不要进去。','key_owned','B',{'open','move_to'}),
    ('door','选择独自调查，拿到钥匙并打开门，但不要进入。','door_open','B',{'move_to'}),
    ('password','选择帮助 Anna，获得终端密码。','password_known','A',set()),
    ('look','别继续之前的计划，只看看左边。',None,None,set()),
    ('unsupported','走到桥边，再回头看山。',None,None,set()),
]


def solve(state,check,route,forbidden):
    rules,graph=RuleEngine(),StoryGraph()
    queue=deque([(state,[])]); seen=set()
    while queue and len(seen)<128:
        current,path=queue.popleft(); key=current.model_dump_json()
        if key in seen: continue
        seen.add(key)
        if check_goal(current,[check]): return path or ['observe:room']
        if len(path)>=8 or current.story.terminal: continue
        for tool_id,(action,target,dialogue) in TOOL_IDS.items():
            if action in forbidden: continue
            plan=Plan(player_intent='reference_solver',semantic_action=action,target=target,dialogue_action=dialogue)
            if effect_already_done(current,plan): continue
            valid=rules.validate(current,plan)
            if not valid.valid: continue
            try: after=graph.transition(current.apply_delta(valid.state_delta),valid.story_event)
            except ValueError: continue
            if after.story.route not in (None,route): continue
            if after!=current: queue.append((after,path+[tool_id]))
    return None


def build_goal_references():
    # Keep every request at an identical starting state in the same split.
    unique={}
    for state,path in reachable_states():
        if state.story.terminal: continue
        unique.setdefault(state.model_dump_json(),(state,path))
    ordered=sorted(unique.values(),key=lambda item:(len(item[1]),hashlib.sha256(item[0].model_dump_json().encode()).hexdigest()))
    buckets={route:[item for item in ordered if item[0].story.route==route] for route in (None,'A','B','C')}
    # Cover early long-horizon states, not a hash-selected majority of late route C states.
    selected=[]
    for split,quota in [('train',{None:2,'A':3,'B':4,'C':3}),('dev',{None:1,'A':1,'B':1}),
                         ('test',{None:2,'A':1,'B':1,'C':1})]:
        for route,count in quota.items():
            for _ in range(count):
                if not buckets[route]: raise ValueError('Insufficient reachable state coverage')
                selected.append((split,buckets[route].pop(0)))
    states=[item for split,item in selected]
    rows=[]
    for split,(state,path) in selected:
        family='goal-state-'+hashlib.sha256(state.model_dump_json().encode()).hexdigest()[:12]
        for name,text,check,route,forbidden in GOALS:
            if name=='look':
                draft=ToolGoalDraft(status='ready',completion_mode='action_execution',
                    steps=[{'instruction':text,'tool_id':'inspect:left'}])
            elif name=='unsupported' or (route and state.story.route not in (None,route)):
                draft=ToolGoalDraft(status='needs_clarification',clarification='当前实验室能力或已选路线不支持该目标；请修改目标。')
            else:
                result=solve(state,check,route,forbidden)
                if result is None: continue  # Bounded search exhaustion is NOT proof of impossibility.
                draft=ToolGoalDraft(status='ready',success_checks=[check],
                    steps=[{'instruction':'执行 '+t,'tool_id':t} for t in result])
            draft.compile()  # Validate the same contract as runtime, no rendering.
            rows.append({'id':family+'-'+name,'seed_group':family,'split':split,
                'player_input':text,'world_state':state.model_dump(mode='json'),'reachability_path':path,
                'expected_output':draft.model_dump(mode='json'),'source':'authored_goal_rule_search_reference',
                'scenario':name,'curriculum_version':2,'human_review':'pending','model_success':False,'visual_ground_truth':False})
    return rows,{'records':len(rows),'initial_state_families':len(states),'source':'authored task + finite rule search',
        'splits':{s:sum(r['split']==s for r in rows) for s in ('train','dev','test')},
        'multi_step_ready':{s:sum(r['split']==s and len(r['expected_output']['steps'])>=2 for r in rows) for s in ('train','dev','test')},
        'training_started':False,'limitations':'Same single world and five authored goals; not 100 independent tasks. Human review pending. No visual labels.'}


def _state_family(state):
    """Collapse optional entrance/visibility differences irrelevant to these goals.

    This is an in-world state holdout, not a held-out language or world benchmark.
    """
    return json.dumps([state.story.node,state.story.route,state.inventory.items,
        state.objects['red_key'].state,state.objects['door_a'].state,
        state.objects['terminal'].state,state.characters['anna'].trust,
        {k:bool(state.story.flags.get(k)) for k in ('met_anna','helped_anna',
            'password_known','found_key','log_retrieved','met_leo')}],sort_keys=True)


def _completion_states():
    from .models import WorldState
    rules,graph=RuleEngine(),StoryGraph(); queue=deque([(WorldState(),[])])
    states={}
    while queue and len(states)<64:
        state,path=queue.popleft(); family=_state_family(state)
        if family in states: continue
        states[family]=(state,path)
        if state.story.terminal or len(path)>=8: continue
        for tool_id,(action,target,dialogue) in TOOL_IDS.items():
            plan=Plan(player_intent='authored_reference',semantic_action=action,target=target,dialogue_action=dialogue)
            result=rules.validate(state,plan)
            if not result.valid: continue
            try: after=graph.transition(state.apply_delta(result.state_delta),result.story_event)
            except ValueError: continue
            if after!=state:
                queue.append((after,path+[{'plan':plan.model_dump(mode='json'),
                    'story_event':result.story_event,'reason':result.reason,'tool_id':tool_id}]))
    return {k:v for k,v in states.items() if not v[0].story.terminal}


def build_goal_completion_references():
    """New finite corpus; never overwrites the original five-goal/data-v3 set."""
    states=_completion_states(); grouped={}
    for key,(state,path) in sorted(states.items()):
        grouped.setdefault(state.story.route or 'none',[]).append((key,state,path))
    families=[]
    for values in grouped.values():
        for index,(key,state,path) in enumerate(values):
            # Every route with at least three current-state families reaches all splits.
            split='dev' if index==len(values)-2 else 'test' if index==len(values)-1 else 'train'
            families.append((state,path,split))
    specs=[
        ('key','选择独自调查，拿到红钥匙；不要开门，也不要进去。','key_owned','B',{'open','move_to'}),
        ('door','独自调查，找到钥匙后打开储藏室门；不要进去。','door_open','B',{'move_to'}),
        ('ending_key','选择独自调查，找到钥匙开门，再进入储藏室。','ending_key','B',set()),
        ('password','帮助 Anna 并取得密码；不要操作终端，也不要进入核心区域。','password_known','A',{'use','move_to','inspect'}),
        ('ending_anna','选择相信并帮助 Anna，取得密码、解锁终端并进入核心区域。','ending_anna','A',set()),
        ('public_logs','选择终端调查路线，读取公开日志；不要去找 Leo。','public_logs','C',{'approach'}),
        ('ending_terminal','选择终端路线，恢复日志、找到 Leo 并向他核实记录。','ending_terminal','C',set()),
    ]
    rows=[]
    for number,(state,path,split) in enumerate(families):
        family=f'completion-state-{number:03d}'
        def append(name,text,draft,forbidden=(),route=None,feedback=None,allowed_tools=None):
            row={'id':family+'-'+name,'seed_group':family,'split':split,
                'world_state':state.model_dump(mode='json'),'reachability_path':path,
                'player_input':text,'expected_output':draft.model_dump(mode='json'),
                'source':'authored_goal_rule_search_reference','scenario':name,
                'forbidden_actions':sorted(forbidden),'required_route':route,
                'feedback':feedback,'allowed_tools':allowed_tools,
                'human_review':'pending','model_success':False,'visual_ground_truth':False}
            rows.append(row)
        for name,text,check,route,forbidden in specs:
            if state.story.route not in (None,route):
                draft=ToolGoalDraft(status='needs_clarification',clarification='当前已选路线不兼容，不能擅自改写分支；请重置或修改目标。')
            else:
                result=solve(state,check,route,forbidden)
                if result is None: continue  # finite search exhaustion is not an impossibility label
                draft=ToolGoalDraft(status='ready',success_checks=[check],
                    steps=[{'instruction':'执行 '+t,'tool_id':t} for t in result])
            append(name,text,draft,forbidden,route)
            if (draft.status=='ready' and state.story.route==route and
                    not check_goal(state,[check])):
                # Authored failure context, NOT a claimed model trajectory or visual label.
                feedback={'failure':'先前提案没有覆盖完整目标，请仅补齐尚未完成的步骤。',
                    'latest_turn':None,'committed_steps':[],
                    'required_success_checks':[check],'required_completion_mode':'rule_state',
                    'remaining_only':True}
                append(name+'-repair',text,draft,forbidden,route,feedback)
        for name,text,tool in [('look','别继续之前的计划，只看看左边。','inspect:left'),
                              ('wait','暂停推进剧情，只在这里等待。','wait:room')]:
            append(name,text,ToolGoalDraft(status='ready',completion_mode='action_execution',
                steps=[{'instruction':text,'tool_id':tool}]),allowed_tools=[tool])
        append('unsupported','走到桥边，再回头看山。',ToolGoalDraft(status='needs_clarification',
            clarification='当前仅支持实验室，没有桥或山，不能承诺精确导航或视角目标。'))
        # Explicitly requested direct action: the model may NOT invent missing prerequisites.
        for name,text,tool,check,route in [
            ('direct_open','只打开门，不允许搜索、拿钥匙、对话或做其他动作。','open:door_a','door_open','B'),
            ('direct_password','只向 Anna 问密码，不允许接近她、帮助她或做其他动作。','ask:anna:password','password_known','A')]:
            action,target,dialogue=TOOL_IDS[tool]
            plan=Plan(player_intent=name,semantic_action=action,target=target,dialogue_action=dialogue)
            valid=RuleEngine().validate(state,plan)
            try:
                if not valid.valid: raise ValueError(valid.reason)
                StoryGraph().transition(state.apply_delta(valid.state_delta),valid.story_event)
                draft=ToolGoalDraft(status='ready',success_checks=[check],steps=[{'instruction':text,'tool_id':tool}])
            except ValueError:
                draft=ToolGoalDraft(status='needs_clarification',clarification='直接动作的前置条件不满足，且用户未授权补齐前置动作。')
            append(name,text,draft,route=route,allowed_tools=[tool])
    # Validate references and compare family/state separation before any training launch.
    from .dataset import validate_row
    for row in rows:
        try: validate_row(row)
        except ValueError as exc: raise ValueError(f"Invalid authored reference {row['id']}: {exc}") from exc
    summary={'records':len(rows),'state_families':len(families),'training_started':False,
        'splits':{s:sum(r['split']==s for r in rows) for s in ('train','dev','test')},
        'source':'Authored laboratory goals and bounded rule-search references; no teacher API or visual labels.',
        'limitations':['Same goal wording and same authored world repeat across different current states.',
            'State-family holdout measures in-world combinatorial planning, not unseen-language/world generalization.',
            'Reference feedback is authored, not captured model failure data. Human review pending.',
            'Rule-goal completion does not establish generated-video action success.']}
    return rows,summary


def score_goal_predictions(rows,predictions):
    """Goal-aware execution scoring; wrong but executable subgoals never pass."""
    from .models import WorldState
    from .feedback import effect_already_done
    metrics={k:{'correct':0,'total':0} for k in ('json_valid','status_correct','ready_goal_success',
        'clarification_correct','goal_contract_preserved','constraint_preserved',
        'preconditions_valid','remaining_only','joint_success')}
    details=[]
    for row in rows:
        expected=row['expected_output']; ready=expected['status']=='ready'
        values={k:False for k in metrics}; values['remaining_only']=True
        applicable={'json_valid','status_correct','joint_success'}
        applicable.update(('ready_goal_success','goal_contract_preserved','constraint_preserved',
                           'preconditions_valid') if ready else ('clarification_correct',))
        if ready and row.get('feedback'): applicable.add('remaining_only')
        detail={'id':row['id'],'scenario':row.get('scenario'),'split':row.get('split')}
        try:
            text=predictions.get(row['id'],'').strip()
            if text.startswith('```') and text.endswith('```'): text='\n'.join(text.splitlines()[1:-1])
            draft=ToolGoalDraft.model_validate(json.loads(text)); proposal=draft.compile()
            values['json_valid']=True; values['status_correct']=draft.status==expected['status']
            if not ready:
                values['clarification_correct']=draft.status=='needs_clarification'
                values['joint_success']=values['clarification_correct']
            elif draft.status=='ready':
                values['goal_contract_preserved']=(set(draft.success_checks)==set(expected['success_checks']) and
                    draft.completion_mode==expected['completion_mode'])
                state=WorldState.model_validate(row['world_state']); rules,graph=RuleEngine(),StoryGraph()
                forbidden=set(row.get('forbidden_actions',[])); allowed=row.get('allowed_tools')
                values['constraint_preserved']=all(
                    step.plan.semantic_action not in forbidden and (not allowed or compact.tool_id in allowed)
                    for step,compact in zip(proposal.steps,draft.steps))
                valid=True
                for step in proposal.steps:
                    if effect_already_done(state,step.plan):
                        if row.get('feedback'): values['remaining_only']=False
                        continue  # runtime skips committed effects; never credit replay as progress
                    result=rules.validate(state,step.plan)
                    if not result.valid: valid=False; detail['rule_error']=result.reason; break
                    try: state=graph.transition(state.apply_delta(result.state_delta),result.story_event)
                    except ValueError as exc: valid=False; detail['rule_error']=str(exc); break
                    if row.get('required_route') and state.story.route not in (None,row['required_route']):
                        values['constraint_preserved']=False
                values['preconditions_valid']=valid
                achieved=(check_goal(state,expected['success_checks']) if expected['completion_mode']=='rule_state'
                    else bool(proposal.steps))
                values['ready_goal_success']=all(values[k] for k in (
                    'goal_contract_preserved','constraint_preserved','preconditions_valid','remaining_only')) and achieved
                values['joint_success']=values['ready_goal_success']
        except (ValueError,TypeError,AttributeError) as exc:
            detail['error']=str(exc)[:500]
        for key in applicable:
            metrics[key]['total']+=1; metrics[key]['correct']+=int(values[key])
        detail.update({key:values[key] for key in applicable}); details.append(detail)
    for value in metrics.values(): value['rate']=value['correct']/value['total'] if value['total'] else None
    return {'metrics':metrics,'details':details,'protocol':'goal_completion_state_holdout_v1',
        'limitations':'Strict first-proposal same-contract scoring; no retries, no renderer, no visual success. Authored in-world state holdout only.'}
