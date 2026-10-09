"""Narrow authored key skill, explicitly NOT learned reasoning or general NL parsing.

Used only when the model's plan fails. No rules are bypassed, and this module
never executes a tool or changes the world. Unknown constraints disable fallback.
"""
import re
from .models import Plan
from .feedback import effect_already_done


def key_request(text):
    clauses=[s.strip() for s in re.split(r'[，,。；;]',text) if s.strip()]
    positives=[]
    for clause in clauses:
        neg=re.match(r'^(?:但|也|并且|同时)?\s*(?:不要|别|不允许)(.*)$',clause)
        if neg:
            if not re.fullmatch(r'(?:开门|打开门|打开储藏室门|进入储藏室|进去|进入|(?:接近|靠近|走近)\s*Anna)',neg[1],re.I):
                return False
        else: positives.append(clause)
    positive='，'.join(positives)
    if re.search(r'桥|山|终端|密码|开门|打开.*门|进去|进入|只(?:拿|拾取)|不准|不能|不要|不允许',positive): return False
    return bool(re.search(r'(?:找(?:到)?|寻找|拿(?:到|起)?|取得|获取|拾取)(?:红)?钥匙',positive))


def key_skill_proposal(engine,text):
    if not key_request(text): return None
    from .tasks import ToolGoalDraft
    if engine.state.story.route not in (None,'B'):
        return ToolGoalDraft(status='needs_clarification',clarification='规则技能兜底：当前路线不支持取钥匙，不能擅自切换分支。').compile()
    if (not engine.state.story.flags.get('met_anna') and
            re.search(r'(?:不要|别|不允许)(?:接近|靠近|走近)\s*Anna',text,re.I)):
        return ToolGoalDraft(status='needs_clarification',clarification='规则技能兜底：取钥匙需要先接近 Anna，与本次禁止接近的要求冲突；未执行任何动作。').compile()
    calls=[('approach','anna',None),('talk','anna','distrust'),('search','desk',None),('pick_up','red_key',None)]
    state=engine.state.model_copy(deep=True); steps=[]
    for action,target,dialogue in calls:
        plan=Plan(player_intent='authored retrieve_key skill',semantic_action=action,target=target,dialogue_action=dialogue)
        if 'red_key' in state.inventory.items: break
        if effect_already_done(state,plan): continue
        result=engine.rules.validate(state,plan)
        if not result.valid: return None
        try: state=engine.graph.transition(state.apply_delta(result.state_delta),result.story_event)
        except ValueError: return None
        tool_id=':'.join(x for x in (action,target,dialogue) if x)
        steps.append({'instruction':'规则技能建议：'+tool_id,'tool_id':tool_id})
    if 'red_key' not in state.inventory.items: return None
    if not steps: steps=[{'instruction':'钥匙已在背包，不重复拾取，仅保持观察。','tool_id':'observe:room'}]
    return ToolGoalDraft(status='ready',success_checks=['key_owned'],steps=steps).compile()


def key_route_request(text):
    """Conservative supported-language recognizer, NOT general model intent parsing.

    Every clause must be known; unsupported objects, prohibitions or extra goals
    disable the fallback. A key-only request never becomes a whole story route.
    """
    patterns={
        'route':r'(?:我)?不(?:相信|信任)\s*Anna|选择独自调查',
        'search':r'(?:搜索|搜查)桌子',
        'key':r'(?:拿起|拾取|拿到)(?:红)?钥匙',
        'door':r'(?:用(?:红)?钥匙)?(?:打开|开)(?:储藏室)?门',
        'enter':r'进入储藏室',
        'ending':r'(?:完成|走完)钥匙路线',
        'no_repeat':r'已经完成的动作不要重复|不要重复已经完成的动作',
    }
    found=set()
    for clause in [s.strip() for s in re.split(r'[，,。；;]',text) if s.strip()]:
        clause=re.sub(r'^(?:然后|接着|再)','',clause)
        matches=[key for key,pattern in patterns.items() if re.fullmatch(pattern,clause,re.I)]
        if not matches: return False
        found.update(matches)
    return {'route','key','door','enter','ending'}<=found


def key_route_skill_proposal(engine,text):
    if not key_route_request(text): return None
    from .skill_contracts import preview_skill
    from .tasks import GoalProposal,GoalStep,GoalTask
    preview=preview_skill(engine,'retrieve_key',goal=text,allow_route_selection=True)
    if preview['status']=='completed_rule_only': steps=[]
    elif preview['status']=='rule_plan_ready':
        steps=GoalProposal.model_validate(preview['proposal']).steps
    else: return None
    # Route and terminal checks still apply when the key is already held.
    if engine.state.story.route not in (None,'B') or engine.state.story.terminal: return None
    for action,target in [('open','door_a'),('move_to','storage_room')]:
        plan=Plan(player_intent='authored full key-route skill',semantic_action=action,target=target)
        if not effect_already_done(engine.state,plan):
            steps.append(GoalStep(instruction='规则技能建议：'+action+':'+target,plan=plan))
    if not steps: return None
    proposal=GoalProposal(status='ready',steps=steps,success_checks=['ending_key'])
    try: GoalTask(text,proposal).preflight(engine)
    except ValueError: return None
    return proposal
