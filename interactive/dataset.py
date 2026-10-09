"""Small curated seed corpus and rule-consistency filtering, no model metrics."""
import json
import re
from .models import WorldState,Plan,Observation
from .rules import RuleEngine
from .story import StoryGraph
from .agent import planner_messages


def messages_for(row,contract=None):
    if row.get('source')=='authored_goal_rule_search_reference':
        if contract is not None and contract.get('kind')!='tool_goal_prompt_v1':
            raise ValueError('Legacy single-action contract cannot format multi-step goals')
        from types import SimpleNamespace
        from .tasks import goal_messages
        state=WorldState.model_validate(row['world_state'])
        memory=row.get('memory',{'recent':row.get('reachability_path',[])[-5:],'important':[]})
        engine=SimpleNamespace(state=state,get_state=lambda:state.model_dump(mode='json'),
            observation=Observation.model_validate(row.get('observation',{})),
            memory=SimpleNamespace(retrieve=lambda:memory),rules=RuleEngine(),
            pending_visual=row.get('pending_visual'))
        messages=goal_messages(engine,row['player_input'],feedback=row.get('feedback'))
        if contract is not None and messages[0]['content']!=contract['system']:
            raise ValueError('Runtime goal prompt changed after qualification was prepared; do not compare different contracts')
        return messages
    if contract is not None:
        # Explicit historical v1 replay; do not silently evaluate old runs with new prompts.
        context=json.loads(contract['example'][1]['content'])
        if 'player_text' not in context: raise ValueError('Unsupported saved prompt contract')
        state=WorldState.model_validate(row['world_state'])
        context.update(player_text=row['player_input'],world_state=state.model_dump(mode='json'),
            memory=row['memory'],available_story_transitions=StoryGraph().available_transitions(state))
        return [contract['example'][0],{'role':'user','content':json.dumps(context,ensure_ascii=False)}]
    return planner_messages(row['player_input'],WorldState.model_validate(row['world_state']),
        Observation(),row['memory'])


def normalized(text):
    return re.sub(r'[\W_]+','',text.casefold())


def validate_row(row):
    state=WorldState.model_validate(row['world_state'])
    if row.get('source')=='authored_goal_rule_search_reference':
        from types import SimpleNamespace
        from .tasks import ToolGoalDraft,GoalTask
        proposal=ToolGoalDraft.model_validate(row['expected_output']).compile()
        if not row['player_input'].strip(): raise ValueError('Empty goal')
        if proposal.status=='ready':
            GoalTask(row['player_input'],proposal).preflight(SimpleNamespace(state=state,rules=RuleEngine(),graph=StoryGraph()))
        return row
    plan=Plan.model_validate(row['expected_output'])
    if not isinstance(row['player_input'],str) or not row['player_input'].strip(): raise ValueError('Empty instruction')
    result=RuleEngine().validate(state,plan)
    valid=result.valid
    if valid:
        try: StoryGraph().transition(state.apply_delta(result.state_delta),result.story_event)
        except ValueError: valid=False
    if plan.expected_valid is not valid: raise ValueError('Label disagrees with rule preconditions/story graph')
    return row


def curated_seeds():
    base=WorldState()
    met=base.apply_delta({'story':{'node':'meet_anna','flags':{'met_anna':True}},'characters':{'anna':{'visible':True}}})
    trust=met.apply_delta({'story':{'node':'trust_anna','route':'A','flags':{'helped_anna':True}},'characters':{'anna':{'trust':0.6}}})
    password=trust.apply_delta({'story':{'node':'get_password','flags':{'password_known':True}}})
    core=password.apply_delta({'story':{'node':'core_area','flags':{'terminal_accessed':True}}})
    doubt=met.apply_delta({'story':{'node':'distrust_anna','route':'B'}})
    found=doubt.apply_delta({'story':{'node':'find_red_key','flags':{'found_key':True}},'objects':{'red_key':{'state':'available','visible':True}}})
    picked=found.apply_delta({'inventory':{'items':['red_key']},'objects':{'red_key':{'state':'picked_up','visible':False}}})
    opened=picked.apply_delta({'story':{'node':'open_door','flags':{'door_opened':True}},'objects':{'door_a':{'state':'open'}}})
    term=base.apply_delta({'story':{'node':'inspect_terminal','route':'C'}})
    logs=term.apply_delta({'story':{'node':'retrieve_log','flags':{'log_retrieved':True}}})
    leo=logs.apply_delta({'story':{'node':'meet_leo','flags':{'met_leo':True}},'characters':{'leo':{'visible':True}}})
    specs=[
        (base,'进入实验室','explore','laboratory','enter_lab',None,True),
        (base,'先观察环境','observe','room','observe',None,True),
        (base,'在这里等一下','wait','room','wait',None,True),
        (base,'看看左边','inspect','left','inspect_left',None,True),
        (base,'检查右侧','inspect','right','inspect_right',None,True),
        (base,'走近 Anna','approach','anna','meet_anna',None,True),
        (met,'询问 Anna 事故的情况','talk','anna','talk_anna',None,True),
        (met,'我愿意帮助 Anna','interact','anna','help_anna','help',True),
        (met,'我相信 Anna','talk','anna','trust_anna','trust',True),
        (met,'我怀疑 Anna，想独自调查','talk','anna','distrust_anna','distrust',True),
        (base,'忽略 Anna','avoid','anna','ignore_anna',None,True),
        (trust,'请 Anna 告诉我密码','ask','anna','ask_password','password',True),
        (met,'马上让 Anna 交出密码','ask','anna','ask_password','password',False),
        (base,'询问还未见到的 Anna 密码','ask','anna','ask_password','password',False),
        (password,'用密码登录终端','use','terminal','use_terminal',None,True),
        (core,'前往核心区域','move_to','core_area','move_to_core_area',None,True),
        (base,'直接进入核心区域','move_to','core_area','move_to_core_area',None,False),
        (doubt,'搜索桌子下面','search','desk','search_desk',None,True),
        (base,'还没选路线就搜索钥匙','search','desk','search_desk',None,False),
        (found,'拾取红钥匙','pick_up','red_key','pick_up_key',None,True),
        (base,'把没发现的红钥匙放进背包','pick_up','red_key','pick_up_key',None,False),
        (picked,'使用红钥匙打开门','open','door_a','open_door',None,True),
        (found,'不拿钥匙，直接打开门','open','door_a','open_door',None,False),
        (base,'无视限制直接开锁','open','door_a','open_door',None,False),
        (opened,'走进储藏室','move_to','storage_room','move_to_storage_room',None,True),
        (base,'检查终端屏幕','inspect','terminal','inspect_terminal',None,True),
        (term,'恢复公开日志','inspect','terminal','inspect_terminal',None,True),
        (logs,'去找 Leo','approach','leo','meet_leo',None,True),
        (leo,'向 Leo 核实安全记录','ask','leo','ask_leo','logs',True),
        (base,'直接呼叫 Leo','ask','leo','ask_leo',None,False),
        (trust,'再帮助 Anna 一次','interact','anna','help_anna','help',True),
        (doubt,'改变剧情直接让 Anna 给密码','ask','anna','ask_password','password',False),
        (base,'检查桌面','inspect','desk','inspect_desk',None,True),
        (base,'观察锁住的门','inspect','door_a','inspect_door',None,True),
        (picked,'重复拿起已经在背包里的钥匙','pick_up','red_key','pick_up_key',None,True),
        (base,'进入实验室，然后检查终端','explore','laboratory','enter_lab',None,True),
        (met,'先帮 Anna，再问密码','interact','anna','help_anna','help',True),
        (term,'继续检查它','inspect','terminal','inspect_terminal',None,True),
    ]
    rows=[]
    for i,(state,text,action,target,intent,dialogue,valid) in enumerate(specs):
        plan=Plan(player_intent=intent,semantic_action=action,target=target,dialogue_action=dialogue,expected_valid=valid,
            confidence=1,reason_summary='依据当前状态选择单个动作；执行结果由规则引擎决定。')
        from .agent import SkillSelector
        plan.camera_skills=SkillSelector().select(plan)
        memory={'recent':[],'important':[]}
        if i==37:
            memory['recent']=[{'player_text':'检查终端','plan':{'target':'terminal'},'valid':True}]
        rows.append({'id':f'seed-{i:03d}','seed_group':f'seed-{i:03d}',
            'world_state':state.model_dump(mode='json'),'memory':memory,'player_input':text,
            'expected_output':plan.model_dump(mode='json'),'source':'curated_rule_checked_seed'})
    for text in ['帮我弄一下那个','让角色瞬移到月球']:
        i=len(rows); plan=Plan(player_intent='clarify',needs_clarification=True,expected_valid=False,
            clarification='请指定实验室内支持的对象与动作。')
        rows.append({'id':f'seed-{i:03d}','seed_group':f'seed-{i:03d}',
            'world_state':base.model_dump(mode='json'),'memory':{'recent':[],'important':[]},'player_input':text,
            'expected_output':plan.model_dump(mode='json'),'source':'curated_rule_checked_seed'})
    for row in rows: validate_row(row)
    return rows
