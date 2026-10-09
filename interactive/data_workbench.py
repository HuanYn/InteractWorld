"""Auditable, rule-reachable task data. No LLM paraphrase is silently relabelled."""
import hashlib
import json
from collections import deque
from .models import WorldState,Plan
from .rules import RuleEngine
from .story import StoryGraph
from .tools import TOOL_CALLS
from .agent import SkillSelector
from .dataset import validate_row


TEXT = {
    ('observe','room',None):'先观察房间，不改变剧情。',
    ('wait','room',None):'暂时等待，不执行其他操作。',
    ('explore','laboratory',None):'进入实验室查看情况。',
    ('inspect','left',None):'只检查左侧的环境。',
    ('inspect','right',None):'先看看右边。',
    ('inspect','desk',None):'检查桌面，不搜索抽屉。',
    ('inspect','door_a',None):'观察门，不尝试开门。',
    ('approach','anna',None):'走近 Anna，但先不帮助她。',
    ('talk','anna',None):'与 Anna 交谈，暂时不选立场。',
    ('interact','anna','help'):'帮助 Anna 稳定情绪。',
    ('talk','anna','trust'):'告诉 Anna 我信任她。',
    ('talk','anna','distrust'):'告诉 Anna 我不信任她，决定独自调查。',
    ('ask','anna','password'):'向 Anna 询问终端密码；如果前置条件不满足，解释原因，不擅自执行其他动作。',
    ('avoid','anna',None):'暂时不与 Anna 接触，选择终端调查路线。',
    ('search','desk',None):'搜索桌子的抽屉寻找钥匙；若当前不允许，不擅自更换路线。',
    ('pick_up','red_key',None):'拿起红钥匙；若当前无法拾取，解释原因，不执行其他步骤。',
    ('open','door_a',None):'尝试打开储藏室门；若没有钥匙，告诉我原因，不自行寻找钥匙。',
    ('inspect','terminal',None):'继续检查终端，执行当前规则允许的下一步调查。',
    ('use','terminal',None):'操作终端，按当前授权查看信息。',
    ('move_to','core_area',None):'进入核心区域；如果还未解锁则说明原因，不自行索取密码。',
    ('move_to','storage_room',None):'走进储藏室；如果不能进入，解释原因。',
    ('approach','leo',None):'去找 Leo；如果还不知道位置，先说明限制。',
    ('ask','leo','logs'):'向 Leo 核实安全日志；如果还不能询问，请说明原因。',
}


def reachable_states():
    """Finite BFS over real rule transitions, never arbitrary state-field mutation."""
    rules,graph=RuleEngine(),StoryGraph()
    queue=deque([(WorldState(),[])])
    seen=set(); found=[]
    while queue and len(found)<64:
        state,path=queue.popleft()
        key=state.model_dump_json()
        if key in seen: continue
        seen.add(key); found.append((state,path))
        if state.story.terminal or len(path)>=8: continue
        for action,target,dialogue in TOOL_CALLS:
            plan=Plan(player_intent='scripted_reachability',semantic_action=action,target=target,dialogue_action=dialogue)
            result=rules.validate(state,plan)
            if not result.valid: continue
            try: after=graph.transition(state.apply_delta(result.state_delta),result.story_event)
            except ValueError: continue
            if after==state: continue
            queue.append((after,path+[{'plan':plan.model_dump(mode='json'),'reason':result.reason,
                                      'story_event':result.story_event}]))
    return found


def build_decisions(limit=100):
    """State/tool families, not 100 independent environments or human labels."""
    if not 20<=limit<=300: raise ValueError('Keep the first corpus bounded to 20..300 families')
    rules,graph=RuleEngine(),StoryGraph(); candidates=[]
    states=reachable_states()
    for state,path in states:
        if state.story.terminal: continue
        # Collapse states differing only in whether the optional entrance step occurred.
        family_state={'node':state.story.node,'route':state.story.route,
                      'key':state.objects['red_key'].state,'door':state.objects['door_a'].state,
                      'terminal':state.objects['terminal'].state,
                      'password':bool(state.story.flags.get('password_known'))}
        for action,target,dialogue in TOOL_CALLS:
            if (action,target,dialogue) not in TEXT: continue
            plan=Plan(player_intent='requested_action',semantic_action=action,target=target,
                      dialogue_action=dialogue,confidence=1,reason_summary='只处理用户请求的动作，按当前状态核验前置条件。')
            result=rules.validate(state,plan); valid=result.valid
            if valid:
                try: graph.transition(state.apply_delta(result.state_delta),result.story_event)
                except ValueError: valid=False
            plan.expected_valid=valid; plan.camera_skills=SkillSelector().select(plan)
            identity=json.dumps([family_state,action,target,dialogue],sort_keys=True)
            fid='state-tool-'+hashlib.sha256(identity.encode()).hexdigest()[:12]
            row={'id':fid,'seed_group':fid,'source':'programmatic_reachable_state_authored_template',
                 'world_state':state.model_dump(mode='json'),'memory':{'recent':path[-5:],'important':[]},
                 'player_input':TEXT[(action,target,dialogue)],'expected_output':plan.model_dump(mode='json'),
                 'reachability_path':path,'rule_valid':valid,'rule_reason':result.reason,
                 'human_review':'pending','visual_ground_truth':False,
                 'source_note':'Project-generated rule-world reference, not a real-player trajectory or model success.'}
            validate_row(row); candidates.append(row)
    # Unique family and balanced labels; use a deterministic order, not model scores.
    unique={}
    for row in candidates: unique.setdefault(row['seed_group'],row)
    bins={valid:sorted([r for r in unique.values() if r['rule_valid']==valid],key=lambda r:r['id']) for valid in (True,False)}
    rows=[]
    for index in range(max(map(len,bins.values()))):
        for valid in (True,False):
            if index<len(bins[valid]) and len(rows)<limit: rows.append(bins[valid][index])
    for index,row in enumerate(rows):
        row['split']='train' if index%20<12 else 'dev' if index%20<15 else 'test'
    summary={'schema_version':1,'source':'rule-reachable generated references',
             'reachable_states':len(states),'candidate_records':len(candidates),'unique_families':len(unique),
             'retained_families':len(rows),'retained_records':len(rows),
             'splits':{s:sum(r['split']==s for r in rows) for s in ('train','dev','test')},
             'valid':sum(r['rule_valid'] for r in rows),'invalid':sum(not r['rule_valid'] for r in rows),
             'human_review':'pending','model_rollouts':0,'training_started':False,
             'limitations':['Single authored rule world; not independent worlds or external generalization.',
                            'Templates repeat across states; tests measure state/tool combinations, not unseen language.',
                            'Final human-reviewed task-family test must be added before final project claims.']}
    return rows,summary


def visual_candidates(session_exports):
    """Reuse actual raw outputs, leaving all human judgments unfilled."""
    records=[]; seen=set()
    for path in session_exports:
        if path.suffix=='.jsonl':
            payload={'session_id':path.parent.name,'turns':[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]}
        else:
            payload=json.loads(path.read_text(encoding='utf-8'))
        for turn in payload.get('turns',[]):
            meta=turn.get('renderer',{})
            if (turn.get('renderer_status')!='completed' or meta.get('synthetic_storyboard') or meta.get('pre_generated')
                    or not str(meta.get('kind','')).startswith('wan')):
                continue
            video=meta.get('raw_video_path') or meta.get('video_path') or turn.get('video')
            if not video or video in seen: continue
            seen.add(video)
            records.append({'id':f'visual-{len(records):04d}','source_session':payload.get('session_id',str(path.parent)),
                'source_type':'own_generated_video','source_record':str(path),'source_turn_id':turn.get('turn_id'),
                'raw_video':video,'before_frame':turn.get('previous_frame') or meta.get('conditioning_image'),
                'after_frame':turn.get('frame'),'requested_plan':turn.get('plan'),
                'rule_committed':turn.get('rule_committed',turn.get('valid')),
                'model_judgment':turn.get('visual_assessment'),
                'actual_event':None,'event_start_seconds':None,'event_end_seconds':None,
                'human_verdict':None,'human_reason':None,'review_status':'pending',
                'training_eligible':False,'action_source':'see_original_renderer_receipt',
                'ground_truth':False})
    return records
