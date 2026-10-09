"""Resume an operator-pinned local archive into a new session, without generation.

No uploaded state, rule effects, visual approvals or old queued actions are executed.
The source archive is retained; each source can create only one continuation.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from .memory import Memory
from .models import WorldState, Observation, Plan


def logical_state(state):
    value=state.model_dump(mode='json')
    value.pop('environment')
    value['player'].pop('goal')
    return value


def resume_archive(engine,session_id,expected_sha256,*,allow_demo_advisory=False):
    if getattr(engine.renderer,'story_lineage_guard',False):
        raise ValueError('Story lineage forbids resuming a foreign historical session')
    if not re.fullmatch(r'[a-f0-9]{32}',session_id) or not re.fullmatch(r'[a-f0-9]{64}',expected_sha256):
        raise ValueError('A pinned server session and archive digest are required')
    with engine.lock:
        if engine.history or engine.task is not None or engine.frame is not None:
            raise ValueError('Cannot overwrite an active session')
        source=(engine.root/session_id).resolve()
        if source.parent!=engine.root: raise ValueError('Archive outside session root')
        archive=source/'session-export.json'; raw=archive.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected_sha256:
            raise ValueError('Archive changed; inspect current session before resuming')
        data=json.loads(raw)
        if data.get('story_profile','legacy')!=getattr(engine,'story_profile','legacy'):
            raise ValueError('Archive story profile mismatch; preserve the original session rather than migrate its effects')
        # A continuation may reference media in its immutable ancestors. Verify
        # every archived parent digest, not arbitrary caller-supplied paths.
        roots=[source]; lineage=data
        for _ in range(8):
            parent_info=lineage.get('session_resume')
            if not parent_info: break
            parent_id=parent_info['source_session_id']
            if not re.fullmatch(r'[a-f0-9]{32}',parent_id): raise ValueError('Invalid parent session')
            parent=(engine.root/parent_id).resolve()
            if parent.parent!=engine.root or parent in roots: raise ValueError('Invalid archive ancestry')
            parent_raw=(parent/'session-export.json').read_bytes()
            if hashlib.sha256(parent_raw).hexdigest()!=parent_info['source_archive_sha256']:
                raise ValueError('Parent archive changed')
            roots.append(parent); lineage=json.loads(parent_raw)
        else: raise ValueError('Archive ancestry too deep')
        policy_change=(allow_demo_advisory and engine.visual_review_policy=='demo_advisory'
            and data['visual_review_policy'] in ('manual_critical','model_gate'))
        if (data['session_id']!=session_id or data['renderer_kind']!='wan_laboratory'
            or getattr(engine.renderer,'kind',None)!='wan_laboratory'
            or data['agent_kind']!=engine.agent.kind
            or (data['visual_review_policy']!=engine.visual_review_policy and not policy_change)):
            raise ValueError('Archive runtime/policy mismatch')
        if (data.get('desk_inspection') or data.get('visual_reviews')
            or ((data.get('pending_visual') or data.get('pending_visual_queue')) and not policy_change)):
            raise ValueError('This resume path cannot discard visual review/retry/inspection state')
        if (data.get('task') or {}).get('status') not in ('needs_clarification','paused','completed_action_goal','completed_rule_goal'):
            raise ValueError('Only a stopped session may be resumed')
        rows=data['turns']
        disk_rows=[json.loads(line) for line in (source/'turns.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        if not rows or rows!=disk_rows: raise ValueError('Export differs from committed turn log')
        memory=Memory(); operations={}; previous=WorldState(); seeds=[]
        def artifact(name):
            path=Path(name).resolve()
            if not any(path.is_relative_to(root) for root in roots) or not path.is_file():
                raise ValueError('Missing or out-of-scope saved artifact')
            return str(path)
        initial=json.loads((roots[-1]/'initial/render.json').read_text())
        def check_render(meta):
            if (meta.get('checkpoint')!='R005-step1040' or meta.get('frames')!=97
                or meta.get('sampler',{}).get('cfg')!=engine.renderer.cfg
                or meta.get('lora_strength')!=engine.renderer.lora_strength or engine.renderer.seconds!=6):
                raise ValueError('Saved renderer settings differ from current runtime')
            seeds.append(meta['seed'])
        check_render(initial)
        for index,row in enumerate(rows,1):
            before=WorldState.model_validate(row['previous_state'])
            if row['turn_id']!=index or logical_state(before)!=logical_state(previous):
                raise ValueError('State chain does not match recorded history')
            if not row.get('rule_committed') or row.get('renderer_status')!='completed':
                raise ValueError('Only fully committed, stopped histories are supported here')
            plan=Plan.model_validate(row['plan']); valid=engine.rules.validate(before,plan)
            if not valid.valid or valid.state_delta!=row['state_delta'] or valid.story_event!=row['story_event']:
                raise ValueError('Saved effect does not match current rules')
            simulated=engine.graph.transition(before.apply_delta(valid.state_delta),valid.story_event)
            previous=WorldState.model_validate(row['state'])
            if logical_state(simulated)!=logical_state(previous):
                raise ValueError('Saved logical state fails CPU replay')
            artifact(row['video']); artifact(row['frame'])
            check_render(row['renderer'])
            op=row['operation_id']
            if op in operations: raise ValueError('Duplicate operation ID')
            operations[op]={'signature':{'text':row['player_text'],
                'plan':row['plan'] if row['plan_source']=='reviewed_goal_plan' else None},'row':deepcopy(row)}
            memory.update(row,previous)
        state=WorldState.model_validate(data['state'])
        if logical_state(state)!=logical_state(previous) or memory.retrieve()!=data['memory']:
            raise ValueError('Archive state/memory does not match committed turns')
        if data['current_frame']!=rows[-1]['frame'] or data['current_video']!=rows[-1]['video']:
            raise ValueError('Archive output differs from last committed turn')
        if seeds!=list(range(engine.renderer.seed,engine.renderer.seed+len(seeds))):
            raise ValueError('Unexpected generation sequence; do not reuse a seed')
        observation=Observation.model_validate(data['observation'])
        frame=artifact(data['current_frame']); video=artifact(data['current_video'])
        # Exclusive source claim prevents two browser sessions resuming the same work.
        receipt={'source_session_id':session_id,'source_archive_sha256':expected_sha256,
            'continuation_session_id':engine.session_id,'restored_turns':len(rows),
            'generated_videos':0,'applied_rule_effects':0,'old_plan_auto_resumed':False,
            'visual_success_verified':False,
            'policy_transition':({'from':data['visual_review_policy'],'to':engine.visual_review_policy,
                'authority':'user_20261006_complete_demo_relaxed_visual_quality',
                'human_visual_approval':False,
                'preserved_pending_visual':deepcopy(data.get('pending_visual')),
                'preserved_pending_visual_queue':deepcopy(data.get('pending_visual_queue',[]))}
                if policy_change else None)}
        with (source/'continuation-claim.json').open('x',encoding='utf-8') as stream:
            json.dump(receipt,stream,ensure_ascii=False,indent=2)
        engine.state=state; engine.memory=memory; engine.history=deepcopy(rows)
        engine.completed_operations=operations; engine.observation=observation
        engine.frame=frame; engine.video=video; engine.renderer.counter=len(seeds)
        engine.task_history=deepcopy(data.get('task_history',[]))+[deepcopy(data['task'])]
        engine.goal_planner_calls=deepcopy(data.get('goal_planner_calls',[]))
        engine.initial_recovery=data.get('initial_recovery')
        engine.visual_warnings=deepcopy(data.get('visual_warnings',[]))
        if policy_change:
            warnings=([data['pending_visual']] if data.get('pending_visual') else [])+data.get('pending_visual_queue',[])
            engine.visual_warnings.extend(dict(deepcopy(w),source='demo_policy_transition',
                visual_success_verified=False,blocks_story=False) for w in warnings)
        engine.session_resume=receipt
        # Persist inherited history in the child. Original archive/media remain untouched.
        (engine.folder/'source-session-export.json').write_bytes(raw)
        (engine.folder/'turns.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        (engine.folder/'state.json').write_text(state.serialize(),encoding='utf-8')
        engine.export_session()
        return receipt
