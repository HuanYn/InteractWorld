"""Disclosed typed-tool completion compiler, not a learned planner repair.

Only an explicit single direction observation can acquire action-only semantics.
No tool is inserted, removed, retargeted or executed; raw model output is retained.
"""
from copy import deepcopy
import re


def direction_observation_target(goal):
    match=re.fullmatch(r'\s*(?:别继续之前的计划[，,。\s]*)?(?:只)?(?:看看|看|观察)'
        r'(左边|右边)(?:[，,]\s*(?:不要|别)(?:移动或操作物品|移动|操作物品))?[。.!\s]*',goal)
    return {'左边':'left','右边':'right'}.get(match[1]) if match else None


def compile_direction_completion(goal, raw):
    """Return unchanged input or a separately attributable compiled JSON draft.

    The original completion labels can be incorrect but must have the original
    bounded scalar/list shape. All other schema fields and the sole model tool
    are validated before any metadata is derived.
    """
    target=direction_observation_target(goal)
    if (target is None or not isinstance(raw,dict) or raw.get('status')!='ready'
            or raw.get('clarification','')!='' or not isinstance(raw.get('steps'),list)
            or len(raw['steps'])!=1 or not isinstance(raw['steps'][0],dict)
            or raw['steps'][0].get('tool_id')!='inspect:'+target):
        return raw,None
    checks=raw.get('success_checks',[]);mode=raw.get('completion_mode','rule_state')
    if (mode not in ('rule_state','action_execution') or not isinstance(checks,list)
            or len(checks)>6 or any(not isinstance(x,str) or not 1<=len(x)<=100 for x in checks)):
        return raw,None
    from .spatial_skills import SpatialToolGoalDraft
    compiled=deepcopy(raw)
    compiled.update(completion_mode='action_execution',success_checks=[])
    # Unknown fields, malformed instructions and all other constraints still fail.
    SpatialToolGoalDraft.model_validate(compiled).compile()
    if mode=='action_execution' and checks==[]:
        return raw,None
    receipt=dict(version='pure_direction_completion_v1',
        source='typed_tool_completion_compiler_not_model_repair',goal=goal,
        original_draft=deepcopy(raw),compiled_draft=deepcopy(compiled),
        original_completion_mode=mode,original_success_checks=deepcopy(checks),
        steps_unchanged=compiled['steps']==raw['steps'],added_or_removed_actions=False,
        model_autonomous_repair=False,model_calls=0,world_state_changed=False,
        completion_means='supported_action_executed_not_visual_direction_verified',
        visual_success_verified=False)
    return compiled,receipt
