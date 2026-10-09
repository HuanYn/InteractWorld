"""Explicit visual evidence requirements; authored task semantics, not ground truth."""

def event_contract(plan):
    if plan.semantic_action=='inspect' and plan.target in ('left','right'):
        return {'id':'camera_pan_'+plan.target,'requirements':{
            'viewpoint_motion':f'The CAMERA turns toward the {plan.target} side of the scene. Judge background landmarks and newly revealed space, not a person looking or moving. Left/right is a direction, not an object label.',
            'scene_continuity':'Enough stable background structure remains to distinguish a camera turn from scene morphing or a cut.'}}
    if plan.semantic_action=='search' and plan.target in ('desk','room','red_key'):
        return {'id':'search_and_reveal_key','requirements':{
            'search_activity':'The requested search area is shown. Mere character proximity or general gesturing is insufficient.',
            'key_evidence':'A small red key with distinguishable key geometry is visibly revealed in the search area. A red bottle, light, blob or unresolvable object is not a key. Missing detail means uncertain, not supported.',
            'object_continuity':'The key and searched furniture have interpretable temporal continuity, rather than appearing through unrelated object morphing.'}}
    if plan.semantic_action=='approach':
        return {'id':'camera_approach','requirements':{
            'viewpoint_motion':'The CAMERA moves closer to the requested target. Subject enlargement alone, character motion toward the camera, or a cut is insufficient.',
            'scene_continuity':'Background and target continuity provide coherent evidence of approach; name/identity is not proven.'}}
    return None

def qualify_contract(result,contract):
    result=dict(result)
    criteria=result.get('criteria',{})
    required=set(contract['requirements'])
    allowed={'supported','contradicted','uncertain','not_visually_testable'}
    if set(criteria)!=required or any(v not in allowed for v in criteria.values()):
        result.update(status='uncertain',qualification='Missing or invalid required visual criteria')
    elif 'contradicted' in criteria.values():
        result.update(status='contradicted',qualification='At least one required visual criterion contradicted')
    elif any(v!='supported' for v in criteria.values()):
        result.update(status='uncertain',qualification='Not all required visual criteria are supported')
    result['visual_contract']=contract
    return result
