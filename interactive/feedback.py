"""Conservative visual gates. Model opinions never become narrative truth."""


def assess_prebuilt(metadata, observer_assessment):
    """A saved clip review can veto a known failure, not verify a fresh action."""
    review=metadata.get('asset_review')
    if not metadata.get('pre_generated') or not isinstance(review,dict):
        return observer_assessment
    verdict=review.get('ai_verdict')
    # Even a supported offline clip does not prove current-state alignment or
    # that the current instruction caused this event. Keep it uncertain.
    status='contradicted' if verdict=='mismatch' or review.get('quality_exclusion') else 'uncertain'
    return {'status':status,'source':'prebuilt_asset_ai_review',
        'evidence':review.get('actual_event','预生成素材缺少已验证的事件证据。'),
        'asset_id':metadata.get('asset_id'),'offline_verdict':verdict,
        'quality_notes':review.get('quality_notes',''),
        'observer_assessment':observer_assessment,
        'ground_truth':False,'authoritative_for_story':False,
        'current_request_visual_success':False,
        'scope':'Saved clip only; not a new generation, human truth or current-state correspondence proof.'}


def visual_problem(row, renderer_kind='unknown', review_policy='model_gate'):
    if not row.get('valid'):
        return 'generation_failure' if row.get('renderer_status') == 'failed' else 'decision_failure'
    assessment = row.get('visual_assessment', {})
    if assessment.get('status') == 'contradicted':
        return 'visual_mismatch'
    plan = row.get('plan', {})
    control=row.get('renderer',{}).get('control_compilation',{})
    if (plan.get('semantic_action')=='approach' and control.get('candidate_path') and
            control.get('semantic_approach_verified') is False):
        # Image-space zoom is disclosed as a visual proxy, not physical approach.
        return 'visual_uncertain'
    critical = (plan.get('semantic_action') in ('pick_up', 'open', 'search') or
                plan.get('target') in ('storage_room', 'core_area') or
                (plan.get('target') == 'terminal' and
                 plan.get('semantic_action') in ('inspect', 'use', 'interact')))
    if critical and renderer_kind.startswith('wan') and review_policy == 'manual_critical':
        # A fallible VLM saying "supported" is not qualification of this gate.
        return 'manual_review_required'
    if critical and renderer_kind.startswith('wan') and assessment.get('status') != 'supported':
        return 'visual_uncertain'
    if critical and assessment.get('source')=='prebuilt_asset_ai_review':
        return 'visual_uncertain'
    motion = (plan.get('semantic_action') in ('approach','move_to','explore','follow','avoid') or
              (plan.get('semantic_action')=='inspect' and plan.get('target') in ('left','right')))
    if motion and renderer_kind.startswith('wan') and assessment.get('status') in ('uncertain','not_visually_testable'):
        # A moving NPC or an unresolved view change does not verify player/camera motion.
        # Advisory mode records this warning; strict mode retains its existing pending gate.
        return 'visual_uncertain'
    return None


def may_act_while_pending(plan):
    return (plan.semantic_action in ('observe', 'wait') or
            (plan.semantic_action == 'inspect' and plan.target in ('left', 'right', 'room', 'door_a', 'desk')))


def effect_already_done(state, plan):
    """Only idempotent narrative effects, never arbitrary repeated observations."""
    a, t, d = plan.semantic_action, plan.target, plan.dialogue_action
    f = state.story.flags
    return bool(
        (a == 'pick_up' and t == 'red_key' and 'red_key' in state.inventory.items) or
        (a in ('open', 'use') and t == 'door_a' and state.objects['door_a'].state == 'open') or
        (a == 'search' and t in ('desk', 'room', 'red_key') and f.get('found_key')) or
        (t == 'anna' and d in ('help', 'trust') and f.get('helped_anna')) or
        (t == 'anna' and d == 'password' and f.get('password_known')) or
        (t == 'anna' and d == 'distrust' and f.get('distrusted_anna')) or
        (a == 'approach' and t == 'anna' and f.get('met_anna')) or
        (a == 'approach' and t == 'leo' and f.get('met_leo')))
