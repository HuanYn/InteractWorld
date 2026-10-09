"""Opt-in presentation policy for text-only NPC dialogue, not video synthesis.

The interaction engine must still validate and commit the rule action itself.
Reusing an existing view does not validate NPC speech, physical interaction,
world memory, model quality, or a new video's duration.
"""
from pathlib import Path

from .models import CameraSkill, Plan
from .renderers import RenderResult


POLICY = 'npc_rule_text_retained_view_v1'


def _existing_media(path):
    """Check only a nonempty local file; do not decode, copy, hash or rewrite it."""
    if not isinstance(path, (str, Path)) or not str(path).strip():
        return None
    try:
        value = Path(path).resolve(strict=True)
        if not value.is_file() or value.stat().st_size <= 0:
            return None
    except (OSError, RuntimeError, ValueError):
        return None
    return str(value)


def retained_dialogue_result(plan: Plan, frame, video, enabled=False, *, allow_image_only=False) -> RenderResult | None:
    """Return the current media only for explicitly enabled, static NPC text.

    ``None`` means this policy does not apply; it is not approval to generate or
    execute the action. No media is created, and no rule/world state is touched.
    Source encoding validity remains the caller's existing renderer contract.
    """
    if enabled is not True or not isinstance(plan, Plan):
        return None
    if plan.needs_clarification or plan.expected_valid is False:
        return None
    if plan.target not in ('anna', 'leo'):
        return None
    textual = (plan.semantic_action in ('talk', 'ask') or
               (plan.semantic_action == 'interact' and plan.target == 'anna'
                and plan.dialogue_action in ('help', 'trust', 'distrust')))
    if not textual:
        return None
    skills = plan.camera_skills
    if not isinstance(skills, list) or any(
            skill not in (CameraSkill.static, 'static') for skill in skills):
        return None
    existing_frame = _existing_media(frame)
    existing_video = _existing_media(video)
    if existing_frame is None or (existing_video is None and not (allow_image_only is True and video is None)):
        return None
    assessment = {
        'status': 'not_visually_testable',
        'source': 'dialogue_presentation_policy',
        'evidence': '本轮仅显示规则 NPC 文字对话，保留已有画面；未生成新视频。',
        'ground_truth': False,
        'authoritative_for_story': False,
        'current_request_visual_success': False,
        'visual_success_verified': False,
        'scope': 'Text presentation only, not VLM verification, speech or lip-sync.'
    }
    metadata = {
        'kind': 'dialogue_only_retained_view',
        'policy': POLICY,
        'policy_opt_in': True,
        'explicit_no_new_generation': True,
        'new_video_generated': False,
        'existing_frame_reused': True,
        'existing_video_reused': existing_video is not None,
        'image_only_retained': existing_video is None,
        'duration_added_seconds': 0,
        'requested_semantic_action': plan.semantic_action,
        'requested_target': plan.target,
        'requested_dialogue_action': plan.dialogue_action,
        'requested_camera_skills': [getattr(skill, 'value', skill) for skill in skills],
        'provenance': {
            'source': 'existing_current_session_media',
            'source_frame': existing_frame,
            'source_video': existing_video,
            'source_files_modified': False,
            'media_validation': 'Existing nonempty local files only; no decode verification.',
            'new_model_call': False,
            'state_or_rule_effects_applied': False
        },
        'audio_generated': False,
        'lip_sync_verified': False,
        'visual_success_verified': False,
        'video_model_quality_improvement_verified': False,
        'persistent_memory_verified': False,
        'visual_assessment': assessment,
        'notice': '纯文字对话保留已有镜头，不新增时长，不代表视频模型画质或物体交互改善。'
    }
    return RenderResult(video_path=existing_video, last_frame=existing_frame,
                        metadata=metadata)
