"""Opt-in story input lineage; not a camera/visual/navigation success policy.

Keep the existing story rules, event text and advisory warnings. Only the
engine's owned current RGB may condition a new visual action. No asset, goal,
crop, foreign session, model call or world effect is selected by this module.
"""
from pathlib import Path
import re

from .live_continuation import LiveFrameBinding, rgb_sha256


def reset_story_session(renderer, session_id):
    if not renderer.story_lineage_guard:
        return
    if not isinstance(session_id, str) or not re.fullmatch(r'[a-f0-9]{32}', session_id):
        raise ValueError('Story lineage requires the actual new engine session ID')
    if getattr(renderer, '_story_request_active', False):
        raise ValueError('Cannot reset story lineage during an active render request')
    renderer._story_session_id = session_id
    renderer._story_binding = None
    renderer._story_input_initialized = False
    renderer._story_has_generated = False
    renderer._declared_room_context = None


def establish_story_input(renderer, session_id, owned_frame, session_dir):
    if not renderer.story_lineage_guard:
        raise ValueError('Explicit story input requires the story lineage opt-in')
    if (getattr(renderer, '_story_request_active', False)
            or getattr(renderer, '_story_input_initialized', False)
            or getattr(renderer, '_story_has_generated', False)):
        raise ValueError('Story RGB can only be established in a fresh idle session')
    if getattr(renderer, '_story_session_id', None) != session_id:
        raise ValueError('Story RGB requires an explicitly reset session')
    folder = Path(session_dir).resolve(strict=True)
    frame = Path(owned_frame).resolve(strict=True)
    if folder.name != session_id or not frame.is_file() or not frame.is_relative_to(folder):
        raise ValueError('Story RGB must belong to this engine session')
    binding = LiveFrameBinding(session_id, folder, frame, rgb_sha256(frame), False)
    renderer._story_input_initialized = True
    renderer._story_binding = binding
    return binding


def bind_story_frame(renderer, session_id, frame, *, session_dir=None):
    """Called under Engine.lock; no initial-image fallback exists here."""
    if not renderer.story_lineage_guard:
        return None
    if getattr(renderer, '_story_request_active', False):
        raise ValueError('Cannot replace an in-flight story frame binding')
    if (not isinstance(session_id, str) or not re.fullmatch(r'[a-f0-9]{32}', session_id)
            or getattr(renderer, '_story_session_id', None) != session_id):
        raise ValueError('Story frame belongs to another or uninitialized session')
    if not getattr(renderer, '_story_input_initialized', False) or frame is None or session_dir is None:
        raise ValueError('Story continuation requires its established current RGB; no opening fallback')
    folder = Path(session_dir).resolve(strict=True)
    current = Path(frame).resolve(strict=True)
    if folder.name != session_id or not current.is_file() or not current.is_relative_to(folder):
        raise ValueError('Story current frame must belong to this engine session')
    binding = LiveFrameBinding(session_id, folder, current, rgb_sha256(current), False)
    renderer._story_binding = binding
    return binding


def request_evidence(renderer, init_frame, output_dir, plan, skills):
    binding = getattr(renderer, '_story_binding', None)
    if not isinstance(binding, LiveFrameBinding):
        raise ValueError('Engine must bind its actual current frame before every story render')
    if renderer.npc_framing or renderer.live_current_frame:
        raise ValueError('Story lineage forbids goal framing and does not enable strict live-camera mode')
    current = Path(init_frame).resolve(strict=True) if init_frame is not None else None
    if current is None or current != binding.frame:
        raise ValueError('Story request differs from the bound current frame; no initial fallback')
    if rgb_sha256(current) != binding.frame_rgb_sha256:
        raise ValueError('Story current pixels changed after engine binding')
    out = Path(output_dir).resolve()
    if out == binding.session_dir or not out.is_relative_to(binding.session_dir):
        raise ValueError('Story output must belong to this engine session')
    return dict(mode='story_lineage_guard', session_id=binding.session_id,
                initial=False, actual_previous_last=str(current),
                actual_previous_last_rgb_sha256=binding.frame_rgb_sha256,
                output_dir=str(out), expected_skills=list(skills),
                expected_prompt_key=getattr(renderer,'effective_prompt_for',renderer.prompt_for)(plan), expected_frames=renderer.seconds*16+1,
                source='engine_locked_current_story_frame',
                input_initialized_without_video=not renderer._story_has_generated,
                goal_condition_allowed=False, pre_generated_allowed=False,
                visual_success_verified=False, camera_motion_verified=False,
                world_navigation_verified=False, same_door_position_verified=False)


def verify_result(result, evidence, plan):
    """Reuse the unchanged RGB/artifact/action validator, not its live policy.

    The shared validator checks real input.png/actions.npy and request-owned
    files. Its temporary live evidence key stays on a local metadata copy.
    Story event prompts and advisory decisions are not altered.
    """
    from .live_continuation import verify_result as verify_visual_input_contract
    if not isinstance(result.get('metadata'), dict):
        raise ValueError('Missing actual story renderer metadata')
    proxy = dict(result, metadata=dict(result['metadata']))
    checked = dict(evidence)
    try:
        verify_visual_input_contract(proxy, checked, plan)
    except (ValueError, KeyError, OSError) as exc:
        raise ValueError('Story input lineage rejected: '+str(exc)) from exc
    checked['lineage_is_not_visual_success'] = True
    result['metadata']['story_lineage_evidence'] = checked
    return checked


def verify_retained_dialogue(renderer, binding, result, previous_video):
    """A current image can host text without fabricating an opening video."""
    if not isinstance(binding, LiveFrameBinding):
        raise ValueError('Retained story dialogue requires the current frame binding')
    metadata = result.metadata
    frame = Path(result.last_frame).resolve(strict=True) if result.last_frame else None
    video = Path(result.video_path).resolve(strict=True) if result.video_path else None
    expected_video = Path(previous_video).resolve(strict=True) if previous_video else None
    if (metadata.get('kind') != 'dialogue_only_retained_view'
            or not metadata.get('explicit_no_new_generation')
            or metadata.get('new_video_generated') is not False
            or frame != binding.frame or rgb_sha256(frame) != binding.frame_rgb_sha256
            or video != expected_video
            or (video is not None and not video.is_relative_to(binding.session_dir))):
        raise ValueError('Dialogue may only retain this session actual current media')
    metadata['story_dialogue_evidence'] = dict(
        session_id=binding.session_id, actual_current_frame=str(frame),
        actual_current_rgb_sha256=binding.frame_rgb_sha256, current_frame_unchanged=True,
        new_video_generated=False, model_conditioning_checked=False,
        image_only=video is None, visual_action_performed=False,
        visual_success_verified=False, world_navigation_verified=False)
    renderer._story_binding = None
    return metadata['story_dialogue_evidence']
