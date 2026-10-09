"""Opt-in current-session RGB lineage checks, not visual/motion verification.

The engine binds its actual current frame while holding its existing lock.
No media selection, image crop, future goal, model call or world update here.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re


@dataclass(frozen=True)
class LiveFrameBinding:
    session_id: str
    session_dir: Path
    frame: Path | None
    frame_rgb_sha256: str | None
    initial: bool


def rgb_sha256(path, *, exact=True):
    from PIL import Image,ImageOps
    with Image.open(path) as source:
        image=source.convert('RGB')
        if exact and image.size!=(832,480):
            raise ValueError('Live continuation requires the actual832x480 current frame, not a crop/resize substitute')
        if not exact:
            image=ImageOps.fit(image,(832,480),method=Image.Resampling.LANCZOS)
        return hashlib.sha256(image.tobytes()).hexdigest()


def reset_live_session(renderer,session_id):
    """Explicit new-story lifecycle, only called by Engine.reset under its lock."""
    if not renderer.live_current_frame:return
    if not isinstance(session_id,str) or not re.fullmatch(r'[a-f0-9]{32}',session_id):
        raise ValueError('Live reset requires a new actual engine session ID')
    if getattr(renderer,'_live_request_active',False):
        raise ValueError('Cannot reset a live session during an active render request')
    renderer._live_session_id=session_id
    renderer._live_binding=None
    renderer._live_has_generated=False
    renderer._live_input_initialized=False


def establish_live_input(renderer,session_id,owned_frame,session_dir):
    """Establish exact current RGB for a fresh session, without rendering it."""
    if not renderer.live_current_frame:
        raise ValueError('Explicit current RGB initialization requires live mode')
    if (getattr(renderer,'_live_request_active',False) or
            getattr(renderer,'_live_has_generated',False) or
            getattr(renderer,'_live_input_initialized',False)):
        raise ValueError('Current RGB initialization is only allowed in a fresh idle session')
    if getattr(renderer,'_live_session_id',None)!=session_id:
        raise ValueError('Current RGB initialization requires the explicitly reset engine session')
    folder=Path(session_dir).resolve(strict=True)
    path=Path(owned_frame).resolve(strict=True)
    if folder.name!=session_id or not path.is_file() or not path.is_relative_to(folder):
        raise ValueError('Established RGB must belong to the new engine session')
    digest=rgb_sha256(path)
    renderer._live_input_initialized=True
    renderer._live_binding=LiveFrameBinding(session_id,folder,path,digest,False)
    return renderer._live_binding


def bind_current_frame(renderer,session_id,frame,*,initial=False,session_dir=None):
    """Bind only the engine's canonical frame; call under the engine lock."""
    if not renderer.live_current_frame:
        return None
    if not isinstance(session_id,str) or not re.fullmatch(r'[a-f0-9]{32}',session_id):
        raise ValueError('Live binding requires the actual engine session ID')
    if type(initial) is not bool or session_dir is None:
        raise ValueError('Live binding needs explicit phase and engine session directory')
    folder=Path(session_dir).resolve()
    if folder.name!=session_id:
        raise ValueError('Session directory/identity mismatch')
    owner=getattr(renderer,'_live_session_id',None)
    if owner is not None and owner!=session_id:
        raise ValueError('A live renderer cannot replace another session frame')
    generated=getattr(renderer,'_live_has_generated',False)
    initialized=getattr(renderer,'_live_input_initialized',False)
    if initial and (generated or initialized):
        raise ValueError('Initial image is allowed only before the first live render')
    if not initial and (not (generated or initialized) or frame is None):
        raise ValueError('A continuation must use the established current session last frame')
    path=Path(frame).resolve(strict=True) if frame is not None else None
    if path is not None and not path.is_file():
        raise ValueError('Current frame is not a file')
    if not initial and not path.is_relative_to(folder):
        raise ValueError('Current frame must belong to this engine session, not another source')
    digest=rgb_sha256(path,exact=not initial) if path is not None else None
    binding=LiveFrameBinding(session_id,folder,path,digest,initial)
    renderer._live_session_id=session_id
    renderer._live_binding=binding
    return binding


def compile_live_skills(plan,requested,*,full_window_pan=False):
    """Directional look intent owns realJ/L keys; noI/K or image-space zoom."""
    if plan is not None and plan.semantic_action in ('inspect','observe'):
        if plan.target=='right':return ['pan_right'] if full_window_pan else ['pan_right','static']
        if plan.target=='left':return ['pan_left'] if full_window_pan else ['pan_left','static']
    return [getattr(value,'value',value) for value in requested]


def request_evidence(renderer,init_frame,output_dir,plan,skills):
    binding=getattr(renderer,'_live_binding',None)
    if not isinstance(binding,LiveFrameBinding):
        raise ValueError('Engine must bind its current frame before every live request')
    if renderer.npc_framing:
        raise ValueError('Live current-frame mode forbids NPC framing goals')
    supplied=Path(init_frame).resolve(strict=True) if init_frame is not None else None
    if supplied!=binding.frame:
        raise ValueError('Request frame differs from the engine current-frame binding')
    if not binding.initial and supplied is None:
        raise ValueError('Missing current frame; initial-asset fallback is forbidden')
    source=supplied if supplied is not None else Path(renderer.initial_image).resolve(strict=True)
    digest=rgb_sha256(source,exact=not binding.initial)
    if binding.frame_rgb_sha256 is not None and digest!=binding.frame_rgb_sha256:
        raise ValueError('Current-frame pixels changed after engine binding')
    out=Path(output_dir).resolve()
    if not out.is_relative_to(binding.session_dir) or out==binding.session_dir:
        raise ValueError('Live output must stay inside this engine session')
    return dict(mode='live_current_frame',session_id=binding.session_id,
        initial=binding.initial,actual_previous_last=str(source),
        actual_previous_last_rgb_sha256=digest,output_dir=str(out),
        expected_skills=list(skills),expected_prompt_key=renderer.live_prompt_key,
        expected_frames=renderer.seconds*16+1,full_window_pan=renderer.live_full_window_pan,
        input_initialized_without_video=bool(getattr(renderer,'_live_input_initialized',False)
                                             and not getattr(renderer,'_live_has_generated',False)),
        source='engine_locked_current_frame_binding',
        pre_generated_allowed=False,goal_condition_allowed=False,
        current_request_visual_success=False,visual_success_verified=False,
        camera_motion_verified=False,world_navigation_verified=False)


def verify_result(result,evidence,plan):
    """Check returned input/actions/artifact ownership before logical commit.

    RGB equality proves conditioning lineage, not artifact quality or motion.
    Existing video decoding/observer remain the renderer/engine responsibility.
    """
    import numpy as np
    from .agent import SkillSelector
    metadata=result.get('metadata')
    if not isinstance(metadata,dict) or metadata.get('kind')!='wan_laboratory':
        raise ValueError('Live mode requires the actual laboratory Wan response')
    if not evidence.get('expected_prompt_key') or metadata.get('prompt_key')!=evidence['expected_prompt_key']:
        raise ValueError('Returned live prompt key differs from the explicitly configured scene cache key')
    forbidden=('pre_generated','explicit_goal_input','authored_state_insert','recovered_existing_initial',
        'existing_frame_reused','existing_video_reused','framing_constraint','framing_condition')
    if any(metadata.get(key) for key in forbidden):
        raise ValueError('A prebuilt, retained, recovered or goal-conditioned response is not a live visual action')
    out=Path(evidence['output_dir'])
    video=Path(result['video_path']).resolve(strict=True)
    last=Path(result['last_frame']).resolve(strict=True)
    inp=Path(metadata['conditioning_image']).resolve(strict=True)
    if (video!=out/'video.mp4' or last!=out/'last.png' or inp!=out/'input.png'
            or not video.is_file() or not video.stat().st_size):
        raise ValueError('Returned artifacts do not belong to this live request output')
    if any((out/name).exists() for name in ('supplied-framing-goal.png','endpoint-trace.json','goal.png')):
        raise ValueError('Live action must not include endpoint/goal conditioning artifacts')
    if rgb_sha256(inp)!=evidence['actual_previous_last_rgb_sha256']:
        raise ValueError('Actual model input differs from the engine-bound current RGB frame')
    if metadata.get('checkpoint')!='R005-step1040' or metadata.get('fps')!=16:
        raise ValueError('Unexpected live renderer identity or timing')
    frames=metadata.get('frames')
    if frames not in (49,97) or frames!=evidence['expected_frames']:
        raise ValueError('Unexpected bounded live frame count')
    actions=np.load(out/'actions.npy',allow_pickle=False)
    if actions.shape!=(frames-1,8) or not np.isfinite(actions).all() or not np.isin(actions,[0,1]).all():
        raise ValueError('Invalid actual live action tensor')
    expected=np.zeros_like(actions);cursor=0
    for segment in SkillSelector().action_segments(evidence['expected_skills'],frames=frames-1):
        end=cursor+segment['frames']
        for key in segment['keys']:expected[cursor:end,['W','A','S','D','I','J','K','L'].index(key)]=1
        cursor=end
    if not np.array_equal(actions,expected):raise ValueError('Actual action schedule differs from live request')
    directional=bool(plan is not None and plan.semantic_action in ('inspect','observe') and plan.target in ('left','right'))
    if directional and not np.any(actions):raise ValueError('Directional observation cannot be static all-zero actions')
    evidence.update(conditioning_rgb_match=True,actual_action_tensor_shape=list(actions.shape),
        actual_nonzero_action_values=int(np.count_nonzero(actions)),
        directional_action_requested=directional,goal_condition_used=False,pre_generated_used=False,
        lineage_check_passed=True,lineage_is_not_visual_success=True)
    metadata['live_current_frame_evidence']=dict(evidence)
    return evidence
