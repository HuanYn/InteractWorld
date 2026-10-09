"""Opt-in current-RGB UI routing. No rendering or model calls live here."""
from pathlib import Path


LIVE_CURRENT_FRAME_CSS = """
#iw-live-current-frame{position:absolute!important;inset:0;z-index:0;
width:100%!important;height:100%!important;min-height:0!important;border:none!important}
#iw-live-current-frame img{width:100%;height:100%;object-fit:contain}
#iw-live-video{position:relative;z-index:1;background:#07101a}
"""


def start_ui_world(engine, image=None, *, live_current_frame=False, story_lineage_guard=False, configured_image=None):
    """Legacy initialization is unchanged; live never generates a static opening."""
    if not (live_current_frame or story_lineage_guard):
        return engine.start_world(image)
    selected = image if image is not None else configured_image
    if not selected or not Path(selected).is_file():
        raise ValueError('请提供明确的当前 RGB 图片，再开始当前帧交互。')
    return engine.start_from_current_frame(selected)


def require_ui_frame(engine, *, live_current_frame=False, story_lineage_guard=False):
    if engine.frame is not None:
        return
    if live_current_frame or story_lineage_guard:
        raise ValueError('请先从明确的当前 RGB 图片开始；不会自动生成静止开场。')
    engine.start_world()


def live_media_state(engine):
    """The input is an image, never a fabricated/generated video."""
    video = getattr(engine, 'video', None) if engine is not None else None
    frame = getattr(engine, 'frame', None) if engine is not None else None
    resume=getattr(engine,'session_resume',None) if engine is not None else None
    if (isinstance(resume,dict) and resume.get('kind') in ('pinned_story_contract_continuation_v1','pinned_story_oom_continuation_v1')
            and video==resume.get('owned_current_video') and frame==resume.get('owned_current_frame')):
        # The copied parent video is history, never a newly played opening.
        # Display its actual final RGB until a genuinely new child clip exists.
        video=None
    return {'video': video, 'video_visible': bool(video),
            'image': frame if not video else None, 'image_visible': bool(frame and not video)}
