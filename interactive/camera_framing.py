"""Input-only 2D framing constraint; never a 3D navigation controller."""
import hashlib
from PIL import Image

MODE = 'current_frame_zoom_1p08'


def current_frame_goal(image, mode):
    if mode != MODE or image.mode != 'RGB' or image.size != (832, 480):
        raise ValueError('Only fixed 832x480 RGB, 1.08x framing is supported')
    width, height = image.size
    crop_width, crop_height = round(width / 1.08), round(height / 1.08)
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    box = (left, top, left + crop_width, top + crop_height)
    target = image.crop(box).resize(image.size, Image.Resampling.LANCZOS)
    return target, {
        'mode': MODE, 'controlled_entity': 'camera_framing_2d',
        'crop_box': list(box), 'nominal_zoom': 1.08,
        'source_rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
        'goal_rgb_sha256': hashlib.sha256(target.tobytes()).hexdigest(),
        'goal_source': 'deterministic crop of current RGB only',
        'goal_is_input': True, 'future_image_used': False,
        'goal_encoding': 'last temporal latent of nine repeated goal RGB frames',
        'final_rgb_pasted': False, 'navigation_3d_verified': False,
        'player_action_control_verified': False, 'visual_success_verified': False,
    }
