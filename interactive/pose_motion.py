"""CPU-only camera motion representation. Never infer recorded keyboard actions.

Input poses are camera-to-world, position in metres and quaternion xyzw, in
the documented TartanGround NED camera basis (forward, right, down). This is
an intermediate data product, NOT the R005 binary action-conditioning format.
No model imports, interpolation, RGB changes or runtime registration.
"""
from __future__ import annotations

import math
from typing import Sequence


def _finite_vector(values: Sequence[float], size: int) -> list[float]:
    if len(values) != size or any(isinstance(v, bool) for v in values):
        raise ValueError(f"expected {size} finite real numbers")
    result = [float(v) for v in values]
    if not all(math.isfinite(v) for v in result):
        raise ValueError("nonfinite motion data")
    return result


def _quaternion(values: Sequence[float]) -> list[float]:
    q = _finite_vector(values, 4)
    norm = math.sqrt(sum(v*v for v in q))
    if abs(norm-1) > 1e-3:
        raise ValueError("quaternion is not approximately unit length")
    return [v/norm for v in q]


def _multiply(a, b):
    x, y, z, w = a
    X, Y, Z, W = b
    return [w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X,
            w*Z+x*Y-y*X+z*W, w*W-x*X-y*Y-z*Z]


def rotation_matrix(q):
    x, y, z, w = _quaternion(q)
    return [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]


def _body_vector(rotation, vector):
    return [sum(rotation[k][j]*vector[k] for k in range(3)) for j in range(3)]


def _relative_rotation(a, b):
    # q_i^-1 * q_j: shortest rotation in the start-camera basis. q and -q
    # represent the same orientation. Canonicalise the pi case as well.
    relative = _quaternion(_multiply([-a[0], -a[1], -a[2], a[3]], b))
    sign = relative[3]
    if abs(sign) < 1e-12:
        sign = next((v for v in relative[:3] if abs(v) > 1e-12), 1.0)
    if sign < 0:
        relative = [-v for v in relative]
    length = math.sqrt(sum(v*v for v in relative[:3]))
    if length < 1e-12:
        return relative, [2*v for v in relative[:3]]
    angle = 2*math.atan2(length, max(0.0, relative[3]))
    return relative, [angle*v/length for v in relative[:3]]


def motion_intervals(poses, frame_indices, timestamps_seconds):
    """Retain all measured adjacent SE(3) deltas; frame0 has no action row.

    Linear velocity is interval-average displacement expressed in the starting
    camera basis, not instantaneous body velocity. Rotvec/dt is an equivalent
    average angular displacement, not a sensor gyro or Euler-angle derivative.
    """
    if len(poses) < 2 or not (len(poses) == len(frame_indices) == len(timestamps_seconds)):
        raise ValueError("need matching pose/frame/time arrays with at least two poses")
    if any(type(v) is not int or v < 0 for v in frame_indices):
        raise ValueError("source frame indices must be nonnegative integers")
    times = _finite_vector(timestamps_seconds, len(poses))
    if times[0] < 0:
        raise ValueError("negative source time")
    parsed = []
    for pose in poses:
        row = _finite_vector(pose, 7)
        parsed.append((row[:3], _quaternion(row[3:])))
    rows = []
    for i, ((p, q), (p_next, q_next)) in enumerate(zip(parsed, parsed[1:])):
        dt = times[i+1]-times[i]
        if dt <= 0 or frame_indices[i+1] != frame_indices[i]+1:
            raise ValueError("missing/reordered frames or nonincreasing source time")
        displacement = [b-a for a, b in zip(p, p_next)]
        local = _body_vector(rotation_matrix(q), displacement)
        relative, rotvec = _relative_rotation(q, q_next)
        rows.append({
            "frame_from": frame_indices[i], "frame_to": frame_indices[i+1],
            "time_from_seconds": times[i], "time_to_seconds": times[i+1],
            "dt_seconds": dt, "translation_start_camera_m": local,
            "relative_quaternion_xyzw": relative,
            "rotation_vector_start_camera_rad": rotvec,
            "interval_velocity_start_camera_m_s": [v/dt for v in local],
            "equivalent_angular_velocity_rad_s": [v/dt for v in rotvec],
            "translation_speed_m_s": math.sqrt(sum(v*v for v in displacement))/dt,
        })
    return rows


def r005_compatibility(poses, timestamps_seconds, *, source_fps: int):
    """Fail closed: geometry alone cannot invert an uncalibrated game controller."""
    if type(source_fps) is not int or source_fps <= 0:
        raise ValueError("source_fps must be a positive integer")
    n = len(poses)
    if n < 2 or len(timestamps_seconds) != n:
        raise ValueError("invalid source samples")
    times = _finite_vector(timestamps_seconds, n)
    if any(abs((b-a)-1/source_fps) > 1e-7 for a,b in zip(times,times[1:])):
        raise ValueError("timestamps do not agree with declared native fps")
    span = times[-1]-times[0]
    # Arithmetic illustration only. Do NOT materialise a duplicated-frame video
    # or pretend nearest-neighbour output samples are measured at 16 Hz.
    output_count = round(span*16)+1
    offsets = [math.floor(i*source_fps/16+0.5) for i in range(output_count)]
    return {
        "canonical_key_order": ["W","A","S","D","I","J","K","L"],
        "binary_action_required": True, "direct_r005_training_qualified": False,
        "eight_key_labels_created": False, "recorded_controller_available": False,
        "reason": "pose does not identify binary keys or original game controller dynamics",
        "native_source_fps": source_fps, "native_frames": n,
        "native_span_seconds": span, "native_playback_seconds": n/source_fps,
        "relabel_same_frames_as_16fps_span_seconds": (n-1)/16,
        "relabel_same_frames_speed_multiplier": 16/source_fps,
        "hypothetical_16fps_nearest_neighbour_frames": output_count,
        "hypothetical_duplicate_samples": output_count-len(set(offsets)),
        "resampling_performed": False, "first_frame_condition_only": True,
        "motion_interval_count": n-1,
        "future_intervals_divisible_by_four": (n-1)%4 == 0,
        "packing_shape_compatible_is_not_semantic_compatibility": True,
    }


def fit_projection(source_size, target_size, intrinsics):
    """Describe ImageOps.fit centre crop and K' for caller-supplied pinhole K.

    Coordinates use continuous pixel edges, not OpenCV integer pixel centres.
    With integer-centre intrinsics a Pillow resize also needs the half-pixel
    term (scale-1)/2; callers must not mix the conventions. This helper is a
    crop/FOV diagnostic, not calibration. It does not resize RGB; basis is NED.
    """
    sw, sh = _finite_vector(source_size, 2)
    tw, th = _finite_vector(target_size, 2)
    if min(sw, sh, tw, th) <= 0:
        raise ValueError("invalid image sizes")
    fx, fy, cx, cy = _finite_vector(intrinsics, 4)
    if fx <= 0 or fy <= 0:
        raise ValueError("invalid focal length")
    scale = max(tw/sw, th/sh)
    crop_w, crop_h = tw/scale, th/scale
    left, top = (sw-crop_w)/2, (sh-crop_h)/2
    return {
        "source_size": [sw, sh], "target_size": [tw, th],
        "source_crop_box": [left, top, left+crop_w, top+crop_h],
        "uniform_scale": scale,
        "intrinsics_fx_fy_cx_cy": [fx*scale, fy*scale, (cx-left)*scale, (cy-top)*scale],
        "assumes_centered_fit": True, "rgb_transform_performed": False,
    }


def require_recorded_r005_actions(motion_record):
    """An explicit guard for future loaders; not registered in the live renderer."""
    if motion_record.get("kind") == "camera_pose_motion_v1":
        raise ValueError("continuous camera poses are not recorded R005 binary controls")
    raise ValueError("this helper cannot certify recorded controls; use the canonical action schema")
