"""Isolated NumPy cubemap projection and an explicit synthetic yaw controller.

No files, model, game controller, loader or renderer are touched. A supplied
calibration declaration is checked numerically; that is not verification that
six real source images were captured at the same time/place. One panorama can
provide rotation-only views, not dynamic-world or object-interaction truth.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


FACE_ORDER = ("front", "right", "back", "left", "top", "bottom")
KEY_ORDER = ("W", "A", "S", "D", "I", "J", "K", "L")
# Columns are face forward/right/down, expressed in the front-camera NED basis.
_FACE_ROTATIONS = (
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
    ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
)


def cube_geometry_contract() -> dict:
    """Fresh JSON-compatible declaration, NOT an inferred source calibration."""
    return {
        "kind": "colocated_90deg_cube_v1",
        "coordinate_system": "NED_x_forward_y_right_z_down",
        "face_order": list(FACE_ORDER),
        "face_size": [640, 640],
        "face_hfov_degrees": 90.0,
        "face_positions_m": {name: [0.0, 0.0, 0.0] for name in FACE_ORDER},
        "face_camera_to_front_ned": {
            name: [list(row) for row in matrix]
            for name, matrix in zip(FACE_ORDER, _FACE_ROTATIONS)
        },
        "declaration_not_physical_verification": True,
    }


def _finite_scalar(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _size(value, name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be [width, height]")
    if any(type(v) is not int or v <= 0 for v in value):
        raise ValueError(f"{name} must contain positive integers")
    return value[0], value[1]


def _validate_cube(faces: Mapping[str, np.ndarray], metadata: dict) -> None:
    if not isinstance(faces, Mapping) or set(faces) != set(FACE_ORDER):
        raise ValueError("all six named cube faces are required, without extra faces")
    for name in FACE_ORDER:
        image = faces[name]
        if not isinstance(image, np.ndarray) or image.shape != (640, 640, 3) or image.dtype != np.uint8:
            raise ValueError(f"{name} must be 640x640x3 uint8 RGB")
    expected = cube_geometry_contract()
    if not isinstance(metadata, dict):
        raise ValueError("explicit cube geometry metadata is required")
    for key in ("kind", "coordinate_system", "face_order"):
        if metadata.get(key) != expected[key]:
            raise ValueError(f"unsupported cube metadata: {key}")
    if _size(metadata.get("face_size"), "face_size") != (640, 640):
        raise ValueError("source cube faces must be 640 square")
    if _finite_scalar(metadata.get("face_hfov_degrees"), "face_hfov_degrees") != 90.0:
        raise ValueError("cube face hFOV must be exactly 90 degrees")
    for key, shape in (("face_positions_m", (3,)), ("face_camera_to_front_ned", (3, 3))):
        values = metadata.get(key)
        if not isinstance(values, dict) or set(values) != set(FACE_ORDER):
            raise ValueError(f"all six {key} declarations are required")
        for name in FACE_ORDER:
            try:
                actual = np.asarray(values[name], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid {key}: {name}") from exc
            if actual.shape != shape or not np.isfinite(actual).all():
                raise ValueError(f"invalid {key}: {name}")
            if not np.allclose(actual, expected[key][name], atol=1e-12, rtol=0):
                raise ValueError(f"non-colocated or differently oriented cube: {key}/{name}")


@dataclass(frozen=True)
class VirtualView:
    image: np.ndarray
    chosen_faces: np.ndarray  # HxW uint8 indices into metadata['face_order'].
    metadata: dict


def project_cube_yaw(
    faces: Mapping[str, np.ndarray],
    yaw_degrees: float,
    *,
    metadata: dict,
    output_size: tuple[int, int] = (832, 480),
    output_hfov_degrees: float = 90.0,
) -> VirtualView:
    """Project rotation-only views; positive NED yaw looks to the right.

    Intrinsics use continuous pixel edges; ray centres are (u+.5, v+.5).
    Source image indices are fx*y/x+cx-.5 and fy*z/x+cy-.5. Select the
    greatest positive face-forward dot product, then bilinearly sample ONLY
    that face. At a seam clamp to its edge pixel: no cross-face mixing/inpaint.
    No assumption of seamless texture, measured calibration or scene validity.
    """
    _validate_cube(faces, metadata)
    width, height = _size(output_size, "output_size")
    yaw = math.radians(_finite_scalar(yaw_degrees, "yaw_degrees"))
    hfov = _finite_scalar(output_hfov_degrees, "output_hfov_degrees")
    if not 0 < hfov < 180:
        raise ValueError("output hFOV must lie strictly between 0 and 180 degrees")
    focal = width / (2 * math.tan(math.radians(hfov) / 2))
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    local_rays = np.stack((np.ones_like(u), (u + .5 - width / 2) / focal,
                           (v + .5 - height / 2) / focal), axis=-1)
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=np.float64)
    rays = local_rays @ rotation.T
    matrices = np.asarray(_FACE_ROTATIONS, dtype=np.float64)
    forward_dots = rays @ matrices[:, :, 0].T
    chosen = np.argmax(forward_dots, axis=-1).astype(np.uint8)
    if np.any(np.max(forward_dots, axis=-1) <= 0):
        raise ValueError("cube does not cover all output rays")
    output = np.empty((height, width, 3), dtype=np.uint8)
    counts = {}
    clamped = 0
    for face_id, name in enumerate(FACE_ORDER):
        mask = chosen == face_id
        count = int(np.count_nonzero(mask))
        counts[name] = count
        if not count:
            continue
        face_rays = rays[mask] @ matrices[face_id]
        x = 320.0 * face_rays[:, 1] / face_rays[:, 0] + 320.0 - .5
        y = 320.0 * face_rays[:, 2] / face_rays[:, 0] + 320.0 - .5
        clamped += int(np.count_nonzero((x < 0) | (x > 639) | (y < 0) | (y > 639)))
        x, y = np.clip(x, 0, 639), np.clip(y, 0, 639)
        x0, y0 = np.floor(x).astype(np.intp), np.floor(y).astype(np.intp)
        x1, y1 = np.minimum(x0 + 1, 639), np.minimum(y0 + 1, 639)
        wx, wy = (x - x0)[:, None], (y - y0)[:, None]
        image = faces[name]
        sampled = ((1-wy)*((1-wx)*image[y0, x0] + wx*image[y0, x1])
                   + wy*((1-wx)*image[y1, x0] + wx*image[y1, x1]))
        output[mask] = np.rint(sampled).clip(0, 255).astype(np.uint8)
    return VirtualView(output, chosen, {
        "kind": "synthetic_rotation_only_cube_projection_v1",
        "face_order": list(FACE_ORDER), "chosen_face_pixel_counts": counts,
        "covered_pixels": width*height, "coverage_fraction": 1.0,
        "edge_clamped_pixels": clamped, "sampling": "bilinear_selected_face_edge_clamp",
        "face_tie_break": "first_in_face_order", "cross_face_blending": False,
        "inpainting": False, "translation_m": [0.0, 0.0, 0.0],
        "yaw_degrees_positive_right": float(yaw_degrees),
        "output_size": [width, height], "output_hfov_degrees": hfov,
        "output_intrinsics_pixel_edges": [focal, focal, width/2, height/2],
        "source_intrinsics_pixel_edges": [320.0, 320.0, 320.0, 320.0],
        "pixel_coordinate_convention": "continuous_edges_centres_at_index_plus_half",
        "declared_geometry_checked": True, "physical_source_geometry_verified": False,
        "model_generated": False, "recorded_game_actions": False,
        "training_qualified": False,
    })


def virtual_yaw_sequence(key: str, *, initial_yaw_degrees: float = 0.0) -> dict:
    """49 synthetic poses / 48 binary intervals at16Hz, fixed10deg/s.

    J explicitly commands left, L right for THIS virtual controller; no keys
    are inferred from source poses. Frame0 is condition-only; action row i
    moves pose i to i+1. The single-panorama product is not training-qualified.
    """
    if key not in ("J", "L"):
        raise ValueError("the virtual yaw controller supports only J or L")
    initial = _finite_scalar(initial_yaw_degrees, "initial_yaw_degrees")
    rate = 10.0 if key == "L" else -10.0
    times = [i/16 for i in range(49)]
    yaws = [initial + rate*t for t in times]
    poses = [[0.0, 0.0, 0.0, 0.0, 0.0, math.sin(math.radians(y)/2),
              math.cos(math.radians(y)/2)] for y in yaws]
    action = [int(name == key) for name in KEY_ORDER]
    return {
        "kind": "synthetic_virtual_yaw_controller_v1", "fps": 16,
        "key_order": list(KEY_ORDER), "commanded_key": key,
        "angular_rate_degrees_per_second": rate,
        "timestamps_seconds": times, "yaw_degrees": yaws,
        "poses_position_m_quaternion_xyzw": poses,
        "binary_actions": [action.copy() for _ in range(48)],
        "frame_count": 49, "action_interval_count": 48,
        "span_seconds": 3.0, "playback_seconds": 49/16,
        "first_rgb_condition_only": True, "first_frame_has_action": False,
        "action_alignment": "row_i_moves_pose_i_to_i_plus_1",
        "translation_m": [0.0, 0.0, 0.0],
        "action_provenance": "explicit_synthetic_virtual_controller_not_original_game_keys",
        "source_pose_keys_inferred": False, "recorded_game_actions": False,
        "dynamic_world_truth": False, "training_qualified": False,
    }
