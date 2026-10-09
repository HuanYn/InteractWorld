"""Isolated CPU reader for the saved virtual49 diagnostic, not a training cache.

The legacy reader decodes lossy H.264 previews. The separate source-cube reader
performs NEW deterministic CPU processing of the acquired six PNGs in memory;
it does not recover missing old RGB or change old previews/receipts. Neither
path loads a model, writes features, assigns splits or qualifies training.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image

from interactive.pose_motion import rotation_matrix
from interactive.virtual_camera import (
    FACE_ORDER, cube_geometry_contract, project_cube_yaw, virtual_yaw_sequence,
)
from training.models.action_adapter import pack_canonical_actions


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Virtual49Sample:
    rgb: torch.Tensor                 # CPU uint8 [C=3,F=49,H=480,W=832]
    actions: torch.Tensor             # CPU float32 [48,8], no frame-0 action
    packed_actions: torch.Tensor      # CPU float32 [12,32], key-major
    metadata: dict

    def vae_input(self) -> torch.Tensor:
        """The existing cache encoder's BCFHW [-1,1] input; no encoding here."""
        return self.rgb.unsqueeze(0).float().div(127.5).sub(1)


def _decode49(path: Path) -> torch.Tensor:
    decoder = imageio_ffmpeg.read_frames(
        str(path), pix_fmt="rgb24", input_params=["-threads", "1"],
        output_params=["-threads", "1", "-vsync", "0"],
    )
    try:
        info = next(decoder)
        if tuple(info["size"]) != (832, 480) or abs(info["fps"] - 16) > 1e-6:
            raise ValueError("virtual MP4 must be native 832x480 at 16 Hz; no resizing/resampling")
        frames = []
        for raw in decoder:
            if len(frames) >= 49 or len(raw) != 832 * 480 * 3:
                raise ValueError("virtual MP4 has extra or malformed frames")
            frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(480, 832, 3))
        if len(frames) != 49:
            raise ValueError(f"virtual MP4 must decode exactly 49 frames, got {len(frames)}")
        return torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).contiguous()
    finally:
        decoder.close()


def load_virtual49(directory: str | Path, key: str, *, source_receipt: str | Path) -> Virtual49Sample:
    """Read the actual saved result/controller/source binding and full MP4.

    A diagnostic may be inspected while unqualified. ``require_profile_ready``
    separately fails closed for optimization use. Original remote face paths
    remain provenance; this reader does not fetch them or regenerate views.
    """
    if key not in ("J", "L"):
        raise ValueError("only the frozen virtual J/L arms are supported")
    root, receipt_path = Path(directory), Path(source_receipt)
    paths = {"result": root / "result.json", "source_binding": root / "source-binding.json",
             "controller": root / f"{key}-virtual-controller.json", "source_receipt": receipt_path,
             "video": root / f"{key}-virtual-yaw-native16fps.mp4"}
    docs = {name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in paths.items() if name != "video"}
    result, binding, controller, receipt = (docs[k] for k in
        ("result", "source_binding", "controller", "source_receipt"))
    if not result.get("status", "").startswith("CPU_virtual_pair_completed"):
        raise ValueError("completed virtual worker result required")
    if result.get("source_receipt_sha256") != _sha(receipt_path):
        raise ValueError("source receipt SHA differs from actual worker binding")
    episode = result["source_episode"]
    if receipt.get("source_episode") != episode or receipt.get("source_frame_index") != result["source_frame"]:
        raise ValueError("source episode/cube time mismatch")
    if receipt.get("revision") != "388faf9c800568cfc6828fa47e063f8369397eb3":
        raise ValueError("unapproved upstream revision")
    expected_faces = {row["face"]: row["sha256"] for row in receipt["rows"]}
    faces = binding.get("bindings", [])
    if len(faces) != 6 or set(expected_faces) != set(FACE_ORDER) or {
        row["face"]: row["sha256"] for row in faces
    } != expected_faces:
        raise ValueError("six source face bindings differ from acquisition receipt")
    expected_controller = virtual_yaw_sequence(key)
    for field, value in expected_controller.items():
        if controller.get(field) != value:
            raise ValueError(f"virtual controller differs at {field}")
    actions = torch.tensor(controller["binary_actions"], dtype=torch.float32)
    packed = pack_canonical_actions(actions)[0]
    if tuple(actions.shape) != (48, 8) or tuple(packed.shape) != (12, 32) or (
        controller.get("packed_12x32_key_major") != packed.tolist()
    ):
        raise ValueError("future48/key-major32 contract mismatch")
    arms = [arm for arm in result["arms"] if arm.get("key") == key]
    if len(arms) != 1:
        raise ValueError("exactly one arm receipt required")
    arm = arms[0]
    if (arm.get("frame_count"), arm.get("fps"), arm.get("action_rows"), arm.get("packed_shape")) != (49, 16, 48, [12, 32]):
        raise ValueError("worker frame/action receipt mismatch")
    if _sha(paths["video"]) != arm["preview_sha256"]:
        raise ValueError("actual MP4 SHA differs from worker result")
    rgb = _decode49(paths["video"])
    decoded_first = rgb[:, 0].permute(1, 2, 0).contiguous().numpy().tobytes()
    environment, camera, trajectory = episode.split("/")
    return Virtual49Sample(rgb, actions, packed, {
        "key": key, "source_episode": episode, "environment": environment,
        "trajectory": trajectory, "camera": camera, "source_frame": result["source_frame"],
        "group_id": f"{episode}/cube-{result['source_frame']:06d}",
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
        "sha256": {name: _sha(path) for name, path in paths.items()},
        "decoded_first_RGB_sha256": hashlib.sha256(decoded_first).hexdigest(),
        "preencode_first_RGB_sha256": arm["first_RGB_sha256"],
        "decoded_first_equals_preencode": hashlib.sha256(decoded_first).hexdigest() == arm["first_RGB_sha256"],
        "rgb_source": "actual_H264_CRF18_yuv420p_MP4_decode_lossy_not_full_PNG_archive",
        "frame_timestamps_seconds": [i / 16 for i in range(49)],
        "action_alignment": "row_i_controls_RGB_i_to_RGB_i_plus_1",
        "original_game_keys": False, "split": "unassigned",
        "training_qualified": False, "model_generated": False,
        "latent_shape_after_existing_encoder": [13, 48, 30, 52],
        "latents_created": False,
    })


def paired_start_check(samples: Sequence[Virtual49Sample]) -> dict:
    """Check decoded RGB starts, never replace them with zero832/source PNG."""
    hashes = [sample.metadata["decoded_first_RGB_sha256"] for sample in samples]
    same_group = len({sample.metadata["group_id"] for sample in samples}) == 1
    matched = len(samples) == 2 and same_group and len(set(hashes)) == 1
    return {"decoded_first_matched": matched, "same_source_group": same_group,
            "decoded_first_RGB_sha256": hashes, "training_qualified": False,
            "encoding_confound": not matched}


def load_source_cube_virtual49(
    source_receipt: str | Path, key: str, *, expected_receipt_sha256: str,
    expected_episode: str, face_paths: Mapping[str, str | Path] | None = None,
) -> Virtual49Sample:
    """New CPU-only raw49 from acquired PNGs, never old MP4 reconstruction.

    The caller must pin the exact acquisition receipt/episode. Optional path
    relocation requires the SAME source PNG SHA, not a substitute image. This
    intentionally remains inspection-only even for independently reviewed data.
    """
    receipt_path = Path(source_receipt)
    if _sha(receipt_path) != expected_receipt_sha256:
        raise ValueError("pinned source acquisition receipt SHA mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") not in (
        "six_native_source_faces_acquired_not_projection_or_training_qualified",
        "six_native_source_faces_acquired_pending_AI_review_not_training_qualified",
    ):
        raise ValueError("completed six-face source acquisition required")
    if (receipt.get("source_episode") != expected_episode
        or receipt.get("source_frame_index") != 0
        or receipt.get("source_timestamp_seconds") != 0
        or receipt.get("native_fps") != 10
        or receipt.get("revision") != "388faf9c800568cfc6828fa47e063f8369397eb3"):
        raise ValueError("fixed source episode/frame0/revision/time contract mismatch")
    rows = receipt.get("rows", [])
    if len(rows) != 6 or {row["face"] for row in rows} != set(FACE_ORDER):
        raise ValueError("exactly six unique acquired source faces required")
    if face_paths is not None and set(face_paths) != set(FACE_ORDER):
        raise ValueError("relocated source paths must provide all six exact faces")
    contract = cube_geometry_contract()
    # The two frozen acquisition workers serialized poses in different named
    # fields. Select by exact completed schema/status, not a silent fallback.
    poses = (receipt["six_camera_geometry"]
             if receipt["status"] == "six_native_source_faces_acquired_pending_AI_review_not_training_qualified"
             else receipt["original_pose_sources"])
    front_pose = np.asarray(poses["front"]["frame0_camera_to_world_xyz_xyzw"], dtype=float)
    if front_pose.shape != (7,) or not np.isfinite(front_pose).all():
        raise ValueError("invalid source frame0 pose")
    front_rotation = np.asarray(rotation_matrix(front_pose[3:]))
    faces, bindings = {}, []
    for name in FACE_ORDER:
        row = next(row for row in rows if row["face"] == name)
        if row.get("frame_index", 0) != 0 or row.get("timestamp_seconds", 0) != 0:
            raise ValueError("source faces are not synchronized frame0")
        path = Path(face_paths[name] if face_paths else row.get("original_path", row.get("local_path", "")))
        if not path.is_file() or path.stat().st_size > 1_000_000 or _sha(path) != row["sha256"]:
            raise ValueError(f"exact acquired PNG unavailable or SHA mismatch: {name}")
        with Image.open(path) as image:
            if image.format != "PNG" or image.size != (640, 640):
                raise ValueError("source faces must be original 640-square PNGs")
            faces[name] = np.asarray(image.convert("RGB")).copy()
        pose = np.asarray(poses[name]["frame0_camera_to_world_xyz_xyzw"], dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all() or not np.array_equal(pose[:3], front_pose[:3]):
            raise ValueError("source optical centres differ or pose is invalid")
        relative = front_rotation.T @ np.asarray(rotation_matrix(pose[3:]))
        if not np.allclose(relative, contract["face_camera_to_front_ned"][name], atol=1e-6, rtol=0):
            raise ValueError("actual source axes differ from nominal cube contract")
        bindings.append({"face": name, "path": str(path.resolve()), "sha256": row["sha256"],
                         "original_pose": pose.tolist(), "relative_rotation": relative.tolist()})
    controller = virtual_yaw_sequence(key)
    # Preallocate only one raw RGB array; no PNG, MP4, latent or cache output.
    rgb = np.empty((49, 480, 832, 3), dtype=np.uint8)
    for frame, yaw in enumerate(controller["yaw_degrees"]):
        rgb[frame] = project_cube_yaw(faces, yaw, metadata=contract).image
    first_hash = hashlib.sha256(rgb[0].tobytes()).hexdigest()
    actions = torch.tensor(controller["binary_actions"], dtype=torch.float32)
    packed = pack_canonical_actions(actions)[0]
    environment, camera, trajectory = expected_episode.split("/")
    return Virtual49Sample(torch.from_numpy(rgb).permute(3, 0, 1, 2), actions, packed, {
        "key": key, "source_episode": expected_episode, "environment": environment,
        "trajectory": trajectory, "camera": camera, "source_frame": 0,
        "group_id": f"{expected_episode}/cube-000000",
        "source_receipt_path": str(receipt_path.resolve()),
        "source_receipt_sha256": expected_receipt_sha256,
        "source_bindings": bindings, "nominal_cube_contract": contract,
        "controller": controller, "first_raw_RGB_sha256": first_hash,
        "rgb_source": "NEW_on_demand_source_cube_CPU_RGB_not_MP4_recovery",
        "old_MP4_read_or_replaced": False, "encoding_applied": False,
        "frame_timestamps_seconds": controller["timestamps_seconds"],
        "action_alignment": controller["action_alignment"],
        "physical_calibration_verified": False, "split": "unassigned",
        "training_qualified": False, "model_generated": False,
        "latents_created": False, "gpu_seconds": 0, "network_requests": 0,
    })


def source_cube_start_check(samples: Sequence[Virtual49Sample]) -> dict:
    """Compare actual raw starting pixels, not source or preencode claims."""
    eligible = len(samples) == 2 and {s.metadata["key"] for s in samples} == {"J", "L"}
    same_group = len({s.metadata["group_id"] for s in samples}) == 1
    same_pixels = eligible and same_group and torch.equal(samples[0].rgb[:, 0], samples[1].rgb[:, 0])
    return {"actual_raw_first_pixels_equal": same_pixels, "same_source_group": same_group,
            "first_raw_RGB_sha256": [s.metadata["first_raw_RGB_sha256"] for s in samples],
            "encoding_applied": False, "training_qualified": False}


def require_profile_ready(samples: Sequence[Virtual49Sample], split_by_group: Mapping[str, str]) -> None:
    """Guard only; it does not qualify sources or start a profile/optimizer.

    Environment and whole trajectory stay in one split, including every cube
    frame and both yaw arms. One panorama cannot be its own development set.
    """
    environments, trajectories = {}, {}
    for sample in samples:
        meta = sample.metadata
        split = split_by_group.get(meta["group_id"])
        if split not in ("train", "dev", "test"):
            raise ValueError("explicit group split required; no automatic assignment")
        for table, identity in ((environments, meta["environment"]),
                                (trajectories, meta["source_episode"])):
            if identity in table and table[identity] != split:
                raise ValueError("source environment/trajectory/cube leaks across splits")
            table[identity] = split
    if len(environments) < 2 or not {"train", "dev"}.issubset(set(environments.values())):
        raise ValueError("independent train/dev environments required; single Office group is inspection-only")
    if not samples or any(sample.metadata["training_qualified"] is not True for sample in samples):
        raise ValueError("source qualification missing; profile/training refused")
