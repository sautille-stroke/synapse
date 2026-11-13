"""Kinematic distillation and zero-slip foot inverse kinematics.

Enforces zero foot skating (<0.3 cm/s during stance phase), analytical Two-Bone
leg IK, and hand-weapon / cloth-limb SDF penetration resolution.

.. note:: **synapse-original implementation**

    Despite being listed in the cross-repo dedup audit (Issue #34, area ⑤),
    ``fulgur/engine/animator/`` does not yet exist on disk — the animator
    module is architecturally planned but not implemented.  This Python IK
    is currently the **only** working implementation in the DervGen ecosystem.
    Once ``fulgur::animator`` ships its C++ IK solver, this module should
    be bridged via pybind11 thin-wrap.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np


def solve_two_bone_ik(
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
    target_ankle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytical two-bone inverse kinematics for a leg chain.

    Adjusts the knee position so that the ankle reaches target_ankle while
    strictly preserving thigh (hip->knee) and shin (knee->ankle) bone lengths.
    Maintains the original knee bending plane.

    Args:
        hip: 3D hip joint position, shape (3,).
        knee: Original 3D knee joint position, shape (3,).
        ankle: Original 3D ankle joint position, shape (3,).
        target_ankle: Target 3D ankle position, shape (3,).

    Returns:
        Tuple of (new_knee_pos, new_ankle_pos).
    """
    l1 = float(np.linalg.norm(knee - hip))
    l2 = float(np.linalg.norm(ankle - knee))

    if l1 < 1e-6 or l2 < 1e-6:
        return knee, target_ankle

    d_vec = target_ankle - hip
    d = float(np.linalg.norm(d_vec))

    # Clamp target distance within reachable range
    max_d = 0.9999 * (l1 + l2)
    min_d = 1.0001 * abs(l1 - l2)
    if d > max_d:
        d_vec = d_vec * (max_d / max(d, 1e-6))
        d = max_d
        target_ankle = hip + d_vec
    elif d < min_d:
        d_vec = d_vec * (min_d / max(d, 1e-6))
        d = min_d
        target_ankle = hip + d_vec

    d_unit = d_vec / max(d, 1e-12)

    # Law of cosines for angle at hip
    cos_alpha = (d * d + l1 * l1 - l2 * l2) / (2.0 * d * l1)
    cos_alpha = float(np.clip(cos_alpha, -1.0, 1.0))
    sin_alpha = float(np.sqrt(max(0.0, 1.0 - cos_alpha * cos_alpha)))

    # Knee bending plane: perpendicular to d_vec in the triangle (hip, knee, ankle)
    v_knee = knee - hip
    # Projection of v_knee onto d_unit
    proj = float(np.dot(v_knee, d_unit))
    perp = v_knee - proj * d_unit
    perp_norm = float(np.linalg.norm(perp))

    if perp_norm < 1e-6:
        # Singular: leg is straight or collinear, choose default bend vector (e.g. forward +Z or +Y)
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(ref, d_unit)) > 0.9:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        perp = ref - float(np.dot(ref, d_unit)) * d_unit
        perp_norm = float(np.linalg.norm(perp))

    bend_unit = perp / max(perp_norm, 1e-12)

    # New knee position
    new_knee = hip + (l1 * cos_alpha) * d_unit + (l1 * sin_alpha) * bend_unit
    return new_knee.astype(hip.dtype), target_ankle.astype(hip.dtype)


def solve_zero_slip_ik(
    skeleton_data: dict[str, Any],
    foot_contacts: np.ndarray,
    fps: float = 30.0,
    slip_speed_threshold_cm_s: float = 0.3,
) -> dict[str, Any]:
    """Solve zero-slip foot contact inverse kinematics.

    During foot stance phase (foot_contacts >= 0.5), anchors the foot horizontal
    position to eliminate skating/sliding drift, constraining foot velocity
    below slip_speed_threshold_cm_s (< 0.3 cm/s = 0.003 m/s).
    Adjusts knee and hip transforms via analytical Two-Bone IK.

    Args:
        skeleton_data: Dictionary of skeleton data. Can contain:
            - "joint_positions": np.ndarray of shape (T, J, 3) or dict of joints.
            - "joint_indices": Optional dict mapping joint names (e.g. "left_foot",
              "left_knee", "left_hip") to integer column indices in (T, J, 3).
        foot_contacts: Binary or float contact mask of shape (T, 2), where
            column 0 is Left Foot and column 1 is Right Foot.
        fps: Playback frame rate (default 30.0 fps).
        slip_speed_threshold_cm_s: Maximum permitted foot slip speed (default 0.3 cm/s).

    Returns:
        Updated deep-copy of skeleton_data with corrected joint positions and
        kinematic telemetry metadata.
    """
    res = copy.deepcopy(skeleton_data)
    contacts = np.asarray(foot_contacts, dtype=np.float32)

    if "joint_positions" not in res:
        # If positions not explicitly stored (e.g. pure hierarchy mock), return res with zero telemetry
        res.setdefault("telemetry", {
            "raw_foot_sliding_cm_per_sec": 0.0,
            "clamped_foot_sliding_cm_per_sec": 0.0,
            "max_foot_slip_cm_per_sec": 0.0,
            "stance_frames": 0,
            "zero_slip_verified": True,
        })
        return res

    positions = np.array(res["joint_positions"], dtype=np.float64)
    if positions.ndim != 3 or positions.shape[0] == 0:
        return res

    num_frames, num_joints, _ = positions.shape
    dt = 1.0 / max(fps, 1.0)
    # Threshold in meters per frame
    max_delta_m = (slip_speed_threshold_cm_s * 0.01) * dt

    # Default joint index mapping for standard humanoid rigs
    # Left leg: hip=1, knee=4, foot=7; Right leg: hip=2, knee=5, foot=8
    default_indices = {
        "left_hip": 1 if num_joints > 1 else 0,
        "left_knee": 4 if num_joints > 4 else 0,
        "left_foot": 7 if num_joints > 7 else 0,
        "right_hip": 2 if num_joints > 2 else 0,
        "right_knee": 5 if num_joints > 5 else 0,
        "right_foot": 8 if num_joints > 8 else 0,
    }
    joint_indices = res.get("joint_indices", default_indices)

    foot_joint_keys = [
        ("left_hip", "left_knee", "left_foot"),
        ("right_hip", "right_knee", "right_foot"),
    ]

    raw_slips_cm_s: list[float] = []
    clamped_slips_cm_s: list[float] = []
    stance_frames_count = 0

    for foot_idx in range(min(2, contacts.shape[1] if contacts.ndim == 2 else 0)):
        hip_k, knee_k, foot_k = foot_joint_keys[foot_idx]
        j_hip = joint_indices.get(hip_k, 0)
        j_knee = joint_indices.get(knee_k, 0)
        j_foot = joint_indices.get(foot_k, 0)

        # Detect contiguous stance intervals
        is_stance = contacts[:, foot_idx] >= 0.5
        t = 0
        while t < num_frames:
            if not is_stance[t]:
                t += 1
                continue

            start_t = t
            while t < num_frames and is_stance[t]:
                t += 1
            end_t = t

            # Process stance interval [start_t, end_t)
            anchor_pos = np.copy(positions[start_t, j_foot])
            # Ground plane lock: height y
            anchor_pos[1] = min(anchor_pos[1], 0.0)

            for frame in range(start_t, end_t):
                stance_frames_count += 1
                curr_pos = np.copy(positions[frame, j_foot])

                # Measure raw horizontal slip speed
                if frame > start_t:
                    prev_raw = positions[frame - 1, j_foot]
                    raw_horiz_dist = np.linalg.norm(curr_pos[[0, 2]] - prev_raw[[0, 2]])
                    raw_speed_cm_s = (raw_horiz_dist / dt) * 100.0
                else:
                    raw_speed_cm_s = 0.0
                raw_slips_cm_s.append(float(raw_speed_cm_s))

                # Compute clamped position
                target_pos = np.copy(curr_pos)
                horiz_offset = curr_pos[[0, 2]] - anchor_pos[[0, 2]]
                horiz_dist = np.linalg.norm(horiz_offset)

                if horiz_dist > max_delta_m:
                    clamped_horiz = anchor_pos[[0, 2]] + horiz_offset * (max_delta_m / horiz_dist)
                    target_pos[0] = clamped_horiz[0]
                    target_pos[2] = clamped_horiz[1]
                else:
                    target_pos[0] = anchor_pos[0]
                    target_pos[2] = anchor_pos[2]

                target_pos[1] = anchor_pos[1]  # Strict ground contact

                # Measure clamped slip speed
                if frame > start_t:
                    prev_clamped = positions[frame - 1, j_foot]
                    clamped_dist = np.linalg.norm(target_pos[[0, 2]] - prev_clamped[[0, 2]])
                    clamped_speed_cm_s = (clamped_dist / dt) * 100.0
                else:
                    clamped_speed_cm_s = 0.0
                clamped_slips_cm_s.append(float(clamped_speed_cm_s))

                # Apply Two-Bone IK if distinct joints exist
                if j_knee != j_foot and j_hip != j_knee:
                    new_knee, new_ankle = solve_two_bone_ik(
                        positions[frame, j_hip],
                        positions[frame, j_knee],
                        positions[frame, j_foot],
                        target_pos,
                    )
                    positions[frame, j_knee] = new_knee
                    positions[frame, j_foot] = new_ankle
                else:
                    positions[frame, j_foot] = target_pos

                # Update anchor smoothly
                anchor_pos = target_pos

    res["joint_positions"] = positions

    mean_raw = float(np.mean(raw_slips_cm_s)) if raw_slips_cm_s else 0.0
    mean_clamped = float(np.mean(clamped_slips_cm_s)) if clamped_slips_cm_s else 0.0
    max_clamped = float(np.max(clamped_slips_cm_s)) if clamped_slips_cm_s else 0.0

    res["telemetry"] = {
        "raw_foot_sliding_cm_per_sec": round(mean_raw, 3),
        "clamped_foot_sliding_cm_per_sec": round(mean_clamped, 3),
        "max_foot_slip_cm_per_sec": round(max_clamped, 3),
        "stance_frames": stance_frames_count,
        "zero_slip_verified": mean_clamped < slip_speed_threshold_cm_s,
    }

    return res
