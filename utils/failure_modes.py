"""
failure_modes.py
By: Natiq Ghafoor

Failure-mode simulations for stress testing the pose → rep pipeline.

Supported simulations:
- Occlusion: forces joint confidence to 0.0 for selected joints/frames
- Jitter: adds Gaussian noise to joint (x, y) for selected joints/frames
- Low-confidence handling: replaces low-confidence joint positions with the last
  known reliable position

Notes:
- These helpers intentionally mutate the pose dict in-place to keep overhead low.
- They are designed for testing robustness; they should be disabled for baseline runs.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import numpy as np


PoseDict = Dict[str, Tuple[float, float, float]]


def simulate_occlusion(pose_dict: Optional[PoseDict], frame_index: int, config: Dict[str, Any]) -> Optional[PoseDict]:
    """
    Sets confidence to 0.0 for selected joints, optionally within a frame range.

    Config styles:
      1) Range-based:
         {"start": 50, "end": 150, "joints": [...], "drop_prob": 0.7}

      2) Frequency-based (legacy):
         {"joints": [...], "frequency": 20, "drop_prob": 0.5}
    """
    if not pose_dict:
        return pose_dict

    joints = config.get("joints", []) or []
    drop_prob = float(config.get("drop_prob", config.get("prob", 0.7)) or 0.0)

    # Range-based mode
    if "start" in config or "end" in config:
        start = int(config.get("start", 0))
        end = int(config.get("end", 0))
        if frame_index < start or frame_index > end:
            return pose_dict

        for joint in joints:
            v = pose_dict.get(joint)
            if v is None:
                continue
            if random.random() < drop_prob:
                x, y, _conf = v
                pose_dict[joint] = (x, y, 0.0)
        return pose_dict

    # Frequency-based mode (legacy)
    frequency = int(config.get("frequency", 20))
    if frequency <= 0:
        return pose_dict

    if frame_index % frequency == 0:
        for joint in joints:
            v = pose_dict.get(joint)
            if v is None:
                continue
            if random.random() < drop_prob:
                x, y, _conf = v
                pose_dict[joint] = (x, y, 0.0)

    return pose_dict


def simulate_jitter(pose_dict: Optional[PoseDict], frame_index: int, config: Dict[str, Any]) -> Optional[PoseDict]:
    """
    Adds Gaussian jitter to joint (x, y) for selected joints within a frame range.

    Expected config:
      {"start": 200, "end": 400, "joints": [...], "prob": 0.2, "jitter_std": 10.0}
    """
    if not pose_dict:
        return pose_dict

    start = int(config.get("start", 0))
    end = int(config.get("end", 0))
    joints = config.get("joints", []) or []
    prob = float(config.get("prob", config.get("drop_prob", 0.0)) or 0.0)
    jitter_std = float(config.get("jitter_std", 0.0) or 0.0)
    verbose = bool(config.get("verbose", False))

    if prob <= 0 or jitter_std <= 0:
        return pose_dict
    if frame_index < start or frame_index > end:
        return pose_dict

    for joint in joints:
        v = pose_dict.get(joint)
        if v is None:
            continue

        x, y, conf = v
        if np.random.rand() < prob:
            dx = float(np.random.normal(0, jitter_std))
            dy = float(np.random.normal(0, jitter_std))
            pose_dict[joint] = (x + dx, y + dy, conf)

            if verbose:
                print(f"[DEBUG] jitter frame={frame_index} joint={joint} dx={dx:.2f} dy={dy:.2f}")

    return pose_dict


def simulate_low_confidence_handling(
    pose_dict: Optional[PoseDict],
    frame_index: int,
    config: Dict[str, Any],
    last_good_pose: Dict[str, Tuple[float, float]],
) -> Tuple[Optional[PoseDict], Dict[str, Tuple[float, float]]]:
    """
    Replaces low-confidence joint positions with the last known reliable values.

    Config:
      {"joints": [...], "threshold": 0.35}
    """
    if not pose_dict:
        return pose_dict, last_good_pose

    joints = config.get("joints", []) or []
    threshold = float(config.get("threshold", 0.0) or 0.0)

    if threshold <= 0:
        return pose_dict, last_good_pose

    for joint in joints:
        v = pose_dict.get(joint)
        if v is None:
            continue

        x, y, conf = v
        if float(conf) < threshold:
            # Fall back to last good if available; keep current confidence value.
            if joint in last_good_pose:
                x_prev, y_prev = last_good_pose[joint]
                pose_dict[joint] = (float(x_prev), float(y_prev), float(conf))
        else:
            last_good_pose[joint] = (float(x), float(y))

    return pose_dict, last_good_pose


# Backward-compatible alias used in older code paths.
simulate_low_confidence = simulate_low_confidence_handling


def apply_failure_modes(
    pose_dict: Optional[PoseDict],
    frame_index: int,
    occlusion_config: Dict[str, Any],
    jitter_config: Dict[str, Any],
    low_conf_config: Dict[str, Any],
    last_good_pose: Dict[str, Tuple[float, float]],
) -> Tuple[Optional[PoseDict], Dict[str, Tuple[float, float]]]:
    """
    Applies enabled failure modes in a consistent order:
      occlusion -> jitter -> low-confidence handling
    """
    pose_dict = simulate_occlusion(pose_dict, frame_index, occlusion_config)
    pose_dict = simulate_jitter(pose_dict, frame_index, jitter_config)
    pose_dict, last_good_pose = simulate_low_confidence_handling(pose_dict, frame_index, low_conf_config, last_good_pose)
    return pose_dict, last_good_pose
