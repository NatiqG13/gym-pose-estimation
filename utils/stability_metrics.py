"""
stability_metrics.py
By: Natiq Ghafoor

Rep-level stability metrics derived from joint motion.

Current approach:
- For each joint, compute frame-to-frame displacement over a rep window
- Use the standard deviation of those displacements as the joint "jitter" score
- Average joint scores to produce a single rep-level stability score

Notes:
- This is intentionally lightweight and camera-space based (pixels).
- It is used as a diagnostic signal rather than a strict pass/fail criterion.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np


PoseDict = Dict[str, Tuple[float, float, float]]


def smooth_sequence(seq: List[float], window_size: int = 5) -> List[float]:
    """
    Moving-average smoother for a 1D sequence.

    This is used for optional visualization/debugging. It is not required for
    the displacement-based jitter metric.
    """
    if not seq:
        return []

    window_size = int(window_size) if window_size and window_size > 0 else 1
    if window_size == 1:
        return list(seq)

    smoothed: List[float] = []
    half = window_size // 2

    for i in range(len(seq)):
        start = max(0, i - half)
        end = min(len(seq), i + half + 1)
        window = seq[start:end]

        # Window is never empty in normal conditions, but keep this defensive.
        if not window:
            smoothed.append(float(seq[i]))
        else:
            smoothed.append(float(sum(window) / len(window)))

    return smoothed


def compute_joint_jitter(pose_sequence: List[Optional[PoseDict]], joint_name: str, start: int, end: int) -> float:
    """
    Computes a joint jitter score over [start, end] (inclusive).

    The score is stddev of per-frame displacement magnitudes. Frames where the
    joint is missing or confidence is 0 are skipped.
    """
    prev_coords: Optional[Tuple[float, float]] = None
    displacements: List[float] = []

    start = int(start)
    end = int(end)
    if end < start:
        start, end = end, start

    end = min(end, max(0, len(pose_sequence) - 1))
    if start > end or not pose_sequence:
        return 0.0

    for frame_idx in range(start, end + 1):
        pose_dict = pose_sequence[frame_idx]
        if not pose_dict:
            continue

        v = pose_dict.get(joint_name)
        if v is None:
            continue

        x, y, conf = v
        if float(conf) <= 0.0:
            continue

        coords = (float(x), float(y))

        if prev_coords is not None:
            dx = coords[0] - prev_coords[0]
            dy = coords[1] - prev_coords[1]
            displacements.append(float((dx * dx + dy * dy) ** 0.5))

        prev_coords = coords

    return float(np.std(displacements)) if displacements else 0.0


def compute_stability_for_reps(
    pose_sequence: List[Optional[PoseDict]],
    reps: List[Tuple[int, int]],
    stability_joints: List[str],
) -> List[Dict]:
    """
    Computes stability metrics per rep by averaging joint jitter scores.
    """
    if not reps:
        return []

    stability_results: List[Dict] = []
    joints = stability_joints or []

    for start, end in reps:
        joint_jitters: Dict[str, float] = {}

        for joint_name in joints:
            joint_jitters[joint_name] = compute_joint_jitter(pose_sequence, joint_name, start, end)

        # Avoid division by zero if the caller supplies an empty joint list.
        stability_score = (sum(joint_jitters.values()) / len(joint_jitters)) if joint_jitters else 0.0

        stability_results.append(
            {
                "start": int(start),
                "end": int(end),
                "joint_stability": joint_jitters,
                "stability_score": float(stability_score),
            }
        )

    return stability_results
