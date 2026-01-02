"""
bench.py
By: Natiq Ghafoor

Bench press exercise module.

This file provides the small API that the main pipeline expects for an exercise:
- Overlay configuration (which limbs/joints to draw)
- Per-frame elbow angle extraction (rep signal)
- Per-rep feature computation (duration, min/max angle, ROM)
- Rep evaluation using calibration thresholds

Rep signal:
Elbow angle computed from shoulder → elbow → wrist. For each frame, the side
(left/right) with better keypoint confidence is selected when available.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def get_limb_pairs() -> List[Tuple[str, str]]:
    """
    Returns limb connections used by the overlay renderer for bench.
    """
    return [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
    ]


def get_segmentation_joint() -> str:
    """
    Returns the joint key used by calibration and segmentation for bench.
    """
    return "right_elbow"


def get_tracked_joints() -> List[str]:
    """
    Returns joints used for basic subject continuity (when multiple people may appear).
    """
    return [
        "right_shoulder", "right_elbow", "right_wrist",
        "left_shoulder", "left_elbow", "left_wrist",
        "right_hip", "left_hip",
    ]


def get_stability_joints() -> List[str]:
    """
    Returns joints used for stability metrics (movement steadiness over time).
    """
    return [
        "right_wrist", "right_elbow", "right_shoulder",
        "left_wrist", "left_elbow", "left_shoulder",
    ]


# Allows minor naming differences across pose outputs without breaking feature extraction.
_ALIASES = {
    "right_shoulder": ["right_shoulder", "r_shoulder", "RShoulder"],
    "right_elbow": ["right_elbow", "r_elbow", "RElbow"],
    "right_wrist": ["right_wrist", "r_wrist", "RWrist"],
    "left_shoulder": ["left_shoulder", "l_shoulder", "LShoulder"],
    "left_elbow": ["left_elbow", "l_elbow", "LElbow"],
    "left_wrist": ["left_wrist", "l_wrist", "LWrist"],
}


def _get_joint(pose: Dict[str, Any], name: str) -> Optional[Tuple[float, float, float]]:
    """
    Fetches a joint from a pose dictionary and returns (x, y, conf).

    Accepted formats:
    - (x, y, conf)
    - (x, y)  -> treated as conf=1.0
    """
    if not pose:
        return None

    for key in _ALIASES.get(name, [name]):
        v = pose.get(key)
        if v is None:
            continue

        if isinstance(v, (tuple, list)) and len(v) >= 2:
            x = float(v[0])
            y = float(v[1])
            c = float(v[2]) if len(v) >= 3 else 1.0
            return (x, y, c)

    return None


def _angle_deg(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    Computes the angle ABC in degrees.
    """
    bax = a[0] - b[0]
    bay = a[1] - b[1]
    bcx = c[0] - b[0]
    bcy = c[1] - b[1]

    dot = bax * bcx + bay * bcy
    mag1 = math.hypot(bax, bay)
    mag2 = math.hypot(bcx, bcy)

    # Degenerate segments produce unstable angles.
    if mag1 < 1e-6 or mag2 < 1e-6:
        return float("nan")

    cosv = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cosv))


def _elbow_angle_for_side(pose: Dict[str, Any], side: str) -> Tuple[Optional[float], float]:
    """
    Computes elbow angle for one side ("right" or "left").

    Returns:
      (angle_degrees_or_None, confidence_score)

    confidence_score is the mean confidence across shoulder/elbow/wrist.
    """
    if side not in ("right", "left"):
        return None, 0.0

    sh = _get_joint(pose, f"{side}_shoulder")
    el = _get_joint(pose, f"{side}_elbow")
    wr = _get_joint(pose, f"{side}_wrist")

    if sh is None or el is None or wr is None:
        return None, 0.0

    conf = (sh[2] + el[2] + wr[2]) / 3.0

    # Very low confidence frames are treated as missing to avoid noisy spikes.
    if conf < 0.05:
        return None, conf

    ang = _angle_deg((sh[0], sh[1]), (el[0], el[1]), (wr[0], wr[1]))
    if math.isnan(ang):
        return None, conf

    # Keep within a valid elbow-angle range.
    ang = max(0.0, min(180.0, float(ang)))
    return ang, conf


def extract_frame_features(
    pose_sequence: List[Dict[str, Any]],
    fps: float,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Builds per-frame features used by the rep segmentation step.

    Primary output:
      bench_elbow_angle (degrees)

    The side (left/right) may switch between frames based on keypoint confidence.
    """
    features: List[Dict[str, Any]] = []

    for i, pose in enumerate(pose_sequence):
        right_ang, right_conf = _elbow_angle_for_side(pose, "right")
        left_ang, left_conf = _elbow_angle_for_side(pose, "left")

        # Default to right; use left when it is clearly more reliable for this frame.
        angle = right_ang
        chosen = "right"

        if angle is None and left_ang is not None:
            angle = left_ang
            chosen = "left"
        elif right_ang is not None and left_ang is not None:
            if left_conf > right_conf + 0.05:
                angle = left_ang
                chosen = "left"

        features.append(
            {
                "frame_idx": i,
                "t": (i / fps) if fps and fps > 0 else 0.0,
                "bench_elbow_angle": angle,
                "bench_angle_side": chosen,
                "right_elbow_angle": right_ang,
                "left_elbow_angle": left_ang,
                "right_conf": right_conf,
                "left_conf": left_conf,
            }
        )

    return features


def get_angle_sequence(frame_features: List[Dict[str, Any]]) -> List[Optional[float]]:
    """
    Extracts the primary angle signal from a frame feature list.
    """
    return [f.get("bench_elbow_angle") for f in frame_features]


def extract_features(
    pose_sequence: List[Dict[str, Any]],
    reps: List[Tuple[int, int]],
    fps: float,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Computes per-rep metrics used by evaluation and CSV export.

    reps:
      List of (start_frame, end_frame)
    """
    frame_feats = extract_frame_features(pose_sequence, fps=fps, config=config)
    angles = get_angle_sequence(frame_feats)

    rep_list: List[Dict[str, Any]] = []

    for (start_idx, end_idx) in reps:
        start_idx = int(start_idx)
        end_idx = int(end_idx)

        # Normalize ordering in case indices arrive swapped.
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx

        seg = angles[start_idx : end_idx + 1]
        clean = [a for a in seg if isinstance(a, (int, float))]

        rep: Dict[str, Any] = {
            "start": start_idx,
            "end": end_idx,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "anomalies": [],
        }

        # Duration based on frame count and FPS.
        if fps and fps > 0:
            rep_duration = (end_idx - start_idx + 1) / float(fps)
        else:
            rep_duration = 0.0
            rep["anomalies"].append("invalid_fps")

        rep["rep_duration"] = float(rep_duration)
        rep["duration"] = round(float(rep_duration), 2)

        # If the angle signal is missing for the rep window, keep a placeholder record.
        if not clean:
            rep.update(
                {
                    "rep_min_angle": 0.0,
                    "rep_max_angle": 0.0,
                    "rom": 0.0,
                    "ROM": 0.0,
                }
            )
            rep["anomalies"].append("no_angle_data")
            rep_list.append(rep)
            continue

        rep_min = float(min(clean))
        rep_max = float(max(clean))
        rom = float(rep_max - rep_min)

        rep["rep_min_angle"] = rep_min
        rep["rep_max_angle"] = rep_max
        rep["rom"] = rom
        rep["ROM"] = rom  # compatibility key used in a few places

        rep_list.append(rep)

    return rep_list


def extract_rep_features(
    pose_sequence: List[Dict[str, Any]],
    reps: List[Tuple[int, int]],
    fps: float,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Backward-compatible alias for older code paths.
    """
    return extract_features(pose_sequence, reps, fps=fps, config=config)


def evaluate_rep(rep: Dict[str, Any], config: Dict[str, Any]) -> Tuple[str, str]:
    """
    Evaluates a rep using calibration thresholds.

    Supported calibration keys:
      - min_angle: bottom threshold (bench bottom corresponds to smaller elbow angles)
      - max_angle: top/lockout threshold
      - min_rom:   minimum required ROM
      - min_duration / max_duration: timing limits
      - lockout_tolerance / depth_tolerance: noise buffers (degrees)

    Returns:
      ("pass" or "fail", short_reason)
    """
    rep_min = float(rep.get("rep_min_angle", rep.get("min_angle", 0.0)) or 0.0)
    rep_max = float(rep.get("rep_max_angle", rep.get("max_angle", 0.0)) or 0.0)
    rom = float(rep.get("ROM", rep.get("rom", 0.0)) or 0.0)
    duration = float(rep.get("rep_duration", rep.get("duration", 0.0)) or 0.0)

    min_angle_thr = float(config.get("min_angle", 60.0))
    max_angle_thr = float(config.get("max_angle", 160.0))
    min_rom_thr = float(config.get("min_rom", 70.0))
    min_dur_thr = float(config.get("min_duration", 0.3))
    max_dur_thr = float(config.get("max_duration", 6.0))

    # Tolerances reduce false fails caused by frame-to-frame pose jitter.
    lockout_tol = float(config.get("lockout_tolerance", 5.0))
    depth_tol = float(config.get("depth_tolerance", 5.0))

    if duration > max_dur_thr:
        return "fail", "Too slow"
    if duration < min_dur_thr:
        return "fail", "Too fast"

    if rep_max < (max_angle_thr - lockout_tol):
        return "fail", "No lockout"

    if rep_min > (min_angle_thr + depth_tol):
        return "fail", "Not deep enough"

    if rom < min_rom_thr:
        return "fail", "Not enough ROM"

    return "pass", "Good rep"
