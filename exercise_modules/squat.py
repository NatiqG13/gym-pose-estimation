"""
squat.py
By: Natiq Ghafoor

Squat exercise module.

This file provides the squat-specific hooks used by the main pipeline:
- Overlay configuration (limb pairs)
- Joints used for tracking/stability
- The joint used for rep segmentation (knee angle)
- Rep-level feature extraction (knee ROM + supporting signals)
- Rep evaluation driven by calibration thresholds

Primary signal:
Right knee angle computed from hip → knee → ankle.

Supporting signals:
- Right hip angle (shoulder → hip → knee)
- Torso lean (shoulder–hip line relative to vertical), used as a simple "hinge" indicator.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from src.rep_feature_builder import compute_angle  # returns radians


def get_limb_pairs() -> List[Tuple[str, str]]:
    """
    Returns limb connections used by the overlay renderer for squats.
    """
    return [
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("left_hip", "right_hip"),
        ("left_shoulder", "right_shoulder"),
    ]


def get_tracked_joints() -> List[str]:
    """
    Returns joints used to keep tracking centered on the lower body.
    """
    return ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]


def get_stability_joints() -> List[str]:
    """
    Returns joints used for stability metrics (lower-body steadiness).
    """
    return ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]


def get_segmentation_joint() -> str:
    """
    Returns the joint key used by calibration and segmentation for squats.
    """
    return "right_knee"


def _torso_lean_deg(
    shoulder: Optional[Tuple[float, float, float]],
    hip: Optional[Tuple[float, float, float]],
) -> float:
    """
    Computes torso lean (degrees) relative to vertical using the shoulder→hip vector.

    Interpretation:
      0°  = upright (vertical)
      larger values = more forward/back lean
    """
    if not shoulder or not hip:
        return 0.0

    dx = float(shoulder[0]) - float(hip[0])
    dy = float(shoulder[1]) - float(hip[1])

    # Angle to vertical. Using abs(dx) keeps this as a "magnitude of lean".
    return abs(math.degrees(math.atan2(dx, dy)))


def extract_features(
    keypoints_sequence: List[Dict[str, Any]],
    reps: List[Tuple[int, int]],
    fps: float = 30.0,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Computes per-rep features for squat evaluation.

    Each rep includes:
      - min_angle / max_angle / rom (right knee)
      - hip_rom (right hip angle range)
      - torso_lean_rom (range of torso lean during the rep)
      - rep_duration (seconds)

    Notes:
    - Knee ROM is the primary movement metric.
    - Hip/torso signals are used for simple “hinge vs knee-bend” detection.
    """
    rep_features: List[Dict[str, Any]] = []

    if not keypoints_sequence or not reps:
        return rep_features

    # Avoid division by zero if fps is missing/invalid.
    fps_val = float(fps) if fps and fps > 0 else 30.0

    for (start, end) in reps:
        start = int(start)
        end = int(end)

        if end < start:
            start, end = end, start

        knee_angles: List[float] = []
        hip_angles: List[float] = []
        torso_lean: List[float] = []

        for f in range(start, end + 1):
            kps = keypoints_sequence[f]

            # Knee angle (right side): hip - knee - ankle
            A_knee = kps.get("right_hip")
            B_knee = kps.get("right_knee")
            C_knee = kps.get("right_ankle")
            knee_rad = compute_angle(A_knee, B_knee, C_knee)
            knee_deg = math.degrees(knee_rad)
            knee_angles.append(float(knee_deg))

            # Hip angle (right side): shoulder - hip - knee
            A_hip = kps.get("right_shoulder")
            B_hip = kps.get("right_hip")
            C_hip = kps.get("right_knee")
            hip_rad = compute_angle(A_hip, B_hip, C_hip)
            hip_deg = math.degrees(hip_rad)
            hip_angles.append(float(hip_deg))

            # Torso lean: shoulder-hip line vs vertical
            torso_lean.append(_torso_lean_deg(kps.get("right_shoulder"), kps.get("right_hip")))

        if not knee_angles:
            continue

        knee_min = float(min(knee_angles))
        knee_max = float(max(knee_angles))
        knee_rom = float(knee_max - knee_min)

        hip_rom = float(max(hip_angles) - min(hip_angles)) if hip_angles else 0.0
        torso_rom = float(max(torso_lean) - min(torso_lean)) if torso_lean else 0.0
        duration = float(end - start + 1) / fps_val

        rep_features.append(
            {
                "start": start,
                "end": end,
                "min_angle": knee_min,
                "max_angle": knee_max,
                "rom": knee_rom,              # primary metric: knee ROM
                "hip_rom": hip_rom,           # supporting metric: hip movement range
                "torso_lean_rom": torso_rom,  # supporting metric: torso lean range
                "rep_duration": duration,
            }
        )

    return rep_features


def evaluate_rep(rep: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """
    Evaluates a squat rep using knee thresholds, tempo constraints, and basic cheat heuristics.

    Expected calibration keys:
      - min_angle / max_angle: depth + lockout targets (knee angle degrees)
      - rom_threshold: minimum knee ROM
      - min_duration / max_duration: tempo limits

    Optional cheat detection keys:
      - knee_small_rom_limit: knee ROM below this is treated as "minimal knee bend"
      - torso_lean_cheat_thr: torso lean range above this suggests hinging/swinging
      - hip_knee_ratio_thr: hip ROM much larger than knee ROM suggests a hinge-dominant rep

    Returns:
      ("pass" or "fail", short_reason)
    """
    cfg = config or {}

    min_angle_thresh = float(cfg.get("min_angle", 70))     # depth target (lower = deeper)
    max_angle_thresh = float(cfg.get("max_angle", 165))    # top/lockout target
    rom_threshold = float(cfg.get("rom_threshold", 60))
    min_duration = float(cfg.get("min_duration", 0.8))
    max_duration = float(cfg.get("max_duration", 5.0))

    lockout_margin = float(cfg.get("lockout_margin", 15))

    knee_small_rom_limit = float(cfg.get("knee_small_rom_limit", 35))
    torso_lean_cheat_thr = float(cfg.get("torso_lean_cheat_thr", 15))
    hip_knee_ratio_thr = float(cfg.get("hip_knee_ratio_thr", 1.5))

    knee_rom = float(rep.get("rom", 0))
    hip_rom = float(rep.get("hip_rom", 0))
    torso_rom = float(rep.get("torso_lean_rom", 0))
    duration = float(rep.get("rep_duration", 0))
    kmin = float(rep.get("min_angle", 999))
    kmax = float(rep.get("max_angle", 0))

    # Tempo checks
    if duration < min_duration:
        return "fail", "Squat too fast"
    if duration > max_duration:
        return "fail", "Squat too slow"

    # Lockout, depth, and ROM checks
    if kmax < (max_angle_thresh - lockout_margin):
        return "fail", "No lockout at top"
    if kmin > min_angle_thresh:
        return "fail", "Not deep enough"
    if knee_rom < rom_threshold:
        return "fail", "Not enough range of motion"

    # Cheat heuristic: small knee motion combined with large torso/hip motion.
    if knee_rom < knee_small_rom_limit:
        ratio = hip_rom / max(knee_rom, 1e-3)
        if torso_rom > torso_lean_cheat_thr or (hip_rom > 0 and ratio > hip_knee_ratio_thr):
            return "fail", "Back hinge, minimal knee bend"

    return "pass", "Good squat"
