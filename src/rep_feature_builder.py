"""
rep_feature_builder.py
By: Natiq Ghafoor

Builds per-rep feature dictionaries from pose keypoints and rep segments.

Primary usage:
- Given keypoints per frame and rep windows (start_frame, end_frame),
  compute min angle, max angle, ROM, duration, and basic quality counters.

Key behavior:
- Always returns exactly one feature dict per rep segment, even if the angle
  signal is missing for that window. This keeps indexing consistent across
  reps, stability metrics, and overlay rendering.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional


def compute_angle(a, b, c) -> float:
    """
    Returns the angle (radians) at point b given points a, b, c.

    If any point is missing or degenerate, returns 0.0.
    """
    if a is None or b is None or c is None:
        return 0.0

    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot_prod = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    cos_angle = dot_prod / (mag_ba * mag_bc)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.acos(cos_angle)


def _pick_elbow_triplet(
    kps: Dict,
    primary: str = "right",
    fallback: str = "left",
):
    """
    Selects the (shoulder, elbow, wrist) triplet for elbow-angle computation.

    Preference order:
    - primary side (typically right)
    - fallback side (typically left)
    """
    s = kps.get(f"{primary}_shoulder")
    e = kps.get(f"{primary}_elbow")
    w = kps.get(f"{primary}_wrist")
    if s is not None and e is not None and w is not None:
        return s, e, w

    s = kps.get(f"{fallback}_shoulder")
    e = kps.get(f"{fallback}_elbow")
    w = kps.get(f"{fallback}_wrist")
    if s is not None and e is not None and w is not None:
        return s, e, w

    return None, None, None


def extract_rep_features(
    keypoints_sequence: List[Dict],
    reps: List[Tuple[int, int]],
    fps: float = 30.0,
    config: Optional[Dict] = None,
) -> List[Dict]:
    """
    Extracts per-rep statistics from elbow-angle frames.

    config keys:
      - angle_low_cutoff: angles below this are ignored (filters extreme glitches)
      - angle_max_jump: consecutive angle jump limit (filters single-frame spikes)
    """
    config = config or {}

    low_cutoff = float(config.get("angle_low_cutoff", 15))
    max_jump = float(config.get("angle_max_jump", 60))

    rep_features: List[Dict] = []

    for (start_frame, end_frame) in reps:
        segment_angles: List[float] = []

        # Build a cleaned angle sequence for this rep window.
        for f in range(start_frame, end_frame + 1):
            kps = keypoints_sequence[f]
            shoulder, elbow, wrist = _pick_elbow_triplet(kps, primary="right", fallback="left")

            angle_rad = compute_angle(shoulder, elbow, wrist)
            angle_deg = math.degrees(angle_rad) if angle_rad else 0.0

            # Drop angles that are implausibly low (often bad keypoints).
            if angle_deg < low_cutoff:
                continue

            # Drop sudden jumps that typically come from a single bad frame.
            if segment_angles and abs(angle_deg - segment_angles[-1]) > max_jump:
                continue

            segment_angles.append(angle_deg)

        rep_duration = (end_frame - start_frame + 1) / float(fps) if fps else 0.0

        # Keep alignment: always emit one dict per rep.
        if not segment_angles:
            rep_features.append(
                {
                    "start": start_frame,
                    "end": end_frame,
                    "min_angle": None,
                    "max_angle": None,
                    "rom": 0.0,
                    "rep_duration": rep_duration,
                    "valid_angle_frames": 0,
                }
            )
            continue

        min_angle = float(min(segment_angles))
        max_angle = float(max(segment_angles))
        rom = float(max_angle - min_angle)

        rep_features.append(
            {
                "start": start_frame,
                "end": end_frame,
                "min_angle": min_angle,
                "max_angle": max_angle,
                "rom": rom,
                "rep_duration": rep_duration,
                "valid_angle_frames": len(segment_angles),
            }
        )

    return rep_features
