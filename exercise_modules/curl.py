"""
curl.py
By: Natiq Ghafoor

Bicep curl exercise module.

This module plugs into the main pipeline and provides:
- The joint used for rep segmentation (right elbow)
- A shared rep-feature extractor (delegated to src.rep_feature_builder)
- Rep evaluation rules driven by calibration thresholds (easy vs strict)
- Optional overlay/stability configuration hooks used elsewhere in the project

The rep signal for curls is elbow angle: shoulder → elbow → wrist.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.rep_feature_builder import extract_rep_features


def get_tracked_joints() -> List[str]:
    """
    Returns the joints used to keep tracking focused on the primary arm.
    """
    return ["right_elbow", "right_shoulder", "right_wrist"]


def get_segmentation_joint() -> str:
    """
    Returns the joint key used by calibration and segmentation for curls.
    """
    return "right_elbow"


def extract_features(
    keypoints_sequence: List[Dict[str, Any]],
    reps: List[Tuple[int, int]],
    fps: float = 30.0,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Computes per-rep features for curls.

    This delegates to the shared feature builder so the rep dictionaries remain
    consistent across exercises (duration, ROM, min/max angle, etc.).
    """
    return extract_rep_features(keypoints_sequence, reps, fps=fps, config=config)


def evaluate_rep(rep: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """
    Evaluates a single curl rep against calibration thresholds.

    Expected calibration keys (defaults applied if missing):
      - min_angle:     bottom position threshold (smaller = more flexion)
      - max_angle:     top/lockout threshold (larger = more extension)
      - rom_threshold: minimum ROM required for a passing rep
      - min_duration:  minimum allowed rep time (seconds)
      - max_duration:  maximum allowed rep time (seconds)

    Margin keys (used to separate easy vs strict behavior):
      - lockout_margin: relaxed allowance on the top angle
      - bottom_margin:  relaxed allowance on the bottom angle

    Returns:
      ("pass" or "fail", short_reason)
    """
    cfg = config or {}

    # Core thresholds
    min_angle_thresh = float(cfg.get("min_angle", 60))
    max_angle_thresh = float(cfg.get("max_angle", 155))
    rom_threshold = float(cfg.get("rom_threshold", 55))
    min_duration = float(cfg.get("min_duration", 0.4))
    max_duration = float(cfg.get("max_duration", 6.0))

    # Margins are the main lever that makes easy vs strict meaningfully different.
    lockout_margin = float(cfg.get("lockout_margin", 15))
    bottom_margin = float(cfg.get("bottom_margin", 0))

    # Rep metrics produced by the feature builder.
    # These defaults avoid crashes if a value is missing, but missing data will usually fail a rep.
    min_angle = float(rep.get("min_angle", 999))
    max_angle = float(rep.get("max_angle", 0))
    rom = float(rep.get("rom", 0))
    duration = float(rep.get("rep_duration", 0))

    # Tempo checks: allow a small buffer so minor FPS jitter does not flip a label.
    if duration < (min_duration * 0.8):
        return "fail", "Curl too fast"
    if duration > (max_duration * 1.2):
        return "fail", "Curl too slow"

    # Top position: require elbow extension near the configured threshold.
    if max_angle < (max_angle_thresh - lockout_margin):
        return "fail", "No lockout at top"

    # Bottom position: require elbow flexion near the configured threshold.
    if min_angle > (min_angle_thresh + bottom_margin):
        return "fail", "Curl not deep enough"

    # ROM check: ensures the rep moved through enough range to count as a full rep.
    if rom < (rom_threshold * 0.9):
        return "fail", "Not enough range of motion"

    return "pass", "Good curl"


def get_limb_pairs() -> List[Tuple[str, str]]:
    """
    Returns limb connections used by the overlay renderer for curls.
    """
    return []


def get_stability_joints() -> List[str]:
    """
    Returns joints used by stability metrics for curl analysis.
    """
    return ["right_elbow", "right_shoulder", "right_wrist"]
