"""
overlay_renderer.py
By: Natiq Ghafoor

Overlay rendering utilities for annotated videos.

Responsibilities:
- Draw a skeleton overlay using provided limb pairs
- Render readable text (outlined) without OpenCV Unicode issues
- Display rep count and a short-lived PASS/FAIL line when a rep ends

Notes:
- Overlay smoothing (EMA) is applied for visualization only.
  Rep segmentation is computed earlier and is not affected by smoothing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2


# -----------------------------
# Module state (frame-based)
# -----------------------------
_last_overlay_text: Optional[str] = None
_last_update_frame: int = -10_000_000
_display_seconds: float = 2.0

# EMA cache for keypoint smoothing (overlay only)
_ema_pose: Dict[str, Tuple[float, float, float]] = {}


def draw_text_outlined(
    frame,
    text: str,
    org: Tuple[int, int],
    font_scale: float = 0.7,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    outline_color: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
    outline_thickness: int = 6,
) -> None:
    """
    Draws outlined text and clamps X so it does not clip off the right edge.
    """
    if frame is None:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX

    (w, h), _baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = int(org[0]), int(org[1])

    # Keep text inside the frame bounds with a small margin.
    max_x = frame.shape[1] - w - 10
    x = max(10, min(x, max_x))
    y = max(10 + h, y)

    # Outline first, then the main text.
    cv2.putText(frame, text, (x, y), font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def _smooth_pose_for_overlay(
    pose_dict: Dict[str, Any],
    min_conf: float,
    alpha: float,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Exponential moving average smoothing for pose keypoints.

    alpha near 1.0 yields a smoother overlay but adds more visual lag.
    """
    global _ema_pose

    if not pose_dict:
        return {}

    try:
        alpha = float(alpha)
    except Exception:
        alpha = 0.0
    alpha = max(0.0, min(0.95, alpha))

    out: Dict[str, Tuple[float, float, float]] = {}
    min_conf = float(min_conf)

    # Update EMA for joints that are currently reliable.
    for name, v in pose_dict.items():
        if not v or len(v) < 3:
            continue

        x, y, c = float(v[0]), float(v[1]), float(v[2])
        if c < min_conf:
            continue

        if name in _ema_pose:
            px, py, _pc = _ema_pose[name]
            sx = (alpha * px) + ((1.0 - alpha) * x)
            sy = (alpha * py) + ((1.0 - alpha) * y)
        else:
            sx, sy = x, y

        _ema_pose[name] = (sx, sy, c)
        out[name] = (sx, sy, c)

    # Light decay when detections drop out, so points do not disappear instantly.
    for name, (px, py, pc) in list(_ema_pose.items()):
        if name in out:
            continue
        pc2 = pc * 0.92
        if pc2 < min_conf:
            _ema_pose.pop(name, None)
            continue
        _ema_pose[name] = (px, py, pc2)
        out[name] = (px, py, pc2)

    return out


def draw_skeleton(
    frame,
    pose_dict: Dict[str, Any],
    limb_pairs: List[Tuple[str, str]],
    min_conf: float = 0.30,
    smooth_alpha: float = 0.75,
    enable_smoothing: bool = True,
) -> None:
    """
    Draws limbs and joint markers for the current frame.
    """
    if frame is None or not pose_dict:
        return

    if enable_smoothing and smooth_alpha > 0:
        pose_dict = _smooth_pose_for_overlay(pose_dict, min_conf=min_conf, alpha=smooth_alpha)

    # Limbs first so joint dots sit on top.
    for (a_name, b_name) in limb_pairs or []:
        a = pose_dict.get(a_name)
        b = pose_dict.get(b_name)
        if not a or not b or len(a) < 3 or len(b) < 3:
            continue

        ax, ay, ac = a
        bx, by, bc = b
        if float(ac) < min_conf or float(bc) < min_conf:
            continue

        cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)), (0, 255, 0), 2, cv2.LINE_AA)

    # Joint dots
    for v in pose_dict.values():
        if not v or len(v) < 3:
            continue
        x, y, c = v
        if float(c) < min_conf:
            continue
        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1, cv2.LINE_AA)


def build_rep_map(reps: List[Dict[str, Any]]) -> Dict[int, Dict[str, str]]:
    """
    Maps each rep end frame -> overlay label/reason.

    The overlay logic triggers updates only at rep end frames and holds the text
    for a short duration.
    """
    rep_map: Dict[int, Dict[str, str]] = {}

    for rep in reps or []:
        end = rep.get("end_idx", rep.get("end_frame", rep.get("end")))
        if end is None:
            continue
        end = int(end)

        label_raw = str(rep.get("label", rep.get("passed", "fail"))).strip().lower()
        label = "pass" if label_raw == "pass" else "fail"

        reason = (rep.get("reason") or rep.get("fail_reason") or "").strip()
        if not reason:
            reason = "Good rep" if label == "pass" else "Form issue"

        rep_map[end] = {"label": label, "reason": reason}

    return rep_map


def annotate_frame(
    frame,
    frame_index: int,
    exercise_name: str,
    rep_count: int,
    rep_map: Dict[int, Dict[str, str]],
    fps: float,
    font_scale: float = 0.7,
) -> None:
    """
    Renders rep count and the most recent rep result (PASS/FAIL + reason).
    """
    global _last_overlay_text, _last_update_frame

    if frame is None:
        return

    # Main header line
    title = f"{exercise_name.title()} | Reps: {rep_count}"
    draw_text_outlined(frame, title, (30, 50), font_scale=float(font_scale))

    # Update the "last rep result" line only when a rep ends.
    if rep_map and frame_index in rep_map:
        info = rep_map[frame_index]
        label = info.get("label", "fail")
        reason = info.get("reason", "Form issue")

        # ASCII-only: avoids OpenCV font rendering issues.
        prefix = "PASS" if label == "pass" else "FAIL"
        _last_overlay_text = f"{prefix} - {reason}"
        _last_update_frame = int(frame_index)

    # Hold the last result line for N frames (based on FPS).
    hold_frames = max(1, int(round(_display_seconds * float(fps or 30.0))))
    if _last_overlay_text and (int(frame_index) - int(_last_update_frame)) <= hold_frames:
        draw_text_outlined(frame, _last_overlay_text, (30, 92), font_scale=float(font_scale) * 0.95)


def reset_overlay_state() -> None:
    """
    Clears cached overlay state (useful when processing multiple clips in one run).
    """
    global _last_overlay_text, _last_update_frame, _ema_pose
    _last_overlay_text = None
    _last_update_frame = -10_000_000
    _ema_pose = {}
