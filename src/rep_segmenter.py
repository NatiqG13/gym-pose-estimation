"""
rep_segmenter.py
By: Natiq Ghafoor

Segments reps from a joint angle sequence (degrees).

High-level behavior
- Bench: detects a rep as TOP -> BOTTOM -> TOP.
  Includes handling for partial lockout cases so reps are not silently dropped.
- Curl/Squat: arms from bottom and ends the rep near the peak after reaching the top.

Output format
- List of rep dicts containing:
  start_idx, end_idx, rom, duration, label, reason
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def segment_reps_from_angle(angle_sequence, calibration, fps=30, debug=False, exercise="generic"):
    reps: List[Dict[str, Any]] = []
    exercise = (exercise or "").lower()
    cfg = calibration or {}

    # Thresholds (degrees) and timing limits (seconds).
    min_angle = float(cfg.get("min_angle", 30))
    max_angle = float(cfg.get("max_angle", 170))
    rom_threshold = float(cfg.get("rom_threshold", 40))
    min_duration = float(cfg.get("min_duration", 0.2))
    max_duration = float(cfg.get("max_duration", 10.0))

    # Segmentation tuning.
    top_margin = float(cfg.get("top_margin", 15))
    bottom_margin = float(cfg.get("bottom_margin", 15))
    refractory_frames = int(cfg.get("refractory_frames", max(1, int(fps * 0.25))))
    hold_start_frames = int(cfg.get("hold_start_frames", 2))

    # Bench tuning.
    hold_end_frames = int(cfg.get("hold_end_frames", 2))
    peak_drop = float(cfg.get("peak_drop", 6))
    allow_partial_lockout = bool(cfg.get("allow_partial_lockout", True))
    asc_timeout_s = float(cfg.get("asc_timeout_s", max_duration * 1.25))
    asc_timeout_frames = int(max(1, asc_timeout_s * fps))

    # Curl/squat tuning.
    bottom_arm_margin = float(cfg.get("bottom_arm_margin", 25))
    rise_delta = float(cfg.get("rise_delta", 10))
    hold_end_frames_cs = int(cfg.get("hold_end_frames", 2))
    seg_rom_min = float(cfg.get("seg_rom_min", 0))

    # Bench: top -> bottom -> top (full rep)
    # Also emits an incomplete rep as fail when lockout is not reached,
    # which prevents "missing reps" caused by occlusion/jitter near the top.

    if exercise == "bench":
        state = "WAIT_TOP"
        start_idx: Optional[int] = None
        bottom_idx: Optional[int] = None

        top_hold = 0
        bottom_hold = 0
        last_rep_end = -999999

        # Tracks the highest angle during ascent, used for peak-drop and timeouts.
        max_seen = -9999.0
        peak_idx: Optional[int] = None

        for i, angle in enumerate(angle_sequence):
            if angle is None:
                continue

            # Refractory window prevents immediate double-counting.
            if (i - last_rep_end) < refractory_frames:
                continue

            top_thr = max_angle - top_margin
            bottom_thr = min_angle + bottom_margin

            if state == "WAIT_TOP":
                # Arm the start once the angle stays near the top for a few frames.
                if angle >= top_thr:
                    top_hold += 1
                    if top_hold >= hold_start_frames:
                        start_idx = i - hold_start_frames + 1
                        state = "DESC"
                        bottom_idx = None
                        bottom_hold = 0
                        max_seen = angle
                        peak_idx = i
                        if debug:
                            print(f"[DEBUG] Armed at TOP (frame {start_idx}) angle={angle:.1f}°")
                else:
                    top_hold = 0

            elif state == "DESC":
                # Confirm bottom with a small hold to reduce false triggers.
                if angle <= bottom_thr:
                    bottom_hold += 1
                    if bottom_hold >= 2:
                        bottom_idx = i
                        state = "ASC"
                        top_hold = 0
                        max_seen = angle
                        peak_idx = i
                        if debug:
                            print(f"[DEBUG] Reached BOTTOM (frame {bottom_idx}) angle={angle:.1f}°")
                else:
                    bottom_hold = 0

            elif state == "ASC":
                # Track the peak as the arm extends.
                if angle > max_seen:
                    max_seen = angle
                    peak_idx = i

                # Full lockout path.
                if angle >= top_thr:
                    top_hold += 1
                    if top_hold >= hold_end_frames:
                        end_idx = i
                        _finalize_bench_rep(
                            reps=reps,
                            angle_sequence=angle_sequence,
                            start_idx=start_idx,
                            end_idx=end_idx,
                            fps=fps,
                            rom_threshold=rom_threshold,
                            min_duration=min_duration,
                            max_duration=max_duration,
                            max_angle=max_angle,
                            lockout_margin=float(cfg.get("lockout_margin", 15)),
                            seg_rom_min=seg_rom_min,
                            debug=debug,
                        )
                        state = "WAIT_TOP"
                        start_idx = None
                        bottom_idx = None
                        top_hold = 0
                        bottom_hold = 0
                        last_rep_end = end_idx
                        max_seen = -9999.0
                        peak_idx = None
                else:
                    top_hold = 0

                    # Partial lockout handling:
                    # - If ascent reaches a peak and then drops by peak_drop, finalize at the peak.
                    if (
                        allow_partial_lockout
                        and start_idx is not None
                        and bottom_idx is not None
                        and peak_idx is not None
                    ):
                        if (i - bottom_idx) >= max(2, int(0.15 * fps)):
                            if angle <= (max_seen - peak_drop):
                                end_idx = peak_idx
                                _finalize_bench_rep(
                                    reps=reps,
                                    angle_sequence=angle_sequence,
                                    start_idx=start_idx,
                                    end_idx=end_idx,
                                    fps=fps,
                                    rom_threshold=rom_threshold,
                                    min_duration=min_duration,
                                    max_duration=max_duration,
                                    max_angle=max_angle,
                                    lockout_margin=float(cfg.get("lockout_margin", 15)),
                                    seg_rom_min=seg_rom_min,
                                    debug=debug,
                                )
                                state = "WAIT_TOP"
                                start_idx = None
                                bottom_idx = None
                                top_hold = 0
                                bottom_hold = 0
                                last_rep_end = end_idx
                                max_seen = -9999.0
                                peak_idx = None
                                continue

                        # Timeout handling:
                        # If ascent drags on too long without lockout, finalize as FAIL at the best peak.
                        if (i - bottom_idx) >= asc_timeout_frames:
                            end_idx = peak_idx if peak_idx is not None else i
                            _finalize_bench_rep(
                                reps=reps,
                                angle_sequence=angle_sequence,
                                start_idx=start_idx,
                                end_idx=end_idx,
                                fps=fps,
                                rom_threshold=rom_threshold,
                                min_duration=min_duration,
                                max_duration=max_duration,
                                max_angle=max_angle,
                                lockout_margin=float(cfg.get("lockout_margin", 15)),
                                seg_rom_min=seg_rom_min,
                                debug=debug,
                                force_reason="Incomplete rep (timeout / no lockout)",
                            )
                            state = "WAIT_TOP"
                            start_idx = None
                            bottom_idx = None
                            top_hold = 0
                            bottom_hold = 0
                            last_rep_end = end_idx
                            max_seen = -9999.0
                            peak_idx = None

        # Video ended mid-rep; finalize as FAIL so the segment is still visible in outputs.
        if state in {"DESC", "ASC"} and start_idx is not None:
            end_idx = peak_idx if peak_idx is not None else (len(angle_sequence) - 1)
            _finalize_bench_rep(
                reps=reps,
                angle_sequence=angle_sequence,
                start_idx=start_idx,
                end_idx=end_idx,
                fps=fps,
                rom_threshold=rom_threshold,
                min_duration=min_duration,
                max_duration=max_duration,
                max_angle=max_angle,
                lockout_margin=float(cfg.get("lockout_margin", 15)),
                seg_rom_min=seg_rom_min,
                debug=debug,
                force_reason="Incomplete rep (video ended before lockout)",
            )

        if debug:
            clean = [a for a in angle_sequence if a is not None]
            if clean:
                print(f"[DEBUG] Angle range seen: min={min(clean):.1f}°, max={max(clean):.1f}°")

        return reps

    # curl / squat: arm from bottom -> rise -> finalize near peak
    armed_from_bottom = False
    direction = None
    start_idx = None

    min_seen = 9999.0
    max_seen = -9999.0
    peak_idx = None
    last_rep_end = -999999

    for i, angle in enumerate(angle_sequence):
        if angle is None:
            continue

        if (i - last_rep_end) < refractory_frames:
            continue

        top_thr = max_angle - top_margin
        bottom_arm_thr = min_angle + bottom_arm_margin

        # Arm the rep once the angle reaches the bottom region.
        if not armed_from_bottom:
            if angle <= bottom_arm_thr:
                armed_from_bottom = True
                direction = None
                start_idx = i
                min_seen = angle
                max_seen = angle
                peak_idx = None
                if debug:
                    print(f"[DEBUG] Armed from BOTTOM (frame {start_idx}) angle={angle:.1f}°")
            continue

        # Track local min/max and a candidate peak index during the rise.
        if angle < min_seen:
            min_seen = angle
        if angle > max_seen:
            max_seen = angle
            peak_idx = i

        # Require a minimum rise before treating movement as an active rep.
        if direction is None:
            if angle >= (min_seen + rise_delta):
                direction = "up"
        else:
            # Finalize near the peak after reaching the top threshold and dropping by peak_drop.
            if direction == "up" and angle >= top_thr:
                if peak_idx is not None and angle <= (max_seen - peak_drop):
                    end_idx = peak_idx
                    _finalize_cs_rep(
                        reps=reps,
                        angle_sequence=angle_sequence,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        fps=fps,
                        rom_threshold=rom_threshold,
                        min_duration=min_duration,
                        max_duration=max_duration,
                        max_angle=max_angle,
                        lockout_margin=float(cfg.get("lockout_margin", 15)),
                        seg_rom_min=seg_rom_min,
                        debug=debug,
                        hold_end_frames=hold_end_frames_cs,
                    )
                    armed_from_bottom = False
                    last_rep_end = end_idx

    if debug:
        clean = [a for a in angle_sequence if a is not None]
        if clean:
            print(f"[DEBUG] Angle range seen: min={min(clean):.1f}°, max={max(clean):.1f}°")

    return reps


def _finalize_bench_rep(
    reps: List[Dict[str, Any]],
    angle_sequence,
    start_idx: int,
    end_idx: int,
    fps: float,
    rom_threshold: float,
    min_duration: float,
    max_duration: float,
    max_angle: float,
    lockout_margin: float,
    seg_rom_min: float,
    debug: bool,
    force_reason: Optional[str] = None,
):
    seg = [a for a in angle_sequence[start_idx : end_idx + 1] if a is not None]
    if not seg:
        return

    rom = float(max(seg) - min(seg))
    if seg_rom_min and rom < seg_rom_min:
        return

    duration = (end_idx - start_idx + 1) / float(fps) if fps else 0.0
    rep_max_angle = float(max(seg))

    label, reason = classify_rep(
        rom,
        duration,
        rom_threshold,
        min_duration,
        max_duration,
        max_angle,
        rep_max_angle,
        lockout_margin,
    )

    # Forced reasons are used for timeout/end-of-video cases.
    if force_reason:
        label = "fail"
        reason = force_reason

    reps.append(
        {
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "rom": rom,
            "duration": duration,
            "label": label,
            "reason": reason,
        }
    )

    if debug:
        print(f"[DEBUG] Rep {len(reps)}: ROM={rom:.1f}°, Duration={duration:.2f}s → {label.upper()} ({reason})")


def _finalize_cs_rep(
    reps: List[Dict[str, Any]],
    angle_sequence,
    start_idx: int,
    end_idx: int,
    fps: float,
    rom_threshold: float,
    min_duration: float,
    max_duration: float,
    max_angle: float,
    lockout_margin: float,
    seg_rom_min: float,
    debug: bool,
    hold_end_frames: int,
):
    seg = [a for a in angle_sequence[start_idx : end_idx + 1] if a is not None]
    if not seg:
        return

    rom = float(max(seg) - min(seg))
    if seg_rom_min and rom < seg_rom_min:
        return

    duration = (end_idx - start_idx + 1) / float(fps) if fps else 0.0
    rep_max_angle = float(max(seg))

    label, reason = classify_rep(
        rom,
        duration,
        rom_threshold,
        min_duration,
        max_duration,
        max_angle,
        rep_max_angle,
        lockout_margin,
    )

    reps.append(
        {
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "rom": rom,
            "duration": duration,
            "label": label,
            "reason": reason,
        }
    )

    if debug:
        print(f"[DEBUG] Rep {len(reps)}: ROM={rom:.1f}°, Duration={duration:.2f}s → {label.upper()} ({reason})")


def classify_rep(
    rom,
    duration,
    rom_threshold,
    min_duration,
    max_duration,
    max_angle=None,
    rep_max_angle=None,
    lockout_margin=15,
):
    """
    Pass/fail checks applied during segmentation.
    """
    if rom < rom_threshold * 0.9:
        return "fail", "Low ROM"

    if duration < min_duration:
        return "fail", "Too fast"

    if duration > max_duration:
        return "fail", "Too slow"

    if max_angle is not None and rep_max_angle is not None:
        try:
            if float(rep_max_angle) < float(max_angle) - float(lockout_margin):
                return "fail", "No lockout"
        except Exception:
            pass

    return "pass", "Good rep"
