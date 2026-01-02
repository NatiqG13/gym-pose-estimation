"""
form_evaluator.py
By: Natiq Ghafoor

Rep-level form evaluator based on joint angle thresholds.

Given:
- angle_sequence (radians)
- reps as (start_frame, end_frame)

For each rep:
- Converts the segment to degrees
- Computes min angle, max angle, and ROM
- Applies configurable thresholds and returns pass/fail with a short reason
"""

from __future__ import annotations

import math


class FormEvaluator:
    """
    Evaluates segmented reps using simple angle-based rules.
    """

    def __init__(self, angle_sequence, reps):
        """
        Args:
          angle_sequence: List[float] joint angles in radians
          reps: List[(start_frame, end_frame)]
        """
        self.angle_sequence = angle_sequence
        self.reps = reps

    def evaluate_reps(self, config=None):
        """
        Evaluates each rep using min angle, max angle, and ROM thresholds.

        config keys (degrees):
          - min_angle
          - max_angle
          - rom_threshold

        Returns:
          List[(status, reason)] where status is "Passed." or "Failed."
        """
        output = []

        default_config = {
            "min_angle": 60,
            "max_angle": 130,
            "rom_threshold": 80,
        }

        # Apply defaults without mutating the caller’s dict unexpectedly.
        if config is None:
            config = dict(default_config)
        else:
            for key, value in default_config.items():
                config.setdefault(key, value)

        for start_frame, end_frame in self.reps:
            # Slice angles for this rep, then convert to degrees for thresholding.
            segment = self.angle_sequence[start_frame : end_frame + 1]
            segment_deg = [math.degrees(a) for a in segment]

            min_angle = min(segment_deg)
            max_angle = max(segment_deg)
            rom = max_angle - min_angle

            fail_reasons = []

            # Depth check (smaller angle typically means more bend/depth).
            if min_angle > config["min_angle"]:
                fail_reasons.append("not deep enough")

            # Lockout/top check.
            if max_angle < config["max_angle"]:
                fail_reasons.append("below lockout")

            # ROM check (overall movement quality).
            if rom < config["rom_threshold"]:
                fail_reasons.append("below ROM")

            if fail_reasons:
                output.append(("Failed.", fail_reasons[0]))
            else:
                output.append(("Passed.", "Full ROM"))

        return output
