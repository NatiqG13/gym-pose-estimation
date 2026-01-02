"""
joint_plotter.py
By: Natiq Ghafoor

Plots the tracked joint angle over time and overlays rep regions.

Output:
- joint_angles.png

Inputs:
- angle_sequence: list of smoothed angles in degrees (None allowed)
- rep_feature_list: list of {start, end, label} per rep (frame indices)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_joint_angles(
    angle_sequence: List[Optional[float]],
    rep_feature_list: List[Dict[str, Any]],
    output_dir: str,
    save_name: str = "joint_angles.png",
    title: str = "Joint Angle Over Time",
) -> str:
    """
    Saves a joint-angle time series plot with pass/fail rep shading.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not angle_sequence:
        raise ValueError("angle_sequence is empty")

    x = np.arange(len(angle_sequence))
    y = np.array([np.nan if (v is None or not np.isfinite(v)) else float(v) for v in angle_sequence], dtype=float)

    plt.figure(figsize=(12, 4))
    plt.plot(x, y, linewidth=1.5, label="Joint Angle (Smoothed)")

    # Rep shading: green for pass, red for fail.
    for rep in rep_feature_list or []:
        start = rep.get("start")
        end = rep.get("end")
        if start is None or end is None:
            continue

        try:
            s = int(start)
            e = int(end)
        except Exception:
            continue

        label = str(rep.get("label", "fail")).lower()
        color = "green" if label == "pass" else "red"
        plt.axvspan(s, e, color=color, alpha=0.12)

    plt.title(title)
    plt.xlabel("Frame Index")
    plt.ylabel("Angle (deg)")
    plt.grid(alpha=0.2)
    plt.legend(loc="upper right")
    plt.tight_layout()

    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
