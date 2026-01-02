"""
timeline_plotter.py
By: Natiq Ghafoor

Rep timeline visualization (each rep is a horizontal bar).

Output:
- rep_timeline.png

Inputs (per rep in rep_feature_list):
- rep_index (1-based)
- start, end (frame indices)
- label ("pass"/"fail")
- rom (deg)
- duration (s)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def plot_rep_timeline(
    rep_feature_list: List[Dict[str, Any]],
    fps: float,
    output_dir: str,
    save_name: str = "rep_timeline.png",
    title: str = "Rep Timeline",
) -> str:
    """
    Saves a timeline chart showing start/end times for each rep.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not rep_feature_list:
        raise ValueError("rep_feature_list is empty")

    reps = sorted(rep_feature_list, key=lambda r: int(r.get("rep_index", 0)))

    y: List[int] = []
    left: List[float] = []
    width: List[float] = []
    colors: List[str] = []
    labels: List[str] = []

    fps_val = float(fps or 30.0)

    for rep in reps:
        idx = int(rep.get("rep_index", 0))
        s = int(rep.get("start", 0))
        e = int(rep.get("end", 0))

        # Convert frames to seconds for the horizontal axis.
        start_s = s / fps_val
        dur_s = max(0.0, (e - s) / fps_val)

        y.append(idx)
        left.append(start_s)
        width.append(dur_s)

        lab = str(rep.get("label", "fail")).lower()
        colors.append("green" if lab == "pass" else "red")

        rom = rep.get("rom", "")
        dur = rep.get("duration", "")
        if isinstance(rom, (int, float)) and isinstance(dur, (int, float)):
            labels.append(f"ROM {float(rom):.0f} deg | {float(dur):.2f}s")
        else:
            labels.append("")

    plt.figure(figsize=(12, max(3, 0.45 * len(reps))))
    plt.barh(y, width, left=left, height=0.35, color=colors)

    # Place text to the right of each bar.
    for yi, le, wi, txt in zip(y, left, width, labels):
        if txt:
            plt.text(le + wi + 0.05, yi, txt, va="center", fontsize=9)

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Rep #")
    plt.yticks(y, [str(v) for v in y])
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()

    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
