"""
rep_feature_aggregator.py
By: Natiq Ghafoor

Aggregates per-frame feature dictionaries into per-rep summary statistics.

For each numeric key in the frame window, computes:
- mean, std, min, max, delta (last - first)

Also includes:
- rep_duration
- start_frame / end_frame
- label / fail_reason (if present on the rep dict)

Compatibility:
Some older code imports compute_rep_metrics, so this file keeps that wrapper.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np


def _get_start_end(rep: Dict[str, Any]) -> (int, int):
    """
    Resolves a rep's start/end frame indices across supported key names.
    """
    start = rep.get("start")
    end = rep.get("end")

    if start is None or end is None:
        start = rep.get("start_idx", start)
        end = rep.get("end_idx", end)

    start = int(start) if start is not None else 0
    end = int(end) if end is not None else start

    if end < start:
        start, end = end, start

    return start, end


def aggregate_features(
    rep: Dict[str, Any],
    frame_features: List[Dict[str, Any]],
    fps: float = 30.0,
) -> Dict[str, float]:
    """
    Aggregates numeric per-frame features over a single rep window.
    """
    start, end = _get_start_end(rep)
    end = min(end, max(0, len(frame_features) - 1))

    window = frame_features[start : end + 1] if frame_features else []
    result: Dict[str, float] = {}

    # Collect numeric series per feature key.
    series_by_key: Dict[str, List[float]] = {}
    for ff in window:
        for k, v in (ff or {}).items():
            if isinstance(v, (int, float, np.number)):
                series_by_key.setdefault(k, []).append(float(v))

    # Compute summary stats for each key.
    for k, values in series_by_key.items():
        if not values:
            continue
        arr = np.array(values, dtype=float)
        result[f"{k}_mean"] = float(np.mean(arr))
        result[f"{k}_std"] = float(np.std(arr))
        result[f"{k}_min"] = float(np.min(arr))
        result[f"{k}_max"] = float(np.max(arr))
        result[f"{k}_delta"] = float(arr[-1] - arr[0])

    # Rep window metadata.
    result["rep_duration"] = float((end - start + 1) / fps) if fps else 0.0
    result["start_frame"] = int(start)
    result["end_frame"] = int(end)

    # Label/reason are carried through for downstream plots and CSV export.
    result["label"] = rep.get("label", "fail")
    fr = (rep.get("fail_reason") or rep.get("reason") or "").strip()
    result["fail_reason"] = fr if fr else "Form issue"

    if rep.get("rep_index") is not None:
        result["rep_index"] = rep.get("rep_index")

    return result


def compute_rep_metrics(
    rep_or_reps: Union[Dict[str, Any], List[Dict[str, Any]]],
    frame_features: List[Dict[str, Any]],
    fps: float = 30.0,
):
    """
    Backwards-compatible wrapper.

    - If rep_or_reps is a dict -> returns a single aggregated dict.
    - If rep_or_reps is a list -> returns a list of aggregated dicts.
    """
    if isinstance(rep_or_reps, list):
        return [aggregate_features(r, frame_features, fps=fps) for r in rep_or_reps]
    return aggregate_features(rep_or_reps, frame_features, fps=fps)
