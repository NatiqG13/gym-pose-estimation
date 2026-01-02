"""
rep_map.py
By: Natiq Ghafoor

Compatibility shim.

Older code paths import build_rep_map from src.rep_map.
The implementation is maintained in visualization.overlay_renderer.
"""

from visualization.overlay_renderer import build_rep_map

__all__ = ["build_rep_map"]
