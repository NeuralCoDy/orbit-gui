"""Greedy, mutually-separated peak selection over a 2D score map -- shared
between orbit.qc_traces (picking representative pixels for QC plots) and
orbit.roi_extraction_corr (auto-seeding candidate ROIs from local
correlation peaks).
"""

from __future__ import annotations

import numpy as np


def select_separated_pixels(score_map: np.ndarray, n_points: int, min_dist: float, descending: bool) -> list[tuple[int, int]]:
    """Greedily picks up to ``n_points`` pixels in ranked order of
    ``score_map`` (highest first if ``descending``, else lowest first),
    skipping any candidate closer than ``min_dist`` to an already-picked
    point -- so the result isn't just a cluster of neighboring pixels
    that all happen to share the extreme score."""
    height, width = score_map.shape
    order = np.argsort(score_map, axis=None)
    if descending:
        order = order[::-1]

    selected: list[tuple[int, int]] = []
    for idx in order:
        r, c = divmod(int(idx), width)
        if all(np.hypot(r - sr, c - sc) >= min_dist for sr, sc in selected):
            selected.append((r, c))
            if len(selected) == n_points:
                break
    return selected
