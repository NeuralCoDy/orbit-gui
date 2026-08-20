"""Auto-picks a handful of representative pixel locations from a movie's
local correlation map, for qualitative before/after sanity checks in any
stage tab (Denoising's traces, Normalization's value distributions,
...) -- shared here rather than re-picked per tab.
"""

from __future__ import annotations

import numpy as np

from ._peak_picking import select_separated_pixels
from .projections import local_correlation_projection


def qc_trace_samples(
    before: np.ndarray, after: np.ndarray, n_peaks: int = 2, n_low: int = 2, min_separation_frac: float = 0.2
) -> list[dict]:
    """Picks representative pixel locations from ``before``'s local
    correlation map -- ``n_peaks`` correlation maxima (likely cell
    footprints) and ``n_low`` correlation minima (likely background/
    noise), each set mutually separated by at least
    ``min_separation_frac`` of the field of view's longer dimension --
    and returns each location's before/after time trace, for a
    qualitative sanity check alongside the aggregate metrics."""
    corr_map = local_correlation_projection(before)
    min_dist = min_separation_frac * max(corr_map.shape)
    locations = [("peak", r, c) for r, c in select_separated_pixels(corr_map, n_peaks, min_dist, descending=True)]
    locations += [("low", r, c) for r, c in select_separated_pixels(corr_map, n_low, min_dist, descending=False)]
    return [
        {
            "kind": kind,
            "row": r,
            "col": c,
            "corr": float(corr_map[r, c]),
            "before": np.asarray(before[r, c, :], dtype=float),
            "after": np.asarray(after[r, c, :], dtype=float),
        }
        for kind, r, c in locations
    ]
