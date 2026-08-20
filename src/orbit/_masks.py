"""Shared mask-based trace extraction -- used by every ROI source
(correlation-based, PCA-ICA, CNMF) and by neuropil ring computation.
"""

from __future__ import annotations

import numpy as np

EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)  # scipy.ndimage label/binary_dilation structure, shared by every ROI post-processing step


def masked_mean_trace(movie: np.ndarray, mask: np.ndarray, frame_limit: int | None = None) -> np.ndarray:
    """Per-frame mean of ``movie`` over ``mask``'s True pixels; zeros if
    the mask is empty."""
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        n_frames = movie.shape[2] if frame_limit is None else min(frame_limit, movie.shape[2])
        return np.zeros(n_frames)
    frames = movie[rows, cols, :frame_limit]
    return np.asarray(frames, dtype=np.float64).mean(axis=0)


def disk_mask(height: int, width: int, center: tuple[float, float], radius: float) -> np.ndarray:
    """Boolean disk of ``radius`` around ``center`` on a (height, width)
    grid -- shared by correlation-based seed growth and CNMF's search-
    radius restricted spatial update."""
    rr, cc = np.ogrid[:height, :width]
    d2 = (rr - center[0]) ** 2 + (cc - center[1]) ** 2
    return d2 <= radius**2
