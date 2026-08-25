"""Zeroing out background/uninformative pixels via an auto-thresholded
binary mask -- the same triangle-method auto-threshold pyGraFT's own
"Mask" tab uses (graft.preprocessing.triangle_mask, itself a thin
wrapper around skimage.filters.threshold_triangle). Reimplemented
directly against skimage here rather than depending on graft for it,
since scikit-image is already this project's own dependency and the
wrapper adds nothing beyond what's ported below.
"""

from __future__ import annotations

import numpy as np
from skimage.filters import threshold_triangle


def triangle_mask(projection: np.ndarray) -> np.ndarray:
    """Auto-thresholds a 2D projection (e.g. a movie's mean image) into
    a boolean mask via the triangle method -- a histogram-shape-based
    automatic threshold, well suited to the strongly right-skewed
    pixel-intensity histograms a mostly-background field of view
    typically has (a small bright peak of real signal on a much larger,
    dim background)."""
    thresh = threshold_triangle(projection)
    return projection > thresh


def apply_mask(movie: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zeroes every pixel outside ``mask`` (H, W) across the whole
    (H, W, T) movie."""
    return movie * mask[:, :, None]
