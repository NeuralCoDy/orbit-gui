"""Zeroing out background/uninformative pixels via a thresholded binary
mask. ``triangle_mask`` replicates pyGraFT's own "Mask" tab default
(graft.preprocessing.triangle_mask, itself a thin wrapper around
skimage.filters.threshold_triangle) -- reimplemented directly against
skimage here rather than depending on graft for it, since scikit-image
is already this project's own dependency. ``otsu_mask``/``manual_mask``/
``percentile_mask`` are further options exposed the same way, for data
whose histogram shape doesn't suit the triangle method well, or where
direct user control is preferable to any auto-threshold.

Every function here takes/returns a projection of any dimensionality
(2D for the normal pipeline, 3D for volumetric) -- none do any
shape-specific unpacking, so no separate "_3d" variants are needed
(unlike apply_mask, which is genuinely axis-order-specific)."""

from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu, threshold_triangle


def triangle_mask(projection: np.ndarray) -> np.ndarray:
    """Auto-thresholds a projection (e.g. a movie's mean image) into a
    boolean mask via the triangle method -- a histogram-shape-based
    automatic threshold, well suited to the strongly right-skewed
    pixel-intensity histograms a mostly-background field of view
    typically has (a small bright peak of real signal on a much larger,
    dim background)."""
    thresh = threshold_triangle(projection)
    return projection > thresh


def otsu_mask(projection: np.ndarray) -> np.ndarray:
    """Auto-thresholds via Otsu's method -- maximizes the between-class
    intensity variance, a better fit than the triangle method for a
    projection whose histogram is closer to two comparably-sized modes
    (foreground/background) than a small bright peak on a big dim tail."""
    thresh = threshold_otsu(projection)
    return projection > thresh


def manual_mask(projection: np.ndarray, threshold: float) -> np.ndarray:
    """A user-supplied, fixed intensity threshold -- for data where none
    of the automatic methods land where the user wants, or a known
    threshold value from a prior run should be reused exactly."""
    return projection > threshold


def percentile_mask(projection: np.ndarray, percentile: float) -> np.ndarray:
    """Keeps the brightest ``percentile`` percent of pixels (0-100) --
    e.g. ``percentile=10`` keeps the top 10% most intense pixels,
    regardless of the projection's absolute intensity scale."""
    thresh = np.percentile(projection, 100.0 - percentile)
    return projection > thresh


_MASK_FUNCS = {
    "triangle": lambda projection, **_kwargs: triangle_mask(projection),
    "otsu": lambda projection, **_kwargs: otsu_mask(projection),
    "manual": lambda projection, **kwargs: manual_mask(projection, kwargs["threshold"]),
    "percentile": lambda projection, **kwargs: percentile_mask(projection, kwargs["percentile"]),
}


def compute_mask(projection: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """Dispatches to the mask function named by ``method`` -- the single
    entry point orbitapp.tabs.mask_tab calls, so the choice of method is
    a parameter, not a different function to import. ``kwargs`` holds
    whichever of ``threshold``/``percentile`` that method needs."""
    if method not in _MASK_FUNCS:
        raise ValueError(f"Unknown mask method: {method!r} (expected one of {sorted(_MASK_FUNCS)})")
    return _MASK_FUNCS[method](projection, **kwargs)


def apply_mask(movie: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zeroes every pixel outside ``mask`` (H, W) across the whole
    (H, W, T) movie."""
    return movie * mask[:, :, None]


def apply_mask_3d(movie: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zeroes every voxel outside ``mask`` (L, W, D) across the whole
    (T, L, W, D) volumetric movie -- T-first counterpart of apply_mask.
    triangle_mask itself needs no such counterpart: threshold_triangle
    has no shape assumptions, so it already works unchanged on a genuine
    3D (L, W, D) projection."""
    return movie * mask[None, :, :, :]
