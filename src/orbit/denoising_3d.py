"""3D (volumetric) counterparts of orbit.denoising's Gaussian and median
filters -- gaussian_filter/median_filter are already dimension-agnostic
(sigma/size just needs to match the array's ndim), so these are a
straight T-first (T, L, W, D) rewrite of the same two scipy calls, not a
new algorithm: the spatial filtering width applies equally to all three
spatial axes (L, W, D) instead of just two.

Wavelet and PCA denoising aren't generalized here -- neither has a
literal "spatial width" parameter the way Gaussian/median do (wavelet
uses a wavelet family + decomposition level; PCA uses a block tiling),
so they're left as a 2D-only, per-frame/per-pixel operation for now.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


def denoise_gaussian_3d(mov: np.ndarray, spatial_sigma: float = 0.0, temporal_sigma: float = 0.0) -> np.ndarray:
    """Gaussian-smooth a (T, L, W, D) volumetric movie -- spatial_sigma
    applies to all three spatial axes (L, W, D) equally; see
    orbit.denoising.denoise_gaussian for the (H, W, T) 2D version."""
    return gaussian_filter(mov, sigma=(temporal_sigma, spatial_sigma, spatial_sigma, spatial_sigma))


def denoise_median_3d(mov: np.ndarray, space_window: int = 3, time_window: int = 1) -> np.ndarray:
    """Median filter over a (time_window, space_window, space_window,
    space_window) window for a (T, L, W, D) volumetric movie -- see
    orbit.denoising.denoise_median for the (H, W, T) 2D version."""
    return median_filter(mov, size=(time_window, space_window, space_window, space_window), mode="nearest")
