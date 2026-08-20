"""Greedy CNMF initialization, ported from CaImAn's initialization.greedyROI:
repeatedly seed a new component at the brightest pixel of a (spatially
blurred) residual, refine its footprint/trace via a small rank-1 nonnegative
alternating least squares fit, then subtract it out before picking the next
seed. A final low-rank NMF over what's left initializes the background.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import NMF

from ._masks import disk_mask


def gaussian_blur_movie(movie: np.ndarray, sigma: float) -> np.ndarray:
    """Per-frame spatial Gaussian blur (no blurring across time) -- makes
    greedy peak-picking robust to single-pixel noise spikes."""
    return gaussian_filter(movie, sigma=(sigma, sigma, 0))


def finetune_component(movie: np.ndarray, mask: np.ndarray, n_iter: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Rank-1 nonnegative alternating-least-squares refinement of a
    candidate footprint/trace pair, restricted to ``mask``'s pixels --
    CaImAn's greedyROI "finetune" step. Returns a full (H, W) footprint
    (zero outside ``mask``) and a (T,) trace."""
    rows, cols = np.nonzero(mask)
    patch = np.asarray(movie[rows, cols, :], dtype=np.float64)  # (n_pixels, T)

    trace = np.clip(patch.mean(axis=0), 0, None)
    spatial = np.ones(len(rows))
    for _ in range(n_iter):
        trace_energy = np.dot(trace, trace)
        if trace_energy > 0:
            spatial = np.clip(patch @ trace / trace_energy, 0, None)
        norm = np.linalg.norm(spatial)
        if norm > 0:
            spatial = spatial / norm
        spatial_energy = np.dot(spatial, spatial)
        if spatial_energy > 0:
            trace = np.clip(spatial @ patch / spatial_energy, 0, None)

    footprint = np.zeros(movie.shape[:2])
    footprint[rows, cols] = spatial
    return footprint, trace


def greedy_roi_init(
    movie: np.ndarray, n_components: int, gauss_sigma: float = 2.0, init_radius: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """Seeds ``n_components`` footprints one at a time at the brightest
    remaining pixel of the (blurred) residual's per-pixel energy, refines
    each via finetune_component, and subtracts it before picking the
    next seed. Returns (footprints (K, H, W), traces (K, T)).

    The per-pixel energy map and the residual are both updated only
    where a just-subtracted footprint could have actually changed them
    (its own, typically small, seed disk) rather than recomputed over
    the whole (H, W[, T]) array every iteration -- profiling showed the
    full recompute was this function's dominant cost. footprint is exact
    zero outside seed_mask by construction (see finetune_component), so
    restricting the update to those pixels is algebraically identical to
    the whole-array version, not an approximation."""
    height, width, _n_frames = movie.shape
    residual = gaussian_blur_movie(movie, gauss_sigma)
    energy = (residual**2).sum(axis=2)

    footprints, traces = [], []
    for _ in range(n_components):
        peak = np.unravel_index(np.argmax(energy), energy.shape)
        seed_mask = disk_mask(height, width, peak, init_radius)
        footprint, trace = finetune_component(residual, seed_mask)
        footprints.append(footprint)
        traces.append(trace)

        rows, cols = np.nonzero(seed_mask)
        residual[rows, cols, :] -= footprint[rows, cols, None] * trace[None, :]
        energy[rows, cols] = (residual[rows, cols, :] ** 2).sum(axis=1)

    return np.stack(footprints), np.stack(traces)


def estimate_background(residual_movie: np.ndarray, n_components: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Low-rank NMF background model over whatever's left after the cell
    components are subtracted out -- CaImAn's own greedyROI background
    step. Returns (spatial (H, W, n_bg), temporal (n_bg, T))."""
    height, width, n_frames = residual_movie.shape
    flat = np.clip(residual_movie.reshape(-1, n_frames), 0, None)
    nmf = NMF(n_components=n_components, init="nndsvda", max_iter=200)
    spatial = nmf.fit_transform(flat)  # (H*W, n_bg)
    temporal = nmf.components_  # (n_bg, T)
    return spatial.reshape(height, width, n_components), temporal
