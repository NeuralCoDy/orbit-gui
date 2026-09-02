"""Greedy CNMF initialization, ported from CaImAn's initialization.greedyROI:
repeatedly seed a new component at the brightest pixel of a (spatially
blurred) residual, refine its footprint/trace via a small rank-1 nonnegative
alternating least squares fit, then subtract it out before picking the next
seed. A final low-rank NMF over what's left initializes the background.
"""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.decomposition import NMF

from ._masks import disk_mask

# scipy.ndimage.gaussian_filter's own default truncation radius (its
# `truncate` parameter): kernel half-width = truncate*sigma, rounded like
# scipy's own `int(truncate * sd + 0.5)` -- matched here so the cv2 kernel
# below is the exact same size/shape scipy would have used.
_GAUSSIAN_TRUNCATE = 4.0


def gaussian_blur_movie(movie: np.ndarray, sigma: float) -> np.ndarray:
    """Per-frame spatial Gaussian blur (no blurring across time) -- makes
    greedy peak-picking robust to single-pixel noise spikes.

    cv2.GaussianBlur per frame, not scipy.ndimage.gaussian_filter: ~3.6x
    faster on a real (256, 256, 2000) movie (profiling found this the
    single largest cost in a CNMF/CNMF-E run, ~38% of a whole-FOV CNMF
    call), confirmed bit-identical (~1e-15, floating-point noise) across
    sigma in {0, 0.5, 1.0, 2.0, 3.7} and non-square field-of-view shapes --
    once matched to scipy's own kernel size (_GAUSSIAN_TRUNCATE) AND
    border mode: scipy's default ``mode='reflect'`` duplicates the edge
    pixel (``d c b a | a b c d``), which is cv2's ``BORDER_REFLECT``, NOT
    the more commonly reached-for ``BORDER_REFLECT_101`` (which doesn't
    duplicate it and gives visibly different, wrong results here)."""
    if sigma <= 0:
        return movie.copy()
    radius = int(_GAUSSIAN_TRUNCATE * sigma + 0.5)
    ksize = 2 * radius + 1
    # (T, H, W): cv2.GaussianBlur only blurs one 2D frame at a time, so the
    # loop below is over the movie's OWN first axis -- moving time there
    # first, rather than looping over movie[:, :, i] slices of the
    # original (H, W, T) layout, keeps each frame contiguous for cv2.
    movie_thw = np.ascontiguousarray(np.transpose(movie, (2, 0, 1)))
    blurred_thw = np.empty_like(movie_thw)
    for i in range(movie_thw.shape[0]):
        cv2.GaussianBlur(
            movie_thw[i], (ksize, ksize), sigmaX=sigma, sigmaY=sigma, dst=blurred_thw[i],
            borderType=cv2.BORDER_REFLECT,
        )
    return np.ascontiguousarray(np.transpose(blurred_thw, (1, 2, 0)))


def _rank1_als(patch: np.ndarray, n_iter: int) -> tuple[np.ndarray, np.ndarray]:
    """Alternating nonnegative least-squares rank-1 factorization of
    ``patch`` (n_pixels, T): a ``spatial`` (n_pixels,) x ``trace`` (T,)
    pair minimizing ``||patch - outer(spatial, trace)||``, both clipped
    nonnegative each step -- CaImAn's greedyROI "finetune" step. Shared
    by finetune_component (a small masked patch per cell component) and
    estimate_background's n_components=1 fast path (the whole
    flattened residual at once) -- same ALS, applied at different
    scales."""
    trace = np.clip(patch.mean(axis=0), 0, None)
    spatial = np.ones(patch.shape[0])
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
    return spatial, trace


def finetune_component(movie: np.ndarray, mask: np.ndarray, n_iter: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Rank-1 nonnegative alternating-least-squares refinement of a
    candidate footprint/trace pair, restricted to ``mask``'s pixels.
    Returns a full (H, W) footprint (zero outside ``mask``) and a (T,)
    trace."""
    rows, cols = np.nonzero(mask)
    patch = np.asarray(movie[rows, cols, :], dtype=np.float64)  # (n_pixels, T)
    spatial, trace = _rank1_als(patch, n_iter)

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
    step. Returns (spatial (H, W, n_bg), temporal (n_bg, T)).

    Clips ``residual_movie`` to nonnegative in place (reshape is a view,
    so this mutates the caller's array too) rather than allocating a
    fresh clipped (H, W, T) copy -- callers of this internal helper
    don't need their residual afterward, and this movie-sized array is
    typically the single biggest temporary in a CNMF run.

    n_components=1 -- this function's default, and the only value the
    GUI exposes -- uses the same rank-1 ALS finetune_component uses per
    cell component instead of sklearn's NMF: sklearn's fit was the
    single largest cost in a whole-FOV CNMF run, a rank-1-specific
    non-convergence pathology in its coordinate-descent solver (never
    converges within max_iter=200 for rank 1, vs. 3-4 iterations for
    rank 2-3 on equivalent data) -- the ALS reaches the same
    reconstruction quality in ~1 iteration instead. n_components > 1
    (not reachable from the GUI) still uses sklearn's NMF: a
    from-scratch ALS generalization was tried and measured slower at
    this scale, since sklearn's own convergence isn't the bottleneck for
    rank >1 the way it was for rank 1."""
    height, width, n_frames = residual_movie.shape
    flat = residual_movie.reshape(-1, n_frames)
    np.clip(flat, 0, None, out=flat)
    if n_components == 1:
        spatial, temporal = _rank1_als(flat, n_iter=10)
        spatial, temporal = spatial[:, None], temporal[None, :]
    else:
        nmf = NMF(n_components=n_components, init="nndsvda", max_iter=200)
        spatial = nmf.fit_transform(flat)  # (H*W, n_bg)
        temporal = nmf.components_  # (n_bg, T)
    return spatial.reshape(height, width, n_components), temporal
