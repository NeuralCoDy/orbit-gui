"""Greedy CNMF initialization, ported from CaImAn's initialization.greedyROI:
repeatedly seed a new component at the brightest pixel of a (spatially
blurred) residual, refine its footprint/trace via a small rank-1 nonnegative
alternating least squares fit, then subtract it out before picking the next
seed. A final low-rank NMF over what's left initializes the background.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import NMF

from ._masks import disk_mask


def gaussian_blur_movie(movie: np.ndarray, sigma: float) -> np.ndarray:
    """Per-frame spatial Gaussian blur (no blurring across time) -- makes
    greedy peak-picking robust to single-pixel noise spikes.

    A 2D Gaussian blur is separable (blur rows, then blur columns), and
    blurring a FIXED axis of the whole (H, W, T) movie at once is itself
    just a linear operator on that axis -- so each pass is one matrix
    multiply against an (H, H) or (W, W) operator, built once by probing
    scipy.ndimage.gaussian_filter1d with an identity matrix (this
    guarantees bit-identical results to calling it directly, without
    hand-deriving its own kernel/boundary-reflection conventions).
    Confirmed bit-identical (~1e-15, floating-point noise) to
    scipy.ndimage.gaussian_filter across sigma in {0, 0.5, 1.0, 2.0, 3.7}
    and non-square field-of-view shapes.

    This replaced an earlier cv2.GaussianBlur-per-frame version (itself
    ~3.6x faster than a single whole-movie scipy.ndimage.gaussian_filter
    call, profiling's original ~38%-of-a-whole-FOV-CNMF-run finding) --
    profiling THAT version found ~76% of its own cost was the transpose
    round-trip cv2 needed (one frame at a time, so time had to become
    the leading axis first and get moved back after), not the blur
    itself. This matmul form needs no transpose at all -- confirmed
    ~2x faster again on the same real (256, 256, 2000) movie (~2.0s vs
    ~4.0s), and slightly LOWER peak memory too (no longer needs a
    transposed working copy on top of the blurred output)."""
    if sigma <= 0:
        return movie.copy()
    height, width, _n_frames = movie.shape
    # mode="reflect" (duplicates the edge pixel: d c b a | a b c d) is
    # gaussian_filter1d's own default too, matching what the whole-movie
    # gaussian_filter call this replaced always used -- named explicitly
    # here anyway, since getting this wrong (e.g. "mirror", which does
    # NOT duplicate the edge pixel) previously gave silently-wrong,
    # not-obviously-wrong-looking results for a similar cv2 border-mode
    # mismatch elsewhere in this codebase's own history.
    row_op = gaussian_filter1d(np.eye(height), sigma, axis=0, mode="reflect")
    col_op = row_op if width == height else gaussian_filter1d(np.eye(width), sigma, axis=0, mode="reflect")
    blurred_rows = np.tensordot(row_op, movie, axes=([1], [0]))  # blur the H axis; result stays (H, W, T)
    # einsum, not a second tensordot, to land directly on (H, W, T) --
    # tensordot's own convention would give (H, T, W) here (the
    # contracted array's un-contracted axes come first, then the other
    # operand's), needing a further transpose(+copy) to fix, defeating
    # the whole point of avoiding a transpose in the first place.
    return np.einsum("hwt,wv->hvt", blurred_rows, col_op, optimize=True)


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
