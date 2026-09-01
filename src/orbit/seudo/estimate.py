"""The core SEUDO solver, ported from the `seudo` package's estimate.py
(itself a port of SEUDO's estimateTimeCoursesWithSEUDO.m, 'static' dyn_mod
path only). Each frame is fit independently as

    minimize_{x >= 0}  0.5 * ||A x - frame||^2 + lambda' * x

where A's columns are the (padded, windowed) known cell profiles plus a
bank of Gaussian "blob" basis functions covering every pixel in the window
-- one blob per pixel, meant to absorb activity from unmodeled ("not
found") sources. The frame's cell activation is then taken from whichever
of (a) plain least squares against the known profiles, or (b) this sparse
fit, has lower cost.

Deliberately dropped from the upstream `seudo` package (not a fidelity
gap in the math, just infrastructure orbit-gui doesn't need): the
n_jobs thread-pool option (sequential; orbit-gui already runs every
long computation off the GUI thread via run_worker) -- confirmed by direct
benchmark this session that thread-based per-cell parallelism is actively
harmful at these per-cell-fit sizes here (dispatch overhead dominates),
not just an unneeded convenience.

use_native (bool, default False): the upstream package's own compiled C++
accelerator was ALSO tried this session (built from its vendored source)
and measured slower here at every window size and every l_mode/stop_mode
combination -- its blob-basis convolution is real-space ("stamp the
kernel at each active pixel"), asymptotically worse than FFT-based
convolution at these sizes, with the gap widening as the window grows.
What `use_native` enables instead is a from-scratch port of orbit-gui's
OWN fista_nonneg_weighted_l1 to C++ (see seudo/_native/fista_native.cpp),
using FFTW with the kernel's transform cached per cell window -- same
algorithm, bit-identical results, confirmed ~1.3x faster end-to-end on
real data on top of make_cached_blob_conv's own ~1.6x win in pure Python.
Silently no-ops back to the pure-Python solver if seudo/_native hasn't
been built (see its own build_native.sh) -- never required.

frame_blocks: an optional list of (start, end) 0-indexed INCLUSIVE frame
ranges. When given, only frames within these ranges are processed --
everywhere else in the output arrays stays NaN. This is what makes
running SEUDO tractable on a real movie: restrict it to just the frames
spanned by a cell's detected transients (see run_on_transients.py)
rather than the whole movie.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import irfft2, next_fast_len, rfft2

from . import _native
from .blob import make_seudo_blob
from .geometry import compute_roi_coms
from .solver import fista_nonneg_weighted_l1


def make_cached_blob_conv(kernel: np.ndarray, a_shape: tuple[int, int]):
    """A same-mode "convolve with `kernel`" closure for arrays shaped
    `a_shape`, numerically equivalent to
    ``scipy.signal.fftconvolve(a, kernel, mode="same")`` (confirmed to
    ~1e-14 across the window sizes real data actually produces) but
    2-6x faster at those sizes for repeated calls against the SAME kernel:
    `kernel` (one_blob -- the single fixed Gaussian blob basis function,
    shared by every cell and every frame for the life of a run) never
    changes, yet fftconvolve retransforms it via FFT from scratch on
    every single call regardless. A/At are each called at least once per
    FISTA iteration, per frame, per cell -- profiling a real 3000-frame
    run found fftconvolve alone accounting for ~81% of total wall-clock
    time, the large majority of it this exact redundant kernel transform
    plus scipy.signal's own per-call dispatch overhead (its newer
    array-API compatibility layer adds real per-call cost at these small
    -- ~20x20 to ~55x55 -- window sizes, confirmed by direct microbenchmark
    against scipy.fft's lower-level rfft2/irfft2 used here directly).
    Precomputing the kernel's rfft2 once per window size (the caller's
    job -- typically once per cell-setup, reused across every subsequent
    frame/iteration until that setup is invalidated) is Real-SEUDO's
    single largest real-data performance win found this session."""
    full_shape = (a_shape[0] + kernel.shape[0] - 1, a_shape[1] + kernel.shape[1] - 1)
    fft_shape = tuple(next_fast_len(s) for s in full_shape)
    kernel_fft = rfft2(kernel, s=fft_shape)
    # matches scipy.signal._signaltools._centered's own convention: the
    # full linear-convolution result, centered down to a_shape.
    start = tuple((full_shape[i] - a_shape[i]) // 2 for i in range(2))
    sl = tuple(slice(start[i], start[i] + a_shape[i]) for i in range(2))

    def conv(a: np.ndarray) -> np.ndarray:
        full = irfft2(rfft2(a, s=fft_shape) * kernel_fft, s=fft_shape)[:full_shape[0], :full_shape[1]]
        return full[sl]

    return conv


def _get_frame(movie: np.ndarray, frame_index: int, zero_level: float) -> np.ndarray:
    return movie[:, :, frame_index].astype(float) - zero_level


def _cell_window_bounds(
    prof: np.ndarray, pad_space: int, mov_y: int, mov_x: int, use_com: bool,
) -> tuple[int, int, int, int]:
    """One cell's fit window: either a single pixel at its center of mass
    (use_com) or its full outer bounding box, padded by pad_space and
    clamped to the movie's own extent -- shared by this module's own
    per-cell precompute loop below and streaming.py's online per-cell
    setup, so both use identical window-bounds math."""
    coms, outer_bounds = compute_roi_coms(prof)
    if use_com:
        cy, cx = int(round(coms[0, 1])), int(round(coms[0, 0]))
        y0 = y1 = cy
        x0 = x1 = cx
    else:
        y0, y1, x0, x1 = outer_bounds[0]
    y0 = max(0, int(y0) - pad_space)
    y1 = min(mov_y - 1, int(y1) + pad_space)
    x0 = max(0, int(x0) - pad_space)
    x1 = min(mov_x - 1, int(x1) + pad_space)
    return y0, y1, x0, x1


def _setup_cell_window(
    profiles: np.ndarray, this_cell: int, y0: int, y1: int, x0: int, x1: int, min_pix_for_inclusion: int,
    lambda_prof: float, lambda_blob: float, sigma2_ds: float, p: float, one_blob: np.ndarray,
    use_native: bool = False, native_blob_conv_cache: dict | None = None,
) -> dict:
    """Build the ROI/blob-basis/lambda setup for one cell's window --
    shared by the per-frame loop below and seudo_residual.py's
    per-transient single-image fit, so both use identical math for "what
    SEUDO regresses against" in a given window.

    ``native_blob_conv_cache``, if given, memoizes native BlobConv handles
    by (n_y, n_x): a cell's setup gets rebuilt far more often than "once
    per cell's whole lifetime" (overlap invalidation rebuilds nearby cells'
    setups too -- see streaming.py's _invalidate_overlapping_setups), and
    different cells often share the same window size, so sharing the
    handle across both rebuilds AND cells avoids reconstructing FFTW's
    plans (a real, measured cost -- see fista_native.cpp) far more often
    than necessary. The caller owns the cache's lifetime (one per
    StreamingState, or one per estimate_time_courses_with_seudo call);
    None (the default) always builds a fresh handle, matching the old
    always-rebuild behavior.

    ``use_native``: also build a native (compiled) BlobConv handle for
    this window (see module docstring / seudo/_native) -- a no-op falling
    back to None whenever seudo._native.NATIVE_AVAILABLE is False, so
    passing True is always safe regardless of whether the accelerator has
    been built."""
    n_y = y1 - y0 + 1
    n_x = x1 - x0 + 1
    n_blobs = n_y * n_x

    pix_per_profile = np.sum(profiles[y0 : y1 + 1, x0 : x1 + 1, :] > 0, axis=(0, 1))
    include = pix_per_profile > min_pix_for_inclusion
    include[this_cell] = True

    n_cells_window = int(np.sum(include))
    window_profiles = profiles[y0 : y1 + 1, x0 : x1 + 1, include]
    rois = window_profiles.reshape(n_y * n_x, n_cells_window)

    prof_norms = np.sqrt(np.sum(rois**2, axis=0))

    lambdas0 = np.concatenate([np.full(n_cells_window, lambda_prof), np.full(n_blobs, lambda_blob)])
    k1 = 2 * sigma2_ds * lambdas0
    pos = lambdas0 > 0
    k2 = 2 * sigma2_ds * (np.log(1.0 / p - 1.0) - np.sum(np.log(lambdas0[pos])))

    norm_factors = np.concatenate([prof_norms, np.ones(n_blobs)])
    lambdas = k1 / norm_factors

    cell_index_within = int(np.where(np.flatnonzero(include) == this_cell)[0][0])
    rois_scaled = rois / prof_norms[np.newaxis, :]

    # A cached-kernel convolution (see make_cached_blob_conv) rather than a
    # fresh fftconvolve call every time -- profiling showed fftconvolve
    # alone accounting for ~81% of a real run's wall-clock time (A/At are
    # called at least once per FISTA iteration, per frame, per cell), the
    # bulk of it needlessly re-transforming the SAME fixed one_blob kernel
    # via FFT on every single call. Built once per (n_y, n_x) window here,
    # reused by every subsequent A/At call against this cell's setup.
    blob_conv = make_cached_blob_conv(one_blob, (n_y, n_x))

    def A(z):
        z_cells = z[:n_cells_window]
        z_blob = z[n_cells_window:].reshape(n_y, n_x)
        return rois_scaled @ z_cells + blob_conv(z_blob).ravel()

    def At(v):
        top = rois_scaled.T @ v
        bottom = blob_conv(v.reshape(n_y, n_x)).ravel()
        return np.concatenate([top, bottom])

    # Same caching granularity as blob_conv above, PLUS shared across every
    # other cell/rebuild with the same window size when a cache is given
    # (see this function's own docstring) -- FFTW plan construction is a
    # real, measured cost, confirmed by profiling a real streaming run.
    native_blob_conv = None
    if use_native and _native.NATIVE_AVAILABLE:
        if native_blob_conv_cache is None:
            native_blob_conv = _native.make_native_blob_conv(one_blob, n_y, n_x)
        else:
            key = (n_y, n_x)
            native_blob_conv = native_blob_conv_cache.get(key)
            if native_blob_conv is None:
                native_blob_conv = _native.make_native_blob_conv(one_blob, n_y, n_x)
                native_blob_conv_cache[key] = native_blob_conv

    return dict(
        n_y=n_y, n_x=n_x, rois=rois, rois_scaled=rois_scaled, norm_factors=norm_factors,
        lambdas=lambdas, k1=k1, k2=k2, cell_index_within=cell_index_within,
        n_cells_window=n_cells_window, operators=(A, At), include=include, blob_conv=blob_conv,
        native_blob_conv=native_blob_conv,
    )


def _solve_one_frame_cell(
    this_frame: np.ndarray, rois: np.ndarray, rois_scaled: np.ndarray, lambdas: np.ndarray, norm_factors: np.ndarray,
    k1: np.ndarray, k2: float, n_y: int, n_x: int, blob_conv, operators: tuple, solver_tol: float,
    solver_max_iter: int, native_blob_conv=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Solve one (frame, cell) pair -- independent of every other frame and cell.
    ``blob_conv`` (see make_cached_blob_conv) is the same cached "convolve
    with one_blob" closure A/At already use internally -- reused here for
    bob_cost's own convolution rather than a separate, uncached fftconvolve
    call, whenever the pure-Python solver path is taken.

    ``native_blob_conv``, if given (see seudo/_native), runs the compiled
    C++/FFTW FISTA solver instead of the pure-Python one -- same problem
    (rois_scaled, lambdas), bit-identical results (confirmed by tests/
    test_seudo_native.py), just faster -- and its own .convolve() is reused
    for bob_cost's blob_contrib too, so no separate Python-side convolution
    runs at all once native is active (profiling a real run found this
    leftover Python fftconvolve call was still ~14% of total time even
    with the solve itself already native). None (the default) always
    falls back to the Python path for both."""
    n_cells_window = rois.shape[1]
    tc_lsq_frame = np.linalg.solve(rois.T @ rois, rois.T @ this_frame)

    if native_blob_conv is not None:
        fit_weights, _n_iter = _native.fista_native(
            native_blob_conv, rois_scaled, this_frame, lambdas, solver_tol, solver_max_iter,
        )
    else:
        A, At = operators
        x0 = np.zeros(n_cells_window + n_y * n_x)
        fit_weights = fista_nonneg_weighted_l1(
            A, At, this_frame, lambdas, x0, tol=solver_tol, max_iter=solver_max_iter,
        )
    fit_weights = fit_weights / norm_factors
    fit_x = fit_weights[:n_cells_window]
    fit_bob = fit_weights[n_cells_window:]

    lsq_cost = np.sum((rois @ tc_lsq_frame - this_frame) ** 2)
    if native_blob_conv is not None:
        blob_contrib = native_blob_conv.convolve(fit_bob.reshape(n_y, n_x)).ravel()
    else:
        blob_contrib = blob_conv(fit_bob.reshape(n_y, n_x)).ravel()
    bob_cost = (
        np.sum((this_frame - rois @ fit_x - blob_contrib) ** 2) + np.sum(np.abs(k1 * fit_weights)) - k2
    )

    if lsq_cost < bob_cost:
        fit_fancy = tc_lsq_frame
        fit_x = np.zeros_like(fit_x)
    else:
        fit_fancy = fit_x

    return tc_lsq_frame, fit_fancy, fit_x, lsq_cost, bob_cost


def _merge_frame_blocks(blocks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    blocks = sorted((int(s), int(e)) for s, e in blocks)
    merged: list[tuple[int, int]] = []
    for s, e in blocks:
        if merged and s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def estimate_time_courses_with_seudo(
    movie: np.ndarray,
    profiles: np.ndarray,
    which_cells: np.ndarray | None = None,
    p: float = 1e-5,
    sigma2: float = 0.01,
    lambda_blob: float = 20.0,
    blob_radius: float = 1.2,
    ds_time: int = 1,
    lambda_prof: float = 0.0,
    min_pix_for_inclusion: int = 1,
    pad_space: int = 10,
    use_com: bool = False,
    zero_level: float = 0.0,
    solver_tol: float = 0.01,
    solver_max_iter: int = 1000,
    frame_blocks: list[tuple[int, int]] | None = None,
    progress_callback=None,
    use_native: bool = True,
) -> dict:
    """movie: (Y, X, F) array. profiles: (Y, X, nCellsTotal) array.
    which_cells: 0-indexed cell indices to analyze (default: all).
    progress_callback, if given, is called as progress_callback(frames_done,
    total_frames) each time a frame finishes.

    ``use_native``: use the compiled C++/FFTW FISTA accelerator (see
    module docstring / seudo/_native) when available -- silently falls
    back to the pure-Python solver otherwise, so True (the default) is
    always safe.

    Returns a dict with keys 'tc', 'tc_lsq', 'params', 'extras'.
    """
    profiles = np.asarray(profiles, dtype=float)
    mov_y, mov_x, n_frames = movie.shape
    n_cells_total = profiles.shape[2]

    which_cells = np.arange(n_cells_total) if which_cells is None else np.asarray(which_cells)
    n_cells_analyzed = len(which_cells)

    sigma2_ds = sigma2 / ds_time
    min_pix_for_inclusion = max(min_pix_for_inclusion, 1)
    one_blob = make_seudo_blob(blob_radius)

    # ---- per-cell precomputation ----

    which_pixels = np.zeros((mov_y * mov_x, n_cells_analyzed), dtype=bool)
    which_profiles = np.zeros((n_cells_total, n_cells_analyzed), dtype=bool)
    n_y_list, n_x_list = [], []
    rois_list, rois_scaled_list, norm_factors_list, lambdas_list = [], [], [], []
    k1_list, k2_list, cell_index_within_list, operators, blob_conv_list = [], [], [], [], []
    native_blob_conv_list = []
    native_blob_conv_cache: dict = {}  # (n_y, n_x) -> BlobConv, shared across every cell this call analyzes

    for cc, this_cell in enumerate(which_cells):
        prof = profiles[:, :, this_cell]
        y0, y1, x0, x1 = _cell_window_bounds(prof, pad_space, mov_y, mov_x, use_com)

        pix_mask = np.zeros((mov_y, mov_x), dtype=bool)
        pix_mask[y0 : y1 + 1, x0 : x1 + 1] = True
        which_pixels[:, cc] = pix_mask.ravel()

        setup = _setup_cell_window(
            profiles, this_cell, y0, y1, x0, x1, min_pix_for_inclusion, lambda_prof, lambda_blob, sigma2_ds, p,
            one_blob, use_native, native_blob_conv_cache,
        )
        n_y_list.append(setup["n_y"])
        n_x_list.append(setup["n_x"])
        rois_list.append(setup["rois"])
        k1_list.append(setup["k1"])
        k2_list.append(setup["k2"])
        norm_factors_list.append(setup["norm_factors"])
        lambdas_list.append(setup["lambdas"])
        cell_index_within_list.append(setup["cell_index_within"])
        rois_scaled_list.append(setup["rois_scaled"])
        operators.append(setup["operators"])
        blob_conv_list.append(setup["blob_conv"])
        native_blob_conv_list.append(setup["native_blob_conv"])
        which_profiles[:, cc] = setup["include"]

    # ---- main per-frame loop ----

    tc_seudo = np.full((n_frames, n_cells_analyzed), np.nan)
    tc_lsq = np.full((n_frames, n_cells_analyzed), np.nan)
    lsq_costs = np.full((n_frames, n_cells_analyzed), np.nan)
    bob_costs = np.full((n_frames, n_cells_analyzed), np.nan)
    tc_cells_with_blobs = np.full((n_frames, n_cells_analyzed), np.nan)

    blocks = [(0, n_frames - 1)] if frame_blocks is None else _merge_frame_blocks(frame_blocks)
    n_block_frames = sum(e - s + 1 for s, e in blocks)
    frames_done = 0

    for block_start, block_end in blocks:
        # the ds_time moving average has a genuine order dependency (each
        # frame's average depends on the raw pixels of the preceding
        # ds_time-1 frames) -- reset at the start of each block rather
        # than blending across the (possibly large) gap between blocks.
        sliding_window = np.full((mov_y * mov_x, ds_time), np.nan)
        for ff in range(block_start, block_end + 1):
            local_ff = ff - block_start
            frame_flat = _get_frame(movie, ff, zero_level).reshape(-1)
            if local_ff < ds_time:
                sliding_window[:, local_ff] = frame_flat
            else:
                sliding_window[:, :-1] = sliding_window[:, 1:]
                sliding_window[:, -1] = frame_flat
            frame_full = np.nanmean(sliding_window, axis=1)

            for cc in range(n_cells_analyzed):
                this_frame = frame_full[which_pixels[:, cc]]
                tc_lsq_frame, fit_fancy, fit_x, lsq_cost, bob_cost = _solve_one_frame_cell(
                    this_frame, rois_list[cc], rois_scaled_list[cc], lambdas_list[cc], norm_factors_list[cc],
                    k1_list[cc], k2_list[cc], n_y_list[cc], n_x_list[cc], blob_conv_list[cc], operators[cc],
                    solver_tol, solver_max_iter, native_blob_conv_list[cc],
                )
                idx_within = cell_index_within_list[cc]
                tc_seudo[ff, cc] = fit_fancy[idx_within]
                tc_cells_with_blobs[ff, cc] = fit_x[idx_within]
                tc_lsq[ff, cc] = tc_lsq_frame[idx_within]
                lsq_costs[ff, cc] = lsq_cost
                bob_costs[ff, cc] = bob_cost

            frames_done += 1
            if progress_callback is not None:
                progress_callback(frames_done, n_block_frames)

    params = dict(
        p=p, sigma2=sigma2, lambda_blob=lambda_blob, blob_radius=blob_radius, ds_time=ds_time,
        lambda_prof=lambda_prof, min_pix_for_inclusion=min_pix_for_inclusion, pad_space=pad_space, use_com=use_com,
        which_cells=which_cells, frame_blocks=blocks,
    )
    extras = dict(
        lsq_costs=lsq_costs, bob_costs=bob_costs, tc_cells_with_blobs=tc_cells_with_blobs, one_blob=one_blob,
        which_pixels=which_pixels, which_profiles=which_profiles,
    )
    return dict(tc=tc_seudo, tc_lsq=tc_lsq, params=params, extras=extras)
