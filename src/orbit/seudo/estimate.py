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
compiled C++ accelerator (pure-Python FISTA solver only) and the
n_jobs thread-pool option (sequential; orbit-gui already runs every
long computation off the GUI thread via run_worker).

frame_blocks: an optional list of (start, end) 0-indexed INCLUSIVE frame
ranges. When given, only frames within these ranges are processed --
everywhere else in the output arrays stays NaN. This is what makes
running SEUDO tractable on a real movie: restrict it to just the frames
spanned by a cell's detected transients (see run_on_transients.py)
rather than the whole movie.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from .blob import make_seudo_blob
from .geometry import compute_roi_coms
from .solver import fista_nonneg_weighted_l1


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
) -> dict:
    """Build the ROI/blob-basis/lambda setup for one cell's window --
    shared by the per-frame loop below and seudo_residual.py's
    per-transient single-image fit, so both use identical math for "what
    SEUDO regresses against" in a given window."""
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

    # fftconvolve rather than the spatial-domain convolve2d -- profiling
    # showed the latter was the dominant cost of a SEUDO run (A/At are
    # called at least once per FISTA iteration, per frame, per cell); the
    # two are numerically equivalent (confirmed to ~1e-14) but
    # fftconvolve is 2-6x faster at the window sizes this solves.
    def A(z):
        z_cells = z[:n_cells_window]
        z_blob = z[n_cells_window:].reshape(n_y, n_x)
        return rois_scaled @ z_cells + fftconvolve(z_blob, one_blob, mode="same").ravel()

    def At(v):
        top = rois_scaled.T @ v
        bottom = fftconvolve(v.reshape(n_y, n_x), one_blob, mode="same").ravel()
        return np.concatenate([top, bottom])

    return dict(
        n_y=n_y, n_x=n_x, rois=rois, rois_scaled=rois_scaled, norm_factors=norm_factors,
        lambdas=lambdas, k1=k1, k2=k2, cell_index_within=cell_index_within,
        n_cells_window=n_cells_window, operators=(A, At), include=include,
    )


def _solve_one_frame_cell(
    this_frame: np.ndarray, rois: np.ndarray, rois_scaled: np.ndarray, lambdas: np.ndarray, norm_factors: np.ndarray,
    k1: np.ndarray, k2: float, n_y: int, n_x: int, one_blob: np.ndarray, operators: tuple, solver_tol: float,
    solver_max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Solve one (frame, cell) pair -- independent of every other frame and cell."""
    n_cells_window = rois.shape[1]
    tc_lsq_frame = np.linalg.solve(rois.T @ rois, rois.T @ this_frame)

    A, At = operators
    x0 = np.zeros(n_cells_window + n_y * n_x)
    fit_weights = fista_nonneg_weighted_l1(A, At, this_frame, lambdas, x0, tol=solver_tol, max_iter=solver_max_iter)
    fit_weights = fit_weights / norm_factors
    fit_x = fit_weights[:n_cells_window]
    fit_bob = fit_weights[n_cells_window:]

    lsq_cost = np.sum((rois @ tc_lsq_frame - this_frame) ** 2)
    blob_contrib = fftconvolve(fit_bob.reshape(n_y, n_x), one_blob, mode="same").ravel()
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
) -> dict:
    """movie: (Y, X, F) array. profiles: (Y, X, nCellsTotal) array.
    which_cells: 0-indexed cell indices to analyze (default: all).
    progress_callback, if given, is called as progress_callback(frames_done,
    total_frames) each time a frame finishes.

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
    k1_list, k2_list, cell_index_within_list, operators = [], [], [], []

    for cc, this_cell in enumerate(which_cells):
        prof = profiles[:, :, this_cell]
        y0, y1, x0, x1 = _cell_window_bounds(prof, pad_space, mov_y, mov_x, use_com)

        pix_mask = np.zeros((mov_y, mov_x), dtype=bool)
        pix_mask[y0 : y1 + 1, x0 : x1 + 1] = True
        which_pixels[:, cc] = pix_mask.ravel()

        setup = _setup_cell_window(
            profiles, this_cell, y0, y1, x0, x1, min_pix_for_inclusion, lambda_prof, lambda_blob, sigma2_ds, p,
            one_blob,
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
                    k1_list[cc], k2_list[cc], n_y_list[cc], n_x_list[cc], one_blob, operators[cc], solver_tol,
                    solver_max_iter,
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
