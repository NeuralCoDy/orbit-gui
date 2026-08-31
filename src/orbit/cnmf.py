"""CNMF source extraction (Pnevmatikakis et al. 2016), ported from CaImAn's
cnmf/spatial.py + temporal.py + merging.py as a compact, single-patch
pipeline: greedy init (cnmf_init.py) followed by alternating spatial/
temporal updates -- both regularized least-squares solves -- with AR(1)
OASIS deconvolution (cnmf_deconvolution.py) folded into the temporal step,
finishing with a correlation-based merge of duplicate/split components.
Deliberately skips CaImAn's own patch/memmap machinery, which this app's
scale doesn't need -- patch_cnmf_source_extraction below is this app's own,
much simpler patch wrapper (runs cnmf_source_extraction per patch, in
parallel worker processes, then merges).
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import numpy as np
from graft import solvers as graft_solvers

from ._masks import masked_mean_trace, threshold_footprint
from ._merge import find_merge_groups
from ._patches import make_patches_2d
from .cnmf_deconvolution import constrained_oasis_ar1, estimate_ar1_coefficient, estimate_noise_std
from .cnmf_init import estimate_background, greedy_roi_init

# Patches run in separate processes (not threads): a patch's own OASIS
# step is pure-Python/GIL-bound, so threads wouldn't overlap it. Capped
# at a small constant, not os.cpu_count(), to avoid oversubscribing
# against each patch's own BLAS-threaded solves -- see
# _single_threaded_blas_for_children below for the measured impact.
_DEFAULT_MAX_WORKERS = 4

# spawn, not fork (the POSIX default): this runs inside a PySide6 GUI
# with its own background threads, and forking a multi-threaded process
# risks the child deadlocking on a lock held elsewhere at fork time.
_MP_CONTEXT = multiprocessing.get_context("spawn")


@dataclass
class CNMFResult:
    masks: list[np.ndarray]  # each (H, W) bool
    traces: list[np.ndarray]  # each (T,) float, plain masked-mean of the fluorescence movie (matches every other
    # extraction method's trace convention -- NOT the OASIS-denoised "c" reconstruction, which is a model fit
    # rather than the measured signal; see spike_traces below for the deconvolution output)
    spike_traces: list[np.ndarray]  # each (T,) float, deconvolved spikes (OASIS "s")


def _centroid(footprint: np.ndarray) -> np.ndarray:
    if footprint.sum() <= 0:
        return np.array(footprint.shape, dtype=np.float64) / 2
    rows, cols = np.nonzero(footprint)
    weights = footprint[rows, cols]
    return np.array([np.average(rows, weights=weights), np.average(cols, weights=weights)])


def update_spatial_components(
    movie: np.ndarray, footprints: np.ndarray, traces: np.ndarray, background_temporal: np.ndarray,
    search_radius: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel nonnegative least squares: each pixel's timeseries is
    regressed against the traces of every component whose search disk
    covers it, plus the (always-candidate) background -- CaImAn's own
    spatial update restricts candidates the same way, both for speed and
    to keep footprints spatially local. Only the union of components'
    search disks is scanned, not the whole frame.

    Pixels are grouped by their exact candidate set (nearby pixels
    usually share one -- the set only changes where a pixel crosses a
    search-disk boundary) and each group's NNLS solves are batched into
    one call to ``graft.solvers.solve_nonneg_qp_batch`` (pygraft-gui's
    own native accelerator), which factors the group's shared Gram
    matrix once and solves every pixel in the group in parallel, rather
    than scipy.optimize.nnls's from-scratch solve per pixel.
    ``min_x ||Ax-b||^2 s.t. x>=0`` is equivalent to solve_nonneg_qp_batch's
    ``min_x x'Hx - c'x`` with ``H=A'A``, ``c=2*A'b`` (see tests for the
    numerical check against scipy.optimize.nnls)."""
    height, width, _n_frames = movie.shape
    n_background = len(background_temporal)
    n_components = len(footprints)
    centroids = np.array([_centroid(f) for f in footprints])

    new_footprints = np.zeros_like(footprints)
    new_background_spatial = np.zeros((height, width, n_background))

    row_lo = max(0, int(np.floor(centroids[:, 0].min() - search_radius)))
    row_hi = min(height - 1, int(np.ceil(centroids[:, 0].max() + search_radius)))
    col_lo = max(0, int(np.floor(centroids[:, 1].min() - search_radius)))
    col_hi = min(width - 1, int(np.ceil(centroids[:, 1].max() + search_radius)))
    rows = np.arange(row_lo, row_hi + 1)
    cols = np.arange(col_lo, col_hi + 1)

    row_dist2 = (rows[:, None] - centroids[:, 0]) ** 2  # (n_rows, K)
    col_dist2 = (cols[:, None] - centroids[:, 1]) ** 2  # (n_cols, K)
    within = (row_dist2[:, None, :] + col_dist2[None, :, :]) <= search_radius**2  # (n_rows, n_cols, K)
    n_rows_scanned, n_cols_scanned = within.shape[:2]
    within_flat = np.ascontiguousarray(within.reshape(-1, n_components))
    # np.unique(..., axis=0) sorts rows via a generic (slow) per-element
    # comparator; viewing each row as one opaque `void` value first lets
    # it use a plain, much faster 1D sort instead -- confirmed ~20x
    # faster than axis=0 for this grouping, which otherwise dominated
    # this function's own runtime (the actual batched solves below are
    # comparatively fast).
    row_keys = within_flat.view(np.dtype((np.void, within_flat.dtype.itemsize * n_components))).ravel()
    _unique_keys, inverse = np.unique(row_keys, return_inverse=True)
    inverse = inverse.reshape(n_rows_scanned, n_cols_scanned)

    for group_idx in range(inverse.max(initial=-1) + 1):
        local_rows, local_cols = np.nonzero(inverse == group_idx)
        if len(local_rows) == 0:
            continue
        candidates = np.where(within[local_rows[0], local_cols[0]])[0]
        if len(candidates) == 0:
            continue
        abs_rows, abs_cols = rows[local_rows], cols[local_cols]

        design = np.vstack([traces[candidates], background_temporal]).T  # (T, n_candidates + n_bg)
        gram = design.T @ design  # (n_candidates + n_bg, n_candidates + n_bg), shared across this group
        pixel_traces = movie[abs_rows, abs_cols, :]  # (n_pixels_in_group, T)
        linear_term = 2 * (pixel_traces @ design)  # (n_pixels_in_group, n_candidates + n_bg)
        coeffs = graft_solvers.solve_nonneg_qp_batch(gram, linear_term)

        for i, component in enumerate(candidates):
            new_footprints[component, abs_rows, abs_cols] = coeffs[:, i]
        new_background_spatial[abs_rows, abs_cols, :] = coeffs[:, len(candidates) :]

    return new_footprints, new_background_spatial


def update_temporal_components(
    movie: np.ndarray, footprints: np.ndarray, traces: np.ndarray, background_spatial: np.ndarray,
    background_temporal: np.ndarray, g_list: list[float], noise_stds: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HALS (block coordinate descent): each component's trace is re-fit
    against the movie with every OTHER component's and the background's
    current contribution subtracted out, then denoised via constrained
    OASIS AR(1) deconvolution. Returns (C, S, updated background trace).

    Maintains the "sufficient statistics" A^T @ residual (K, T) --
    standard HALS/NMF practice -- rather than the full (P, T) residual
    against the movie: profiling showed materializing a fresh (P, T)
    array per component (either via a full re-multiply or an outer
    product update) was the dominant cost of a CNMF run, since P = H*W
    is typically >>K. The per-component update here is an outer product
    of two size-K/size-T vectors instead, using the small (K, K) Gram
    matrix AtA computed once up front. Verified numerically equivalent
    to the direct from-scratch version to floating-point precision (see
    tests)."""
    height, width, n_frames = movie.shape
    n_components = len(footprints)
    y_flat = movie.reshape(-1, n_frames)
    spatial_flat = footprints.reshape(n_components, -1).T  # (P, K)
    background_flat = background_spatial.reshape(-1, background_spatial.shape[-1])  # (P, n_bg)

    c_updated = traces.copy()
    new_c = np.zeros_like(traces)
    new_s = np.zeros_like(traces)

    AtA = spatial_flat.T @ spatial_flat  # (K, K)
    AtY = spatial_flat.T @ y_flat  # (K, T)
    AtB = spatial_flat.T @ background_flat  # (K, n_bg)
    denom = np.diag(AtA)

    at_residual = AtY - AtA @ c_updated - AtB @ background_temporal  # (K, T)
    for k in range(n_components):
        if denom[k] <= 0:
            continue
        raw_trace = at_residual[k] / denom[k] + c_updated[k]
        c_k, s_k = constrained_oasis_ar1(raw_trace, g_list[k], noise_stds[k])
        at_residual -= np.outer(AtA[:, k], c_k - c_updated[k])
        new_c[k] = c_k
        new_s[k] = s_k
        c_updated[k] = c_k  # Gauss-Seidel: later components see this update immediately

    residual_no_bg = y_flat - spatial_flat @ c_updated
    if background_flat.shape[1] > 0:
        # Normal equations, not lstsq: background_flat is (P, n_bg) with
        # P=H*W >> n_bg (typically 1), so solving the tiny (n_bg, n_bg)
        # Gram system directly is the same least-squares answer as
        # lstsq's general SVD-based solve (see tests) but 5-14x faster
        # at realistic frame sizes, since it skips a P-sized
        # factorization. Ridge term guards a near-singular Gram matrix.
        gram = background_flat.T @ background_flat  # (n_bg, n_bg)
        rhs = background_flat.T @ residual_no_bg  # (n_bg, T)
        jitter = 1e-10 * np.trace(gram) / max(gram.shape[0], 1)
        new_background_temporal = np.clip(np.linalg.solve(gram + jitter * np.eye(gram.shape[0]), rhs), 0, None)
    else:
        new_background_temporal = background_temporal

    return new_c, new_s, new_background_temporal


def merge_overlapping_components(
    footprints: np.ndarray, traces: np.ndarray, spike_traces: np.ndarray, g_list: list[float],
    merge_thresh: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Groups components wherever footprints overlap AND traces are
    highly correlated (CaImAn's own merge criterion needs both -- see
    _merge.find_merge_groups, shared with volumetric GraFT's own merge
    step), then collapses each group into one footprint-weighted-mean
    trace, re-deconvolved via OASIS."""
    groups = find_merge_groups([f > 0 for f in footprints], traces, merge_thresh)

    merged_footprints, merged_traces, merged_spikes, merged_g = [], [], [], []
    for members in groups:
        if len(members) == 1:
            (k,) = members
            merged_footprints.append(footprints[k])
            merged_traces.append(traces[k])
            merged_spikes.append(spike_traces[k])
            merged_g.append(g_list[k])
            continue

        weights = np.array([footprints[k].sum() for k in members])
        weights = weights / weights.sum()
        merged_footprint = np.sum([footprints[k] for k in members], axis=0)
        merged_trace = np.sum([w * traces[k] for w, k in zip(weights, members)], axis=0)
        g = float(np.mean([g_list[k] for k in members]))
        c, s = constrained_oasis_ar1(merged_trace, g, estimate_noise_std(merged_trace))
        merged_footprints.append(merged_footprint)
        merged_traces.append(c)
        merged_spikes.append(s)
        merged_g.append(g)

    return np.stack(merged_footprints), np.stack(merged_traces), np.stack(merged_spikes), merged_g


def _finalize_result(movie: np.ndarray, footprints: np.ndarray, spike_traces: np.ndarray) -> CNMFResult:
    """Boolean masks + measured (masked-mean) traces from final
    footprints -- shared by the single-patch and patch-based entry
    points below, since both need this identical last step. The ROI's
    *primary* trace is the measured signal (masked-mean of the movie
    itself, e.g. dF/F if the movie was normalized upstream), not the
    OASIS-denoised model reconstruction -- that's still available via
    spike_traces' deconvolution, just not what callers see as "the
    trace"."""
    masks = [f > 0 for f in footprints]
    traces = [masked_mean_trace(movie, mask) for mask in masks]
    return CNMFResult(masks=masks, traces=traces, spike_traces=list(spike_traces))


def cnmf_source_extraction(
    movie: np.ndarray,
    n_components: int = 20,
    gauss_sigma: float = 2.0,
    init_radius: float = 5.0,
    search_radius: float = 10.0,
    n_background_components: int = 1,
    merge_thresh: float = 0.8,
    n_iterations: int = 2,
) -> CNMFResult:
    """Full pipeline: greedy init -> [spatial update -> threshold ->
    temporal update (+ OASIS) -> merge] x n_iterations -> final masks.
    See patch_cnmf_source_extraction for a version that splits a large
    field of view into patches first, for the frames/movies where this
    whole-FOV version doesn't scale well."""
    footprints, traces = greedy_roi_init(movie, n_components, gauss_sigma, init_radius)
    height, width, _n_frames = movie.shape
    # reshape+matmul instead of einsum("khw,kt->hwt", ...) -- identical
    # math, but einsum doesn't recognize this as a plain matrix product
    # and falls back to a slow generic loop instead of BLAS (>50x slower
    # at realistic movie sizes).
    recon = (footprints.reshape(n_components, -1).T @ traces).reshape(height, width, -1)
    residual = movie - recon
    # residual isn't read again after this call, so estimate_background is
    # allowed to clip it in place -- avoids a second full-(H, W, T) clipped
    # copy on top of the one `movie - recon` already allocated above.
    np.clip(residual, 0, None, out=residual)
    background_spatial, background_temporal = estimate_background(residual, n_background_components)

    g_list = [estimate_ar1_coefficient(t) for t in traces]
    noise_stds = [estimate_noise_std(t) for t in traces]
    spike_traces = np.zeros_like(traces)

    for iteration in range(n_iterations):
        footprints, background_spatial = update_spatial_components(
            movie, footprints, traces, background_temporal, search_radius
        )
        footprints = np.stack([threshold_footprint(f) for f in footprints])
        traces, spike_traces, background_temporal = update_temporal_components(
            movie, footprints, traces, background_spatial, background_temporal, g_list, noise_stds
        )
        if iteration < n_iterations - 1:
            footprints, traces, spike_traces, g_list = merge_overlapping_components(
                footprints, traces, spike_traces, g_list, merge_thresh
            )
            noise_stds = [estimate_noise_std(t) for t in traces]

    return _finalize_result(movie, footprints, spike_traces)


_BLAS_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@contextmanager
def _single_threaded_blas_for_children():
    """Pins BLAS thread-count env vars to 1 in this process for spawned
    workers to inherit (spawn copies os.environ at spawn time), restored
    on exit so this process's own BLAS calls aren't affected. Without
    this, each (already process-parallel) worker also fans its NNLS/NMF
    solves out across every core via OpenBLAS's own threading --
    measured 4 workers running patches ~5x SLOWER than 1, not just
    failing to speed up, from that oversubscription.

    A ProcessPoolExecutor `initializer` doesn't work here: resolving a
    pickled reference to it re-imports this module (and numpy) in the
    child BEFORE the initializer body runs, by which point OpenBLAS's
    thread pool is already sized -- setting the env var here, in the
    parent before any worker spawns, is the earliest point that
    actually works."""
    previous = {var: os.environ.get(var) for var in _BLAS_THREAD_ENV_VARS}
    for var in _BLAS_THREAD_ENV_VARS:
        os.environ[var] = "1"
    try:
        yield
    finally:
        for var, value in previous.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


def _run_patch(movie_patch: np.ndarray, n_components_per_patch: int, merge_thresh: float, cnmf_kwargs: dict) -> CNMFResult:
    """Module-level (rather than a closure) so ProcessPoolExecutor can
    pickle a reference to it for each worker process."""
    return cnmf_source_extraction(
        movie_patch, n_components=n_components_per_patch, merge_thresh=merge_thresh, **cnmf_kwargs
    )


def _run_patches_and_merge(
    movie: np.ndarray,
    per_patch_fn: Callable[[np.ndarray, int, float, dict], CNMFResult],
    patch_size: tuple[int, int],
    overlap: int,
    n_components_per_patch: int,
    merge_thresh: float,
    progress_callback,
    max_workers: int | None,
    patch_kwargs: dict,
) -> CNMFResult:
    """Shared patch-orchestration core for patch_cnmf_source_extraction
    below and cnmf_e.patch_cnmf_e_source_extraction: splits the field of
    view into overlapping patches, runs ``per_patch_fn`` (a module-level,
    ProcessPoolExecutor-picklable per-patch entry point -- see _run_patch/
    cnmf_e._run_patch_e) on each in its own worker process, re-embeds
    each patch's locally-indexed components into full-FOV-sized
    footprints, and merges components found in more than one patch's
    overlap region via merge_overlapping_components -- the same logic a
    single-patch run already uses to resolve split/duplicate components.

    Patching exists because several of a single-patch run's costs scale
    with the *whole frame* regardless of how many components are
    actually in it (background estimation, the initial reconstruction,
    the spatial-update scan region) -- restricting each run to a patch
    keeps those bounded by patch_size instead of the full (H, W).
    ``patch_kwargs`` are forwarded to every patch's ``per_patch_fn`` call
    (gauss_sigma, init_radius, search_radius, n_iterations, ... or, for
    CNMF-E, also the ring-model parameters). progress_callback, if
    given, is called as progress_callback(patches_done, total_patches)
    once per patch, in patch order, as each patch's result becomes
    available (patches themselves may finish out of order across worker
    processes). max_workers caps how many patches run concurrently --
    None falls back to min(_DEFAULT_MAX_WORKERS, os.cpu_count(),
    len(patches)); see _DEFAULT_MAX_WORKERS' comment for why that stays
    a small constant rather than just os.cpu_count()."""
    height, width, _n_frames = movie.shape
    patches = make_patches_2d(height, width, patch_size, overlap)
    workers = max_workers if max_workers is not None else min(_DEFAULT_MAX_WORKERS, os.cpu_count() or 1, len(patches))

    all_footprints, all_traces, all_spikes, all_g = [], [], [], []
    with _single_threaded_blas_for_children(), ProcessPoolExecutor(max_workers=workers, mp_context=_MP_CONTEXT) as pool:
        futures = [
            pool.submit(per_patch_fn, movie[r0:r1, c0:c1, :], n_components_per_patch, merge_thresh, patch_kwargs)
            for r0, r1, c0, c1 in patches
        ]
        for i, (future, (r0, r1, c0, c1)) in enumerate(zip(futures, patches)):
            result = future.result()
            for mask, trace, spike in zip(result.masks, result.traces, result.spike_traces):
                if not mask.any():
                    continue
                full_footprint = np.zeros((height, width))
                full_footprint[r0:r1, c0:c1] = mask
                all_footprints.append(full_footprint)
                all_traces.append(trace)
                all_spikes.append(spike)
                all_g.append(estimate_ar1_coefficient(trace))
            if progress_callback is not None:
                progress_callback(i + 1, len(patches))

    if not all_footprints:
        return CNMFResult(masks=[], traces=[], spike_traces=[])

    merged_footprints, _merged_traces, merged_spikes, _g = merge_overlapping_components(
        np.stack(all_footprints), np.stack(all_traces), np.stack(all_spikes), all_g, merge_thresh
    )
    return _finalize_result(movie, merged_footprints, merged_spikes)


def patch_cnmf_source_extraction(
    movie: np.ndarray,
    patch_size: tuple[int, int] = (80, 80),
    overlap: int = 20,
    n_components_per_patch: int = 10,
    merge_thresh: float = 0.8,
    progress_callback=None,
    max_workers: int | None = _DEFAULT_MAX_WORKERS,
    **cnmf_kwargs,
) -> CNMFResult:
    """Runs cnmf_source_extraction independently on overlapping spatial
    patches instead of the whole field of view at once -- see
    _run_patches_and_merge for the shared orchestration/merge mechanics."""
    return _run_patches_and_merge(
        movie, _run_patch, patch_size, overlap, n_components_per_patch, merge_thresh, progress_callback,
        max_workers, cnmf_kwargs,
    )
