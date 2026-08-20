"""CNMF source extraction (Pnevmatikakis et al. 2016), ported from CaImAn's
cnmf/spatial.py + temporal.py + merging.py as a compact, single-patch
pipeline: greedy init (cnmf_init.py) followed by alternating spatial/
temporal updates -- both regularized least-squares solves -- with AR(1)
OASIS deconvolution (cnmf_deconvolution.py) folded into the temporal step,
finishing with a correlation-based merge of duplicate/split components.
Deliberately skips CaImAn's patch-parallelism/memmap machinery, which this
app's scale doesn't need.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label
from scipy.optimize import nnls

from ._masks import EIGHT_CONNECTED, masked_mean_trace
from .cnmf_deconvolution import constrained_oasis_ar1, estimate_ar1_coefficient, estimate_noise_std
from .cnmf_init import estimate_background, greedy_roi_init


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


def threshold_footprint(footprint: np.ndarray, quantile: float = 0.5) -> np.ndarray:
    """Drops the bottom ``quantile`` of a footprint's own nonzero weights
    (nnls/least-squares noise scattered thinly across the search radius),
    then keeps only the connected component still covering the peak
    pixel -- CaImAn's own spatial post-processing cleanup."""
    if not footprint.any():
        return footprint
    nz = footprint[footprint > 0]
    thresh = np.quantile(nz, quantile)
    cleaned = np.where(footprint >= thresh, footprint, 0.0)
    if not cleaned.any():
        return cleaned
    peak = np.unravel_index(np.argmax(footprint), footprint.shape)
    labeled, _n = label(cleaned > 0, structure=EIGHT_CONNECTED)
    keep_label = labeled[peak]
    if keep_label == 0:
        return cleaned
    return np.where(labeled == keep_label, cleaned, 0.0)


def update_spatial_components(
    movie: np.ndarray, footprints: np.ndarray, traces: np.ndarray, background_temporal: np.ndarray,
    search_radius: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel nonnegative least squares: each pixel's timeseries is
    regressed against the traces of every component whose search disk
    covers it, plus the (always-candidate) background -- CaImAn's own
    spatial update restricts candidates the same way, both for speed and
    to keep footprints spatially local. Only the union of components'
    search disks is scanned, not the whole frame."""
    height, width, _n_frames = movie.shape
    n_background = len(background_temporal)
    centroids = np.array([_centroid(f) for f in footprints])

    new_footprints = np.zeros_like(footprints)
    new_background_spatial = np.zeros((height, width, n_background))

    row_lo = max(0, int(np.floor(centroids[:, 0].min() - search_radius)))
    row_hi = min(height - 1, int(np.ceil(centroids[:, 0].max() + search_radius)))
    col_lo = max(0, int(np.floor(centroids[:, 1].min() - search_radius)))
    col_hi = min(width - 1, int(np.ceil(centroids[:, 1].max() + search_radius)))

    for row in range(row_lo, row_hi + 1):
        row_dist2 = (centroids[:, 0] - row) ** 2
        for col in range(col_lo, col_hi + 1):
            candidates = np.where(row_dist2 + (centroids[:, 1] - col) ** 2 <= search_radius**2)[0]
            if len(candidates) == 0:
                continue
            design = np.vstack([traces[candidates], background_temporal]).T  # (T, n_candidates + n_bg)
            coeffs, _residual = nnls(design, movie[row, col, :])
            new_footprints[candidates, row, col] = coeffs[: len(candidates)]
            new_background_spatial[row, col, :] = coeffs[len(candidates) :]

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
        new_background_temporal = np.clip(np.linalg.lstsq(background_flat, residual_no_bg, rcond=None)[0], 0, None)
    else:
        new_background_temporal = background_temporal

    return new_c, new_s, new_background_temporal


def merge_overlapping_components(
    footprints: np.ndarray, traces: np.ndarray, spike_traces: np.ndarray, g_list: list[float],
    merge_thresh: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Union-finds components into merge groups wherever footprints
    overlap AND traces are highly correlated (CaImAn's own merge
    criterion needs both), then collapses each group into one
    footprint-weighted-mean trace, re-deconvolved via OASIS."""
    n_components = len(footprints)
    parent = list(range(n_components))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n_components):
        for j in range(i + 1, n_components):
            if not np.any((footprints[i] > 0) & (footprints[j] > 0)):
                continue
            if np.corrcoef(traces[i], traces[j])[0, 1] >= merge_thresh:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(n_components):
        groups.setdefault(find(i), []).append(i)

    merged_footprints, merged_traces, merged_spikes, merged_g = [], [], [], []
    for members in groups.values():
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
    residual = movie - np.einsum("khw,kt->hwt", footprints, traces)
    background_spatial, background_temporal = estimate_background(np.clip(residual, 0, None), n_background_components)

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


def _patch_bounds(size: int, patch_extent: int, overlap: int) -> list[int]:
    """Start offsets of overlapping patches of length ``patch_extent``
    tiling ``[0, size)`` -- the last one is pulled back to end exactly at
    ``size`` (rather than running past it) so every pixel is covered by
    at least one patch, matching however unevenly `size` divides. When
    the regular stride already lands within half a stride of that edge
    position, the last regular start is shifted to the edge instead of
    an extra patch being appended there -- otherwise a small remainder
    (e.g. tiling 256px with a 100px/25px-overlap patch leaves a 6px
    remainder) adds a near-duplicate patch just a few pixels over from
    the previous one, close to doubling total patch coverage for
    negligible extra frame coverage."""
    if size <= patch_extent:
        return [0]
    stride = max(1, patch_extent - overlap)
    starts = list(range(0, size - patch_extent + 1, stride))
    edge = size - patch_extent
    if starts[-1] != edge:
        # Shifting (rather than appending) only when there's already a
        # second-to-last start to keep the near-0 edge covered -- if
        # starts is just [0], shifting it away from 0 would leave [0,
        # edge) uncovered entirely, since nothing else covers that end.
        if len(starts) > 1 and edge - starts[-1] < stride / 2:
            starts[-1] = edge
        else:
            starts.append(edge)
    return starts


def _make_patches(height: int, width: int, patch_size: tuple[int, int], overlap: int) -> list[tuple[int, int, int, int]]:
    """(row0, row1, col0, col1) bounds of every patch tiling (height, width)."""
    patch_h, patch_w = min(patch_size[0], height), min(patch_size[1], width)
    row_starts = _patch_bounds(height, patch_h, overlap)
    col_starts = _patch_bounds(width, patch_w, overlap)
    return [(r0, r0 + patch_h, c0, c0 + patch_w) for r0 in row_starts for c0 in col_starts]


def patch_cnmf_source_extraction(
    movie: np.ndarray,
    patch_size: tuple[int, int] = (80, 80),
    overlap: int = 20,
    n_components_per_patch: int = 10,
    merge_thresh: float = 0.8,
    progress_callback=None,
    **cnmf_kwargs,
) -> CNMFResult:
    """Runs cnmf_source_extraction independently on overlapping spatial
    patches instead of the whole field of view at once, then merges
    components found in more than one patch's overlap region -- reusing
    merge_overlapping_components, the same logic a single-patch run
    already uses to resolve split/duplicate components.

    Patching exists because several of cnmf_source_extraction's costs
    scale with the *whole frame*, regardless of how many components are
    actually in it or where (confirmed by profiling: the background NMF
    fit dominates on a large FOV) -- restricting each run to a patch
    keeps those bounded by patch_size instead of the full (H, W).
    cnmf_kwargs are forwarded to every patch's cnmf_source_extraction
    call (gauss_sigma, init_radius, search_radius, n_iterations, ...).
    progress_callback, if given, is called as progress_callback(
    patches_done, total_patches)."""
    height, width, _n_frames = movie.shape
    patches = _make_patches(height, width, patch_size, overlap)

    all_footprints, all_traces, all_spikes, all_g = [], [], [], []
    for i, (r0, r1, c0, c1) in enumerate(patches):
        result = cnmf_source_extraction(
            movie[r0:r1, c0:c1, :], n_components=n_components_per_patch, merge_thresh=merge_thresh, **cnmf_kwargs
        )
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
