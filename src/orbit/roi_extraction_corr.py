"""Correlation-based ROI selection: grow a mask from one seed pixel, ported
from roiapp's corr_based_masks.py (itself a port of corrBasedMasks_new.m).
Adapted from roiapp's `datamap.IMG` object wrapper to a plain (H, W, T)
ndarray, and from roiapp's "monkey_ind"-calibrated neuropil radii (a
recording-rig-specific constant with no orbit-gui equivalent) to explicit
pixel radii.

Three growth strategies, selected via ``growth_method``:

- 'local_corr_threshold' (this module's default): thresholds an
  already-computed local-correlation image directly and takes the
  connected component containing the seed pixel. Recovers exactly what a
  Local Correlation projection visually shows, by construction.
- 'fixed_seed': correlates every pixel's full timeseries (within a
  max_dist search disk) against ONE fixed seed timeseries, thresholds
  that correlation map, and cleans the thresholded blob. Correlation
  necessarily decays with distance from that one fixed reference point.
- 'flood_fill': iterative region growing -- starts from the seed, then
  repeatedly correlates every pixel adjacent to the CURRENT mask against
  the CURRENT mask's own (progressively less noisy) mean trace, adding
  whichever pass threshold, until nothing new is added.

All three support the same auto-threshold search: when no explicit
threshold is given, sweeps a fixed threshold range and picks whichever
maximizes an auto-threshold objective (see ``_method_objective``) computed
against a neuropil-ring trace -- used only to pick the best threshold
during the search, not to correct the final returned trace (see
``roi_from_seed``'s docstring).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops

from ._masks import EIGHT_CONNECTED
from ._masks import disk_mask as _shared_disk_mask
from ._masks import masked_mean_trace
from ._peak_picking import select_separated_pixels
from .neuropil import neuropil_ring_mask

_THRESH_RANGE = np.linspace(0.6, 0.9, 31)  # 0.6:0.01:0.9 inclusive
_SEARCH_FRAME_LIMIT = 1000  # cap frames used during the auto-threshold search for speed


@dataclass
class CorrMaskResult:
    mask: np.ndarray  # (H, W) bool
    trace: np.ndarray  # (T,) float, plain masked-mean of the fluorescence movie
    thresh: float


def _safe_nanargmax(values: np.ndarray) -> int:
    """Like np.nanargmax, but returns 0 instead of raising when every value
    is NaN (reachable when a block-averaged seed's self-correlation isn't
    guaranteed to exceed every candidate threshold, making every sweep
    candidate's mask empty)."""
    if np.all(np.isnan(values)):
        return 0
    return int(np.nanargmax(values))


def _make_distance_map(height: int, width: int, pix_loc: tuple[int, int], max_dist: float) -> np.ndarray:
    """Boolean disk of radius max_dist centered at pix_loc, cropped to image bounds."""
    return _shared_disk_mask(height, width, pix_loc, max_dist)


def _clean_roi(roi_init: np.ndarray, pix_loc: tuple[int, int]) -> np.ndarray:
    """Post-process a thresholded binary blob: keep the connected component
    containing pix_loc (or nearest to it), drop any strictly-larger
    component and any island smaller than half its area, then fill holes."""
    roi_init = np.asarray(roi_init, dtype=bool)
    if not roi_init.any():
        return np.zeros_like(roi_init)

    labeled = label(roi_init, connectivity=2)
    props = regionprops(labeled)
    if not props:
        return np.zeros_like(roi_init)

    row, col = int(pix_loc[0]), int(pix_loc[1])
    target = np.array([row, col])

    main_idx = None
    distances = np.full(len(props), np.inf)
    for i, region in enumerate(props):
        coords = region.coords
        if np.any((coords[:, 0] == row) & (coords[:, 1] == col)):
            main_idx = i
            break
        distances[i] = np.linalg.norm(coords - target, axis=1).min()
    if main_idx is None:
        main_idx = int(np.argmin(distances))
    main_area = props[main_idx].area

    keep_mask = np.zeros_like(roi_init)
    for region in props:
        if region.area > main_area or region.area < main_area * 0.5:
            continue
        keep_mask[labeled == region.label] = True

    filled = np.zeros_like(roi_init)
    for region in regionprops(label(keep_mask, connectivity=2)):
        min_row, min_col, max_row, max_col = region.bbox
        filled_component = getattr(region, "image_filled", None)
        if filled_component is None:
            filled_component = region.filled_image
        filled[min_row:max_row, min_col:max_col] |= filled_component
    return filled


def _clamped_block_bounds(height: int, width: int, pix_loc: tuple[int, int], radius: int) -> tuple[int, int, int, int]:
    """Row/col bounds (r0, r1, c0, c1) of a (2*radius+1)-square centered at
    pix_loc, clamped to image bounds -- shared by every seed-block-radius
    use (seed trace extraction, flood-fill's initial mask)."""
    radius = max(radius, 0)
    r0 = max(0, pix_loc[0] - radius)
    r1 = min(height, pix_loc[0] + radius + 1)
    c0 = max(0, pix_loc[1] - radius)
    c1 = min(width, pix_loc[1] + radius + 1)
    return r0, r1, c0, c1


def _extract_seed_trace(movie: np.ndarray, pix_loc: tuple[int, int], seed_block_radius: int) -> np.ndarray:
    """The seed timeseries correlated against every pixel in the search
    disk. seed_block_radius=0: just the clicked pixel's own trace.
    seed_block_radius>0: the mean trace over a
    (2*seed_block_radius+1)-square block centered on the click, clamped at
    the image border -- less noisy than a single pixel's raw timeseries."""
    if seed_block_radius <= 0:
        return np.asarray(movie[pix_loc[0], pix_loc[1], :], dtype=np.float64)

    height, width = movie.shape[0], movie.shape[1]
    r0, r1, c0, c1 = _clamped_block_bounds(height, width, pix_loc, seed_block_radius)
    block = np.asarray(movie[r0:r1, c0:c1, :], dtype=np.float64)
    return block.reshape(-1, block.shape[-1]).mean(axis=0)


def _corr_to_reference(traces: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Pearson correlation of each row in ``traces`` against a zero-mean,
    unit-norm ``reference`` vector -- shared by the fixed-seed and
    flood-fill correlation-map computations."""
    centered = traces - traces.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    normalized = np.divide(centered, norms[:, None], out=np.zeros_like(centered), where=norms[:, None] > 0)
    return normalized @ reference


def _compute_corr_coef_map(
    movie: np.ndarray, pix_loc: tuple[int, int], disk_mask: np.ndarray, seed_block_radius: int
) -> np.ndarray:
    seed = _extract_seed_trace(movie, pix_loc, seed_block_radius)
    seed = seed - seed.mean()
    seed = seed / np.linalg.norm(seed)

    rows, cols = np.nonzero(disk_mask)
    pixel_traces = np.asarray(movie[rows, cols, :], dtype=np.float64)
    corr_vals = _corr_to_reference(pixel_traces, seed)

    corr_map = np.zeros(disk_mask.shape, dtype=np.float64)
    corr_map[rows, cols] = corr_vals
    return corr_map


def _flood_fill_corr_mask(
    movie: np.ndarray, pix_loc: tuple[int, int], thresh: float, disk_mask: np.ndarray, seed_block_radius: int
) -> np.ndarray:
    """Iterative region growing: starts the mask at the seed, then repeats:
    compute the CURRENT mask's own mean trace (normalized), test every
    pixel 8-connected-adjacent to the mask and within disk_mask against
    it, add whichever exceed thresh, and repeat until a round adds
    nothing. Vectorized per round rather than one-pixel-at-a-time."""
    height, width = movie.shape[0], movie.shape[1]
    mask = np.zeros((height, width), dtype=bool)
    r0, r1, c0, c1 = _clamped_block_bounds(height, width, pix_loc, seed_block_radius)
    mask[r0:r1, c0:c1] = True
    mask &= disk_mask

    while True:
        mask_rows, mask_cols = np.nonzero(mask)
        mask_trace = np.asarray(movie[mask_rows, mask_cols, :], dtype=np.float64).mean(axis=0)
        mask_trace = mask_trace - mask_trace.mean()
        mask_trace_norm = np.linalg.norm(mask_trace)
        if mask_trace_norm == 0:
            break
        mask_trace = mask_trace / mask_trace_norm

        frontier = ndimage.binary_dilation(mask, structure=EIGHT_CONNECTED) & disk_mask & ~mask
        f_rows, f_cols = np.nonzero(frontier)
        if len(f_rows) == 0:
            break

        f_traces = np.asarray(movie[f_rows, f_cols, :], dtype=np.float64)
        corr_vals = _corr_to_reference(f_traces, mask_trace)

        to_add = corr_vals > thresh
        if not to_add.any():
            break
        mask[f_rows[to_add], f_cols[to_add]] = True

    return mask


def _threshold_local_corr_mask(
    pix_loc: tuple[int, int], thresh: float, disk_mask: np.ndarray, local_corr_image: np.ndarray
) -> np.ndarray:
    return _clean_roi((local_corr_image > thresh) & disk_mask, pix_loc)


def _method_objective(method: str, trace: np.ndarray, nptrace: np.ndarray) -> float:
    if method == "cellnpdiff":
        return float(np.sum(trace - nptrace) / np.var(nptrace, ddof=1))
    if method == "cellnpdecorr":
        r = np.corrcoef(trace, nptrace)[0, 1]
        return float(-r)
    raise ValueError(f"Unrecognized method: {method!r}")


def _search_best_threshold(
    movie: np.ndarray, pix_loc: tuple[int, int], disk_mask: np.ndarray, method: str, candidate_mask_fn
) -> float:
    """Sweeps _THRESH_RANGE, scoring each candidate mask's cell trace
    against its neuropil-ring trace, and returns the threshold maximizing
    _method_objective. ``candidate_mask_fn(thresh) -> (H, W) bool``."""
    objective_vals = np.full(len(_THRESH_RANGE), np.nan)
    for n, candidate in enumerate(_THRESH_RANGE):
        candidate_mask = candidate_mask_fn(candidate)
        if not candidate_mask.any():
            continue
        ring = neuropil_ring_mask(candidate_mask)
        if not ring.any():
            continue
        trace = masked_mean_trace(movie, candidate_mask, _SEARCH_FRAME_LIMIT)
        nptrace = masked_mean_trace(movie, ring, _SEARCH_FRAME_LIMIT)
        objective_vals[n] = _method_objective(method, trace, nptrace)
    return float(_THRESH_RANGE[_safe_nanargmax(objective_vals)])


def roi_from_seed(
    movie: np.ndarray,
    pix_loc: tuple[int, int],
    thresh: float | None = None,
    max_dist: float = 15.0,
    method: str = "cellnpdecorr",
    seed_block_radius: int = 0,
    growth_method: str = "local_corr_threshold",
    local_corr_image: np.ndarray | None = None,
) -> CorrMaskResult:
    """Grows one ROI mask from a single seed pixel. ``thresh=None`` triggers
    an auto-threshold search (see module docstring); ``method`` is the
    search objective ('cellnpdiff' or 'cellnpdecorr'), ignored when
    ``thresh`` is given explicitly. The returned trace is a plain
    masked-mean of the fluorescence movie -- neuropil correction is not
    applied to it, only used internally to pick the best threshold."""
    if growth_method not in ("fixed_seed", "flood_fill", "local_corr_threshold"):
        raise ValueError(f"Unrecognized growth_method: {growth_method!r}")
    if growth_method == "local_corr_threshold" and local_corr_image is None:
        raise ValueError("local_corr_image is required when growth_method='local_corr_threshold'")
    if thresh is not None:
        thresh = abs(thresh)
        if thresh > 1:
            raise ValueError("threshold must be less than 1")

    row, col = int(pix_loc[0]), int(pix_loc[1])
    height, width = movie.shape[0], movie.shape[1]
    disk_mask = _make_distance_map(height, width, (row, col), max_dist)

    if growth_method == "local_corr_threshold":
        candidate_fn = lambda t: _threshold_local_corr_mask((row, col), t, disk_mask, local_corr_image)  # noqa: E731
    elif growth_method == "flood_fill":
        candidate_fn = lambda t: _clean_roi(  # noqa: E731
            _flood_fill_corr_mask(movie, (row, col), t, disk_mask, seed_block_radius), (row, col)
        )
    else:
        corr_map = _compute_corr_coef_map(movie, (row, col), disk_mask, seed_block_radius)
        candidate_fn = lambda t: _clean_roi(corr_map > t, (row, col))  # noqa: E731

    if thresh is None:
        thresh = _search_best_threshold(movie, (row, col), disk_mask, method, candidate_fn)

    mask = candidate_fn(thresh)
    trace = masked_mean_trace(movie, mask)
    return CorrMaskResult(mask=mask, trace=trace, thresh=thresh)


def find_seed_candidates(
    local_corr_image: np.ndarray, n_seeds: int, min_separation_frac: float = 0.05, min_corr: float = 0.3
) -> list[tuple[int, int]]:
    """Auto-picks up to ``n_seeds`` candidate seed pixels from local
    correlation peaks, mutually separated by at least
    ``min_separation_frac`` of the field of view's longer dimension, and
    above ``min_corr`` -- feeds the "auto-select seeds" supervised mode:
    each candidate still gets grown into a mask via roi_from_seed and
    reviewed like a manual click, nothing is accepted automatically."""
    min_dist = min_separation_frac * max(local_corr_image.shape)
    candidates = select_separated_pixels(local_corr_image, n_seeds, min_dist, descending=True)
    return [(r, c) for r, c in candidates if local_corr_image[r, c] >= min_corr]
