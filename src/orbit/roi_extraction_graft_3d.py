"""GraFT source extraction for volumetric (T, L, W, D) data -- requires a
mask (see orbitapp.tabs.mask_tab): the masked region's voxels are
reorganized into a 2D (n_pixels, T) array ("2D array of pixel by time"),
GraFT is run on that directly, and each resulting spatial component is
scattered back into a (L, W, D) volume using the voxel coordinates saved
when the array was built.

Two findings from pyGraFT's own source (not just its public docstrings)
make this a direct, non-hacky use of the API rather than a workaround:

- ``graft.graft``'s ``data_obj`` may be ``(H, W, T)`` *or* a flat
  ``(n_pix, T)`` array directly (confirmed in graft/core.py's
  ``_get_problem_sizes``/``graft()``) -- when given the latter with no
  ``params["mask"]``/``nRows``/``nCols``, the returned spatial dictionary
  comes back unreshaped, as ``(n_pix, N)``. A flat pixel list *is* a
  first-class input mode, not a shape this module has to trick the
  algorithm into accepting.
- This app's own default kernel (``corrType: "embedding"``) builds its
  spatial-smoothness graph from each pixel's own time-trace similarity
  (a kNN graph over PCA-reduced traces -- see graft/kernels.py's
  ``make_data_embedding``), not from 2D grid adjacency. It flattens
  ``(H, W, T)`` to ``(n_pix, T)`` internally regardless. That means the
  order voxels are listed in here is irrelevant to the algorithm's own
  spatial regularization -- no adjacency-preserving reshape trick is
  needed, an arbitrary (but *saved*) voxel order is exactly as correct
  as any other.

Merging across overlapping patches shares its overlap+correlation
grouping with ``cnmf.merge_overlapping_components`` (see
``_merge.find_merge_groups``), but not that function's own
post-grouping step -- ``cnmf``'s is coupled to OASIS spike
re-deconvolution, which has no GraFT equivalent (GraFT ROIs carry a
measured trace only, same as the 2D ``roi_extraction_graft.py``), so
_merge_overlapping_masks_3d below does its own (simpler) mask-union +
recomputed-trace collapse instead.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import graft
import numpy as np

from ._masks import threshold_footprint
from ._merge import find_merge_groups
from ._patches import make_patches_3d
from .roi_extraction_graft import _DEFAULT_CORR_KERN, _MAX_PATCH_WORKERS


@dataclass
class GraFTResult3D:
    masks: list[np.ndarray]  # each (L, W, D) bool
    traces: list[np.ndarray]  # each (T,) float, masked-mean fluorescence -- same convention as 2D GraFTResult


def _require_mask(mask: np.ndarray | None) -> None:
    """Raises unless ``mask`` is a real, restricting mask: not given, not
    empty (nothing selected), and not all-True (an explicit Clear Mask,
    or an auto-threshold that happened to keep everything, doesn't
    reduce the voxel count -- the whole reason a mask is required here)."""
    if mask is None or not mask.any() or mask.all():
        raise ValueError(
            "GraFT on volumetric data requires a real, restricting mask -- run Mask tab (Auto-threshold) "
            "first; an empty or all-kept (Clear Mask) mask isn't enough."
        )


def _masked_mean_trace_3d(movie: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-timepoint mean of ``movie`` (T, L, W, D) over ``mask``'s
    (L, W, D) True voxels; zeros if the mask is empty. T-first, 3D-mask
    counterpart of orbit._masks.masked_mean_trace."""
    if not mask.any():
        return np.zeros(movie.shape[0])
    return np.asarray(movie[:, mask], dtype=np.float64).mean(axis=1)


def _run_one_region(
    movie: np.ndarray, mask_full: np.ndarray, l0: int, l1: int, w0: int, w1: int, d0: int, d1: int,
    n_dict: int, rng: np.random.Generator | None, **graft_params,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The shared core both graft_source_extraction_3d (one region
    spanning the whole volume) and patch_graft_source_extraction_3d
    (many smaller, overlapping regions) call. Returns full-volume-sized
    masks + their traces, or ``([], [])`` if this region has no masked
    voxels at all (skipped without ever calling GraFT)."""
    local_mask = mask_full[l0:l1, w0:w1, d0:d1]
    if not local_mask.any():
        return [], []

    coords = np.argwhere(local_mask)  # (n_pix, 3) local (within-region) voxel coordinates -- "saving the 3D location"
    region = movie[:, l0:l1, w0:w1, d0:d1]
    flat = region[:, coords[:, 0], coords[:, 1], coords[:, 2]].T  # (n_pix, T) -- "2D array of pixel by time"

    _dict_temporal, S, _extras = graft.graft(
        flat, corr_kern=_DEFAULT_CORR_KERN, params={"n_dict": n_dict, **graft_params}, rng=rng,
    )  # S is (n_pix, N), unreshaped -- same flat pixel order as `coords`

    masks: list[np.ndarray] = []
    traces: list[np.ndarray] = []
    L, W, D = mask_full.shape
    for i in range(S.shape[1]):
        local_footprint = np.zeros(local_mask.shape)
        local_footprint[coords[:, 0], coords[:, 1], coords[:, 2]] = S[:, i]
        cleaned = threshold_footprint(local_footprint) > 0  # real 3D connected-component cleanup
        if not cleaned.any():
            continue
        full_mask = np.zeros((L, W, D), dtype=bool)
        full_mask[l0:l1, w0:w1, d0:d1] = cleaned  # "rebuilding the 3D ROI ... into the volume"
        masks.append(full_mask)
        traces.append(_masked_mean_trace_3d(movie, full_mask))
    return masks, traces


def _merge_overlapping_masks_3d(
    movie: np.ndarray, masks: list[np.ndarray], traces: list[np.ndarray], merge_thresh: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Groups masks wherever they overlap AND traces are highly
    correlated (see _merge.find_merge_groups, shared with
    cnmf.merge_overlapping_components's identical criterion), then
    collapses each group into the union of its masks with a trace
    recomputed from that union -- consistent with every other ROI's
    "trace = masked-mean over its own final mask" convention, rather
    than averaging each patch's already-computed (smaller-mask) trace."""
    groups = find_merge_groups(masks, traces, merge_thresh)

    merged_masks, merged_traces = [], []
    for members in groups:
        union_mask = np.logical_or.reduce([masks[i] for i in members])
        merged_masks.append(union_mask)
        merged_traces.append(_masked_mean_trace_3d(movie, union_mask))
    return merged_masks, merged_traces


def graft_source_extraction_3d(
    movie: np.ndarray, mask: np.ndarray | None, n_dict: int = 20, rng: np.random.Generator | None = None,
    **graft_params,
) -> GraFTResult3D:
    """Whole-masked-volume GraFT. ``movie`` is (T, L, W, D); ``mask`` is
    (L, W, D) and required -- raises ValueError if not given/empty (see
    _require_mask). Not memmap-safe (reads every masked voxel across the
    whole movie at once) -- use patch_graft_source_extraction_3d for a
    memory-mapped movie, same requirement as the 2D CNMF/GraFT methods."""
    _require_mask(mask)
    L, W, D = movie.shape[1:]
    masks, traces = _run_one_region(movie, mask, 0, L, 0, W, 0, D, n_dict, rng, **graft_params)
    return GraFTResult3D(masks=masks, traces=traces)


def patch_graft_source_extraction_3d(
    movie: np.ndarray,
    mask: np.ndarray | None,
    patch_size: tuple[int, int, int] = (20, 20, 20),
    overlap: int = 4,
    n_dict_per_patch: int = 10,
    rng: np.random.Generator | None = None,
    max_workers: int | None = _MAX_PATCH_WORKERS,
    merge_thresh: float = 0.85,
    **graft_params,
) -> GraFTResult3D:
    """Patch-based GraFT -- splits the masked volume into overlapping 3D
    spatial patches (make_patches_3d), runs _run_one_region independently
    per patch (regions with no masked voxels are skipped without ever
    calling GraFT), then merges components found in more than one
    patch's overlap region. ``mask`` is required -- see _require_mask.

    Already memmap-safe: each patch only ever reads its own bounded
    slice of ``movie``. ``max_workers`` caps concurrent patch threads --
    see roi_extraction_graft.py's _MAX_PATCH_WORKERS docstring for why
    this defaults to a small constant (each graft.graft call may itself
    run an OpenMP-parallel native solve internally, so too many
    concurrent patches can oversubscribe badly on a high-core-count
    machine)."""
    _require_mask(mask)
    L, W, D = movie.shape[1:]
    regions = make_patches_3d(L, W, D, patch_size, overlap)
    workers = max_workers if max_workers is not None else min(_MAX_PATCH_WORKERS, os.cpu_count() or 1, len(regions))

    all_masks: list[np.ndarray] = []
    all_traces: list[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_run_one_region, movie, mask, l0, l1, w0, w1, d0, d1, n_dict_per_patch, rng, **graft_params)
            for l0, l1, w0, w1, d0, d1 in regions
        ]
        for future in futures:
            masks, traces = future.result()
            all_masks.extend(masks)
            all_traces.extend(traces)

    if not all_masks:
        return GraFTResult3D(masks=[], traces=[])
    merged_masks, merged_traces = _merge_overlapping_masks_3d(movie, all_masks, all_traces, merge_thresh)
    return GraFTResult3D(masks=merged_masks, traces=merged_traces)
