"""Neuropil ring computation for ROI traces -- ported from roiapp's
algorithms/neuropil.py and morphology.py as closely as possible: a
dilation ring around an ROI (dilate to an outer radius minus dilate to
an inner radius), with any pixel already claimed by another ROI removed.
The ring estimates the local background/out-of-focus fluorescence
around a cell, used to correct its own signal trace.

Unlike roiapp's interactive per-click flow (which excludes only
already-added ROIs from a brand-new one's ring, without retroactively
updating older ones), every ROI's ring here gets recomputed from the
FULL current set whenever that set changes -- matching how roiapp's own
bulk-import path (load_suite2p) computes one shared exclusion union
up front for every ROI at once, just triggered on every add/delete
instead of only at bulk-import time.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ._masks import disk_mask, masked_mean_trace

INNER_RADIUS = 2  # pixels -- matches roiapp's monkey_ind=1 (default) ring
OUTER_RADIUS = 4  # pixels


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    footprint = disk_mask(2 * radius + 1, 2 * radius + 1, (radius, radius), radius)
    return ndimage.binary_dilation(np.asarray(mask, dtype=bool), structure=footprint, border_value=0)


def neuropil_ring_mask(
    roi_mask: np.ndarray,
    inner_radius: int = INNER_RADIUS,
    outer_radius: int = OUTER_RADIUS,
    exclusion_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Dilation ring around ``roi_mask``: dilate(mask, outer_radius) minus
    dilate(mask, inner_radius). ``exclusion_mask`` (typically the union of
    every OTHER known ROI) is subtracted out if given."""
    ring = _dilate(roi_mask, outer_radius) & ~_dilate(roi_mask, inner_radius)
    if exclusion_mask is not None:
        ring = ring & ~np.asarray(exclusion_mask, dtype=bool)
    return ring


def compute_neuropil_traces(movie: np.ndarray, rois: list, inner_radius: int = INNER_RADIUS, outer_radius: int = OUTER_RADIUS) -> None:
    """Recomputes every ROI's ``neuropil_trace`` IN PLACE: a dilated ring
    around its own mask, excluding pixels claimed by any OTHER ROI in
    ``rois``. Call this whenever the ROI set changes (one is added or
    removed) -- adding/removing an ROI can change every other ROI's ring.
    ``rois`` is duck-typed (each needs ``.mask`` and a settable
    ``.neuropil_trace``) rather than orbitapp.state.ROI directly, since
    orbit/ has no dependency on the Qt-layer package."""
    if not rois:
        return
    full_union = np.logical_or.reduce([roi.mask for roi in rois])
    for roi in rois:
        ring = neuropil_ring_mask(roi.mask, inner_radius, outer_radius, exclusion_mask=full_union)
        roi.neuropil_trace = masked_mean_trace(movie, ring)
