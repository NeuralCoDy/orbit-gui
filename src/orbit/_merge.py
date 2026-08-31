"""Shared union-find merge-by-overlap-and-correlation logic: reused by
cnmf.merge_overlapping_components (2D CNMF/CNMF-E) and
roi_extraction_graft_3d._merge_overlapping_masks_3d (volumetric GraFT),
both of which need to combine duplicate/split components discovered
independently (e.g. across overlapping patches) using the exact same
criterion -- spatial overlap alone isn't enough (two genuinely separate
but touching/adjacent cells would also overlap), so a merge additionally
requires the pair's measured traces to be highly correlated.
"""

from __future__ import annotations

import numpy as np


def find_merge_groups(masks: list[np.ndarray], traces: list[np.ndarray], merge_thresh: float) -> list[list[int]]:
    """Groups indices 0..len(masks) by transitive overlap+correlation --
    each returned group (in no particular order) is one final merged
    component; a singleton group means that component merged with
    nothing. ``masks`` must already be boolean -- 2D CNMF's continuous-
    valued footprints need their own `> 0` thresholding first (done at
    the call site, not here, since a footprint and a mask aren't the
    same kind of array to every caller)."""
    n = len(masks)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if not np.any(masks[i] & masks[j]):
                continue
            if np.corrcoef(traces[i], traces[j])[0, 1] >= merge_thresh:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())
