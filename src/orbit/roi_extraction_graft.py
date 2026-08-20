"""GraFT (Graph-Filtered Temporal dictionary learning, Charles et al.)
source extraction, via the ``pygraft-gui`` PyPI package's ``graft``
algorithm module (not its own GUI, ``graftapp``, which orbit-gui doesn't
use). ``graft`` is a dependency, not a port -- see the pyGraFT
repository for the algorithm itself.

Both entry points below adapt ``graft``'s own output convention to this
app's: ``graft.graft``/``graft.patch_graft`` return continuous-valued
spatial dictionary atoms (not boolean masks) and a *model* temporal
dictionary (not the measured signal) -- ``_finalize`` threshold-cleans
the former (reusing ``orbit._masks.threshold_footprint``, the same
cleanup CNMF's own continuous footprints get) and replaces the latter
with each mask's masked-mean fluorescence, matching every other
extraction method's ``ROI.trace`` convention in this app.
"""

from __future__ import annotations

from dataclasses import dataclass

import graft
import numpy as np

from ._masks import masked_mean_trace, threshold_footprint

# The corr_kern graftapp's own GUI actually uses (graftapp/workers.py) --
# graft.graft's own default (None) exists mainly for programmatic/testing
# use, not real data.
_DEFAULT_CORR_KERN = {"corrType": "embedding", "reduce_dim": True, "w_time": 0}

# patch_graft's own default is min(n_patches, cpu_count, 8) concurrent
# ThreadPoolExecutor workers -- each of which ALSO runs its own
# OpenMP-parallel native solve internally. On a high-core-count machine
# (confirmed: reproducible segfault on an 80-core box with the default,
# every time, with enough patches in flight) that combination
# oversubscribes so badly it crashes, not just runs slower -- matches
# what patch_graft's own docstring already flags as a *speed* concern on
# a similar box, just worse here. Capping outright at a small constant
# (verified crash-free at this same repro) is safer than trying to
# compute "how many is too many" for an unknown machine.
_MAX_PATCH_WORKERS = 4


@dataclass
class GraFTResult:
    masks: list[np.ndarray]  # each (H, W) bool -- thresholded from graft's continuous spatial dictionary atoms
    traces: list[np.ndarray]  # each (T,) float, masked-mean of the fluorescence movie (matches every other
    # extraction method's trace convention -- NOT graft's own learned temporal dictionary column, which is a
    # model reconstruction basis rather than the measured signal)


def _finalize(movie: np.ndarray, spatial: np.ndarray) -> GraFTResult:
    """``spatial`` is (H, W, N) -- graft's/patch_graft's own ``S``/``Sm``
    output. Drops any dictionary atom that thresholds down to nothing
    (no real spatially-localized support -- not every learned atom
    corresponds to an actual cell)."""
    masks: list[np.ndarray] = []
    for i in range(spatial.shape[-1]):
        mask = threshold_footprint(spatial[:, :, i]) > 0
        if mask.any():
            masks.append(mask)
    traces = [masked_mean_trace(movie, mask) for mask in masks]
    return GraFTResult(masks=masks, traces=traces)


def graft_source_extraction(
    movie: np.ndarray, n_dict: int = 20, rng: np.random.Generator | None = None, **graft_params,
) -> GraFTResult:
    """Whole-FOV GraFT. ``rng`` is forwarded directly to ``graft.graft``
    (its own top-level parameter, controlling dictionary initialization
    -- NOT part of the ``params`` dict). ``graft_params`` are merged into
    the ``params`` dict ``graft.graft`` itself takes (e.g. ``max_learn``,
    ``lambda``) -- see ``graft.core._GRAFT_DEFAULTS`` for the full set;
    only ``n_dict`` is exposed as a named parameter here since it's the
    one every other extraction method in this app also surfaces directly
    (component count)."""
    _dict_temporal, spatial, _extras = graft.graft(
        movie, corr_kern=_DEFAULT_CORR_KERN, params={"n_dict": n_dict, **graft_params}, rng=rng,
    )
    return _finalize(movie, spatial)


def patch_graft_source_extraction(
    movie: np.ndarray,
    patch_size: tuple[int, int] = (50, 50),
    overlap: tuple[int, int] = (10, 10),
    n_dict_per_patch: int = 10,
    rng: np.random.Generator | None = None,
    max_workers: int | None = _MAX_PATCH_WORKERS,
    **graft_params,
) -> GraFTResult:
    """Patch-based GraFT -- splits the FOV into overlapping patches,
    runs GraFT independently per patch, and merges duplicate components
    found in each patch's overlap region (``graft.patch_graft`` handles
    all of this internally, including the merge step -- unlike orbit's
    own ``patch_cnmf_source_extraction``, which merges via a separate
    call to ``merge_overlapping_components``). ``rng``/``graft_params``
    -- see ``graft_source_extraction`` above. ``max_workers`` caps
    concurrent patch threads -- see ``_MAX_PATCH_WORKERS``'s comment for
    why this defaults to a small constant rather than ``patch_graft``'s
    own (much less conservative) default.

    Already memmap-safe by construction (confirmed in ``graft.patch_graft``'s
    own docstring): each patch only ever reads its own slice of
    ``movie``, so a real ``numpy.memmap`` input stays bounded by patch
    size rather than the whole movie -- this is why patch-based GraFT is
    required (not just offered) for a memmap-backed movie in the UI,
    same as patch-based CNMF.
    """
    patches = graft.construct_patches(movie.shape[:2], patch_size, overlap)
    _dict_temporal, spatial, _extras = graft.patch_graft(
        movie, n_dict=n_dict_per_patch, patches=patches, corr_kern=_DEFAULT_CORR_KERN, params=graft_params,
        rng=rng, max_workers=max_workers,
    )
    return _finalize(movie, spatial)
