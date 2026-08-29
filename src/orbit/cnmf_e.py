"""CNMF-E source extraction (Zhou et al. 2018) for 1P/microendoscopic
data: the same alternating spatial/temporal update loop as cnmf.py's
plain CNMF, but with two differences that matter for 1P recordings'
strong, spatially-varying background fluorescence -- (1) init seeds at
correlation-image x peak-to-noise-ratio peaks instead of plain
intensity peaks (cnmf_e_init.py), and (2) a per-pixel "ring model"
background (cnmf_e_background.py) is fit once and subtracted from the
movie before/during the loop, instead of relying solely on plain CNMF's
global low-rank background.

Deliberately reuses cnmf.py's update_spatial_components/
update_temporal_components/merge_overlapping_components/CNMFResult/
_finalize_result UNCHANGED -- neither update function cares whether the
movie it's given already had a background subtracted from it, so the
ring-corrected movie is a legitimate drop-in for the raw movie those
functions were written against. patch_cnmf_e_source_extraction below
mirrors cnmf.py's patch_cnmf_source_extraction structure exactly.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ._masks import threshold_footprint
from ._patches import make_patches_2d
from .cnmf import (
    _DEFAULT_MAX_WORKERS,
    _MP_CONTEXT,
    CNMFResult,
    _finalize_result,
    _single_threaded_blas_for_children,
    merge_overlapping_components,
    update_spatial_components,
    update_temporal_components,
)
from .cnmf_deconvolution import estimate_ar1_coefficient, estimate_noise_std
from .cnmf_e_background import fit_ring_model, predict_ring_model_background
from .cnmf_e_init import cnmf_e_init
from .cnmf_init import estimate_background


def cnmf_e_source_extraction(
    movie: np.ndarray,
    n_components: int = 20,
    gauss_sigma: float = 2.0,
    init_radius: float = 5.0,
    min_corr: float = 0.8,
    min_pnr: float = 8.0,
    search_radius: float = 10.0,
    ring_inner_radius: float = 20.0,
    ring_outer_radius: float = 25.0,
    ring_downsample: int = 4,
    ring_max_fit_frames: int = 500,
    n_background_components: int = 1,
    merge_thresh: float = 0.8,
    n_iterations: int = 2,
) -> CNMFResult:
    """Full pipeline: corr*PNR init -> fit the ring background ONCE ->
    [subtract the current ring-background prediction -> spatial update ->
    threshold -> temporal update (+ OASIS) -> merge] x n_iterations ->
    final masks. See patch_cnmf_e_source_extraction for a version that
    splits a large field of view into patches first."""
    footprints, traces = cnmf_e_init(movie, n_components, gauss_sigma, init_radius, min_corr, min_pnr)
    height, width, _n_frames = movie.shape
    n_found = len(footprints)
    if n_found == 0:
        return CNMFResult(masks=[], traces=[], spike_traces=[])

    recon = (footprints.reshape(n_found, -1).T @ traces).reshape(height, width, -1)
    residual = movie - recon
    neuron_mask = footprints.sum(axis=0) > 0

    # The expensive step -- fit once. real CNMF-E also only refits ring
    # WEIGHTS rarely, then cheaply re-applies them to the current
    # residual each iteration (predict_ring_model_background below).
    ring_model = fit_ring_model(
        residual, neuron_mask, ring_inner_radius, ring_outer_radius, ring_downsample, ring_max_fit_frames
    )
    ring_background = predict_ring_model_background(residual, ring_model)
    ring_corrected = movie - ring_background

    # A small rank-1 low-rank term on top of the ring-corrected movie
    # mops up whatever slowly-varying drift the ring model doesn't
    # capture -- the same call cnmf_source_extraction makes today, just
    # on the ring-corrected residual instead of the raw one. The two
    # background models don't conflict, they're stacked.
    background_spatial, background_temporal = estimate_background(
        np.clip(ring_corrected - recon, 0, None), n_background_components
    )

    g_list = [estimate_ar1_coefficient(t) for t in traces]
    noise_stds = [estimate_noise_std(t) for t in traces]
    spike_traces = np.zeros_like(traces)

    for iteration in range(n_iterations):
        footprints, background_spatial = update_spatial_components(
            ring_corrected, footprints, traces, background_temporal, search_radius
        )
        footprints = np.stack([threshold_footprint(f) for f in footprints])
        traces, spike_traces, background_temporal = update_temporal_components(
            ring_corrected, footprints, traces, background_spatial, background_temporal, g_list, noise_stds
        )
        if iteration < n_iterations - 1:
            footprints, traces, spike_traces, g_list = merge_overlapping_components(
                footprints, traces, spike_traces, g_list, merge_thresh
            )
            noise_stds = [estimate_noise_std(t) for t in traces]
            # Refresh the ring-corrected movie from the just-updated
            # footprints/traces (cheap: same fitted ring_model, just
            # re-predicted from the new residual -- see
            # predict_ring_model_background) so the next iteration's
            # spatial/temporal update sees an up-to-date background
            # subtraction. This allocates one extra movie-sized array
            # per iteration beyond what cnmf_source_extraction does
            # today; n_iterations is small (default 2) so it's a minor,
            # not dominant, cost.
            n_found = len(footprints)
            recon = (footprints.reshape(n_found, -1).T @ traces).reshape(height, width, -1)
            residual = movie - recon
            ring_background = predict_ring_model_background(residual, ring_model)
            ring_corrected = movie - ring_background

    # Measured traces come from the RAW movie, not ring_corrected --
    # matches CNMFResult's own documented convention (every extraction
    # method's "trace" is the measured signal from the movie as
    # committed, not any one method's internal background-subtracted
    # working copy) and plain CNMF's own precedent (its background
    # estimate isn't subtracted from its traces either).
    return _finalize_result(movie, footprints, spike_traces)


def _run_patch_e(
    movie_patch: np.ndarray, n_components_per_patch: int, merge_thresh: float, cnmf_e_kwargs: dict
) -> CNMFResult:
    """Module-level (rather than a closure) so ProcessPoolExecutor can
    pickle a reference to it for each worker process."""
    return cnmf_e_source_extraction(
        movie_patch, n_components=n_components_per_patch, merge_thresh=merge_thresh, **cnmf_e_kwargs
    )


def patch_cnmf_e_source_extraction(
    movie: np.ndarray,
    patch_size: tuple[int, int] = (80, 80),
    overlap: int = 20,
    n_components_per_patch: int = 10,
    merge_thresh: float = 0.8,
    progress_callback=None,
    max_workers: int | None = _DEFAULT_MAX_WORKERS,
    **cnmf_e_kwargs,
) -> CNMFResult:
    """Runs cnmf_e_source_extraction independently on overlapping
    spatial patches, then merges components found in more than one
    patch's overlap region -- mirrors
    cnmf.patch_cnmf_source_extraction's exact structure (same
    ProcessPoolExecutor/_single_threaded_blas_for_children/
    make_patches_2d/merge pattern), with cnmf_e_source_extraction as the
    per-patch call.

    Warns if ``overlap`` is smaller than the ring model's own
    ``ring_outer_radius`` (default 25.0, or whatever's passed via
    cnmf_e_kwargs): a too-small overlap starves the ring model of
    context near patch boundaries, silently degrading the background
    fit there rather than raising an error -- worth an explicit warning
    since this is a correctness footgun, not just a style nit (mirrors
    e.g. io.py's memmap-fallback warning)."""
    ring_outer_radius = cnmf_e_kwargs.get("ring_outer_radius", 25.0)
    if overlap < ring_outer_radius:
        warnings.warn(
            f"patch_cnmf_e_source_extraction: overlap ({overlap}) is smaller than ring_outer_radius "
            f"({ring_outer_radius}) -- the ring model near patch boundaries will be starved of context, "
            "degrading the background fit there. Increase overlap or shrink ring_outer_radius.",
            stacklevel=2,
        )

    height, width, _n_frames = movie.shape
    patches = make_patches_2d(height, width, patch_size, overlap)
    workers = max_workers if max_workers is not None else min(_DEFAULT_MAX_WORKERS, os.cpu_count() or 1, len(patches))

    all_footprints, all_traces, all_spikes, all_g = [], [], [], []
    with _single_threaded_blas_for_children(), ProcessPoolExecutor(max_workers=workers, mp_context=_MP_CONTEXT) as pool:
        futures = [
            pool.submit(_run_patch_e, movie[r0:r1, c0:c1, :], n_components_per_patch, merge_thresh, cnmf_e_kwargs)
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
