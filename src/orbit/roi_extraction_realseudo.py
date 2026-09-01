"""Real-SEUDO source extraction: streams a movie through
orbit.seudo.streaming.realSEUDOfit one frame at a time, discovering and
promoting new cells as their activity is detected, rather than requiring a
whole-movie decomposition (PCA-ICA) or a fixed initial guess refined via
alternating updates (CNMF/CNMF-E) or a global sparse dictionary fit
(GraFT). See orbit.seudo.streaming's own module docstring for the
algorithm and its extensive real-data-tuning notes.

Unlike every other Source Extraction method in this app, this one is
memmap-safe by construction with no special casing needed: it reads and
fits exactly one frame at a time (`movie[:, :, t]`), so a numpy.memmap
movie never needs a preview cap (PCA-ICA) or a separate patch-based mode
(CNMF/CNMF-E/GraFT) to stay bounded in memory -- there simply isn't a
"whole movie in RAM" step to avoid here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._masks import masked_mean_trace
from .seudo.streaming import DetectionParams, FitParams, PromotionParams, StreamingState, realSEUDOfit


@dataclass
class RealSeudoResult:
    masks: list[np.ndarray]  # each (H, W) bool
    traces: list[np.ndarray]  # each (T,) float, masked-mean of the fluorescence movie (matches every other
    # extraction method's trace convention -- NOT the SEUDO-fitted activity coefficient, which is an internal
    # detail of the discovery process rather than something stored per-ROI in this first pass)


def real_seudo_source_extraction(
    movie: np.ndarray,
    sigma2: float = 0.0020,
    lambda_blob: float = 10.0,
    blob_radius: float = 3.0,
    pad_space: int = 5,
    lookahead_frames: int = 3,
    cutoff_multiplier: float = 4.0,
    min_roi_size: int = 50,
    min_avg_px: float = -1.0,
    mask_blur_rad: int = 1,
    consecutive_frames_required: int = 5,
    max_track_gap: int = 1,
    eq8_merge_threshold: float = 0.2,
    eq9_merge_threshold: float = 0.2,
    progress_callback=None,
) -> RealSeudoResult:
    """Runs realSEUDOfit across every frame of ``movie`` in order, starting
    from no prior knowledge of cell locations, then builds boolean masks
    from the final discovered profiles (thresholded the same way CNMF/
    GraFT's own continuous-valued footprints are, ``> 0``) and measured
    (masked-mean) traces from them.

    Defaults are the source project's own real-data-tuned values (see
    orbit.seudo.streaming's module docstring and its source repo's
    run_realseudo_full_movie.py) for the parameters exposed here -- a
    curated subset of FitParams/DetectionParams/PromotionParams' full set,
    matching this app's own "expose the impactful subset" convention (see
    e.g. cnmf_e_source_extraction's ring-model parameters). eq8_merge_
    threshold/eq9_merge_threshold=0.2 (not the paper's own 0.75) is
    required, not optional, given known-cell exclusion is unconditionally
    off (see streaming.py's own DetectionParams docstring) -- confirmed on
    real data that 0.75 here lets a rejected candidate re-spawn on top of
    an already-known cell every frame with nothing to catch it, an
    unbounded "cell" count (200+ on a 10,000-frame real recording) with
    per-frame cost climbing right along with it. ``progress_callback``, if
    given, is called as progress_callback(frames_done, total_frames) once
    per frame."""
    height, width, n_frames = movie.shape
    fit = FitParams(
        sigma2=sigma2, lambda_blob=lambda_blob, blob_radius=blob_radius,
        pad_space=pad_space, lookahead_frames=lookahead_frames,
    )
    detection = DetectionParams(
        cutoff_multiplier=cutoff_multiplier, min_roi_size=min_roi_size, min_avg_px=min_avg_px,
        mask_blur_rad=mask_blur_rad,
    )
    promotion = PromotionParams(
        consecutive_frames_required=consecutive_frames_required, max_track_gap=max_track_gap,
        eq8_merge_threshold=eq8_merge_threshold, eq9_merge_threshold=eq9_merge_threshold,
    )

    with StreamingState((height, width), fit=fit, detection=detection, promotion=promotion) as state:
        for t in range(n_frames):
            realSEUDOfit(movie[:, :, t], state)
            if progress_callback is not None:
                progress_callback(t + 1, n_frames)

        n_cells = state.profiles.shape[2]
        masks = [state.profiles[:, :, i] > 0 for i in range(n_cells)]

    traces = [masked_mean_trace(movie, mask) for mask in masks]
    return RealSeudoResult(masks=masks, traces=traces)
