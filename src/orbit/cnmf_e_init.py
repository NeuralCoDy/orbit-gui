"""CNMF-E initialization (Zhou et al. 2018): seeds components at local
peaks of correlation-image x peak-to-noise-ratio (PNR), rather than
plain CNMF's greedy_roi_init blurred-energy-peak-then-subtract loop --
1P/microendoscopic recordings have strong, spatially-varying background
fluorescence that swamps plain intensity peaks, but a real cell still
stands out as BOTH locally correlated (its footprint's pixels covary)
AND high-PNR (a real calcium transient's peak is many noise-std above
its own baseline). This is CNMF-E's actual init strategy, not a
simplification of it: the real algorithm also picks all qualifying,
mutually-separated peaks up front rather than a sequential greedy
subtraction.

Reuses cnmf_init.gaussian_blur_movie/finetune_component, _masks.disk_mask,
projections.local_correlation_projection, and
_peak_picking.select_separated_pixels (the same peak-picker
roi_extraction_corr.find_seed_candidates already uses) -- only the PNR
projection and the corr*PNR combination/thresholding are new.
"""

from __future__ import annotations

import numpy as np

from ._blocks import chunked_median
from ._masks import disk_mask
from ._peak_picking import select_separated_pixels
from .cnmf_deconvolution import _fast_welch_psd
from .cnmf_init import finetune_component, gaussian_blur_movie
from .projections import local_correlation_projection


def noise_std_projection(movie: np.ndarray, freq_range: tuple[float, float] = (0.25, 0.5)) -> np.ndarray:
    """Per-pixel high-frequency-band noise estimate -- the same Welch-PSD
    formula as cnmf_deconvolution.estimate_noise_std, applied to every
    pixel's trace at once via _fast_welch_psd's own batched last-axis
    support instead of a Python loop calling that function P times
    (confirmed numerically identical to the per-trace function, output
    for output)."""
    height, width, n_frames = movie.shape
    flat = np.ascontiguousarray(movie, dtype=np.float64).reshape(-1, n_frames)
    freqs, psd = _fast_welch_psd(flat, nperseg=min(n_frames, 256))
    band = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if not band.any():
        band = freqs >= freq_range[0]
    noise = np.sqrt(np.exp(np.mean(np.log(psd[:, band] / 2 + 1e-10), axis=1)))
    return noise.reshape(height, width)


def peak_to_noise_ratio_projection(movie: np.ndarray) -> np.ndarray:
    """(peak - median) / noise_std per pixel -- CNMF-E's PNR image: a
    real calcium transient's peak sits many noise-std above its own
    quiescent baseline, unlike a purely noisy or slowly-drifting
    background pixel."""
    peak = movie.max(axis=2) - chunked_median(movie, axis=2, max_block=150)
    noise = noise_std_projection(movie)
    return peak / np.maximum(noise, 1e-6)


def cnmf_e_seed_candidates(
    movie: np.ndarray, gauss_sigma: float, min_corr: float, min_pnr: float, n_components: int,
    min_separation_frac: float = 0.05, blurred_movie: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Auto-picks up to ``n_components`` seed pixels from peaks of
    corr-image x PNR-image, mutually separated by at least
    ``min_separation_frac`` of the field of view's longer dimension, and
    above BOTH ``min_corr`` and ``min_pnr`` -- mirrors
    roi_extraction_corr.find_seed_candidates's exact shape (rank via
    select_separated_pixels first, then post-filter the returned
    candidates by threshold, rather than pre-zeroing the score map,
    since select_separated_pixels doesn't itself filter by value).

    ``blurred_movie``, if given, is used directly instead of re-blurring
    ``movie`` -- cnmf_e_init already builds its own blurred copy (needed
    for finetune_component too) and passes it through here, since blurring
    the whole movie is real, non-trivial cost (profiling found it ~19% of
    a whole-FOV CNMF-E run) not worth paying twice per call. None (the
    default) blurs movie itself, for standalone/test use."""
    blurred = blurred_movie if blurred_movie is not None else gaussian_blur_movie(movie, gauss_sigma)
    corr_map = local_correlation_projection(blurred)
    pnr_map = peak_to_noise_ratio_projection(blurred)
    score = corr_map * pnr_map

    min_dist = min_separation_frac * max(score.shape)
    candidates = select_separated_pixels(score, n_components, min_dist, descending=True)
    return [(r, c) for r, c in candidates if corr_map[r, c] >= min_corr and pnr_map[r, c] >= min_pnr]


def cnmf_e_init(
    movie: np.ndarray, n_components: int, gauss_sigma: float = 2.0, init_radius: float = 5.0,
    min_corr: float = 0.8, min_pnr: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Seeds up to ``n_components`` footprints at corr*PNR peaks (see
    cnmf_e_seed_candidates) and refines each via finetune_component --
    identical per-seed refinement to greedy_roi_init, just with CNMF-E's
    own seed selection swapped in for greedy_roi_init's sequential
    blurred-energy-peak-then-subtract loop. Returns (footprints
    (K, H, W), traces (K, T)); K may be less than n_components if fewer
    pixels qualify."""
    height, width, n_frames = movie.shape
    blurred = gaussian_blur_movie(movie, gauss_sigma)
    # cnmf_e_seed_candidates's own default min_separation_frac (0.05,
    # inherited from roi_extraction_corr.find_seed_candidates's
    # manual-review context, where a duplicate seed on the same cell is
    # harmless -- a human just discards it) is too small relative to a
    # typical cell footprint for this fully-automated pipeline: unlike
    # greedy_roi_init, seeds here aren't picked sequentially with
    # subtraction in between, so a strong cell's own extended
    # corr*PNR plateau can otherwise soak up every requested seed
    # before a second, real cell is ever reached. Deriving the minimum
    # separation from init_radius (this pipeline's own expected cell
    # size) instead of a generic image-size fraction keeps seeds spread
    # across distinct cells regardless of image size.
    min_separation_frac = (1.5 * init_radius) / max(height, width)
    seeds = cnmf_e_seed_candidates(
        movie, gauss_sigma, min_corr, min_pnr, n_components, min_separation_frac, blurred_movie=blurred,
    )

    if not seeds:
        return np.zeros((0, height, width)), np.zeros((0, n_frames))

    footprints, traces = [], []
    for seed in seeds:
        mask = disk_mask(height, width, seed, init_radius)
        footprint, trace = finetune_component(blurred, mask)
        footprints.append(footprint)
        traces.append(trace)

    return np.stack(footprints), np.stack(traces)
