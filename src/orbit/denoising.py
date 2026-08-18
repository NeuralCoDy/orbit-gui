"""Denoising for a (H, W, T) movie: wavelet shrinkage (ported from
pyGraFT's graft/preprocessing.py denoise_wavelet_time/denoise_wavelet_space
-- functionally equivalent VisuShrink/BayesShrink denoisers via
PyWavelets, batched over every pixel-trace or every frame in one pywt
call rather than looping in Python), plus Gaussian and median filtering
via scipy.ndimage (each a single vectorized call, no Python-level loop
over pixels/frames either).

residual_energy_fraction is the "residual energy" assessment metric --
baked in here rather than left to a separate QC pass, per this project's
whole premise.
"""

from __future__ import annotations

import numpy as np
import pywt
from scipy.ndimage import gaussian_filter, median_filter

from .normalization import robust_std
from .projections import local_correlation_projection


def _soft_threshold(x: np.ndarray, thresh: np.ndarray | float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


def _threshold_subband(detail: np.ndarray, sigma_noise: np.ndarray, method: str, axis) -> np.ndarray:
    """Soft-thresholds one wavelet subband. ``sigma_noise`` is already
    shaped to broadcast against ``detail`` via keepdims (one noise
    estimate per pixel-trace or per-frame, batched along a leading
    axis). ``axis`` is the subband's own spatial axis/axes (-1 for 1D,
    (-2, -1) for 2D), matching what ``sigma_noise`` was reduced over."""
    if method == "bayes":
        sigma_w = np.std(detail, axis=axis, keepdims=True)
        sigma_x = np.sqrt(np.maximum(sigma_w**2 - sigma_noise**2, 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            safe_sigma_x = np.where(sigma_x > 0, sigma_x, 1.0)
            thresh = np.where(
                sigma_x > 0, sigma_noise**2 / safe_sigma_x, np.abs(detail).max(axis=axis, keepdims=True)
            )
    elif method == "universal":
        n = np.prod([detail.shape[a] for a in np.atleast_1d(axis)])
        thresh = sigma_noise * np.sqrt(2.0 * np.log(max(n, 2)))
    else:
        raise ValueError(f"Unknown wavelet threshold method: {method!r}")
    return _soft_threshold(detail, thresh)


def _denoise_wavelet_batch(flat: np.ndarray, wavelet: str, level: int, method: str) -> np.ndarray:
    """Denoise every row of ``flat`` (n_signals, T) independently along
    the last axis in one pywt call per subband."""
    n_frames = flat.shape[-1]
    max_level = pywt.dwt_max_level(n_frames, pywt.Wavelet(wavelet).dec_len)
    lvl = min(level, max_level)
    if lvl < 1:
        return flat.copy()
    coeffs = pywt.wavedec(flat, wavelet, level=lvl, axis=-1)
    sigma = robust_std(coeffs[-1], axis=-1, keepdims=True)
    new_coeffs = [coeffs[0]] + [_threshold_subband(d, sigma, method, axis=-1) for d in coeffs[1:]]
    denoised = pywt.waverec(new_coeffs, wavelet, axis=-1)
    return denoised[..., :n_frames]


def _denoise_wavelet_2d_batch(cube: np.ndarray, wavelet: str, level: int, method: str) -> np.ndarray:
    """Denoise every frame of ``cube`` (T, H, W) independently (2D
    wavelet shrinkage), batched over the leading T axis."""
    height, width = cube.shape[-2:]
    max_level = pywt.dwt_max_level(min(height, width), pywt.Wavelet(wavelet).dec_len)
    lvl = min(level, max_level)
    if lvl < 1:
        return cube.copy()
    coeffs = pywt.wavedec2(cube, wavelet, level=lvl, axes=(-2, -1))
    sigma = robust_std(coeffs[-1][2], axis=(-2, -1), keepdims=True)  # finest diagonal-detail subband
    new_coeffs = [coeffs[0]]
    for level_details in coeffs[1:]:
        new_coeffs.append(tuple(_threshold_subband(d, sigma, method, axis=(-2, -1)) for d in level_details))
    denoised = pywt.waverec2(new_coeffs, wavelet, axes=(-2, -1))
    return denoised[..., :height, :width]


def denoise_wavelet_time(mov: np.ndarray, wavelet: str = "sym4", level: int = 4, method: str = "bayes") -> np.ndarray:
    """Denoise each pixel's time-trace independently via wavelet
    shrinkage. ``mov`` is (H, W, T)."""
    orig_shape = mov.shape
    flat = np.asarray(mov, dtype=float).reshape(-1, orig_shape[-1])
    return _denoise_wavelet_batch(flat, wavelet, level, method).reshape(orig_shape)


def denoise_wavelet_space(mov: np.ndarray, wavelet: str = "sym4", level: int = 4, method: str = "bayes") -> np.ndarray:
    """Denoise each frame independently (2D) via wavelet shrinkage.
    ``mov`` is (H, W, T)."""
    cube = np.moveaxis(np.asarray(mov, dtype=float), -1, 0)  # (H, W, T) -> (T, H, W)
    denoised = _denoise_wavelet_2d_batch(cube, wavelet, level, method)
    return np.moveaxis(denoised, 0, -1)


def denoise_gaussian(mov: np.ndarray, spatial_sigma: float = 0.0, temporal_sigma: float = 0.0) -> np.ndarray:
    """Gaussian-smooth a movie with independent spatial and temporal
    widths. A sigma of 0 skips smoothing along that axis entirely, so
    spatial_sigma=0 is purely temporal smoothing and temporal_sigma=0 is
    purely spatial (per-frame) smoothing."""
    return gaussian_filter(mov, sigma=(spatial_sigma, spatial_sigma, temporal_sigma))


def denoise_median(mov: np.ndarray, space_window: int = 3, time_window: int = 1) -> np.ndarray:
    """Median filter over a window that's square in space
    (``space_window`` x ``space_window``) and ``time_window`` frames
    deep. ``mode="nearest"`` (edge-replicated) avoids the boundary
    darkening a zero-padded median filter would introduce."""
    return median_filter(mov, size=(space_window, space_window, time_window), mode="nearest")


def _select_separated_pixels(score_map: np.ndarray, n_points: int, min_dist: float, descending: bool) -> list[tuple[int, int]]:
    """Greedily picks up to ``n_points`` pixels in ranked order of
    ``score_map`` (highest first if ``descending``, else lowest first),
    skipping any candidate closer than ``min_dist`` to an already-picked
    point -- so the result isn't just a cluster of neighboring pixels
    that all happen to share the extreme score."""
    height, width = score_map.shape
    order = np.argsort(score_map, axis=None)
    if descending:
        order = order[::-1]

    selected: list[tuple[int, int]] = []
    for idx in order:
        r, c = divmod(int(idx), width)
        if all(np.hypot(r - sr, c - sc) >= min_dist for sr, sc in selected):
            selected.append((r, c))
            if len(selected) == n_points:
                break
    return selected


def qc_trace_samples(
    before: np.ndarray, after: np.ndarray, n_peaks: int = 2, n_low: int = 2, min_separation_frac: float = 0.2
) -> list[dict]:
    """Picks representative pixel locations from ``before``'s local
    correlation map -- ``n_peaks`` correlation maxima (likely cell
    footprints) and ``n_low`` correlation minima (likely background/
    noise), each set mutually separated by at least
    ``min_separation_frac`` of the field of view's longer dimension --
    and returns each location's before/after time trace, for a
    qualitative sanity check alongside the aggregate metrics."""
    corr_map = local_correlation_projection(before)
    min_dist = min_separation_frac * max(corr_map.shape)
    locations = [("peak", r, c) for r, c in _select_separated_pixels(corr_map, n_peaks, min_dist, descending=True)]
    locations += [("low", r, c) for r, c in _select_separated_pixels(corr_map, n_low, min_dist, descending=False)]
    return [
        {
            "kind": kind,
            "row": r,
            "col": c,
            "corr": float(corr_map[r, c]),
            "before": np.asarray(before[r, c, :], dtype=float),
            "after": np.asarray(after[r, c, :], dtype=float),
        }
        for kind, r, c in locations
    ]


def residual_energy_fraction(before: np.ndarray, after: np.ndarray) -> float:
    """Fraction of signal energy removed by a denoising step:
    ||before - after||^2 / ||before||^2. Higher means more of the
    original signal was treated as noise and subtracted -- a value so
    high it looks implausible for genuine noise is a sign real signal is
    being removed, not just noise."""
    residual = before.astype(np.float64) - after.astype(np.float64)
    denom = np.sum(before.astype(np.float64) ** 2)
    return float(np.sum(residual**2) / denom) if denom > 0 else 0.0
