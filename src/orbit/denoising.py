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


def denoise_gaussian_time(mov: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth each pixel's time-trace independently (no
    spatial blur). ``sigma`` is the Gaussian's width, in frames."""
    return gaussian_filter(mov, sigma=(0, 0, sigma))


def denoise_gaussian_space(mov: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-blur each frame independently (no temporal blur).
    ``sigma`` is the Gaussian's width, in pixels."""
    return gaussian_filter(mov, sigma=(sigma, sigma, 0))


def denoise_median(mov: np.ndarray, space_window: int = 3, time_window: int = 1) -> np.ndarray:
    """Median filter over a window that's square in space
    (``space_window`` x ``space_window``) and ``time_window`` frames
    deep. ``mode="nearest"`` (edge-replicated) avoids the boundary
    darkening a zero-padded median filter would introduce."""
    return median_filter(mov, size=(space_window, space_window, time_window), mode="nearest")


def residual_energy_fraction(before: np.ndarray, after: np.ndarray) -> float:
    """Fraction of signal energy removed by a denoising step:
    ||before - after||^2 / ||before||^2. Higher means more of the
    original signal was treated as noise and subtracted -- a value so
    high it looks implausible for genuine noise is a sign real signal is
    being removed, not just noise."""
    residual = before.astype(np.float64) - after.astype(np.float64)
    denom = np.sum(before.astype(np.float64) ** 2)
    return float(np.sum(residual**2) / denom) if denom > 0 else 0.0
