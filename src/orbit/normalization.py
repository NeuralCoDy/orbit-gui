"""Centering and scale normalization for a (H, W, T) movie -- e.g. a
Delta-F/F-style transform. Ported from pyGraFT's graft/preprocessing.py
preprocess_data (the centering/normalization half only; filtering/
temporal averaging are orbit.denoising's job instead).
"""

from __future__ import annotations

import numpy as np

from .projections import _half_sample_mode_1d, mode_projection

_MAD_CONST = 0.6741891400433162  # matches robustSTD.m's exact constant


def robust_std(x: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    """Median-absolute-deviation-based robust standard deviation."""
    x = np.asarray(x, dtype=float)
    centered = x - np.median(x, axis=axis, keepdims=True)
    return np.median(np.abs(centered), axis=axis, keepdims=keepdims) / _MAD_CONST


def _mode_baseline(x: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    """half-sample mode, called the same way as np.median/np.mean/etc so
    it can sit in _CENTER_BASELINES alongside them. Only ``axis=None``
    (whole-array scalar) and ``axis=2`` (this module's per-pixel case)
    are used here; the latter reuses mode_projection's own native-C++
    fast path rather than re-deriving a slow apply_along_axis loop.
    """
    if axis is None:
        result = _half_sample_mode_1d(x.ravel())
        return np.full((1,) * x.ndim, result) if keepdims else result
    result = mode_projection(x)
    return result[:, :, None] if keepdims else result


_CENTER_BASELINES = {"median": np.median, "mean": np.mean, "min": np.min, "mode": _mode_baseline}
_NORM_BASELINES = {"median": np.median, "mean": np.mean, "max": np.max, "robuststd": robust_std}


def _center(movie: np.ndarray, baseline_name: str, pixel_wise: bool) -> np.ndarray:
    baseline_fn = _CENTER_BASELINES[baseline_name]
    baseline = baseline_fn(movie, axis=2, keepdims=True) if pixel_wise else baseline_fn(movie)
    return movie - baseline


def _normalize(movie: np.ndarray, baseline_name: str, pixel_wise: bool) -> np.ndarray:
    baseline_fn = _NORM_BASELINES[baseline_name]
    if pixel_wise:
        baseline = baseline_fn(movie, axis=2, keepdims=True)
        baseline = np.where(baseline == 0, 1.0, baseline)
    else:
        baseline = baseline_fn(movie)
        baseline = 1.0 if baseline == 0 else baseline
    return movie / baseline


def normalize_movie(
    movie: np.ndarray,
    *,
    center: bool = True,
    normalize: bool = True,
    center_baseline: str = "min",
    norm_baseline: str = "median",
    pixel_center: bool = False,
    pixel_norm: bool = False,
) -> np.ndarray:
    """Subtract a baseline (center) then divide by a scale (normalize).
    ``pixel_center``/``pixel_norm`` compute the baseline per pixel
    (reduced over time) rather than one global scalar for the whole
    movie. ``center_baseline`` in {"min", "median", "mean", "mode"},
    ``norm_baseline`` in {"median", "mean", "max", "robuststd"}.
    """
    movie = np.asarray(movie, dtype=float)
    if center:
        movie = _center(movie, center_baseline, pixel_center)
    if normalize:
        movie = _normalize(movie, norm_baseline, pixel_norm)
    return np.nan_to_num(movie, nan=0.0)


def summary_stats(movie: np.ndarray) -> dict[str, float]:
    """Basic distribution summary -- a quick sanity check that a
    centering/normalization transform did something reasonable (values
    aren't blowing up or collapsing to zero)."""
    return {
        "min": float(movie.min()),
        "max": float(movie.max()),
        "mean": float(movie.mean()),
        "std": float(movie.std()),
    }


def pixel_value_histogram(trace: np.ndarray, n_bins: int = 20) -> dict:
    """Binned histogram of one pixel's value distribution over time,
    plus its mean/median/mode."""
    trace = np.asarray(trace, dtype=float)
    counts, edges = np.histogram(trace, bins=n_bins)
    return {
        "edges": edges,
        "counts": counts,
        "mean": float(trace.mean()),
        "median": float(np.median(trace)),
        "mode": float(_half_sample_mode_1d(trace)),
    }
