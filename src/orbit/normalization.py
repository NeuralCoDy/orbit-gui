"""Centering and scale normalization for a (H, W, T) movie -- e.g. a
Delta-F/F-style transform. Ported from pyGraFT's graft/preprocessing.py
preprocess_data (the centering/normalization half only; filtering/
temporal averaging are orbit.denoising's job instead).
"""

from __future__ import annotations

import numpy as np

from ._blocks import _MEDIAN_SLAB_BYTES, chunked_median, iter_axis_slices
from .projections import _half_sample_mode_1d, mode_projection

_MAD_CONST = 0.6741891400433162  # matches robustSTD.m's exact constant
_MEDIAN_BLOCK_PIXELS = 150  # cap on the chunked spatial axis -- see orbit._blocks.chunked_median


def robust_std(x: np.ndarray, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> np.ndarray:
    """Median-absolute-deviation-based robust standard deviation.

    A single non-None ``axis`` -- the per-pixel/per-voxel baseline case,
    on a movie/volume that can be multi-GB -- is chunked (see
    _chunked_robust_std). ``axis=None`` (one global scalar) or a
    multi-axis tuple (e.g. orbit.denoising's per-subband
    ``robust_std(coeffs, axis=(-2, -1))``, reducing a small wavelet
    coefficient array over both its spatial axes at once) fall back to
    the plain formula unchanged: an exact global/multi-axis median
    needs every value in that reduction at once, so it can't be spatially
    blocked the way a single-axis-at-a-time reduction can -- and neither
    case here is ever movie-scale anyway (multi-axis calls are always on
    one already-small wavelet subband, not a full movie)."""
    x = np.asarray(x, dtype=float)
    if isinstance(axis, int):
        return _chunked_robust_std(x, axis, keepdims)
    centered = x - np.median(x, axis=axis, keepdims=True)
    return np.median(np.abs(centered), axis=axis, keepdims=keepdims) / _MAD_CONST


def _chunked_robust_std(x: np.ndarray, axis: int, keepdims: bool) -> np.ndarray:
    """``robust_std(x, axis=axis, keepdims=keepdims)`` computed one
    spatial block at a time. The two-pass MAD algorithm (median, then
    median of abs deviations from it) would otherwise need a full
    movie-sized ``x - median`` array between passes on top of
    np.median's own internal partition copy -- each block instead runs
    both passes on just its own slab, in place, before moving on, so
    neither of those is ever materialized at full size."""
    moved = np.moveaxis(x, axis, 0)  # reduction axis -> 0 (a view)
    out = np.empty(moved.shape[1:], dtype=np.float64)
    for sl in iter_axis_slices(moved.shape, 1, target_bytes=_MEDIAN_SLAB_BYTES, itemsize=8, max_step=_MEDIAN_BLOCK_PIXELS):
        block = np.array(moved[:, sl], dtype=np.float64)  # own copy -- safe to overwrite in place
        center = np.median(block, axis=0, keepdims=True)
        block -= center
        np.abs(block, out=block)
        out[sl] = np.median(block, axis=0, overwrite_input=True)
    out /= _MAD_CONST
    return np.expand_dims(out, axis) if keepdims else out


def _chunked_median_baseline(movie: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    """Drop-in for np.median in _CENTER_BASELINES/_NORM_BASELINES: a
    real ``axis`` (the pixel-wise baseline case) routes through the one
    shared chunked_median implementation instead of a plain np.median
    call on the whole movie; ``axis=None`` (a single global scalar
    baseline) falls back to plain np.median -- see robust_std's
    docstring for why a global median can't be chunked the same way."""
    if axis is None:
        return np.median(movie)
    return chunked_median(movie, axis=axis, keepdims=keepdims, max_block=_MEDIAN_BLOCK_PIXELS)


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


_CENTER_BASELINES = {"median": _chunked_median_baseline, "mean": np.mean, "min": np.min, "mode": _mode_baseline}
_NORM_BASELINES = {"median": _chunked_median_baseline, "mean": np.mean, "max": np.max, "robuststd": robust_std}


def _center_baseline(movie: np.ndarray, baseline_name: str, pixel_wise: bool) -> np.ndarray:
    baseline_fn = _CENTER_BASELINES[baseline_name]
    return baseline_fn(movie, axis=2, keepdims=True) if pixel_wise else baseline_fn(movie)


def _norm_baseline(movie: np.ndarray, baseline_name: str, pixel_wise: bool) -> np.ndarray:
    baseline_fn = _NORM_BASELINES[baseline_name]
    if pixel_wise:
        baseline = baseline_fn(movie, axis=2, keepdims=True)
        return np.where(baseline == 0, 1.0, baseline)
    baseline = baseline_fn(movie)
    return 1.0 if baseline == 0 else baseline


def compute_baselines(
    movie: np.ndarray,
    *,
    center: bool = True,
    normalize: bool = True,
    center_baseline: str = "min",
    norm_baseline: str = "median",
    pixel_center: bool = False,
    pixel_norm: bool = False,
) -> dict:
    """Computes (without applying) the center/normalize baselines
    normalize_movie() would use for ``movie``. Split out from
    normalize_movie so a caller can fit baselines once from a
    representative sample and apply them to a *different*, possibly
    much larger array later via apply_baselines() -- see
    orbitapp.tabs.normalization_tab.NormalizationTab._chunked_commit,
    which fits from the Apply preview and reuses the result across every
    chunk of a memory-mapped movie's full-length Commit, since baseline
    statistics like ``median``/``mode``/``robuststd`` can't be computed
    exactly from streamed chunks without keeping every value anyway.
    """
    movie = np.asarray(movie, dtype=float)
    baselines: dict = {}
    if center:
        baselines["center"] = _center_baseline(movie, center_baseline, pixel_center)
    if normalize:
        baselines["normalize"] = _norm_baseline(movie, norm_baseline, pixel_norm)
    return baselines


def apply_baselines(movie: np.ndarray, baselines: dict, *, center: bool = True, normalize: bool = True) -> np.ndarray:
    """Applies baselines from compute_baselines() (computed against this
    same movie, or a different one -- e.g. a representative sample) to
    ``movie``. A per-pixel baseline broadcasts against any T, so this
    works the same whether ``movie`` is the whole movie the baseline was
    fit from or just one time-chunk of a much longer one."""
    movie = np.asarray(movie, dtype=float)
    if center and "center" in baselines:
        movie = movie - baselines["center"]
    if normalize and "normalize" in baselines:
        movie = movie / baselines["normalize"]
    return np.nan_to_num(movie, nan=0.0)


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
    baselines = compute_baselines(
        movie, center=center, normalize=normalize, center_baseline=center_baseline, norm_baseline=norm_baseline,
        pixel_center=pixel_center, pixel_norm=pixel_norm,
    )
    return apply_baselines(movie, baselines, center=center, normalize=normalize)


def describe_normalization(
    center: bool, center_baseline: str, pixel_center: bool, normalize: bool, norm_baseline: str, pixel_norm: bool
) -> str:
    """Human-readable summary of a normalize_movie() call, e.g.
    "(pixel-mode, pixel-median)" or "([], all-median)" -- used as the
    pipeline-breadcrumb detail for a committed normalization step, since
    "Normalize" alone doesn't say which baseline/scope was actually used."""

    def _describe(enabled: bool, baseline: str, pixel_wise: bool) -> str:
        if not enabled:
            return "[]"
        return f"{'pixel' if pixel_wise else 'all'}-{baseline}"

    return f"({_describe(center, center_baseline, pixel_center)}, {_describe(normalize, norm_baseline, pixel_norm)})"


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
