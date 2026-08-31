"""Photobleaching/intensity-drift correction: divides the movie by a
slowly-varying running-percentile trend of its own field-of-view-average
intensity, so a monotonic (or otherwise slow) drift in overall
brightness doesn't get mistaken for real signal by later stages
(Normalization, Source Extraction). Multiplicative (divide), matching
how photobleaching actually behaves -- fluorophore depletion scales
brightness down over time, it doesn't just shift it by a constant --
not an additive subtraction.

The trend is a *trailing* (causal) running percentile: trend[t] only
depends on frames up to and including t, never frames after it, the
same convention as a causal rolling-baseline filter (e.g. Suite2p's
rolling-percentile baseline).

fov_average_trace/detrend_movie are the (H, W, T) 2D versions;
fov_average_trace_3d/detrend_movie_3d are the (T, L, W, D) volumetric
counterparts (mean over all three spatial axes per volume instead of
the two per frame) -- running_percentile_trend itself is dimension-
agnostic (just a 1D trace in, 1D trend out), so it's shared unmodified.
"""

from __future__ import annotations

import numpy as np


def fov_average_trace(movie: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """(T,) mean pixel intensity per frame across the field of view,
    normalized to the FIRST frame's own mean (so the trace and trend
    both start at 1.0 and read directly as "fraction of initial
    brightness" -- comparable across movies of very different absolute
    intensity scale). Purely a display/inspection convenience: rescaling
    a trace by a positive constant changes neither running_percentile_trend's
    trend shape nor detrend_movie's actual `scale = trend / trend.mean()`
    correction (both percentile and mean commute with positive scalar
    multiplication, so the constant cancels out of that ratio) --
    confirmed via test_detrend_movie_correction_is_unaffected_by_the_
    trace_s_own_normalization.

    ``mask`` (H, W) bool, e.g. from a committed Mask stage -- restricts
    the average to True pixels only, so blank background/empty FOV space
    doesn't dilute a real intensity change happening only in the imaged
    tissue. None (the default) averages every pixel, unmasked."""
    movie = np.asarray(movie, dtype=np.float64)
    trace = movie.mean(axis=(0, 1)) if mask is None else movie[mask].mean(axis=0)
    baseline = trace[0]
    return trace / baseline if baseline != 0 else trace


def fov_average_trace_3d(movie: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """(T,) mean voxel intensity per volume -- (T, L, W, D) counterpart of
    fov_average_trace; ``mask`` is (L, W, D). See that function's
    docstring for the first-volume normalization and mask semantics,
    identical here."""
    movie = np.asarray(movie, dtype=np.float64)
    trace = movie.mean(axis=(1, 2, 3)) if mask is None else movie[:, mask].mean(axis=1)
    baseline = trace[0]
    return trace / baseline if baseline != 0 else trace


def running_percentile_trend(trace: np.ndarray, percentile: float, window: int) -> np.ndarray:
    """Trailing-window Xth-percentile trend of a 1D trace: trend[t] is
    the ``percentile``-th percentile of trace[max(0, t-window+1) : t+1].
    The interior (once a full window is available, t >= window-1) is
    computed in one vectorized call via a sliding-window view rather
    than a per-frame Python loop; only the short growing-window prefix
    (t < window-1, at most window-1 frames) is looped, since that
    region can't share the same fixed-size window shape."""
    trace = np.asarray(trace, dtype=np.float64)
    n = len(trace)
    window = int(np.clip(window, 1, max(n, 1)))
    trend = np.empty(n)

    for t in range(min(window - 1, n)):
        trend[t] = np.percentile(trace[: t + 1], percentile)

    if n >= window:
        windows = np.lib.stride_tricks.sliding_window_view(trace, window)  # (n-window+1, window)
        trend[window - 1 :] = np.percentile(windows, percentile, axis=-1)

    return trend


def detrend_movie(
    movie: np.ndarray, percentile: float = 8.0, window: int = 200, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full pipeline: fov_average_trace -> running_percentile_trend ->
    divide every pixel's trace by the trend, rescaled to the trend's own
    mean so the movie's overall intensity scale is preserved (only the
    slow drift is removed, not the average brightness level). Returns
    (corrected movie, fov trace, trend) -- the trace/trend are what the
    GUI plots. ``mask`` (H, W) bool restricts fov_average_trace to True
    pixels only -- see that function's docstring."""
    trace = fov_average_trace(movie, mask)
    trend = running_percentile_trend(trace, percentile, window)
    scale = trend / trend.mean()
    corrected = movie / scale[None, None, :]
    return corrected, trace, trend


def detrend_movie_3d(
    movie: np.ndarray, percentile: float = 8.0, window: int = 200, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Volumetric (T, L, W, D) counterpart of detrend_movie -- ``mask``
    is (L, W, D). See fov_average_trace_3d for the per-volume average."""
    trace = fov_average_trace_3d(movie, mask)
    trend = running_percentile_trend(trace, percentile, window)
    scale = trend / trend.mean()
    corrected = movie / scale[:, None, None, None]
    return corrected, trace, trend
