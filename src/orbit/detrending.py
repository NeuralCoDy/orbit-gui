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
"""

from __future__ import annotations

import numpy as np


def fov_average_trace(movie: np.ndarray) -> np.ndarray:
    """(T,) mean pixel intensity per frame, across the whole field of view."""
    return np.asarray(movie, dtype=np.float64).mean(axis=(0, 1))


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
    movie: np.ndarray, percentile: float = 8.0, window: int = 200
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full pipeline: fov_average_trace -> running_percentile_trend ->
    divide every pixel's trace by the trend, rescaled to the trend's own
    mean so the movie's overall intensity scale is preserved (only the
    slow drift is removed, not the average brightness level). Returns
    (corrected movie, raw fov trace, trend) -- the trace/trend are what
    the GUI plots."""
    trace = fov_average_trace(movie)
    trend = running_percentile_trend(trace, percentile, window)
    scale = trend / trend.mean()
    corrected = movie / scale[None, None, :]
    return corrected, trace, trend
