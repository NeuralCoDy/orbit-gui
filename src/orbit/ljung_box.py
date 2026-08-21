"""Ljung-Box whiteness test, adapted for calcium imaging traces: the
smallest ``n_exclude`` lags (nearest lag 0) are dropped from the test
statistic before summing, since a real fluorescence trace's own
indicator decay kinetics create strong, expected autocorrelation at
small lags that would otherwise trivially fail any whiteness test --
excluding them tests only for structure BEYOND that expected
short-range correlation.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from . import _native


def _default_max_lag(n_frames: int) -> int:
    """Standard rule of thumb for the largest lag to test (``10*log10(T)``,
    capped at ``T-1`` so every lag has at least one valid pair)."""
    return max(1, min(int(10 * np.log10(n_frames)), n_frames - 1))


def ljung_box_test_movie(movie: np.ndarray, n_exclude: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Ljung-Box test on every pixel's own (T,) trace, vectorized across
    the whole (H, W, T) movie at once. Returns ``(passed, alpha)``, each
    (H, W): ``passed[h, w]`` is True if that pixel's trace shows no
    significant autocorrelation beyond the excluded near-zero lags
    (fails to reject the null at ``alpha > 0.05``); ``alpha[h, w]`` is
    that p-value.

    The largest lag tested is chosen automatically from the trace
    length (see _default_max_lag). A pixel with a perfectly constant
    trace (zero variance) has no autocorrelation to test and always
    passes (alpha=1.0).

    Uses the native (C++) kernel in _native/ when built, falling back
    to the pure-numpy implementation below otherwise (see
    _native/build_native.sh) -- computing every pixel's own Q statistic
    is an O(H*W*T*n_lags) reduction that stays single-threaded in
    numpy; the chi-squared p-value step below is cheap regardless and
    stays in Python either way."""
    height, width, n_frames = movie.shape
    max_lag = _default_max_lag(n_frames)
    if n_exclude >= max_lag:
        raise ValueError(f"n_exclude ({n_exclude}) leaves no lags to test (max_lag={max_lag})")

    if _native.NATIVE_AVAILABLE:
        q_stat = np.asarray(
            _native.ljung_box_q_statistic_native(np.asarray(movie, dtype=np.float64), n_exclude, max_lag)
        )
    else:
        centered = movie - movie.mean(axis=2, keepdims=True)
        denom = (centered**2).sum(axis=2)  # (H, W)

        q_stat = np.zeros((height, width))
        for lag in range(n_exclude + 1, max_lag + 1):
            cov = (centered[:, :, :-lag] * centered[:, :, lag:]).sum(axis=2)  # (H, W)
            rho = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
            q_stat += rho**2 / (n_frames - lag)
        q_stat *= n_frames * (n_frames + 2)

    df = max_lag - n_exclude
    alpha = chi2.sf(q_stat, df)  # zero-variance pixels have q_stat=0 -> alpha=1.0, no special-casing needed
    passed = alpha > 0.05
    return passed, alpha


def ljung_box_test(trace: np.ndarray, n_exclude: int = 0) -> tuple[bool, float]:
    """Ljung-Box test on a single (T,) trace -- see ljung_box_test_movie
    for the full description; this is that same computation applied to
    one trace (reshaped to a 1x1 "movie" so both share one
    implementation)."""
    trace = np.asarray(trace, dtype=np.float64)
    passed, alpha = ljung_box_test_movie(trace.reshape(1, 1, -1), n_exclude)
    return bool(passed[0, 0]), float(alpha[0, 0])
