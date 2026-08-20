"""AR(1) calcium deconvolution for CNMF, ported from CaImAn's
deconvolution.py: noise/AR(1)-coefficient estimation plus OASIS (Friedrich,
Zhou & Paninski 2017) -- a pool-adjacent-violators algorithm that finds the
exact solution to the L1-penalized, nonnegative-spike-constrained least
squares fit ``min_c,s ||y - c||^2 + lam*sum(s)`` s.t. ``c_t = g*c_{t-1} + s_t``,
``s_t >= 0``, without ever forming the T x T system explicitly.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


def estimate_noise_std(trace: np.ndarray, freq_range: tuple[float, float] = (0.25, 0.5)) -> float:
    """High-frequency-band noise estimate, ported from CaImAn's
    preprocessing.GetSn (Welch PSD, geometric mean over ``freq_range`` of
    the Nyquist band -- signal power is assumed concentrated at lower
    frequencies, noise flat across the band)."""
    freqs, psd = welch(np.asarray(trace, dtype=np.float64), nperseg=min(len(trace), 256))
    band = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if not band.any():
        band = freqs >= freq_range[0]
    return float(np.sqrt(np.exp(np.mean(np.log(psd[band] / 2 + 1e-10)))))


def estimate_ar1_coefficient(trace: np.ndarray, lag: int = 5) -> float:
    """AR(1) decay coefficient from the trace's own lag-1 autocovariance
    ratio -- the order-1 case of CaImAn's preprocessing.estimate_time_constant,
    clipped to a plausible decay range."""
    x = np.asarray(trace, dtype=np.float64)
    x = x - x.mean()
    cov0 = np.dot(x, x) / len(x)
    if cov0 <= 0:
        return 0.0
    cov1 = np.dot(x[:-1], x[1:]) / len(x)
    return float(np.clip(cov1 / cov0, 0.0, 0.998))


def oasis_ar1(trace: np.ndarray, g: float, lam: float = 0.0, s_min: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Exact AR(1) OASIS solve. Maintains a list of "pools" -- contiguous
    stretches sharing one spike-free exponential decay from a single
    fitted height -- merging a newly-added pool into its predecessor
    whenever keeping them separate would require a negative spike to
    explain the join, which is what enforces s_t >= 0 without ever
    solving the full system. ``lam`` is an L1 sparsity penalty on the
    spike train; ``s_min`` hard-zeros any spike below that height."""
    y = np.asarray(trace, dtype=np.float64)
    n_frames = len(y)
    pools: list[list[float]] = []  # each: [start, length, value, weight]
    for i in range(n_frames):
        yi = y[i] - (lam if i == n_frames - 1 else lam * (1 - g))
        pools.append([i, 1, yi, 1.0])
        while len(pools) > 1:
            t1, l1, v1, w1 = pools[-2]
            _t2, l2, v2, w2 = pools[-1]
            if v2 / w2 < (g**l1) * v1 / w1 + s_min:
                pools[-2] = [t1, l1 + l2, v1 + (g**l1) * v2, w1 + (g ** (2 * l1)) * w2]
                pools.pop()
            else:
                break

    c = np.empty(n_frames)
    for start, length, value, weight in pools:
        start = int(start)
        length = int(length)
        c[start : start + length] = (value / weight) * g ** np.arange(length)

    s = np.empty(n_frames)
    s[0] = c[0]
    s[1:] = c[1:] - g * c[:-1]
    s[s < s_min] = 0.0
    return c, s


def constrained_oasis_ar1(
    trace: np.ndarray, g: float, noise_std: float, s_min: float = 0.0, max_iter: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Sparsest OASIS fit whose residual doesn't exceed the known noise
    floor -- CaImAn's "constrained foopsi": residual error grows
    monotonically with the L1 penalty ``lam``, so this bisects for the
    LARGEST ``lam`` (i.e. sparsest spike train) whose residual std still
    stays within ``noise_std`` -- not just the first one that does, which
    lam=0's near-exact fit would trivially satisfy without sparsifying
    anything."""
    y = np.asarray(trace, dtype=np.float64)
    lam_lo, lam_hi = 0.0, float(np.abs(y).max()) or 1.0
    c, s = oasis_ar1(y, g, lam=lam_lo, s_min=s_min)  # feasible by construction (lam=0 minimizes residual)

    # Grow the upper bound until it's actually infeasible, so the
    # bisection below brackets a real feasible/infeasible crossing.
    for _ in range(10):
        c_hi, s_hi = oasis_ar1(y, g, lam=lam_hi, s_min=s_min)
        if np.std(y - c_hi) > noise_std:
            break
        lam_hi *= 2

    for _ in range(max_iter):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        c_mid, s_mid = oasis_ar1(y, g, lam=lam_mid, s_min=s_min)
        if np.std(y - c_mid) > noise_std:
            lam_hi = lam_mid
        else:
            lam_lo = lam_mid
            c, s = c_mid, s_mid  # keep the largest-lam solution seen that still satisfies the budget
    return c, s
