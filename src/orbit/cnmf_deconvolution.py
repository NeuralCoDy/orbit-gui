"""AR(1) calcium deconvolution for CNMF, ported from CaImAn's
deconvolution.py: noise/AR(1)-coefficient estimation plus OASIS (Friedrich,
Zhou & Paninski 2017) -- a pool-adjacent-violators algorithm that finds the
exact solution to the L1-penalized, nonnegative-spike-constrained least
squares fit ``min_c,s ||y - c||^2 + lam*sum(s)`` s.t. ``c_t = g*c_{t-1} + s_t``,
``s_t >= 0``, without ever forming the T x T system explicitly.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window

from . import _native


def _fast_welch_psd(x: np.ndarray, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    """Welch's method PSD estimate along the last axis, matching
    scipy.signal.welch(x, nperseg=nperseg, axis=-1)'s own defaults
    (Hann window, 50% overlap, one-sided, density scaling at fs=1.0,
    per-segment constant detrend) -- confirmed numerically equivalent
    (~1e-14 relative difference, i.e. floating-point noise) at a
    fraction of the cost: profiling showed scipy's welch spends ~90% of
    its time in its general-purpose ShortTimeFFT machinery (built for
    streaming/arbitrary boundary handling) rather than the FFT itself,
    none of which this module needs -- every call here wants exactly
    one fixed-size, non-streaming PSD estimate with the same window/
    overlap/scaling every time (~2.3x faster on a realistic (65536,
    2000) batch in cnmf_e_init.noise_std_projection).

    ``x`` can be 1D (a single trace, for estimate_noise_std below) or
    2D (n_signals, n_samples), matching cnmf_e_init.noise_std_projection's
    batched (P, T) shape -- the last axis is always the one transformed.
    Returns (freqs, psd), same shape/convention as scipy.signal.welch."""
    x = np.asarray(x, dtype=np.float64)
    n_samples = x.shape[-1]
    nperseg = min(nperseg, n_samples)  # as_strided below is unsafe if nperseg could exceed n_samples
    noverlap = nperseg // 2
    step = nperseg - noverlap
    n_segments = max(1, (n_samples - noverlap) // step)

    window = get_window("hann", nperseg, fftbins=True).astype(np.float64)
    win_scale = (window**2).sum()

    shape = x.shape[:-1] + (n_segments, nperseg)
    strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
    segments = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides, writeable=False)
    segments = segments - segments.mean(axis=-1, keepdims=True)  # scipy welch's default detrend="constant"
    segments = segments * window

    spectrum = np.fft.rfft(segments, axis=-1)
    psd = (spectrum.real**2 + spectrum.imag**2) / win_scale
    psd[..., 1:-1] *= 2  # one-sided scaling: fold the negative-frequency half back in
    psd = psd.mean(axis=-2)  # average over segments

    freqs = np.fft.rfftfreq(nperseg)
    return freqs, psd


def estimate_noise_std(trace: np.ndarray, freq_range: tuple[float, float] = (0.25, 0.5)) -> float:
    """High-frequency-band noise estimate, ported from CaImAn's
    preprocessing.GetSn (Welch PSD, geometric mean over ``freq_range`` of
    the Nyquist band -- signal power is assumed concentrated at lower
    frequencies, noise flat across the band)."""
    trace = np.asarray(trace, dtype=np.float64)
    freqs, psd = _fast_welch_psd(trace, nperseg=min(len(trace), 256))
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
    spike train; ``s_min`` hard-zeros any spike below that height.

    Uses the native (C++) kernel in _native/ when built, falling back to
    the pure-Python implementation below otherwise (see
    _native/build_native.sh) -- OASIS's pool-merging is inherently
    sequential per trace (can't vectorize with numpy), and gets called
    dozens of times per component via constrained_oasis_ar1's bisection
    below, so this dominated per-patch CNMF runtime before the native
    port."""
    if _native.NATIVE_AVAILABLE:
        c, s = _native.oasis_ar1_native(np.asarray(trace, dtype=np.float64), float(g), float(lam), float(s_min))
        return np.asarray(c), np.asarray(s)
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
