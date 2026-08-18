"""Per-pixel summary projections of a (H, W, T) movie -- the first visual
diagnostic shown for any loaded movie, before any processing is applied.
"""

from __future__ import annotations

import numpy as np

from . import _native


def mean_projection(movie: np.ndarray) -> np.ndarray:
    """Per-pixel mean across time."""
    return movie.mean(axis=2)


def median_projection(movie: np.ndarray) -> np.ndarray:
    """Per-pixel median across time -- robust to bright transients."""
    return np.median(movie, axis=2)


def variance_projection(movie: np.ndarray) -> np.ndarray:
    """Per-pixel variance across time -- highlights active pixels."""
    return movie.var(axis=2)


def fano_factor_projection(movie: np.ndarray) -> np.ndarray:
    """Per-pixel Fano factor (variance / mean); zero-mean pixels -> 0."""
    mean = movie.mean(axis=2)
    var = movie.var(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(mean > 0, var / mean, 0.0)


def _half_sample_mode_1d(x: np.ndarray) -> float:
    """Half-sample mode (Bickel & Fruehwirth 2006): repeatedly shrink to
    the densest half of the sorted sample until <=3 points remain. Ported
    from pyGraFT's _half_sample_mode_1d (itself a port of
    CoDybase-MATLAB's halfSampleMode.cpp), the same algorithm SEUDO's
    upstream toolchain compiles to C++ -- a genuinely serial, recursive
    per-trace algorithm that can't be vectorized across pixels the way
    mean/median/variance can.
    """
    x = np.sort(np.asarray(x, dtype=float))
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return float("nan")

    offset = 0
    while True:
        if n == 1:
            return float(x[offset])
        if n == 2:
            return float((x[offset] + x[offset + 1]) / 2.0)
        if n == 3:
            a, b, c = x[offset], x[offset + 1], x[offset + 2]
            diff = (b - a) - (c - b)
            if diff < 0:
                return float((a + b) / 2.0)
            if diff > 0:
                return float((b + c) / 2.0)
            return float(b)

        N = int(np.ceil(0.5 * n))
        widths = x[offset + N - 1 : offset + n - 1] - x[offset : offset + n - N]
        j = int(np.argmin(widths))  # first occurrence, matching the C++ strict "<" comparison
        offset += j
        n = N


def mode_projection(movie: np.ndarray) -> np.ndarray:
    """Per-pixel half-sample mode across time -- a robust "typical
    baseline value" estimate, less pulled toward bright transients than
    the mean and, unlike the median, not fixed to an actual observed
    sample.

    Recursive/sequential per pixel -- can't be vectorized across pixels
    the way mean/median/variance can -- so this is the case where the
    multi-threaded C++ kernel in _native/ (falls back to the pure-numpy
    ``np.apply_along_axis`` path below if not built -- see
    _native/build_native.sh) matters most. Both paths take a real
    contiguous copy first for the same reason as local_correlation_projection.
    """
    movie = np.ascontiguousarray(movie, dtype=np.float32)

    if _native.NATIVE_AVAILABLE:
        return np.asarray(_native.half_sample_mode_native(movie))

    return np.apply_along_axis(_half_sample_mode_1d, 2, movie)


def local_correlation_projection(movie: np.ndarray) -> np.ndarray:
    """Correlation of each pixel's trace with the mean trace of its up to
    8 neighbors -- surfaces cell footprints (co-varying pixels) against
    independent per-pixel noise, as in Suite2p/CaImAn. Ported from
    roiapp's compute_image_local_corr.

    ``np.ascontiguousarray`` (rather than a plain dtype cast) matters a
    lot here: a movie loaded via ``np.moveaxis`` (as orbitapp's loader
    does, to go from a file's native (T, H, W) to (H, W, T)) is a *view*
    with the time axis at the largest stride, so every per-pixel
    time-reduction reads memory in the worst possible order. Forcing a
    real contiguous copy plus float32 (visualization-grade precision is
    enough) alone measured a 16x speedup (54s -> 3.4s on a real
    256x256x2000 two-photon recording); the multi-threaded C++ kernel in
    _native/ (falls back to the pure-numpy path below if not built --
    see _native/build_native.sh) goes further still.
    """
    movie = np.ascontiguousarray(movie, dtype=np.float32)

    if _native.NATIVE_AVAILABLE:
        return np.asarray(_native.local_correlation_native(movie))

    height, width, _n_frames = movie.shape

    padded = np.pad(movie, ((1, 1), (1, 1), (0, 0)))
    valid = np.pad(np.ones((height, width), dtype=np.float32), 1)

    neighbor_sum = np.zeros_like(movie)
    neighbor_count = np.zeros((height, width), dtype=np.float32)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == dc == 0:
                continue
            r, c = 1 + dr, 1 + dc
            neighbor_sum += padded[r : r + height, c : c + width, :]
            neighbor_count += valid[r : r + height, c : c + width]
    neighbor_avg = neighbor_sum / neighbor_count[:, :, None]

    own_c = movie - movie.mean(axis=2, keepdims=True)
    nb_c = neighbor_avg - neighbor_avg.mean(axis=2, keepdims=True)
    num = (own_c * nb_c).sum(axis=2)
    den = np.sqrt((own_c**2).sum(axis=2) * (nb_c**2).sum(axis=2))
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)
