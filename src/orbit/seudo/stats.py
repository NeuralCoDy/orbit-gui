"""Small statistics helpers, ported from the `seudo` package's stats.py
(itself a port of SEUDO's robustSTD and correlationVectorMatrix).
"""

from __future__ import annotations

import numpy as np


def robust_std(data: np.ndarray) -> np.ndarray:
    """Median-absolute-deviation standard deviation estimate, rescaled to
    match ordinary std for normally-distributed data. NaNs ignored."""
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return np.array([])
    centered = data - np.nanmedian(data)
    return np.nanmedian(np.abs(centered)) / 0.6741891400433162


def correlation_vector_matrix(v: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Pearson correlation of vector v (length T) against each column of
    matrix m (T, C), ignoring NaNs. Returns a length-C array."""
    v = np.asarray(v, dtype=float).reshape(-1)
    m = np.asarray(m, dtype=float)
    assert v.shape[0] == m.shape[0]

    v = v - np.nanmean(v)
    m = m - np.nanmean(m, axis=0, keepdims=True)

    numers = v[:, np.newaxis] * m
    denom_v = np.nansum(v**2)
    denom_m = np.nansum(m**2, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(numers, axis=0) / np.sqrt(denom_v * denom_m)
