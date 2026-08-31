"""Gaussian blob basis function, ported from the `seudo` package's blob.py
(matching the blob built in SEUDO's estimateTimeCoursesWithSEUDO.m).
"""

from __future__ import annotations

import numpy as np


def _matlab_fspecial_gaussian(hsize: int, sigma: float) -> np.ndarray:
    """MATLAB's fspecial('gaussian', hsize, sigma) for a square, odd-sized kernel."""
    siz = (hsize - 1) / 2
    y, x = np.mgrid[-siz : siz + 1, -siz : siz + 1]
    h = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    h[h < np.finfo(float).eps * h.max()] = 0.0
    total = h.sum()
    return h / total if total != 0 else h


def make_seudo_blob(blob_radius: float, clip_height: float = 0.01) -> np.ndarray:
    """The single normalized Gaussian blob used as the basis function for
    unmodeled ("blob") activity in SEUDO."""
    crop_rad = int(np.ceil(blob_radius * 2.5 + np.finfo(float).eps))
    hsize = crop_rad * 2 + 1
    blob = _matlab_fspecial_gaussian(hsize, blob_radius)
    blob = blob * (blob > clip_height * blob.max())
    return blob / np.sqrt(np.sum(blob**2))


def make_smoothing_kernel(radius: float) -> np.ndarray:
    """Sum-normalized (not L2-normalized) Gaussian smoothing kernel, for
    denoising a frame via convolution -- preserves the frame's overall
    pixel-intensity scale (mirrors _matlab_fspecial_gaussian's own
    normalization), unlike make_seudo_blob's L2-normalized (sum of squares
    = 1) SEUDO basis-function convention, which would badly distort a
    frame's amplitude if used as a plain smoothing filter instead."""
    crop_rad = int(np.ceil(radius * 2.5 + np.finfo(float).eps))
    hsize = crop_rad * 2 + 1
    return _matlab_fspecial_gaussian(hsize, radius)
