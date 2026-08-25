"""Shared helper for volumetric (T, L, W, D) movies -- projecting a
volume's depth axis away lands the movie in the exact (H, W, T)
shape/axis convention every 2D display widget and quality metric in this
codebase already expects, so those can be reused unmodified on a
volumetric candidate rather than duplicated in 3D.
"""

from __future__ import annotations

import numpy as np


def depth_project(movie: np.ndarray, mode: str = "max") -> np.ndarray:
    """(T, L, W, D) -> (L, W, T) via a projection across depth.

    ``mode="max"`` (default) is a maximum-intensity projection -- keeps
    bright puncta as sharp peaks, so residual x/y/z misregistration shows
    up as blur/doubling rather than being washed out the way a mean
    projection would. ``mode="mean"`` is also available.
    """
    if mode == "max":
        projected = movie.max(axis=3)
    elif mode == "mean":
        projected = movie.mean(axis=3)
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected 'max' or 'mean')")
    return np.moveaxis(projected, 0, -1)
