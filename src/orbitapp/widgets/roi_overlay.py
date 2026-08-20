"""Pure-numpy RGBA compositing for the Source Extraction review overlay --
one distinct color per candidate ROI, alpha modulated by review status
(accepted/pending/rejected). Kept independent of Qt so the compositing
math is unit-testable on its own, same pattern as qc_panel.py's
split_by_kind.
"""

from __future__ import annotations

import colorsys

import numpy as np

from ..state import ROI

_ALPHA_BY_STATUS = {"accepted": 1.0, "pending": 0.5, "rejected": 0.08}


def _roi_color(index: int, total: int) -> tuple[float, float, float]:
    """Cycles hue around the color wheel so ROIs are visually distinct
    regardless of how many there are."""
    hue = (index / max(total, 1)) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.85, 1.0)


def _over(
    base_rgb: np.ndarray, base_alpha: np.ndarray, top_rgb: np.ndarray, top_alpha: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Porter-Duff 'over': composite (top_rgb, top_alpha) over (base_rgb, base_alpha)."""
    out_alpha = top_alpha + base_alpha * (1.0 - top_alpha)
    numer = top_rgb * top_alpha[..., None] + base_rgb * base_alpha[..., None] * (1.0 - top_alpha[..., None])
    out_rgb = np.divide(numer, out_alpha[..., None], out=np.zeros_like(numer), where=out_alpha[..., None] > 1e-12)
    return out_rgb, out_alpha


def render_roi_overlay(rois: list[ROI], shape: tuple[int, int]) -> np.ndarray | None:
    """RGBA (uint8) composite of every ROI's mask -- one color per ROI
    cycling around the hue wheel, alpha set by review status (full for
    accepted, medium for pending, faint for rejected so it stays
    reconsiderable rather than vanishing). None if there's nothing to draw."""
    if not rois:
        return None

    rgb = np.zeros((*shape, 3), dtype=np.float64)
    alpha = np.zeros(shape, dtype=np.float64)
    for i, roi in enumerate(rois):
        color = _roi_color(i, len(rois))
        roi_rgb = np.empty((*shape, 3))
        roi_rgb[..., 0], roi_rgb[..., 1], roi_rgb[..., 2] = color
        roi_alpha = roi.mask.astype(np.float64) * _ALPHA_BY_STATUS.get(roi.status, 0.5)
        rgb, alpha = _over(rgb, alpha, roi_rgb, roi_alpha)

    out = np.zeros((*shape, 4), dtype=np.uint8)
    out[..., :3] = np.clip(rgb, 0.0, 1.0) * 255
    out[..., 3] = np.clip(alpha, 0.0, 1.0) * 255
    return out
