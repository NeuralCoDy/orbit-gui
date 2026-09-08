"""CNMF-E's defining feature: a per-pixel "ring model" background, in
place of plain CNMF's global low-rank background (see cnmf_init.
estimate_background). 1P/microendoscopic recordings have strong,
spatially-varying out-of-focus fluorescence that a single global
low-rank term can't capture -- CNMF-E instead predicts each pixel's
background as a weighted sum of its own ring (annulus) of neighboring
pixels' fluorescence, with weights fit by per-pixel least-squares
regression against a neuron-subtracted residual (Zhou et al. 2018).

Not the same computation as neuropil.py's neuropil_ring_mask, despite
the shared "ring" naming: that builds ONE boolean dilation-ring for ONE
already-known ROI mask (dilate to outer radius minus dilate to inner
radius). This module instead needs a ring PER PIXEL of the whole field
of view, from raw offset geometry, to fit an independent regression at
every pixel -- a fundamentally different computation, not a reuse
opportunity.

Fitting a genuinely independent regression at every pixel of a full-
resolution frame is where real CNMF-E spends much of its runtime, so
(matching the real algorithm's own documented performance technique,
not a shortcut introduced here) the fit runs on a spatially
block-downsampled grid, then the resulting background is upsampled back
to full resolution -- see ring_model_background.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.transform import downscale_local_mean

_MIN_RING_NEIGHBORS = 3  # below this, a per-pixel ridge fit is too underdetermined to trust

# H * width_ds * chunk_size at the exact configuration profiling measured
# (256x256 field of view, ds_ratio=4 -> 64x64 downsampled grid, chunk=50):
# ~40MB peak "extra" memory for that one temporary, ~5x faster than
# scipy.ndimage.zoom -- see predict_ring_model_background's own docstring
# for the full rationale/measurements. Chosen per call (see
# _zoom_chunk_frames) to keep H * width_ds * chunk_size at this same
# value regardless of field-of-view size, so a much larger FOV gets a
# smaller chunk automatically (bounded extra memory) rather than the
# extra growing unbounded with FOV size.
_ZOOM_CHUNK_ELEMENTS = 256 * 64 * 50


def _zoom_chunk_frames(height: int, width_ds: int) -> int:
    """How many frames to upsample at once in predict_ring_model_background's
    own chunked replacement for scipy.ndimage.zoom -- see
    _ZOOM_CHUNK_ELEMENTS' own comment."""
    return max(1, _ZOOM_CHUNK_ELEMENTS // (height * width_ds))


@dataclass
class RingModel:
    """Fitted ring weights (the expensive step -- see fit_ring_model)
    plus everything needed to cheaply re-predict a background from a
    NEW residual without refitting: real CNMF-E only refits the ring
    weights rarely, then re-applies them to the current residual each
    iteration."""

    weights: np.ndarray  # (Hd, Wd, n_ring)
    valid_mask: np.ndarray  # (Hd, Wd, n_ring) bool
    offsets: np.ndarray  # (n_ring, 2) int, in downsampled-grid units
    ds_ratio: int
    full_shape: tuple[int, int]  # (H, W) -- the resolution predictions upsample back to
    row_interp: np.ndarray  # (H, Hd) -- see predict_ring_model_background's own docstring
    col_interp: np.ndarray  # (Wd, W)


def ring_offsets(inner_radius: float, outer_radius: float) -> np.ndarray:
    """Integer (dr, dc) pixel offsets forming an annulus: every offset
    with ``inner_radius < hypot(dr, dc) <= outer_radius``. Radii are in
    whatever grid's units the caller is working in -- this function has
    no hidden unit assumption; ring_model_background is what converts
    full-resolution radii to downsampled-grid units before calling this.
    Symmetric under negation by construction (the grid and threshold
    condition are both symmetric about the origin)."""
    radius = int(np.ceil(outer_radius))
    dr, dc = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1), indexing="ij")
    dist = np.hypot(dr, dc)
    keep = (dist > inner_radius) & (dist <= outer_radius)
    return np.stack([dr[keep], dc[keep]], axis=1).astype(int)


def fit_ring_weights(
    residual: np.ndarray, neuron_mask: np.ndarray, offsets: np.ndarray, max_fit_frames: int = 500,
    ridge: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel independent ridge regression of each non-neuron pixel's
    trace against its own (in-bounds, non-neuron) ring-neighbor traces,
    temporally subsampled to at most ``max_fit_frames`` evenly-spaced
    frames. A genuine per-pixel Python loop -- there's no shared design
    matrix to batch here (unlike cnmf.update_spatial_components's
    identically-candidate-set grouping): every pixel's ring-neighbor set
    differs via image-boundary clipping and neuron-mask exclusion.

    Returns (weights (H, W, n_ring), valid_mask (H, W, n_ring) bool) --
    ``valid_mask[r, c, k]`` is True only where offset k was actually used
    as a predictor for pixel (r, c) (in-bounds, non-neuron, and the pixel
    itself had enough such neighbors to fit); weights default to 0
    everywhere else, so an unfit/invalid slot contributes nothing to
    predict_ring_background regardless."""
    height, width, n_frames = residual.shape
    n_ring = len(offsets)
    weights = np.zeros((height, width, n_ring))
    valid_mask = np.zeros((height, width, n_ring), dtype=bool)
    if n_ring == 0:
        return weights, valid_mask

    n_fit = min(max_fit_frames, n_frames)
    fit_idx = np.unique(np.linspace(0, n_frames - 1, n_fit).astype(int))

    for r in range(height):
        for c in range(width):
            if neuron_mask[r, c]:
                continue
            neighbor_rows = r + offsets[:, 0]
            neighbor_cols = c + offsets[:, 1]
            in_bounds = (
                (neighbor_rows >= 0) & (neighbor_rows < height) & (neighbor_cols >= 0) & (neighbor_cols < width)
            )
            candidate_idx = np.nonzero(in_bounds)[0]
            if len(candidate_idx) == 0:
                continue
            nr, nc = neighbor_rows[candidate_idx], neighbor_cols[candidate_idx]
            not_neuron = ~neuron_mask[nr, nc]
            candidate_idx, nr, nc = candidate_idx[not_neuron], nr[not_neuron], nc[not_neuron]
            if len(candidate_idx) < _MIN_RING_NEIGHBORS:
                continue

            design = residual[nr, nc][:, fit_idx].T  # (n_fit, n_valid_neighbors)
            target = residual[r, c, fit_idx]  # (n_fit,)
            gram = design.T @ design
            rhs = design.T @ target
            # Ridge term scaled relative to the Gram matrix's own trace
            # (same style as cnmf.py's update_temporal_components
            # background solve) -- keeps `ridge` a dimensionless
            # regularization strength regardless of the movie's own
            # fluorescence units, with a small floor so an all-zero
            # neighborhood (e.g. in a synthetic test) doesn't leave the
            # system exactly singular.
            jitter = ridge * (np.trace(gram) / max(gram.shape[0], 1)) + 1e-12
            solved = np.linalg.solve(gram + jitter * np.eye(gram.shape[0]), rhs)

            weights[r, c, candidate_idx] = solved
            valid_mask[r, c, candidate_idx] = True

    return weights, valid_mask


def predict_ring_background(
    residual: np.ndarray, weights: np.ndarray, valid_mask: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """Applies already-fit per-pixel ring weights to ``residual``'s full
    frame count, producing a same-shape background prediction. Cheap
    relative to fit_ring_weights: vectorized over the whole grid at once
    per ring offset (a handful of shift-multiply-accumulate passes),
    rather than a per-pixel Python loop."""
    height, width, n_frames = residual.shape
    background = np.zeros_like(residual)
    for k in range(len(offsets)):
        dr, dc = offsets[k]
        r0_dst, r1_dst = max(0, -dr), min(height, height - dr)
        c0_dst, c1_dst = max(0, -dc), min(width, width - dc)
        if r0_dst >= r1_dst or c0_dst >= c1_dst:
            continue
        r0_src, r1_src, c0_src, c1_src = r0_dst + dr, r1_dst + dr, c0_dst + dc, c1_dst + dc

        contribution = residual[r0_src:r1_src, c0_src:c1_src, :] * weights[r0_dst:r1_dst, c0_dst:c1_dst, k, None]
        contribution = np.where(valid_mask[r0_dst:r1_dst, c0_dst:c1_dst, k, None], contribution, 0.0)
        background[r0_dst:r1_dst, c0_dst:c1_dst, :] += contribution
    return background


def fit_ring_model(
    residual: np.ndarray, neuron_mask: np.ndarray, ring_inner_radius: float, ring_outer_radius: float,
    ds_ratio: int = 4, max_fit_frames: int = 500, ridge: float = 1e-2,
) -> RingModel:
    """The expensive one-time step: spatially block-downsamples
    ``residual``/``neuron_mask`` by ``ds_ratio``
    (skimage.transform.downscale_local_mean -- averages blocks, which
    also reduces per-pixel noise before the regression fit, not just a
    speed shortcut) and fits per-pixel ring weights on that downsampled
    grid. ``ring_inner_radius``/``ring_outer_radius`` are in
    full-resolution pixel units, converted to downsampled-grid units
    (divided by ``ds_ratio``) before being passed to ring_offsets, which
    itself stays unit-agnostic. See predict_ring_model_background for
    the cheap step that reapplies this fit to a (possibly different)
    residual without refitting."""
    height, width, _n_frames = residual.shape
    residual_ds = downscale_local_mean(residual, (ds_ratio, ds_ratio, 1))
    mask_ds = downscale_local_mean(neuron_mask.astype(np.float64), (ds_ratio, ds_ratio)) > 0
    height_ds, width_ds = mask_ds.shape

    offsets = ring_offsets(ring_inner_radius / ds_ratio, ring_outer_radius / ds_ratio)
    weights, valid_mask = fit_ring_weights(residual_ds, mask_ds, offsets, max_fit_frames, ridge)
    # The two 1D linear operators predict_ring_model_background's own
    # upsample needs, built once here (not per predict call, since real
    # CNMF-E calls that far more often than it refits this model) by
    # probing scipy.ndimage.zoom with an identity matrix -- guarantees
    # bit-identical results to calling zoom directly, without having to
    # hand-derive its own order=1 interpolation/boundary conventions.
    # See predict_ring_model_background's own docstring for why this
    # replaced a direct zoom call there.
    row_interp = ndimage.zoom(np.eye(height_ds), (height / height_ds, 1), order=1)  # (H, Hd)
    col_interp = ndimage.zoom(np.eye(width_ds), (1, width / width_ds), order=1)  # (Wd, W)
    return RingModel(
        weights=weights, valid_mask=valid_mask, offsets=offsets, ds_ratio=ds_ratio, full_shape=(height, width),
        row_interp=row_interp, col_interp=col_interp,
    )


def predict_ring_model_background(residual: np.ndarray, model: RingModel) -> np.ndarray:
    """Cheaply re-applies an already-fit RingModel to (possibly a new)
    ``residual``: downsamples it the same way fit_ring_model did,
    predicts on that grid (a handful of vectorized shift-multiply-
    accumulate passes -- see predict_ring_background), then upsamples
    back to full resolution. No refitting -- real CNMF-E also only
    refits ring weights rarely, then re-applies them to the current
    residual each iteration.

    The upsample is NOT scipy.ndimage.zoom (used here previously) --
    profiling a real 256x256x2000 CNMF-E run found that one call the
    single largest cost in the whole function (~12.7s of ~40.3s, called
    twice per run), because zoom is a general N-dimensional geometric
    transform doing real per-pixel coordinate-mapping work for every one
    of T frames, even though the actual transform is the SAME fixed,
    data-independent linear map applied identically to every frame (the
    grid only changes shape once, when fit_ring_model builds this
    RingModel -- never per predict call). Since order=1 (bilinear)
    upsampling is linear and separable (row pass then column pass),
    model.row_interp/col_interp (built once in fit_ring_model, see its
    own docstring) let both passes become ordinary matrix multiplies
    against the whole (Hd, Wd, T) grid at once -- confirmed bit-identical
    to the old zoom call (~1e-15, floating-point noise) and ~5x faster on
    real data. Processed a chunk of frames at a time (not all T at once)
    to bound the temporary's own size regardless of frame count -- see
    _ZOOM_CHUNK_ELEMENTS' own comment; this adds negligible overhead
    (confirmed: the same total time within measurement noise) since the
    temporary was already going to be much smaller than the full output
    either way."""
    residual_ds = downscale_local_mean(residual, (model.ds_ratio, model.ds_ratio, 1))
    background_ds = predict_ring_background(residual_ds, model.weights, model.valid_mask, model.offsets)

    height, width = model.full_shape
    height_ds, width_ds, n_frames = background_ds.shape
    chunk = _zoom_chunk_frames(height, width_ds)
    out = np.empty((height, width, n_frames))
    for t0 in range(0, n_frames, chunk):
        t1 = min(t0 + chunk, n_frames)
        rows_blurred = np.tensordot(model.row_interp, background_ds[:, :, t0:t1], axes=([1], [0]))
        # einsum, not a second tensordot, to land directly on (H, W, chunk)
        # -- tensordot's own convention would give (H, chunk, W) here,
        # needing a further transpose(+copy) to fix (see cnmf_init.
        # gaussian_blur_movie's own docstring for the same lesson learned
        # the hard way there).
        np.einsum("hdt,dw->hwt", rows_blurred, model.col_interp, optimize=True, out=out[:, :, t0:t1])
    return out


def ring_model_background(
    residual: np.ndarray, neuron_mask: np.ndarray, ring_inner_radius: float, ring_outer_radius: float,
    ds_ratio: int = 4, max_fit_frames: int = 500, ridge: float = 1e-2,
) -> np.ndarray:
    """Convenience one-shot wrapper (fit + predict on the same residual)
    for simple/single-use callers and tests; cnmf_e_source_extraction
    itself calls fit_ring_model once and predict_ring_model_background
    per iteration directly, to avoid refitting on every call."""
    model = fit_ring_model(residual, neuron_mask, ring_inner_radius, ring_outer_radius, ds_ratio, max_fit_frames, ridge)
    return predict_ring_model_background(residual, model)
