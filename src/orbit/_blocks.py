"""Tiny helper for streaming a large array through a reduction one slab
at a time -- so a whole-movie float64 working copy is never held, only
one block, while the reduction itself still accumulates at full width.

Used by the assessment metrics that are pure reductions over an axis
(``orbit.denoising.residual_energy_fraction`` over frames-as-rows,
``orbit.motion_metrics.mean_correlation_to_reference`` over frames) and
by the depth-projected residual the Denoising tab shows for a volumetric
movie -- all cases where the input can be a multi-GB (T, L, W, D) volume
but the result is a scalar or a single 2D-per-frame projection.

``chunked_median``/``chunked_variance`` are the one place ``np.median``
and ``.var()`` should be called on a potentially movie-sized array
anywhere in orbit: both build a second, often-larger copy of their whole
input internally (a partition copy; a promoted-dtype ``arr - mean``
array). Every such caller routes through here so a memory-bound fix
lands in one place."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

_DEFAULT_TARGET_BYTES = 64 * 1024 * 1024  # ~64 MiB float64 working slab per block
_MEDIAN_SLAB_BYTES = 256 * 1024 * 1024  # working-slab budget for chunked_median/chunked_variance
_BLOCK_PIXELS = 150  # default hard cap on a block's span of the chunked axis (on top of the byte budget)


def iter_axis_slices(
    shape: tuple[int, ...],
    axis: int,
    target_bytes: int = _DEFAULT_TARGET_BYTES,
    itemsize: int = 8,
    max_block: int | None = None,
) -> Iterator[slice]:
    """Yield ``slice`` objects that partition ``axis`` of an array of
    ``shape`` into consecutive runs, each about ``target_bytes`` once
    widened to ``itemsize`` bytes per element (default 8 -> float64).

    ``max_block``, if given, further caps the step regardless of what
    the byte budget alone would allow -- the byte budget still applies
    underneath it, so a chunk is never larger than either limit allows.

    Index the array as ``arr[s]`` for ``axis == 0`` or, in general,
    ``arr[(slice(None),) * axis + (s,)]`` / ``arr[..., s]`` for the last
    axis. A small array yields a single full-extent slice, so callers
    don't need a separate "small input" path."""
    axis_len = shape[axis]
    other_elems = 1
    for i, n in enumerate(shape):
        if i != axis:
            other_elems *= n
    step = max(1, target_bytes // max(1, other_elems * itemsize))
    if max_block is not None:
        step = min(step, max_block)
    for start in range(0, axis_len, step):
        yield slice(start, min(start + step, axis_len))


def _promoted_dtype(dtype: np.dtype) -> np.dtype:
    """np.median's/.var()'s own output-dtype promotion rule: int or a
    float narrower than float64 -> float64; float64 (or wider) stays."""
    return dtype if np.issubdtype(dtype, np.floating) else np.dtype(np.float64)


def _chunked_reduce(
    arr: np.ndarray,
    axis: int,
    reduce_fn: Callable[[np.ndarray], np.ndarray],
    keepdims: bool = False,
    out_dtype: np.dtype | None = None,
    max_block: int | None = _BLOCK_PIXELS,
) -> np.ndarray:
    """Shared block-at-a-time driver behind chunked_median/
    chunked_variance: moves ``axis`` to the front, walks the first
    remaining axis in blocks (bounded by ``_MEDIAN_SLAB_BYTES`` and
    ``max_block`` together), and writes ``reduce_fn(block)`` (block's
    reduction axis at 0) into the matching output slice. Only one block
    is ever copied to ``out_dtype`` at a time -- ``arr`` is never
    mutated or fully materialized at that dtype."""
    moved = np.moveaxis(arr, axis, 0)  # reduction axis -> 0 (a view)
    if out_dtype is None:
        out_dtype = _promoted_dtype(moved.dtype)
    out = np.empty(moved.shape[1:], dtype=out_dtype)
    itemsize = np.dtype(out_dtype).itemsize
    for sl in iter_axis_slices(moved.shape, 1, target_bytes=_MEDIAN_SLAB_BYTES, itemsize=itemsize, max_block=max_block):
        block = np.array(moved[:, sl], dtype=out_dtype)  # own copy -- safe to overwrite
        out[sl] = reduce_fn(block)
    return np.expand_dims(out, axis) if keepdims else out


def chunked_median(
    arr: np.ndarray,
    axis: int | None = None,
    keepdims: bool = False,
    out_dtype: np.dtype | None = None,
    max_block: int | None = _BLOCK_PIXELS,
) -> np.ndarray:
    """``np.median(arr, axis=axis, keepdims=keepdims)`` computed one
    block of the first non-reduction axis at a time -- see this module's
    docstring. ``axis=None`` (a single global scalar, which genuinely
    needs every value at once) falls straight through to ``np.median``,
    so this is a drop-in wherever ``np.median`` was called.

    ``out_dtype`` defaults to np.median's own promotion rule; pass one
    (e.g. float32) for a caller that wants a narrower working dtype
    regardless of the input's. Element-wise identical to ``np.median``
    for any block partitioning -- each block's per-pixel median only
    needs that pixel's own full time series."""
    if axis is None:
        return np.median(arr, keepdims=keepdims)
    return _chunked_reduce(
        arr, axis, lambda block: np.median(block, axis=0, overwrite_input=True), keepdims, out_dtype, max_block,
    )


def chunked_variance(arr: np.ndarray, axis: int, max_block: int | None = _BLOCK_PIXELS) -> np.ndarray:
    """``arr.var(axis=axis)`` (ddof=0) computed one block at a time --
    ndarray.var() builds a full promoted-dtype ``arr - mean`` array
    before squaring/summing it, and each block instead only needs its
    own. Element-wise identical to ``arr.var(axis=axis)``, for the same
    per-pixel-independence reason as chunked_median."""
    return _chunked_reduce(arr, axis, lambda block: block.var(axis=0), max_block=max_block)
