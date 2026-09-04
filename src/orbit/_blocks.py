"""Tiny helper for streaming a large array through a reduction one slab
at a time -- so a whole-movie float64 working copy is never held, only
one block, while the reduction itself still accumulates at full width.

Used by the assessment metrics that are pure reductions over an axis
(``orbit.denoising.residual_energy_fraction`` over frames-as-rows,
``orbit.motion_metrics.mean_correlation_to_reference`` over frames) and
by the depth-projected residual the Denoising tab shows for a volumetric
movie -- all cases where the input can be a multi-GB (T, L, W, D) volume
but the result is a scalar or a single 2D-per-frame projection.

``chunked_median``/``chunked_variance`` are the one place ``np.median``/
``.var()`` should be called on a potentially movie-sized array anywhere
in orbit -- both make an internal copy the size of their input (a
partition copy for median; ``.var()``'s own docs show it builds a full
``arr - mean`` centered array, promoted to float64 for a narrower
input, before squaring and summing it), which on a multi-GB movie/
volume is a second multi-GB array just to draw one statistic. Every
caller with that shape of problem (orbit.projections'
median_projection/variance_projection/fano_factor_projection and their
_volumetric counterparts, orbit.cnmf_e_init's
peak_to_noise_ratio_projection, orbit.normalization's pixel-wise
median/robuststd baselines, orbit.motion_correction's per-chunk
template refresh) goes through one of these rather than each hand-
rolling its own chunking loop, so a memory-bound fix (or a bug fix to
the chunking itself) lands everywhere at once."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

_DEFAULT_TARGET_BYTES = 64 * 1024 * 1024  # ~64 MiB float64 working slab per block
_MEDIAN_SLAB_BYTES = 256 * 1024 * 1024  # working-slab budget for chunked_median/chunked_variance


def iter_axis_slices(
    shape: tuple[int, ...],
    axis: int,
    target_bytes: int = _DEFAULT_TARGET_BYTES,
    itemsize: int = 8,
    max_step: int | None = None,
) -> Iterator[slice]:
    """Yield ``slice`` objects that partition ``axis`` of an array of
    ``shape`` into consecutive runs, each about ``target_bytes`` once
    widened to ``itemsize`` bytes per element (default 8 -> float64).

    ``max_step``, if given, further caps the step regardless of what
    the byte budget alone would allow -- useful when a caller wants a
    predictable, bounded chunk count (e.g. "at most 150 pixels") on top
    of the memory bound; the byte budget still applies underneath it,
    so a chunk is never larger than either limit allows.

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
    if max_step is not None:
        step = min(step, max_step)
    for start in range(0, axis_len, step):
        yield slice(start, min(start + step, axis_len))


def _promoted_dtype(dtype: np.dtype) -> np.dtype:
    """np.median's/.var()'s own output-dtype promotion rule: int or a
    float narrower than float64 -> float64; float64 (or wider) stays."""
    return dtype if np.issubdtype(dtype, np.floating) else np.dtype(np.float64)


def chunked_reduce(
    arr: np.ndarray,
    axis: int,
    reduce_fn: Callable[[np.ndarray], np.ndarray],
    keepdims: bool = False,
    out_dtype: np.dtype | None = None,
    target_bytes: int = _MEDIAN_SLAB_BYTES,
    max_block: int | None = None,
) -> np.ndarray:
    """Shared block-at-a-time driver behind chunked_median/
    chunked_variance: moves ``axis`` to the front, walks the first
    remaining axis in blocks (bounded by ``target_bytes``/``max_block``
    together -- see iter_axis_slices), and writes ``reduce_fn(block)``
    (block's reduction axis at 0, so ``reduce_fn`` calls ...(axis=0))
    into the matching output slice. Only one block is ever copied to
    ``out_dtype`` at a time -- ``arr`` is never mutated or fully
    materialized at that dtype."""
    moved = np.moveaxis(arr, axis, 0)  # reduction axis -> 0 (a view)
    if out_dtype is None:
        out_dtype = _promoted_dtype(moved.dtype)
    out = np.empty(moved.shape[1:], dtype=out_dtype)
    itemsize = np.dtype(out_dtype).itemsize
    for sl in iter_axis_slices(moved.shape, 1, target_bytes=target_bytes, itemsize=itemsize, max_step=max_block):
        block = np.array(moved[:, sl], dtype=out_dtype)  # own copy -- safe to overwrite
        out[sl] = reduce_fn(block)
    return np.expand_dims(out, axis) if keepdims else out


def chunked_median(
    arr: np.ndarray,
    axis: int,
    keepdims: bool = False,
    out_dtype: np.dtype | None = None,
    target_bytes: int = _MEDIAN_SLAB_BYTES,
    max_block: int | None = None,
) -> np.ndarray:
    """``np.median(arr, axis=axis, keepdims=keepdims)``, computed one
    block of the first non-reduction axis at a time -- see this module's
    docstring for why every movie/volume-scale median in orbit should
    go through this rather than calling ``np.median`` directly.

    ``target_bytes``/``max_block`` bound one block's size, by memory
    and by a hard pixel count respectively (whichever is smaller wins --
    see iter_axis_slices); the defaults suit a per-pixel/per-voxel time
    reduction on a 2D or volumetric movie. ``out_dtype`` defaults to
    matching np.median's own promotion rule -- pass one explicitly
    (e.g. float32) for a caller like motion correction that wants a
    narrower working dtype regardless of the input's own.

    Element-wise identical to ``np.median`` on the ``out_dtype`` cast,
    for any ``axis``/block partitioning -- each block's median only
    ever needs that block's own pixels' full time series, so slicing
    the non-reduction axis can't change any individual result."""
    return chunked_reduce(
        arr, axis, lambda block: np.median(block, axis=0, overwrite_input=True),
        keepdims, out_dtype, target_bytes, max_block,
    )


def chunked_variance(
    arr: np.ndarray,
    axis: int,
    keepdims: bool = False,
    out_dtype: np.dtype | None = None,
    target_bytes: int = _MEDIAN_SLAB_BYTES,
    max_block: int | None = None,
) -> np.ndarray:
    """``arr.var(axis=axis, keepdims=keepdims)`` (ddof=0), computed one
    block at a time -- ndarray.var() builds a full ``arr - mean``
    centered array (promoted to float64 for a narrower input) before
    squaring and summing it, a second, often-LARGER-than-``arr`` copy
    of the whole thing; each block instead only ever needs its own
    (block-sized) centered copy. See chunked_median for the shared
    ``target_bytes``/``max_block``/``out_dtype`` semantics.

    Element-wise identical to ``arr.var(axis=axis)`` on the
    ``out_dtype`` cast, for the same reason chunked_median is: each
    pixel's variance is independent of every other pixel's."""
    return chunked_reduce(arr, axis, lambda block: block.var(axis=0), keepdims, out_dtype, target_bytes, max_block)
