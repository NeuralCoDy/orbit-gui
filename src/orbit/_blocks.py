"""Tiny helper for streaming a large array through a reduction one slab
at a time -- so a whole-movie float64 working copy is never held, only
one block, while the reduction itself still accumulates at full width.

Used by the assessment metrics that are pure reductions over an axis
(``orbit.denoising.residual_energy_fraction`` over frames-as-rows,
``orbit.motion_metrics.mean_correlation_to_reference`` over frames) and
by the depth-projected residual the Denoising tab shows for a volumetric
movie -- all cases where the input can be a multi-GB (T, L, W, D) volume
but the result is a scalar or a single 2D-per-frame projection.
"""

from __future__ import annotations

from collections.abc import Iterator

_DEFAULT_TARGET_BYTES = 64 * 1024 * 1024  # ~64 MiB float64 working slab per block


def iter_axis_slices(
    shape: tuple[int, ...],
    axis: int,
    target_bytes: int = _DEFAULT_TARGET_BYTES,
    itemsize: int = 8,
) -> Iterator[slice]:
    """Yield ``slice`` objects that partition ``axis`` of an array of
    ``shape`` into consecutive runs, each about ``target_bytes`` once
    widened to ``itemsize`` bytes per element (default 8 -> float64).

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
    for start in range(0, axis_len, step):
        yield slice(start, min(start + step, axis_len))
