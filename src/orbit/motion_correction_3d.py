"""3D (whole-volume) rigid motion correction for volumetric (T, L, W, D)
movies -- the volumetric analog of orbit.motion_correction's rigid path.

_apply_shift and _estimate_shift over there are already dimension-agnostic
(fourier_shift/fftn and _phase_correlate_shift both operate on an array of
any dimensionality, given a shift vector of matching length), so they're
reused unmodified here. Only the axis-order plumbing around
them -- which axis is time, how the template is bootstrapped, how chunks
are looped -- is specific to a movie's shape convention, and (T, L, W, D)
puts time on axis 0 rather than axis -1 (orbit.motion_correction's (H, W,
T) convention), so that plumbing is rewritten here rather than shared.

No patch-based/non-rigid 3D path yet -- only whole-volume rigid shifts.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

from .motion_correction import (
    _apply_shift,
    _as_float_working_copy,
    _estimate_shift,
    _median_over_axis,
    _resolve_max_workers,
)

# One whole-volume FFT buffer is complex64 the size of the volume
# (gigabytes at realistic sizes), and phase_cross_correlation holds
# several at once -- concurrent workers multiply that, the single
# largest transient in the 3D path. So it defaults to serial; callers
# with RAM headroom can pass ``max_workers`` explicitly.
_DEFAULT_MAX_WORKERS_3D = 1


def _bootstrap_template_3d(
    movie: np.ndarray, template: np.ndarray | None, init_batch: int
) -> tuple[np.ndarray, np.ndarray]:
    """Starting reference volume: ``template`` if supplied, else the
    median of the first ``init_batch`` volumes. Returns ``(template,
    initial_template)``.

    _median_over_axis casts to float32 a spatial slab at a time -- see
    _bootstrap_template (2D). It reads straight from ``movie``, so a
    batch of ``init_batch`` raw volumes is never materialized as float32
    (that would be another ~4x the raw uint8 movie, gigabytes on a real
    set)."""
    if template is None:
        template = _median_over_axis(movie[: min(init_batch, movie.shape[0])], axis=0)
    return template, template.copy()


def _register_in_chunks_3d(
    T: int,
    bin_width: int,
    max_workers: int | None,
    process_one: Callable[[int, np.ndarray], tuple[np.ndarray, np.ndarray]],
    registered: np.ndarray,
    accum: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    """Register volumes 0..T against a template refreshed every
    ``bin_width`` volumes -- same chunked/threaded structure as
    orbit.motion_correction._register_in_chunks, indexed on axis 0."""
    for chunk_start in range(0, T, bin_width):
        chunk_end = min(chunk_start + bin_width, T)
        chunk_template = template

        def _worker(t: int, _template: np.ndarray = chunk_template) -> tuple[int, np.ndarray, np.ndarray]:
            reg_vol, delta = process_one(t, _template)
            return t, reg_vol, delta

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for t, reg_vol, delta in pool.map(_worker, range(chunk_start, chunk_end)):
                registered[t] = reg_vol
                accum[t] += delta

        template = _median_over_axis(registered[chunk_start:chunk_end], axis=0)

    return template


def _setup_registration_3d(
    movie: np.ndarray, template: np.ndarray | None, init_batch: int, bin_width: int,
    max_workers: int | None, output: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, int, int]:
    """Shared setup, T-first analog of
    orbit.motion_correction._setup_registration -- see that function's
    docstring for the ``output``/``initial_read_source`` semantics."""
    if output is not None:
        T = movie.shape[0]
        template, initial_template = _bootstrap_template_3d(movie, template, init_batch)
        registered = output
        initial_read_source = movie
    else:
        movie = _as_float_working_copy(movie)
        T = movie.shape[0]
        template, initial_template = _bootstrap_template_3d(movie, template, init_batch)
        registered = movie  # movie is already a private float32 copy -- no second copy needed
        initial_read_source = None
    workers = _resolve_max_workers(max_workers, bin_width, default=_DEFAULT_MAX_WORKERS_3D)
    return registered, initial_read_source, template, initial_template, T, workers


def rigid_motion_correct_3d(
    movie: np.ndarray,
    template: np.ndarray | None = None,
    max_shift: float = 15.0,
    upsample_factor: int = 50,
    bin_width: int = 200,
    init_batch: int = 100,
    n_iter: int = 1,
    phase_flag: bool = False,
    max_workers: int | None = None,
    output: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Whole-volume rigid motion correction -- 3D analog of
    orbit.motion_correction.rigid_motion_correct.

    ``movie`` is (T, L, W, D). Returns ``(registered_movie, shifts,
    template, initial_template)``:

    - ``shifts`` is (T, 3) subpixel shifts, cumulative across iterations
      and relative to the returned (final) ``template``.
    - ``template`` is the final (possibly iteratively refined) reference
      volume; ``initial_template`` is the bootstrap template before any
      registration.

    ``output``, if given, is written into instead of an in-RAM working
    copy -- see orbit.motion_correction._setup_registration.
    """
    normalization = "phase" if phase_flag else None
    registered, read_source, template, initial_template, T, workers = _setup_registration_3d(
        movie, template, init_batch, bin_width, max_workers, output
    )
    shifts = np.zeros((T, 3))

    for _ in range(n_iter):
        source = read_source if read_source is not None else registered

        def _process_one(t: int, chunk_template: np.ndarray, _source: np.ndarray = source) -> tuple[np.ndarray, np.ndarray]:
            # float32 per volume -- a no-op view when _source is the
            # float32 working copy, but when it's the caller's raw
            # (uint16/float32 memmap) movie -- the ``output=`` path --
            # this bounds each worker's whole-volume FFT transient to one
            # float32 volume rather than one complex128 one.
            volume = np.asarray(_source[t], dtype=np.float32)
            shift = _estimate_shift(chunk_template, volume, upsample_factor, normalization, max_shift)
            return _apply_shift(volume, shift), shift

        template = _register_in_chunks_3d(T, bin_width, workers, _process_one, registered, shifts, template)
        read_source = None  # subsequent iterations refine `registered` in place

    return registered, shifts, template, initial_template
