"""Motion correction (rigid + piecewise-rigid), ported from pyGraFT's
graft/motion_correction.py (same author, a Python port of the core
NoRMCorre registration algorithm). See that module's docstring for the
Guizar-Sicairos subpixel-registration and displacement-field-warp design
notes -- unchanged here, just relocated.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
from scipy.ndimage import fourier_shift, map_coordinates
from skimage.registration import phase_cross_correlation

_DEFAULT_MAX_WORKERS = 4


def _resolve_max_workers(max_workers: int | None, bin_width: int) -> int:
    if max_workers is not None:
        return max_workers
    return min(_DEFAULT_MAX_WORKERS, os.cpu_count() or 1, bin_width)


def _as_float_working_copy(movie: np.ndarray) -> np.ndarray:
    """A float64 (H, W, T) array this function owns and can register
    into in place, without mutating whatever the caller passed in.

    np.asarray(movie, dtype=float) already allocates a fresh, private
    array whenever a dtype conversion is needed (the common case: raw
    microscopy movies are usually uint16 or float32) -- calling .copy()
    on top of that in every case, regardless of whether a conversion
    happened, doubled peak memory for exactly the inputs most likely to
    be large (measured: ~13x a uint16 movie's raw size, vs ~7x once this
    redundant copy is skipped). Only allocate the extra copy when
    asarray returned the caller's own array unchanged (dtype was already
    float64), since that's the one case where skipping it would let
    per-frame registration mutate data the caller still holds a
    reference to."""
    converted = np.asarray(movie, dtype=float)
    return converted if converted is not movie else converted.copy()


def _apply_shift(frame: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Apply a rigid (dy, dx) subpixel shift via frequency-domain warping."""
    shifted_fft = fourier_shift(np.fft.fftn(frame), shift)
    return np.real(np.fft.ifftn(shifted_fft))


def _apply_displacement_field(frame: np.ndarray, disp_y: np.ndarray, disp_x: np.ndarray) -> np.ndarray:
    """Apply a smooth per-pixel (dy, dx) displacement field via spline warping."""
    H, W = frame.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coords = [yy - disp_y, xx - disp_x]
    return map_coordinates(frame, coords, order=3, mode="nearest")


def _bootstrap_template(
    movie: np.ndarray, template: np.ndarray | None, init_batch: int
) -> tuple[np.ndarray, np.ndarray]:
    """Starting reference image: ``template`` if supplied, else the median
    of the first ``init_batch`` frames. Returns ``(template, initial_template)``."""
    if template is None:
        template = np.median(movie[:, :, : min(init_batch, movie.shape[-1])], axis=2)
    return template, template.copy()


def _estimate_shift(
    reference: np.ndarray, moving: np.ndarray, upsample_factor: int, normalization: str | None, max_shift: float
) -> np.ndarray:
    """Subpixel (dy, dx) shift of ``moving`` relative to ``reference``, clipped to ``max_shift``."""
    shift, _error, _phasediff = phase_cross_correlation(
        reference, moving, upsample_factor=upsample_factor, normalization=normalization
    )
    return np.clip(shift, -max_shift, max_shift)


def _register_in_chunks(
    T: int,
    bin_width: int,
    max_workers: int | None,
    process_one: Callable[[int, np.ndarray], tuple[np.ndarray, np.ndarray]],
    registered: np.ndarray,
    accum: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    """Register frames 0..T against a template refreshed every ``bin_width``
    frames. Frames within one chunk share a template and don't depend on
    each other, so they register concurrently in a thread pool (skimage/
    scipy's FFT/spline routines release the GIL)."""
    for chunk_start in range(0, T, bin_width):
        chunk_end = min(chunk_start + bin_width, T)
        chunk_template = template

        def _worker(t: int, _template: np.ndarray = chunk_template) -> tuple[int, np.ndarray, np.ndarray]:
            reg_frame, delta = process_one(t, _template)
            return t, reg_frame, delta

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for t, reg_frame, delta in pool.map(_worker, range(chunk_start, chunk_end)):
                registered[:, :, t] = reg_frame
                accum[t] += delta

        template = np.median(registered[:, :, chunk_start:chunk_end], axis=2)

    return template


def _setup_registration(
    movie: np.ndarray, template: np.ndarray | None, init_batch: int, bin_width: int,
    max_workers: int | None, output: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, int, int]:
    """Shared setup for rigid_motion_correct/patch_motion_correct.

    When ``output`` is given (typically a FITS-backed memmap -- see
    StageTab._chunked_commit), registration reads frames from ``movie``
    (possibly itself a memmap, left untouched) and writes corrected
    frames into ``output`` instead of allocating a second in-RAM working
    copy -- this is what keeps a memmap Commit's peak memory bounded by
    chunk size rather than the whole movie. Returns (registered,
    initial_read_source, template, initial_template, T, workers);
    ``initial_read_source`` is ``movie`` itself when writing to a
    separate ``output`` (registered/output starts uninitialized), or
    None when registering in place (the usual, non-memmap case), meaning
    the first pass should read from ``registered`` directly."""
    if output is not None:
        T = movie.shape[-1]
        template, initial_template = _bootstrap_template(movie, template, init_batch)
        registered = output
        initial_read_source = movie
    else:
        movie = _as_float_working_copy(movie)
        T = movie.shape[-1]
        template, initial_template = _bootstrap_template(movie, template, init_batch)
        registered = movie  # movie is already a private float64 copy -- no second copy needed
        initial_read_source = None
    workers = _resolve_max_workers(max_workers, bin_width)
    return registered, initial_read_source, template, initial_template, T, workers


def rigid_motion_correct(
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
    """Whole-frame rigid motion correction (port of NoRMCorre's rigid path).

    ``movie`` is (H, W, T). Returns ``(registered_movie, shifts, template,
    initial_template)``:

    - ``shifts`` is (T, 2) (row, col) subpixel shifts, cumulative across
      iterations and relative to the returned (final) ``template``.
    - ``template`` is the final (possibly iteratively refined) reference
      image; ``initial_template`` is the bootstrap template before any
      registration -- comparing the two is a useful diagnostic (see
      orbit.motion_metrics).

    ``output``, if given, is written into instead of an in-RAM working
    copy -- see _setup_registration.
    """
    normalization = "phase" if phase_flag else None
    registered, read_source, template, initial_template, T, workers = _setup_registration(
        movie, template, init_batch, bin_width, max_workers, output
    )
    shifts = np.zeros((T, 2))

    for _ in range(n_iter):
        source = read_source if read_source is not None else registered

        def _process_one(t: int, chunk_template: np.ndarray, _source: np.ndarray = source) -> tuple[np.ndarray, np.ndarray]:
            frame = _source[:, :, t]
            shift = _estimate_shift(chunk_template, frame, upsample_factor, normalization, max_shift)
            return _apply_shift(frame, shift), shift

        template = _register_in_chunks(T, bin_width, workers, _process_one, registered, shifts, template)
        read_source = None  # subsequent iterations refine `registered` in place

    return registered, shifts, template, initial_template


def _patch_centers(size: int, grid_size: int) -> np.ndarray:
    n_patches = max(1, round(size / grid_size))
    edges = np.linspace(0, size, n_patches + 1)
    return 0.5 * (edges[:-1] + edges[1:]), edges.astype(int)


def _low_signal_patches(
    template: np.ndarray, y_edges: np.ndarray, x_edges: np.ndarray, min_relative_std: float
) -> list[tuple[int, int]]:
    """(i, j) indices of patches whose local std in ``template`` is below
    ``min_relative_std`` of the whole-template std -- little texture for
    phase correlation to lock onto."""
    global_std = template.std()
    if global_std == 0:
        return []
    ny, nx = len(y_edges) - 1, len(x_edges) - 1
    low = []
    for i in range(ny):
        for j in range(nx):
            r0, r1 = y_edges[i], y_edges[i + 1]
            c0, c1 = x_edges[j], x_edges[j + 1]
            if template[r0:r1, c0:c1].std() < min_relative_std * global_std:
                low.append((i, j))
    return low


def _estimate_patch_shift_field(
    template: np.ndarray,
    frame: np.ndarray,
    y_edges: np.ndarray,
    x_edges: np.ndarray,
    rigid_shift: np.ndarray,
    max_dev: float,
    upsample_factor: int,
    normalization: str | None,
) -> np.ndarray:
    """Per-patch (dy, dx) shift estimate, each clipped to within
    ``max_dev`` of the whole-frame ``rigid_shift``."""
    ny, nx = len(y_edges) - 1, len(x_edges) - 1
    patch_shift = np.zeros((ny, nx, 2))
    for i in range(ny):
        r0, r1 = y_edges[i], y_edges[i + 1]
        for j in range(nx):
            c0, c1 = x_edges[j], x_edges[j + 1]
            p_shift = _estimate_shift(
                template[r0:r1, c0:c1], frame[r0:r1, c0:c1], upsample_factor, normalization, max_shift=np.inf
            )
            patch_shift[i, j] = np.clip(p_shift, rigid_shift - max_dev, rigid_shift + max_dev)
    return patch_shift


def _warn_if_low_signal(
    template: np.ndarray, y_edges: np.ndarray, x_edges: np.ndarray, min_patch_contrast: float
) -> None:
    """Warn about patches too low-contrast for reliable phase correlation
    (typically blank background in sparse ROI movies). No-op if
    ``min_patch_contrast <= 0``."""
    if min_patch_contrast <= 0:
        return
    ny, nx = len(y_edges) - 1, len(x_edges) - 1
    low_signal = _low_signal_patches(template, y_edges, x_edges, min_patch_contrast)
    if not low_signal:
        return
    warnings.warn(
        f"patch_motion_correct: {len(low_signal)}/{ny * nx} patch(es) "
        f"{low_signal} have local contrast below {min_patch_contrast:.0%} of "
        "the reference template's overall std -- their shift estimates may "
        "be unreliable. Consider a larger grid_size (fewer, bigger patches).",
        stacklevel=3,
    )


def _upsample_shift_field(
    patch_shift: np.ndarray, y_centers: np.ndarray, x_centers: np.ndarray, H: int, W: int
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly upsample a coarse (ny, nx, 2) per-patch shift grid to a
    smooth (H, W) per-pixel displacement field."""
    ny, nx = patch_shift.shape[:2]
    y_query = np.interp(np.arange(H), y_centers, np.arange(ny)) if ny > 1 else np.zeros(H)
    x_query = np.interp(np.arange(W), x_centers, np.arange(nx)) if nx > 1 else np.zeros(W)
    yq, xq = np.meshgrid(y_query, x_query, indexing="ij")
    disp_y = map_coordinates(patch_shift[..., 0], [yq, xq], order=1, mode="nearest")
    disp_x = map_coordinates(patch_shift[..., 1], [yq, xq], order=1, mode="nearest")
    return disp_y, disp_x


def patch_motion_correct(
    movie: np.ndarray,
    template: np.ndarray | None = None,
    grid_size: int = 32,
    max_shift: float = 15.0,
    max_dev: float = 3.0,
    upsample_factor: int = 50,
    bin_width: int = 200,
    init_batch: int = 100,
    n_iter: int = 1,
    phase_flag: bool = False,
    min_patch_contrast: float = 0.1,
    max_workers: int | None = None,
    output: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Piecewise-rigid (non-rigid) motion correction (port of NoRMCorre's
    non-rigid path): splits each frame into a ``grid_size``-ish grid of
    patches, estimates a shift per patch (bounded to within ``max_dev`` of
    the whole-frame rigid shift), and warps via a bilinearly-upsampled
    per-pixel displacement field rather than NoRMCorre's overlap-blended
    patches.

    ``movie`` is (H, W, T). Returns ``(registered_movie, shift_fields,
    template, initial_template)`` -- ``shift_fields`` is (T, n_patches_y,
    n_patches_x, 2), the coarse per-patch shifts (pre-upsampling),
    accumulated across iterations.

    ``min_patch_contrast`` (default 0.1): warns (doesn't fail) if any
    patch's local contrast is too low for phase correlation to lock onto
    -- pass 0 to disable. ``output``, if given, is written into instead
    of an in-RAM working copy -- see _setup_registration.
    """
    normalization = "phase" if phase_flag else None
    registered, read_source, template, initial_template, T, workers = _setup_registration(
        movie, template, init_batch, bin_width, max_workers, output
    )
    H, W = registered.shape[:2]

    y_centers, y_edges = _patch_centers(H, grid_size)
    x_centers, x_edges = _patch_centers(W, grid_size)
    ny, nx = len(y_centers), len(x_centers)
    _warn_if_low_signal(initial_template, y_edges, x_edges, min_patch_contrast)

    shift_fields = np.zeros((T, ny, nx, 2))

    for _ in range(n_iter):
        source = read_source if read_source is not None else registered

        def _process_one(t: int, chunk_template: np.ndarray, _source: np.ndarray = source) -> tuple[np.ndarray, np.ndarray]:
            frame = _source[:, :, t]
            rigid_shift = _estimate_shift(chunk_template, frame, upsample_factor, normalization, max_shift)
            patch_shift = _estimate_patch_shift_field(
                chunk_template, frame, y_edges, x_edges, rigid_shift, max_dev, upsample_factor, normalization
            )
            disp_y, disp_x = _upsample_shift_field(patch_shift, y_centers, x_centers, H, W)
            return _apply_displacement_field(frame, disp_y, disp_x), patch_shift

        template = _register_in_chunks(T, bin_width, workers, _process_one, registered, shift_fields, template)
        read_source = None  # subsequent iterations refine `registered` in place

    return registered, shift_fields, template, initial_template


def motion_correct(
    movie: np.ndarray, method: str = "rigid", **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dispatches to ``rigid_motion_correct``, ``patch_motion_correct``,
    or ``patchwarp_motion_correct`` by name -- the single entry point
    stage tabs/pipelines should call, so the choice of algorithm is a
    parameter, not a different function to import."""
    if method == "rigid":
        return rigid_motion_correct(movie, **kwargs)
    if method == "patch":
        return patch_motion_correct(movie, **kwargs)
    if method == "patchwarp":
        from .patchwarp import patchwarp_motion_correct  # deferred: patchwarp imports rigid_motion_correct from here

        return patchwarp_motion_correct(movie, **kwargs)
    raise ValueError(f"Unknown motion correction method: {method!r} (expected 'rigid', 'patch', or 'patchwarp')")
