"""Motion correction (rigid + piecewise-rigid), ported from pyGraFT's
graft/motion_correction.py (same author, a Python port of the core
NoRMCorre registration algorithm). See that module's docstring for the
Guizar-Sicairos subpixel-registration and displacement-field-warp design
notes -- unchanged here, just relocated.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
from scipy.fft import fftn, ifftn, next_fast_len
from scipy.ndimage import fourier_shift, map_coordinates

from ._blocks import chunked_median, iter_axis_slices
from ._concurrency import available_cpu_count

_DEFAULT_MAX_WORKERS = 4
_SMOOTH_ENOUGH_PRIME = 53  # an axis whose length's largest prime factor is <= this FFTs fast enough without padding


def _largest_prime_factor(n: int) -> int:
    f = 2
    while f * f <= n:
        while n % f == 0:
            n //= f
        f += 1
    return n


def _pad_to_fast_len(image: np.ndarray, mode: str = "constant") -> np.ndarray:
    """Pads ``image`` (bottom/right of each axis, via ``mode``) up to the
    next FFT-efficient size per axis (scipy.fft.next_fast_len) --
    dimension-agnostic, so this works for both a 2D frame and a 3D
    volume (see motion_correction_3d.py, which reuses _apply_shift/
    _estimate_shift as-is).

    An "unlucky" size -- one with a large prime factor, e.g. a 500px
    frame split into a grid_size=32 grid of patches lands most patches
    at 31px wide, and 31 is prime -- can make an FFT several times
    slower than a same-ballpark size with only small prime factors
    (confirmed empirically: a 509x509 FFT took ~5x longer than padding
    up to 512x512 first, including the padding's own cost).

    An axis is only padded if its length has a prime factor larger than
    ``_SMOOTH_ENOUGH_PRIME`` -- pocketfft handles a moderately-composite
    size efficiently enough (a factor of, say, 53 costs maybe 1.5x a
    fully smooth transform), and for a whole 3D volume the padding copy
    is gigabytes, dwarfing that. A genuinely pathological size (a large
    prime like 509, or a small prime like the 31px patch above) still
    gets padded. No-op (no copy) when every axis is smooth enough."""
    target_shape = tuple(
        next_fast_len(n) if _largest_prime_factor(n) > _SMOOTH_ENOUGH_PRIME else n for n in image.shape
    )
    if target_shape == image.shape:
        return image
    pad_width = [(0, target - n) for n, target in zip(image.shape, target_shape)]
    return np.pad(image, pad_width, mode=mode)


def _resolve_max_workers(max_workers: int | None, bin_width: int, default: int = _DEFAULT_MAX_WORKERS) -> int:
    if max_workers is not None:
        return max_workers
    return max(1, min(default, available_cpu_count(), bin_width))


def _median_over_axis(arr: np.ndarray, axis: int) -> np.ndarray:
    """Chunked ``np.median(arr, axis=axis)`` -- float32 output (motion
    correction never needs more), and ``max_block=None`` since only the
    byte budget matters here (a chunk safely spanning thousands of
    pixels is fine; capping it lower would just mean more np.median
    calls for no memory benefit). ``arr`` is never mutated."""
    return chunked_median(arr, axis, out_dtype=np.float32, max_block=None)


def _as_float_working_copy(movie: np.ndarray) -> np.ndarray:
    """A float32 (H, W, T) array this function owns and can register
    into in place, without mutating whatever the caller passed in.

    float32, not float64: phase-correlation shift estimates are
    unaffected by the narrower mantissa at realistic upsample_factors
    (20-50), and raw microscopy movies -- usually uint16 or float32 --
    are exactly the inputs large enough for a float64 working copy to
    matter (a uint16 volume becomes a 4x-larger array as float64 vs 2x
    as float32, with the original still referenced alongside it). Matches
    the float32 FITS memmap the chunked-Commit path already writes into
    (see _setup_registration's ``output``).

    A dtype conversion (the common case: raw movies are uint16 or, when
    already float, usually float64) already allocates a fresh, private,
    writable array -- calling .copy() on top of that in every case
    doubled peak memory for exactly the inputs most likely to be large
    (measured: ~13x a uint16 movie's raw size, vs ~7x once this
    redundant copy is skipped). So the explicit copy is made only when
    the input is *already* float32, the one case where np.asarray would
    hand back something backed by the caller's data. That test is on the
    dtype, not on object identity: np.asarray of a float32 np.memmap
    returns a plain-ndarray *view* (fails ``is``) that is also read-only,
    and registering into that raises rather than copying."""
    if movie.dtype == np.float32:
        return np.array(movie, dtype=np.float32)  # our own writable copy
    return np.asarray(movie, dtype=np.float32)  # conversion already made a fresh, writable array


def _apply_shift(frame: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Apply a rigid (dy, dx) subpixel shift via frequency-domain warping.

    Pads to an FFT-fast size first (see _pad_to_fast_len) and crops back
    to ``frame``'s own shape afterward -- "edge" padding (not zeros),
    since unlike _estimate_shift (which only ever reads off a shift
    scalar and discards the padded array), this array IS the output:
    zero-padding would risk shifting real content into view of a
    sharp-edged all-zero region near the boundary. _pad_to_fast_len is a
    no-op (no copy) when ``frame`` is already a fast size, so the crop
    below is then a full-extent, effectively free slice.

    scipy.fft (not np.fft) so a float32 ``frame`` -- the working dtype
    since _as_float_working_copy -- stays complex64 through the
    transform rather than being upcast to complex128, halving the
    transient a whole-volume 3D FFT costs (see motion_correction_3d.py,
    which reuses this unchanged). ``overwrite_x``/``output=`` let fftn,
    fourier_shift, and ifftn all reuse the same complex buffer rather
    than each allocating their own -- one whole-volume complex64 array
    (gigabytes at a real volume's size) instead of three."""
    padded = _pad_to_fast_len(frame, mode="edge")
    spectrum = fftn(padded, overwrite_x=True)
    fourier_shift(spectrum, shift, output=spectrum)
    shifted = np.real(ifftn(spectrum, overwrite_x=True))
    return shifted[tuple(slice(0, n) for n in frame.shape)].astype(frame.dtype, copy=False)


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
    of the first ``init_batch`` frames. Returns ``(template, initial_template)``.

    _median_over_axis casts to float32 a slab at a time, so the
    bootstrap template is identical whether the movie is already the
    float32 working copy (in-RAM path) or still the caller's raw array
    (``output=`` path), without materializing ``init_batch`` frames as
    float32 up front."""
    if template is None:
        template = _median_over_axis(movie[:, :, : min(init_batch, movie.shape[-1])], axis=2)
    return template, template.copy()


def _upsampled_dft(data, upsampled_region_size, upsample_factor, axis_offsets):
    """Upsampled DFT of ``data`` by matrix multiplication, evaluating only
    an ``upsampled_region_size``-sized neighbourhood rather than a full
    upsample_factor-times-larger FFT -- the matrix-multiply DFT trick from
    Guizar-Sicairos, Thurman & Fienup, Opt. Lett. 33, 156-158 (2008).

    Copied from skimage.registration._phase_cross_correlation._upsampled_dft
    (BSD-3-Clause) unmodified -- see _phase_correlate_shift for why this
    isn't just called from skimage directly."""
    im2pi = 1j * 2 * np.pi
    for n_items, ups_size, ax_offset in zip(data.shape[::-1], upsampled_region_size[::-1], axis_offsets[::-1]):
        kernel = (np.arange(ups_size) - ax_offset)[:, None] * np.fft.fftfreq(n_items, upsample_factor)
        kernel = np.exp(-im2pi * kernel).astype(data.dtype, copy=False)
        data = np.tensordot(kernel, data, axes=(1, -1))
    return data


def _argmax_magnitude(arr: np.ndarray) -> tuple[int, ...]:
    """``np.unravel_index(np.argmax(np.abs(arr)), arr.shape)`` without
    materializing a whole-array magnitude buffer -- one axis-0 slab (see
    orbit._blocks) at a time instead. Exact same result; on a whole
    volume's complex spectrum this is the difference between a ~64 MiB
    transient and a multi-GB one."""
    best_value = -1.0
    best_index: tuple[int, ...] | None = None
    for sl in iter_axis_slices(arr.shape, 0, itemsize=arr.itemsize):
        block_mag = np.abs(arr[sl])
        local_index = np.unravel_index(np.argmax(block_mag), block_mag.shape)
        value = block_mag[local_index]
        if value > best_value:
            best_value = value
            best_index = (local_index[0] + sl.start, *local_index[1:])
    return best_index


def _phase_correlate_shift(reference: np.ndarray, moving: np.ndarray, upsample_factor: int, normalization: str | None) -> np.ndarray:
    """Subpixel shift via phase correlation -- a leaner reimplementation of
    ``skimage.registration.phase_cross_correlation`` (same Guizar-Sicairos
    algorithm; validated bit-for-bit identical against it across upsample
    factors, both normalization modes, and 2D/3D shapes), returning only
    the shift vector.

    skimage's version keeps several whole-array complex buffers alive at
    once (it also computes an error/phasediff we never read, at the cost
    of a couple more) -- at a real volume's size, gigabytes each. This
    version reuses buffers in place (``out=``/``overwrite_x=True``) and
    never materializes a whole-array magnitude (see _argmax_magnitude),
    so at most ~2 whole-array complex64 buffers are alive at once:
    measured ~10x -> ~4x one volume's size on a real (150, 3200, 530)
    -shaped pair."""
    src_freq = fftn(reference, overwrite_x=False)
    target_freq = fftn(moving, overwrite_x=True)
    np.conjugate(target_freq, out=target_freq)
    image_product = src_freq
    np.multiply(src_freq, target_freq, out=image_product)
    del target_freq  # its data lives on, conjugated, inside image_product

    shape = image_product.shape
    float_dtype = image_product.real.dtype

    if normalization == "phase":
        eps = np.finfo(float_dtype).eps
        denom = np.maximum(np.abs(image_product), 100 * eps)
        image_product /= denom
        del denom
    elif normalization is not None:
        raise ValueError("normalization must be either phase or None")

    cross_correlation = ifftn(image_product)  # a 2nd whole-array buffer -- image_product is still needed below
    midpoint = np.array([n // 2 for n in shape])
    shift = np.array(_argmax_magnitude(cross_correlation), dtype=float_dtype)
    shift[shift > midpoint] -= np.array(shape)[shift > midpoint]
    del cross_correlation  # back down to one whole-array buffer

    if upsample_factor != 1:
        upsample_factor = np.asarray(upsample_factor, dtype=float_dtype)
        shift = np.round(shift * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        dftshift = np.trunc(upsampled_region_size / 2.0)
        sample_region_offset = dftshift - shift * upsample_factor
        np.conjugate(image_product, out=image_product)  # image_product's last use -- conjugate it in place
        local_cc = _upsampled_dft(
            image_product, [upsampled_region_size] * len(shape), upsample_factor, list(sample_region_offset)
        ).conj()  # tiny (an upsampled_region_size-per-axis neighbourhood, not whole-array)
        local_max = np.array(np.unravel_index(np.argmax(np.abs(local_cc)), local_cc.shape), dtype=float_dtype)
        shift += (local_max - dftshift) / upsample_factor

    for dim, n in enumerate(shape):
        if n == 1:
            shift[dim] = 0.0
    return shift


def _estimate_shift(
    reference: np.ndarray, moving: np.ndarray, upsample_factor: int, normalization: str | None, max_shift: float
) -> np.ndarray:
    """Subpixel (dy, dx) shift of ``moving`` relative to ``reference``, clipped to ``max_shift``.

    Both arrays are padded to an FFT-fast size first (see
    _pad_to_fast_len) -- unlike _apply_shift, nothing here is cropped
    back afterward, since only a shift vector is read off, not pixel
    data. Padding mode matters a lot here, confirmed empirically: zero
    padding (this function's first attempt) is NOT a harmless no-op the
    way it sounds -- it adds a hard, perfectly-aligned-between-the-two-
    images edge discontinuity that dominates the cross-power spectrum
    and made phase_cross_correlation report a shift of exactly zero
    regardless of the images' true relative shift on a synthetic
    known-shift test. "edge" padding (extending each border's own pixel
    values outward, not dropping to zero) avoids that dominant artifact
    and tracked the true shift closely in the same test (within ~0.1px
    at upsample_factor=50) -- still a small approximation, not exact,
    but a reasonable trade for the speedup on an unlucky size. Patch-
    based correction is the main beneficiary -- _patch_centers's patch
    widths have no reason to land on an FFT-friendly size (e.g. a 500px
    frame split grid_size=32-ish lands most patches at 31px, which is
    prime), and this function is called once per patch per frame there.

    Shift estimation itself is _phase_correlate_shift, a leaner
    reimplementation of skimage's phase_cross_correlation -- see its own
    docstring. Only ``shift`` is ever needed here (not skimage's
    error/phasediff), which is also why that version never hits the
    float32 overflow skimage's error term used to warn about on a whole
    volume."""
    reference = _pad_to_fast_len(reference, mode="edge")
    moving = _pad_to_fast_len(moving, mode="edge")
    shift = _phase_correlate_shift(reference, moving, upsample_factor, normalization)
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
    each other, so they register concurrently in a thread pool (scipy's
    FFT/tensordot and spline routines release the GIL)."""
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

        template = _median_over_axis(registered[:, :, chunk_start:chunk_end], axis=2)

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
        registered = movie  # movie is already a private float32 copy -- no second copy needed
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
            # float32 per frame -- a no-op view when _source is already
            # the float32 working copy, but when it's the caller's raw
            # (uint16/float32 memmap) movie -- the ``output=`` path --
            # this bounds each worker's FFT transient to one float32
            # frame rather than one complex128 one.
            frame = np.asarray(_source[:, :, t], dtype=np.float32)
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
            frame = np.asarray(_source[:, :, t], dtype=np.float32)  # see rigid_motion_correct's _process_one
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
