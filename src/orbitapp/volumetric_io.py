"""Loading for volumetric (time x length x width x depth) movies -- a
totally separate code path from orbitapp.io's (H, W, T) 2D+time movie
loader, used only when the Load tab's "Volumetric" toggle is on (see
AppState.volumetric). Reuses established, already-tested pieces where
they apply directly (orbitapp.fits_io.open_fits_memmap for the memmap
case, orbitapp.io.is_memmap to detect it) rather than duplicating them,
but is otherwise independent: nothing here is imported by, or changes
the behavior of, the existing 2D pipeline.

Two source formats, both producing a (T, L, W, D) array:

load_volumetric_movie -- a single 4D FITS file already shaped that way.
Astropy's own read/write already undoes FITS's axis-order-reversed-vs-
numpy convention (confirmed empirically: round-tripping a (T, L, W, D)
array through fits.PrimaryHDU/fits.open gives back that exact shape,
not reversed), so no manual axis handling is needed here.

load_volumetric_tiff_folder -- a folder of TIFF files, for data that
isn't already packaged as one 4D FITS file. A folder alone is
ambiguous about where one volume ends and the next begins (unlike a
single FITS file's own header), so the caller must resolve that first
-- see the ``mode`` argument (orbitapp.widgets.volumetric_load_dialog
is the popup that asks the user).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import numpy as np
from astropy.io import fits

from .fits_io import open_fits_memmap
from .io import is_memmap

ONE_VOLUME_PER_STACK = "one_volume_per_stack"
INTERLEAVED = "interleaved"


def preview_slice_volumetric(
    movie: np.ndarray, max_frames: int = 5000, max_voxels: int | None = None
) -> np.ndarray:
    """The leading timepoints of a (T, L, W, D) ``movie`` -- same purpose
    as orbitapp.io.preview_slice (a cheap view, no copy) but capped on
    axis 0, not axis -1 (volumetric movies put time first).

    ``max_frames`` only trims a disk-backed movie, matching preview_slice.
    ``max_voxels``, if given, additionally trims *any* movie (memmap or
    in-RAM) so the preview stays under that many voxels total -- a
    per-timepoint cap is meaningless for a wide volume, where even a
    handful of (L, W, D) timepoints is many GB once converted to float
    for registration. Callers that run heavy per-voxel compute on the
    preview (the stage tabs' Apply) pass this; display-only callers
    don't."""
    T = movie.shape[0]
    limit = T
    if is_memmap(movie) and T > max_frames:
        limit = max_frames
    if max_voxels is not None:
        per_timepoint = movie.size // T
        limit = min(limit, max(1, max_voxels // per_timepoint))
    return movie[:limit] if limit < T else movie


def load_volumetric_movie(path: str | Path, mmap: bool = False) -> np.ndarray:
    """Loads a 4D (T, L, W, D) volumetric movie from a FITS file.

    ``mmap=True`` keeps it disk-backed (reusing
    orbitapp.fits_io.open_fits_memmap, which already works for any
    standard single-HDU FITS file, not just ones this app wrote itself)
    -- falls back to a full in-RAM load with a warning if the file's
    dtype can't be memory-mapped (FITS has no native unsigned-integer
    representation; astropy adds a BZERO/BSCALE header offset for those,
    which then makes astropy itself refuse a plain memmap read -- see
    fits_io.py's own docstring for the same limitation on the write
    side).
    """
    path = Path(path)
    if mmap:
        try:
            movie = open_fits_memmap(path, mode="r")
        except ValueError as exc:
            warnings.warn(
                f"Memory mapping requested but {path} can't be memory-mapped ({exc}) -- loaded "
                "fully into RAM instead.", stacklevel=2,
            )
            movie = _load_full(path)
    else:
        movie = _load_full(path)

    if movie.ndim != 4:
        raise ValueError(f"Expected a 4D (T, L, W, D) volumetric FITS file, got shape {movie.shape} from {path}")
    return movie


def _load_full(path: Path) -> np.ndarray:
    with fits.open(path) as hdul:
        return np.ascontiguousarray(hdul[0].data)


def load_volumetric_tiff_folder(
    path: str | Path, mode: str, depth: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Loads every TIFF file in ``path`` (sorted by name) as a (T, L, W,
    D) volumetric movie. ``mode`` resolves the file-vs-volume ambiguity
    a folder has that a single 4D FITS file doesn't:

    ``ONE_VOLUME_PER_STACK``: each file IS one full volume at one time
    point -- T is the number of files, and (L, W, D) come directly from
    each file's own multi-page stack shape (every file must share the
    same shape; a single-page file is treated as a 1-slice volume).

    ``INTERLEAVED``: every page across every file, in file-then-page
    order, is one continuous stream of 2D slices -- every ``depth``
    consecutive slices become one volume, so ``depth`` is required and
    the total slice count must be an exact multiple of it.

    Reading is dominated by per-file TIFF decompression (tens of
    seconds a file on a large, heavily-compressed stack), done one file
    at a time -- ``progress_callback``, if given, is called
    ``(files_read, total_files)`` after each file.
    """
    import tifffile

    path = Path(path)
    files = sorted(path.glob("*.tif")) + sorted(path.glob("*.tiff"))
    if not files:
        raise ValueError(f"No .tif/.tiff files found in {path}")

    stacks = []
    for i, f in enumerate(files):
        stack = tifffile.imread(f)
        if stack.ndim == 2:
            stack = stack[None, ...]  # a single-page file is a 1-slice stack
        elif stack.ndim != 3:
            raise ValueError(f"Expected a 2D or 3D (page, H, W) TIFF stack, got shape {stack.shape} from {f}")
        stacks.append(stack)  # each (n_pages, L, W)
        if progress_callback is not None:
            progress_callback(i + 1, len(files))

    if mode == ONE_VOLUME_PER_STACK:
        shapes = {s.shape for s in stacks}
        if len(shapes) > 1:
            raise ValueError(
                f"One volume per TIFF stack requires every file to share the same shape, got: "
                f"{ {str(f.name): s.shape for f, s in zip(files, stacks)} }"
            )
        return np.stack([np.moveaxis(s, 0, -1) for s in stacks], axis=0)  # (T, L, W, D)

    if mode == INTERLEAVED:
        if not depth or depth < 1:
            raise ValueError("depth (slices per volume) is required for interleaved mode")
        flat = np.concatenate(stacks, axis=0)  # (total_slices, L, W)
        total_slices = flat.shape[0]
        if total_slices % depth != 0:
            raise ValueError(
                f"Total slice count ({total_slices}) across all files isn't an exact multiple of "
                f"depth ({depth})"
            )
        volumes = flat.reshape(total_slices // depth, depth, *flat.shape[1:])  # (T, D, L, W)
        return np.moveaxis(volumes, 1, -1)  # (T, L, W, D)

    raise ValueError(f"Unknown mode: {mode!r} (expected {ONE_VOLUME_PER_STACK!r} or {INTERLEAVED!r})")
