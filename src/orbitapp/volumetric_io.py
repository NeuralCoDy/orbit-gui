"""Loading for volumetric (time x length x width x depth) movies -- a
totally separate code path from orbitapp.io's (H, W, T) 2D+time movie
loader, used only when the Load tab's "Volumetric" toggle is on (see
AppState.volumetric). Reuses established, already-tested pieces where
they apply directly (orbitapp.fits_io.open_fits_memmap for the memmap
case, orbitapp.io.is_memmap to detect it) rather than duplicating them,
but is otherwise independent: nothing here is imported by, or changes
the behavior of, the existing 2D pipeline.

Assumes the source is a single-HDU 4D FITS file already shaped
(T, L, W, D) -- astropy's own read/write already undoes FITS's
axis-order-reversed-vs-numpy convention (confirmed empirically:
round-tripping a (T, L, W, D) array through fits.PrimaryHDU/fits.open
gives back that exact shape, not reversed), so no manual axis handling
is needed here.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits

from .fits_io import open_fits_memmap


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
