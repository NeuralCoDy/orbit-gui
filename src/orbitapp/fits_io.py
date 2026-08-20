"""Disk-backed (H, W, T) arrays via FITS, for a stage's chunked Commit
output when the input movie is memory-mapped and too large to hold a
full in-RAM result.

astropy's own ``HDUList[0].data`` (even with ``memmap=True``) does NOT
return a real ``numpy.memmap`` -- it's a view backed by a raw
``mmap.mmap`` a couple of ``.base`` hops down, which fails
``isinstance(array, np.memmap)`` (confirmed empirically: `arr.base.base`
is a plain ``mmap.mmap``, not a ``numpy.memmap``). Since
``orbitapp.io.is_memmap`` is the single check the rest of the app uses
to decide whether to take the bounded-memory chunked path, that
mismatch would silently break memmap detection for every committed
FITS output. Both functions below therefore use astropy only to manage
the FITS header (so files stay standard, tool-interoperable FITS), and
construct the actual data array as a genuine ``numpy.memmap`` directly
against the file, at the byte offset astropy's own header reports.

Only signed-integer and floating-point dtypes are supported (float32,
float64, int16, int32, int64, uint8 -- FITS's BITPIX values). Unsigned
16/32/64-bit dtypes aren't: FITS has no native representation for them,
so astropy's writer always adds a BZERO/BSCALE header offset for those
-- which then makes astropy itself refuse a plain memmap read back
("Cannot load a memory-mapped image: BZERO/BSCALE/BLANK header keywords
present"). Every stage's committed output through this module is
floating-point anyway (subpixel shifts, filtering, normalization all
produce continuous values), so this isn't a real limitation in
practice; callers should convert to float32 before writing rather than
trying to preserve e.g. a raw uint16 movie's dtype exactly.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from astropy.io import fits

_BLOCK_SIZE = 2880  # FITS header/data records are always padded to a multiple of this many bytes


def _round_up_to_block(n_bytes: int) -> int:
    return ((n_bytes + _BLOCK_SIZE - 1) // _BLOCK_SIZE) * _BLOCK_SIZE


def create_fits_memmap(path: str | Path, shape: tuple[int, ...], dtype: np.dtype | type) -> np.memmap:
    """Pre-allocates a new FITS file of the given ``shape``/``dtype`` on
    disk (without ever materializing the full array in RAM -- writes a
    zero-length stub HDU, then extends the file to its true size with a
    single seek + 1-byte write) and returns a writable ``numpy.memmap``
    onto its data block. Caller writes into slices of the returned array
    (e.g. one time-chunk at a time) and should call ``.flush()`` when
    done to guarantee everything reaches disk.

    FITS stores data big-endian by convention -- the returned memmap's
    dtype reflects that (e.g. ``>f4`` for float32), which numpy handles
    transparently (native-endian values written into it are byte-swapped
    on the way in); nothing downstream needs to special-case this.
    """
    path = str(path)
    dtype = np.dtype(dtype).newbyteorder(">")

    # A stub with 0 rows along the first axis keeps this from ever
    # materializing `shape`-sized data in RAM -- only the header (a few
    # KB) is written by tofile(); NAXIS1..NAXISn are then corrected to
    # the true final shape (FITS axis order is numpy's shape reversed)
    # before the file is extended to match on disk.
    stub_shape = (0,) + tuple(shape[1:])
    hdu = fits.PrimaryHDU(data=np.zeros(stub_shape, dtype=dtype))
    header = hdu.header
    while len(header) < (36 * 4 - 1):  # reserve header room so later edits don't need a rewrite
        header.append()
    for i, dim in enumerate(reversed(shape), start=1):
        header[f"NAXIS{i}"] = dim
    header.tofile(path, overwrite=True)

    header_bytes = os.path.getsize(path)
    data_bytes = int(np.prod(shape)) * dtype.itemsize
    with open(path, "rb+") as f:
        f.seek(header_bytes + _round_up_to_block(data_bytes) - 1)
        f.write(b"\0")

    return np.memmap(path, dtype=dtype, mode="r+", offset=header_bytes, shape=tuple(shape))


def open_fits_memmap(path: str | Path, mode: str = "r") -> np.memmap:
    """Reopens a FITS file written by ``create_fits_memmap`` (or any
    standard single-HDU FITS file) as a ``numpy.memmap`` -- ``mode`` is
    forwarded to ``numpy.memmap`` (``"r"`` read-only, ``"r+"`` to keep
    writing into it)."""
    path = str(path)
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        shape, dtype = data.shape, data.dtype
        offset = hdul.fileinfo(0)["datLoc"]
    return np.memmap(path, dtype=dtype, mode=mode, offset=offset, shape=shape)
