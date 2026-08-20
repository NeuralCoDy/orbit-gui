import numpy as np
import pytest

pytest.importorskip("astropy")

from orbitapp.fits_io import create_fits_memmap, open_fits_memmap  # noqa: E402
from orbitapp.io import is_memmap  # noqa: E402


def test_create_fits_memmap_is_a_real_memmap_and_starts_zeroed(tmp_path):
    out = create_fits_memmap(tmp_path / "movie.fits", (4, 5, 20), np.float32)

    assert is_memmap(out)
    assert out.shape == (4, 5, 20)
    assert np.array_equal(np.asarray(out), np.zeros((4, 5, 20)))


def test_chunked_writes_persist_and_round_trip_through_open_fits_memmap(tmp_path):
    path = tmp_path / "movie.fits"
    shape = (4, 5, 100)
    out = create_fits_memmap(path, shape, np.float32)

    for t0 in range(0, shape[-1], 30):
        t1 = min(t0 + 30, shape[-1])
        out[:, :, t0:t1] = np.full((shape[0], shape[1], t1 - t0), t0, dtype=np.float32)
    out.flush()
    del out

    readback = open_fits_memmap(path)
    assert is_memmap(readback)
    assert readback.shape == shape
    assert np.array_equal(readback[:, :, 0], np.zeros((4, 5)))
    assert np.array_equal(readback[:, :, 30], np.full((4, 5), 30))
    assert np.array_equal(readback[:, :, 90], np.full((4, 5), 90))


def test_dtype_is_preserved_for_supported_types(tmp_path):
    # Unsigned 16/32/64-bit dtypes aren't supported (see fits_io.py's
    # module docstring) -- every stage's committed output is
    # floating-point regardless, so this covers what's actually used.
    for dtype in (np.float32, np.float64, np.int16, np.uint8):
        path = tmp_path / f"{np.dtype(dtype).name}.fits"
        out = create_fits_memmap(path, (2, 2, 5), dtype)
        assert out.dtype.itemsize == np.dtype(dtype).itemsize
        out[:] = 7
        out.flush()
        del out
        readback = open_fits_memmap(path)
        assert np.array_equal(np.asarray(readback), np.full((2, 2, 5), 7))


def test_open_fits_memmap_readonly_mode_rejects_writes(tmp_path):
    path = tmp_path / "movie.fits"
    out = create_fits_memmap(path, (3, 3, 10), np.float32)
    out.flush()
    del out

    readonly = open_fits_memmap(path, mode="r")
    with pytest.raises(ValueError):
        readonly[0, 0, 0] = 1.0
