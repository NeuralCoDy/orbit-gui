import numpy as np
import pytest

pytest.importorskip("astropy")

from astropy.io import fits  # noqa: E402

from orbitapp.io import is_memmap  # noqa: E402
from orbitapp.volumetric_io import load_volumetric_movie  # noqa: E402


def _write_4d_fits(path, shape=(3, 4, 5, 6), dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(shape).astype(dtype)
    fits.PrimaryHDU(data=arr).writeto(path, overwrite=True)
    return arr


def test_load_volumetric_movie_full_load_matches_original_shape_and_values(tmp_path):
    path = tmp_path / "volume.fits"
    original = _write_4d_fits(path)

    movie = load_volumetric_movie(path)

    assert movie.shape == (3, 4, 5, 6)  # (T, L, W, D), astropy's own axis handling, not reversed
    assert not is_memmap(movie)
    np.testing.assert_allclose(np.asarray(movie, dtype=np.float32), original)


def test_load_volumetric_movie_mmap_true_returns_a_real_memmap(tmp_path):
    path = tmp_path / "volume.fits"
    original = _write_4d_fits(path)

    movie = load_volumetric_movie(path, mmap=True)

    assert is_memmap(movie)
    assert movie.shape == (3, 4, 5, 6)
    np.testing.assert_allclose(np.asarray(movie, dtype=np.float32), original)


def test_load_volumetric_movie_rejects_a_non_4d_file(tmp_path):
    path = tmp_path / "flat.fits"
    fits.PrimaryHDU(data=np.zeros((5, 6, 7), dtype=np.float32)).writeto(path, overwrite=True)

    with pytest.raises(ValueError, match="4D"):
        load_volumetric_movie(path)


def test_load_volumetric_movie_mmap_falls_back_with_warning_for_unmappable_dtype(tmp_path):
    # Unsigned dtypes get a BZERO/BSCALE header offset that astropy
    # itself refuses to memory-map (see fits_io.py) -- confirmed the
    # same limitation applies when reading an externally-provided file,
    # not just ones this app wrote itself.
    path = tmp_path / "volume_uint16.fits"
    original = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
    fits.PrimaryHDU(data=original).writeto(path, overwrite=True)

    with pytest.warns(UserWarning, match="can't be memory-mapped"):
        movie = load_volumetric_movie(path, mmap=True)

    assert not is_memmap(movie)
    assert movie.shape == (2, 3, 4, 5)
    np.testing.assert_array_equal(np.asarray(movie), original)


def test_load_volumetric_movie_preserves_dtype_values_for_integer_data(tmp_path):
    path = tmp_path / "volume_int16.fits"
    original = np.arange(-10, 10, dtype=np.int16).reshape(2, 2, 1, 5)
    fits.PrimaryHDU(data=original).writeto(path, overwrite=True)

    movie = load_volumetric_movie(path)
    np.testing.assert_array_equal(np.asarray(movie), original)
