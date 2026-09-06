import numpy as np
import pytest

pytest.importorskip("astropy")

from astropy.io import fits  # noqa: E402

from orbitapp.io import is_memmap  # noqa: E402
from orbitapp.volumetric_io import (  # noqa: E402
    INTERLEAVED,
    ONE_VOLUME_PER_STACK,
    load_volumetric_movie,
    load_volumetric_tiff_folder,
    preview_slice_volumetric,
)


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


def _write_tiff_stacks(folder, volumes):
    """``volumes`` is (T, D, L, W) -- writes one multi-page TIFF file
    per timepoint, each holding that timepoint's D pages."""
    tifffile = pytest.importorskip("tifffile")
    folder.mkdir(exist_ok=True)
    for t, vol in enumerate(volumes):
        tifffile.imwrite(folder / f"vol_{t:03d}.tif", vol)


def test_one_volume_per_stack_reads_each_file_as_one_timepoint(tmp_path):
    T, D, L, W = 3, 4, 5, 6
    volumes = np.arange(T * D * L * W, dtype=np.float32).reshape(T, D, L, W)
    _write_tiff_stacks(tmp_path, volumes)

    result = load_volumetric_tiff_folder(tmp_path, ONE_VOLUME_PER_STACK)

    assert result.shape == (T, L, W, D)
    np.testing.assert_array_equal(result, np.moveaxis(volumes, 1, -1))


def test_one_volume_per_stack_reports_progress_per_file(tmp_path):
    T, D, L, W = 4, 2, 3, 3
    volumes = np.arange(T * D * L * W, dtype=np.float32).reshape(T, D, L, W)
    _write_tiff_stacks(tmp_path, volumes)

    calls = []
    load_volumetric_tiff_folder(
        tmp_path, ONE_VOLUME_PER_STACK, progress_callback=lambda done, total: calls.append((done, total))
    )

    assert calls == [(i, T) for i in range(1, T + 1)]


def test_one_volume_per_stack_rejects_mismatched_file_shapes(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    tifffile.imwrite(tmp_path / "a.tif", np.zeros((4, 5, 6), dtype=np.float32))
    tifffile.imwrite(tmp_path / "b.tif", np.zeros((4, 5, 7), dtype=np.float32))  # different width

    with pytest.raises(ValueError, match="same shape"):
        load_volumetric_tiff_folder(tmp_path, ONE_VOLUME_PER_STACK)


def test_one_volume_per_stack_treats_a_single_page_file_as_a_one_slice_volume(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    tifffile.imwrite(tmp_path / "a.tif", np.full((5, 6), 1.0, dtype=np.float32))
    tifffile.imwrite(tmp_path / "b.tif", np.full((5, 6), 2.0, dtype=np.float32))

    result = load_volumetric_tiff_folder(tmp_path, ONE_VOLUME_PER_STACK)

    assert result.shape == (2, 5, 6, 1)
    np.testing.assert_array_equal(result[0, :, :, 0], np.full((5, 6), 1.0))
    np.testing.assert_array_equal(result[1, :, :, 0], np.full((5, 6), 2.0))


def test_interleaved_chops_a_continuous_stream_of_slices_into_volumes(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    n_files, pages_per_file, depth, L, W = 2, 6, 3, 5, 6
    all_slices = np.arange(n_files * pages_per_file * L * W, dtype=np.float32).reshape(
        n_files * pages_per_file, L, W
    )
    for i in range(n_files):
        tifffile.imwrite(
            tmp_path / f"f_{i:03d}.tif", all_slices[i * pages_per_file : (i + 1) * pages_per_file]
        )

    result = load_volumetric_tiff_folder(tmp_path, INTERLEAVED, depth=depth)

    n_volumes = (n_files * pages_per_file) // depth
    expected = np.moveaxis(all_slices.reshape(n_volumes, depth, L, W), 1, -1)
    assert result.shape == (n_volumes, L, W, depth)
    np.testing.assert_array_equal(result, expected)


def test_interleaved_requires_depth(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    tifffile.imwrite(tmp_path / "a.tif", np.zeros((4, 5, 6), dtype=np.float32))

    with pytest.raises(ValueError, match="depth"):
        load_volumetric_tiff_folder(tmp_path, INTERLEAVED, depth=None)


def test_interleaved_rejects_a_slice_count_not_divisible_by_depth(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    tifffile.imwrite(tmp_path / "a.tif", np.zeros((5, 4, 4), dtype=np.float32))  # 5 slices total

    with pytest.raises(ValueError, match="multiple"):
        load_volumetric_tiff_folder(tmp_path, INTERLEAVED, depth=3)


def test_load_volumetric_tiff_folder_raises_for_an_empty_folder(tmp_path):
    with pytest.raises(ValueError, match="No .tif"):
        load_volumetric_tiff_folder(tmp_path, ONE_VOLUME_PER_STACK)


def test_load_volumetric_tiff_folder_rejects_an_unknown_mode(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    tifffile.imwrite(tmp_path / "a.tif", np.zeros((4, 5, 6), dtype=np.float32))

    with pytest.raises(ValueError, match="Unknown mode"):
        load_volumetric_tiff_folder(tmp_path, "bogus")


def test_preview_slice_volumetric_caps_a_long_memmap_but_leaves_short_or_in_ram_movies_alone(tmp_path):
    path = tmp_path / "movie.fits"
    movie = _write_4d_fits(path, shape=(20, 3, 4, 5))
    mmap_movie = load_volumetric_movie(path, mmap=True)

    capped = preview_slice_volumetric(mmap_movie, max_frames=5)
    assert capped.shape == (5, 3, 4, 5)
    assert is_memmap(capped)
    assert np.array_equal(capped, movie[:5])

    # a memmap shorter than max_frames, or a plain in-RAM array, pass through unchanged
    assert preview_slice_volumetric(mmap_movie, max_frames=100) is mmap_movie
    in_ram = np.array(movie)
    assert preview_slice_volumetric(in_ram, max_frames=5) is in_ram
