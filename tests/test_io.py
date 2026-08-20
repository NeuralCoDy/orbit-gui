import numpy as np
import pytest

from orbitapp.io import is_memmap, load_movie, preview_slice


def test_load_npy_roundtrip(tmp_path):
    movie = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    path = tmp_path / "movie.npy"
    np.save(path, movie)

    loaded = load_movie(path)

    assert loaded.shape == movie.shape
    assert np.array_equal(loaded, movie)


def test_load_npy_mmap_returns_a_real_memmap(tmp_path):
    movie = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    path = tmp_path / "movie.npy"
    np.save(path, movie)

    loaded = load_movie(path, mmap=True)

    assert is_memmap(loaded)
    assert np.array_equal(loaded, movie)


def test_load_tiff_stack_without_mmap_is_never_a_memmap_even_though_the_loader_tries_one_internally(tmp_path):
    # _load_tiff_stack opportunistically tries tifffile.memmap() even
    # when mmap=False was requested (fewer copies either way) -- a real
    # memmap result from that internal attempt must not leak out as the
    # return value unless the caller actually asked for mmap=True, or
    # every ordinary (non-memmap) load of a well-formed contiguous TIFF
    # would silently come back disk-backed instead of a plain in-RAM
    # array, which downstream code relies on by default.
    tifffile = pytest.importorskip("tifffile")

    frames = np.arange(3 * 5 * 4, dtype="uint16").reshape(3, 5, 4)
    path = tmp_path / "movie.tif"
    tifffile.imwrite(path, frames)

    loaded = load_movie(path)  # mmap=False (default)

    assert not is_memmap(loaded)
    assert np.array_equal(loaded, np.moveaxis(frames, 0, -1))


def test_load_tiff_stack_mmap_returns_a_real_memmap(tmp_path):
    tifffile = pytest.importorskip("tifffile")

    frames = np.arange(3 * 5 * 4, dtype="uint16").reshape(3, 5, 4)  # (T, H, W)
    path = tmp_path / "movie.tif"
    tifffile.imwrite(path, frames)

    loaded = load_movie(path, mmap=True)

    assert is_memmap(loaded)
    assert loaded.shape == (5, 4, 3)  # (H, W, T)
    assert np.array_equal(loaded, np.moveaxis(frames, 0, -1))


def test_load_mmap_unsupported_format_falls_back_and_warns(tmp_path):
    h5py = pytest.importorskip("h5py")

    movie = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
    path = tmp_path / "movie.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("movie", data=movie)

    with pytest.warns(UserWarning, match="Memory mapping requested"):
        loaded = load_movie(path, mmap=True)

    assert not is_memmap(loaded)
    assert np.array_equal(loaded, movie)


def test_preview_slice_caps_a_long_memmap_but_leaves_short_or_in_ram_movies_alone(tmp_path):
    movie = np.arange(2 * 3 * 20, dtype=float).reshape(2, 3, 20)
    path = tmp_path / "movie.npy"
    np.save(path, movie)
    mmap_movie = load_movie(path, mmap=True)

    capped = preview_slice(mmap_movie, max_frames=5)
    assert capped.shape == (2, 3, 5)
    assert is_memmap(capped)
    assert np.array_equal(capped, movie[:, :, :5])

    # a memmap shorter than max_frames, or a plain in-RAM array, pass through unchanged
    assert preview_slice(mmap_movie, max_frames=100) is mmap_movie
    in_ram = np.array(movie)
    assert preview_slice(in_ram, max_frames=5) is in_ram


def test_load_unsupported_suffix_raises(tmp_path):
    path = tmp_path / "movie.xyz"
    path.write_text("not a movie")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_movie(path)


def test_load_tiff_stack_transposes_to_h_w_t(tmp_path):
    tifffile = pytest.importorskip("tifffile")

    frames = np.arange(3 * 5 * 4, dtype="uint16").reshape(3, 5, 4)  # (T, H, W)
    path = tmp_path / "movie.tif"
    tifffile.imwrite(path, frames)

    loaded = load_movie(path)

    assert loaded.shape == (5, 4, 3)  # (H, W, T)
    assert np.array_equal(loaded, np.moveaxis(frames, 0, -1))


def test_load_h5_roundtrip(tmp_path):
    h5py = pytest.importorskip("h5py")

    movie = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)  # already (H, W, T)
    path = tmp_path / "movie.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("movie", data=movie)

    loaded = load_movie(path)

    assert loaded.shape == movie.shape
    assert np.array_equal(loaded, movie)


def test_load_h5_picks_the_largest_dataset_over_metadata(tmp_path):
    h5py = pytest.importorskip("h5py")

    movie = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
    path = tmp_path / "movie.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("framerate", data=30.0)  # sorts before "movie" alphabetically
        f.create_dataset("movie", data=movie)

    loaded = load_movie(path)

    assert loaded.shape == movie.shape
    assert np.array_equal(loaded, movie)


def test_load_mat_v5_roundtrip(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")

    movie = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)  # already (H, W, T)
    path = tmp_path / "movie.mat"
    scipy_io.savemat(path, {"movie": movie})

    loaded = load_movie(path)

    assert loaded.shape == movie.shape
    assert np.array_equal(loaded, movie)


def test_load_mat_v73_undoes_matlabs_hdf5_axis_reversal(tmp_path):
    # MATLAB v7.3 .mat files are HDF5, but MATLAB's own writer stores
    # arrays axis-reversed relative to their MATLAB size() (HDF5 is
    # row-major, MATLAB column-major) -- hdf5storage reproduces that
    # real MATLAB-compatible layout, unlike a plain h5py-written file.
    hdf5storage = pytest.importorskip("hdf5storage")

    movie = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)  # (H, W, T) per MATLAB's size()
    path = tmp_path / "movie_v73.mat"
    hdf5storage.savemat(str(path), {"movie": movie}, format="7.3")

    loaded = load_movie(path)

    assert loaded.shape == movie.shape  # (H, W, T), not h5py's raw (T, W, H)
    assert np.array_equal(loaded, movie)


def test_load_mat_v73_skips_refs_group_and_metadata(tmp_path):
    hdf5storage = pytest.importorskip("hdf5storage")

    movie = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
    path = tmp_path / "movie_v73_multi.mat"
    # A string variable forces hdf5storage to emit a "#refs#" bookkeeping
    # group (sorts before "movie" alphabetically) alongside a small
    # unrelated numeric variable -- both must be skipped in favor of the
    # much larger movie array.
    hdf5storage.savemat(
        str(path), {"movie": movie, "framerate": 30.0, "note": "hello"}, format="7.3", matlab_compatible=True
    )

    loaded = load_movie(path)

    assert loaded.shape == movie.shape
    assert np.array_equal(loaded, movie)
