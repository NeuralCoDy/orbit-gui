import numpy as np
import pytest

from orbitapp.io import load_movie


def test_load_npy_roundtrip(tmp_path):
    movie = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    path = tmp_path / "movie.npy"
    np.save(path, movie)

    loaded = load_movie(path)

    assert loaded.shape == movie.shape
    assert np.array_equal(loaded, movie)


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
