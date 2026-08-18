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
