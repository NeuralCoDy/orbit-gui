import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from orbit.masking import apply_mask, triangle_mask  # noqa: E402
from orbitapp.io import load_movie  # noqa: E402
from orbitapp.state import AppState  # noqa: E402
from orbitapp.tabs.mask_tab import MaskTab  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def _wait(tab):
    if tab.worker is not None:
        tab.worker.wait(15000)
    for _ in range(50):
        QApplication.processEvents()


def _bright_blob_movie(height=20, width=20, n_frames=60, seed=0):
    rng = np.random.default_rng(seed)
    movie = rng.standard_normal((height, width, n_frames)).astype(np.float32) * 0.05 + 0.1
    movie[5:12, 5:12, :] += 3.0
    return np.clip(movie, 0, None).astype(np.float32)


def _memmapped_movie(tmp_path, movie):
    path = tmp_path / "movie.npy"
    np.save(path, movie)
    return path, load_movie(path, mmap=True)


def test_mask_tab_auto_threshold_apply_and_commit_on_a_non_memmap_movie():
    movie = _bright_blob_movie()
    state = AppState()
    state.load("movie.npy", movie)
    tab = MaskTab(state)
    tab.on_data_loaded()

    tab._apply()
    _wait(tab)
    tab._commit()
    _wait(tab)

    committed = state.active_data()
    expected_mask = triangle_mask(np.asarray(movie, dtype=np.float64).mean(axis=2))
    expected = apply_mask(movie, expected_mask)
    np.testing.assert_allclose(np.asarray(committed), expected)
    assert "Mask" in state.pipeline[-1]


def test_mask_tab_clear_mask_is_a_no_op_and_commits_unchanged_movie():
    movie = _bright_blob_movie()
    state = AppState()
    state.load("movie.npy", movie)
    tab = MaskTab(state)
    tab.on_data_loaded()

    tab._on_clear_clicked()  # synchronous, no worker to wait on
    tab._commit()
    _wait(tab)

    committed = state.active_data()
    np.testing.assert_allclose(np.asarray(committed), movie)


def test_mask_tab_metrics_label_reports_fraction_kept():
    movie = _bright_blob_movie()
    state = AppState()
    state.load("movie.npy", movie)
    tab = MaskTab(state)
    tab.on_data_loaded()

    tab._apply()
    _wait(tab)

    assert "%" in tab.metrics_label.text()
    metrics = tab._extract_metrics(tab._pending_result)
    assert 0.0 < metrics["fraction_kept"] < 1.0  # the blob is a small fraction of the frame


def test_mask_tab_auto_threshold_commit_of_memmap_movie_matches_whole_movie_result(tmp_path):
    movie = _bright_blob_movie()
    path, mmapped = _memmapped_movie(tmp_path, movie)
    state = AppState()
    state.load(str(path), mmapped)
    tab = MaskTab(state)
    tab.on_data_loaded()
    tab._chunk_frames = 17  # force multiple chunks over 60 frames

    tab._apply()
    _wait(tab)
    tab._commit()
    _wait(tab)

    committed = state.active_data()
    expected_mask = triangle_mask(np.asarray(mmapped, dtype=np.float64).mean(axis=2))
    expected = apply_mask(np.asarray(mmapped, dtype=np.float32), expected_mask)
    np.testing.assert_allclose(np.asarray(committed), expected, atol=1e-5)


def test_mask_tab_clear_commit_of_memmap_movie_matches_whole_movie_result(tmp_path):
    movie = _bright_blob_movie()
    path, mmapped = _memmapped_movie(tmp_path, movie)
    state = AppState()
    state.load(str(path), mmapped)
    tab = MaskTab(state)
    tab.on_data_loaded()
    tab._chunk_frames = 17

    tab._on_clear_clicked()
    tab._commit()
    _wait(tab)

    committed = state.active_data()
    np.testing.assert_allclose(np.asarray(committed), np.asarray(mmapped), atol=1e-5)
