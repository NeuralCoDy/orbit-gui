import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from orbit.denoising import denoise_gaussian, denoise_median, denoise_wavelet_space  # noqa: E402
from orbit.pca_denoise import pca_denoise  # noqa: E402
from orbitapp.io import is_memmap, load_movie  # noqa: E402
from orbitapp.state import AppState  # noqa: E402
from orbitapp.tabs.denoising_tab import DenoisingTab  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def _wait(tab):
    if tab.worker is not None:
        tab.worker.wait(15000)
    for _ in range(50):
        QApplication.processEvents()


def _memmapped_movie(tmp_path, height=12, width=12, n_frames=47, seed=0):
    rng = np.random.default_rng(seed)
    movie = rng.standard_normal((height, width, n_frames)).astype(np.float32)
    path = tmp_path / "movie.npy"
    np.save(path, movie)
    return path, load_movie(path, mmap=True)


def test_gaussian_commit_of_memmap_movie_matches_whole_movie_result_exactly(tmp_path):
    path, movie = _memmapped_movie(tmp_path)
    state = AppState()
    state.load(str(path), movie)
    tab = DenoisingTab(state)
    tab.on_data_loaded()

    tab.method_combo.setCurrentText("Gaussian Filter")
    tab.gaussian_spatial_spin.setValue(1.5)
    tab.gaussian_temporal_spin.setValue(2.0)  # nonzero temporal component needs the margin logic
    tab._chunk_frames = 9
    tab._apply()
    _wait(tab)
    tab._commit()
    _wait(tab)

    committed = state.active_data()
    assert is_memmap(committed)
    expected = denoise_gaussian(np.asarray(movie, dtype=np.float32), spatial_sigma=1.5, temporal_sigma=2.0)
    np.testing.assert_allclose(np.asarray(committed), expected, atol=1e-5)


def test_median_commit_of_memmap_movie_matches_whole_movie_result_exactly(tmp_path):
    path, movie = _memmapped_movie(tmp_path)
    state = AppState()
    state.load(str(path), movie)
    tab = DenoisingTab(state)
    tab.on_data_loaded()

    tab.method_combo.setCurrentText("Median Filter")
    tab.median_space_spin.setValue(3)
    tab.median_time_spin.setValue(5)  # nonzero temporal window needs the margin logic
    tab._chunk_frames = 8
    tab._apply()
    _wait(tab)
    tab._commit()
    _wait(tab)

    committed = state.active_data()
    expected = denoise_median(np.asarray(movie, dtype=np.float32), space_window=3, time_window=5)
    np.testing.assert_allclose(np.asarray(committed), expected, atol=1e-5)


def test_wavelet_space_commit_of_memmap_movie_matches_whole_movie_result(tmp_path):
    path, movie = _memmapped_movie(tmp_path)
    state = AppState()
    state.load(str(path), movie)
    tab = DenoisingTab(state)
    tab.on_data_loaded()

    tab.method_combo.setCurrentText("Wavelet - Spatial (per frame)")
    tab._chunk_frames = 10
    tab._apply()
    _wait(tab)
    tab._commit()
    _wait(tab)

    committed = state.active_data()
    expected = denoise_wavelet_space(np.asarray(movie, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(committed), expected, atol=1e-4)


def test_wavelet_time_commit_of_memmap_movie_fails_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr("orbitapp.tabs.stage_tab.QMessageBox.critical", lambda *a, **k: None)

    path, movie = _memmapped_movie(tmp_path)
    state = AppState()
    state.load(str(path), movie)
    tab = DenoisingTab(state)
    tab.on_data_loaded()

    tab.method_combo.setCurrentText("Wavelet - Temporal (per pixel)")
    tab._apply()
    _wait(tab)

    tab._commit()
    _wait(tab)

    assert state.active_data() is movie  # nothing got committed
    assert tab.commit_controls.commit_btn.isEnabled()  # candidate still there to retry with a different method


def test_gaussian_with_zero_temporal_sigma_needs_no_margin_and_still_matches(tmp_path):
    # spatial-only denoising (the tab's own default) shouldn't accidentally
    # read/write outside its own chunk.
    path, movie = _memmapped_movie(tmp_path)
    state = AppState()
    state.load(str(path), movie)
    tab = DenoisingTab(state)
    tab.on_data_loaded()

    tab.method_combo.setCurrentText("Gaussian Filter")
    tab.gaussian_spatial_spin.setValue(1.0)
    tab.gaussian_temporal_spin.setValue(0.0)
    assert tab._temporal_margin("gaussian", {"temporal_sigma": 0.0}) == 0
    tab._chunk_frames = 11
    tab._apply()
    _wait(tab)
    tab._commit()
    _wait(tab)

    committed = state.active_data()
    expected = denoise_gaussian(np.asarray(movie, dtype=np.float32), spatial_sigma=1.0, temporal_sigma=0.0)
    np.testing.assert_allclose(np.asarray(committed), expected, atol=1e-5)


def test_pca_denoising_apply_and_commit_on_a_non_memmap_movie():
    rng = np.random.default_rng(0)
    height, width, n_frames = 15, 15, 60
    movie = rng.standard_normal((height, width, n_frames)).astype(np.float32)
    state = AppState()
    state.load("movie.npy", movie)
    tab = DenoisingTab(state)
    tab.on_data_loaded()

    tab.method_combo.setCurrentText("PCA Denoising")
    tab.pca_n_components_spin.setValue(3)
    tab._apply()
    _wait(tab)
    tab._commit()
    _wait(tab)

    committed = state.active_data()
    expected = pca_denoise(movie, n_components=3, block_size=(250, 250), block_frames=5000)
    np.testing.assert_allclose(np.asarray(committed), expected, atol=1e-4)


def test_pca_denoising_commit_of_memmap_movie_fails_cleanly(tmp_path, monkeypatch):
    # PCA Denoising's blocks are a global, blended tiling of the whole
    # movie -- it can't be committed chunk-by-chunk against a memmap
    # movie without changing the result (see _chunked_commit), so it
    # should fail the same clean way wavelet_time already does rather
    # than silently commit something inconsistent with its own preview.
    monkeypatch.setattr("orbitapp.tabs.stage_tab.QMessageBox.critical", lambda *a, **k: None)

    path, movie = _memmapped_movie(tmp_path)
    state = AppState()
    state.load(str(path), movie)
    tab = DenoisingTab(state)
    tab.on_data_loaded()

    tab.method_combo.setCurrentText("PCA Denoising")
    tab._apply()
    _wait(tab)

    tab._commit()
    _wait(tab)

    assert state.active_data() is movie  # nothing got committed
    assert tab.commit_controls.commit_btn.isEnabled()  # candidate still there to retry with a different method
