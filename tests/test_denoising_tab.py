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


@pytest.mark.parametrize(
    "method_label, expected_extra_keys",
    [
        ("Wavelet - Temporal (per pixel)", {"wavelet", "level", "threshold_method"}),
        ("Wavelet - Spatial (per frame)", {"wavelet", "level", "threshold_method"}),
        ("Gaussian Filter", {"spatial_sigma", "temporal_sigma"}),
        ("Median Filter", {"space_window", "time_window"}),
        ("PCA Denoising", {
            "pca_n_components", "pca_block_size", "pca_block_frames",
            "pca_spatial_overlap", "pca_temporal_overlap",
        }),
    ],
)
def test_current_fingerprint_only_includes_the_selected_algorithms_params(method_label, expected_extra_keys):
    state = AppState()
    tab = DenoisingTab(state)
    tab.method_combo.setCurrentText(method_label)

    params = tab._current_fingerprint()

    assert set(params) == {"algorithm"} | expected_extra_keys


def test_current_fingerprint_switching_algorithms_drops_the_other_algorithms_fields():
    state = AppState()
    tab = DenoisingTab(state)
    tab.method_combo.setCurrentText("PCA Denoising")
    assert "pca_n_components" in tab._current_fingerprint()

    tab.method_combo.setCurrentText("Gaussian Filter")

    params = tab._current_fingerprint()
    assert "pca_n_components" not in params
    assert "wavelet" not in params
    assert set(params) == {"algorithm", "spatial_sigma", "temporal_sigma"}


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


def test_ljung_box_metric_appears_after_apply():
    rng = np.random.default_rng(0)
    movie = rng.standard_normal((10, 10, 100)).astype(np.float32)
    state = AppState()
    state.load("movie.npy", movie)
    tab = DenoisingTab(state)
    tab.on_data_loaded()

    tab.method_combo.setCurrentText("Gaussian Filter")
    tab.gaussian_spatial_spin.setValue(1.5)
    tab._apply()
    _wait(tab)

    metrics = tab._extract_metrics(tab._pending_result)
    assert 0 <= metrics["ljung_box_failed"] <= metrics["ljung_box_total"] == 100
    assert "Ljung-Box" in tab.metrics_label.text()


def test_residual_movie_is_playable_after_apply():
    rng = np.random.default_rng(0)
    movie = rng.standard_normal((10, 10, 100)).astype(np.float32)
    state = AppState()
    state.load("movie.npy", movie)
    tab = DenoisingTab(state)
    tab.on_data_loaded()

    tab.method_combo.setCurrentText("Gaussian Filter")
    tab.gaussian_spatial_spin.setValue(1.5)
    tab._apply()
    _wait(tab)

    denoised = tab._pending_result["denoised"]
    expected_residual = movie.astype(np.float64) - denoised.astype(np.float64)
    np.testing.assert_allclose(tab.panel._movies["residual"], expected_residual)

    tab.panel._play_movie("residual")
    assert tab.panel._players["residual"] is not None
    assert tab.panel._players["residual"].windowTitle() == "Movie Player - Play Residual Movie"


def test_ljung_box_n_exclude_matches_each_algorithms_own_reach():
    from orbitapp.tabs.denoising_tab import _ljung_box_n_exclude

    assert _ljung_box_n_exclude("gaussian", {"temporal_sigma": 2.0}) == 8  # ceil(4*2.0), same as _temporal_margin
    assert _ljung_box_n_exclude("gaussian", {"temporal_sigma": 0.0}) == 0  # spatial-only: no temporal reach at all
    assert _ljung_box_n_exclude("median", {"time_window": 5}) == 5
    assert _ljung_box_n_exclude("wavelet_time", {"level": 4}) == 16  # 2**4
    assert _ljung_box_n_exclude("wavelet_space", {}) == 0
    assert _ljung_box_n_exclude("pca", {}) == 0


def test_ljung_box_metric_substantially_improves_on_pure_noise_with_temporal_smoothing():
    # Without excluding the smoothing filter's own reach, the residual
    # of essentially ANY temporal filter fails the whiteness test almost
    # everywhere even for pure noise input with no real signal removed
    # -- this pins that the per-algorithm exclusion (_ljung_box_n_exclude)
    # brings that down substantially, not that it's eliminated (some
    # residual bias is inherent to smoothing/rank filters -- see that
    # function's own docstring).
    rng = np.random.default_rng(0)
    height, width, n_frames = 20, 20, 500
    movie = rng.standard_normal((height, width, n_frames)).astype(np.float32)
    state = AppState()
    state.load("movie.npy", movie)
    tab = DenoisingTab(state)
    tab.on_data_loaded()

    tab.method_combo.setCurrentText("Gaussian Filter")
    tab.gaussian_spatial_spin.setValue(0.0)
    tab.gaussian_temporal_spin.setValue(2.0)
    tab._apply()
    _wait(tab)

    metrics = tab._extract_metrics(tab._pending_result)
    fail_fraction = metrics["ljung_box_failed"] / metrics["ljung_box_total"]
    assert fail_fraction < 0.5  # far below what n_exclude=0 gives here (measured ~100%)


# -- Volumetric (state.volumetric) path ----------------------------------


def _volumetric_movie(shape=(6, 10, 10, 5), seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32) * 3 + 50


def _memmapped_volumetric_movie(tmp_path, shape=(20, 8, 8, 4), seed=0):
    fits = pytest.importorskip("astropy.io.fits")
    from orbitapp.volumetric_io import load_volumetric_movie

    movie = _volumetric_movie(shape, seed)
    path = tmp_path / "movie.fits"
    fits.PrimaryHDU(data=movie).writeto(path, overwrite=True)
    return str(path), load_volumetric_movie(path, mmap=True)


def test_on_modality_changed_restricts_method_combo_to_gaussian_or_median_when_volumetric():
    state = AppState()
    tab = DenoisingTab(state)
    tab.method_combo.setCurrentText("PCA Denoising")

    state.volumetric = True
    tab.on_modality_changed()

    assert tab.method_combo.currentText() == "Gaussian Filter"
    model = tab.method_combo.model()
    assert not model.item(0).isEnabled()  # Wavelet - Temporal
    assert not model.item(1).isEnabled()  # Wavelet - Spatial
    assert model.item(2).isEnabled()  # Gaussian
    assert model.item(3).isEnabled()  # Median
    assert not model.item(4).isEnabled()  # PCA

    # Median stays selected across the toggle -- it already has a 3D implementation.
    tab.method_combo.setCurrentText("Median Filter")
    state.volumetric = False
    tab.on_modality_changed()
    assert tab.method_combo.currentText() == "Median Filter"
    assert model.item(0).isEnabled()
    assert model.item(4).isEnabled()


def test_volumetric_gaussian_apply_and_commit_produces_a_true_4d_array(tmp_path):
    pytest.importorskip("astropy")
    state = AppState()
    state.volumetric = True
    movie = _volumetric_movie()
    state.load(str(tmp_path / "movie.fits"), movie)
    tab = DenoisingTab(state)
    tab.on_data_loaded()

    tab.method_combo.setCurrentText("Gaussian Filter")
    tab.gaussian_spatial_spin.setValue(1.0)
    tab.gaussian_temporal_spin.setValue(0.5)
    tab._apply()
    _wait(tab)
    assert tab.commit_controls.commit_btn.isEnabled()

    tab._commit()
    _wait(tab)

    committed = state.active_data()
    assert not is_memmap(committed)
    assert committed.shape == movie.shape  # true (T, L, W, D), not the depth-projected display movie
    assert np.all(np.isfinite(committed))
    assert state.pipeline == ["Load", "Gaussian Denoising"]


def test_volumetric_median_commit_of_a_memmap_movie_produces_a_4d_fits_memmap(tmp_path):
    path, movie = _memmapped_volumetric_movie(tmp_path)
    state = AppState()
    state.volumetric = True
    state.load(path, movie)
    tab = DenoisingTab(state)
    tab.on_data_loaded()

    tab.method_combo.setCurrentText("Median Filter")
    tab.median_space_spin.setValue(3)
    tab.median_time_spin.setValue(3)
    tab._chunk_frames = 6  # small, so a 20-timepoint movie spans multiple chunks
    tab._apply()
    _wait(tab)

    tab._commit()
    _wait(tab)

    committed = state.active_data()
    assert is_memmap(committed)
    assert committed.shape == movie.shape  # the WHOLE volumetric movie, not just the preview
    assert np.all(np.isfinite(np.asarray(committed)))


def test_turning_volumetric_off_leaves_the_2d_denoising_path_unaffected():
    # A regression check for the core constraint this whole feature was
    # built under: with the toggle off, Apply/Commit behave exactly as
    # they did before any volumetric code existed.
    state = AppState()
    movie = np.random.default_rng(2).standard_normal((10, 10, 12)).astype(np.float32)
    state.load("movie.tif", movie)
    tab = DenoisingTab(state)
    tab.on_data_loaded()

    tab.method_combo.setCurrentText("Gaussian Filter")
    tab._apply()
    _wait(tab)
    tab._commit()

    assert not is_memmap(state.active_data())
    assert state.active_data().shape == movie.shape
    assert state.pipeline == ["Load", "Gaussian Denoising"]
