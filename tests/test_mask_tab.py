import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from orbit.masking import apply_mask, apply_mask_3d, compute_mask, triangle_mask  # noqa: E402
from orbitapp.io import is_memmap, load_movie  # noqa: E402
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


def test_mask_tab_method_combo_offers_every_method():
    state = AppState()
    tab = MaskTab(state)
    items = [tab.method_combo.itemText(i) for i in range(tab.method_combo.count())]
    assert items == ["Triangle (auto)", "Otsu (auto)", "Manual threshold", "Percentile"]


def test_mask_tab_params_dialog_only_shows_the_selected_methods_group():
    state = AppState()
    tab = MaskTab(state)

    tab.method_combo.setCurrentText("Triangle (auto)")
    assert not tab.params_dialog.form.isRowVisible(tab.manual_threshold_spin)
    assert not tab.params_dialog.form.isRowVisible(tab.percentile_spin)

    tab.method_combo.setCurrentText("Manual threshold")
    assert tab.params_dialog.form.isRowVisible(tab.manual_threshold_spin)
    assert not tab.params_dialog.form.isRowVisible(tab.percentile_spin)

    tab.method_combo.setCurrentText("Percentile")
    assert not tab.params_dialog.form.isRowVisible(tab.manual_threshold_spin)
    assert tab.params_dialog.form.isRowVisible(tab.percentile_spin)


def test_mask_tab_manual_threshold_apply_and_commit():
    movie = _bright_blob_movie()
    state = AppState()
    state.load("movie.npy", movie)
    tab = MaskTab(state)
    tab.on_data_loaded()

    tab.method_combo.setCurrentText("Manual threshold")
    tab.manual_threshold_spin.setValue(1.0)
    tab._apply()
    _wait(tab)
    tab._commit()
    _wait(tab)

    projection = np.asarray(movie, dtype=np.float64).mean(axis=2)
    expected_mask = compute_mask(projection, "manual", threshold=1.0)
    expected = apply_mask(movie, expected_mask)
    np.testing.assert_allclose(np.asarray(state.active_data()), expected)
    np.testing.assert_array_equal(state.mask, expected_mask)


def test_mask_tab_percentile_apply_and_commit():
    movie = _bright_blob_movie()
    state = AppState()
    state.load("movie.npy", movie)
    tab = MaskTab(state)
    tab.on_data_loaded()

    tab.method_combo.setCurrentText("Percentile")
    tab.percentile_spin.setValue(15.0)
    tab._apply()
    _wait(tab)
    tab._commit()
    _wait(tab)

    projection = np.asarray(movie, dtype=np.float64).mean(axis=2)
    expected_mask = compute_mask(projection, "percentile", percentile=15.0)
    expected = apply_mask(movie, expected_mask)
    np.testing.assert_allclose(np.asarray(state.active_data()), expected)


def test_mask_tab_otsu_apply_and_commit():
    movie = _bright_blob_movie()
    state = AppState()
    state.load("movie.npy", movie)
    tab = MaskTab(state)
    tab.on_data_loaded()

    tab.method_combo.setCurrentText("Otsu (auto)")
    tab._apply()
    _wait(tab)
    tab._commit()
    _wait(tab)

    projection = np.asarray(movie, dtype=np.float64).mean(axis=2)
    expected_mask = compute_mask(projection, "otsu")
    expected = apply_mask(movie, expected_mask)
    np.testing.assert_allclose(np.asarray(state.active_data()), expected)


def test_mask_tab_restore_params_maps_legacy_auto_to_triangle():
    state = AppState()
    tab = MaskTab(state)
    tab.method_combo.setCurrentText("Manual threshold")

    tab.restore_params({"action": "auto"})

    assert tab.method_combo.currentText() == "Triangle (auto)"


def test_mask_tab_restore_params_restores_method_and_value():
    state = AppState()
    tab = MaskTab(state)

    tab.restore_params({"action": "percentile", "percentile": 42.0})

    assert tab.method_combo.currentText() == "Percentile"
    assert tab.percentile_spin.value() == 42.0


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


def test_mask_tab_auto_threshold_commit_persists_the_mask_in_state():
    movie = _bright_blob_movie()
    state = AppState()
    state.load("movie.npy", movie)
    tab = MaskTab(state)
    tab.on_data_loaded()

    assert state.mask is None
    tab._apply()
    _wait(tab)
    tab._commit()
    _wait(tab)

    expected_mask = triangle_mask(np.asarray(movie, dtype=np.float64).mean(axis=2))
    np.testing.assert_array_equal(state.mask, expected_mask)


def test_mask_tab_clear_mask_commit_persists_an_all_true_mask_in_state():
    movie = _bright_blob_movie()
    state = AppState()
    state.load("movie.npy", movie)
    tab = MaskTab(state)
    tab.on_data_loaded()

    tab._on_clear_clicked()
    tab._commit()
    _wait(tab)

    assert state.mask is not None
    assert state.mask.all()


def test_loading_new_data_resets_the_mask():
    movie = _bright_blob_movie()
    state = AppState()
    state.load("movie.npy", movie)
    tab = MaskTab(state)
    tab.on_data_loaded()
    tab._apply()
    _wait(tab)
    tab._commit()
    _wait(tab)
    assert state.mask is not None

    state.load("movie2.npy", movie)
    assert state.mask is None


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


# -- Volumetric (state.volumetric) path -----------------------------------


def _bright_blob_volume(shape=(20, 10, 10, 6), seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    movie = rng.standard_normal(shape).astype(np.float32) * 0.05 + 0.1
    movie[:, 3:8, 3:8, 2:5] += 3.0
    return np.clip(movie, 0, None).astype(np.float32)


def test_volumetric_auto_threshold_apply_and_commit_produces_a_true_4d_array():
    movie = _bright_blob_volume()
    state = AppState()
    state.volumetric = True
    state.load("movie.fits", movie)
    tab = MaskTab(state)
    tab.on_data_loaded()

    tab._apply()
    _wait(tab)
    assert tab.commit_controls.commit_btn.isEnabled()
    tab._commit()
    _wait(tab)

    committed = state.active_data()
    assert not is_memmap(committed)
    assert committed.shape == movie.shape  # true (T, L, W, D), not the depth-projected display movie
    expected_mask = triangle_mask(np.asarray(movie, dtype=np.float64).mean(axis=0))
    np.testing.assert_array_equal(state.mask, expected_mask)
    np.testing.assert_allclose(np.asarray(committed), apply_mask_3d(movie, expected_mask))
    assert "Mask" in state.pipeline[-1]


def test_volumetric_clear_mask_commits_unchanged_movie_and_persists_all_true_mask():
    movie = _bright_blob_volume()
    state = AppState()
    state.volumetric = True
    state.load("movie.fits", movie)
    tab = MaskTab(state)
    tab.on_data_loaded()

    tab._on_clear_clicked()  # synchronous
    tab._commit()
    _wait(tab)

    committed = state.active_data()
    np.testing.assert_allclose(np.asarray(committed), movie)
    assert state.mask is not None
    assert state.mask.all()


def test_volumetric_commit_of_a_memmap_movie_produces_a_4d_fits_memmap(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    from orbitapp.volumetric_io import load_volumetric_movie

    movie = _bright_blob_volume(shape=(20, 8, 8, 4))
    path = tmp_path / "movie.fits"
    fits.PrimaryHDU(data=movie).writeto(path, overwrite=True)
    mmapped = load_volumetric_movie(path, mmap=True)

    state = AppState()
    state.volumetric = True
    state.load(str(path), mmapped)
    tab = MaskTab(state)
    tab.on_data_loaded()
    tab._chunk_frames = 6

    tab._apply()
    _wait(tab)
    tab._commit()
    _wait(tab)

    committed = state.active_data()
    assert is_memmap(committed)
    assert committed.shape == movie.shape
    expected_mask = triangle_mask(np.asarray(mmapped, dtype=np.float64).mean(axis=0))
    expected = apply_mask_3d(np.asarray(mmapped, dtype=np.float32), expected_mask)
    np.testing.assert_allclose(np.asarray(committed), expected, atol=1e-5)
    np.testing.assert_array_equal(state.mask, expected_mask)


def test_turning_volumetric_off_leaves_the_2d_masking_path_unaffected():
    state = AppState()
    movie = _bright_blob_movie()
    state.load("movie.tif", movie)
    tab = MaskTab(state)
    tab.on_data_loaded()

    tab._apply()
    _wait(tab)
    tab._commit()

    assert not is_memmap(state.active_data())
    assert state.active_data().shape == movie.shape
    assert "Mask" in state.pipeline[-1]
