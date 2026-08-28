import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from orbitapp.io import is_memmap, load_movie  # noqa: E402
from orbitapp.state import AppState  # noqa: E402
from orbitapp.tabs.motion_correction_tab import MotionCorrectionTab  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def _wait(tab):
    if tab.worker is not None:
        tab.worker.wait(15000)
    for _ in range(50):
        QApplication.processEvents()


def _memmapped_movie(tmp_path, height=16, width=16, n_frames=40, seed=0):
    rng = np.random.default_rng(seed)
    movie = gaussian_filter(rng.standard_normal((height, width, n_frames)), (2, 2, 0)).astype(np.float32) * 10 + 100
    path = tmp_path / "movie.npy"
    np.save(path, movie)
    return path, load_movie(path, mmap=True)


def test_rigid_commit_of_a_memmap_movie_produces_a_fits_memmap_covering_the_whole_movie(tmp_path):
    path, movie = _memmapped_movie(tmp_path, n_frames=30)
    state = AppState()
    state.load(str(path), movie)
    tab = MotionCorrectionTab(state)
    tab.on_data_loaded()

    tab.method_combo.setCurrentText("Rigid")
    tab.max_shift_spin.setValue(5.0)
    tab.upsample_spin.setValue(10)
    tab._chunk_frames = 8  # small, so a 30-frame movie spans multiple chunks
    tab._apply()
    _wait(tab)
    assert tab.commit_controls.commit_btn.isEnabled()

    tab._commit()
    _wait(tab)

    committed = state.active_data()
    assert is_memmap(committed)
    assert committed.shape == movie.shape  # the WHOLE movie, not just the 5000-frame preview
    assert np.all(np.isfinite(np.asarray(committed)))
    assert state.pipeline == ["Load", "Rigid"]


def test_patch_based_commit_of_a_memmap_movie_also_produces_a_fits_memmap(tmp_path):
    path, movie = _memmapped_movie(tmp_path, height=24, width=24, n_frames=20)
    state = AppState()
    state.load(str(path), movie)
    tab = MotionCorrectionTab(state)
    tab.on_data_loaded()

    tab.method_combo.setCurrentText("Patch-based (non-rigid)")
    tab.max_shift_spin.setValue(5.0)
    tab.grid_size_spin.setValue(12)
    tab.upsample_spin.setValue(10)
    tab._chunk_frames = 7
    tab._apply()
    _wait(tab)

    tab._commit()
    _wait(tab)

    committed = state.active_data()
    assert is_memmap(committed)
    assert committed.shape == movie.shape


def test_apply_on_a_memmap_movie_only_previews_the_first_5000_frames(tmp_path, monkeypatch):
    path, movie = _memmapped_movie(tmp_path, n_frames=20)
    state = AppState()
    state.load(str(path), movie)
    tab = MotionCorrectionTab(state)
    tab.on_data_loaded()

    seen = {}
    real_start = tab._start_worker

    def _spy(m):
        seen["shape"] = m.shape
        real_start(m)

    monkeypatch.setattr(tab, "_start_worker", _spy)
    tab._apply()
    _wait(tab)

    assert seen["shape"][-1] == 20  # movie is shorter than 5000, so preview == whole movie
    assert tab._input_movie.shape[-1] == 20  # commit-time source stays the real (memmap) movie


def test_patchwarp_commit_of_a_memmap_movie_fails_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr("orbitapp.tabs.stage_tab.QMessageBox.critical", lambda *a, **k: None)

    path, movie = _memmapped_movie(tmp_path, n_frames=15)
    state = AppState()
    state.load(str(path), movie)
    tab = MotionCorrectionTab(state)
    tab.on_data_loaded()

    tab.method_combo.setCurrentText("PatchWarp (piecewise-affine)")
    tab._apply()
    _wait(tab)

    tab._commit()
    _wait(tab)

    assert state.active_data() is movie  # nothing got committed
    assert tab.commit_controls.commit_btn.isEnabled()  # candidate still there to retry with a different method


def test_non_memmap_movie_commit_is_unaffected_direct_promotion(tmp_path):
    state = AppState()
    movie = gaussian_filter(np.random.default_rng(1).standard_normal((10, 10, 15)), (1, 1, 0))
    state.load("movie.tif", movie)
    tab = MotionCorrectionTab(state)
    tab.on_data_loaded()

    tab.method_combo.setCurrentText("Rigid")
    tab._apply()
    _wait(tab)
    tab._commit()

    assert not is_memmap(state.active_data())
    assert state.active_data().shape == movie.shape


def test_current_fingerprint_only_includes_rigid_relevant_params():
    state = AppState()
    tab = MotionCorrectionTab(state)
    tab.method_combo.setCurrentText("Rigid")

    params = tab._current_fingerprint()

    assert set(params) == {"method", "max_shift", "n_iter", "n_components", "upsample_factor"}


def test_current_fingerprint_only_includes_patch_based_relevant_params():
    state = AppState()
    tab = MotionCorrectionTab(state)
    tab.method_combo.setCurrentText("Patch-based (non-rigid)")

    params = tab._current_fingerprint()

    assert set(params) == {"method", "max_shift", "n_iter", "n_components", "upsample_factor", "grid_size"}


def test_current_fingerprint_only_includes_patchwarp_relevant_params():
    state = AppState()
    tab = MotionCorrectionTab(state)
    tab.method_combo.setCurrentText("PatchWarp (piecewise-affine)")

    params = tab._current_fingerprint()

    assert set(params) == {
        "method", "max_shift", "n_iter", "n_components",
        "patchwarp_grid", "overlap_frac", "ecc_iterations", "pyramid_levels",
    }


def test_current_fingerprint_switching_methods_drops_the_other_methods_fields():
    state = AppState()
    tab = MotionCorrectionTab(state)
    tab.method_combo.setCurrentText("PatchWarp (piecewise-affine)")
    assert "patchwarp_grid" in tab._current_fingerprint()

    tab.method_combo.setCurrentText("Rigid")

    assert "patchwarp_grid" not in tab._current_fingerprint()
    assert "grid_size" not in tab._current_fingerprint()


# -- Volumetric (state.volumetric) path ---------------------------------


def _volumetric_movie(shape=(6, 10, 10, 5), seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.standard_normal(shape), (0, 2, 2, 0)).astype(np.float32) * 10 + 100


def _memmapped_volumetric_movie(tmp_path, shape=(20, 8, 8, 4), seed=0):
    fits = pytest.importorskip("astropy.io.fits")
    from orbitapp.volumetric_io import load_volumetric_movie

    movie = _volumetric_movie(shape, seed)
    path = tmp_path / "movie.fits"
    fits.PrimaryHDU(data=movie).writeto(path, overwrite=True)
    return str(path), load_volumetric_movie(path, mmap=True)


def test_on_modality_changed_restricts_method_combo_to_rigid_when_volumetric():
    state = AppState()
    tab = MotionCorrectionTab(state)
    tab.method_combo.setCurrentText("Patch-based (non-rigid)")

    state.volumetric = True
    tab.on_modality_changed()

    assert tab.method_combo.currentText() == "Rigid"
    model = tab.method_combo.model()
    assert not model.item(1).isEnabled()
    assert not model.item(2).isEnabled()

    state.volumetric = False
    tab.on_modality_changed()
    assert model.item(1).isEnabled()
    assert model.item(2).isEnabled()


def test_volumetric_apply_and_commit_produces_a_true_4d_array(tmp_path):
    pytest.importorskip("astropy")
    state = AppState()
    state.volumetric = True
    movie = _volumetric_movie()
    state.load(str(tmp_path / "movie.fits"), movie)
    tab = MotionCorrectionTab(state)
    tab.on_data_loaded()

    tab.max_shift_spin.setValue(3.0)
    tab.upsample_spin.setValue(10)
    tab._apply()
    _wait(tab)
    assert tab.commit_controls.commit_btn.isEnabled()

    tab._commit()
    _wait(tab)

    committed = state.active_data()
    assert not is_memmap(committed)
    assert committed.shape == movie.shape  # true (T, L, W, D), not the depth-projected display movie
    assert np.all(np.isfinite(committed))
    assert state.pipeline == ["Load", "Rigid (3D)"]


def test_volumetric_commit_of_a_memmap_movie_produces_a_4d_fits_memmap(tmp_path):
    path, movie = _memmapped_volumetric_movie(tmp_path)
    state = AppState()
    state.volumetric = True
    state.load(path, movie)
    tab = MotionCorrectionTab(state)
    tab.on_data_loaded()

    tab.max_shift_spin.setValue(3.0)
    tab.upsample_spin.setValue(10)
    tab._chunk_frames = 6  # small, so a 20-timepoint movie spans multiple chunks
    tab._apply()
    _wait(tab)

    tab._commit()
    _wait(tab)

    committed = state.active_data()
    assert is_memmap(committed)
    assert committed.shape == movie.shape  # the WHOLE volumetric movie, not just the preview
    assert np.all(np.isfinite(np.asarray(committed)))


def test_turning_volumetric_off_leaves_the_2d_path_unaffected(tmp_path):
    # A regression check for the core constraint this whole feature was
    # built under: with the toggle off, Apply/Commit behave exactly as
    # they did before any volumetric code existed.
    state = AppState()
    movie = gaussian_filter(np.random.default_rng(2).standard_normal((10, 10, 12)), (1, 1, 0))
    state.load("movie.tif", movie)
    tab = MotionCorrectionTab(state)
    tab.on_data_loaded()

    tab.method_combo.setCurrentText("Rigid")
    tab._apply()
    _wait(tab)
    tab._commit()

    assert not is_memmap(state.active_data())
    assert state.active_data().shape == movie.shape
    assert state.pipeline == ["Load", "Rigid"]
