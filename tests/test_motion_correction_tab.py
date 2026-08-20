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
