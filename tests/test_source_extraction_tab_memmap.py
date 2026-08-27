"""Memmap-specific SourceExtractionTab coverage -- separate from
test_source_extraction_tab.py since these need real memmapped movies on
disk rather than plain in-RAM arrays.
"""

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from orbit._masks import masked_mean_trace  # noqa: E402
from orbitapp.io import load_movie  # noqa: E402
from orbitapp.state import AppState  # noqa: E402
from orbitapp.tabs.source_extraction_tab import SourceExtractionTab  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def _wait(tab, timeout_ms=15000):
    if tab.worker is not None:
        tab.worker.wait(timeout_ms)
    for _ in range(50):
        QApplication.processEvents()


def _memmapped_movie(tmp_path, height=30, width=30, n_frames=60, seed=0):
    rng = np.random.default_rng(seed)
    movie = rng.standard_normal((height, width, n_frames)).astype(np.float32) * 0.1 + 5.0
    movie[5:10, 5:10, :] += rng.standard_normal(n_frames)
    movie[20:25, 20:25, :] += rng.standard_normal(n_frames)
    path = tmp_path / "movie.npy"
    np.save(path, movie)
    return path, load_movie(path, mmap=True)


def test_correlation_image_uses_a_bounded_preview_for_a_memmap_movie(tmp_path):
    path, movie = _memmapped_movie(tmp_path, n_frames=8000)
    state = AppState()
    state.load(str(path), movie)
    tab = SourceExtractionTab(state)

    tab.on_data_loaded()
    _wait(tab)

    assert tab._corr_image is not None
    assert tab._corr_image.shape == (30, 30)  # spatial shape unaffected -- only T was capped internally


def test_whole_fov_cnmf_is_refused_for_a_memmap_movie_without_patch_based_checked(tmp_path, monkeypatch):
    warned = []
    monkeypatch.setattr(
        "orbitapp.tabs.source_extraction_tab.QMessageBox.warning", lambda *a, **k: warned.append(1)
    )
    path, movie = _memmapped_movie(tmp_path)
    state = AppState()
    state.load(str(path), movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait(tab)

    assert not tab.cnmf_patch_check.isChecked()
    tab._on_run_cnmf_clicked()

    assert warned == [1]
    assert tab._candidates == []


def test_patch_based_cnmf_runs_on_a_memmap_movie_and_commit_reextracts_traces_from_the_full_movie(tmp_path):
    path, movie = _memmapped_movie(tmp_path)
    state = AppState()
    state.load(str(path), movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait(tab)

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("CNMF"))
    tab.cnmf_patch_check.setChecked(True)
    tab.cnmf_patch_size_spin.setValue(18)
    tab.cnmf_patch_overlap_spin.setValue(6)
    tab.cnmf_components_per_patch_spin.setValue(3)
    tab.cnmf_search_radius_spin.setValue(8)
    tab._on_run_cnmf_clicked()
    _wait(tab)

    assert len(tab._candidates) > 0
    for roi in tab._candidates:
        roi.status = "accepted"

    tab._commit()
    _wait(tab)

    assert len(state.rois) > 0
    for roi in state.rois:
        expected_trace = masked_mean_trace(np.asarray(movie), roi.mask)
        np.testing.assert_allclose(roi.trace, expected_trace)


def test_pca_ica_on_a_memmap_movie_uses_a_bounded_preview_and_commit_reextracts_traces(tmp_path):
    path, movie = _memmapped_movie(tmp_path, n_frames=8000)
    state = AppState()
    state.load(str(path), movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait(tab)

    tab.n_pca_components_spin.setValue(10)
    tab.n_ica_components_spin.setValue(6)
    tab._on_run_pca_ica_clicked()
    _wait(tab, timeout_ms=30000)

    assert len(tab._candidates) >= 1
    for roi in tab._candidates:
        roi.status = "accepted"

    tab._commit()
    _wait(tab)

    assert len(state.rois) >= 1
    for roi in state.rois:
        expected_trace = masked_mean_trace(np.asarray(movie), roi.mask)
        np.testing.assert_allclose(roi.trace, expected_trace)


def test_correlation_based_rois_are_unaffected_by_memmap_reextraction(tmp_path):
    # Correlation-based traces are already computed from the true full
    # movie (never preview-capped) -- commit-time re-extraction should
    # reproduce the identical value, not change it.
    path, movie = _memmapped_movie(tmp_path)
    state = AppState()
    state.load(str(path), movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait(tab)

    tab._grow_from_seeds(movie, [(7, 7)])
    _wait(tab)
    assert len(tab._candidates) == 1
    original_trace = tab._candidates[0].trace.copy()
    tab._candidates[0].status = "accepted"

    tab._commit()
    _wait(tab)

    assert len(state.rois) == 1
    np.testing.assert_allclose(state.rois[0].trace, original_trace)


def test_whole_fov_graft_is_refused_for_a_memmap_movie_without_patch_based_checked(tmp_path, monkeypatch):
    warned = []
    monkeypatch.setattr(
        "orbitapp.tabs.source_extraction_tab.QMessageBox.warning", lambda *a, **k: warned.append(1)
    )
    path, movie = _memmapped_movie(tmp_path)
    state = AppState()
    state.load(str(path), movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait(tab)

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("GraFT"))
    assert not tab.graft_patch_check.isChecked()
    tab._on_run_graft_clicked()

    assert warned == [1]
    assert tab._candidates == []


def test_patch_based_graft_runs_on_a_memmap_movie_and_commit_reextracts_traces_from_the_full_movie(tmp_path):
    path, movie = _memmapped_movie(tmp_path, n_frames=150)  # GraFT needs more frames to converge reliably
    state = AppState()
    state.load(str(path), movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait(tab)

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("GraFT"))
    tab.graft_patch_check.setChecked(True)
    tab.graft_patch_size_spin.setValue(18)
    tab.graft_patch_overlap_spin.setValue(6)
    tab.graft_n_dict_per_patch_spin.setValue(5)
    tab._on_run_graft_clicked()
    _wait(tab, timeout_ms=30000)

    assert len(tab._candidates) > 0
    for roi in tab._candidates:
        roi.status = "accepted"

    tab._commit()
    _wait(tab)

    assert len(state.rois) > 0
    for roi in state.rois:
        expected_trace = masked_mean_trace(np.asarray(movie), roi.mask)
        np.testing.assert_allclose(roi.trace, expected_trace)


def _memmapped_volumetric_movie(tmp_path, shape=(20, 8, 8, 4), seed=0):
    fits = pytest.importorskip("astropy.io.fits")
    from orbitapp.volumetric_io import load_volumetric_movie

    rng = np.random.default_rng(seed)
    movie = rng.standard_normal(shape).astype(np.float32) * 0.1 + 1.0
    length, width, depth = shape[1:]
    movie[:, 0 : length // 2, 0 : width // 2, :] += 3.0
    movie[:, length // 2 :, width // 2 :, :] += 3.0
    path = tmp_path / "movie.fits"
    fits.PrimaryHDU(data=movie).writeto(path, overwrite=True)
    return str(path), load_volumetric_movie(path, mmap=True)


def test_whole_volume_graft_is_refused_for_a_memmap_movie_without_patch_based_checked(tmp_path, monkeypatch):
    warned = []
    monkeypatch.setattr(
        "orbitapp.tabs.source_extraction_tab.QMessageBox.warning", lambda *a, **k: warned.append(1)
    )
    path, movie = _memmapped_volumetric_movie(tmp_path)
    state = AppState()
    state.volumetric = True
    state.load(path, movie)
    state.mask = np.zeros(movie.shape[1:], dtype=bool)
    state.mask[0:4, 0:4, :] = True
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()

    assert not tab.graft_patch_check.isChecked()
    tab._on_run_graft_clicked()

    assert warned == [1]
    assert tab._candidates == []


def test_patch_based_graft_volumetric_runs_on_a_memmap_movie_and_commit_reextracts_traces(tmp_path):
    from orbit.roi_extraction_graft_3d import _masked_mean_trace_3d

    path, movie = _memmapped_volumetric_movie(tmp_path)
    state = AppState()
    state.volumetric = True
    state.load(path, movie)
    mask = np.zeros(movie.shape[1:], dtype=bool)
    mask[0:4, 0:4, :] = True
    mask[4:8, 4:8, :] = True
    state.mask = mask
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()

    tab.graft_patch_check.setChecked(True)
    tab.graft_patch_size_spin.setValue(4)
    tab.graft_patch_overlap_spin.setValue(1)
    tab.graft_n_dict_per_patch_spin.setValue(3)
    tab._on_run_graft_clicked()
    _wait(tab, timeout_ms=30000)

    assert len(tab._candidates) > 0
    for roi in tab._candidates:
        roi.status = "accepted"

    tab._commit()
    _wait(tab)

    assert len(state.rois) > 0
    for roi in state.rois:
        expected_trace = _masked_mean_trace_3d(np.asarray(movie), roi.mask)
        np.testing.assert_allclose(roi.trace, expected_trace)


def test_non_memmap_movie_commit_skips_the_reextraction_worker(tmp_path):
    rng = np.random.default_rng(2)
    movie = rng.standard_normal((30, 30, 60)).astype(np.float64) * 0.1 + 5.0
    state = AppState()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait(tab)

    tab._grow_from_seeds(movie, [(7, 7)])
    _wait(tab)
    tab._candidates[0].status = "accepted"

    tab._commit()  # must commit synchronously -- no worker involved

    assert len(state.rois) == 1
