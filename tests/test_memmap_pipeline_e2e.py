"""End-to-end verification of the memmap pipeline: Load (mmap=True) ->
Motion Correction -> Denoising -> Normalization -> Source Extraction,
each stage's Commit chaining into the next as a FITS-backed memmap,
exactly the workflow a user would drive through the GUI for a movie too
large to hold fully in RAM.
"""

import numpy as np
import pytest

pytest.importorskip("PySide6")
tifffile = pytest.importorskip("tifffile")

from PySide6.QtWidgets import QApplication  # noqa: E402

from orbitapp.app import MainWindow  # noqa: E402
from orbitapp.io import is_memmap  # noqa: E402

_HEIGHT, _WIDTH, _N_FRAMES = 40, 40, 6000  # > 5000, so the preview cap is genuinely exercised


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def _wait(tab, timeout_ms=30000):
    if tab.worker is not None:
        tab.worker.wait(timeout_ms)
    for _ in range(50):
        QApplication.processEvents()


def _write_synthetic_movie(path):
    rng = np.random.default_rng(0)
    movie = rng.standard_normal((_HEIGHT, _WIDTH, _N_FRAMES)).astype(np.float32) * 0.1 + 5.0
    movie[10:16, 10:16, :] += 3 * np.clip(rng.standard_normal(_N_FRAMES), 0, None)
    movie[25:31, 25:31, :] += 3 * np.clip(rng.standard_normal(_N_FRAMES), 0, None)
    tifffile.imwrite(path, np.moveaxis(movie, -1, 0).astype("float32"))


def test_full_memmap_pipeline_load_through_source_extraction(tmp_path):
    movie_path = tmp_path / "movie.tif"
    _write_synthetic_movie(movie_path)

    win = MainWindow()
    win.load_tab.mmap_check.setChecked(True)
    win.load_tab._load(str(movie_path))
    _wait(win.load_tab)

    assert is_memmap(win.state.original_data)
    assert win.state.original_data.shape[-1] == _N_FRAMES
    assert win.load_tab.movie_view.state.movie.shape[-1] == 5000  # viewer capped to the preview

    # --- Motion Correction ---
    mc = win.motion_correction_tab
    mc.on_data_loaded()
    mc.method_combo.setCurrentText("Rigid")
    mc.upsample_spin.setValue(5)
    mc._chunk_frames = 1500
    mc._apply()
    _wait(mc)
    assert mc._pending_result["registered"].shape[-1] <= 5000  # Apply stayed bounded to the preview
    mc._commit()
    _wait(mc)

    after_mc = win.state.active_data()
    assert is_memmap(after_mc)
    assert after_mc.shape == (_HEIGHT, _WIDTH, _N_FRAMES)  # Commit covered the WHOLE movie
    assert np.all(np.isfinite(np.asarray(after_mc[:, :, :50])))  # spot-check without reading everything

    # --- Denoising (chained onto Motion Correction's committed output) ---
    dn = win.denoising_tab
    dn.on_data_loaded()
    dn.method_combo.setCurrentText("Gaussian Filter")
    dn.gaussian_spatial_spin.setValue(1.0)
    dn._chunk_frames = 1500
    dn._apply()
    _wait(dn)
    dn._commit()
    _wait(dn)

    after_dn = win.state.active_data()
    assert is_memmap(after_dn)
    assert after_dn.shape == (_HEIGHT, _WIDTH, _N_FRAMES)

    # --- Normalization (chained onto Denoising's committed output) ---
    nm = win.normalization_tab
    nm.on_data_loaded()
    nm._chunk_frames = 1500
    nm._apply()
    _wait(nm)
    nm._commit()
    _wait(nm)

    after_nm = win.state.active_data()
    assert is_memmap(after_nm)
    assert after_nm.shape == (_HEIGHT, _WIDTH, _N_FRAMES)
    assert win.state.pipeline == ["Load", "Rigid", "Gaussian Denoising", nm.state.steps[-1].label]

    # --- Source Extraction (patch-based CNMF, required for a memmap movie) ---
    se = win.source_extraction_tab
    se.on_data_loaded()
    _wait(se)
    se.method_combo.setCurrentIndex(se.method_combo.findText("CNMF"))
    se.cnmf_patch_check.setChecked(True)
    se.cnmf_patch_size_spin.setValue(20)
    se.cnmf_patch_overlap_spin.setValue(6)
    se.cnmf_components_per_patch_spin.setValue(3)
    se.cnmf_search_radius_spin.setValue(8)
    se._on_run_cnmf_clicked()
    _wait(se)

    assert len(se._candidates) > 0
    for roi in se._candidates:
        roi.status = "accepted"
    se._commit()
    _wait(se)

    assert len(win.state.rois) > 0
    for roi in win.state.rois:
        assert roi.trace.shape[-1] == _N_FRAMES  # re-extracted from the full (post-pipeline) movie
