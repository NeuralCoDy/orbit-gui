import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("h5py")
tifffile = pytest.importorskip("tifffile")

from PySide6.QtWidgets import QApplication  # noqa: E402

from orbit.seudo.constants import VAL_TRUE  # noqa: E402
from orbitapp.app import MainWindow  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def _wait_for_worker(tab, timeout_ms=10000):
    if tab.worker is not None:
        tab.worker.wait(timeout_ms)
    for _ in range(50):
        QApplication.processEvents()


def _write_synthetic_movie(path, height=30, width=30, n_frames=100, seed=0):
    rng = np.random.default_rng(seed)
    movie = rng.standard_normal((height, width, n_frames)).astype(np.float64) * 0.1 + 5.0
    movie[5:15, 5:15, :] += 3 * np.clip(rng.standard_normal(n_frames), 0, None)
    tifffile.imwrite(path, np.moveaxis(movie, -1, 0).astype("float32"))


def test_session_round_trip_restores_movie_rois_params_and_validation(tmp_path, monkeypatch):
    movie_path = tmp_path / "movie.tif"
    _write_synthetic_movie(movie_path)

    win = MainWindow()
    win.load_tab._load(str(movie_path))
    _wait_for_worker(win.load_tab)

    se = win.source_extraction_tab
    se.on_data_loaded()
    _wait_for_worker(se)
    se._on_pixel_clicked(9, 9)
    _wait_for_worker(se)
    assert se._preview_roi is not None
    se._on_add_roi_clicked()
    se._candidates[0].status = "accepted"
    se._commit()
    assert len(win.state.rois) == 1
    original_roi = win.state.rois[0]

    rv = win.roi_validation_tab
    rv._on_load_rois_clicked()
    classified_any = False
    if rv.se is not None:
        ti = rv._tc_struct()["transient_info"][0]
        if ti["times"].shape[0] > 0:
            ti["classification"][:] = VAL_TRUE
            classified_any = True

    base_path = tmp_path / "session.h5"
    monkeypatch.setattr(
        "orbitapp.tabs.save_tab.QFileDialog.getSaveFileName", lambda *a, **k: (str(base_path), "")
    )
    win.save_tab._on_full_save_clicked()

    win2 = MainWindow()
    monkeypatch.setattr(
        "orbitapp.tabs.save_tab.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(tmp_path / "session_pipeline.h5"), ""),
    )
    win2.save_tab._on_load_session_clicked()

    assert win2.state.data_path == str(movie_path)
    assert win2.state.active_data() is not None
    assert win2.state.active_data().shape == win.state.active_data().shape
    assert win2.state.pipeline == win.state.pipeline

    assert len(win2.state.rois) == 1
    restored_roi = win2.state.rois[0]
    assert restored_roi.seed_loc == original_roi.seed_loc
    assert restored_roi.params == original_roi.params
    assert np.array_equal(restored_roi.mask, original_roi.mask)
    assert np.array_equal(restored_roi.trace, original_roi.trace)

    if classified_any:
        assert win2.roi_validation_tab.se is not None
        restored_ti = win2.roi_validation_tab._tc_struct()["transient_info"][0]
        assert restored_ti["classification"][0] == VAL_TRUE


def test_pipeline_only_load_restores_params_without_rois(tmp_path, monkeypatch):
    movie_path = tmp_path / "movie.tif"
    _write_synthetic_movie(movie_path)

    win = MainWindow()
    win.load_tab._load(str(movie_path))
    _wait_for_worker(win.load_tab)

    mc = win.motion_correction_tab
    mc.on_data_loaded()
    mc.max_shift_spin.setValue(42.0)
    mc._apply()
    _wait_for_worker(mc)
    mc._commit()

    path = tmp_path / "pipeline_only.h5"
    monkeypatch.setattr(
        "orbitapp.tabs.save_tab.QFileDialog.getSaveFileName", lambda *a, **k: (str(path), "")
    )
    win.save_tab._on_save_pipeline_clicked()

    win2 = MainWindow()
    monkeypatch.setattr(
        "orbitapp.tabs.save_tab.QFileDialog.getOpenFileName", lambda *a, **k: (str(path), "")
    )
    informed = []
    monkeypatch.setattr(
        "orbitapp.app.QMessageBox.information", lambda *a, **k: informed.append(1)
    )
    win2.save_tab._on_load_session_clicked()

    assert win2.motion_correction_tab.max_shift_spin.value() == 42.0
    assert win2.state.rois == []


def test_generate_report_with_nothing_committed_shows_a_guard_dialog(monkeypatch):
    informed = []
    monkeypatch.setattr("orbitapp.app.QMessageBox.information", lambda *a, **k: informed.append(1))

    win = MainWindow()
    win._on_generate_report_clicked()

    assert informed == [1]
    assert win.report_worker is None


def test_generate_report_happy_path_writes_a_pdf_and_shows_the_path(tmp_path, monkeypatch):
    pytest.importorskip("shutil")
    import shutil

    if shutil.which("pdflatex") is None:
        pytest.skip("pdflatex not on PATH")

    movie_path = tmp_path / "movie.tif"
    _write_synthetic_movie(movie_path)

    win = MainWindow()
    win.load_tab._load(str(movie_path))
    _wait_for_worker(win.load_tab)

    mc = win.motion_correction_tab
    mc.on_data_loaded()
    mc._apply()
    _wait_for_worker(mc)
    mc._commit()

    report_path = tmp_path / "report.pdf"
    monkeypatch.setattr(
        "orbitapp.app.QFileDialog.getSaveFileName", lambda *a, **k: (str(report_path), "")
    )
    informed = []
    monkeypatch.setattr("orbitapp.app.QMessageBox.information", lambda *a, **k: informed.append(1))

    win._on_generate_report_clicked()
    win.report_worker.wait(30000)
    for _ in range(50):
        QApplication.processEvents()

    assert report_path.exists()
    assert informed == [1]
    assert win.header.report_btn.isEnabled()
    assert win.header.report_btn.text() == "Generate Report..."


def test_generate_report_failure_shows_critical_dialog_and_reenables_button(tmp_path, monkeypatch):
    movie_path = tmp_path / "movie.tif"
    _write_synthetic_movie(movie_path)

    win = MainWindow()
    win.load_tab._load(str(movie_path))
    _wait_for_worker(win.load_tab)

    mc = win.motion_correction_tab
    mc.on_data_loaded()
    mc._apply()
    _wait_for_worker(mc)
    mc._commit()

    monkeypatch.setattr(
        "orbitapp.app.QFileDialog.getSaveFileName", lambda *a, **k: (str(tmp_path / "report.pdf"), "")
    )
    monkeypatch.setattr(
        "orbitapp.report.render_report", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    critical = []
    monkeypatch.setattr("orbitapp.app.QMessageBox.critical", lambda *a, **k: critical.append(a[2] if len(a) > 2 else ""))

    win._on_generate_report_clicked()
    win.report_worker.wait(15000)
    for _ in range(50):
        QApplication.processEvents()

    assert len(critical) == 1
    assert "boom" in critical[0]
    assert win.header.report_btn.isEnabled()
