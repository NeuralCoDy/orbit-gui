import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from orbit.seudo.constants import VAL_FALSE, VAL_TRUE  # noqa: E402
from orbitapp.state import AppState, ROI  # noqa: E402
from orbitapp.tabs.roi_validation_tab import ROIValidationTab  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def _synthetic_movie(height=30, width=30, n_frames=200, seed=0):
    rng = np.random.default_rng(seed)
    movie = rng.standard_normal((height, width, n_frames)).astype(np.float64) * 0.1 + 5.0
    movie[5:15, 5:15, :] += 3 * np.clip(rng.standard_normal(n_frames), 0, None)
    movie[10:20, 10:20, :] += 4 * np.clip(rng.standard_normal(n_frames), 0, None)
    return movie


def _rois_for(movie):
    mask1 = np.zeros(movie.shape[:2], dtype=bool)
    mask1[5:15, 5:15] = True
    mask2 = np.zeros(movie.shape[:2], dtype=bool)
    mask2[10:20, 10:20] = True
    return [
        ROI(id=0, mask=mask1, trace=movie[mask1].mean(axis=0), source_method="pca_ica", status="accepted"),
        ROI(id=1, mask=mask2, trace=movie[mask2].mean(axis=0), source_method="pca_ica", status="accepted"),
    ]


def _loaded_tab():
    movie = _synthetic_movie()
    state = AppState()
    state.load("movie.tif", movie)
    state.rois = _rois_for(movie)
    tab = ROIValidationTab(state)
    tab.on_data_loaded()
    tab._on_load_rois_clicked()
    return tab


def test_load_rois_without_committed_rois_warns(monkeypatch):
    monkeypatch.setattr("orbitapp.tabs.roi_validation_tab.QMessageBox.warning", lambda *a, **k: None)
    state = AppState()
    state.load("movie.tif", _synthetic_movie())
    tab = ROIValidationTab(state)
    tab.on_data_loaded()

    tab._on_load_rois_clicked()

    assert tab.se is None


def test_load_rois_builds_a_seudo_session_and_enables_navigation():
    tab = _loaded_tab()

    assert tab.se is not None
    assert tab.se.n_cells == 2
    assert tab.cell_slider.isEnabled()
    assert tab.cell_slider.maximum() == 2


def test_on_data_loaded_resets_a_stale_session():
    tab = _loaded_tab()
    assert tab.se is not None

    tab.on_data_loaded()

    assert tab.se is None
    assert not tab.cell_slider.isEnabled()


def test_thumbnail_click_cycles_classification_unclassified_to_false():
    tab = _loaded_tab()
    ti = tab._cell_transient_info()
    assert ti["times"].shape[0] >= 1
    assert np.isnan(ti["classification"][0])

    fake_ax = object()
    tab._ax_to_trans = {fake_ax: 0}

    class _FakeEvent:
        inaxes = fake_ax

    tab._on_thumbnail_click(_FakeEvent())

    assert ti["classification"][0] == VAL_FALSE


def test_artifact_checkbox_sets_is_artifact_on_the_current_cell():
    tab = _loaded_tab()
    ti = tab._cell_transient_info()
    assert ti["is_artifact"] is False

    tab.artifact_checkbox.setChecked(True)

    assert ti["is_artifact"] is True


def test_go_to_cell_updates_nav_widgets_and_current_cell():
    tab = _loaded_tab()
    tab._go_to_cell(1)
    assert tab.this_cell == 1
    assert tab.cell_slider.value() == 2
    assert tab.cell_spin.value() == 2


def test_go_to_next_and_prev_unclassified():
    tab = _loaded_tab()
    tab._cell_transient_info()["classification"][:] = VAL_TRUE  # classify cell 0's only transient

    tab._go_to_cell(0)
    tab.go_to_next_unclassified()
    assert tab.this_cell == 1  # cell 1 is still fully unclassified

    # cell 0 is no longer unclassified, so there's nowhere earlier to go
    tab.go_to_prev_unclassified()
    assert tab.this_cell == 1


def test_save_and_load_classification_round_trips(tmp_path, monkeypatch):
    tab = _loaded_tab()
    tab._cell_transient_info()["classification"][:] = VAL_TRUE

    path = tmp_path / "classification.pkl"
    monkeypatch.setattr(
        "orbitapp.tabs.roi_validation_tab.QFileDialog.getSaveFileName", lambda *a, **k: (str(path), "")
    )
    tab._on_save()
    assert path.exists()

    tab._cell_transient_info()["classification"][:] = np.nan  # forget it locally
    monkeypatch.setattr(
        "orbitapp.tabs.roi_validation_tab.QFileDialog.getOpenFileName", lambda *a, **k: (str(path), "")
    )
    tab._on_load()

    assert tab._cell_transient_info()["classification"][0] == VAL_TRUE


def test_export_results_is_none_before_a_session_is_loaded():
    state = AppState()
    state.load("movie.tif", _synthetic_movie())
    tab = ROIValidationTab(state)
    tab.on_data_loaded()

    assert tab.export_results() is None


def test_export_then_import_results_round_trips_classification_and_artifact():
    tab = _loaded_tab()
    tab._go_to_cell(0)
    tab._cell_transient_info()["classification"][:] = VAL_TRUE
    tab._go_to_cell(1)
    tab._cell_transient_info()["is_artifact"] = True

    exported = tab.export_results()
    assert exported is not None
    assert len(exported) == tab.se.n_cells

    # simulate a fresh session (e.g. after reloading the app) then restore
    fresh = _loaded_tab()
    fresh.import_results(exported)

    assert fresh._tc_struct()["transient_info"][0]["classification"][0] == VAL_TRUE
    assert fresh._tc_struct()["transient_info"][1]["is_artifact"] is True


def test_export_pipeline_params_reflects_last_auto_classify_and_seudo_runs():
    tab = _loaded_tab()
    assert tab.export_pipeline_params() == {"auto_classify": {}, "run_seudo": {}}

    tab._last_auto_classify_params = {"criterion_label": "correlation", "threshold": 0.4}
    tab._last_seudo_params = {"sigma2": 0.002}

    params = tab.export_pipeline_params()
    assert params["auto_classify"] == {"criterion_label": "correlation", "threshold": 0.4}
    assert params["run_seudo"] == {"sigma2": 0.002}
