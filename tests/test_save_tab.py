from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("h5py")

from PySide6.QtWidgets import QApplication  # noqa: E402

from orbitapp import session_io  # noqa: E402
from orbitapp.state import AppState, ROI  # noqa: E402
from orbitapp.tabs.roi_validation_tab import ROIValidationTab  # noqa: E402
from orbitapp.tabs.save_tab import SaveTab, _derive_paths, _load_with_sibling, _sibling_path  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def test_derive_paths_appends_pipeline_and_output_suffixes():
    pipeline_path, output_path = _derive_paths("/tmp/myrun.h5")
    assert pipeline_path == Path("/tmp/myrun_pipeline.h5")
    assert output_path == Path("/tmp/myrun_output.h5")


def test_derive_paths_strips_an_existing_suffix_first():
    pipeline_path, output_path = _derive_paths("/tmp/myrun_pipeline.h5")
    assert pipeline_path == Path("/tmp/myrun_pipeline.h5")
    assert output_path == Path("/tmp/myrun_output.h5")


def test_sibling_path_swaps_the_tag():
    assert _sibling_path("/tmp/myrun_pipeline.h5", "_pipeline", "_output") == Path("/tmp/myrun_output.h5")
    assert _sibling_path("/tmp/myrun.h5", "_pipeline", "_output") is None


def _state_with_a_committed_roi() -> AppState:
    state = AppState()
    state.load("movie.tif", np.zeros((10, 10, 5)))
    state.commit_rois(
        [ROI(id=0, mask=np.eye(10, dtype=bool), trace=np.arange(5, dtype=float), source_method="correlation",
             status="accepted", seed_loc=(3, 3), params={"threshold": 0.7})],
        "Correlation ROIs",
    )
    return state


def test_load_with_sibling_finds_the_output_file_from_the_pipeline_file(tmp_path):
    state = _state_with_a_committed_roi()
    pipeline_path = tmp_path / "run_pipeline.h5"
    output_path = tmp_path / "run_output.h5"
    session_io.save_pipeline(state, {}, pipeline_path)
    session_io.save_output(state, None, output_path)

    pipeline_data, output_data = _load_with_sibling(str(pipeline_path))

    assert pipeline_data is not None
    assert output_data is not None
    assert len(output_data["rois"]) == 1


def test_load_with_sibling_returns_none_for_missing_sibling(tmp_path):
    state = _state_with_a_committed_roi()
    pipeline_path = tmp_path / "run_pipeline.h5"
    session_io.save_pipeline(state, {}, pipeline_path)

    pipeline_data, output_data = _load_with_sibling(str(pipeline_path))

    assert pipeline_data is not None
    assert output_data is None


def test_save_pipeline_button_writes_a_file_and_emits_nothing(tmp_path, monkeypatch):
    state = _state_with_a_committed_roi()
    roi_tab = ROIValidationTab(state)
    tab = SaveTab(state, roi_tab)

    path = tmp_path / "chosen.h5"
    monkeypatch.setattr("orbitapp.tabs.save_tab.QFileDialog.getSaveFileName", lambda *a, **k: (str(path), ""))

    tab._on_save_pipeline_clicked()

    assert path.exists()
    loaded = session_io.load_pipeline(path)
    assert loaded["source_extraction_rois"][0]["seed_loc"] == (3, 3)


def test_full_save_button_writes_both_files(tmp_path, monkeypatch):
    state = _state_with_a_committed_roi()
    roi_tab = ROIValidationTab(state)
    tab = SaveTab(state, roi_tab)

    base = tmp_path / "session.h5"
    monkeypatch.setattr("orbitapp.tabs.save_tab.QFileDialog.getSaveFileName", lambda *a, **k: (str(base), ""))

    tab._on_full_save_clicked()

    assert (tmp_path / "session_pipeline.h5").exists()
    assert (tmp_path / "session_output.h5").exists()


def test_load_session_button_emits_session_loaded_with_both_parts(tmp_path, monkeypatch):
    state = _state_with_a_committed_roi()
    roi_tab = ROIValidationTab(state)
    tab = SaveTab(state, roi_tab)

    pipeline_path = tmp_path / "run_pipeline.h5"
    output_path = tmp_path / "run_output.h5"
    session_io.save_pipeline(state, {}, pipeline_path)
    session_io.save_output(state, None, output_path)
    monkeypatch.setattr(
        "orbitapp.tabs.save_tab.QFileDialog.getOpenFileName", lambda *a, **k: (str(pipeline_path), "")
    )

    received = []
    tab.session_loaded.connect(received.append)
    tab._on_load_session_clicked()

    assert len(received) == 1
    assert received[0]["pipeline"] is not None
    assert received[0]["output"] is not None


def test_load_session_button_shows_error_for_an_unrecognized_file(tmp_path, monkeypatch):
    bad_path = tmp_path / "not_a_session.h5"
    import h5py
    with h5py.File(bad_path, "w") as f:
        f.attrs["unrelated"] = "data"

    state = AppState()
    roi_tab = ROIValidationTab(state)
    tab = SaveTab(state, roi_tab)
    monkeypatch.setattr("orbitapp.tabs.save_tab.QFileDialog.getOpenFileName", lambda *a, **k: (str(bad_path), ""))
    errors = []
    monkeypatch.setattr(
        "orbitapp.tabs.save_tab.QMessageBox.critical", lambda *a, **k: errors.append(1)
    )

    received = []
    tab.session_loaded.connect(received.append)
    tab._on_load_session_clicked()

    assert errors == [1]
    assert received == []
