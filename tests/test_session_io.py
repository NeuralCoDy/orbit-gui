import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from orbitapp import session_io  # noqa: E402
from orbitapp.state import AppState, ROI  # noqa: E402


def _populated_state() -> AppState:
    state = AppState()
    state.load("/some/movie.tif", np.zeros((6, 6, 10)))
    state.commit(
        np.zeros((6, 6, 10)), "Rigid", stage="motion_correction",
        params={"method": "Rigid", "max_shift": 15.0}, metrics={"mmd": 1.23, "ecc": 0.98},
    )
    state.rois = [
        ROI(id=0, mask=np.eye(6, dtype=bool), trace=np.arange(10, dtype=float), source_method="correlation",
            status="accepted", seed_loc=(3, 3), params={"threshold": 0.73, "max_dist": 15.0}),
        ROI(id=1, mask=np.eye(6, dtype=bool)[::-1], trace=np.arange(10, dtype=float) * 2, source_method="pca_ica",
            status="accepted", spike_trace=np.zeros(10), neuropil_trace=np.ones(10),
            params={"n_pca_components": 50, "n_ica_components": 40}),
    ]
    return state


def test_save_and_load_pipeline_round_trips_steps_and_roi_metadata(tmp_path):
    state = _populated_state()
    roi_validation_params = {"auto_classify": {"criterion_label": "correlation", "threshold": 0.4}, "run_seudo": {}}
    path = tmp_path / "session_pipeline.h5"

    session_io.save_pipeline(state, roi_validation_params, path)
    loaded = session_io.load_pipeline(path)

    assert loaded["data_path"] == "/some/movie.tif"
    assert loaded["pipeline"] == state.pipeline
    assert [s.label for s in loaded["steps"]] == [s.label for s in state.steps]
    assert loaded["steps"][1].stage == "motion_correction"
    assert loaded["steps"][1].params == {"method": "Rigid", "max_shift": 15.0}

    roi0, roi1 = loaded["source_extraction_rois"]
    assert roi0["id"] == 0
    assert roi0["source_method"] == "correlation"
    assert roi0["seed_loc"] == (3, 3)
    assert roi0["params"] == {"threshold": 0.73, "max_dist": 15.0}
    assert roi1["seed_loc"] is None
    assert roi1["params"] == {"n_pca_components": 50, "n_ica_components": 40}

    assert loaded["roi_validation"] == roi_validation_params


def test_save_and_load_output_round_trips_roi_arrays_and_metrics(tmp_path):
    state = _populated_state()
    roi_validation_results = [
        {"times": [[10, 20], [50, 60]], "classification": [1.0, float("nan")], "is_artifact": False},
        {"times": np.zeros((0, 2), dtype=int), "classification": np.array([]), "is_artifact": True},
    ]
    path = tmp_path / "session_output.h5"

    session_io.save_output(state, roi_validation_results, path)
    loaded = session_io.load_output(path)

    assert len(loaded["rois"]) == 2
    roi0, roi1 = loaded["rois"]
    assert roi0.id == 0
    assert np.array_equal(roi0.mask, state.rois[0].mask)
    assert np.array_equal(roi0.trace, state.rois[0].trace)
    assert roi0.spike_trace is None
    assert roi1.spike_trace is not None
    assert np.array_equal(roi1.spike_trace, state.rois[1].spike_trace)
    assert np.array_equal(roi1.neuropil_trace, state.rois[1].neuropil_trace)

    assert loaded["metrics_by_label"]["Rigid"] == {"mmd": 1.23, "ecc": 0.98}

    rv0, rv1 = loaded["roi_validation_results"]
    assert np.array_equal(rv0["times"], [[10, 20], [50, 60]])
    assert rv0["classification"][0] == 1.0
    assert np.isnan(rv0["classification"][1])
    assert rv0["is_artifact"] is False
    assert rv1["is_artifact"] is True


def test_save_output_without_roi_validation_results_omits_that_group(tmp_path):
    state = _populated_state()
    path = tmp_path / "session_output.h5"

    session_io.save_output(state, None, path)
    loaded = session_io.load_output(path)

    assert loaded["roi_validation_results"] is None


def test_load_pipeline_handles_no_committed_rois(tmp_path):
    state = AppState()
    state.load("/x.tif", np.zeros((4, 4, 5)))
    path = tmp_path / "empty_pipeline.h5"

    session_io.save_pipeline(state, {}, path)
    loaded = session_io.load_pipeline(path)

    assert loaded["source_extraction_rois"] == []
    assert [s.label for s in loaded["steps"]] == ["Load"]
