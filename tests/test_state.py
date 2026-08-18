import numpy as np

from orbitapp.state import AppState


def test_active_data_prefers_preprocessed_over_original():
    state = AppState()
    original = np.zeros((2, 2, 3))
    processed = np.ones((2, 2, 3))

    state.original_data = original
    assert state.active_data() is original

    state.preprocessed_data = processed
    assert state.active_data() is processed


def test_load_resets_pipeline_and_preprocessed_data():
    state = AppState()
    state.preprocessed_data = np.ones((2, 2, 3))
    state.pipeline = ["Load", "Rigid"]

    movie = np.zeros((4, 4, 5))
    state.load("/some/movie.tif", movie)

    assert state.data_path == "/some/movie.tif"
    assert state.original_data is movie
    assert state.preprocessed_data is None
    assert state.pipeline == ["Load"]
    assert state.active_data() is movie


def test_commit_updates_active_data_and_appends_pipeline_step():
    state = AppState()
    state.load("/some/movie.tif", np.zeros((4, 4, 5)))

    corrected = np.ones((4, 4, 5))
    state.commit(corrected, "Rigid")

    assert state.active_data() is corrected
    assert state.pipeline == ["Load", "Rigid"]


def test_commit_can_be_called_multiple_times():
    state = AppState()
    state.load("/some/movie.tif", np.zeros((4, 4, 5)))

    state.commit(np.ones((4, 4, 5)), "Patch Warp")
    state.commit(np.full((4, 4, 5), 2.0), "Normalize")

    assert state.pipeline == ["Load", "Patch Warp", "Normalize"]
    assert np.all(state.active_data() == 2.0)
