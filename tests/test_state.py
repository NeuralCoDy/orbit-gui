import numpy as np

from orbitapp.state import ROI, AppState


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


def test_modality_modifiers_empty_by_default():
    state = AppState()
    assert state.modality_modifiers() == []


def test_modality_modifiers_in_fixed_display_order():
    state = AppState()
    state.volumetric = True
    state.dendrites = True
    assert state.modality_modifiers() == ["dendrites", "volumetric"]


def test_load_does_not_reset_modality_toggles():
    # Unlike pipeline/rois/steps, the modality toggles describe the
    # *kind* of dataset being analyzed, independent of any one file --
    # a fresh Load shouldn't silently clear the user's selection.
    state = AppState()
    state.widefield = True
    state.load("/some/movie.tif", np.zeros((4, 4, 5)))
    assert state.widefield


def test_modality_modifiers_includes_somatic_1p_and_somatic_2p_in_fixed_order():
    state = AppState()
    state.somatic_2p = True
    state.dendrites = True
    state.somatic_1p = True
    assert state.modality_modifiers() == ["dendrites", "somatic_1p", "somatic_2p"]


def test_load_does_not_reset_somatic_toggles():
    state = AppState()
    state.somatic_1p = True
    state.load("/some/movie.tif", np.zeros((4, 4, 5)))
    assert state.somatic_1p


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


def _make_roi(roi_id: int, source_method: str = "correlation") -> ROI:
    return ROI(
        id=roi_id,
        mask=np.zeros((4, 4), dtype=bool),
        trace=np.zeros(5),
        source_method=source_method,
    )


def test_commit_rois_accumulates_across_calls_and_methods():
    state = AppState()
    state.load("/some/movie.tif", np.zeros((4, 4, 5)))

    state.commit_rois([_make_roi(1, "correlation"), _make_roi(2, "correlation")], "Correlation ROIs")
    state.commit_rois([_make_roi(3, "pca_ica")], "PCA-ICA")

    assert [r.id for r in state.rois] == [1, 2, 3]
    assert {r.source_method for r in state.rois} == {"correlation", "pca_ica"}
    assert state.pipeline == ["Load", "Correlation ROIs", "PCA-ICA"]


def test_load_resets_rois():
    state = AppState()
    state.load("/some/movie.tif", np.zeros((4, 4, 5)))
    state.commit_rois([_make_roi(1)], "Correlation ROIs")

    state.load("/other/movie.tif", np.zeros((4, 4, 5)))

    assert state.rois == []
    assert state.pipeline == ["Load"]


def test_clear_rois_empties_committed_rois_and_removes_their_pipeline_blocks():
    state = AppState()
    state.load("/some/movie.tif", np.zeros((4, 4, 5)))
    state.commit_rois([_make_roi(1, "correlation"), _make_roi(2, "correlation")], "Correlation ROIs")
    state.commit_rois([_make_roi(3, "real_seudo")], "Real-SEUDO")

    state.clear_rois()

    assert state.rois == []
    # Unlike every other stage's history (left as-is), the now-undone
    # source-extraction blocks are removed -- they'd otherwise claim ROIs
    # were committed that no longer exist anywhere in the session.
    assert state.pipeline == ["Load"]
    assert [s.stage for s in state.steps] == ["load"]


def test_clear_rois_leaves_non_source_extraction_pipeline_history_alone():
    state = AppState()
    state.load("/some/movie.tif", np.zeros((4, 4, 5)))
    state.commit(np.zeros((4, 4, 5)), "Motion Correction", stage="motion_correction")
    state.commit_rois([_make_roi(1, "correlation")], "Correlation ROIs")

    state.clear_rois()

    assert state.pipeline == ["Load", "Motion Correction"]
    assert [s.stage for s in state.steps] == ["load", "motion_correction"]


def test_commit_rois_records_params_and_metrics():
    state = AppState()
    state.load("/some/movie.tif", np.zeros((4, 4, 5)))

    state.commit_rois(
        [_make_roi(1, "real_seudo")], "Real-SEUDO",
        params={"sigma2": 0.002, "lambda_blob": 10.0}, metrics={"rois_committed": 1, "rois_deleted": 3},
    )

    step = state.steps[-1]
    assert step.params == {"sigma2": 0.002, "lambda_blob": 10.0}
    assert step.metrics == {"rois_committed": 1, "rois_deleted": 3}
