import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from orbit.roi_extraction_pca_ica import pca_ica_source_extraction  # noqa: E402
from orbitapp.state import AppState  # noqa: E402
from orbitapp.tabs.source_extraction_tab import SourceExtractionTab  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def _synthetic_movie(height=30, width=30, n_frames=60, seed=0):
    rng = np.random.default_rng(seed)
    movie = rng.standard_normal((height, width, n_frames)).astype(np.float64) * 0.1 + 5.0
    movie[5:10, 5:10, :] += rng.standard_normal(n_frames)
    movie[20:25, 20:25, :] += rng.standard_normal(n_frames)
    return movie


def _wait_for_worker(tab, timeout_ms=10000):
    if tab.worker is not None:
        tab.worker.wait(timeout_ms)
    for _ in range(50):
        QApplication.processEvents()


def test_on_data_loaded_computes_correlation_image_and_resets_candidates():
    state = AppState()
    state.load("movie.tif", _synthetic_movie())
    tab = SourceExtractionTab(state)

    tab.on_data_loaded()
    _wait_for_worker(tab)

    assert tab._corr_image is not None
    assert tab._corr_image.shape == (30, 30)
    assert tab._candidates == []


def test_method_combo_only_offers_batch_algorithms():
    state = AppState()
    tab = SourceExtractionTab(state)

    labels = [tab.method_combo.itemText(i) for i in range(tab.method_combo.count())]
    assert labels == ["PCA-ICA", "CNMF", "GraFT"]  # correlation click-to-add is not a Method option


def test_method_combo_switches_parameter_group_and_action_widget():
    state = AppState()
    tab = SourceExtractionTab(state)

    idx = tab.method_combo.findText("PCA-ICA")
    tab.method_combo.setCurrentIndex(idx)

    assert tab.params_dialog.form.isRowVisible(tab.n_pca_components_spin)
    assert tab.action_stack.currentWidget() is tab.action_stack.widget(0)

    idx = tab.method_combo.findText("CNMF")
    tab.method_combo.setCurrentIndex(idx)
    assert not tab.params_dialog.form.isRowVisible(tab.n_pca_components_spin)
    assert tab.params_dialog.form.isRowVisible(tab.cnmf_n_components_spin)
    assert tab.action_stack.currentWidget() is tab.action_stack.widget(1)


def test_correlation_params_live_on_the_main_screen_not_the_dialog():
    # Correlation click-to-add is always active, independent of Method --
    # its params live on the main screen, not behind Parameters..., so
    # they're unaffected by the dialog's per-method show_only_group.
    state = AppState()
    tab = SourceExtractionTab(state)

    assert tab.max_dist_spin.parent() is not tab.params_dialog
    assert tab.growth_method_combo.parent() is not tab.params_dialog
    assert tab.seed_block_radius_spin.parent() is not tab.params_dialog
    assert tab.n_auto_seeds_spin.parent() is not tab.params_dialog


def test_params_dialog_only_shows_selected_methods_group():
    state = AppState()
    tab = SourceExtractionTab(state)

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("PCA-ICA"))
    assert tab.params_dialog.form.isRowVisible(tab.n_pca_components_spin)
    assert not tab.params_dialog.form.isRowVisible(tab.cnmf_n_components_spin)

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("CNMF"))
    assert not tab.params_dialog.form.isRowVisible(tab.n_pca_components_spin)
    assert tab.params_dialog.form.isRowVisible(tab.cnmf_n_components_spin)


def test_cnmf_patch_rows_only_visible_for_cnmf_method_and_when_checked():
    state = AppState()
    tab = SourceExtractionTab(state)

    # Default: PCA-ICA selected, checkbox unchecked -- patch rows hidden either way.
    assert not tab.params_dialog.form.isRowVisible(tab.cnmf_patch_size_spin)

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("CNMF"))
    assert tab.params_dialog.form.isRowVisible(tab.cnmf_patch_check)
    assert not tab.params_dialog.form.isRowVisible(tab.cnmf_patch_size_spin)  # checkbox still unchecked

    tab.cnmf_patch_check.setChecked(True)
    assert tab.params_dialog.form.isRowVisible(tab.cnmf_patch_size_spin)
    assert tab.params_dialog.form.isRowVisible(tab.cnmf_patch_overlap_spin)
    assert tab.params_dialog.form.isRowVisible(tab.cnmf_components_per_patch_spin)

    # Switching away and back to CNMF must not lose the checked state's rows.
    tab.method_combo.setCurrentIndex(tab.method_combo.findText("PCA-ICA"))
    assert not tab.params_dialog.form.isRowVisible(tab.cnmf_patch_size_spin)
    tab.method_combo.setCurrentIndex(tab.method_combo.findText("CNMF"))
    assert tab.params_dialog.form.isRowVisible(tab.cnmf_patch_size_spin)


def test_fov_view_panning_is_always_disabled():
    # Clicking the FOV view always seeds a new ROI now (no separate "click
    # mode" toggle), so pyqtgraph's default left-drag panning -- which a
    # real mouse's few pixels of jitter could trigger mid-click -- is
    # permanently off there rather than only while some mode is armed.
    state = AppState()
    tab = SourceExtractionTab(state)
    view_box = tab.review_panel.fov_view.getView()
    assert view_box.state["mouseEnabled"] == [False, False]


def test_grow_from_seeds_adds_pending_candidates_and_does_not_auto_accept():
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab._grow_from_seeds(movie, [(7, 7)])
    _wait_for_worker(tab)

    assert len(tab._candidates) == 1
    roi = tab._candidates[0]
    assert roi.status == "pending"
    assert roi.source_method == "correlation"
    assert roi.mask[7, 7]
    assert roi.trace.shape == (movie.shape[2],)


def test_auto_seed_click_proposes_seeds_needing_review():
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab.n_auto_seeds_spin.setValue(2)
    tab._on_auto_seed_clicked()
    _wait_for_worker(tab)

    assert len(tab._candidates) == 2
    assert all(roi.status == "pending" for roi in tab._candidates)


def test_commit_moves_only_accepted_rois_into_state_and_accumulates():
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab._grow_from_seeds(movie, [(7, 7), (22, 22)])
    _wait_for_worker(tab)
    assert len(tab._candidates) == 2

    tab._candidates[0].status = "accepted"
    tab._candidates[1].status = "rejected"

    committed = []
    tab.data_changed.connect(lambda: committed.append(1))
    tab._commit()

    assert len(state.rois) == 1
    assert state.rois[0].status == "accepted"
    assert state.pipeline == ["Load", "Correlation ROIs"]
    assert committed == [1]
    # the rejected candidate stays around for continued review, not silently dropped
    assert len(tab._candidates) == 1
    assert tab._candidates[0].status == "rejected"

    # a second commit accumulates rather than replacing
    tab._grow_from_seeds(movie, [(7, 7)])
    _wait_for_worker(tab)
    tab._candidates[-1].status = "accepted"
    tab._commit()
    assert len(state.rois) == 2


def test_clicking_grows_a_preview_but_does_not_add_it_to_the_collection():
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab._on_pixel_clicked(7, 7)  # clicking is always active, no mode toggle needed
    _wait_for_worker(tab)

    assert tab._preview_roi is not None
    assert tab._preview_roi.id == -1
    assert tab._preview_roi.mask[7, 7]
    assert tab._candidates == []  # not added yet
    assert tab.add_roi_btn.isEnabled()


def test_preview_is_listed_as_roi_minus_one_and_auto_selected_for_its_trace():
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab._on_pixel_clicked(7, 7)
    _wait_for_worker(tab)

    panel = tab.review_panel
    assert panel.table.rowCount() == 1
    assert panel.table.item(0, 0).text() == "-1"
    # auto-selected -- trace shows without an explicit click on the row
    assert len(panel.trace_plot.getPlotItem().listDataItems()) == 1


def test_add_roi_button_moves_preview_into_the_collection():
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab._on_pixel_clicked(7, 7)
    _wait_for_worker(tab)

    tab._on_add_roi_clicked()

    assert len(tab._candidates) == 1
    assert tab._candidates[0].id == 0
    assert tab._candidates[0].status == "pending"
    assert tab._preview_roi is None
    assert not tab.add_roi_btn.isEnabled()
    assert tab.review_panel.table.item(0, 0).text() == "0"


def test_clicking_again_replaces_the_previous_unadded_preview():
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab._on_pixel_clicked(7, 7)
    _wait_for_worker(tab)
    tab._on_pixel_clicked(22, 22)
    _wait_for_worker(tab)

    assert tab._preview_roi.seed_loc == (22, 22)
    assert tab._candidates == []  # neither preview was ever added
    assert tab.review_panel.table.rowCount() == 1  # still just the one preview row


def test_review_panel_collection_click_selects_matching_table_row():
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab._grow_from_seeds(movie, [(7, 7), (22, 22)])
    _wait_for_worker(tab)

    panel = tab.review_panel
    idx = panel._find_roi_at_pixel(22, 22)
    assert idx == 1

    selected = []
    panel.roi_selected.connect(selected.append)
    panel.table.selectRow(idx)
    assert selected == [tab._candidates[1].id]


def test_delete_selected_removes_roi_from_collection_entirely():
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab._grow_from_seeds(movie, [(7, 7), (22, 22)])
    _wait_for_worker(tab)
    assert len(tab._candidates) == 2

    tab.review_panel.table.selectRow(0)
    tab.review_panel._delete_selected()

    assert len(tab._candidates) == 1
    assert tab._candidates[0].seed_loc == (22, 22)


def test_commit_with_nothing_accepted_is_a_no_op(monkeypatch):
    # _commit() pops a modal QMessageBox.information when there's nothing
    # accepted -- stub it out so the test doesn't block waiting for a click.
    monkeypatch.setattr("orbitapp.tabs.source_extraction_tab.QMessageBox.information", lambda *a, **k: None)

    state = AppState()
    state.load("movie.tif", _synthetic_movie())
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab._commit()

    assert state.rois == []
    assert state.pipeline == ["Load"]


def test_neuropil_traces_are_computed_when_candidates_are_added():
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab._grow_from_seeds(movie, [(7, 7), (22, 22)])
    _wait_for_worker(tab)

    assert all(roi.neuropil_trace is not None for roi in tab._candidates)
    assert all(roi.neuropil_trace.shape == (movie.shape[2],) for roi in tab._candidates)


def test_deleting_a_roi_recomputes_neuropil_for_the_rest():
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab._grow_from_seeds(movie, [(7, 7), (22, 22), (8, 8)])
    _wait_for_worker(tab)
    assert len(tab._candidates) == 3

    tab.review_panel.table.selectRow(0)
    tab.review_panel._delete_selected()

    assert len(tab._candidates) == 2
    assert all(roi.neuropil_trace is not None for roi in tab._candidates)


def test_run_pca_ica_adds_candidates_from_both_synthetic_blobs():
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab.n_pca_components_spin.setValue(10)
    tab.n_ica_components_spin.setValue(6)
    tab._on_run_pca_ica_clicked()
    _wait_for_worker(tab)

    assert len(tab._candidates) >= 2
    assert all(roi.source_method == "pca_ica" for roi in tab._candidates)
    assert all(roi.status == "pending" for roi in tab._candidates)
    assert all(roi.neuropil_trace is not None for roi in tab._candidates)

    centers = [np.argwhere(roi.mask).mean(axis=0) for roi in tab._candidates]
    near_a = any(np.hypot(*(c - (7.5, 7.5))) < 4 for c in centers)
    near_b = any(np.hypot(*(c - (22.5, 22.5))) < 4 for c in centers)
    assert near_a and near_b


def test_run_pca_ica_without_data_warns(monkeypatch):
    monkeypatch.setattr("orbitapp.tabs.source_extraction_tab.QMessageBox.warning", lambda *a, **k: None)
    state = AppState()
    tab = SourceExtractionTab(state)

    tab._on_run_pca_ica_clicked()

    assert tab._candidates == []


def test_run_cnmf_adds_candidates_from_both_synthetic_blobs():
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab.cnmf_n_components_spin.setValue(3)
    tab.cnmf_search_radius_spin.setValue(8)
    tab._on_run_cnmf_clicked()
    _wait_for_worker(tab)

    assert len(tab._candidates) == 3
    assert all(roi.source_method == "cnmf" for roi in tab._candidates)
    assert all(roi.status == "pending" for roi in tab._candidates)
    assert all(roi.neuropil_trace is not None for roi in tab._candidates)
    assert all(roi.spike_trace is not None for roi in tab._candidates)

    centers = [np.argwhere(roi.mask).mean(axis=0) for roi in tab._candidates if roi.mask.any()]
    near_a = any(np.hypot(*(c - (7.5, 7.5))) < 4 for c in centers)
    near_b = any(np.hypot(*(c - (22.5, 22.5))) < 4 for c in centers)
    assert near_a and near_b


def test_run_patch_cnmf_adds_candidates_from_both_synthetic_blobs():
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("CNMF"))
    tab.cnmf_patch_check.setChecked(True)
    tab.cnmf_patch_size_spin.setValue(18)
    tab.cnmf_patch_overlap_spin.setValue(6)
    tab.cnmf_components_per_patch_spin.setValue(2)
    tab.cnmf_search_radius_spin.setValue(8)
    tab._on_run_cnmf_clicked()
    _wait_for_worker(tab)

    assert len(tab._candidates) > 0
    assert all(roi.source_method == "cnmf" for roi in tab._candidates)
    assert all(roi.status == "pending" for roi in tab._candidates)
    assert all(roi.spike_trace is not None for roi in tab._candidates)

    centers = [np.argwhere(roi.mask).mean(axis=0) for roi in tab._candidates if roi.mask.any()]
    near_a = any(np.hypot(*(c - (7.5, 7.5))) < 4 for c in centers)
    near_b = any(np.hypot(*(c - (22.5, 22.5))) < 4 for c in centers)
    assert near_a and near_b


def test_run_patch_cnmf_records_patch_params_on_each_roi():
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("CNMF"))
    tab.cnmf_patch_check.setChecked(True)
    tab.cnmf_patch_size_spin.setValue(18)
    tab.cnmf_patch_overlap_spin.setValue(6)
    tab.cnmf_components_per_patch_spin.setValue(2)
    tab._on_run_cnmf_clicked()
    _wait_for_worker(tab)

    assert len(tab._candidates) > 0
    for roi in tab._candidates:
        assert roi.params["patch_size"] == (18, 18)
        assert roi.params["overlap"] == 6
        assert roi.params["n_components_per_patch"] == 2


def test_run_cnmf_without_data_warns(monkeypatch):
    monkeypatch.setattr("orbitapp.tabs.source_extraction_tab.QMessageBox.warning", lambda *a, **k: None)
    state = AppState()
    tab = SourceExtractionTab(state)

    tab._on_run_cnmf_clicked()

    assert tab._candidates == []


def _pca_ica_fingerprint(tab, movie):
    kwargs = dict(n_pca_components=tab.n_pca_components_spin.value(), n_ica_components=tab.n_ica_components_spin.value())
    return (id(movie), pca_ica_source_extraction, tuple(sorted(kwargs.items())))


def test_run_pca_ica_with_unchanged_parameters_prompts_and_skips_if_declined(monkeypatch):
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)
    tab._last_batch_run = _pca_ica_fingerprint(tab, state.active_data())  # simulate a prior successful run

    monkeypatch.setattr("orbitapp.tabs.source_extraction_tab.confirm_recompute", lambda *a, **k: False)
    tab._on_run_pca_ica_clicked()
    _wait_for_worker(tab)

    assert tab._candidates == []


def test_run_pca_ica_with_unchanged_parameters_reruns_if_confirmed(monkeypatch):
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)
    tab._last_batch_run = _pca_ica_fingerprint(tab, state.active_data())

    monkeypatch.setattr("orbitapp.tabs.source_extraction_tab.confirm_recompute", lambda *a, **k: True)
    tab._on_run_pca_ica_clicked()
    _wait_for_worker(tab)

    assert len(tab._candidates) >= 1


def test_run_pca_ica_with_changed_parameters_does_not_prompt(monkeypatch):
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)
    tab._last_batch_run = _pca_ica_fingerprint(tab, state.active_data())
    tab.n_pca_components_spin.setValue(tab.n_pca_components_spin.value() + 1)

    prompted = []
    monkeypatch.setattr(
        "orbitapp.tabs.source_extraction_tab.confirm_recompute", lambda *a, **k: prompted.append(1) or False
    )
    tab._on_run_pca_ica_clicked()
    _wait_for_worker(tab)

    assert prompted == []
    assert len(tab._candidates) >= 1


def test_clear_all_wipes_pending_candidates_and_committed_rois_when_confirmed(monkeypatch):
    monkeypatch.setattr("orbitapp.tabs.source_extraction_tab.QMessageBox.question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab._grow_from_seeds(movie, [(7, 7), (22, 22)])
    _wait_for_worker(tab)
    tab._candidates[0].status = "accepted"
    tab._commit()
    assert len(state.rois) == 1
    assert len(tab._candidates) == 1  # the un-accepted one stays pending

    changed = []
    tab.data_changed.connect(lambda: changed.append(1))
    tab._on_clear_all_clicked()

    assert tab._candidates == []
    assert state.rois == []
    assert state.pipeline == ["Load", "Correlation ROIs"]  # history log untouched
    assert changed == [1]
    assert tab._next_id == 0


def test_clear_all_does_nothing_when_declined(monkeypatch):
    monkeypatch.setattr("orbitapp.tabs.source_extraction_tab.QMessageBox.question", lambda *a, **k: QMessageBox.StandardButton.No)
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab._grow_from_seeds(movie, [(7, 7)])
    _wait_for_worker(tab)
    tab._candidates[0].status = "accepted"
    tab._commit()

    tab._on_clear_all_clicked()

    assert len(tab._candidates) == 0  # the accepted one was already committed, none left pending
    assert len(state.rois) == 1


def test_clear_all_with_nothing_to_clear_does_not_prompt(monkeypatch):
    prompted = []
    monkeypatch.setattr(
        "orbitapp.tabs.source_extraction_tab.QMessageBox.question", lambda *a, **k: prompted.append(1)
    )
    state = AppState()
    tab = SourceExtractionTab(state)

    tab._on_clear_all_clicked()

    assert prompted == []


def test_correlation_preview_records_seed_loc_and_resolved_threshold():
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab._on_pixel_clicked(7, 7)
    _wait_for_worker(tab)

    roi = tab._preview_roi
    assert roi.seed_loc == (7, 7)
    assert roi.params is not None
    assert isinstance(roi.params["threshold"], float)
    assert roi.params["max_dist"] == tab.max_dist_spin.value()
    assert roi.params["growth_method"] == tab.growth_method_combo.currentText()
    assert "local_corr_image" not in roi.params  # not JSON-serializable, and not a "parameter" per se


def test_correlation_batch_seeds_each_record_their_own_seed_loc_and_threshold():
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab._grow_from_seeds(movie, [(7, 7), (22, 22)])
    _wait_for_worker(tab)

    seed_locs = {roi.seed_loc for roi in tab._candidates}
    assert seed_locs == {(7, 7), (22, 22)}
    for roi in tab._candidates:
        assert roi.params is not None
        assert isinstance(roi.params["threshold"], float)


def test_pca_ica_rois_record_the_batch_parameters_used():
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab.n_pca_components_spin.setValue(10)
    tab.n_ica_components_spin.setValue(6)
    tab._on_run_pca_ica_clicked()
    _wait_for_worker(tab)

    assert len(tab._candidates) >= 1
    for roi in tab._candidates:
        assert roi.params == {"n_pca_components": 10, "n_ica_components": 6}
        assert roi.seed_loc is None


def test_cnmf_rois_record_the_batch_parameters_used():
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab.cnmf_n_components_spin.setValue(3)
    tab.cnmf_search_radius_spin.setValue(8)
    tab._on_run_cnmf_clicked()
    _wait_for_worker(tab)

    assert len(tab._candidates) == 3
    for roi in tab._candidates:
        assert roi.params == {"n_components": 3, "search_radius": 8.0, "merge_thresh": tab.cnmf_merge_thresh_spin.value()}


def test_run_graft_adds_candidates_from_both_synthetic_blobs():
    state = AppState()
    movie = _synthetic_movie(n_frames=150)  # GraFT needs more frames than CNMF/PCA-ICA to converge reliably
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("GraFT"))
    # graft_n_dict_spin's own default (20) -- confirmed empirically to
    # converge reliably against this shared, comparatively low-contrast
    # _synthetic_movie fixture; fewer dictionary components sometimes
    # miss one of the two blobs on this particular movie.
    tab._on_run_graft_clicked()
    _wait_for_worker(tab, timeout_ms=30000)

    assert len(tab._candidates) > 0
    assert all(roi.source_method == "graft" for roi in tab._candidates)
    assert all(roi.status == "pending" for roi in tab._candidates)
    assert all(roi.neuropil_trace is not None for roi in tab._candidates)
    assert all(roi.spike_trace is None for roi in tab._candidates)  # GraFT doesn't produce one, unlike CNMF

    centers = [np.argwhere(roi.mask).mean(axis=0) for roi in tab._candidates if roi.mask.any()]
    near_a = any(np.hypot(*(c - (7.5, 7.5))) < 4 for c in centers)
    near_b = any(np.hypot(*(c - (22.5, 22.5))) < 4 for c in centers)
    assert near_a and near_b


def test_run_patch_graft_adds_candidates_from_both_synthetic_blobs():
    state = AppState()
    movie = _synthetic_movie(n_frames=150)
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("GraFT"))
    tab.graft_patch_check.setChecked(True)
    tab.graft_patch_size_spin.setValue(18)
    tab.graft_patch_overlap_spin.setValue(6)
    tab.graft_n_dict_per_patch_spin.setValue(5)
    tab._on_run_graft_clicked()
    _wait_for_worker(tab, timeout_ms=30000)

    assert len(tab._candidates) > 0
    assert all(roi.source_method == "graft" for roi in tab._candidates)
    centers = [np.argwhere(roi.mask).mean(axis=0) for roi in tab._candidates if roi.mask.any()]
    near_a = any(np.hypot(*(c - (7.5, 7.5))) < 4 for c in centers)
    near_b = any(np.hypot(*(c - (22.5, 22.5))) < 4 for c in centers)
    assert near_a and near_b


def test_run_graft_without_data_warns(monkeypatch):
    monkeypatch.setattr("orbitapp.tabs.source_extraction_tab.QMessageBox.warning", lambda *a, **k: None)
    state = AppState()
    tab = SourceExtractionTab(state)

    tab._on_run_graft_clicked()

    assert tab._candidates == []


def test_graft_rois_record_the_batch_parameters_used():
    state = AppState()
    movie = _synthetic_movie(n_frames=150)
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("GraFT"))
    tab.graft_n_dict_spin.setValue(10)
    tab._on_run_graft_clicked()
    _wait_for_worker(tab, timeout_ms=30000)

    assert len(tab._candidates) > 0
    for roi in tab._candidates:
        assert roi.params == {"n_dict": 10}
