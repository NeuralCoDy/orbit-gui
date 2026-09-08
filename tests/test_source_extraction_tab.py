import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from orbit.cnmf import CNMFResult  # noqa: E402
from orbit.roi_extraction_graft import GraFTResult  # noqa: E402
from orbit.roi_extraction_pca_ica import PCAICAResult, pca_ica_source_extraction  # noqa: E402
from orbit.roi_extraction_realseudo import RealSeudoResult  # noqa: E402
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


def _calcium_trace(n_frames, rng, rate=0.03, decay=0.9, amp=4.0):
    # A smooth exponential-decay transient, not per-frame i.i.d. noise --
    # CNMF-E's corr*PNR seeding needs actual temporal structure to read
    # a "cell" as different from noise (unlike _synthetic_movie's plain
    # per-frame-random bump, which CNMF's own intensity-peak seeding
    # doesn't care about).
    spikes = (rng.random(n_frames) < rate).astype(float) * rng.uniform(1, 2, n_frames)
    trace = np.zeros(n_frames)
    for t in range(1, n_frames):
        trace[t] = decay * trace[t - 1] + spikes[t]
    return amp * trace


def _synthetic_1p_movie(height=40, width=40, n_frames=200, seed=0):
    rng = np.random.default_rng(seed)
    movie = rng.standard_normal((height, width, n_frames)).astype(np.float64) * 0.05 + 1.0
    movie[8:13, 8:13, :] += _calcium_trace(n_frames, rng)
    movie[28:33, 28:33, :] += _calcium_trace(n_frames, rng)
    return np.clip(movie, 0, None)


def _set_cnmf_e_params(tab, n_components=4):
    tab.cnmf_e_n_components_spin.setValue(n_components)
    tab.cnmf_e_search_radius_spin.setValue(8)
    tab.cnmf_e_min_corr_spin.setValue(0.8)
    tab.cnmf_e_min_pnr_spin.setValue(8.0)
    tab.cnmf_e_ring_inner_radius_spin.setValue(6)
    tab.cnmf_e_ring_outer_radius_spin.setValue(10)
    tab.cnmf_e_ring_downsample_spin.setValue(2)
    tab.cnmf_e_ring_max_fit_frames_spin.setValue(150)


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
    # correlation click-to-add is not a Method option
    assert labels == ["PCA-ICA", "CNMF", "CNMF-E", "GraFT", "Real-SEUDO"]


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


def test_method_combo_includes_cnmf_e():
    state = AppState()
    tab = SourceExtractionTab(state)

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("CNMF-E"))
    assert not tab.params_dialog.form.isRowVisible(tab.cnmf_n_components_spin)
    assert tab.params_dialog.form.isRowVisible(tab.cnmf_e_n_components_spin)


def test_cnmf_e_patch_rows_only_visible_for_cnmf_e_method_and_when_checked():
    state = AppState()
    tab = SourceExtractionTab(state)

    assert not tab.params_dialog.form.isRowVisible(tab.cnmf_e_patch_size_spin)

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("CNMF-E"))
    assert tab.params_dialog.form.isRowVisible(tab.cnmf_e_patch_check)
    assert not tab.params_dialog.form.isRowVisible(tab.cnmf_e_patch_size_spin)  # checkbox still unchecked

    tab.cnmf_e_patch_check.setChecked(True)
    assert tab.params_dialog.form.isRowVisible(tab.cnmf_e_patch_size_spin)
    assert tab.params_dialog.form.isRowVisible(tab.cnmf_e_patch_overlap_spin)
    assert tab.params_dialog.form.isRowVisible(tab.cnmf_e_components_per_patch_spin)

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("PCA-ICA"))
    assert not tab.params_dialog.form.isRowVisible(tab.cnmf_e_patch_size_spin)
    tab.method_combo.setCurrentIndex(tab.method_combo.findText("CNMF-E"))
    assert tab.params_dialog.form.isRowVisible(tab.cnmf_e_patch_size_spin)


def test_modality_warning_hidden_by_default():
    state = AppState()
    tab = SourceExtractionTab(state)
    assert tab.modality_warning_label.isHidden()


def test_modality_warning_for_cnmf_e_with_2p_somatic():
    state = AppState()
    tab = SourceExtractionTab(state)

    state.somatic_2p = True
    tab.method_combo.setCurrentText("CNMF-E")

    assert not tab.modality_warning_label.isHidden()
    assert "CNMF-E" in tab.modality_warning_label.text()
    assert "1P" in tab.modality_warning_label.text()


def test_modality_warning_absent_for_cnmf_e_with_1p_somatic():
    # The correct pairing shouldn't warn.
    state = AppState()
    tab = SourceExtractionTab(state)

    state.somatic_1p = True
    tab.on_modality_changed()
    tab.method_combo.setCurrentText("CNMF-E")

    assert tab.modality_warning_label.isHidden()


@pytest.mark.parametrize("method_label", ["CNMF", "PCA-ICA"])
def test_modality_warning_for_cnmf_or_pca_ica_with_1p_somatic(method_label):
    state = AppState()
    tab = SourceExtractionTab(state)

    state.somatic_1p = True
    tab.on_modality_changed()  # simulates the Load tab checkbox toggle's signal chain
    tab.method_combo.setCurrentText(method_label)
    tab._update_modality_warning()  # setCurrentText is a no-op (and fires no signal) if already selected

    assert not tab.modality_warning_label.isHidden()
    assert method_label in tab.modality_warning_label.text()
    assert "1P-Somatic" in tab.modality_warning_label.text()


@pytest.mark.parametrize("method_label", ["PCA-ICA", "CNMF", "CNMF-E"])
@pytest.mark.parametrize("modality_field", ["dendrites", "axons", "widefield"])
def test_modality_warning_for_dendritic_axonal_widefield_with_non_graft_method(method_label, modality_field):
    state = AppState()
    tab = SourceExtractionTab(state)

    setattr(state, modality_field, True)
    tab.on_modality_changed()  # simulates the Load tab checkbox toggle's signal chain
    tab.method_combo.setCurrentText(method_label)
    tab._update_modality_warning()  # setCurrentText is a no-op (and fires no signal) if already selected

    assert not tab.modality_warning_label.isHidden()
    assert "GraFT" in tab.modality_warning_label.text()


@pytest.mark.parametrize("modality_field", ["dendrites", "axons", "widefield"])
def test_modality_warning_absent_for_graft_with_dendritic_axonal_widefield(modality_field):
    state = AppState()
    tab = SourceExtractionTab(state)

    setattr(state, modality_field, True)
    tab.on_modality_changed()
    tab.method_combo.setCurrentText("GraFT")

    assert tab.modality_warning_label.isHidden()


def test_modality_warning_combines_multiple_applicable_messages():
    state = AppState()
    tab = SourceExtractionTab(state)

    state.somatic_1p = True
    state.dendrites = True
    tab.on_modality_changed()
    tab.method_combo.setCurrentText("CNMF")

    text = tab.modality_warning_label.text()
    assert "1P-Somatic" in text
    assert "GraFT" in text


def test_modality_warning_updates_live_when_load_tab_toggle_changes():
    state = AppState()
    tab = SourceExtractionTab(state)
    tab.method_combo.setCurrentText("CNMF-E")
    assert tab.modality_warning_label.isHidden()

    state.somatic_2p = True
    tab.on_modality_changed()

    assert not tab.modality_warning_label.isHidden()

    state.somatic_2p = False
    tab.on_modality_changed()

    assert tab.modality_warning_label.isHidden()


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


def test_commit_records_the_run_s_params_and_a_deleted_count_metric():
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
    del tab._candidates[1]  # simulates the review panel's Delete Selected/Delete All Rejected

    tab._commit()

    step = state.steps[-1]
    assert step.label == "Correlation ROIs"
    assert step.params  # the run's own correlation params, not empty
    assert step.metrics == {"rois_committed": 1, "rois_deleted": 1}


def test_commit_does_not_count_a_still_pending_candidate_as_deleted():
    state = AppState()
    movie = _synthetic_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab._grow_from_seeds(movie, [(7, 7), (22, 22)])
    _wait_for_worker(tab)
    tab._candidates[0].status = "accepted"
    # tab._candidates[1] stays "pending" -- left for later review, not deleted

    tab._commit()

    step = state.steps[-1]
    assert step.metrics == {"rois_committed": 1, "rois_deleted": 0}
    assert len(tab._candidates) == 1  # the pending one is still around


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


def test_run_cnmf_e_adds_candidates_from_both_synthetic_blobs():
    state = AppState()
    movie = _synthetic_1p_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("CNMF-E"))
    _set_cnmf_e_params(tab)
    tab._on_run_cnmf_e_clicked()
    _wait_for_worker(tab)

    assert len(tab._candidates) > 0
    assert all(roi.source_method == "cnmf_e" for roi in tab._candidates)
    assert all(roi.status == "pending" for roi in tab._candidates)
    assert all(roi.neuropil_trace is not None for roi in tab._candidates)
    assert all(roi.spike_trace is not None for roi in tab._candidates)

    centers = [np.argwhere(roi.mask).mean(axis=0) for roi in tab._candidates if roi.mask.any()]
    near_a = any(np.hypot(*(c - (10, 10))) < 5 for c in centers)
    near_b = any(np.hypot(*(c - (30, 30))) < 5 for c in centers)
    assert near_a and near_b


def test_run_patch_cnmf_e_adds_candidates_from_both_synthetic_blobs():
    state = AppState()
    movie = _synthetic_1p_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("CNMF-E"))
    _set_cnmf_e_params(tab)
    tab.cnmf_e_patch_check.setChecked(True)
    tab.cnmf_e_patch_size_spin.setValue(25)
    tab.cnmf_e_patch_overlap_spin.setValue(12)
    tab.cnmf_e_components_per_patch_spin.setValue(2)
    tab._on_run_cnmf_e_clicked()
    _wait_for_worker(tab)

    assert len(tab._candidates) > 0
    assert all(roi.source_method == "cnmf_e" for roi in tab._candidates)
    assert all(roi.spike_trace is not None for roi in tab._candidates)


def test_run_patch_cnmf_e_records_patch_params_on_each_roi():
    state = AppState()
    movie = _synthetic_1p_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("CNMF-E"))
    _set_cnmf_e_params(tab)
    tab.cnmf_e_patch_check.setChecked(True)
    tab.cnmf_e_patch_size_spin.setValue(25)
    tab.cnmf_e_patch_overlap_spin.setValue(12)
    tab.cnmf_e_components_per_patch_spin.setValue(2)
    tab._on_run_cnmf_e_clicked()
    _wait_for_worker(tab)

    assert len(tab._candidates) > 0
    for roi in tab._candidates:
        assert roi.params["patch_size"] == (25, 25)
        assert roi.params["overlap"] == 12
        assert roi.params["n_components_per_patch"] == 2


def test_run_cnmf_e_without_data_warns(monkeypatch):
    monkeypatch.setattr("orbitapp.tabs.source_extraction_tab.QMessageBox.warning", lambda *a, **k: None)
    state = AppState()
    tab = SourceExtractionTab(state)

    tab._on_run_cnmf_e_clicked()

    assert tab._candidates == []


def test_run_cnmf_without_data_warns(monkeypatch):
    monkeypatch.setattr("orbitapp.tabs.source_extraction_tab.QMessageBox.warning", lambda *a, **k: None)
    state = AppState()
    tab = SourceExtractionTab(state)

    tab._on_run_cnmf_clicked()

    assert tab._candidates == []


def _pca_ica_fingerprint(tab, movie):
    kwargs = dict(n_pca_components=tab.n_pca_components_spin.value(), n_ica_components=tab.n_ica_components_spin.value())
    # trailing None matches _run_batch_method's extra_fingerprint default
    # (a normal, non-test run) -- see run_test_btn's n_frames_limit.
    return (id(movie), pca_ica_source_extraction, tuple(sorted(kwargs.items())), None)


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
    # unlike other stages' history, the undone source-extraction block is
    # removed too -- it no longer has any committed ROIs to describe
    assert state.pipeline == ["Load"]
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


def _graft_friendly_movie(height=30, width=30, n_frames=150, seed=0):
    """A higher-contrast synthetic movie than the shared _synthetic_movie
    above -- GraFT's random dictionary initialization makes it more
    sensitive to blob contrast than CNMF/PCA-ICA are (confirmed
    empirically: the shared, lower-contrast fixture only converges to
    both blobs in ~75% of unseeded runs at the UI's default parameters,
    even with regularization all the way down at 0 -- this one is
    reliable in 8/8 trials at the UI's actual (heavier) default
    regularization values, matching what orbit.roi_extraction_graft's
    own tests already use)."""
    rng = np.random.default_rng(seed)
    movie = rng.standard_normal((height, width, n_frames)).astype(np.float64) * 0.1 + 1.0
    movie[5:10, 5:10, :] += 3 * np.clip(rng.standard_normal(n_frames), 0, None)
    movie[20:25, 20:25, :] += 3 * np.clip(rng.standard_normal(n_frames), 0, None)
    return np.clip(movie, 0, None)


def test_run_graft_adds_candidates_from_both_synthetic_blobs():
    state = AppState()
    movie = _graft_friendly_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("GraFT"))
    # graft_n_dict_spin's own default (20), and every other GraFT param
    # left at the UI's own defaults -- this movie profile converges
    # reliably at those actual defaults (see _graft_friendly_movie).
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
    tab.graft_lambda_spin.setValue(0.5)
    tab.graft_lam_forb_spin.setValue(0.2)
    tab.graft_lam_corr_spin.setValue(0.1)
    tab.graft_lam_cont_spin.setValue(0.3)
    tab.graft_learn_eps_spin.setValue(0.005)
    tab._on_run_graft_clicked()
    _wait_for_worker(tab, timeout_ms=30000)

    assert len(tab._candidates) > 0
    for roi in tab._candidates:
        assert roi.params == {
            "n_dict": 10, "lambda": 0.5, "lamForb": 0.2, "lamCorr": 0.1, "lamCont": 0.3, "learn_eps": 0.005,
        }


# -- Volumetric (state.volumetric) path -----------------------------------


def _synthetic_volumetric_movie(length=20, width=20, depth=10, n_frames=200, seed=0):
    rng = np.random.default_rng(seed)
    movie = rng.standard_normal((n_frames, length, width, depth)).astype(np.float64) * 0.1 + 1.0

    blob1 = np.zeros((length, width, depth))
    blob1[3:7, 3:7, 2:5] = 1.0
    blob2 = np.zeros((length, width, depth))
    blob2[12:16, 12:16, 5:8] = 1.0
    act1 = np.clip(rng.standard_normal(n_frames), 0, None) * 3
    act2 = np.clip(rng.standard_normal(n_frames), 0, None) * 3
    for t in range(n_frames):
        movie[t] += blob1 * act1[t] + blob2 * act2[t]

    mask = np.zeros((length, width, depth), dtype=bool)
    mask[0 : length // 2, 0 : width // 2, :] = True
    mask[length // 2 :, width // 2 :, :] = True
    return movie, mask


def test_on_modality_changed_restricts_method_combo_to_graft_when_volumetric():
    state = AppState()
    tab = SourceExtractionTab(state)
    tab.method_combo.setCurrentText("CNMF")

    state.volumetric = True
    tab.on_modality_changed()

    assert tab.method_combo.currentText() == "GraFT"
    model = tab.method_combo.model()
    assert not model.item(0).isEnabled()  # PCA-ICA
    assert not model.item(1).isEnabled()  # CNMF
    assert not model.item(2).isEnabled()  # CNMF-E
    assert model.item(3).isEnabled()  # GraFT
    assert not tab.auto_seed_btn.isEnabled()

    state.volumetric = False
    tab.on_modality_changed()
    assert model.item(0).isEnabled()
    assert model.item(1).isEnabled()
    assert tab.auto_seed_btn.isEnabled()


def test_run_graft_volumetric_without_a_mask_warns_and_does_not_run(monkeypatch):
    monkeypatch.setattr("orbitapp.tabs.source_extraction_tab.QMessageBox.warning", lambda *a, **k: None)
    state = AppState()
    state.volumetric = True
    movie, _mask = _synthetic_volumetric_movie(n_frames=20)
    state.load("movie.fits", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()

    assert state.mask is None
    tab._on_run_graft_clicked()

    assert tab._candidates == []


def test_run_graft_volumetric_with_an_all_true_mask_warns_and_does_not_run(monkeypatch):
    monkeypatch.setattr("orbitapp.tabs.source_extraction_tab.QMessageBox.warning", lambda *a, **k: None)
    state = AppState()
    state.volumetric = True
    movie, mask = _synthetic_volumetric_movie(n_frames=20)
    state.load("movie.fits", movie)
    state.mask = np.ones_like(mask)  # an explicit Clear Mask -- doesn't restrict anything
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()

    tab._on_run_graft_clicked()

    assert tab._candidates == []


def test_run_graft_volumetric_produces_true_3d_rois_and_commits():
    state = AppState()
    state.volumetric = True
    movie, mask = _synthetic_volumetric_movie()
    state.load("movie.fits", movie)
    state.mask = mask
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()

    tab.graft_n_dict_spin.setValue(6)
    tab._on_run_graft_clicked()
    _wait_for_worker(tab, timeout_ms=30000)

    assert len(tab._candidates) > 0
    assert all(roi.source_method == "graft" for roi in tab._candidates)
    assert all(roi.mask.shape == mask.shape for roi in tab._candidates)  # true (L, W, D), not depth-projected
    assert all(roi.neuropil_trace is None for roi in tab._candidates)  # neuropil skipped for volumetric

    for roi in tab._candidates:
        roi.status = "accepted"
    tab._commit()

    assert len(state.rois) > 0
    assert all(roi.mask.ndim == 3 for roi in state.rois)


def test_run_patch_graft_volumetric_produces_true_3d_rois():
    state = AppState()
    state.volumetric = True
    movie, mask = _synthetic_volumetric_movie()
    state.load("movie.fits", movie)
    state.mask = mask
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()

    tab.graft_patch_check.setChecked(True)
    tab.graft_patch_size_spin.setValue(12)
    tab.graft_patch_overlap_spin.setValue(3)
    tab.graft_n_dict_per_patch_spin.setValue(4)
    tab._on_run_graft_clicked()
    _wait_for_worker(tab, timeout_ms=30000)

    assert len(tab._candidates) > 0
    assert all(roi.source_method == "graft" for roi in tab._candidates)
    assert all(roi.mask.shape == mask.shape for roi in tab._candidates)


def test_review_panel_does_not_crash_on_3d_rois():
    # ROIReviewPanel's rendering is inherently 2D -- confirms it depth-
    # projects rather than erroring when fed a real 3D-masked ROI.
    state = AppState()
    state.volumetric = True
    movie, mask = _synthetic_volumetric_movie()
    state.load("movie.fits", movie)
    state.mask = mask
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()

    tab.graft_n_dict_spin.setValue(6)
    tab._on_run_graft_clicked()
    _wait_for_worker(tab, timeout_ms=30000)

    assert len(tab._candidates) > 0  # rendered via _sync_candidates -> review_panel.set_candidates without crashing


def test_turning_volumetric_off_leaves_2d_source_extraction_unaffected():
    state = AppState()
    movie = _graft_friendly_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("GraFT"))
    tab._on_run_graft_clicked()
    _wait_for_worker(tab, timeout_ms=30000)

    assert len(tab._candidates) > 0
    assert all(roi.source_method == "graft" for roi in tab._candidates)
    assert all(roi.neuropil_trace is not None for roi in tab._candidates)


def _single_blob_onset_movie(height=30, width=30, n_frames=40, onset=5, amplitude=5.0, center=(15, 15), seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    blob = np.exp(-((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / (2 * 2.0 ** 2))
    blob /= blob.max()
    activity = np.zeros(n_frames)
    activity[onset:] = amplitude
    movie = np.zeros((height, width, n_frames))
    for t in range(n_frames):
        movie[:, :, t] = blob * activity[t] + rng.normal(scale=0.05, size=(height, width))
    return movie


def _set_lenient_real_seudo_params(tab):
    """Small/lenient enough to reliably discover a single synthetic blob
    within a short test movie -- the wrapper's own defaults are tuned for
    real, much longer recordings."""
    tab.real_seudo_min_roi_size_spin.setValue(5)
    tab.real_seudo_consecutive_frames_spin.setValue(3)
    tab.real_seudo_lookahead_frames_spin.setValue(1)


def test_method_combo_includes_real_seudo():
    state = AppState()
    tab = SourceExtractionTab(state)

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("Real-SEUDO"))
    assert not tab.params_dialog.form.isRowVisible(tab.graft_n_dict_spin)
    assert tab.params_dialog.form.isRowVisible(tab.real_seudo_sigma2_spin)
    assert tab.action_stack.currentWidget() is tab.action_stack.widget(4)  # 5th method, 0-indexed


def test_run_real_seudo_adds_a_candidate_at_the_synthetic_blob():
    state = AppState()
    movie = _single_blob_onset_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("Real-SEUDO"))
    _set_lenient_real_seudo_params(tab)
    tab._on_run_real_seudo_clicked()
    _wait_for_worker(tab, timeout_ms=30000)

    assert len(tab._candidates) >= 1
    assert all(roi.source_method == "real_seudo" for roi in tab._candidates)
    assert all(roi.status == "pending" for roi in tab._candidates)
    assert all(roi.spike_trace is None for roi in tab._candidates)  # Real-SEUDO doesn't produce one, unlike CNMF

    centers = [np.argwhere(roi.mask).mean(axis=0) for roi in tab._candidates if roi.mask.any()]
    assert any(np.hypot(*(c - (15, 15))) < 4 for c in centers)


def test_run_real_seudo_records_the_batch_parameters_used():
    state = AppState()
    movie = _single_blob_onset_movie()
    state.load("movie.tif", movie)
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("Real-SEUDO"))
    _set_lenient_real_seudo_params(tab)
    tab._on_run_real_seudo_clicked()
    _wait_for_worker(tab, timeout_ms=30000)

    assert len(tab._candidates) >= 1
    assert tab._candidates[0].params["min_roi_size"] == 5
    assert tab._candidates[0].params["consecutive_frames_required"] == 3


def test_run_real_seudo_without_data_warns(monkeypatch):
    monkeypatch.setattr("orbitapp.tabs.source_extraction_tab.QMessageBox.warning", lambda *a, **k: None)
    state = AppState()
    tab = SourceExtractionTab(state)

    tab._on_run_real_seudo_clicked()

    assert tab._candidates == []


def test_on_modality_changed_disables_real_seudo_when_volumetric():
    state = AppState()
    state.volumetric = True
    tab = SourceExtractionTab(state)

    idx = tab.method_combo.findText("Real-SEUDO")
    assert not tab.method_combo.model().item(idx).isEnabled()


# --- "Run <N>-frame test" button ---------------------------------------
# _TEST_N_FRAMES is monkeypatched down to a small value throughout so these
# tests don't need a genuinely 10000+-frame synthetic movie to exercise the
# gray-out/enabled boundary or an actual capped run.

def test_run_test_btn_is_disabled_with_no_data_loaded():
    state = AppState()
    tab = SourceExtractionTab(state)
    assert not tab.run_test_btn.isEnabled()


def test_run_test_btn_disabled_when_movie_has_fewer_frames_than_threshold(monkeypatch):
    monkeypatch.setattr("orbitapp.tabs.source_extraction_tab._TEST_N_FRAMES", 50)
    state = AppState()
    state.load("movie.tif", _synthetic_movie(n_frames=49))
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    assert not tab.run_test_btn.isEnabled()


def test_run_test_btn_enabled_when_movie_has_at_least_threshold_frames(monkeypatch):
    monkeypatch.setattr("orbitapp.tabs.source_extraction_tab._TEST_N_FRAMES", 50)
    state = AppState()
    state.load("movie.tif", _synthetic_movie(n_frames=50))
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    assert tab.run_test_btn.isEnabled()


def test_run_test_btn_reflects_a_movie_loaded_after_construction(monkeypatch):
    # on_modality_changed (not just on_data_loaded) also refreshes the
    # button, since either can flip which movie/modality is active.
    monkeypatch.setattr("orbitapp.tabs.source_extraction_tab._TEST_N_FRAMES", 50)
    state = AppState()
    tab = SourceExtractionTab(state)
    assert not tab.run_test_btn.isEnabled()

    state.load("movie.tif", _synthetic_movie(n_frames=50))
    tab.on_data_loaded()
    _wait_for_worker(tab)
    assert tab.run_test_btn.isEnabled()

    tab.on_modality_changed()
    assert tab.run_test_btn.isEnabled()


def test_on_run_test_clicked_dispatches_to_whichever_method_is_selected(monkeypatch):
    state = AppState()
    state.load("movie.tif", _synthetic_movie())
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    calls = []
    for name in ("pca_ica", "cnmf", "cnmf_e", "graft", "real_seudo"):
        monkeypatch.setattr(
            tab, f"_on_run_{name}_clicked", lambda n_frames_limit=None, name=name: calls.append((name, n_frames_limit))
        )

    for label, key in (
        ("PCA-ICA", "pca_ica"), ("CNMF", "cnmf"), ("CNMF-E", "cnmf_e"),
        ("GraFT", "graft"), ("Real-SEUDO", "real_seudo"),
    ):
        tab.method_combo.setCurrentIndex(tab.method_combo.findText(label))
        tab._on_run_test_clicked()

    from orbitapp.tabs.source_extraction_tab import _TEST_N_FRAMES
    assert calls == [
        ("pca_ica", _TEST_N_FRAMES), ("cnmf", _TEST_N_FRAMES), ("cnmf_e", _TEST_N_FRAMES),
        ("graft", _TEST_N_FRAMES), ("real_seudo", _TEST_N_FRAMES),
    ]


def test_clicking_a_run_button_for_real_uses_the_full_movie_not_zero_frames(monkeypatch):
    # Regression test for a real bug: QPushButton.clicked always emits its
    # own `checked: bool` argument. _add_run_action used to connect it
    # straight to slot (btn.clicked.connect(slot)), and every
    # _on_run_*_clicked has an optional n_frames_limit *first* parameter
    # (for run_test_btn's use) -- so a REAL click bound n_frames_limit=False
    # (Qt's checked state), and since False is not None, the code took the
    # "capped" branch and sliced movie[:, :, :False] == 0 frames. Every
    # other test in this file calls the slot directly in Python
    # (tab._on_run_graft_clicked()), which always correctly defaults
    # n_frames_limit to None -- only an actual button.click() (a real Qt
    # signal emission, matching what a user's mouse click produces)
    # reproduces the bug, which is why it went undetected until a user hit
    # it live. Covers all five buttons since they share _add_run_action.
    n_frames = 60
    state = AppState()
    state.load("movie.tif", _synthetic_movie(n_frames=n_frames))
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)

    seen_shapes = {}

    def _stub(name, result):
        def _fn(movie, **kw):
            seen_shapes[name] = movie.shape
            return result
        return _fn

    monkeypatch.setattr(
        "orbitapp.tabs.source_extraction_tab.pca_ica_source_extraction",
        _stub("pca_ica", PCAICAResult(masks=[], traces=[])),
    )
    monkeypatch.setattr(
        "orbitapp.tabs.source_extraction_tab.cnmf_source_extraction",
        _stub("cnmf", CNMFResult(masks=[], traces=[], spike_traces=[])),
    )
    monkeypatch.setattr(
        "orbitapp.tabs.source_extraction_tab.cnmf_e_source_extraction",
        _stub("cnmf_e", CNMFResult(masks=[], traces=[], spike_traces=[])),
    )
    monkeypatch.setattr(
        "orbitapp.tabs.source_extraction_tab.graft_source_extraction",
        _stub("graft", GraFTResult(masks=[], traces=[])),
    )
    monkeypatch.setattr(
        "orbitapp.tabs.source_extraction_tab.real_seudo_source_extraction",
        _stub("real_seudo", RealSeudoResult(masks=[], traces=[])),
    )

    for btn_name in (
        "run_pca_ica_btn", "run_cnmf_btn", "run_cnmf_e_btn", "run_graft_btn", "run_real_seudo_btn",
    ):
        getattr(tab, btn_name).click()  # a REAL Qt signal emission, unlike calling _on_run_*_clicked() directly
        _wait_for_worker(tab)

    assert seen_shapes == {
        "pca_ica": (30, 30, n_frames), "cnmf": (30, 30, n_frames), "cnmf_e": (30, 30, n_frames),
        "graft": (30, 30, n_frames), "real_seudo": (30, 30, n_frames),
    }


def test_run_test_btn_caps_pca_ica_to_the_requested_frame_count(monkeypatch):
    monkeypatch.setattr("orbitapp.tabs.source_extraction_tab._TEST_N_FRAMES", 20)
    state = AppState()
    state.load("movie.tif", _synthetic_movie(n_frames=60))
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)
    # Component counts must fit within the 20-frame test cap below (the
    # spinboxes' own defaults, 50/40, exceed it and would make sklearn
    # raise -- see _set_lenient_real_seudo_params's own precedent).
    tab.n_pca_components_spin.setValue(10)
    tab.n_ica_components_spin.setValue(6)

    seen_shapes = []
    real_fn = pca_ica_source_extraction
    monkeypatch.setattr(
        "orbitapp.tabs.source_extraction_tab.pca_ica_source_extraction",
        lambda movie, **kw: (seen_shapes.append(movie.shape), real_fn(movie, **kw))[1],
    )

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("PCA-ICA"))
    tab._on_run_test_clicked()
    _wait_for_worker(tab)

    assert seen_shapes == [(30, 30, 20)]  # capped, not the full 60-frame movie


def test_run_test_btn_and_a_full_run_are_not_treated_as_the_same_prior_run(monkeypatch):
    # A test run and a full run share every algorithm kwarg, but must not
    # be mistaken for "already ran with these exact parameters" against
    # each other -- extra_fingerprint (the n_frames_limit) keeps them apart.
    monkeypatch.setattr("orbitapp.tabs.source_extraction_tab._TEST_N_FRAMES", 20)
    state = AppState()
    state.load("movie.tif", _synthetic_movie(n_frames=60))
    tab = SourceExtractionTab(state)
    tab.on_data_loaded()
    _wait_for_worker(tab)
    # Must fit within the 20-frame test cap used below, not just the
    # 60-frame full run (see the sibling test above for why).
    tab.n_pca_components_spin.setValue(10)
    tab.n_ica_components_spin.setValue(6)

    prompted = []
    monkeypatch.setattr(
        "orbitapp.tabs.source_extraction_tab.confirm_recompute", lambda *a, **k: prompted.append(1) or False
    )

    tab.method_combo.setCurrentIndex(tab.method_combo.findText("PCA-ICA"))
    tab._on_run_pca_ica_clicked()  # full run
    _wait_for_worker(tab)
    tab._on_run_test_clicked()  # test run, same params -- must not be seen as a repeat of the full run
    _wait_for_worker(tab)

    assert prompted == []
