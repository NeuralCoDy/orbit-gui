import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QLabel, QSpinBox  # noqa: E402

from orbitapp.state import ROI  # noqa: E402
from orbitapp.widgets import (  # noqa: E402
    BusyBar,
    CommitControls,
    HeaderBar,
    ImageSlideshow,
    ParametersDialog,
    QCPlotGrid,
    ROIReviewPanel,
    StagePanel,
    make_spinbox,
    split_by_kind,
)


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def test_header_bar_shows_no_data_by_default():
    header = HeaderBar()
    assert "No data loaded" in header.data_label.text()
    assert header.stage_label.text() == ""


def test_header_bar_reflects_loaded_data_and_active_stage():
    header = HeaderBar()
    header.set_data_info("movie.tif  --  64 x 64, 100 time-steps")
    header.set_active_stage("Motion Correction")

    assert "movie.tif" in header.data_label.text()
    assert header.stage_label.text() == "Stage: Motion Correction"


def test_header_bar_reset_to_no_data():
    header = HeaderBar()
    header.set_data_info("movie.tif  --  64 x 64, 100 time-steps")
    header.set_data_info(None)
    assert "No data loaded" in header.data_label.text()


def test_stage_panel_has_before_after_image_views():
    panel = StagePanel(before_title="Raw", after_title="Corrected")
    assert panel.before_view is not None
    assert panel.after_view is not None

    movie_frame = np.zeros((10, 10))
    panel.before_view.setImage(movie_frame)
    panel.after_view.setImage(movie_frame)


def test_stage_panel_play_movie_is_a_noop_without_a_movie_set():
    panel = StagePanel()
    panel._play_movie("before")  # no movie set yet -- must not raise or create a player
    assert panel._players["before"] is None


def test_stage_panel_play_movie_opens_a_player_for_the_right_panel():
    panel = StagePanel(before_title="Raw", after_title="Corrected")
    before_movie = np.zeros((5, 5, 3))
    after_movie = np.ones((5, 5, 3))
    panel.set_before_movie(before_movie)
    panel.set_after_movie(after_movie)

    panel._play_movie("before")
    assert panel._players["before"] is not None
    assert panel._players["after"] is None
    assert panel._players["before"].windowTitle() == "Movie Player - Raw"

    panel._play_movie("after")
    assert panel._players["after"].windowTitle() == "Movie Player - Corrected"


def test_stage_panel_play_movie_caps_a_memmapped_movie_to_5000_frames(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    from orbitapp.io import load_movie

    movie_path = tmp_path / "movie.tif"
    tifffile.imwrite(movie_path, np.zeros((5200, 4, 4), dtype="uint16"))  # (T, H, W)
    movie = load_movie(movie_path, mmap=True)
    assert movie.shape[-1] == 5200

    panel = StagePanel()
    panel.set_before_movie(movie)
    panel._play_movie("before")

    assert panel._players["before"].state.movie.shape[-1] == 5000


def test_stage_panel_metrics_row_accepts_widgets():
    panel = StagePanel()
    assert panel.metrics_row.count() == 0

    panel.add_metric_widget(QLabel("mMD: -12.3"))
    panel.add_metric_widget(QLabel("ECC: 0.87"))

    assert panel.metrics_row.count() == 2


def test_header_bar_pipeline_breadcrumb():
    header = HeaderBar()
    assert header.pipeline_label.text() == ""

    header.set_pipeline(["Load", "Patch Warp", "Normalize"])
    assert header.pipeline_label.text() == "Pipeline: Load > Patch Warp > Normalize"

    header.set_pipeline([])
    assert header.pipeline_label.text() == ""


def test_busy_bar_starts_hidden_and_toggles_on_start_stop():
    bar = BusyBar()
    assert bar.bar.isVisibleTo(bar) is False

    bar.start("Loading...")
    assert bar.label.text() == "Loading..."
    assert bar.bar.isVisibleTo(bar) is True
    assert bar.bar.value() == 0

    bar.stop("Done.")
    assert bar.label.text() == "Done."
    assert bar.bar.isVisibleTo(bar) is False


def test_busy_bar_set_message_updates_text_without_resetting_progress():
    bar = BusyBar()
    bar.start("Loading...")
    bar._tick()
    bar._tick()
    value_before = bar.bar.value()

    bar.set_message("Rendering...")

    assert bar.label.text() == "Rendering..."
    assert bar.bar.value() == value_before  # unchanged -- still the same continuous operation
    assert bar.bar.isVisibleTo(bar) is True


def test_busy_bar_fills_over_time_but_never_reaches_the_ceiling_while_running():
    bar = BusyBar()
    bar.start("Working...")

    values = []
    for _ in range(10):
        bar._tick()
        values.append(bar.bar.value())

    assert values == sorted(values)  # monotonically non-decreasing
    assert values[-1] > values[0]  # it actually moved
    assert values[-1] < 92  # never reaches (let alone exceeds) the ceiling while still running

    bar.stop("Done.")
    assert bar.bar.value() == 100


def test_commit_controls_starts_with_commit_disabled():
    controls = CommitControls(apply_label="Apply Motion Correction")
    assert controls.apply_btn.isEnabled()
    assert not controls.commit_btn.isEnabled()


def test_commit_controls_enable_disable():
    controls = CommitControls(apply_label="Apply")
    controls.set_commit_enabled(True)
    assert controls.commit_btn.isEnabled()

    controls.set_apply_enabled(False)
    assert not controls.apply_btn.isEnabled()


def test_commit_controls_emit_signals_on_click():
    controls = CommitControls(apply_label="Apply")
    controls.set_commit_enabled(True)

    apply_calls = []
    commit_calls = []
    controls.apply_clicked.connect(lambda: apply_calls.append(1))
    controls.commit_clicked.connect(lambda: commit_calls.append(1))

    controls.apply_btn.click()
    controls.commit_btn.click()

    assert apply_calls == [1]
    assert commit_calls == [1]


def test_parameters_dialog_hosts_added_widgets():
    dialog = ParametersDialog(title="Test Parameters")
    assert dialog.windowTitle() == "Test Parameters"

    spin = QDoubleSpinBox()
    spin.setValue(42)
    dialog.add_row("some_param", spin)

    assert dialog.form.rowCount() == 1
    # the widget is still fully usable/queryable after being added to the dialog
    assert spin.value() == 42


def test_parameters_dialog_group_visibility():
    dialog = ParametersDialog()
    wavelet_field = QDoubleSpinBox()
    gaussian_field = QDoubleSpinBox()
    dialog.add_row("wavelet param", wavelet_field, group="wavelet")
    dialog.add_row("gaussian param", gaussian_field, group="gaussian")

    dialog.set_group_visible("gaussian", False)
    assert dialog.form.isRowVisible(wavelet_field)
    assert not dialog.form.isRowVisible(gaussian_field)


def test_parameters_dialog_show_only_group():
    dialog = ParametersDialog()
    wavelet_field = QDoubleSpinBox()
    gaussian_field = QDoubleSpinBox()
    median_field = QDoubleSpinBox()
    dialog.add_row("wavelet param", wavelet_field, group="wavelet")
    dialog.add_row("gaussian param", gaussian_field, group="gaussian")
    dialog.add_row("median param", median_field, group="median")

    dialog.show_only_group("median")

    assert not dialog.form.isRowVisible(wavelet_field)
    assert not dialog.form.isRowVisible(gaussian_field)
    assert dialog.form.isRowVisible(median_field)

    dialog.show_only_group("wavelet")
    assert dialog.form.isRowVisible(wavelet_field)
    assert not dialog.form.isRowVisible(gaussian_field)
    assert not dialog.form.isRowVisible(median_field)


def test_parameters_dialog_ungrouped_rows_are_unaffected():
    dialog = ParametersDialog()
    ungrouped_field = QDoubleSpinBox()
    grouped_field = QDoubleSpinBox()
    dialog.add_row("always visible", ungrouped_field)
    dialog.add_row("grouped", grouped_field, group="wavelet")

    dialog.show_only_group("median")

    assert dialog.form.isRowVisible(ungrouped_field)


def test_make_spinbox_integer():
    box = make_spinbox(1, 200, 20)
    assert isinstance(box, QSpinBox)
    assert (box.minimum(), box.maximum(), box.value()) == (1, 200, 20)


def test_make_spinbox_decimal_with_step():
    box = make_spinbox(0.0, 0.5, 0.1, step=0.05, decimal=True)
    assert isinstance(box, QDoubleSpinBox)
    assert (box.minimum(), box.maximum(), box.value(), box.singleStep()) == (0.0, 0.5, 0.1, 0.05)


def test_image_slideshow_empty_by_default():
    show = ImageSlideshow()
    assert show.index_label.text() == "0 / 0"


def test_image_slideshow_set_stack_shows_first_frame():
    show = ImageSlideshow()
    stack = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    show.set_stack(stack)
    assert show.index_label.text() == "1 / 5"


def test_image_slideshow_next_prev_wrap_around():
    show = ImageSlideshow()
    show.set_stack(np.zeros((2, 2, 3)))

    show._next()
    show._next()
    assert show.index_label.text() == "3 / 3"
    show._next()
    assert show.index_label.text() == "1 / 3"  # wrapped past the end

    show._prev()
    assert show.index_label.text() == "3 / 3"  # wrapped past the start


def test_image_slideshow_view_large_is_a_noop_without_a_stack():
    show = ImageSlideshow()
    show._view_large()  # no stack set yet -- must not raise or create a player
    assert show._player is None


def test_image_slideshow_view_large_opens_the_full_stack_in_a_player():
    show = ImageSlideshow()
    stack = np.zeros((5, 5, 4))
    show.set_stack(stack)

    show._view_large()

    assert show._player is not None
    assert show._player.windowTitle() == "PC Stack Viewer"
    assert show._player.state.total_frames == 4


def test_image_slideshow_set_stack_clears():
    show = ImageSlideshow()
    show.set_stack(np.zeros((2, 2, 4)))
    assert show.index_label.text() == "1 / 4"

    show.set_stack(None)
    assert show.index_label.text() == "0 / 0"


def test_split_by_kind_separates_peak_and_low_samples():
    samples = [{"kind": "peak", "id": 1}, {"kind": "low", "id": 2}, {"kind": "peak", "id": 3}]
    peaks, lows = split_by_kind(samples)
    assert [s["id"] for s in peaks] == [1, 3]
    assert [s["id"] for s in lows] == [2]


def test_qc_plot_grid_one_plot_per_sample_fills_left_and_right_columns():
    grid = QCPlotGrid("Signal", "Noise", n_rows=2, plots_per_sample=1)
    assert len(grid.left_rows) == 2
    assert all(len(plots) == 1 for plots in grid.left_rows + grid.right_rows)

    calls = []
    grid.fill([{"id": "p1"}, {"id": "p2"}], [{"id": "l1"}], lambda plots, sample: calls.append((len(plots), sample["id"])))
    assert calls == [(1, "p1"), (1, "p2"), (1, "l1")]


def test_qc_plot_grid_multiple_plots_per_sample():
    grid = QCPlotGrid("Signal", "Noise", n_rows=1, plots_per_sample=2, sub_labels=["Before", "After"])
    assert len(grid.left_rows[0]) == 2
    assert len(grid.right_rows[0]) == 2

    calls = []
    grid.fill([{"id": "p1"}], [{"id": "l1"}], lambda plots, sample: calls.append((len(plots), sample["id"])))
    assert calls == [(2, "p1"), (2, "l1")]


def _make_roi(roi_id, status="pending"):
    mask = np.zeros((5, 5), dtype=bool)
    mask[roi_id, roi_id] = True
    return ROI(id=roi_id, mask=mask, trace=np.arange(10, dtype=float), source_method="correlation", status=status)


def test_roi_review_panel_set_candidates_populates_table():
    panel = ROIReviewPanel()
    panel.set_candidates([_make_roi(0), _make_roi(1)])

    assert panel.table.rowCount() == 2
    assert panel.table.item(0, 0).text() == "0"
    assert panel.table.item(1, 2).text() == "pending"


def test_roi_review_panel_accept_all_updates_status_and_emits():
    panel = ROIReviewPanel()
    rois = [_make_roi(0), _make_roi(1)]
    panel.set_candidates(rois)

    changes = []
    panel.roi_status_changed.connect(lambda roi_id, status: changes.append((roi_id, status)))
    panel._set_all_status("accepted")

    assert all(roi.status == "accepted" for roi in rois)
    assert changes == [(0, "accepted"), (1, "accepted")]
    assert panel.table.item(0, 2).text() == "accepted"


def test_roi_review_panel_selection_plots_trace_and_emits_selected():
    panel = ROIReviewPanel()
    panel.set_candidates([_make_roi(0), _make_roi(1)])

    selected = []
    panel.roi_selected.connect(selected.append)
    panel.table.selectRow(1)

    assert selected == [1]
    assert len(panel.trace_plot.getPlotItem().listDataItems()) == 1
    assert len(panel.diff_plot.getPlotItem().listDataItems()) == 0  # no neuropil_trace on this fixture


def test_roi_review_panel_selection_plots_trace_minus_neuropil_in_diff_plot():
    panel = ROIReviewPanel()
    roi = _make_roi(0)
    roi.neuropil_trace = roi.trace - 1.0
    panel.set_candidates([roi])

    panel.table.selectRow(0)

    diff_items = panel.diff_plot.getPlotItem().listDataItems()
    assert len(diff_items) == 1
    assert np.allclose(diff_items[0].yData, roi.trace - roi.neuropil_trace)


def test_roi_review_panel_set_preview_roi_highlights_fov_panel():
    panel = ROIReviewPanel()
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    preview = ROI(id=-1, mask=mask, trace=np.arange(10, dtype=float), source_method="correlation", status="preview")

    panel.set_preview_roi(preview)
    assert panel._highlight_item.image is not None
    assert panel.table.rowCount() == 1  # listed as "ROI -1"

    panel.set_preview_roi(None)
    assert panel._highlight_item.image is None
    assert panel.table.rowCount() == 0


def test_roi_review_panel_selecting_a_roi_highlights_it_in_the_fov_panel():
    # The merged FOV/Selected-ROI panel highlights whichever ROI is
    # selected -- from the table or the Current ROIs view -- and clears
    # once it's gone from the collection.
    panel = ROIReviewPanel()
    panel.set_candidates([_make_roi(0), _make_roi(1)])

    panel.table.selectRow(1)
    assert panel._highlight_item.image is not None

    panel._delete_selected()
    assert panel._highlight_item.image is None


def test_roi_review_panel_find_roi_at_pixel_matches_the_covering_mask():
    panel = ROIReviewPanel()
    panel.set_candidates([_make_roi(0), _make_roi(1)])

    assert panel._find_roi_at_pixel(0, 0) == 0
    assert panel._find_roi_at_pixel(1, 1) == 1
    assert panel._find_roi_at_pixel(4, 4) is None


def test_roi_review_panel_delete_selected_removes_from_collection_and_emits():
    panel = ROIReviewPanel()
    rois = [_make_roi(0), _make_roi(1)]
    panel.set_candidates(rois)

    deleted = []
    panel.roi_deleted.connect(deleted.append)
    panel.table.selectRow(0)
    panel._delete_selected()

    assert deleted == [0]
    assert len(rois) == 1  # panel._rois is the same list object -- mutated in place
    assert rois[0].id == 1
    assert panel.table.rowCount() == 1
