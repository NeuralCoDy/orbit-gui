import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QLabel, QSpinBox  # noqa: E402

from orbitapp.widgets import (  # noqa: E402
    BusyBar,
    CommitControls,
    HeaderBar,
    ParametersDialog,
    StagePanel,
    make_spinbox,
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


def test_make_spinbox_integer():
    box = make_spinbox(1, 200, 20)
    assert isinstance(box, QSpinBox)
    assert (box.minimum(), box.maximum(), box.value()) == (1, 200, 20)


def test_make_spinbox_decimal_with_step():
    box = make_spinbox(0.0, 0.5, 0.1, step=0.05, decimal=True)
    assert isinstance(box, QDoubleSpinBox)
    assert (box.minimum(), box.maximum(), box.value(), box.singleStep()) == (0.0, 0.5, 0.1, 0.05)
