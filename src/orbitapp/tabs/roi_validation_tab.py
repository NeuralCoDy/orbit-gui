"""ROI Validation tab: reproduces the `seudo` PyPI package's
ClassifyTransientsWindow GUI as closely as possible, adapted to pull its
input directly from this app's own committed ROIs (state.rois) and active
movie rather than requiring a hand-built SeudoData object.

Layout (matching upstream): a full-height left sidebar (classification
counts, single-cell profile, clickable field-of-view picture,
contamination-severity scatter, sort/title/blur controls, save/load) next
to a center column (time course with zoom/pan, scrollable thumbnail
grid), a right sidebar (auto-classify + Run SEUDO), with the cell
navigation bar along the bottom.

Dropped relative to upstream (infrastructure orbit-gui doesn't need, not
a fidelity gap in the actual SEUDO math -- see orbit.seudo's own module
docstrings): the compiled native accelerator and n_jobs/native-thread
parallelism controls (this tab's "Run SEUDO"/"Auto-classify" workers
already run off the GUI thread via QThread, matching every other
long-running operation in this app), and the MATLAB .mat-classification
interop in favor of orbit-gui's own state.

Like SourceExtractionTab, this doesn't produce a new candidate movie, so
it doesn't subclass StageTab -- there's no Apply/Commit step here at all,
classification edits apply immediately (matching upstream's own
click-to-classify-immediately interaction).
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6 import QtCore
from PySide6.QtGui import QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import binary_dilation, gaussian_filter

from orbit.seudo import SeudoData, auto_classify_transients, classification_color, cycle_classification, pick_color
from orbit.seudo.classification_io import load_classification, save_classification
from orbit.seudo.run_on_transients import run_seudo_restricted_to_transients

from ..state import AppState

_SORT_ORDERS = ("time", "correlation", "residual ratio")
_TITLE_MODES = ("transient id", "correlation", "residual ratio")
_FOV_HIGHLIGHT_COLOR = (0.0, 1.0, 1.0)
_SEUDO_OVERLAY_COLOR = (0.6, 0.1, 0.8)
_PROFILE_BLUR_SIGMA = 1.5  # boolean ROI masks -> continuous profiles, see _build_profiles_from_rois

_AUTO_CLASSIFY_CRITERIA = {"correlation": "corr", "residual ratio": "res_ratio", "SEUDO residual": "seudo_residual"}
_AUTO_CLASSIFY_DEFAULT_THRESH = {"correlation": 0.4, "residual ratio": 0.5, "SEUDO residual": 0.5}


def _build_profiles_from_rois(rois) -> np.ndarray:
    """(H, W, nCells) continuous profile stack from boolean ROI masks.
    Gaussian-blurred rather than used as raw 0/1 masks -- a hard-edged
    binary mask, cropped to its own support, has zero within-support
    pixel variance, which makes the correlation-based classifier's
    correlation undefined (0/0). Confirmed empirically while building
    this port: real/contaminated transients only separate cleanly once
    profiles vary continuously across the ROI."""
    return np.stack([gaussian_filter(roi.mask.astype(float), sigma=_PROFILE_BLUR_SIGMA) for roi in rois], axis=2)


def _scale_seudo_for_display(tc: np.ndarray, seudo_tc: np.ndarray, valid_mask: np.ndarray) -> float:
    if not np.any(valid_mask):
        return 1.0
    seudo_amp = np.max(np.abs(seudo_tc[valid_mask]))
    if seudo_amp <= 1e-12:
        return 1.0
    return np.max(np.abs(tc[valid_mask])) / seudo_amp


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


class _SeudoRunWorker(QtCore.QThread):
    progress = QtCore.Signal(int, int, int)
    frame_progress = QtCore.Signal(int, int, int, int)
    finished_ok = QtCore.Signal(dict)
    failed = QtCore.Signal(str)

    def __init__(self, se, which_struct, which_cells, seudo_params, parent=None):
        super().__init__(parent)
        self.se, self.which_struct, self.which_cells, self.seudo_params = se, which_struct, which_cells, seudo_params

    def run(self) -> None:
        try:
            result = run_seudo_restricted_to_transients(
                self.se, self.which_struct, which_cells=self.which_cells,
                progress_callback=lambda done, total, cc: self.progress.emit(done, total, cc),
                frame_progress_callback=lambda col, cc, done, total: self.frame_progress.emit(col, cc, done, total),
                **self.seudo_params,
            )
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class _AutoClassifyWorker(QtCore.QThread):
    progress = QtCore.Signal(int, int, int)
    finished_ok = QtCore.Signal(list)
    failed = QtCore.Signal(str)

    def __init__(self, se, which_struct, kwargs, parent=None):
        super().__init__(parent)
        self.se, self.which_struct, self.kwargs = se, which_struct, kwargs

    def run(self) -> None:
        try:
            result = auto_classify_transients(
                self.se, self.which_struct, overwrite=True, save_results=True,
                progress_callback=lambda done, total, cc: self.progress.emit(done, total, cc), **self.kwargs,
            )
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ROIValidationTab(QWidget):
    def __init__(self, state: AppState, n_trans_x: int = 5, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.n_trans_x = n_trans_x

        self.se: SeudoData | None = None
        self.which_struct = "default"
        self.this_cell = 0
        self.sort_order = "time"
        self.title_mode = "transient id"
        self.blur_sigma = 0.0
        self._ax_to_trans: dict = {}
        self._trans_to_ax: dict = {}
        self._tc_transient_lines: dict = {}
        self._tc_transient_labels: dict = {}
        self._scatter_points: dict = {}
        self._fov_backdrop_rgb: np.ndarray | None = None
        self._last_click_yx: tuple[int, int] | None = None
        self._last_click_count = 0
        self._tc_xlim: tuple[float, float] | None = None
        self._tc_pan_start: tuple[float, tuple[float, float]] | None = None
        self._pixel_lookup: dict = {}
        self.auto_class: list[dict] = []
        self._seudo_worker: _SeudoRunWorker | None = None
        self._auto_classify_worker: _AutoClassifyWorker | None = None
        self._last_auto_classify_params: dict | None = None  # for session_io.py's pipeline file
        self._last_seudo_params: dict | None = None

        self._build_ui()
        self._set_session_active(False)

    # ---- top-level layout ----

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        load_row = QHBoxLayout()
        self.load_btn = QPushButton("Load ROIs for validation")
        self.load_btn.clicked.connect(self._on_load_rois_clicked)
        load_row.addWidget(self.load_btn)
        self.status_label = QLabel("No ROIs loaded yet -- commit some in Source Extraction first.")
        load_row.addWidget(self.status_label, 1)
        outer.addLayout(load_row)

        main_row = QHBoxLayout()

        left_widget = QWidget()
        left_widget.setLayout(self._build_left_panel())
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_widget)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMaximumWidth(350)
        left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_row.addWidget(left_scroll, 0)

        main_row.addLayout(self._build_center_panel(), 1)

        right_widget = QWidget()
        right_widget.setLayout(self._build_right_sidebar_panel())
        right_scroll = QScrollArea()
        right_scroll.setWidget(right_widget)
        right_scroll.setWidgetResizable(True)
        right_scroll.setMaximumWidth(300)
        right_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_row.addWidget(right_scroll, 0)

        outer.addLayout(main_row, 1)
        outer.addLayout(self._build_cell_nav_bar())

        QShortcut(QtCore.Qt.Key.Key_PageDown, self, self._scroll_thumbnails_page_down)
        QShortcut(QtCore.Qt.Key.Key_PageUp, self, self._scroll_thumbnails_page_up)
        QShortcut(QtCore.Qt.Key.Key_A, self, self._toggle_artifact_shortcut)

    def _build_counts_panel(self) -> QGroupBox:
        group = QGroupBox("Classification counts")
        grid = QGridLayout(group)
        self.count_labels = {}
        keys = ("true", "false", "mixed", "unclassified", "artifact", "non_artifact")
        for i, key in enumerate(keys):
            lbl = QLabel()
            font = lbl.font()
            font.setBold(True)
            lbl.setFont(font)
            self.count_labels[key] = lbl
            grid.addWidget(lbl, i // 2, i % 2)
        return group

    def _build_left_panel(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.addWidget(self._build_counts_panel())

        self.profile_fig = Figure(figsize=(3.0, 2.4), facecolor="black")
        self.profile_canvas = FigureCanvas(self.profile_fig)
        self.profile_canvas.setMinimumSize(280, 220)
        col.addWidget(self.profile_canvas)

        self.fov_fig = Figure(figsize=(3.2, 3.2), facecolor="black")
        self.fov_canvas = FigureCanvas(self.fov_fig)
        self.fov_canvas.setMinimumSize(300, 300)
        self.fov_canvas.mpl_connect("button_press_event", self._on_fov_click)
        col.addWidget(self.fov_canvas)

        self.scatter_fig = Figure(figsize=(2.8, 1.8), facecolor="black")
        self.scatter_canvas = FigureCanvas(self.scatter_fig)
        self.scatter_canvas.setMinimumSize(280, 180)
        col.addWidget(self.scatter_canvas)

        self.artifact_checkbox = QCheckBox("Is artifact")
        self.artifact_checkbox.stateChanged.connect(self._on_artifact_changed)
        col.addWidget(self.artifact_checkbox)

        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("Sort by:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(_SORT_ORDERS)
        self.sort_combo.currentTextChanged.connect(self._on_sort_changed)
        sort_row.addWidget(self.sort_combo)
        col.addLayout(sort_row)

        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("Title:"))
        self.title_combo = QComboBox()
        self.title_combo.addItems(_TITLE_MODES)
        self.title_combo.currentTextChanged.connect(self._on_title_changed)
        title_row.addWidget(self.title_combo)
        col.addLayout(title_row)

        blur_row = QHBoxLayout()
        blur_row.addWidget(QLabel("Blur (px std):"))
        self.blur_spin = QDoubleSpinBox()
        self.blur_spin.setRange(0.0, 20.0)
        self.blur_spin.setSingleStep(0.5)
        self.blur_spin.valueChanged.connect(self._on_blur_changed)
        blur_row.addWidget(self.blur_spin)
        col.addLayout(blur_row)

        save_btn = QPushButton("Save classification...")
        save_btn.clicked.connect(self._on_save)
        col.addWidget(save_btn)
        load_btn = QPushButton("Load classification...")
        load_btn.clicked.connect(self._on_load)
        col.addWidget(load_btn)

        col.addStretch(1)
        self._left_panel_widgets = [
            self.artifact_checkbox, self.sort_combo, self.title_combo, self.blur_spin, save_btn, load_btn,
        ]
        return col

    def _build_right_sidebar_panel(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.addWidget(self._build_auto_classify_panel())
        col.addWidget(self._build_run_seudo_panel())
        col.addStretch(1)
        return col

    def _build_auto_classify_panel(self) -> QGroupBox:
        group = QGroupBox("Auto-classify")
        form = QFormLayout(group)

        self.auto_classify_criterion_combo = QComboBox()
        self.auto_classify_criterion_combo.addItems(list(_AUTO_CLASSIFY_CRITERIA))
        self.auto_classify_criterion_combo.currentTextChanged.connect(self._on_auto_classify_criterion_changed)
        form.addRow("Criteria:", self.auto_classify_criterion_combo)

        self.auto_classify_thresh_spin = QDoubleSpinBox()
        self.auto_classify_thresh_spin.setDecimals(3)
        self.auto_classify_thresh_spin.setRange(-1e6, 1e6)
        self.auto_classify_thresh_spin.setSingleStep(0.05)
        self.auto_classify_thresh_spin.setValue(_AUTO_CLASSIFY_DEFAULT_THRESH["correlation"])
        form.addRow("Threshold:", self.auto_classify_thresh_spin)

        self.auto_classify_btn = QPushButton("Auto-classify (all cells)")
        self.auto_classify_btn.clicked.connect(self._on_auto_classify_clicked)
        form.addRow(self.auto_classify_btn)

        self.auto_classify_progress = QProgressBar()
        self.auto_classify_progress.setRange(0, 1)
        form.addRow(self.auto_classify_progress)

        self.auto_classify_status_label = QLabel("")
        self.auto_classify_status_label.setWordWrap(True)
        form.addRow(self.auto_classify_status_label)

        self._right_panel_widgets = [self.auto_classify_criterion_combo, self.auto_classify_thresh_spin, self.auto_classify_btn]
        return group

    def _build_run_seudo_panel(self) -> QGroupBox:
        group = QGroupBox("Run SEUDO")
        form = QFormLayout(group)

        self.seudo_sigma2_spin = QDoubleSpinBox()
        self.seudo_sigma2_spin.setDecimals(4)
        self.seudo_sigma2_spin.setRange(0.0001, 10.0)
        self.seudo_sigma2_spin.setSingleStep(0.001)
        self.seudo_sigma2_spin.setValue(0.0020)
        form.addRow("sigma2:", self.seudo_sigma2_spin)

        self.seudo_lambda_blob_spin = QDoubleSpinBox()
        self.seudo_lambda_blob_spin.setRange(0.0, 1000.0)
        self.seudo_lambda_blob_spin.setSingleStep(1.0)
        self.seudo_lambda_blob_spin.setValue(10.0)
        form.addRow("lambdaBlob:", self.seudo_lambda_blob_spin)

        self.seudo_blob_radius_spin = QDoubleSpinBox()
        self.seudo_blob_radius_spin.setRange(0.1, 20.0)
        self.seudo_blob_radius_spin.setSingleStep(0.5)
        self.seudo_blob_radius_spin.setValue(3.0)
        form.addRow("blobRadius:", self.seudo_blob_radius_spin)

        self.seudo_ds_time_spin = QSpinBox()
        self.seudo_ds_time_spin.setRange(1, 50)
        self.seudo_ds_time_spin.setValue(3)
        form.addRow("dsTime:", self.seudo_ds_time_spin)

        self.seudo_pad_space_spin = QSpinBox()
        self.seudo_pad_space_spin.setRange(0, 100)
        self.seudo_pad_space_spin.setValue(5)
        form.addRow("padSpace:", self.seudo_pad_space_spin)

        self.run_seudo_one_btn = QPushButton("Run SEUDO (this cell)")
        self.run_seudo_one_btn.clicked.connect(self._run_seudo_this_cell)
        form.addRow(self.run_seudo_one_btn)

        self.run_seudo_all_btn = QPushButton("Run SEUDO (all cells)")
        self.run_seudo_all_btn.clicked.connect(self._run_seudo_all_cells)
        form.addRow(self.run_seudo_all_btn)

        self.seudo_progress = QProgressBar()
        self.seudo_progress.setRange(0, 1)
        form.addRow(self.seudo_progress)

        self.seudo_status_label = QLabel("")
        self.seudo_status_label.setWordWrap(True)
        form.addRow(self.seudo_status_label)

        self._right_panel_widgets += [
            self.seudo_sigma2_spin, self.seudo_lambda_blob_spin, self.seudo_blob_radius_spin,
            self.seudo_ds_time_spin, self.seudo_pad_space_spin, self.run_seudo_one_btn, self.run_seudo_all_btn,
        ]
        return group

    def _build_center_panel(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.addLayout(self._build_time_course_section())

        self.thumb_fig = Figure(facecolor="black")
        self.thumb_canvas = FigureCanvas(self.thumb_fig)
        self.thumb_canvas.mpl_connect("button_press_event", self._on_thumbnail_click)

        self.thumb_scroll = QScrollArea()
        self.thumb_scroll.setWidget(self.thumb_canvas)
        self.thumb_scroll.setWidgetResizable(False)
        self.thumb_scroll.setMinimumSize(400, 300)
        col.addWidget(self.thumb_scroll, 1)
        return col

    def _build_time_course_section(self) -> QVBoxLayout:
        col = QVBoxLayout()
        self.tc_fig = Figure(figsize=(9, 3), facecolor="black")
        self.tc_canvas = FigureCanvas(self.tc_fig)
        self.tc_canvas.setMinimumHeight(220)
        self.tc_canvas.mpl_connect("scroll_event", self._on_tc_scroll)
        self.tc_canvas.mpl_connect("button_press_event", self._on_tc_pan_press)
        self.tc_canvas.mpl_connect("motion_notify_event", self._on_tc_pan_motion)
        self.tc_canvas.mpl_connect("button_release_event", self._on_tc_pan_release)
        col.addWidget(self.tc_canvas)

        tc_controls_row = QHBoxLayout()
        tc_controls_row.addWidget(QLabel("scroll to zoom, drag to pan (horizontal only)"))
        tc_controls_row.addStretch(1)
        reset_zoom_btn = QPushButton("Reset zoom")
        reset_zoom_btn.clicked.connect(self._reset_tc_zoom)
        tc_controls_row.addWidget(reset_zoom_btn)
        col.addLayout(tc_controls_row)
        return col

    def _build_cell_nav_bar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel("Cell:"))

        self.cell_slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.cell_slider.setMinimum(1)
        self.cell_slider.setMaximum(1)
        self.cell_slider.valueChanged.connect(lambda v: self._go_to_cell(v - 1))
        row.addWidget(self.cell_slider, 1)

        self.cell_spin = QSpinBox()
        self.cell_spin.setMinimum(1)
        self.cell_spin.setMaximum(1)
        self.cell_spin.valueChanged.connect(lambda v: self._go_to_cell(v - 1))
        row.addWidget(self.cell_spin)

        prev_btn = QPushButton("◀ prev unclassified")
        prev_btn.clicked.connect(self.go_to_prev_unclassified)
        row.addWidget(prev_btn)
        next_btn = QPushButton("next unclassified ▶")
        next_btn.clicked.connect(self.go_to_next_unclassified)
        row.addWidget(next_btn)

        self._nav_widgets = [self.cell_slider, self.cell_spin, prev_btn, next_btn]
        return row

    def _set_session_active(self, active: bool) -> None:
        for widget in self._left_panel_widgets + self._right_panel_widgets + self._nav_widgets:
            widget.setEnabled(active)

    # ---- loading ROIs from Source Extraction ----

    def on_data_loaded(self) -> None:
        """A new movie loaded or an earlier stage committed a change --
        any in-progress validation session is now stale (state.rois was
        reset by AppState.load(), or the movie/traces it's plotting from
        may no longer match). Doesn't auto-rebuild -- that's an explicit,
        possibly expensive action (see _on_load_rois_clicked)."""
        self.se = None
        self._set_session_active(False)
        n_rois = len(self.state.rois)
        if n_rois == 0:
            self.status_label.setText("No ROIs loaded yet -- commit some in Source Extraction first.")
        else:
            self.status_label.setText(f"{n_rois} committed ROI(s) available. Click 'Load ROIs for validation'.")

    def _on_load_rois_clicked(self) -> None:
        movie = self.state.active_data()
        if movie is None or not self.state.rois:
            QMessageBox.warning(
                self, "No ROIs", "Commit at least one ROI in Source Extraction before validating."
            )
            return

        rois = self.state.rois
        profiles = _build_profiles_from_rois(rois)
        time_courses = np.stack([roi.trace for roi in rois], axis=1)

        self.se = SeudoData(movie, profiles, name=self.state.data_path or "untitled", time_courses=time_courses)
        self.se.compute_transient_info(self.which_struct)
        self.auto_class = auto_classify_transients(self.se, self.which_struct, save_results=False)
        self._pixel_lookup = self._build_pixel_lookup()
        self._fov_backdrop_rgb = None

        self.cell_slider.setMaximum(max(1, self.se.n_cells))
        self.cell_spin.setMaximum(max(1, self.se.n_cells))
        self._set_session_active(True)
        self.status_label.setText(f"Validating {self.se.n_cells} ROI(s), {self.se.mov_f} frames.")
        self._go_to_cell(0)

    def _build_pixel_lookup(self) -> dict:
        mask = self.se.profiles > 0
        dilated = binary_dilation(mask, structure=np.ones((3, 3, 1), dtype=bool))
        lookup = {}
        ys, xs = np.nonzero(np.any(dilated, axis=2))
        for y, x in zip(ys, xs):
            lookup[(int(y), int(x))] = sorted(np.flatnonzero(dilated[y, x, :]).tolist())
        return lookup

    # ---- cell navigation ----

    def _go_to_cell(self, cell_idx: int) -> None:
        if self.se is None:
            return
        cell_idx = int(min(max(0, cell_idx), self.se.n_cells - 1))
        self.this_cell = cell_idx
        self.cell_slider.blockSignals(True)
        self.cell_spin.blockSignals(True)
        self.cell_slider.setValue(cell_idx + 1)
        self.cell_spin.setValue(cell_idx + 1)
        self.cell_slider.blockSignals(False)
        self.cell_spin.blockSignals(False)
        self._tc_xlim = None
        self.refresh()
        self.thumb_scroll.verticalScrollBar().setValue(0)

    def _cell_transient_info(self) -> dict:
        return self._tc_struct()["transient_info"][self.this_cell]

    def _tc_struct(self) -> dict:
        return self.se._resolve_tc_struct(self.which_struct)

    def _scroll_thumbnails_page_down(self) -> None:
        bar = self.thumb_scroll.verticalScrollBar()
        bar.setValue(bar.value() + bar.pageStep())

    def _scroll_thumbnails_page_up(self) -> None:
        bar = self.thumb_scroll.verticalScrollBar()
        bar.setValue(bar.value() - bar.pageStep())

    def _unclassified_cells(self) -> list[int]:
        info = self._tc_struct()["transient_info"]
        return [
            i for i, ti in enumerate(info)
            if ti["classification"].size > 0 and np.all(np.isnan(ti["classification"])) and not ti["is_artifact"]
        ]

    def go_to_next_unclassified(self) -> None:
        candidates = [c for c in self._unclassified_cells() if c > self.this_cell]
        if candidates:
            self._go_to_cell(min(candidates))

    def go_to_prev_unclassified(self) -> None:
        candidates = [c for c in self._unclassified_cells() if c < self.this_cell]
        if candidates:
            self._go_to_cell(max(candidates))

    # ---- classification interaction ----

    def _cell_auto_class(self) -> dict:
        return self.auto_class[self.this_cell]

    def _plot_order(self) -> list[int]:
        ti = self._cell_transient_info()
        n_trans = ti["times"].shape[0]
        metric = self._cell_auto_class()
        if self.sort_order == "correlation" and metric["corrs"].size == n_trans:
            return list(np.argsort(-metric["corrs"]))
        if self.sort_order == "residual ratio" and metric["resRatios"].size == n_trans:
            return list(np.argsort(metric["resRatios"]))
        return list(range(n_trans))

    def _on_thumbnail_click(self, event) -> None:
        if event.inaxes not in self._ax_to_trans:
            return
        trans_idx = self._ax_to_trans[event.inaxes]
        ti = self._cell_transient_info()
        ti["classification"][trans_idx] = cycle_classification(ti["classification"][trans_idx])
        self._update_transient_colors([trans_idx])

    def _on_fov_click(self, event) -> None:
        if event.xdata is None or event.ydata is None or self.se is None:
            return
        x = int(min(max(0, round(event.xdata)), self.se.mov_x - 1))
        y = int(min(max(0, round(event.ydata)), self.se.mov_y - 1))

        cell_list = self._pixel_lookup.get((y, x), [])
        if not cell_list:
            return

        if self._last_click_yx == (y, x):
            self._last_click_count = (self._last_click_count % len(cell_list)) + 1
        else:
            self._last_click_yx = (y, x)
            self._last_click_count = 1

        new_cell = cell_list[self._last_click_count - 1]
        if new_cell != self.this_cell:
            self._go_to_cell(new_cell)

    def _on_artifact_changed(self, _state: int) -> None:
        ti = self._cell_transient_info()
        ti["is_artifact"] = self.artifact_checkbox.isChecked()
        self._update_transient_colors(range(ti["times"].shape[0]))

    def _toggle_artifact_shortcut(self) -> None:
        self.artifact_checkbox.setChecked(not self.artifact_checkbox.isChecked())

    def _on_sort_changed(self, text: str) -> None:
        self.sort_order = text
        self._draw_thumbnails()

    def _on_title_changed(self, text: str) -> None:
        self.title_mode = text
        self._draw_thumbnails()

    def _on_blur_changed(self, value: float) -> None:
        self.blur_sigma = value
        self._draw_thumbnails()

    def _on_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save classification")
        if path:
            save_classification(self._tc_struct(), path)

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load classification")
        if not path:
            return
        payload = load_classification(path)
        self._tc_struct()["transient_info"] = payload["transient_info"]
        self.auto_class = auto_classify_transients(self.se, self.which_struct, save_results=False)
        self.refresh()

    # ---- auto-classify ----

    def _on_auto_classify_criterion_changed(self, criterion_label: str) -> None:
        self.auto_classify_thresh_spin.setValue(_AUTO_CLASSIFY_DEFAULT_THRESH[criterion_label])

    def _on_auto_classify_clicked(self) -> None:
        criterion_label = self.auto_classify_criterion_combo.currentText()
        criterion = _AUTO_CLASSIFY_CRITERIA[criterion_label]
        threshold = self.auto_classify_thresh_spin.value()

        reply = QMessageBox.question(
            self, "Auto-classify",
            f"This will overwrite the current classification for every transient in every ROI, using "
            f"{criterion_label} vs. threshold {threshold:g}. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        kwargs: dict = dict(criterion=criterion)
        if criterion == "corr":
            kwargs["corr_thresh"] = threshold
        elif criterion == "res_ratio":
            kwargs["res_ratio_thresh"] = threshold
        else:
            kwargs["seudo_residual_thresh"] = threshold
            kwargs["seudo_kwargs"] = dict(
                sigma2=self.seudo_sigma2_spin.value(), lambda_blob=self.seudo_lambda_blob_spin.value(),
                blob_radius=self.seudo_blob_radius_spin.value(),
            )

        if self._auto_classify_worker is not None and self._auto_classify_worker.isRunning():
            return

        self._last_auto_classify_params = dict(criterion_label=criterion_label, threshold=threshold)
        self.auto_classify_btn.setEnabled(False)
        self.auto_classify_progress.setRange(0, self.se.n_cells)
        self.auto_classify_progress.setValue(0)
        self.auto_classify_status_label.setText(f"Auto-classifying {self.se.n_cells} ROI(s)...")

        worker = _AutoClassifyWorker(self.se, self.which_struct, kwargs, self)
        worker.progress.connect(self._on_auto_classify_progress)
        worker.finished_ok.connect(self._on_auto_classify_finished)
        worker.failed.connect(self._on_auto_classify_failed)
        self._auto_classify_worker = worker
        worker.start()

    def _on_auto_classify_progress(self, done: int, total: int, cell_id: int) -> None:
        self.auto_classify_progress.setValue(done)
        self.auto_classify_status_label.setText(f"Auto-classify: {done}/{total} ROI(s) done (last: ROI {cell_id + 1})")

    def _on_auto_classify_finished(self, results: list) -> None:
        self.auto_classify_btn.setEnabled(True)
        self.auto_classify_status_label.setText(f"Auto-classify done ({len(results)} ROI(s)).")
        self.refresh()

    def _on_auto_classify_failed(self, message: str) -> None:
        self.auto_classify_btn.setEnabled(True)
        self.auto_classify_status_label.setText(f"Auto-classify failed: {message}")

    # ---- running SEUDO ----

    def _collect_seudo_params(self) -> dict:
        return dict(
            sigma2=self.seudo_sigma2_spin.value(), lambda_blob=self.seudo_lambda_blob_spin.value(),
            blob_radius=self.seudo_blob_radius_spin.value(), ds_time=self.seudo_ds_time_spin.value(),
            pad_space=self.seudo_pad_space_spin.value(),
        )

    _SEUDO_PROGRESS_RESOLUTION = 1000

    def _run_seudo(self, which_cells: list[int]) -> None:
        if self._seudo_worker is not None and self._seudo_worker.isRunning():
            return

        params = self._collect_seudo_params()
        self._last_seudo_params = dict(params)
        self.run_seudo_one_btn.setEnabled(False)
        self.run_seudo_all_btn.setEnabled(False)
        self.seudo_progress.setRange(0, len(which_cells) * self._SEUDO_PROGRESS_RESOLUTION)
        self.seudo_progress.setValue(0)
        self.seudo_status_label.setText(f"Running SEUDO on {len(which_cells)} ROI(s)...")

        worker = _SeudoRunWorker(self.se, self.which_struct, which_cells, params, self)
        worker.progress.connect(self._on_seudo_progress)
        worker.frame_progress.connect(self._on_seudo_frame_progress)
        worker.finished_ok.connect(self._on_seudo_finished)
        worker.failed.connect(self._on_seudo_failed)
        self._seudo_worker = worker
        worker.start()

    def _run_seudo_this_cell(self) -> None:
        self._run_seudo([self.this_cell])

    def _run_seudo_all_cells(self) -> None:
        self._run_seudo(list(range(self.se.n_cells)))

    def _on_seudo_progress(self, done: int, total: int, cell_idx: int) -> None:
        self.seudo_progress.setValue(done * self._SEUDO_PROGRESS_RESOLUTION)
        self.seudo_status_label.setText(f"SEUDO: {done}/{total} ROI(s) done (last: ROI {cell_idx + 1})")

    def _on_seudo_frame_progress(self, col: int, cell_idx: int, frames_done: int, total_frames: int) -> None:
        if total_frames <= 0:
            return
        within_cell = int(self._SEUDO_PROGRESS_RESOLUTION * frames_done / total_frames)
        self.seudo_progress.setValue(col * self._SEUDO_PROGRESS_RESOLUTION + within_cell)
        self.seudo_status_label.setText(f"SEUDO: ROI {cell_idx + 1}, frame {frames_done}/{total_frames}")

    def _on_seudo_finished(self, result: dict) -> None:
        self.run_seudo_one_btn.setEnabled(True)
        self.run_seudo_all_btn.setEnabled(True)
        self.seudo_status_label.setText(f"SEUDO done ({result['tc'].shape[1]} ROI(s)).")
        self._draw_time_course()

    def _on_seudo_failed(self, message: str) -> None:
        self.run_seudo_one_btn.setEnabled(True)
        self.run_seudo_all_btn.setEnabled(True)
        self.seudo_status_label.setText(f"SEUDO failed: {message}")

    def _latest_seudo_tc_for_cell(self, cell_idx: int) -> np.ndarray | None:
        if not self.se.tc_seudo:
            return None
        result = self.se.tc_seudo[-1]
        which_cells = result.get("params", {}).get("which_cells")
        if which_cells is None or cell_idx not in which_cells:
            return None
        col = list(which_cells).index(cell_idx)
        tc = result["tc"][:, col]
        return None if np.all(np.isnan(tc)) else tc

    # ---- drawing ----

    def refresh(self) -> None:
        ti = self._cell_transient_info()
        self.artifact_checkbox.blockSignals(True)
        self.artifact_checkbox.setChecked(bool(ti["is_artifact"]))
        self.artifact_checkbox.blockSignals(False)

        self._draw_profile()
        self._draw_fov()
        self._draw_time_course()
        self._draw_scatter()
        self._draw_thumbnails()
        self._update_counts()

    def _update_transient_colors(self, trans_indices) -> None:
        ti = self._cell_transient_info()
        touched_thumb = touched_tc = touched_scatter = False

        for trans_idx in trans_indices:
            color = pick_color("artifact") if ti["is_artifact"] else classification_color(ti["classification"][trans_idx])

            ax = self._trans_to_ax.get(trans_idx)
            if ax is not None:
                for spine in ax.spines.values():
                    spine.set_color(color)
                touched_thumb = True

            line = self._tc_transient_lines.get(trans_idx)
            if line is not None:
                line.set_color(color)
                touched_tc = True

            label = self._tc_transient_labels.get(trans_idx)
            if label is not None:
                label.set_color(color)
                touched_tc = True

            point = self._scatter_points.get(trans_idx)
            if point is not None:
                point.set_color(color)
                touched_scatter = True

        if touched_thumb:
            self.thumb_canvas.draw_idle()
        if touched_tc:
            self.tc_canvas.draw_idle()
        if touched_scatter:
            self.scatter_canvas.draw_idle()
        self._update_counts()

    def _style_dark_axes(self, ax) -> None:
        ax.set_facecolor("black")
        for spine in ax.spines.values():
            spine.set_color("#61afef")
        ax.tick_params(colors="#61afef")
        ax.xaxis.label.set_color("#61afef")
        ax.yaxis.label.set_color("#61afef")
        ax.title.set_color("#61afef")

    def _draw_profile(self) -> None:
        ti = self._cell_transient_info()
        y0, y1, x0, x1 = ti["window"]
        prof = self.se.profiles[y0 : y1 + 1, x0 : x1 + 1, self.this_cell]

        self.profile_fig.clf()
        ax = self.profile_fig.add_subplot(111)
        ax.imshow(prof, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"ROI {self.this_cell + 1} profile", fontsize=9)
        self._style_dark_axes(ax)
        self.profile_canvas.draw_idle()

    def _fov_backdrop(self) -> np.ndarray:
        if self._fov_backdrop_rgb is None:
            profiles = self.se.profiles.astype(float)
            per_cell_max = profiles.max(axis=(0, 1), keepdims=True)
            per_cell_max = np.where(per_cell_max > 0, per_cell_max, 1.0)
            prof_max = (profiles / per_cell_max).max(axis=2)
            peak = prof_max.max()
            if peak > 0:
                prof_max = prof_max / peak
            self._fov_backdrop_rgb = np.stack([prof_max] * 3, axis=-1)
        return self._fov_backdrop_rgb

    def _draw_fov(self) -> None:
        backdrop = self._fov_backdrop()
        this_profile = self.se.profiles[:, :, self.this_cell].astype(float)
        peak = this_profile.max()
        if peak > 0:
            this_profile = this_profile / peak
        highlight = np.stack([this_profile * 0, this_profile * 1.0, this_profile * 0.5], axis=-1)

        im = np.clip(1 - (1 - backdrop * 0.7) * (1 - highlight), 0, 1)

        self.fov_fig.clf()
        ax = self.fov_fig.add_subplot(111)
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"all {self.se.n_cells} ROIs", fontsize=10)
        self._style_dark_axes(ax)

        ti = self._cell_transient_info()
        y0, y1, x0, x1 = ti["window"]
        ax.plot(
            [x0 - 0.5, x1 + 0.5, x1 + 0.5, x0 - 0.5, x0 - 0.5], [y0 - 0.5, y0 - 0.5, y1 + 0.5, y1 + 0.5, y0 - 0.5],
            "-", linewidth=1.5, color=_FOV_HIGHLIGHT_COLOR,
        )
        self.fov_canvas.draw_idle()

    def _draw_time_course(self) -> None:
        ti = self._cell_transient_info()
        tc = self._tc_struct()["tc"][:, self.this_cell]
        self.tc_fig.clf()
        ax = self.tc_fig.add_subplot(111)
        ax.plot(tc, color=(0.5, 0.5, 0.5), linewidth=0.5)

        self._tc_transient_lines = {}
        self._tc_transient_labels = {}
        for tt in range(ti["times"].shape[0]):
            s, e = ti["times"][tt]
            color = classification_color(ti["classification"][tt])
            (line,) = ax.plot(range(s, e + 1), tc[s : e + 1], color=color)
            self._tc_transient_lines[tt] = line

            peak_local = int(np.argmax(tc[s : e + 1]))
            peak_frame = s + peak_local
            label = ax.annotate(
                str(tt + 1), xy=(peak_frame, tc[peak_frame]), xytext=(0, 3), textcoords="offset points",
                fontsize=7, ha="center", va="bottom", color=color,
            )
            self._tc_transient_labels[tt] = label

        seudo_tc = self._latest_seudo_tc_for_cell(self.this_cell)
        if seudo_tc is not None:
            valid = ~np.isnan(seudo_tc)
            scale = _scale_seudo_for_display(tc, seudo_tc, valid)
            seudo_tc_display = seudo_tc * scale
            label = "SEUDO" if np.isclose(scale, 1.0) else f"SEUDO (×{scale:0.2f})"
            first = True
            for s, e in _contiguous_runs(valid):
                ax.plot(
                    range(s, e + 1), seudo_tc_display[s : e + 1], "-", color=_SEUDO_OVERLAY_COLOR, linewidth=1.5,
                    label=label if first else None,
                )
                first = False
            legend = ax.legend(fontsize=7, loc="upper right")
            legend.get_frame().set_facecolor("black")
            for text in legend.get_texts():
                text.set_color("#61afef")

        ax.set_title(f"ROI {self.this_cell + 1}", fontsize=10)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        self._style_dark_axes(ax)

        if self._tc_xlim is not None:
            ax.set_xlim(*self._tc_xlim)
        else:
            ax.set_xlim(0, len(tc))
            self._tc_xlim = (0, len(tc))

        self.tc_fig.tight_layout()
        self.tc_canvas.draw_idle()

    # ---- time-course horizontal zoom/pan ----

    def _tc_axes(self):
        return self.tc_fig.axes[0] if self.tc_fig.axes else None

    def _on_tc_scroll(self, event) -> None:
        ax = self._tc_axes()
        if ax is None or event.inaxes != ax or event.xdata is None or self.se is None:
            return
        n_frames = self.se.mov_f
        cur_left, cur_right = ax.get_xlim()
        width = cur_right - cur_left
        scale = 0.8 if event.button == "up" else 1.25
        new_width = max(5.0, min(width * scale, float(n_frames)))
        rel = (event.xdata - cur_left) / width if width else 0.5
        new_left = event.xdata - rel * new_width
        new_right = new_left + new_width
        if new_left < 0:
            new_left, new_right = 0.0, new_width
        elif new_right > n_frames:
            new_right = float(n_frames)
            new_left = new_right - new_width
        self._tc_xlim = (new_left, new_right)
        ax.set_xlim(new_left, new_right)
        self.tc_canvas.draw_idle()

    def _on_tc_pan_press(self, event) -> None:
        ax = self._tc_axes()
        if ax is None or event.inaxes != ax or event.button != 1:
            return
        self._tc_pan_start = (event.xdata, ax.get_xlim())

    def _on_tc_pan_motion(self, event) -> None:
        if self._tc_pan_start is None or event.xdata is None or self.se is None:
            return
        ax = self._tc_axes()
        if ax is None:
            return
        start_xdata, start_xlim = self._tc_pan_start
        n_frames = self.se.mov_f
        width = min(start_xlim[1] - start_xlim[0], float(n_frames))
        dx = start_xdata - event.xdata
        new_left = start_xlim[0] + dx
        new_right = new_left + width
        if new_left < 0:
            new_left, new_right = 0.0, width
        elif new_right > n_frames:
            new_right = float(n_frames)
            new_left = new_right - width
        self._tc_xlim = (new_left, new_right)
        ax.set_xlim(new_left, new_right)
        self.tc_canvas.draw_idle()

    def _on_tc_pan_release(self, _event) -> None:
        self._tc_pan_start = None

    def _reset_tc_zoom(self) -> None:
        self._tc_xlim = None
        if self.se is not None:
            self._draw_time_course()

    def _draw_scatter(self) -> None:
        ti = self._cell_transient_info()
        metric = self._cell_auto_class()
        self.scatter_fig.clf()
        ax = self.scatter_fig.add_subplot(111)

        self._scatter_points = {}
        n_trans = ti["times"].shape[0]
        if n_trans > 0 and metric["corrs"].size == n_trans and metric["resRatios"].size == n_trans:
            for tt in range(n_trans):
                (point,) = ax.plot(
                    metric["corrs"][tt], metric["resRatios"][tt], "o",
                    color=classification_color(ti["classification"][tt]), markersize=4,
                )
                self._scatter_points[tt] = point
        ax.set_xlim(-1, 1)
        ax.set_xlabel("correlation", fontsize=8)
        ax.set_ylabel("contam. severity", fontsize=8)
        ax.tick_params(labelsize=7)
        self._style_dark_axes(ax)
        self.scatter_fig.tight_layout()
        self.scatter_canvas.draw_idle()

    def _thumbnail_title(self, trans_idx: int) -> str:
        metric = self._cell_auto_class()
        if self.title_mode == "correlation" and metric["corrs"].size > trans_idx:
            return f"{metric['corrs'][trans_idx]:0.3f}"
        if self.title_mode == "residual ratio" and metric["resRatios"].size > trans_idx:
            return f"{metric['resRatios'][trans_idx]:0.3f}"
        return str(trans_idx + 1)

    def _draw_thumbnails(self) -> None:
        ti = self._cell_transient_info()
        self.thumb_fig.clf()
        self._ax_to_trans = {}
        self._trans_to_ax = {}

        n_trans = ti["times"].shape[0]
        if n_trans == 0:
            self.thumb_fig.set_size_inches(4, 3)
            self.thumb_canvas.setFixedSize(400, 300)
            ax = self.thumb_fig.add_subplot(111)
            ax.text(0.5, 0.5, "This ROI has no transients", ha="center", va="center", color="#61afef")
            ax.axis("off")
            self.thumb_canvas.draw_idle()
            return

        plot_order = self._plot_order()
        n_cols = self.n_trans_x
        n_rows = -(-n_trans // n_cols)

        cell_px = 150
        dpi = 100
        self.thumb_fig.set_dpi(dpi)
        self.thumb_fig.set_size_inches(n_cols * cell_px / dpi, n_rows * cell_px / dpi)
        self.thumb_canvas.setFixedSize(n_cols * cell_px, n_rows * cell_px)

        axes = self.thumb_fig.subplots(n_rows, n_cols, squeeze=False)
        for i, ax in enumerate(axes.flat):
            if i >= n_trans:
                ax.axis("off")
                continue

            trans_idx = plot_order[i]
            shape = ti["shapes"][:, :, trans_idx]
            if self.blur_sigma > 0:
                shape = gaussian_filter(shape, sigma=self.blur_sigma)
            ax.imshow(shape, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])

            color = pick_color("artifact") if ti["is_artifact"] else classification_color(ti["classification"][trans_idx])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(3)

            ax.set_title(self._thumbnail_title(trans_idx), fontsize=8, color="#61afef")
            self._ax_to_trans[ax] = trans_idx
            self._trans_to_ax[trans_idx] = ax

        total_height_px = n_rows * cell_px
        top = 1 - 15 / total_height_px
        bottom = 5 / total_height_px
        self.thumb_fig.subplots_adjust(left=0.02, right=0.98, top=top, bottom=bottom, wspace=0.15, hspace=0.6)
        self.thumb_canvas.draw_idle()

    def _update_counts(self) -> None:
        info = self._tc_struct()["transient_info"]
        is_artifact = np.array([ti["is_artifact"] for ti in info])
        class_vals = (
            np.concatenate([ti["classification"] for ti, art in zip(info, is_artifact) if not art])
            if np.any(~is_artifact) else np.array([])
        )

        from orbit.seudo import constants as C

        n_true = int(np.sum(class_vals == C.VAL_TRUE))
        n_false = int(np.sum(class_vals == C.VAL_FALSE))
        n_unc = int(np.sum(np.isnan(class_vals)))
        n_mix = int(class_vals.size - n_true - n_false - n_unc)
        n_art = int(np.sum(is_artifact))
        n_non_art = int(np.sum(~is_artifact))

        counts = {
            "true": (n_true, pick_color("true")), "false": (n_false, pick_color("false")),
            "mixed": (n_mix, pick_color("mixed")), "unclassified": (n_unc, pick_color("unclassified")),
            "artifact": (n_art, pick_color("artifact")), "non_artifact": (n_non_art, (1, 1, 1)),
        }
        labels = {
            "true": "True", "false": "False", "mixed": "Mixed", "unclassified": "Unclassified",
            "artifact": "Artifact", "non_artifact": "Non-artifact",
        }
        for key, (count, color) in counts.items():
            r, g, b = (int(255 * c) for c in color)
            self.count_labels[key].setText(f"{labels[key]}: {count}")
            self.count_labels[key].setStyleSheet(f"color: rgb({r},{g},{b})")

    # ---- session save/load (see orbitapp.session_io) ----

    def export_pipeline_params(self) -> dict:
        """Auto-classify/Run-SEUDO parameters last used in this session --
        the "recipe" half, always worth recording even without a full
        save. Empty entries if that action was never taken."""
        return dict(
            auto_classify=self._last_auto_classify_params or {}, run_seudo=self._last_seudo_params or {},
        )

    def export_results(self) -> list[dict] | None:
        """Per-ROI transient times/classification/is_artifact -- the
        actual derived results, only meaningful (and only returned) once
        a validation session has been built. Order matches state.rois."""
        if self.se is None:
            return None
        return [
            {"times": ti["times"], "classification": ti["classification"], "is_artifact": bool(ti["is_artifact"])}
            for ti in self._tc_struct()["transient_info"]
        ]

    def import_results(self, saved: list[dict]) -> None:
        """Restores previously-saved classification/is_artifact onto the
        current session's freshly-detected transients -- reuses
        compute_transient_info's own "preserve classification from an
        existing transient_info" merge logic (exact-match times carry
        over directly; split/merged ones get resolved or left
        unclassified) by seeding it with the loaded data first."""
        if self.se is None or len(saved) != self.se.n_cells:
            return
        self._tc_struct()["transient_info"] = saved
        self.se.compute_transient_info(self.which_struct)
        self.auto_class = auto_classify_transients(self.se, self.which_struct, save_results=False)
        self.refresh()
