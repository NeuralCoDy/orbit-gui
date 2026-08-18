"""Motion Correction tab: rigid, piecewise-rigid, or PatchWarp-style
piecewise-affine registration (orbit.motion_correction.motion_correct),
with quality metrics computed and shown alongside the before/after
images rather than as an afterthought -- mMD/mCM/ECC (PatchWarp) plus
singular-value-spectrum tightening and spatial PC maps (halo/crescent
inspection).

Registration + metrics run together in one background FunctionWorker
(tens of seconds on a real recording) behind a busy indicator, so the
GUI thread never blocks. "Apply" only ever produces a *candidate*
result previewed in this tab -- the shared active dataset (and hence
every other tab) is untouched until "Commit to Active Dataset" is
clicked explicitly, at which point the step is also recorded in the
header's pipeline breadcrumb.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from orbit.motion_correction import motion_correct
from orbit.motion_metrics import (
    enhanced_correlation_coefficient,
    mean_correlation_to_reference,
    mean_max_intensity_difference,
    spatiotemporal_svd,
)

from ..state import AppState
from ..widgets import BusyBar, CommitControls, ImageSlideshow, ParametersDialog, StagePanel, make_spinbox
from ..workers import FunctionWorker, run_worker

_DEFAULT_N_COMPONENTS = 20

# Short, pipeline-breadcrumb-friendly names per method -- "Patch Warp"
# matches how the user refers to it, not the combo box's longer label.
_METHOD_LABELS = {"rigid": "Rigid", "patch": "Patch-based", "patchwarp": "Patch Warp"}


def _run_and_assess(movie: np.ndarray, method: str, n_components: int, **kwargs) -> dict:
    """Runs off the GUI thread: registration plus every metric needed to
    populate the tab, packaged into one dict."""
    registered, shifts, template, initial_template = motion_correct(movie, method=method, **kwargs)
    sv_before, _pc_before = spatiotemporal_svd(movie, n_components=n_components)
    sv_after, pc_after = spatiotemporal_svd(registered, n_components=n_components)
    return {
        "registered": registered,
        "shifts": shifts,
        "template": template,
        "initial_template": initial_template,
        "mmd": mean_max_intensity_difference(movie, registered),
        "mcm_before": mean_correlation_to_reference(movie),
        "mcm_after": mean_correlation_to_reference(registered),
        "ecc": enhanced_correlation_coefficient(initial_template, template),
        "sv_before": sv_before,
        "sv_after": sv_after,
        "pc_after": pc_after,
    }


class MotionCorrectionTab(QWidget):
    data_changed = Signal()  # emitted only on Commit, not on Apply

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.worker: FunctionWorker | None = None
        self._input_movie: np.ndarray | None = None
        self._pending_result: dict | None = None
        self._pending_step_label: str | None = None

        layout = QVBoxLayout(self)

        # All per-algorithm parameters live in the ParametersDialog popup
        # below rather than sprawling across the tab -- this row is the
        # only thing always visible: which algorithm, its parameters
        # button, and the two actions (Apply / Commit).
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Rigid", "Patch-based (non-rigid)", "PatchWarp (piecewise-affine)"])

        self.max_shift_spin = make_spinbox(0, 200, 15, decimal=True)
        self.upsample_spin = make_spinbox(1, 200, 20)
        self.n_iter_spin = make_spinbox(1, 20, 1)
        self.grid_size_spin = make_spinbox(4, 2000, 32)
        self.patchwarp_grid_spin = make_spinbox(1, 16, 4)
        self.overlap_frac_spin = make_spinbox(0.0, 0.5, 0.1, step=0.05, decimal=True)
        self.ecc_iterations_spin = make_spinbox(1, 500, 30)
        self.pyramid_levels_spin = make_spinbox(1, 4, 1)
        self.pc_count_spin = make_spinbox(1, 100, _DEFAULT_N_COMPONENTS)

        self.params_dialog = ParametersDialog(title="Motion Correction Parameters", parent=self)
        self.params_dialog.add_row("max_shift (rigid stage, all methods)", self.max_shift_spin)
        self.params_dialog.add_row("upsample_factor (rigid/patch-based only)", self.upsample_spin)
        self.params_dialog.add_row("n_iter (rigid stage, all methods)", self.n_iter_spin)
        self.params_dialog.add_row("grid_size in pixels (patch-based only)", self.grid_size_spin)
        self.params_dialog.add_row("patch grid (N x N, PatchWarp only)", self.patchwarp_grid_spin)
        self.params_dialog.add_row("overlap_frac (PatchWarp only)", self.overlap_frac_spin)
        self.params_dialog.add_row("ecc_iterations (PatchWarp only)", self.ecc_iterations_spin)
        self.params_dialog.add_row("pyramid_levels (PatchWarp only)", self.pyramid_levels_spin)
        self.params_dialog.add_row("number of spatial PCs (spectrum + slideshow)", self.pc_count_spin)

        self.commit_controls = CommitControls(apply_label="Apply Motion Correction")
        self.commit_controls.set_apply_enabled(False)
        self.commit_controls.apply_clicked.connect(self._apply)
        self.commit_controls.commit_clicked.connect(self._commit)

        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("Method:"))
        controls_row.addWidget(self.method_combo)
        self.params_btn = QPushButton("Parameters...")
        self.params_btn.clicked.connect(self.params_dialog.exec)
        controls_row.addWidget(self.params_btn)
        controls_row.addWidget(self.commit_controls)
        controls_row.addStretch()
        layout.addLayout(controls_row)

        self.busy_bar = BusyBar()
        layout.addWidget(self.busy_bar)

        self.status_label = QLabel("No data loaded.")
        layout.addWidget(self.status_label)

        self.panel = StagePanel(before_title="Raw (mean projection)", after_title="Candidate (mean projection)")
        layout.addWidget(self.panel)

        self.metrics_label = QLabel("Run motion correction to see quality metrics.")
        self.panel.add_metric_widget(self.metrics_label)

        self.sv_plot = pg.PlotWidget(title="Singular value spectrum (tighter after = better)")
        self.sv_plot.addLegend()
        self.sv_plot.setLabel("bottom", "singular value number")
        self.sv_plot.setLabel("left", "normalized singular value")
        self.panel.add_metric_widget(self.sv_plot)

        pc_container = QWidget()
        pc_layout = QVBoxLayout(pc_container)
        pc_layout.addWidget(QLabel("Top spatial PCs (candidate) -- halos/crescents mean residual motion"))
        self.pc_slideshow = ImageSlideshow()
        pc_layout.addWidget(self.pc_slideshow)
        self.panel.add_metric_widget(pc_container)

    def on_data_loaded(self) -> None:
        movie = self.state.active_data()
        self.commit_controls.set_apply_enabled(movie is not None)
        self.commit_controls.set_commit_enabled(False)
        self._pending_result = None
        self._pending_step_label = None
        if movie is not None:
            self.panel.before_view.setImage(movie.mean(axis=2))
            self.panel.set_before_movie(movie)
            self.status_label.setText(f"Ready. shape={movie.shape}")

    def _apply(self) -> None:
        movie = self.state.active_data()
        if movie is None:
            QMessageBox.warning(self, "No data", "Load data on the Load tab first.")
            return

        self._input_movie = movie
        self.commit_controls.set_apply_enabled(False)
        self.commit_controls.set_commit_enabled(False)

        method_text = self.method_combo.currentText()
        if method_text.startswith("Rigid"):
            method = "rigid"
        elif method_text.startswith("Patch-based"):
            method = "patch"
        else:
            method = "patchwarp"
        self._pending_step_label = _METHOD_LABELS[method]

        max_shift = self.max_shift_spin.value()
        n_iter = self.n_iter_spin.value()

        if method in ("rigid", "patch"):
            kwargs = dict(
                max_shift=max_shift,
                upsample_factor=self.upsample_spin.value(),
                n_iter=n_iter,
                init_batch=movie.shape[-1],
            )
            if method == "patch":
                kwargs["grid_size"] = self.grid_size_spin.value()
                kwargs["max_dev"] = max(1.0, max_shift / 3)
        else:
            kwargs = dict(
                grid_size=self.patchwarp_grid_spin.value(),
                overlap_frac=self.overlap_frac_spin.value(),
                rigid_max_shift=max_shift,
                rigid_n_iter=n_iter,
                ecc_iterations=self.ecc_iterations_spin.value(),
                pyramid_levels=self.pyramid_levels_spin.value(),
            )
        kwargs["n_components"] = self.pc_count_spin.value()

        self.worker = run_worker(
            self.busy_bar, "Running motion correction and metrics (this can take a while)...",
            _run_and_assess, movie, method, on_success=self._on_finished, on_failure=self._on_failed, **kwargs,
        )

    def _on_finished(self, result: dict) -> None:
        self._pending_result = result

        self.panel.before_view.setImage(self._input_movie.mean(axis=2))
        self.panel.after_view.setImage(result["registered"].mean(axis=2))
        self.panel.set_after_movie(result["registered"])

        self.metrics_label.setText(
            f"mMD: {result['mmd']:.2f}\n"
            f"self-mCM before -> after: {result['mcm_before']:.3f} -> {result['mcm_after']:.3f}\n"
            f"ECC(initial -> final template): {result['ecc']:.3f}"
        )

        # Both curves normalized to "before"'s own peak, so "after" tightening
        # (or not) reads directly off the same 0-1 scale "before" is plotted on.
        sv_before = result["sv_before"]
        norm = sv_before.max() if sv_before.max() > 0 else 1.0
        self.sv_plot.clear()
        self.sv_plot.plot(sv_before / norm, pen="r", name="Before")
        self.sv_plot.plot(result["sv_after"] / norm, pen="g", name="After")

        self.pc_slideshow.set_stack(result["pc_after"])

        self.busy_bar.stop("")
        self.status_label.setText(
            f"Candidate ready (shape={result['registered'].shape}). "
            "Click 'Commit to Active Dataset' to keep it, or Apply again to discard and retry."
        )
        self.commit_controls.set_apply_enabled(True)
        self.commit_controls.set_commit_enabled(True)

    def _on_failed(self, message: str) -> None:
        self.busy_bar.stop("Failed.")
        self.status_label.setText(f"Failed: {message}")
        QMessageBox.critical(self, "Motion correction failed", message)
        self.commit_controls.set_apply_enabled(True)

    def _commit(self) -> None:
        if self._pending_result is None:
            return
        self.state.commit(self._pending_result["registered"], self._pending_step_label)
        self.status_label.setText(f"Committed as pipeline step '{self._pending_step_label}'.")
        self.commit_controls.set_commit_enabled(False)
        self.data_changed.emit()
