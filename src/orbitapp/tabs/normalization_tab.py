"""Normalization tab: centering + scale normalization (a Delta-F/F-style
transform, orbit.normalization.normalize_movie), with before/after
summary statistics shown alongside the images.

Same Apply-produces-a-candidate / Commit-makes-it-active pattern as
Motion Correction -- see that tab's module docstring.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from orbit.normalization import normalize_movie, summary_stats

from ..state import AppState
from ..widgets import BusyBar, CommitControls, ParametersDialog, StagePanel
from ..workers import FunctionWorker, run_worker

_CENTER_BASELINES = ("min", "median", "mean", "mode")
_NORM_BASELINES = ("median", "mean", "max", "robuststd")


def _run_and_assess(movie: np.ndarray, **kwargs) -> dict:
    """Runs off the GUI thread."""
    normalized = normalize_movie(movie, **kwargs)
    return {
        "normalized": normalized,
        "stats_before": summary_stats(movie),
        "stats_after": summary_stats(normalized),
    }


class NormalizationTab(QWidget):
    data_changed = Signal()  # emitted only on Commit, not on Apply

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.worker: FunctionWorker | None = None
        self._input_movie: np.ndarray | None = None
        self._pending_result: dict | None = None

        layout = QVBoxLayout(self)

        self.center_check = QCheckBox("Center")
        self.center_check.setChecked(True)
        self.center_baseline_combo = QComboBox()
        self.center_baseline_combo.addItems(_CENTER_BASELINES)
        self.pixel_center_check = QCheckBox("Center per-pixel (not globally)")

        self.normalize_check = QCheckBox("Normalize")
        self.normalize_check.setChecked(True)
        self.norm_baseline_combo = QComboBox()
        self.norm_baseline_combo.addItems(_NORM_BASELINES)
        self.pixel_norm_check = QCheckBox("Normalize per-pixel (not globally)")

        self.params_dialog = ParametersDialog(title="Normalization Parameters", parent=self)
        self.params_dialog.add_row("", self.center_check)
        self.params_dialog.add_row("center_baseline", self.center_baseline_combo)
        self.params_dialog.add_row("", self.pixel_center_check)
        self.params_dialog.add_row("", self.normalize_check)
        self.params_dialog.add_row("norm_baseline", self.norm_baseline_combo)
        self.params_dialog.add_row("", self.pixel_norm_check)

        self.commit_controls = CommitControls(apply_label="Apply Normalization")
        self.commit_controls.set_apply_enabled(False)
        self.commit_controls.apply_clicked.connect(self._apply)
        self.commit_controls.commit_clicked.connect(self._commit)

        controls_row = QHBoxLayout()
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

        self.metrics_label = QLabel("Run normalization to see summary statistics.")
        self.panel.add_metric_widget(self.metrics_label)

    def on_data_loaded(self) -> None:
        movie = self.state.active_data()
        self.commit_controls.set_apply_enabled(movie is not None)
        self.commit_controls.set_commit_enabled(False)
        self._pending_result = None
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

        kwargs = dict(
            center=self.center_check.isChecked(),
            center_baseline=self.center_baseline_combo.currentText(),
            pixel_center=self.pixel_center_check.isChecked(),
            normalize=self.normalize_check.isChecked(),
            norm_baseline=self.norm_baseline_combo.currentText(),
            pixel_norm=self.pixel_norm_check.isChecked(),
        )

        self.worker = run_worker(
            self.busy_bar, "Running normalization and metrics...",
            _run_and_assess, movie, on_success=self._on_finished, on_failure=self._on_failed, **kwargs,
        )

    def _on_finished(self, result: dict) -> None:
        self._pending_result = result

        self.panel.before_view.setImage(self._input_movie.mean(axis=2))
        self.panel.after_view.setImage(result["normalized"].mean(axis=2))
        self.panel.set_after_movie(result["normalized"])

        before, after = result["stats_before"], result["stats_after"]
        self.metrics_label.setText(
            "\n".join(f"{key}: {before[key]:.3g} -> {after[key]:.3g}" for key in ("min", "max", "mean", "std"))
        )

        self.busy_bar.stop("")
        self.status_label.setText(
            f"Candidate ready (shape={result['normalized'].shape}). "
            "Click 'Commit to Active Dataset' to keep it, or Apply again to discard and retry."
        )
        self.commit_controls.set_apply_enabled(True)
        self.commit_controls.set_commit_enabled(True)

    def _on_failed(self, message: str) -> None:
        self.busy_bar.stop("Failed.")
        self.status_label.setText(f"Failed: {message}")
        QMessageBox.critical(self, "Normalization failed", message)
        self.commit_controls.set_apply_enabled(True)

    def _commit(self) -> None:
        if self._pending_result is None:
            return
        self.state.commit(self._pending_result["normalized"], "Normalize")
        self.status_label.setText("Committed as pipeline step 'Normalize'.")
        self.commit_controls.set_commit_enabled(False)
        self.data_changed.emit()
