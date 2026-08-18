"""Shared skeleton for a pipeline-stage tab (Motion Correction, Denoising,
Normalization, ...): Apply runs the stage's algorithm in a background
worker and previews the result as a candidate; Commit makes it the
active dataset and records the step in the pipeline breadcrumb. Nothing
overwrites the active dataset until Commit is clicked.

Subclasses provide the algorithm-specific pieces: `_stage_name` and
`_result_key` class attributes, `_build_controls_row`/`_build_metrics`
to lay out their own widgets, `_start_worker` to launch the Apply run,
and `_render_result` to show a finished candidate.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMessageBox, QVBoxLayout, QWidget

from ..state import AppState
from ..widgets import BusyBar, CommitControls, StagePanel
from ..workers import FunctionWorker


class StageTab(QWidget):
    data_changed = Signal()  # emitted only on Commit, not on Apply

    _stage_name = "Stage"  # used in the failure dialog title
    _result_key = "result"  # key into the worker's result dict for the candidate array

    def __init__(
        self,
        state: AppState,
        apply_label: str,
        before_title: str = "Raw (mean projection)",
        after_title: str = "Candidate (mean projection)",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.worker: FunctionWorker | None = None
        self._input_movie: np.ndarray | None = None
        self._pending_result: dict | None = None
        self._pending_step_label: str | None = None

        layout = QVBoxLayout(self)

        self.commit_controls = CommitControls(apply_label=apply_label)
        self.commit_controls.set_apply_enabled(False)
        self.commit_controls.apply_clicked.connect(self._apply)
        self.commit_controls.commit_clicked.connect(self._commit)

        layout.addLayout(self._build_controls_row())

        self.busy_bar = BusyBar()
        layout.addWidget(self.busy_bar)

        self.status_label = QLabel("No data loaded.")
        layout.addWidget(self.status_label)

        self.panel = StagePanel(before_title=before_title, after_title=after_title)
        layout.addWidget(self.panel)

        self._build_metrics()

    def _build_controls_row(self) -> QHBoxLayout:
        """Returns the row above the busy bar: algorithm choice,
        Parameters button, self.commit_controls, etc."""
        raise NotImplementedError

    def _build_metrics(self) -> None:
        """Adds self.metrics_label and any other QC widgets to self.panel."""
        raise NotImplementedError

    def _on_data_reset(self) -> None:
        """Hook for subclass state that also needs clearing on reload
        (e.g. QC location markers). No-op by default."""

    def on_data_loaded(self) -> None:
        movie = self.state.active_data()
        self.commit_controls.set_apply_enabled(movie is not None)
        self.commit_controls.set_commit_enabled(False)
        self._pending_result = None
        self._pending_step_label = None
        self._on_data_reset()
        if movie is not None:
            self.panel.before_view.setImage(movie.mean(axis=2))
            self.panel.set_before_movie(movie)
            self.status_label.setText(f"Ready. shape={movie.shape}")

    def _start_worker(self, movie: np.ndarray) -> None:
        """Sets self._pending_step_label and launches self.worker via
        run_worker, with self._on_finished/self._on_failed as callbacks."""
        raise NotImplementedError

    def _apply(self) -> None:
        movie = self.state.active_data()
        if movie is None:
            QMessageBox.warning(self, "No data", "Load data on the Load tab first.")
            return

        self._input_movie = movie
        self.commit_controls.set_apply_enabled(False)
        self.commit_controls.set_commit_enabled(False)
        self._start_worker(movie)

    def _render_result(self, result: dict) -> None:
        """Updates the panel images/movies and self.metrics_label (plus
        any other QC widgets) for a finished candidate."""
        raise NotImplementedError

    def _on_finished(self, result: dict) -> None:
        self._pending_result = result
        self._render_result(result)

        self.busy_bar.stop("")
        self.status_label.setText(
            f"Candidate ready (shape={result[self._result_key].shape}). "
            "Click 'Commit to Active Dataset' to keep it, or Apply again to discard and retry."
        )
        self.commit_controls.set_apply_enabled(True)
        self.commit_controls.set_commit_enabled(True)

    def _on_failed(self, message: str) -> None:
        self.busy_bar.stop("Failed.")
        self.status_label.setText(f"Failed: {message}")
        QMessageBox.critical(self, f"{self._stage_name} failed", message)
        self.commit_controls.set_apply_enabled(True)

    def _commit(self) -> None:
        if self._pending_result is None:
            return
        self.state.commit(self._pending_result[self._result_key], self._pending_step_label)
        self.status_label.setText(f"Committed as pipeline step '{self._pending_step_label}'.")
        self.commit_controls.set_commit_enabled(False)
        self.data_changed.emit()
