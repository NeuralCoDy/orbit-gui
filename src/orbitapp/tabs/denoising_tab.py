"""Denoising tab: wavelet shrinkage (temporal or spatial), Gaussian
filtering (temporal or spatial), or median filtering, via orbit.denoising.
Residual energy fraction (how much signal was treated as noise and
removed) and the change in mean local-pixel-correlation (denoising
should raise it, since it suppresses spatially-independent noise while
preserving spatially-coherent signal) are shown alongside the images.

Same Apply-produces-a-candidate / Commit-makes-it-active pattern as
Motion Correction -- see that tab's module docstring.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from orbit.denoising import (
    denoise_gaussian,
    denoise_median,
    denoise_wavelet_space,
    denoise_wavelet_time,
    qc_trace_samples,
    residual_energy_fraction,
)
from orbit.projections import local_correlation_projection

from ..state import AppState
from ..widgets import BusyBar, CommitControls, ParametersDialog, StagePanel, make_spinbox
from ..workers import FunctionWorker, run_worker

_WAVELETS = ("sym4", "db4", "haar", "coif2")
_THRESHOLD_METHODS = ("bayes", "universal")
_N_TRACE_PLOTS = 4  # 2 correlation-peak locations + 2 low-correlation locations

# (combo box label, dispatch key) -- key is deliberately not called
# "method", since the wavelet threshold method ("bayes"/"universal") is
# itself passed as a same-named kwarg; a name collision there previously
# raised a TypeError on every denoise run.
_ALGORITHMS = (
    ("Wavelet - Temporal (per pixel)", "wavelet_time"),
    ("Wavelet - Spatial (per frame)", "wavelet_space"),
    ("Gaussian Filter", "gaussian"),
    ("Median Filter", "median"),
)
_ALGORITHM_KEYS = dict(_ALGORITHMS)
_DENOISE_FUNCS = {
    "wavelet_time": denoise_wavelet_time,
    "wavelet_space": denoise_wavelet_space,
    "gaussian": denoise_gaussian,
    "median": denoise_median,
}
# Which ParametersDialog group each algorithm's fields belong to --
# several algorithms can share a group (both wavelet domains use the
# same wavelet/level/threshold fields).
_GROUP_BY_ALGORITHM = {
    "wavelet_time": "wavelet",
    "wavelet_space": "wavelet",
    "gaussian": "gaussian",
    "median": "median",
}


def _run_and_assess(movie: np.ndarray, algorithm: str, **kwargs) -> dict:
    """Runs off the GUI thread."""
    denoised = _DENOISE_FUNCS[algorithm](movie, **kwargs)
    return {
        "denoised": denoised,
        "residual_energy_fraction": residual_energy_fraction(movie, denoised),
        "corr_before": float(local_correlation_projection(movie).mean()),
        "corr_after": float(local_correlation_projection(denoised).mean()),
        "qc_traces": qc_trace_samples(movie, denoised),
    }


class DenoisingTab(QWidget):
    data_changed = Signal()  # emitted only on Commit, not on Apply

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.worker: FunctionWorker | None = None
        self._input_movie: np.ndarray | None = None
        self._pending_result: dict | None = None

        layout = QVBoxLayout(self)

        self.method_combo = QComboBox()
        self.method_combo.addItems([label for label, _key in _ALGORITHMS])
        self.method_combo.currentTextChanged.connect(self._update_visible_params)

        self.wavelet_combo = QComboBox()
        self.wavelet_combo.addItems(_WAVELETS)
        self.level_spin = make_spinbox(1, 10, 4)
        self.threshold_combo = QComboBox()
        self.threshold_combo.addItems(_THRESHOLD_METHODS)
        self.gaussian_spatial_spin = make_spinbox(0.0, 50.0, 2.0, step=0.5, decimal=True)
        self.gaussian_temporal_spin = make_spinbox(0.0, 50.0, 0.0, step=0.5, decimal=True)
        self.median_space_spin = make_spinbox(1, 51, 3)
        self.median_time_spin = make_spinbox(1, 51, 1)

        self.params_dialog = ParametersDialog(title="Denoising Parameters", parent=self)
        self.params_dialog.add_row("wavelet", self.wavelet_combo, group="wavelet")
        self.params_dialog.add_row("level", self.level_spin, group="wavelet")
        self.params_dialog.add_row("threshold method", self.threshold_combo, group="wavelet")
        self.params_dialog.add_row("spatial width (pixels, 0 = temporal only)", self.gaussian_spatial_spin, group="gaussian")
        self.params_dialog.add_row("temporal width (frames, 0 = spatial only)", self.gaussian_temporal_spin, group="gaussian")
        self.params_dialog.add_row("space_window", self.median_space_spin, group="median")
        self.params_dialog.add_row("time_window", self.median_time_spin, group="median")
        self._update_visible_params(self.method_combo.currentText())

        self.commit_controls = CommitControls(apply_label="Apply Denoising")
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

        self.metrics_label = QLabel("Run denoising to see quality metrics.")
        self.panel.add_metric_widget(self.metrics_label)

        traces_container = QWidget()
        traces_layout = QGridLayout(traces_container)
        traces_layout.addWidget(
            QLabel("Representative pixel traces -- 2 local-correlation peaks, 2 low-correlation (before vs. after)"),
            0, 0, 1, 2,
        )
        self.trace_plots = []
        for i in range(_N_TRACE_PLOTS):
            plot = pg.PlotWidget()
            plot.addLegend()
            plot.setLabel("bottom", "frame")
            plot.setLabel("left", "intensity")
            traces_layout.addWidget(plot, 1 + i // 2, i % 2)
            self.trace_plots.append(plot)
        self.panel.add_metric_widget(traces_container)

    def _update_visible_params(self, label: str) -> None:
        """Only the fields relevant to the selected algorithm are shown
        in the Parameters popup -- e.g. wavelet/level/threshold method
        stay hidden while a Gaussian or Median method is selected."""
        group = _GROUP_BY_ALGORITHM[_ALGORITHM_KEYS[label]]
        self.params_dialog.show_only_group(group)

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

        algorithm = _ALGORITHM_KEYS[self.method_combo.currentText()]
        if algorithm in ("wavelet_time", "wavelet_space"):
            kwargs = dict(
                wavelet=self.wavelet_combo.currentText(),
                level=self.level_spin.value(),
                method=self.threshold_combo.currentText(),
            )
        elif algorithm == "gaussian":
            kwargs = dict(
                spatial_sigma=self.gaussian_spatial_spin.value(), temporal_sigma=self.gaussian_temporal_spin.value()
            )
        else:
            kwargs = dict(space_window=self.median_space_spin.value(), time_window=self.median_time_spin.value())

        self.worker = run_worker(
            self.busy_bar, "Running denoising and metrics (this can take a while)...",
            _run_and_assess, movie, algorithm, on_success=self._on_finished, on_failure=self._on_failed, **kwargs,
        )

    def _on_finished(self, result: dict) -> None:
        self._pending_result = result

        self.panel.before_view.setImage(self._input_movie.mean(axis=2))
        self.panel.after_view.setImage(result["denoised"].mean(axis=2))
        self.panel.set_after_movie(result["denoised"])

        self.metrics_label.setText(
            f"Residual energy fraction: {result['residual_energy_fraction']:.3f}\n"
            f"Mean local correlation before -> after: "
            f"{result['corr_before']:.3f} -> {result['corr_after']:.3f}"
        )

        for plot in self.trace_plots:
            plot.clear()
            plot.setTitle("")
        for plot, sample in zip(self.trace_plots, result["qc_traces"]):
            plot.plot(sample["before"], pen="r", name="Before")
            plot.plot(sample["after"], pen="g", name="After")
            plot.setTitle(f"{sample['kind']} @ ({sample['row']}, {sample['col']}), corr={sample['corr']:.2f}")

        self.busy_bar.stop("")
        self.status_label.setText(
            f"Candidate ready (shape={result['denoised'].shape}). "
            "Click 'Commit to Active Dataset' to keep it, or Apply again to discard and retry."
        )
        self.commit_controls.set_apply_enabled(True)
        self.commit_controls.set_commit_enabled(True)

    def _on_failed(self, message: str) -> None:
        self.busy_bar.stop("Failed.")
        self.status_label.setText(f"Failed: {message}")
        QMessageBox.critical(self, "Denoising failed", message)
        self.commit_controls.set_apply_enabled(True)

    def _commit(self) -> None:
        if self._pending_result is None:
            return
        self.state.commit(self._pending_result["denoised"], "Denoise")
        self.status_label.setText("Committed as pipeline step 'Denoise'.")
        self.commit_controls.set_commit_enabled(False)
        self.data_changed.emit()
