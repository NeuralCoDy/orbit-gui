"""Denoising tab: wavelet shrinkage (temporal or spatial), Gaussian
filtering (temporal or spatial), or median filtering, via orbit.denoising.
Residual energy fraction (how much signal was treated as noise and
removed) and the change in mean local-pixel-correlation (denoising
should raise it, since it suppresses spatially-independent noise while
preserving spatially-coherent signal) are shown alongside the images.

Same Apply-produces-a-candidate / Commit-makes-it-active pattern as
every StageTab -- see that module's docstring.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton

from orbit.denoising import (
    denoise_gaussian,
    denoise_median,
    denoise_wavelet_space,
    denoise_wavelet_time,
    residual_energy_fraction,
)
from orbit.projections import local_correlation_projection
from orbit.qc_traces import qc_trace_samples

from ..state import AppState
from ..widgets import ParametersDialog, QCPlotGrid, add_location_markers, make_spinbox, split_by_kind
from ..workers import run_worker
from .stage_tab import StageTab

_WAVELETS = ("sym4", "db4", "haar", "coif2")
_THRESHOLD_METHODS = ("bayes", "universal")

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
# Pipeline-breadcrumb label per algorithm -- specific enough to tell
# denoising methods apart in the header ("Load > Gaussian Denoising >
# ...") rather than a single generic "Denoise" for all of them.
_PIPELINE_LABELS = {
    "wavelet_time": "Wavelet Denoising (Temporal)",
    "wavelet_space": "Wavelet Denoising (Spatial)",
    "gaussian": "Gaussian Denoising",
    "median": "Median Filtering",
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


def _plot_trace(plots, sample: dict) -> None:
    plot = plots[0]
    plot.plot(sample["before"], pen="r", name="Before")
    plot.plot(sample["after"], pen="g", name="After")


class DenoisingTab(StageTab):
    _stage_name = "Denoising"
    _result_key = "denoised"

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, apply_label="Apply Denoising", parent=parent)

    def _build_controls_row(self) -> QHBoxLayout:
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

        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("Method:"))
        controls_row.addWidget(self.method_combo)
        self.params_btn = QPushButton("Parameters...")
        self.params_btn.clicked.connect(self.params_dialog.exec)
        controls_row.addWidget(self.params_btn)
        controls_row.addWidget(self.commit_controls)
        controls_row.addStretch()
        return controls_row

    def _build_metrics(self) -> None:
        self.metrics_label = QLabel("Run denoising to see quality metrics.")
        self.panel.add_metric_widget(self.metrics_label)

        self._location_markers = add_location_markers(self.panel.before_view)
        self.trace_grid = QCPlotGrid("Example signal pixels", "Example noise pixels", xlabel="frame", ylabel="intensity")
        self.panel.add_metric_widget(self.trace_grid)

    def _update_visible_params(self, label: str) -> None:
        """Only the fields relevant to the selected algorithm are shown
        in the Parameters popup -- e.g. wavelet/level/threshold method
        stay hidden while a Gaussian or Median method is selected."""
        group = _GROUP_BY_ALGORITHM[_ALGORITHM_KEYS[label]]
        self.params_dialog.show_only_group(group)

    def _on_data_reset(self) -> None:
        self._location_markers.clear()

    def _start_worker(self, movie: np.ndarray) -> None:
        algorithm = _ALGORITHM_KEYS[self.method_combo.currentText()]
        self._pending_step_label = _PIPELINE_LABELS[algorithm]
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

    def _render_result(self, result: dict) -> None:
        self.panel.before_view.setImage(self._input_movie.mean(axis=2))
        self.panel.after_view.setImage(result["denoised"].mean(axis=2))
        self.panel.set_after_movie(result["denoised"])

        self.metrics_label.setText(
            f"Residual energy fraction: {result['residual_energy_fraction']:.3f}\n"
            f"Mean local correlation before -> after: "
            f"{result['corr_before']:.3f} -> {result['corr_after']:.3f}"
        )

        qc_traces = result["qc_traces"]
        self._location_markers.setData([s["row"] + 0.5 for s in qc_traces], [s["col"] + 0.5 for s in qc_traces])
        peak_samples, low_samples = split_by_kind(qc_traces)
        self.trace_grid.fill(peak_samples, low_samples, _plot_trace)
