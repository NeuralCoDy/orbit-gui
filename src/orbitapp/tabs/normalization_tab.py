"""Normalization tab: centering + scale normalization (a Delta-F/F-style
transform, orbit.normalization.normalize_movie), with before/after
summary statistics shown alongside the images, plus a qualitative check
of each pixel's value histogram at a handful of auto-picked signal/noise
locations. Before/after are plotted as separate, side-by-side histograms
rather than overlaid, since normalization can shift value ranges by
orders of magnitude.

Same Apply-produces-a-candidate / Commit-makes-it-active pattern as
every StageTab -- see that module's docstring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton

from orbit.normalization import (
    apply_baselines,
    compute_baselines,
    describe_normalization,
    normalize_movie,
    pixel_value_histogram,
    summary_stats,
)
from orbit.qc_traces import qc_trace_samples

from ..fits_io import create_fits_memmap
from ..io import preview_slice
from ..state import AppState
from ..widgets import ParametersDialog, QCPlotGrid, add_location_markers, pixels_to_data_pos, split_by_kind
from ..workers import run_worker
from .stage_tab import StageTab

_CENTER_BASELINES = ("min", "median", "mean", "mode")
_NORM_BASELINES = ("median", "mean", "max", "robuststd")
_STAT_LINE_STYLES = {"mean": Qt.PenStyle.SolidLine, "median": Qt.PenStyle.DashLine, "mode": Qt.PenStyle.DotLine}


def _run_and_assess(movie: np.ndarray, **kwargs) -> dict:
    """Runs off the GUI thread."""
    normalized = normalize_movie(movie, **kwargs)
    qc_traces = qc_trace_samples(movie, normalized)
    for sample in qc_traces:
        sample["hist_before"] = pixel_value_histogram(sample["before"])
        sample["hist_after"] = pixel_value_histogram(sample["after"])
    return {
        "normalized": normalized,
        "stats_before": summary_stats(movie),
        "stats_after": summary_stats(normalized),
        "qc_traces": qc_traces,
    }


def _plot_histogram(plot: pg.PlotWidget, hist: dict, color: str) -> None:
    edges = hist["edges"]
    centers = (edges[:-1] + edges[1:]) / 2
    plot.addItem(pg.BarGraphItem(x=centers, height=hist["counts"], width=(edges[1] - edges[0]) * 0.9, brush=color))
    legend = plot.plotItem.legend  # from QCPlotGrid's own add_legend=True default -- see qc_panel.py
    for stat, style in _STAT_LINE_STYLES.items():
        pen = pg.mkPen(color, style=style, width=1.5)
        line = pg.InfiniteLine(pos=hist[stat], angle=90, pen=pen)
        plot.addItem(line)
        if legend is not None:
            # NOT legend.addItem(line, stat) -- pyqtgraph's LegendItem
            # assumes every entry is a PlotDataItem/BarGraphItem/
            # ScatterPlotItem (all of which carry an `.opts` dict);
            # InfiniteLine has no `.opts` at all. That's not a subtle
            # edge case: a minimal repro (one InfiniteLine added to a
            # legend, then rendered once) segfaults immediately --
            # LegendItem's paint() does `self.item.opts` unconditionally,
            # an AttributeError that a Qt paintEvent can't recover from.
            # A standalone PlotDataItem with the same pen, never added to
            # the plot itself, gives the legend swatch something with a
            # real `.opts` to read, with an identical-looking line.
            legend.addItem(pg.PlotDataItem(pen=pen), stat)


def _plot_histograms(plots: list[pg.PlotWidget], sample: dict) -> None:
    before_plot, after_plot = plots
    _plot_histogram(before_plot, sample["hist_before"], "r")
    _plot_histogram(after_plot, sample["hist_after"], "b")


class NormalizationTab(StageTab):
    _stage_name = "Normalization"
    _result_key = "normalized"
    _stage_key = "normalization"

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, apply_label="Apply Normalization", parent=parent)

    def _build_controls_row(self) -> QHBoxLayout:
        self.center_check = QCheckBox("Center")
        self.center_check.setChecked(True)
        self.center_baseline_combo = QComboBox()
        self.center_baseline_combo.addItems(_CENTER_BASELINES)
        self.center_baseline_combo.setCurrentText("mode")
        self.pixel_center_check = QCheckBox("Center per-pixel (not globally)")
        self.pixel_center_check.setChecked(True)

        self.normalize_check = QCheckBox("Normalize")
        self.normalize_check.setChecked(True)
        self.norm_baseline_combo = QComboBox()
        self.norm_baseline_combo.addItems(_NORM_BASELINES)
        self.norm_baseline_combo.setCurrentText("robuststd")
        self.pixel_norm_check = QCheckBox("Normalize per-pixel (not globally)")
        self.pixel_norm_check.setChecked(True)

        self.params_dialog = ParametersDialog(title="Normalization Parameters", parent=self)
        self.params_dialog.add_row("", self.center_check)
        self.params_dialog.add_row("center_baseline", self.center_baseline_combo)
        self.params_dialog.add_row("", self.pixel_center_check)
        self.params_dialog.add_row("", self.normalize_check)
        self.params_dialog.add_row("norm_baseline", self.norm_baseline_combo)
        self.params_dialog.add_row("", self.pixel_norm_check)

        controls_row = QHBoxLayout()
        self.params_btn = QPushButton("Parameters...")
        self.params_btn.clicked.connect(self.params_dialog.exec)
        controls_row.addWidget(self.params_btn)
        controls_row.addWidget(self.commit_controls)
        controls_row.addStretch()
        return controls_row

    def _build_metrics(self) -> None:
        self.metrics_label = QLabel("Run normalization to see summary statistics.")
        self.panel.add_metric_widget(self.metrics_label)

        self._location_markers = add_location_markers(self.panel.before_view)
        self.hist_grid = QCPlotGrid(
            "Example signal pixels", "Example noise pixels",
            plots_per_sample=2, sub_labels=["Before", "After"],
            xlabel="pixel value", ylabel="count",
        )
        # Every "Before" plot shares one x-axis, every "After" plot shares
        # another -- directly comparable within a column (before/after can
        # differ wildly in scale from each other, which is why they're
        # separate plots rather than overlaid in the first place).
        rows = self.hist_grid.left_rows + self.hist_grid.right_rows
        for column in range(2):
            plots_in_column = [row[column] for row in rows]
            for plot in plots_in_column[1:]:
                plot.setXLink(plots_in_column[0])
        self.panel.add_metric_widget(self.hist_grid)

    def _on_data_reset(self) -> None:
        self._location_markers.clear()

    def _current_fingerprint(self) -> dict:
        """center_baseline/pixel_center only matter when center is on
        (same for norm_baseline/pixel_norm and normalize) -- recording
        them while that toggle is off would misrepresent what this
        commit actually ran with, both in a saved session and in the
        "Generate Report" PDF."""
        center = self.center_check.isChecked()
        normalize = self.normalize_check.isChecked()
        params = dict(center=center, normalize=normalize)
        if center:
            params["center_baseline"] = self.center_baseline_combo.currentText()
            params["pixel_center"] = self.pixel_center_check.isChecked()
        if normalize:
            params["norm_baseline"] = self.norm_baseline_combo.currentText()
            params["pixel_norm"] = self.pixel_norm_check.isChecked()
        return params

    def restore_params(self, params: dict) -> None:
        if "center" in params:
            self.center_check.setChecked(params["center"])
        if "center_baseline" in params:
            self.center_baseline_combo.setCurrentText(params["center_baseline"])
        if "pixel_center" in params:
            self.pixel_center_check.setChecked(params["pixel_center"])
        if "normalize" in params:
            self.normalize_check.setChecked(params["normalize"])
        if "norm_baseline" in params:
            self.norm_baseline_combo.setCurrentText(params["norm_baseline"])
        if "pixel_norm" in params:
            self.pixel_norm_check.setChecked(params["pixel_norm"])

    def _extract_metrics(self, result: dict) -> dict:
        return dict(stats_before=result["stats_before"], stats_after=result["stats_after"])

    def _kwargs(self) -> dict:
        return dict(
            center=self.center_check.isChecked(),
            center_baseline=self.center_baseline_combo.currentText(),
            pixel_center=self.pixel_center_check.isChecked(),
            normalize=self.normalize_check.isChecked(),
            norm_baseline=self.norm_baseline_combo.currentText(),
            pixel_norm=self.pixel_norm_check.isChecked(),
        )

    def _start_worker(self, movie: np.ndarray) -> None:
        kwargs = self._kwargs()
        self._pending_step_label = f"Normalize {describe_normalization(**kwargs)}"
        self.worker = run_worker(
            self.busy_bar, "Running normalization and metrics...",
            _run_and_assess, movie, on_success=self._on_finished, on_failure=self._on_failed, **kwargs,
        )

    def _chunked_commit(self, source: np.ndarray, output_path: Path) -> np.ndarray:
        kwargs = self._kwargs()
        # Baselines (median/mode/robuststd are whole-time-axis per-pixel
        # statistics -- not exactly computable from streamed chunks
        # without keeping every value) are fit once from the same
        # <=5000-frame preview Apply already ran against, then applied
        # as a fixed transform to every chunk of the full movie. This
        # matches the preview's own result exactly for that sample, and
        # is the same "fit on the 5000-frame sample" approximation used
        # for the preview itself -- not a further approximation on top.
        preview = preview_slice(source)
        baselines = compute_baselines(
            preview, center=kwargs["center"], normalize=kwargs["normalize"],
            center_baseline=kwargs["center_baseline"], norm_baseline=kwargs["norm_baseline"],
            pixel_center=kwargs["pixel_center"], pixel_norm=kwargs["pixel_norm"],
        )

        T = source.shape[-1]
        output = create_fits_memmap(output_path, source.shape, np.float32)
        for t0 in range(0, T, self._chunk_frames):
            t1 = min(t0 + self._chunk_frames, T)
            chunk = np.asarray(source[:, :, t0:t1], dtype=np.float32)
            output[:, :, t0:t1] = apply_baselines(chunk, baselines, center=kwargs["center"], normalize=kwargs["normalize"])
        output.flush()
        return output

    def _render_result(self, result: dict) -> None:
        self.panel.before_view.setImage(preview_slice(self._input_movie).mean(axis=2))
        self.panel.after_view.setImage(result["normalized"].mean(axis=2))
        self.panel.set_after_movie(result["normalized"])

        before, after = result["stats_before"], result["stats_after"]
        self.metrics_label.setText(
            "\n".join(f"{key}: {before[key]:.3g} -> {after[key]:.3g}" for key in ("min", "max", "mean", "std"))
        )

        qc_traces = result["qc_traces"]
        marker_xs, marker_ys = pixels_to_data_pos(
            self.panel.before_view.getImageItem(),
            [s["row"] + 0.5 for s in qc_traces], [s["col"] + 0.5 for s in qc_traces],
        )
        self._location_markers.setData(marker_xs, marker_ys)
        peak_samples, low_samples = split_by_kind(qc_traces)
        self.hist_grid.fill(peak_samples, low_samples, _plot_histograms)
