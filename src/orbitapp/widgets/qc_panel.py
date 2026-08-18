"""Shared "before/after check at a handful of auto-picked pixels" panel
used by any stage tab built on orbit.qc_traces.qc_trace_samples --
Denoising's before/after traces, Normalization's before/after value
histograms, and any future stage that wants the same qualitative
sanity check rather than re-building the layout each time.
"""

from __future__ import annotations

from typing import Callable

import pyqtgraph as pg
from PySide6.QtWidgets import QGridLayout, QLabel, QWidget


def add_location_markers(image_view: pg.ImageView) -> pg.ScatterPlotItem:
    """Adds (and returns) a red-X scatter overlay to an ImageView --
    marks exactly which pixels a QC panel's plots come from, in place
    of a per-plot title that's unreadable at typical window sizes."""
    markers = pg.ScatterPlotItem(symbol="x", size=14, pen=pg.mkPen("r", width=2), brush=None)
    image_view.getView().addItem(markers)
    return markers


def split_by_kind(samples: list[dict]) -> tuple[list[dict], list[dict]]:
    """Splits qc_trace_samples's output into (peak, low) lists."""
    return [s for s in samples if s["kind"] == "peak"], [s for s in samples if s["kind"] == "low"]


class QCPlotGrid(QWidget):
    """Two labeled columns (e.g. "signal" / "noise" pixels), each row
    holding one sample's ``plots_per_sample`` side-by-side plots -- 1
    for a single overlaid before/after plot, more when before/after
    need separate axes (e.g. value ranges too far apart to share one)."""

    def __init__(
        self,
        left_title: str,
        right_title: str,
        n_rows: int = 2,
        plots_per_sample: int = 1,
        sub_labels: list[str] | None = None,
        xlabel: str = "",
        ylabel: str = "",
        add_legend: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.addWidget(QLabel(left_title), 0, 0, 1, plots_per_sample)
        layout.addWidget(QLabel(right_title), 0, plots_per_sample, 1, plots_per_sample)

        row = 1
        if sub_labels:
            for i, label in enumerate(sub_labels):
                layout.addWidget(QLabel(label), row, i)
                layout.addWidget(QLabel(label), row, plots_per_sample + i)
            row += 1

        self.left_rows = [self._make_plots(plots_per_sample, xlabel, ylabel, add_legend) for _ in range(n_rows)]
        self.right_rows = [self._make_plots(plots_per_sample, xlabel, ylabel, add_legend) for _ in range(n_rows)]
        for rows, col_offset in ((self.left_rows, 0), (self.right_rows, plots_per_sample)):
            for r, plots in enumerate(rows):
                for c, plot in enumerate(plots):
                    layout.addWidget(plot, row + r, col_offset + c)

    @staticmethod
    def _make_plots(count: int, xlabel: str, ylabel: str, add_legend: bool) -> list[pg.PlotWidget]:
        plots = []
        for _ in range(count):
            plot = pg.PlotWidget()
            if add_legend:
                plot.addLegend()
            if xlabel:
                plot.setLabel("bottom", xlabel)
            if ylabel:
                plot.setLabel("left", ylabel)
            plots.append(plot)
        return plots

    def fill(
        self, peak_samples: list[dict], low_samples: list[dict], plot_fn: Callable[[list[pg.PlotWidget], dict], None]
    ) -> None:
        """Clears every plot, then calls ``plot_fn(plots, sample)`` for
        each row -- peak samples into the left column, low samples into
        the right, in the order each list was given."""
        for plots in self.left_rows + self.right_rows:
            for plot in plots:
                plot.clear()
        for rows, samples in ((self.left_rows, peak_samples), (self.right_rows, low_samples)):
            for plots, sample in zip(rows, samples):
                plot_fn(plots, sample)
