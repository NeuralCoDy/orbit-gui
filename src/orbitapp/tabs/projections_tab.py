"""Data Projections tab: a single full-size view with a dropdown to flip
between the available per-pixel summary projections (orbit.projections).
Each projection is computed lazily, the first time it's selected (some,
like local correlation or mode, are expensive) and cached after that, so
loading a movie never blocks on projections nobody's asked to see yet.
Computation runs in a background FunctionWorker with a busy indicator,
same as any other >2s operation.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QComboBox, QLabel, QMessageBox, QVBoxLayout, QWidget

from orbit.projections import (
    fano_factor_projection,
    local_correlation_projection,
    mean_projection,
    median_projection,
    mode_projection,
    variance_projection,
)

from ..state import AppState
from ..widgets import BusyBar
from ..workers import FunctionWorker, run_worker

_PROJECTIONS = (
    ("Mean", mean_projection),
    ("Median", median_projection),
    ("Mode", mode_projection),
    ("Variance", variance_projection),
    ("Fano factor", fano_factor_projection),
    ("Local correlation", local_correlation_projection),
)


class ProjectionsTab(QWidget):
    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self._fns = dict(_PROJECTIONS)
        self._cache: dict[str, np.ndarray] = {}
        self.worker: FunctionWorker | None = None

        layout = QVBoxLayout(self)

        self.status_label = QLabel("No data loaded.")
        layout.addWidget(self.status_label)

        self.busy_bar = BusyBar()
        layout.addWidget(self.busy_bar)

        self.selector = QComboBox()
        self.selector.addItems([name for name, _fn in _PROJECTIONS])
        self.selector.currentTextChanged.connect(self._show_selected)
        layout.addWidget(self.selector)

        self.view = pg.ImageView()
        layout.addWidget(self.view)

    def on_data_loaded(self) -> None:
        self._cache = {}
        movie = self.state.active_data()
        if movie is None:
            self.status_label.setText("No data loaded.")
            self.view.clear()
            return

        self.status_label.setText(f"shape={movie.shape} (H, W, T)")
        self._show_selected(self.selector.currentText())

    def _show_selected(self, name: str) -> None:
        movie = self.state.active_data()
        if movie is None:
            return
        if name in self._cache:
            self.view.setImage(self._cache[name])
            return

        self.selector.setEnabled(False)
        self.worker = run_worker(
            self.busy_bar, f"Computing {name}...", self._fns[name], movie,
            on_success=lambda result, name=name: self._on_computed(name, result),
            on_failure=self._on_failed,
        )

    def _on_computed(self, name: str, result: np.ndarray) -> None:
        self._cache[name] = result
        self.busy_bar.stop("")
        self.selector.setEnabled(True)
        if self.selector.currentText() == name:
            self.view.setImage(result)

    def _on_failed(self, message: str) -> None:
        self.busy_bar.stop("Computation failed.")
        self.selector.setEnabled(True)
        QMessageBox.critical(self, "Projection failed", message)
