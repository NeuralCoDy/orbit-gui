"""Data Projections tab: a single full-size view with a dropdown to flip
between the available per-pixel summary projections (orbit.projections).
Each projection is computed lazily, the first time it's selected (some,
like local correlation or mode, are expensive) and cached after that, so
loading a movie never blocks on projections nobody's asked to see yet.
Computation runs in a background FunctionWorker with a busy indicator,
same as any other >2s operation.

For a volumetric (T, L, W, D) movie (state.volumetric) the projection
reduces over *time* (axis 0), yielding an (L, W, D) volume shown as a
depth-scrollable stack -- the 2D projections all have ``axis=2`` baked
in for the (H, W, T) convention and would otherwise average over width.
Local correlation has no volumetric form yet (its 8-neighbour graph is
2D) and is disabled while volumetric data is loaded.

Volumetric data also gets a "Volume (3D)" entry: the time-mean volume
rendered as a click-drag-rotatable 3D view (see
orbitapp.widgets.VolumeView).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QComboBox, QLabel, QMessageBox, QStackedWidget, QVBoxLayout, QWidget

from orbit.projections import (
    fano_factor_projection,
    fano_factor_projection_volumetric,
    local_correlation_projection,
    mean_projection,
    mean_projection_volumetric,
    median_projection,
    median_projection_volumetric,
    mode_projection,
    mode_projection_volumetric,
    variance_projection,
    variance_projection_volumetric,
)

from ..state import AppState
from ..widgets import BusyBar, VolumeView
from ..workers import FunctionWorker, run_worker

_VOLUME_3D = "Volume (3D)"

# name -> (2D (H, W, T) projection, volumetric (T, L, W, D) projection or
# None if there's no volumetric form yet). _VOLUME_3D is handled specially.
_PROJECTIONS = (
    ("Mean", mean_projection, mean_projection_volumetric),
    ("Median", median_projection, median_projection_volumetric),
    ("Mode", mode_projection, mode_projection_volumetric),
    ("Variance", variance_projection, variance_projection_volumetric),
    ("Fano factor", fano_factor_projection, fano_factor_projection_volumetric),
    ("Local correlation", local_correlation_projection, None),
    (_VOLUME_3D, None, None),
)


class ProjectionsTab(QWidget):
    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self._fns = {name: fn for name, fn, _v in _PROJECTIONS}
        self._fns_volumetric = {name: v for name, _fn, v in _PROJECTIONS}
        self._cache: dict[str, np.ndarray] = {}
        self.worker: FunctionWorker | None = None

        layout = QVBoxLayout(self)

        self.status_label = QLabel("No data loaded.")
        layout.addWidget(self.status_label)

        self.busy_bar = BusyBar()
        layout.addWidget(self.busy_bar)

        self.selector = QComboBox()
        self.selector.addItems([name for name, *_ in _PROJECTIONS])
        self.selector.currentTextChanged.connect(self._show_selected)
        layout.addWidget(self.selector)

        self.view = pg.ImageView()
        self.volume_view = VolumeView()
        self._stack = QStackedWidget()
        self._stack.addWidget(self.view)  # index 0: 2D projections / depth stack
        self._stack.addWidget(self.volume_view)  # index 1: Volume (3D)
        layout.addWidget(self._stack)

    def on_data_loaded(self) -> None:
        self._cache = {}
        self.volume_view.clear()
        self._sync_selector_for_modality()
        movie = self.state.active_data()
        if movie is None:
            self.status_label.setText("No data loaded.")
            self.view.clear()
            return

        if self.state.volumetric:
            self.status_label.setText(
                f"shape={movie.shape} (T, L, W, D) -- projected over time; scroll for depth, "
                "or pick 'Volume (3D)' and drag to rotate"
            )
        else:
            self.status_label.setText(f"shape={movie.shape} (H, W, T)")
        self._show_selected(self.selector.currentText())

    def _sync_selector_for_modality(self) -> None:
        """Enable each entry only where it has a form: everything for 2D
        data; the volumetric projections plus Volume (3D) for volumetric.
        Move the selection off a now-disabled entry -- same pattern as
        MotionCorrectionTab's method combo."""
        volumetric = self.state.volumetric
        model = self.selector.model()
        for i, (name, _fn, vfn) in enumerate(_PROJECTIONS):
            if name == _VOLUME_3D:
                usable = volumetric
            elif volumetric:
                usable = vfn is not None
            else:
                usable = True
            model.item(i).setEnabled(usable)
            model.item(i).setToolTip("" if usable else ("Volumetric data only." if name == _VOLUME_3D
                                                        else "No volumetric form yet."))
        if not self._entry_usable(self.selector.currentText()):
            self.selector.setCurrentText("Mean")

    def _entry_usable(self, name: str) -> bool:
        if name == _VOLUME_3D:
            return self.state.volumetric
        if self.state.volumetric:
            return self._fns_volumetric.get(name) is not None
        return True

    def _show_selected(self, name: str) -> None:
        movie = self.state.active_data()
        if movie is None or not self._entry_usable(name):
            return

        if name == _VOLUME_3D:
            self._show_volume_3d()
            return

        self._stack.setCurrentWidget(self.view)
        if name in self._cache:
            self._set_view(self._cache[name])
            return

        fn = self._fns_volumetric[name] if self.state.volumetric else self._fns[name]
        self.selector.setEnabled(False)
        self.worker = run_worker(
            self.busy_bar, f"Computing {name}...", fn, movie,
            on_success=lambda result, name=name: self._on_computed(name, result),
            on_failure=self._on_failed,
        )

    def _show_volume_3d(self) -> None:
        """The 3D view renders the time-mean volume -- the same (L, W, D)
        array as the "Mean" projection, so reuse/populate that cache."""
        self._stack.setCurrentWidget(self.volume_view)
        if "Mean" in self._cache:
            self.volume_view.set_volume(self._cache["Mean"])
            return
        self.selector.setEnabled(False)
        self.worker = run_worker(
            self.busy_bar, "Computing time-mean volume...", mean_projection_volumetric, self.state.active_data(),
            on_success=lambda result: self._on_computed("Mean", result),
            on_failure=self._on_failed,
        )

    def _set_view(self, projection: np.ndarray) -> None:
        """A 2D (H, W) projection shows directly; a volumetric (L, W, D)
        one is shown depth-first so pg.ImageView's slider scrolls through
        depth."""
        if projection.ndim == 3:
            self.view.setImage(np.moveaxis(projection, -1, 0))  # (L, W, D) -> (D, L, W)
        else:
            self.view.setImage(projection)

    def _on_computed(self, name: str, result: np.ndarray) -> None:
        self._cache[name] = result
        self.busy_bar.stop("")
        self.selector.setEnabled(True)
        current = self.selector.currentText()
        if current == name:
            self._set_view(result)
        elif current == _VOLUME_3D and name == "Mean":
            self.volume_view.set_volume(result)

    def _on_failed(self, message: str) -> None:
        self.busy_bar.stop("Computation failed.")
        self.selector.setEnabled(True)
        QMessageBox.critical(self, "Projection failed", message)
