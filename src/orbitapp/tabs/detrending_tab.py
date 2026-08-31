"""Detrending tab: divides the movie by a running-percentile trend of
its own field-of-view-average intensity (orbit.detrending), correcting
slow photobleaching-style drift before Normalization sees the data.

Same Apply-produces-a-candidate / Commit-makes-it-active pattern as
every StageTab -- see that module's docstring. Unlike every other
StageTab, its main figure isn't a before/after image pair -- it's a
single line plot of the raw field-of-view-average trace and its
running-percentile trend, since that's what "detrending" actually is
(see StageTab._build_panel/_show_before_preview, overridden below).

Supports volumetric (state.volumetric) data too: the "field-of-view
average" is then the mean over each timepoint's whole (L, W, D) volume
(orbit.detrending.fov_average_trace_3d), not a 2D frame -- there's no
image to depth-project for display here (unlike every other volumetric-
aware StageTab), just a differently-computed 1D trace, so
_show_before_preview_volumetric only needs to swap which orbit function
it calls, not follow the StagePanel-shaped default at all.

Both the 2D and 3D trace functions also restrict the average to
state.mask's True pixels/voxels when a mask has been committed (Mask
tab) -- otherwise a large blank/background region would dilute a real
intensity change happening only in the imaged tissue -- and normalize
to the very first frame/volume's own mean, so the plotted trace and
trend read directly as "fraction of initial brightness" regardless of
the movie's absolute intensity scale (see fov_average_trace's
docstring: this normalization doesn't change the actual correction,
only the displayed numbers).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from orbit.detrending import (
    detrend_movie,
    detrend_movie_3d,
    fov_average_trace,
    fov_average_trace_3d,
    running_percentile_trend,
)

from ..fits_io import create_fits_memmap
from ..io import preview_slice
from ..state import AppState
from ..theme import add_legend
from ..volumetric_io import preview_slice_volumetric
from ..widgets import ParametersDialog, make_spinbox
from ..workers import run_worker
from .stage_tab import StageTab


def _run_detrend(movie: np.ndarray, percentile: float, window: int, mask: np.ndarray | None) -> dict:
    """Runs off the GUI thread."""
    corrected, trace, trend = detrend_movie(movie, percentile=percentile, window=window, mask=mask)
    return {"corrected": corrected, "trace": trace, "trend": trend}


def _run_detrend_3d(movie: np.ndarray, percentile: float, window: int, mask: np.ndarray | None) -> dict:
    """Volumetric (T, L, W, D) counterpart of _run_detrend. Unlike every
    other volumetric-aware StageTab, there's no separate depth-projected
    "for display" copy needed under a distinct key -- Detrending never
    shows an image at all (just the trace/trend plot, already a proper
    1D array either way), so _result_key_3d is the SAME key as
    _result_key below rather than its own "..._3d" name."""
    corrected, trace, trend = detrend_movie_3d(movie, percentile=percentile, window=window, mask=mask)
    return {"corrected": corrected, "trace": trace, "trend": trend}


class DetrendingTab(StageTab):
    _stage_name = "Detrending"
    _result_key = "corrected"
    _result_key_3d = "corrected"  # see _run_detrend_3d's docstring -- no separate display-only key needed here
    _stage_key = "detrending"
    _supports_volumetric = True

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, apply_label="Apply Detrending", parent=parent)

    def _build_controls_row(self) -> QHBoxLayout:
        # Only one method for now ("running percentile") -- X/N live in
        # the Parameters popup, same convention as every other stage.
        self.percentile_spin = make_spinbox(0.0, 100.0, 8.0, step=1.0, decimal=True)
        self.window_spin = make_spinbox(1, 100000, 200)

        self.params_dialog = ParametersDialog(title="Detrending Parameters", parent=self)
        self.params_dialog.add_row("percentile (X)", self.percentile_spin)
        self.params_dialog.add_row("window size in frames (N)", self.window_spin)

        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("Method: Running percentile"))
        self.params_btn = QPushButton("Parameters...")
        self.params_btn.clicked.connect(self.params_dialog.exec)
        controls_row.addWidget(self.params_btn)
        controls_row.addWidget(self.commit_controls)
        controls_row.addStretch()
        return controls_row

    def _build_panel(self, before_title: str, after_title: str) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.trace_plot = pg.PlotWidget(title="Field-of-view average intensity")
        add_legend(self.trace_plot)
        self.trace_plot.setLabel("bottom", "frame")
        self.trace_plot.setLabel("left", "relative intensity (frame 0 = 1)")
        layout.addWidget(self.trace_plot)
        return panel

    def _show_before_preview(self, movie: np.ndarray) -> None:
        # preview_slice bounds this to the first 5000 frames for a
        # memmap movie, same reasoning as every other stage's before-image.
        # state.mask (if a Mask stage has been committed) restricts the
        # average to kept pixels only -- see fov_average_trace's docstring.
        trace = fov_average_trace(preview_slice(movie), mask=self.state.mask)
        self.trace_plot.clear()
        self.trace_plot.plot(trace, pen="c", name="Raw")

    def _show_before_preview_volumetric(self, movie: np.ndarray) -> None:
        # preview_slice_volumetric bounds this the same way, per volume
        # rather than per frame; no depth projection needed here (unlike
        # the StagePanel default this overrides) since the plot is
        # already a 1D trace either way -- see the module docstring.
        trace = fov_average_trace_3d(preview_slice_volumetric(movie), mask=self.state.mask)
        self.trace_plot.clear()
        self.trace_plot.plot(trace, pen="c", name="Raw")

    def _build_metrics(self) -> None:
        pass  # the trace plot itself is the whole "metric" here -- nothing else to add

    def _current_fingerprint(self) -> dict:
        return dict(percentile=self.percentile_spin.value(), window=self.window_spin.value())

    def restore_params(self, params: dict) -> None:
        if "percentile" in params:
            self.percentile_spin.setValue(params["percentile"])
        if "window" in params:
            self.window_spin.setValue(params["window"])

    def _start_worker(self, movie: np.ndarray) -> None:
        kwargs = self._current_fingerprint()
        self._pending_step_label = f"Detrend (percentile={kwargs['percentile']:g}, window={kwargs['window']})"
        self.worker = run_worker(
            self.busy_bar, "Computing detrending...", _run_detrend, movie, mask=self.state.mask,
            on_success=self._on_finished, on_failure=self._on_failed, **kwargs,
        )

    def _start_worker_volumetric(self, movie: np.ndarray) -> None:
        kwargs = self._current_fingerprint()
        self._pending_step_label = f"Detrend (percentile={kwargs['percentile']:g}, window={kwargs['window']})"
        self.worker = run_worker(
            self.busy_bar, "Computing detrending...", _run_detrend_3d, movie, mask=self.state.mask,
            on_success=self._on_finished, on_failure=self._on_failed, **kwargs,
        )

    def _render_result(self, result: dict) -> None:
        self.trace_plot.clear()
        self.trace_plot.plot(result["trace"], pen="c", name="Raw")
        self.trace_plot.plot(result["trend"], pen="m", name="Trend (running percentile)")

    def _chunked_commit(self, source: np.ndarray, output_path: Path) -> np.ndarray:
        kwargs = self._current_fingerprint()
        mask = self.state.mask
        n_frames = source.shape[-1]

        # First pass: stream the WHOLE movie's per-frame FOV-average
        # intensity (a cheap H*W-element reduction per chunk). Unlike
        # Normalization's per-pixel baselines (fit once from a bounded
        # preview and reused as-is against every chunk), the trend here
        # is itself a function of time and must cover every frame the
        # correction will actually be applied to -- a preview-only fit
        # would have no trend value at all for frames past the preview.
        # (This trace is purely an internal intermediate for trend/scale
        # below -- unlike fov_average_trace's own first-frame
        # normalization, it's never displayed, so it's left in raw units.)
        trace = np.empty(n_frames, dtype=np.float64)
        for t0 in range(0, n_frames, self._chunk_frames):
            t1 = min(t0 + self._chunk_frames, n_frames)
            chunk = np.asarray(source[:, :, t0:t1], dtype=np.float64)
            trace[t0:t1] = chunk[mask].mean(axis=0) if mask is not None else chunk.mean(axis=(0, 1))
        trend = running_percentile_trend(trace, kwargs["percentile"], kwargs["window"])
        scale = trend / trend.mean()

        # Second pass: apply the now fully-known correction chunk by chunk.
        output = create_fits_memmap(output_path, source.shape, np.float32)
        for t0 in range(0, n_frames, self._chunk_frames):
            t1 = min(t0 + self._chunk_frames, n_frames)
            chunk = np.asarray(source[:, :, t0:t1], dtype=np.float32)
            output[:, :, t0:t1] = chunk / scale[None, None, t0:t1]
        output.flush()
        return output

    def _chunked_commit_volumetric(self, source: np.ndarray, output_path: Path) -> np.ndarray:
        kwargs = self._current_fingerprint()
        mask = self.state.mask
        n_frames = source.shape[0]

        trace = np.empty(n_frames, dtype=np.float64)
        for t0 in range(0, n_frames, self._chunk_frames):
            t1 = min(t0 + self._chunk_frames, n_frames)
            chunk = np.asarray(source[t0:t1], dtype=np.float64)
            trace[t0:t1] = chunk[:, mask].mean(axis=1) if mask is not None else chunk.mean(axis=(1, 2, 3))
        trend = running_percentile_trend(trace, kwargs["percentile"], kwargs["window"])
        scale = trend / trend.mean()

        output = create_fits_memmap(output_path, source.shape, np.float32)
        for t0 in range(0, n_frames, self._chunk_frames):
            t1 = min(t0 + self._chunk_frames, n_frames)
            chunk = np.asarray(source[t0:t1], dtype=np.float32)
            output[t0:t1] = chunk / scale[t0:t1, None, None, None]
        output.flush()
        return output
