"""Detrending tab: divides the movie by a slowly-varying trend of its
own field-of-view-average intensity (orbit.detrending), correcting slow
photobleaching-style drift before Normalization sees the data. Two
methods, same "params live in the group matching the selected Method"
convention as Mask/Denoising/Source Extraction's own multi-method tabs:

- "Running percentile": a trailing running-percentile trend (percentile/
  window parameters).
- "Exponential decay (Huber)": a single global exponential a*exp(-t/b)
  fit via a robust asymmetric-Huber loss (delta_pos_sigma/delta_neg_sigma
  parameters) -- see orbit.detrending.fit_exponential_trend's own
  docstring for the full rationale.

Same Apply-produces-a-candidate / Commit-makes-it-active pattern as
every StageTab -- see that module's docstring. Unlike every other
StageTab, its main figure isn't a before/after image pair -- it's a
single line plot of the raw field-of-view-average trace and its trend,
since that's what "detrending" actually is (see
StageTab._build_panel/_show_before_preview, overridden below).

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
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from orbit.detrending import (
    detrend_movie,
    detrend_movie_3d,
    detrend_movie_3d_exponential,
    detrend_movie_exponential,
    fit_exponential_trend,
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

_METHODS = (
    ("Running percentile", "percentile"),
    ("Exponential decay (Huber)", "exponential"),
)
_METHOD_KEYS = dict(_METHODS)
_METHOD_LABELS = {key: label for label, key in _METHODS}


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


def _run_detrend_exponential(
    movie: np.ndarray, delta_pos_sigma: float, delta_neg_sigma: float, mask: np.ndarray | None,
) -> dict:
    """Runs off the GUI thread. Exponential-decay counterpart of
    _run_detrend -- ``a``/``b`` (the fitted initial brightness and decay
    time constant) ride along in the result dict purely for display,
    same as detrend_movie_exponential's own extra return values."""
    corrected, trace, trend, a, b = detrend_movie_exponential(
        movie, delta_pos_sigma=delta_pos_sigma, delta_neg_sigma=delta_neg_sigma, mask=mask,
    )
    return {"corrected": corrected, "trace": trace, "trend": trend, "a": a, "b": b}


def _run_detrend_exponential_3d(
    movie: np.ndarray, delta_pos_sigma: float, delta_neg_sigma: float, mask: np.ndarray | None,
) -> dict:
    """Volumetric counterpart of _run_detrend_exponential -- see
    _run_detrend_3d's own docstring for why no separate 3D result key is
    needed here either."""
    corrected, trace, trend, a, b = detrend_movie_3d_exponential(
        movie, delta_pos_sigma=delta_pos_sigma, delta_neg_sigma=delta_neg_sigma, mask=mask,
    )
    return {"corrected": corrected, "trace": trace, "trend": trend, "a": a, "b": b}


def _compute_full_trend(trace: np.ndarray, params: dict) -> np.ndarray:
    """Builds the (T,) trend array from a fully-streamed FOV-average
    trace, dispatching to whichever method Apply used -- shared by
    _chunked_commit/_chunked_commit_volumetric, which both need this
    same trend regardless of 2D/3D (the streaming-trace step above it
    differs by dimensionality, this doesn't)."""
    if params["method"] == "percentile":
        return running_percentile_trend(trace, params["percentile"], params["window"])
    a, b = fit_exponential_trend(trace, params["delta_pos_sigma"], params["delta_neg_sigma"])
    t = np.arange(len(trace), dtype=np.float64)
    return a * np.exp(-t / b)


class DetrendingTab(StageTab):
    _stage_name = "Detrending"
    _result_key = "corrected"
    _result_key_3d = "corrected"  # see _run_detrend_3d's docstring -- no separate display-only key needed here
    _stage_key = "detrending"
    _supports_volumetric = True

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, apply_label="Apply Detrending", parent=parent)

    def _build_controls_row(self) -> QHBoxLayout:
        self.method_combo = QComboBox()
        self.method_combo.addItems([label for label, _key in _METHODS])
        self.method_combo.currentTextChanged.connect(self._update_visible_params)

        self.percentile_spin = make_spinbox(0.0, 100.0, 8.0, step=1.0, decimal=True)
        self.window_spin = make_spinbox(1, 100000, 200)
        # Huber transition thresholds, each a multiple of the trace's own
        # estimated noise std -- see fit_exponential_trend's own
        # docstring for why this asymmetry (small positive-side
        # threshold, larger negative-side one) makes the fit track the
        # trace's lower envelope rather than its mean.
        self.delta_pos_sigma_spin = make_spinbox(0.01, 100.0, 1.0, step=0.1, decimal=True)
        self.delta_neg_sigma_spin = make_spinbox(0.01, 100.0, 3.0, step=0.1, decimal=True)

        self.params_dialog = ParametersDialog(title="Detrending Parameters", parent=self)
        self.params_dialog.add_row("percentile (X)", self.percentile_spin, group="percentile")
        self.params_dialog.add_row("window size in frames (N)", self.window_spin, group="percentile")
        self.params_dialog.add_row(
            "Huber threshold, positive side (x noise std)", self.delta_pos_sigma_spin, group="exponential",
        )
        self.params_dialog.add_row(
            "Huber threshold, negative side (x noise std)", self.delta_neg_sigma_spin, group="exponential",
        )
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

    def _update_visible_params(self, label: str) -> None:
        """Only the fields relevant to the selected method are shown in
        the Parameters popup -- same convention as MaskTab's own
        multi-method parameter groups."""
        method = _METHOD_KEYS[label]
        self.params_dialog.show_only_group(method)

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
        method = _METHOD_KEYS[self.method_combo.currentText()]
        params: dict = {"method": method}
        if method == "percentile":
            params["percentile"] = self.percentile_spin.value()
            params["window"] = self.window_spin.value()
        else:
            params["delta_pos_sigma"] = self.delta_pos_sigma_spin.value()
            params["delta_neg_sigma"] = self.delta_neg_sigma_spin.value()
        return params

    def restore_params(self, params: dict) -> None:
        method = params.get("method", "percentile")
        if method in _METHOD_LABELS:
            self.method_combo.setCurrentText(_METHOD_LABELS[method])
        if "percentile" in params:
            self.percentile_spin.setValue(params["percentile"])
        if "window" in params:
            self.window_spin.setValue(params["window"])
        if "delta_pos_sigma" in params:
            self.delta_pos_sigma_spin.setValue(params["delta_pos_sigma"])
        if "delta_neg_sigma" in params:
            self.delta_neg_sigma_spin.setValue(params["delta_neg_sigma"])

    def _start_worker(self, movie: np.ndarray) -> None:
        kwargs = self._current_fingerprint()
        method = kwargs.pop("method")
        if method == "percentile":
            self._pending_step_label = f"Detrend (percentile={kwargs['percentile']:g}, window={kwargs['window']})"
            self.worker = run_worker(
                self.busy_bar, "Computing detrending...", _run_detrend, movie, mask=self.state.mask,
                on_success=self._on_finished, on_failure=self._on_failed, **kwargs,
            )
        else:
            self._pending_step_label = (
                f"Detrend (exponential decay, Huber +{kwargs['delta_pos_sigma']:g}/-{kwargs['delta_neg_sigma']:g} sigma)"
            )
            self.worker = run_worker(
                self.busy_bar, "Fitting exponential decay...", _run_detrend_exponential, movie, mask=self.state.mask,
                on_success=self._on_finished, on_failure=self._on_failed, **kwargs,
            )

    def _start_worker_volumetric(self, movie: np.ndarray) -> None:
        kwargs = self._current_fingerprint()
        method = kwargs.pop("method")
        if method == "percentile":
            self._pending_step_label = f"Detrend (percentile={kwargs['percentile']:g}, window={kwargs['window']})"
            self.worker = run_worker(
                self.busy_bar, "Computing detrending...", _run_detrend_3d, movie, mask=self.state.mask,
                on_success=self._on_finished, on_failure=self._on_failed, **kwargs,
            )
        else:
            self._pending_step_label = (
                f"Detrend (exponential decay, Huber +{kwargs['delta_pos_sigma']:g}/-{kwargs['delta_neg_sigma']:g} sigma)"
            )
            self.worker = run_worker(
                self.busy_bar, "Fitting exponential decay...", _run_detrend_exponential_3d, movie, mask=self.state.mask,
                on_success=self._on_finished, on_failure=self._on_failed, **kwargs,
            )

    def _render_result(self, result: dict) -> None:
        self.trace_plot.clear()
        self.trace_plot.plot(result["trace"], pen="c", name="Raw")
        if "a" in result:
            trend_name = f"Trend (exponential decay: a={result['a']:.3g}, b={result['b']:.1f} frames)"
        else:
            trend_name = "Trend (running percentile)"
        self.trace_plot.plot(result["trend"], pen="m", name=trend_name)

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
        trend = _compute_full_trend(trace, kwargs)
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
        trend = _compute_full_trend(trace, kwargs)
        scale = trend / trend.mean()

        output = create_fits_memmap(output_path, source.shape, np.float32)
        for t0 in range(0, n_frames, self._chunk_frames):
            t1 = min(t0 + self._chunk_frames, n_frames)
            chunk = np.asarray(source[t0:t1], dtype=np.float32)
            output[t0:t1] = chunk / scale[t0:t1, None, None, None]
        output.flush()
        return output
