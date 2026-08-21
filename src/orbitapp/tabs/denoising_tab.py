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

from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton

from orbit.denoising import (
    denoise_gaussian,
    denoise_median,
    denoise_wavelet_space,
    denoise_wavelet_time,
    residual_energy_fraction,
)
from orbit.pca_denoise import pca_denoise
from orbit.projections import local_correlation_projection
from orbit.qc_traces import qc_trace_samples

from ..fits_io import create_fits_memmap
from ..io import preview_slice
from ..state import AppState
from ..widgets import ParametersDialog, QCPlotGrid, add_location_markers, make_spinbox, pixels_to_data_pos, split_by_kind
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
    ("PCA Denoising", "pca"),
)
_ALGORITHM_KEYS = dict(_ALGORITHMS)
_DENOISE_FUNCS = {
    "wavelet_time": denoise_wavelet_time,
    "wavelet_space": denoise_wavelet_space,
    "gaussian": denoise_gaussian,
    "median": denoise_median,
    "pca": pca_denoise,
}
# Which ParametersDialog group each algorithm's fields belong to --
# several algorithms can share a group (both wavelet domains use the
# same wavelet/level/threshold fields).
_GROUP_BY_ALGORITHM = {
    "wavelet_time": "wavelet",
    "wavelet_space": "wavelet",
    "gaussian": "gaussian",
    "median": "median",
    "pca": "pca",
}
# Pipeline-breadcrumb label per algorithm -- specific enough to tell
# denoising methods apart in the header ("Load > Gaussian Denoising >
# ...") rather than a single generic "Denoise" for all of them.
_PIPELINE_LABELS = {
    "wavelet_time": "Wavelet Denoising (Temporal)",
    "wavelet_space": "Wavelet Denoising (Spatial)",
    "gaussian": "Gaussian Denoising",
    "median": "Median Filtering",
    "pca": "PCA Denoising",
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
    _stage_key = "denoising"

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
        self.pca_n_components_spin = make_spinbox(1, 500, 20)
        self.pca_block_size_spin = make_spinbox(10, 2000, 250)
        self.pca_block_frames_spin = make_spinbox(10, 50000, 5000)
        self.pca_spatial_overlap_spin = make_spinbox(0, 500, 30)
        self.pca_temporal_overlap_spin = make_spinbox(0, 20000, 500)

        self.params_dialog = ParametersDialog(title="Denoising Parameters", parent=self)
        self.params_dialog.add_row("wavelet", self.wavelet_combo, group="wavelet")
        self.params_dialog.add_row("level", self.level_spin, group="wavelet")
        self.params_dialog.add_row("threshold method", self.threshold_combo, group="wavelet")
        self.params_dialog.add_row("spatial width (pixels, 0 = temporal only)", self.gaussian_spatial_spin, group="gaussian")
        self.params_dialog.add_row("temporal width (frames, 0 = spatial only)", self.gaussian_temporal_spin, group="gaussian")
        self.params_dialog.add_row("space_window", self.median_space_spin, group="median")
        self.params_dialog.add_row("time_window", self.median_time_spin, group="median")
        self.params_dialog.add_row("number of components", self.pca_n_components_spin, group="pca")
        self.params_dialog.add_row("block size (pixels)", self.pca_block_size_spin, group="pca")
        self.params_dialog.add_row("block length (frames)", self.pca_block_frames_spin, group="pca")
        self.params_dialog.add_row("spatial overlap (pixels)", self.pca_spatial_overlap_spin, group="pca")
        self.params_dialog.add_row("temporal overlap (frames)", self.pca_temporal_overlap_spin, group="pca")
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

    def _current_fingerprint(self) -> dict:
        return dict(
            algorithm=self.method_combo.currentText(), wavelet=self.wavelet_combo.currentText(),
            level=self.level_spin.value(), threshold_method=self.threshold_combo.currentText(),
            spatial_sigma=self.gaussian_spatial_spin.value(), temporal_sigma=self.gaussian_temporal_spin.value(),
            space_window=self.median_space_spin.value(), time_window=self.median_time_spin.value(),
            pca_n_components=self.pca_n_components_spin.value(), pca_block_size=self.pca_block_size_spin.value(),
            pca_block_frames=self.pca_block_frames_spin.value(),
            pca_spatial_overlap=self.pca_spatial_overlap_spin.value(),
            pca_temporal_overlap=self.pca_temporal_overlap_spin.value(),
        )

    def restore_params(self, params: dict) -> None:
        if "algorithm" in params:
            self.method_combo.setCurrentText(params["algorithm"])
        if "threshold_method" in params:
            self.threshold_combo.setCurrentText(params["threshold_method"])
        if "wavelet" in params:
            self.wavelet_combo.setCurrentText(params["wavelet"])
        for key, spin in (
            ("level", self.level_spin), ("spatial_sigma", self.gaussian_spatial_spin),
            ("temporal_sigma", self.gaussian_temporal_spin), ("space_window", self.median_space_spin),
            ("time_window", self.median_time_spin), ("pca_n_components", self.pca_n_components_spin),
            ("pca_block_size", self.pca_block_size_spin), ("pca_block_frames", self.pca_block_frames_spin),
            ("pca_spatial_overlap", self.pca_spatial_overlap_spin),
            ("pca_temporal_overlap", self.pca_temporal_overlap_spin),
        ):
            if key in params:
                spin.setValue(params[key])

    def _extract_metrics(self, result: dict) -> dict:
        return dict(
            residual_energy_fraction=result["residual_energy_fraction"], corr_before=result["corr_before"],
            corr_after=result["corr_after"],
        )

    def _algorithm_and_kwargs(self) -> tuple[str, dict]:
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
        elif algorithm == "median":
            kwargs = dict(space_window=self.median_space_spin.value(), time_window=self.median_time_spin.value())
        else:
            block = self.pca_block_size_spin.value()
            kwargs = dict(
                n_components=self.pca_n_components_spin.value(), block_size=(block, block),
                block_frames=self.pca_block_frames_spin.value(),
                spatial_overlap=self.pca_spatial_overlap_spin.value(),
                temporal_overlap=self.pca_temporal_overlap_spin.value(),
            )
        return algorithm, kwargs

    def _start_worker(self, movie: np.ndarray) -> None:
        algorithm, kwargs = self._algorithm_and_kwargs()
        self._pending_step_label = _PIPELINE_LABELS[algorithm]

        self.worker = run_worker(
            self.busy_bar, "Running denoising and metrics (this can take a while)...",
            _run_and_assess, movie, algorithm, on_success=self._on_finished, on_failure=self._on_failed, **kwargs,
        )

    @staticmethod
    def _temporal_margin(algorithm: str, kwargs: dict) -> int:
        """Extra frames to read on each side of a chunk so a temporal
        filter's edge frames aren't computed from a truncated window --
        0 for algorithms with no temporal reach (wavelet_space, or
        gaussian/median with their temporal parameter left at its
        no-op default). Matches scipy.ndimage's own default truncate=4.0
        for the Gaussian case; verified numerically identical to the
        whole-movie result for both gaussian and median (see tests)."""
        if algorithm == "gaussian":
            temporal_sigma = kwargs["temporal_sigma"]
            return int(np.ceil(4 * temporal_sigma)) if temporal_sigma > 0 else 0
        if algorithm == "median":
            return kwargs["time_window"] // 2
        return 0

    def _chunked_commit(self, source: np.ndarray, output_path: Path) -> np.ndarray:
        algorithm, kwargs = self._algorithm_and_kwargs()
        if algorithm == "wavelet_time":
            raise NotImplementedError(
                "Wavelet Denoising (Temporal) needs each pixel's whole time series and can't be "
                "committed chunk-by-chunk against a memory-mapped movie -- use Wavelet (Spatial), "
                "Gaussian, or Median for very large files, or turn off memory mapping on the Load tab."
            )
        if algorithm == "pca":
            # Unlike the local-window filters below, PCA Denoising's
            # blocks are a *global* tiling of the whole movie, blended
            # where they overlap -- chunking Commit at any granularity
            # other than that exact same tiling would silently diverge
            # from what one whole-movie call produces (not just an
            # approximation of it), and reproducing that tiling exactly
            # would require accumulator arrays sized to the whole movie,
            # defeating chunked Commit's entire bounded-memory point.
            raise NotImplementedError(
                "PCA Denoising's blocks span a large, blended region of the whole movie and can't "
                "be committed chunk-by-chunk without changing the result -- use Wavelet (Spatial), "
                "Gaussian, or Median for very large memory-mapped files, or turn off memory "
                "mapping on the Load tab."
            )

        margin = self._temporal_margin(algorithm, kwargs)
        denoise_fn = _DENOISE_FUNCS[algorithm]
        T = source.shape[-1]
        output = create_fits_memmap(output_path, source.shape, np.float32)

        for t0 in range(0, T, self._chunk_frames):
            t1 = min(t0 + self._chunk_frames, T)
            pad_lo = min(margin, t0)
            pad_hi = min(margin, T - t1)
            chunk = np.asarray(source[:, :, t0 - pad_lo : t1 + pad_hi], dtype=np.float32)
            denoised_chunk = denoise_fn(chunk, **kwargs)
            output[:, :, t0:t1] = denoised_chunk[:, :, pad_lo : pad_lo + (t1 - t0)]

        output.flush()
        return output

    def _render_result(self, result: dict) -> None:
        self.panel.before_view.setImage(preview_slice(self._input_movie).mean(axis=2))
        self.panel.after_view.setImage(result["denoised"].mean(axis=2))
        self.panel.set_after_movie(result["denoised"])

        self.metrics_label.setText(
            f"Residual energy fraction: {result['residual_energy_fraction']:.3f}\n"
            f"Mean local correlation before -> after: "
            f"{result['corr_before']:.3f} -> {result['corr_after']:.3f}"
        )

        qc_traces = result["qc_traces"]
        marker_xs, marker_ys = pixels_to_data_pos(
            self.panel.before_view.getImageItem(),
            [s["row"] + 0.5 for s in qc_traces], [s["col"] + 0.5 for s in qc_traces],
        )
        self._location_markers.setData(marker_xs, marker_ys)
        peak_samples, low_samples = split_by_kind(qc_traces)
        self.trace_grid.fill(peak_samples, low_samples, _plot_trace)
