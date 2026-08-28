"""Motion Correction tab: rigid, piecewise-rigid, or PatchWarp-style
piecewise-affine registration (orbit.motion_correction.motion_correct),
with quality metrics computed and shown alongside the before/after
images rather than as an afterthought -- mMD/mCM/ECC (PatchWarp) plus
singular-value-spectrum tightening and spatial PC maps (halo/crescent
inspection).

Same Apply-produces-a-candidate / Commit-makes-it-active pattern as
every StageTab -- see that module's docstring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from orbit._volumetric import depth_project
from orbit.motion_correction import motion_correct, rigid_motion_correct, patch_motion_correct
from orbit.motion_correction_3d import rigid_motion_correct_3d
from orbit.motion_metrics import (
    enhanced_correlation_coefficient,
    mean_correlation_to_reference,
    mean_max_intensity_difference,
    spatiotemporal_svd,
)

from ..fits_io import create_fits_memmap
from ..io import is_memmap, preview_slice
from ..state import AppState
from ..theme import add_legend
from ..volumetric_io import preview_slice_volumetric
from ..widgets import ImageSlideshow, ParametersDialog, confirm_recompute, make_spinbox
from ..workers import run_worker
from .stage_tab import StageTab

_DEFAULT_N_COMPONENTS = 20

# Short, pipeline-breadcrumb-friendly names per method -- "Patch Warp"
# matches how the user refers to it, not the combo box's longer label.
_METHOD_LABELS = {"rigid": "Rigid", "patch": "Patch-based", "patchwarp": "Patch Warp"}


def _run_and_assess(movie: np.ndarray, method: str, n_components: int, **kwargs) -> dict:
    """Runs off the GUI thread: registration plus every metric needed to
    populate the tab, packaged into one dict."""
    registered, shifts, template, initial_template = motion_correct(movie, method=method, **kwargs)
    sv_before, _pc_before = spatiotemporal_svd(movie, n_components=n_components)
    sv_after, pc_after = spatiotemporal_svd(registered, n_components=n_components)
    return {
        "registered": registered,
        "shifts": shifts,
        "template": template,
        "initial_template": initial_template,
        "mmd": mean_max_intensity_difference(movie, registered),
        "mcm_before": mean_correlation_to_reference(movie),
        "mcm_after": mean_correlation_to_reference(registered),
        "ecc": enhanced_correlation_coefficient(initial_template, template),
        "sv_before": sv_before,
        "sv_after": sv_after,
        "pc_after": pc_after,
    }


def _run_and_assess_3d(movie: np.ndarray, n_components: int, **kwargs) -> dict:
    """Volumetric (T, L, W, D) counterpart of _run_and_assess -- runs the
    3D rigid registration, then depth-projects both the input and the
    registered volume to (L, W, T) so every 2D display widget and
    orbit.motion_metrics function below can be reused unmodified rather
    than duplicated in 3D (see orbit._volumetric.depth_project). ECC
    compares the raw (L, W, D) templates directly instead -- it's just a
    flattened Pearson correlation, already dimension-agnostic."""
    registered, shifts, template, initial_template = rigid_motion_correct_3d(movie, **kwargs)
    projected_before = depth_project(movie)
    projected_after = depth_project(registered)
    sv_before, _pc_before = spatiotemporal_svd(projected_before, n_components=n_components)
    sv_after, pc_after = spatiotemporal_svd(projected_after, n_components=n_components)
    return {
        "registered": projected_after,  # (L, W, T) -- for display/metrics only
        "registered_3d": registered,  # (T, L, W, D) -- the real array, for Commit
        "shifts": shifts,
        "template": template,
        "initial_template": initial_template,
        "mmd": mean_max_intensity_difference(projected_before, projected_after),
        "mcm_before": mean_correlation_to_reference(projected_before),
        "mcm_after": mean_correlation_to_reference(projected_after),
        "ecc": enhanced_correlation_coefficient(initial_template, template),
        "sv_before": sv_before,
        "sv_after": sv_after,
        "pc_after": pc_after,
    }


class MotionCorrectionTab(StageTab):
    _stage_name = "Motion correction"
    _result_key = "registered"
    _stage_key = "motion_correction"

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, apply_label="Apply Motion Correction", parent=parent)
        self.on_modality_changed()  # reflects state.volumetric's initial value, if already set

    def _build_controls_row(self) -> QHBoxLayout:
        # All per-algorithm parameters live in the ParametersDialog popup
        # below rather than sprawling across the tab -- this row is the
        # only thing always visible: which algorithm, its parameters
        # button, and the two actions (Apply / Commit).
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Rigid", "Patch-based (non-rigid)", "PatchWarp (piecewise-affine)"])

        self.max_shift_spin = make_spinbox(0, 200, 15, decimal=True)
        self.upsample_spin = make_spinbox(1, 200, 20)
        self.n_iter_spin = make_spinbox(1, 20, 1)
        self.grid_size_spin = make_spinbox(4, 2000, 32)
        self.patchwarp_grid_spin = make_spinbox(1, 16, 4)
        self.overlap_frac_spin = make_spinbox(0.0, 0.5, 0.1, step=0.05, decimal=True)
        self.ecc_iterations_spin = make_spinbox(1, 500, 30)
        self.pyramid_levels_spin = make_spinbox(1, 4, 1)
        self.pc_count_spin = make_spinbox(1, 100, _DEFAULT_N_COMPONENTS)

        self.params_dialog = ParametersDialog(title="Motion Correction Parameters", parent=self)
        self.params_dialog.add_row("max_shift (rigid stage, all methods)", self.max_shift_spin)
        self.params_dialog.add_row("upsample_factor (rigid/patch-based only)", self.upsample_spin)
        self.params_dialog.add_row("n_iter (rigid stage, all methods)", self.n_iter_spin)
        self.params_dialog.add_row("grid_size in pixels (patch-based only)", self.grid_size_spin)
        self.params_dialog.add_row("patch grid (N x N, PatchWarp only)", self.patchwarp_grid_spin)
        self.params_dialog.add_row("overlap_frac (PatchWarp only)", self.overlap_frac_spin)
        self.params_dialog.add_row("ecc_iterations (PatchWarp only)", self.ecc_iterations_spin)
        self.params_dialog.add_row("pyramid_levels (PatchWarp only)", self.pyramid_levels_spin)
        self.params_dialog.add_row("number of spatial PCs (spectrum + slideshow)", self.pc_count_spin)

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
        self.metrics_label = QLabel("Run motion correction to see quality metrics.")
        self.panel.add_metric_widget(self.metrics_label)

        self.sv_plot = pg.PlotWidget(title="Singular value spectrum (tighter after = better)")
        add_legend(self.sv_plot)
        self.sv_plot.setLabel("bottom", "singular value number")
        self.sv_plot.setLabel("left", "normalized singular value")
        self.panel.add_metric_widget(self.sv_plot)

        pc_container = QWidget()
        pc_layout = QVBoxLayout(pc_container)
        pc_layout.addWidget(QLabel("Top spatial PCs (candidate) -- halos/crescents mean residual motion"))
        self.pc_slideshow = ImageSlideshow()
        pc_layout.addWidget(self.pc_slideshow)
        self.panel.add_metric_widget(pc_container)

    def _current_fingerprint(self) -> dict:
        """Only the fields the *selected* method actually uses -- e.g.
        Rigid never touches patchwarp_grid/ecc_iterations/pyramid_levels,
        so recording them (alongside every other method's own unused
        fields) would misrepresent what this commit actually ran with,
        both in a saved session and in the "Generate Report" PDF. Mirrors
        _method_and_kwargs's own per-method branching, kept separate
        from it since that one also needs init_batch/max_dev (derived,
        not user-set) and algorithm-facing key names (rigid_max_shift)
        rather than these widget-facing ones."""
        method_text = self.method_combo.currentText()
        params = dict(
            method=method_text, max_shift=self.max_shift_spin.value(), n_iter=self.n_iter_spin.value(),
            n_components=self.pc_count_spin.value(),
        )
        if method_text.startswith("Rigid"):
            params["upsample_factor"] = self.upsample_spin.value()
        elif method_text.startswith("Patch-based"):
            params["upsample_factor"] = self.upsample_spin.value()
            params["grid_size"] = self.grid_size_spin.value()
        else:  # PatchWarp
            params["patchwarp_grid"] = self.patchwarp_grid_spin.value()
            params["overlap_frac"] = self.overlap_frac_spin.value()
            params["ecc_iterations"] = self.ecc_iterations_spin.value()
            params["pyramid_levels"] = self.pyramid_levels_spin.value()
        return params

    def restore_params(self, params: dict) -> None:
        if "method" in params:
            self.method_combo.setCurrentText(params["method"])
        for key, spin in (
            ("max_shift", self.max_shift_spin), ("upsample_factor", self.upsample_spin),
            ("n_iter", self.n_iter_spin), ("grid_size", self.grid_size_spin),
            ("patchwarp_grid", self.patchwarp_grid_spin), ("overlap_frac", self.overlap_frac_spin),
            ("ecc_iterations", self.ecc_iterations_spin), ("pyramid_levels", self.pyramid_levels_spin),
            ("n_components", self.pc_count_spin),
        ):
            if key in params:
                spin.setValue(params[key])

    def _extract_metrics(self, result: dict) -> dict:
        return dict(mmd=result["mmd"], mcm_before=result["mcm_before"], mcm_after=result["mcm_after"], ecc=result["ecc"])

    def _method_and_kwargs(self, init_batch: int) -> tuple[str, dict]:
        """The algorithm name + kwargs _start_worker/_chunked_commit both
        need, built from the current widget values. ``init_batch`` is
        passed in rather than derived from a movie array here, since
        Apply (preview, <=5000 frames) and a memmap Commit (the whole
        movie, which could be far larger) need very different bounds --
        the template bootstrap must never scan an entire huge movie."""
        method_text = self.method_combo.currentText()
        if method_text.startswith("Rigid"):
            method = "rigid"
        elif method_text.startswith("Patch-based"):
            method = "patch"
        else:
            method = "patchwarp"

        max_shift = self.max_shift_spin.value()
        n_iter = self.n_iter_spin.value()

        if method in ("rigid", "patch"):
            kwargs = dict(
                max_shift=max_shift, upsample_factor=self.upsample_spin.value(), n_iter=n_iter,
                init_batch=init_batch,
            )
            if method == "patch":
                kwargs["grid_size"] = self.grid_size_spin.value()
                kwargs["max_dev"] = max(1.0, max_shift / 3)
        else:
            kwargs = dict(
                grid_size=self.patchwarp_grid_spin.value(), overlap_frac=self.overlap_frac_spin.value(),
                rigid_max_shift=max_shift, rigid_n_iter=n_iter, ecc_iterations=self.ecc_iterations_spin.value(),
                pyramid_levels=self.pyramid_levels_spin.value(),
            )
        return method, kwargs

    def _start_worker(self, movie: np.ndarray) -> None:
        method, kwargs = self._method_and_kwargs(init_batch=movie.shape[-1])
        self._pending_step_label = _METHOD_LABELS[method]
        kwargs["n_components"] = self.pc_count_spin.value()

        self.worker = run_worker(
            self.busy_bar, "Running motion correction and metrics (this can take a while)...",
            _run_and_assess, movie, method, on_success=self._on_finished, on_failure=self._on_failed, **kwargs,
        )

    def _chunked_commit(self, source: np.ndarray, output_path: Path) -> np.ndarray:
        if self.state.volumetric:
            return self._chunked_commit_volumetric(source, output_path)
        method, kwargs = self._method_and_kwargs(init_batch=min(source.shape[-1], 5000))
        if method == "patchwarp":
            raise NotImplementedError(
                "PatchWarp doesn't yet support committing a memory-mapped movie -- use Rigid or "
                "Patch-based for very large files, or turn off memory mapping on the Load tab."
            )
        kwargs["bin_width"] = self._chunk_frames
        output = create_fits_memmap(output_path, source.shape, np.float32)
        fn = rigid_motion_correct if method == "rigid" else patch_motion_correct
        fn(source, output=output, **kwargs)
        output.flush()
        return output

    def _render_result(self, result: dict) -> None:
        # preview_slice/preview_slice_volumetric bound this to the same
        # (<=5000-frame) preview Apply actually ran against -- self._input_movie
        # can be a much longer memmap movie now that Commit re-runs against the
        # whole thing, and a full mean over that would force a full read just
        # to draw the "before" thumbnail.
        if self.state.volumetric:
            before = depth_project(preview_slice_volumetric(self._input_movie))
        else:
            before = preview_slice(self._input_movie)
        self.panel.before_view.setImage(before.mean(axis=2))
        self.panel.after_view.setImage(result["registered"].mean(axis=2))
        self.panel.set_after_movie(result["registered"])

        self.metrics_label.setText(
            f"mMD: {result['mmd']:.2f}\n"
            f"self-mCM before -> after: {result['mcm_before']:.3f} -> {result['mcm_after']:.3f}\n"
            f"ECC(initial -> final template): {result['ecc']:.3f}"
        )

        # Both curves normalized to "before"'s own peak, so "after" tightening
        # (or not) reads directly off the same 0-1 scale "before" is plotted on.
        sv_before = result["sv_before"]
        norm = sv_before.max() if sv_before.max() > 0 else 1.0
        self.sv_plot.clear()
        self.sv_plot.plot(sv_before / norm, pen="r", name="Before")
        self.sv_plot.plot(result["sv_after"] / norm, pen="g", name="After")

        self.pc_slideshow.set_stack(result["pc_after"])

    # -- Volumetric (state.volumetric) path -----------------------------
    #
    # A totally separate branch through each entry point below -- same
    # pattern LoadTab._browse_folder established for the volumetric
    # loader: check self.state.volumetric first, fall through to the
    # unmodified 2D behavior above when it's off. Only Rigid has a 3D
    # implementation (orbit.motion_correction_3d.rigid_motion_correct_3d),
    # so the method combo is restricted to it whenever volumetric data is
    # in play (see on_modality_changed). Registration always runs against
    # the true (T, L, W, D) volume; depth_project (see _run_and_assess_3d)
    # is only ever used to feed a display-friendly (L, W, T) projection
    # into the existing 2D panel/metrics widgets above, unmodified.

    def on_modality_changed(self) -> None:
        """Reacts to the Load tab's Volumetric toggle (wired in app.py) --
        disables Patch-based/PatchWarp (no 3D implementation yet) and
        forces the combo onto Rigid whenever volumetric data is in play."""
        volumetric = self.state.volumetric
        model = self.method_combo.model()
        for i in (1, 2):  # "Patch-based (non-rigid)", "PatchWarp (piecewise-affine)"
            item = model.item(i)
            item.setEnabled(not volumetric)
            item.setToolTip("Not yet available for volumetric data." if volumetric else "")
        if volumetric and self.method_combo.currentIndex() != 0:
            self.method_combo.setCurrentIndex(0)

    def on_data_loaded(self) -> None:
        if self.state.volumetric:
            self._on_volumetric_data_loaded()
            return
        super().on_data_loaded()

    def _on_volumetric_data_loaded(self) -> None:
        movie = self.state.active_data()
        self.commit_controls.set_apply_enabled(movie is not None)
        self.commit_controls.set_commit_enabled(False)
        self._pending_result = None
        self._pending_step_label = None
        self._last_run = None
        self._on_data_reset()
        if movie is not None:
            projected = depth_project(preview_slice_volumetric(movie))
            self.panel.before_view.setImage(projected.mean(axis=2))
            self.panel.set_before_movie(projected)
            self.status_label.setText(f"Ready. shape={movie.shape} (volumetric)")

    def _apply(self) -> None:
        if self.state.volumetric:
            self._apply_volumetric()
            return
        super()._apply()

    def _apply_volumetric(self) -> None:
        movie = self.state.active_data()
        if movie is None:
            QMessageBox.warning(self, "No data", "Load data on the Load tab first.")
            return

        params = self._current_fingerprint()
        fingerprint = (id(movie), tuple(sorted(params.items())))
        if fingerprint == self._last_run:
            message = f"{self._stage_name} was already run with these exact parameters on this data."
            if not confirm_recompute(self, message):
                return

        self._pending_fingerprint = fingerprint
        self._pending_params = params
        self._input_movie = movie
        self.commit_controls.set_apply_enabled(False)
        self.commit_controls.set_commit_enabled(False)
        self._start_worker_volumetric(preview_slice_volumetric(movie))

    def _start_worker_volumetric(self, movie: np.ndarray) -> None:
        self._pending_step_label = "Rigid (3D)"
        self.worker = run_worker(
            self.busy_bar, "Running 3D motion correction and metrics (this can take a while)...",
            _run_and_assess_3d, movie, self.pc_count_spin.value(),
            max_shift=self.max_shift_spin.value(), upsample_factor=self.upsample_spin.value(),
            n_iter=self.n_iter_spin.value(), init_batch=movie.shape[0],
            on_success=self._on_finished, on_failure=self._on_failed,
        )

    def _commit(self) -> None:
        if self._pending_result is None:
            return
        if is_memmap(self._input_movie):
            self._start_chunked_commit()
            return
        if self.state.volumetric:
            self._finish_commit(self._pending_result["registered_3d"])
            return
        self._finish_commit(self._pending_result[self._result_key])

    def _chunked_commit_volumetric(self, source: np.ndarray, output_path: Path) -> np.ndarray:
        output = create_fits_memmap(output_path, source.shape, np.float32)
        rigid_motion_correct_3d(
            source, output=output, bin_width=self._chunk_frames,
            max_shift=self.max_shift_spin.value(), upsample_factor=self.upsample_spin.value(),
            n_iter=self.n_iter_spin.value(), init_batch=min(source.shape[0], 5000),
        )
        output.flush()
        return output
