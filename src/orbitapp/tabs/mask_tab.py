"""Mask tab: zeroes out background/uninformative pixels via an
auto-thresholded binary mask (orbit.masking), replicating the two
options pyGraFT's own "Mask" tab exposes -- "Auto-threshold Mask
(triangle method)" and "Clear Mask" -- fit into this app's own
Apply-produces-a-candidate / Commit-makes-it-active StageTab pattern
(see that module's docstring) rather than graftapp's more immediate
apply-on-click model.

"Auto-threshold Mask (triangle method)" is this tab's own Apply button
label -- it's the only real "algorithm" here, so it goes through the
usual worker-based Apply machinery. "Clear Mask" (an all-True, no-op
mask) is trivial and instant, so it runs synchronously rather than
through a background worker, but still produces a candidate that goes
through the same _on_finished/preview-before-commit review as the
auto-thresholded path -- clearing a mask is still an explicit,
reviewable pipeline step, not a silent reset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMessageBox, QPushButton

from orbit._volumetric import depth_project
from orbit.masking import apply_mask, apply_mask_3d, triangle_mask

from ..fits_io import create_fits_memmap
from ..io import is_memmap, preview_slice
from ..state import AppState
from ..volumetric_io import preview_slice_volumetric
from ..widgets import confirm_recompute
from ..workers import run_worker
from .stage_tab import StageTab


def _run_and_assess(movie: np.ndarray, action: str) -> dict:
    """Runs off the GUI thread for the "auto" path (called directly,
    synchronously, for the trivial "clear" path)."""
    if action == "clear":
        mask = np.ones(movie.shape[:2], dtype=bool)
    else:
        mask = triangle_mask(np.asarray(movie, dtype=np.float64).mean(axis=2))
    masked = apply_mask(movie, mask)
    return {"masked": masked, "mask": mask, "fraction_kept": float(mask.mean())}


def _run_and_assess_3d(movie: np.ndarray, action: str) -> dict:
    """Volumetric (T, L, W, D) counterpart of _run_and_assess. triangle_mask
    itself needs no volumetric variant (threshold_triangle has no shape
    assumptions, so it already works unchanged on a genuine 3D (L, W, D)
    mean-projection); only apply_mask (T-last) needed a T-first sibling."""
    if action == "clear":
        mask = np.ones(movie.shape[1:], dtype=bool)
    else:
        mask = triangle_mask(np.asarray(movie, dtype=np.float64).mean(axis=0))
    masked = apply_mask_3d(movie, mask)
    return {
        "masked": depth_project(masked),  # (L, W, T) -- for display only
        "masked_3d": masked,  # (T, L, W, D) -- the real array, for Commit
        "mask": mask,  # (L, W, D) bool -- persisted to state.mask on commit
        "fraction_kept": float(mask.mean()),
    }


class MaskTab(StageTab):
    _stage_name = "Mask"
    _result_key = "masked"
    _stage_key = "mask"

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, apply_label="Auto-threshold Mask (triangle method)", parent=parent)

    def _build_controls_row(self) -> QHBoxLayout:
        self.clear_btn = QPushButton("Clear Mask")
        self.clear_btn.clicked.connect(self._on_clear_clicked)

        controls_row = QHBoxLayout()
        controls_row.addWidget(self.commit_controls)
        controls_row.addWidget(self.clear_btn)
        controls_row.addStretch()
        return controls_row

    def _build_metrics(self) -> None:
        self.metrics_label = QLabel("Run masking to see results.")
        self.panel.add_metric_widget(self.metrics_label)

    def _on_clear_clicked(self) -> None:
        if self.state.volumetric:
            self._on_clear_clicked_volumetric()
            return
        movie = self.state.active_data()
        if movie is None:
            QMessageBox.warning(self, "No data", "Load data on the Load tab first.")
            return
        self._pending_fingerprint = (id(movie), "clear")
        self._pending_params = dict(action="clear")
        self._input_movie = movie
        self._pending_step_label = "Mask (cleared)"
        result = _run_and_assess(preview_slice(movie), "clear")
        self._on_finished(result)

    def _current_fingerprint(self) -> dict:
        # The auto-threshold path (this tab's own Apply button) has no
        # tunable parameters of its own -- "clear" goes through
        # _on_clear_clicked instead, which sets _pending_fingerprint/
        # _pending_params directly rather than via this method.
        return dict(action="auto")

    def restore_params(self, params: dict) -> None:
        pass  # nothing to restore -- see _current_fingerprint

    def _extract_metrics(self, result: dict) -> dict:
        return dict(fraction_kept=result["fraction_kept"])

    def _finish_commit(self, data: np.ndarray) -> None:
        # AppState.mask is the only place the boolean mask itself survives
        # past this tab -- committing here only ever replaces the active
        # movie with the already-masked one, so anything downstream that
        # needs to know WHICH pixels were kept (e.g. GraFT on volumetric
        # data) has to read it from here.
        self.state.mask = self._pending_result["mask"]
        super()._finish_commit(data)

    def _start_worker(self, movie: np.ndarray) -> None:
        self._pending_step_label = "Mask (auto-threshold)"
        self.worker = run_worker(
            self.busy_bar, "Computing mask...",
            _run_and_assess, movie, "auto", on_success=self._on_finished, on_failure=self._on_failed,
        )

    def _chunked_commit(self, source: np.ndarray, output_path: Path) -> np.ndarray:
        if self.state.volumetric:
            return self._chunked_commit_volumetric(source, output_path)
        # The mask is fit once from the same <=5000-frame preview Apply
        # (or Clear Mask) already used, then applied as a fixed
        # per-pixel gate to every chunk of the full movie -- same
        # "fit on the preview sample" approximation normalization_tab.py
        # uses for its own baselines.
        action = self._pending_params.get("action", "auto")
        if action == "clear":
            mask = np.ones(source.shape[:2], dtype=bool)
        else:
            preview = preview_slice(source)
            mask = triangle_mask(np.asarray(preview, dtype=np.float64).mean(axis=2))

        T = source.shape[-1]
        output = create_fits_memmap(output_path, source.shape, np.float32)
        for t0 in range(0, T, self._chunk_frames):
            t1 = min(t0 + self._chunk_frames, T)
            chunk = np.asarray(source[:, :, t0:t1], dtype=np.float32)
            output[:, :, t0:t1] = apply_mask(chunk, mask)
        output.flush()
        return output

    def _render_result(self, result: dict) -> None:
        if self.state.volumetric:
            before = depth_project(preview_slice_volumetric(self._input_movie))
        else:
            before = preview_slice(self._input_movie)
        self.panel.before_view.setImage(before.mean(axis=2))
        self.panel.after_view.setImage(result["masked"].mean(axis=2))
        self.panel.set_after_movie(result["masked"])

        mask = result["mask"]
        pct = 100 * result["fraction_kept"]
        self.metrics_label.setText(f"Mask kept {int(mask.sum())}/{mask.size} pixels ({pct:.1f}%).")

    # -- Volumetric (state.volumetric) path -----------------------------
    #
    # Same branching pattern as MotionCorrectionTab/DenoisingTab: each
    # entry point checks self.state.volumetric first, falling through to
    # the unmodified 2D behavior above when it's off.

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

    def _on_clear_clicked_volumetric(self) -> None:
        movie = self.state.active_data()
        if movie is None:
            QMessageBox.warning(self, "No data", "Load data on the Load tab first.")
            return
        self._pending_fingerprint = (id(movie), "clear")
        self._pending_params = dict(action="clear")
        self._input_movie = movie
        self._pending_step_label = "Mask (cleared)"
        result = _run_and_assess_3d(preview_slice_volumetric(movie), "clear")
        self._on_finished(result)

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
        self._pending_step_label = "Mask (auto-threshold)"
        self.worker = run_worker(
            self.busy_bar, "Computing mask...",
            _run_and_assess_3d, movie, "auto", on_success=self._on_finished, on_failure=self._on_failed,
        )

    def _commit(self) -> None:
        if self._pending_result is None:
            return
        if is_memmap(self._input_movie):
            self._start_chunked_commit()
            return
        if self.state.volumetric:
            self._finish_commit(self._pending_result["masked_3d"])
            return
        self._finish_commit(self._pending_result[self._result_key])

    def _chunked_commit_volumetric(self, source: np.ndarray, output_path: Path) -> np.ndarray:
        action = self._pending_params.get("action", "auto")
        if action == "clear":
            mask = np.ones(source.shape[1:], dtype=bool)
        else:
            preview = preview_slice_volumetric(source)
            mask = triangle_mask(np.asarray(preview, dtype=np.float64).mean(axis=0))

        T = source.shape[0]
        output = create_fits_memmap(output_path, source.shape, np.float32)
        for t0 in range(0, T, self._chunk_frames):
            t1 = min(t0 + self._chunk_frames, T)
            chunk = np.asarray(source[t0:t1], dtype=np.float32)
            output[t0:t1] = apply_mask_3d(chunk, mask)
        output.flush()
        return output
