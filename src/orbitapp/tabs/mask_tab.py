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

from orbit.masking import apply_mask, triangle_mask

from ..fits_io import create_fits_memmap
from ..io import preview_slice
from ..state import AppState
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

    def _start_worker(self, movie: np.ndarray) -> None:
        self._pending_step_label = "Mask (auto-threshold)"
        self.worker = run_worker(
            self.busy_bar, "Computing mask...",
            _run_and_assess, movie, "auto", on_success=self._on_finished, on_failure=self._on_failed,
        )

    def _chunked_commit(self, source: np.ndarray, output_path: Path) -> np.ndarray:
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
        self.panel.before_view.setImage(preview_slice(self._input_movie).mean(axis=2))
        self.panel.after_view.setImage(result["masked"].mean(axis=2))
        self.panel.set_after_movie(result["masked"])

        mask = result["mask"]
        pct = 100 * result["fraction_kept"]
        self.metrics_label.setText(f"Mask kept {int(mask.sum())}/{mask.size} pixels ({pct:.1f}%).")
