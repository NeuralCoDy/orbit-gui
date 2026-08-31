"""Mask tab: zeroes out background/uninformative pixels via a
thresholded binary mask (orbit.masking) -- several methods (Triangle,
Otsu, Manual threshold, Percentile), each fit into this app's own
Apply-produces-a-candidate / Commit-makes-it-active StageTab pattern
(see that module's docstring), plus "Clear Mask" (an all-True, no-op
mask, replicating pyGraFT's own "Mask" tab default) rather than
graftapp's more immediate apply-on-click model.

The selected method's parameters (if any) live in the Parameters...
popup, same convention as Denoising/Motion Correction/Source
Extraction's own per-method parameter groups. "Clear Mask" stays its
own always-present button rather than a Method option: it's trivial and
instant, so it runs synchronously rather than through a background
worker, but still produces a candidate that goes through the same
_on_finished/preview-before-commit review as every other method here --
clearing a mask is still an explicit, reviewable pipeline step, not a
silent reset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QMessageBox, QPushButton

from orbit._volumetric import depth_project
from orbit.masking import apply_mask, apply_mask_3d, compute_mask

from ..fits_io import create_fits_memmap
from ..io import preview_slice
from ..state import AppState
from ..volumetric_io import preview_slice_volumetric
from ..widgets import ParametersDialog, make_spinbox
from ..workers import run_worker
from .stage_tab import StageTab

_METHODS = (
    ("Triangle (auto)", "triangle"),
    ("Otsu (auto)", "otsu"),
    ("Manual threshold", "manual"),
    ("Percentile", "percentile"),
)
_METHOD_KEYS = dict(_METHODS)
_METHOD_LABELS = {key: label for label, key in _METHODS}
# Methods with tunable parameters -- shown/hidden in the Parameters
# dialog together, same "group per method" convention as DenoisingTab.
_PARAM_GROUPS = ("manual", "percentile")


def _mask_kwargs(action: str, params: dict) -> dict:
    """The subset of a fingerprint dict this method actually needs,
    stripped of the "action" key itself -- shared by every call site
    below so a new parametrized method only needs a new key here."""
    if action == "manual":
        return {"threshold": params["threshold"]}
    if action == "percentile":
        return {"percentile": params["percentile"]}
    return {}


def _run_and_assess(movie: np.ndarray, action: str, **kwargs) -> dict:
    """Runs off the GUI thread for every Method (called directly,
    synchronously, for the trivial "clear" path)."""
    if action == "clear":
        mask = np.ones(movie.shape[:2], dtype=bool)
    else:
        mask = compute_mask(np.asarray(movie, dtype=np.float64).mean(axis=2), action, **kwargs)
    masked = apply_mask(movie, mask)
    return {"masked": masked, "mask": mask, "fraction_kept": float(mask.mean())}


def _run_and_assess_3d(movie: np.ndarray, action: str, **kwargs) -> dict:
    """Volumetric (T, L, W, D) counterpart of _run_and_assess. Every
    orbit.masking method needs no volumetric variant of its own (none do
    shape-specific unpacking, so they already work unchanged on a
    genuine 3D (L, W, D) mean-projection); only apply_mask (T-last)
    needed a T-first sibling."""
    if action == "clear":
        mask = np.ones(movie.shape[1:], dtype=bool)
    else:
        mask = compute_mask(np.asarray(movie, dtype=np.float64).mean(axis=0), action, **kwargs)
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
    _result_key_3d = "masked_3d"
    _stage_key = "mask"
    _supports_volumetric = True

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, apply_label="Apply Mask", parent=parent)

    def _build_controls_row(self) -> QHBoxLayout:
        self.method_combo = QComboBox()
        self.method_combo.addItems([label for label, _key in _METHODS])
        self.method_combo.currentTextChanged.connect(self._update_visible_params)

        self.manual_threshold_spin = make_spinbox(-1e6, 1e6, 0.0, step=1.0, decimal=True)
        self.percentile_spin = make_spinbox(0.0, 100.0, 10.0, step=1.0, decimal=True)

        self.params_dialog = ParametersDialog(title="Mask Parameters", parent=self)
        self.params_dialog.add_row("threshold value", self.manual_threshold_spin, group="manual")
        self.params_dialog.add_row("percentile (%, brightest kept)", self.percentile_spin, group="percentile")
        self._update_visible_params(self.method_combo.currentText())

        self.clear_btn = QPushButton("Clear Mask")
        self.clear_btn.clicked.connect(self._on_clear_clicked)

        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("Method:"))
        controls_row.addWidget(self.method_combo)
        self.params_btn = QPushButton("Parameters...")
        self.params_btn.clicked.connect(self.params_dialog.exec)
        controls_row.addWidget(self.params_btn)
        controls_row.addWidget(self.commit_controls)
        controls_row.addWidget(self.clear_btn)
        controls_row.addStretch()
        return controls_row

    def _build_metrics(self) -> None:
        self.metrics_label = QLabel("Run masking to see results.")
        self.panel.add_metric_widget(self.metrics_label)

    def _update_visible_params(self, label: str) -> None:
        """Only the fields relevant to the selected method are shown in
        the Parameters popup -- Triangle/Otsu have none of their own."""
        method = _METHOD_KEYS[label]
        self.params_dialog.show_only_group(method)

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
        # "clear" goes through _on_clear_clicked instead, which sets
        # _pending_fingerprint/_pending_params directly rather than via
        # this method.
        action = _METHOD_KEYS[self.method_combo.currentText()]
        params = dict(action=action)
        if action == "manual":
            params["threshold"] = self.manual_threshold_spin.value()
        elif action == "percentile":
            params["percentile"] = self.percentile_spin.value()
        return params

    def restore_params(self, params: dict) -> None:
        action = params.get("action", "triangle")
        if action == "auto":  # legacy saved-session value, from before multiple methods existed
            action = "triangle"
        if action in _METHOD_LABELS:
            self.method_combo.setCurrentText(_METHOD_LABELS[action])
        if "threshold" in params:
            self.manual_threshold_spin.setValue(params["threshold"])
        if "percentile" in params:
            self.percentile_spin.setValue(params["percentile"])

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
        action = self._pending_params["action"]
        self._pending_step_label = f"Mask ({_METHOD_LABELS[action]})"
        self.worker = run_worker(
            self.busy_bar, "Computing mask...",
            _run_and_assess, movie, action, on_success=self._on_finished, on_failure=self._on_failed,
            **_mask_kwargs(action, self._pending_params),
        )

    def _chunked_commit(self, source: np.ndarray, output_path: Path) -> np.ndarray:
        # The mask is fit once from the same <=5000-frame preview Apply
        # (or Clear Mask) already used, then applied as a fixed
        # per-pixel gate to every chunk of the full movie -- same
        # "fit on the preview sample" approximation normalization_tab.py
        # uses for its own baselines.
        action = self._pending_params.get("action", "triangle")
        if action == "clear":
            mask = np.ones(source.shape[:2], dtype=bool)
        else:
            preview = preview_slice(source)
            mask = compute_mask(
                np.asarray(preview, dtype=np.float64).mean(axis=2), action, **_mask_kwargs(action, self._pending_params)
            )

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

    def _start_worker_volumetric(self, movie: np.ndarray) -> None:
        action = self._pending_params["action"]
        self._pending_step_label = f"Mask ({_METHOD_LABELS[action]})"
        self.worker = run_worker(
            self.busy_bar, "Computing mask...",
            _run_and_assess_3d, movie, action, on_success=self._on_finished, on_failure=self._on_failed,
            **_mask_kwargs(action, self._pending_params),
        )

    def _chunked_commit_volumetric(self, source: np.ndarray, output_path: Path) -> np.ndarray:
        action = self._pending_params.get("action", "triangle")
        if action == "clear":
            mask = np.ones(source.shape[1:], dtype=bool)
        else:
            preview = preview_slice_volumetric(source)
            mask = compute_mask(
                np.asarray(preview, dtype=np.float64).mean(axis=0), action, **_mask_kwargs(action, self._pending_params)
            )

        T = source.shape[0]
        output = create_fits_memmap(output_path, source.shape, np.float32)
        for t0 in range(0, T, self._chunk_frames):
            t1 = min(t0 + self._chunk_frames, T)
            chunk = np.asarray(source[t0:t1], dtype=np.float32)
            output[t0:t1] = apply_mask_3d(chunk, mask)
        output.flush()
        return output
