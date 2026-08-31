"""Source Extraction tab: find candidate ROI masks + traces via
Correlation-based click-to-add seeding (ported from roiapp, always
available regardless of the selected batch Method), PCA-ICA, CNMF, or
GraFT.

Unlike every earlier stage, this one doesn't transform the movie into a
new movie -- it produces a variable-size collection of ROIs, so it
doesn't subclass StageTab (see that module's docstring for the
"Apply produces one candidate movie, Commit replaces it" assumption that
doesn't fit here).

"Method" only chooses between the batch algorithms (PCA-ICA, CNMF, more
later); correlation-based click-to-add is a separate, always-active tool
beneath it, not a Method option -- clicking the field-of-view view
always grows a new ROI's preview. Mirrors roiapp's own click workflow:
nothing is saved anywhere until the explicit "Add ROI" action, which is
why a not-yet-added preview is listed as "ROI -1" (see ROIReviewPanel)
and shows its trace immediately on selection, same as any added ROI.
Auto-select-seeds and the batch methods skip the one-at-a-time preview
step and add their whole batch directly, since they're reviewed
afterward via the table instead. Nothing is ever *accepted* automatically
either way -- every candidate needs an explicit Accept before Commit
will include it. Commit accumulates onto AppState.rois rather than
replacing it, and accepted ROIs from different methods can mix in one
committed set.

Every ROI's neuropil trace (a dilated ring around its mask, other ROIs'
pixels excluded -- see orbit.neuropil) is recomputed for the WHOLE
current set whenever that set changes, since one ROI's ring can newly
overlap another's mask.
"""

from __future__ import annotations

import functools

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from orbit._masks import masked_mean_trace
from orbit._volumetric import depth_project
from orbit.cnmf import CNMFResult, cnmf_source_extraction, patch_cnmf_source_extraction
from orbit.cnmf_e import cnmf_e_source_extraction, patch_cnmf_e_source_extraction
from orbit.neuropil import compute_neuropil_traces
from orbit.projections import local_correlation_projection
from orbit.roi_extraction_corr import find_seed_candidates, roi_from_seed
from orbit.roi_extraction_graft import GraFTResult, graft_source_extraction, patch_graft_source_extraction
from orbit.roi_extraction_graft_3d import (
    GraFTResult3D,
    _masked_mean_trace_3d,
    graft_source_extraction_3d,
    patch_graft_source_extraction_3d,
)
from orbit.roi_extraction_pca_ica import PCAICAResult, pca_ica_source_extraction
from orbit.roi_extraction_realseudo import RealSeudoResult, real_seudo_source_extraction

from ..io import is_memmap, preview_slice
from ..state import AppState, ROI
from ..theme import WARNING
from ..volumetric_io import preview_slice_volumetric
from ..widgets import BusyBar, ParametersDialog, ROIReviewPanel, confirm_recompute, make_spinbox
from ..workers import FunctionWorker, run_worker

_METHODS = (
    ("PCA-ICA", "pca_ica"),
    ("CNMF", "cnmf"),
    ("CNMF-E", "cnmf_e"),
    ("GraFT", "graft"),
    ("Real-SEUDO", "real_seudo"),
)
_METHOD_KEYS = dict(_METHODS)
_METHOD_ORDER = [key for _label, key in _METHODS]
_PIPELINE_LABELS = {
    "correlation": "Correlation ROIs", "pca_ica": "PCA-ICA", "cnmf": "CNMF", "cnmf_e": "CNMF-E", "graft": "GraFT",
    "real_seudo": "Real-SEUDO",
}


def _grow_seeds(movie: np.ndarray, seeds: list[tuple[int, int]], kwargs: dict) -> list[dict]:
    """Runs off the GUI thread. Returns plain dicts rather than ROI objects
    -- ROI ids get assigned back on the GUI thread, since self._next_id
    isn't safe to touch from a worker thread."""
    results = []
    for seed in seeds:
        result = roi_from_seed(movie, seed, **kwargs)
        results.append({"seed_loc": seed, "mask": result.mask, "trace": result.trace, "thresh": result.thresh})
    return results


def _reextract_traces(rois: list[ROI], movie: np.ndarray) -> list[ROI]:
    """Runs off the GUI thread. Recomputes each ROI's primary trace from
    the WHOLE movie -- used at Commit time for a memmap-backed dataset,
    since CNMF/PCA-ICA's own trace may only reflect the 5000-frame
    preview they were fit against (see _on_run_cnmf_clicked/
    _on_run_pca_ica_clicked). masked_mean_trace is already bounded per
    ROI (proportional to mask size x T, not the whole FOV), so this
    stays memmap-safe regardless of how the trace was first computed --
    correlation-based ROIs are already exact here too, just redundantly
    recomputed to the same value, since they're not preview-capped."""
    for roi in rois:
        roi.trace = masked_mean_trace(movie, roi.mask)
    return rois


def _reextract_traces_3d(rois: list[ROI], movie: np.ndarray) -> list[ROI]:
    """Volumetric counterpart of _reextract_traces -- uses
    _masked_mean_trace_3d (T-first, 3D-mask) instead of masked_mean_trace."""
    for roi in rois:
        roi.trace = _masked_mean_trace_3d(movie, roi.mask)
    return rois


def _corr_shared_params(kwargs: dict) -> dict:
    """The subset of _corr_kwargs() worth recording as this ROI's
    provenance -- excludes local_corr_image (a big array, not
    JSON-serializable/meaningful as a saved parameter) and thresh (not
    shared: each ROI's own *resolved* threshold, explicit or
    auto-searched, comes back as result.thresh instead)."""
    return {k: v for k, v in kwargs.items() if k not in ("local_corr_image", "thresh")}


class SourceExtractionTab(QWidget):
    data_changed = Signal()  # emitted whenever state.rois changes: Commit, or Clear All

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.worker: FunctionWorker | None = None
        self._candidates: list[ROI] = []
        self._next_id = 0
        self._corr_image: np.ndarray | None = None
        self._preview_roi: ROI | None = None  # grown but not yet added -- listed as "ROI -1"
        self._last_batch_run: tuple | None = None  # (id(movie), fn, sorted kwargs) of the last successful batch run
        self._pending_corr_params: dict = {}  # shared corr_kwargs for the in-flight seed-growing worker
        self._pending_batch_params: dict = {}  # kwargs for the in-flight PCA-ICA/CNMF/GraFT worker

        layout = QVBoxLayout(self)

        self.method_combo = QComboBox()
        self.method_combo.addItems([label for label, _key in _METHODS])
        self.method_combo.currentTextChanged.connect(self._on_method_changed)

        self.max_dist_spin = make_spinbox(1, 200, 15, decimal=True)
        self.growth_method_combo = QComboBox()
        self.growth_method_combo.addItems(["local_corr_threshold", "fixed_seed", "flood_fill"])
        self.seed_block_radius_spin = make_spinbox(0, 10, 0)
        self.n_auto_seeds_spin = make_spinbox(1, 200, 10)
        # The correlation threshold lives on the main panel, not this dialog
        # (see _build_correlation_rows) -- adjusted often enough to not want
        # it behind a popup.
        self.corr_thresh_spin = make_spinbox(0.0, 0.95, 0.0, step=0.05, decimal=True)

        self.n_pca_components_spin = make_spinbox(1, 500, 50)
        self.n_ica_components_spin = make_spinbox(1, 200, 40)
        self.cnmf_n_components_spin = make_spinbox(1, 500, 30)
        self.cnmf_search_radius_spin = make_spinbox(1, 200, 10, decimal=True)
        self.cnmf_merge_thresh_spin = make_spinbox(0.0, 1.0, 0.8, step=0.05, decimal=True)

        # Patch-based CNMF: splits the FOV into overlapping square patches
        # and runs CNMF independently on each (see orbit.cnmf docstring --
        # several of the whole-FOV algorithm's costs scale with frame
        # area regardless of component count, which patching bounds).
        # Rows below live in their own "cnmf_patch" params-dialog group,
        # nested inside "cnmf", so they only show when both the CNMF
        # method AND this checkbox are active (_update_cnmf_patch_rows_visibility).
        self.cnmf_patch_check = QCheckBox("Use patch-based extraction (for large fields of view)")
        self.cnmf_patch_check.toggled.connect(
            lambda: self._update_patch_rows_visibility("cnmf", "cnmf_patch", self.cnmf_patch_check)
        )
        self.cnmf_patch_size_spin = make_spinbox(10, 2000, 80)
        self.cnmf_patch_overlap_spin = make_spinbox(0, 500, 20)
        self.cnmf_components_per_patch_spin = make_spinbox(1, 200, 10)

        # CNMF-E (Zhou et al. 2018) -- for 1P/microendoscopic data: same
        # alternating spatial/temporal update loop as CNMF above, but
        # seeded at correlation x peak-to-noise-ratio peaks (min_corr/
        # min_pnr) instead of plain intensity peaks, and with a per-pixel
        # "ring model" background (ring_inner/outer_radius) instead of
        # relying solely on a global low-rank background -- see
        # orbit.cnmf_e's docstring. ring_downsample/ring_max_fit_frames
        # trade the ring model's own fit speed against fidelity (see
        # orbit.cnmf_e_background's docstring for the performance
        # rationale) -- surfaced here since, unlike the other CNMF-E
        # params, there's no single "obviously correct" default across
        # every field-of-view size.
        self.cnmf_e_n_components_spin = make_spinbox(1, 500, 30)
        self.cnmf_e_search_radius_spin = make_spinbox(1, 200, 10, decimal=True)
        self.cnmf_e_merge_thresh_spin = make_spinbox(0.0, 1.0, 0.8, step=0.05, decimal=True)
        self.cnmf_e_min_corr_spin = make_spinbox(0.0, 1.0, 0.8, step=0.05, decimal=True)
        self.cnmf_e_min_pnr_spin = make_spinbox(0.0, 100.0, 8.0, step=0.5, decimal=True)
        self.cnmf_e_ring_inner_radius_spin = make_spinbox(1, 500, 20, decimal=True)
        self.cnmf_e_ring_outer_radius_spin = make_spinbox(1, 500, 25, decimal=True)
        self.cnmf_e_ring_downsample_spin = make_spinbox(1, 16, 4)
        self.cnmf_e_ring_downsample_spin.setToolTip(
            "How much to spatially shrink the field of view before fitting the ring background model -- "
            "higher is much faster (grid size shrinks quadratically) but coarser. 4 is a reasonable default "
            "at typical recording sizes."
        )
        self.cnmf_e_ring_max_fit_frames_spin = make_spinbox(10, 20000, 500)
        self.cnmf_e_ring_max_fit_frames_spin.setToolTip(
            "How many (evenly-spaced) frames to use when fitting the ring background model's weights -- "
            "more is slower with diminishing accuracy gains past a few hundred."
        )

        self.cnmf_e_patch_check = QCheckBox("Use patch-based extraction (for large fields of view)")
        self.cnmf_e_patch_check.toggled.connect(
            lambda: self._update_patch_rows_visibility("cnmf_e", "cnmf_e_patch", self.cnmf_e_patch_check)
        )
        self.cnmf_e_patch_size_spin = make_spinbox(10, 2000, 80)
        self.cnmf_e_patch_overlap_spin = make_spinbox(0, 500, 20)
        self.cnmf_e_components_per_patch_spin = make_spinbox(1, 200, 10)

        # GraFT (Graph-Filtered Temporal dictionary learning, via the
        # pygraft-gui dependency) -- same whole-FOV/patch-based split as
        # CNMF above, same reason (patch-based is required, not just
        # offered, for a memmap movie -- see _on_run_graft_clicked).
        self.graft_n_dict_spin = make_spinbox(1, 500, 20)
        # lambda/lamForb/lamCorr/lamCont/learn_eps: same params dict keys
        # pyGraFT's own GUI (graftapp/tabs.py's ParametersTab) exposes --
        # shared between whole-FOV and patch-based GraFT below (both take
        # the same keys), not duplicated per mode the way n_dict is, since
        # these regularization/convergence knobs mean the same thing
        # regardless of which mode is selected. Defaults for lamForb/
        # lamCorr/lamCont are set higher than pyGraFT's own GUI defaults
        # (which are all 0.0) per explicit instruction, not carried over
        # from there.
        self.graft_lambda_spin = make_spinbox(0.0, 100.0, 0.6, step=0.05, decimal=True, decimals=4)
        self.graft_lam_forb_spin = make_spinbox(0.0, 100.0, 0.9, step=0.05, decimal=True, decimals=4)
        self.graft_lam_corr_spin = make_spinbox(0.0, 100.0, 0.5, step=0.05, decimal=True, decimals=4)
        self.graft_lam_cont_spin = make_spinbox(0.0, 100.0, 0.1, step=0.05, decimal=True, decimals=4)
        self.graft_learn_eps_spin = make_spinbox(0.0, 10.0, 0.01, decimal=True, decimals=6)
        self.graft_patch_check = QCheckBox("Use patch-based extraction (for large fields of view)")
        self.graft_patch_check.toggled.connect(
            lambda: self._update_patch_rows_visibility("graft", "graft_patch", self.graft_patch_check)
        )
        self.graft_patch_size_spin = make_spinbox(10, 2000, 50)
        self.graft_patch_overlap_spin = make_spinbox(0, 500, 10)
        self.graft_n_dict_per_patch_spin = make_spinbox(1, 200, 10)

        # Real-SEUDO -- fits one frame at a time against a growing known-cell
        # set, discovering and promoting new cells as their activity is
        # detected (see orbit.roi_extraction_realseudo/orbit.seudo.streaming
        # docstrings). Unlike every method above, this never needs the whole
        # movie in RAM (one frame at a time by construction), so there's no
        # patch-based mode here at all -- it's already memmap-safe. Defaults
        # match the wrapper's own real-data-tuned values.
        self.real_seudo_sigma2_spin = make_spinbox(0.0, 10.0, 0.0020, decimal=True, decimals=4)
        self.real_seudo_lambda_blob_spin = make_spinbox(0.0, 1000.0, 10.0, step=0.5, decimal=True)
        self.real_seudo_blob_radius_spin = make_spinbox(0.1, 50.0, 3.0, step=0.1, decimal=True)
        self.real_seudo_pad_space_spin = make_spinbox(0, 200, 5)
        self.real_seudo_lookahead_frames_spin = make_spinbox(1, 100, 3)
        self.real_seudo_lookahead_frames_spin.setToolTip(
            "Forward-looking averaging window before reporting a frame's activity -- improves detection "
            "quality at the cost of this many frames of latency. 1 disables lookahead entirely (immediate, "
            "strictly causal results)."
        )
        self.real_seudo_cutoff_multiplier_spin = make_spinbox(0.1, 100.0, 4.0, step=0.5, decimal=True)
        self.real_seudo_min_roi_size_spin = make_spinbox(1, 100000, 50)
        self.real_seudo_min_avg_px_spin = make_spinbox(-1000.0, 1000.0, -1.0, step=0.5, decimal=True)
        self.real_seudo_min_avg_px_spin.setToolTip(
            "Minimum average brightness for a candidate region -- negative values are multiples of the "
            "local noise level, positive values are an absolute threshold."
        )
        self.real_seudo_mask_blur_rad_spin = make_spinbox(0, 50, 1)
        self.real_seudo_exclude_radius_spin = make_spinbox(-1, 500, 5)
        self.real_seudo_exclude_radius_spin.setToolTip(
            "How far (in pixels) a known cell's footprint is dilated before being excluded from new-"
            "candidate detection. -1 disables known-cell exclusion entirely (recovers more overlapping "
            "cells, at a real precision cost) -- see orbit.seudo.streaming.DetectionParams' docstring."
        )
        self.real_seudo_consecutive_frames_spin = make_spinbox(1, 1000, 5)
        self.real_seudo_max_track_gap_spin = make_spinbox(0, 100, 1)
        self.real_seudo_eq8_merge_thresh_spin = make_spinbox(0.0, 1.0, 0.75, step=0.05, decimal=True)
        self.real_seudo_eq9_merge_thresh_spin = make_spinbox(0.0, 1.0, 0.75, step=0.05, decimal=True)

        # Correlation click-to-add's own parameters live on the main screen
        # (see _build_correlation_rows), not in this dialog -- it's always
        # active, not tied to the Method combo below, so this dialog only
        # ever holds each batch Method's own group (show_only_group in
        # _on_method_changed keeps just the selected one visible, same
        # pattern as DenoisingTab).
        self.params_dialog = ParametersDialog(title="Source Extraction Parameters", parent=self)
        self.params_dialog.add_row("number of PCA components", self.n_pca_components_spin, group="pca_ica")
        self.params_dialog.add_row("number of ICA components", self.n_ica_components_spin, group="pca_ica")
        self.params_dialog.add_row("number of components", self.cnmf_n_components_spin, group="cnmf")
        self.params_dialog.add_row("search radius (px)", self.cnmf_search_radius_spin, group="cnmf")
        self.params_dialog.add_row("merge threshold", self.cnmf_merge_thresh_spin, group="cnmf")
        self.params_dialog.add_row("", self.cnmf_patch_check, group="cnmf")
        self.params_dialog.add_row("patch size (px)", self.cnmf_patch_size_spin, group="cnmf_patch")
        self.params_dialog.add_row("patch overlap (px)", self.cnmf_patch_overlap_spin, group="cnmf_patch")
        self.params_dialog.add_row("components per patch", self.cnmf_components_per_patch_spin, group="cnmf_patch")
        self.params_dialog.add_row("number of components", self.cnmf_e_n_components_spin, group="cnmf_e")
        self.params_dialog.add_row("search radius (px)", self.cnmf_e_search_radius_spin, group="cnmf_e")
        self.params_dialog.add_row("merge threshold", self.cnmf_e_merge_thresh_spin, group="cnmf_e")
        self.params_dialog.add_row("min correlation (seeding)", self.cnmf_e_min_corr_spin, group="cnmf_e")
        self.params_dialog.add_row("min peak-to-noise ratio (seeding)", self.cnmf_e_min_pnr_spin, group="cnmf_e")
        self.params_dialog.add_row("ring model inner radius (px)", self.cnmf_e_ring_inner_radius_spin, group="cnmf_e")
        self.params_dialog.add_row("ring model outer radius (px)", self.cnmf_e_ring_outer_radius_spin, group="cnmf_e")
        self.params_dialog.add_row("ring model downsample factor", self.cnmf_e_ring_downsample_spin, group="cnmf_e")
        self.params_dialog.add_row("ring model max fit frames", self.cnmf_e_ring_max_fit_frames_spin, group="cnmf_e")
        self.params_dialog.add_row("", self.cnmf_e_patch_check, group="cnmf_e")
        self.params_dialog.add_row("patch size (px)", self.cnmf_e_patch_size_spin, group="cnmf_e_patch")
        self.params_dialog.add_row("patch overlap (px)", self.cnmf_e_patch_overlap_spin, group="cnmf_e_patch")
        self.params_dialog.add_row("components per patch", self.cnmf_e_components_per_patch_spin, group="cnmf_e_patch")
        self.params_dialog.add_row("number of dictionary components", self.graft_n_dict_spin, group="graft")
        self.params_dialog.add_row("sparsity (lambda)", self.graft_lambda_spin, group="graft")
        self.params_dialog.add_row("Frobenius regularization (lamForb)", self.graft_lam_forb_spin, group="graft")
        self.params_dialog.add_row("correlation regularization (lamCorr)", self.graft_lam_corr_spin, group="graft")
        self.params_dialog.add_row("continuation regularization (lamCont)", self.graft_lam_cont_spin, group="graft")
        self.params_dialog.add_row("convergence threshold (learn_eps)", self.graft_learn_eps_spin, group="graft")
        self.params_dialog.add_row("", self.graft_patch_check, group="graft")
        self.params_dialog.add_row("patch size (px)", self.graft_patch_size_spin, group="graft_patch")
        self.params_dialog.add_row("patch overlap (px)", self.graft_patch_overlap_spin, group="graft_patch")
        self.params_dialog.add_row("dictionary components per patch", self.graft_n_dict_per_patch_spin, group="graft_patch")
        self.params_dialog.add_row("noise level (sigma2)", self.real_seudo_sigma2_spin, group="real_seudo")
        self.params_dialog.add_row("blob sparsity (lambda_blob)", self.real_seudo_lambda_blob_spin, group="real_seudo")
        self.params_dialog.add_row("blob radius (px)", self.real_seudo_blob_radius_spin, group="real_seudo")
        self.params_dialog.add_row("fit window padding (px)", self.real_seudo_pad_space_spin, group="real_seudo")
        self.params_dialog.add_row("lookahead frames", self.real_seudo_lookahead_frames_spin, group="real_seudo")
        self.params_dialog.add_row("detection cutoff (x noise)", self.real_seudo_cutoff_multiplier_spin, group="real_seudo")
        self.params_dialog.add_row("min ROI size (px)", self.real_seudo_min_roi_size_spin, group="real_seudo")
        self.params_dialog.add_row("min average brightness", self.real_seudo_min_avg_px_spin, group="real_seudo")
        self.params_dialog.add_row("mask blur radius (px)", self.real_seudo_mask_blur_rad_spin, group="real_seudo")
        self.params_dialog.add_row("known-cell exclusion radius (px)", self.real_seudo_exclude_radius_spin, group="real_seudo")
        self.params_dialog.add_row("consecutive frames to promote", self.real_seudo_consecutive_frames_spin, group="real_seudo")
        self.params_dialog.add_row("max track gap (frames)", self.real_seudo_max_track_gap_spin, group="real_seudo")
        self.params_dialog.add_row("merge threshold (candidate-candidate)", self.real_seudo_eq8_merge_thresh_spin, group="real_seudo")
        self.params_dialog.add_row("merge threshold (candidate-known)", self.real_seudo_eq9_merge_thresh_spin, group="real_seudo")

        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("Method:"))
        controls_row.addWidget(self.method_combo)
        self.params_btn = QPushButton("Parameters...")
        self.params_btn.clicked.connect(self.params_dialog.exec)
        controls_row.addWidget(self.params_btn)

        self.action_stack = QStackedWidget()
        self.run_pca_ica_btn = self._add_run_action("Run PCA-ICA", self._on_run_pca_ica_clicked)
        self.run_cnmf_btn = self._add_run_action("Run CNMF", self._on_run_cnmf_clicked)
        self.run_cnmf_e_btn = self._add_run_action("Run CNMF-E", self._on_run_cnmf_e_clicked)
        self.run_graft_btn = self._add_run_action("Run GraFT", self._on_run_graft_clicked)
        self.run_real_seudo_btn = self._add_run_action("Run Real-SEUDO", self._on_run_real_seudo_clicked)
        controls_row.addWidget(self.action_stack)

        self.commit_btn = QPushButton("Commit Accepted ROIs")
        self.commit_btn.clicked.connect(self._commit)
        controls_row.addWidget(self.commit_btn)

        self.clear_all_btn = QPushButton("Clear All Traces / Undo Committed ROIs")
        self.clear_all_btn.clicked.connect(self._on_clear_all_clicked)
        controls_row.addWidget(self.clear_all_btn)
        controls_row.addStretch()
        layout.addLayout(controls_row)

        # Non-blocking, always-current hint (not a modal dialog -- updates
        # live as either Method or the Load tab's modality toggles change,
        # see _update_modality_warning) about a poor method/modality
        # pairing, e.g. CNMF-E on 2P data or CNMF/PCA-ICA on dendritic data.
        self.modality_warning_label = QLabel()
        self.modality_warning_label.setWordWrap(True)
        self.modality_warning_label.setStyleSheet(f"color: {WARNING};")
        self.modality_warning_label.setVisible(False)
        layout.addWidget(self.modality_warning_label)

        layout.addLayout(self._build_correlation_rows())

        self.busy_bar = BusyBar()
        layout.addWidget(self.busy_bar)

        self.status_label = QLabel("No data loaded.")
        layout.addWidget(self.status_label)

        self.review_panel = ROIReviewPanel()
        self.review_panel.pixel_clicked.connect(self._on_pixel_clicked)
        self.review_panel.roi_deleted.connect(self._on_roi_deleted)
        # Clicking the FOV view always seeds a new ROI -- no separate "click
        # mode" toggle -- so panning (pyqtgraph's default for left-drag) is
        # permanently off there instead of only while some mode is armed.
        self.review_panel.fov_view.getView().setMouseEnabled(x=False, y=False)
        layout.addWidget(self.review_panel)

        self._on_method_changed(self.method_combo.currentText())
        self.on_modality_changed()  # reflects state.volumetric's initial value, if already set

    def _build_correlation_rows(self) -> QVBoxLayout:
        """Correlation-based click-to-add: always active, independent of
        the Method combo above (see module docstring) -- its own
        parameters live here on the main screen rather than behind
        Parameters..., since they apply regardless of which batch Method
        is selected."""
        rows = QVBoxLayout()

        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("Correlation growth -- max_dist (px):"))
        params_row.addWidget(self.max_dist_spin)
        params_row.addWidget(QLabel("growth_method:"))
        params_row.addWidget(self.growth_method_combo)
        params_row.addWidget(QLabel("seed_block_radius (px):"))
        params_row.addWidget(self.seed_block_radius_spin)
        params_row.addStretch()
        rows.addLayout(params_row)

        action_row = QHBoxLayout()
        action_row.addWidget(QLabel("Click FOV to add ROI. Correlation threshold (0 = auto-search):"))
        action_row.addWidget(self.corr_thresh_spin)
        self.add_roi_btn = QPushButton("Add ROI")
        self.add_roi_btn.setEnabled(False)
        self.add_roi_btn.clicked.connect(self._on_add_roi_clicked)
        action_row.addWidget(self.add_roi_btn)
        action_row.addWidget(QLabel("number of auto seeds:"))
        action_row.addWidget(self.n_auto_seeds_spin)
        self.auto_seed_btn = QPushButton("Auto-select seeds...")
        self.auto_seed_btn.clicked.connect(self._on_auto_seed_clicked)
        action_row.addWidget(self.auto_seed_btn)
        action_row.addStretch()
        rows.addLayout(action_row)

        return rows

    def _add_run_action(self, label: str, slot) -> QPushButton:
        """One "Run <Method>" button in its own action_stack page --
        shared by every batch method (PCA-ICA, CNMF, ...) so adding a
        future method doesn't need its own near-identical builder."""
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        btn = QPushButton(label)
        btn.clicked.connect(slot)
        row.addWidget(btn)
        self.action_stack.addWidget(widget)
        return btn

    def _on_method_changed(self, label: str) -> None:
        method = _METHOD_KEYS[label]
        self.params_dialog.show_only_group(method)
        # show_only_group above always hides "cnmf_patch"/"cnmf_e_patch"/"graft_patch" -- reapply on top
        self._update_patch_rows_visibility("cnmf", "cnmf_patch", self.cnmf_patch_check)
        self._update_patch_rows_visibility("cnmf_e", "cnmf_e_patch", self.cnmf_e_patch_check)
        self._update_patch_rows_visibility("graft", "graft_patch", self.graft_patch_check)
        self.action_stack.setCurrentIndex(_METHOD_ORDER.index(method))
        self._update_modality_warning()

    def _update_modality_warning(self) -> None:
        """Non-blocking hint (not every mismatch is necessarily wrong for
        a given dataset, so this never prevents Run) about a poor
        Method/modality pairing -- re-evaluated on every Method change
        AND every Load tab modality-toggle change (see _on_method_changed/
        on_modality_changed), against the Load tab's current
        somatic_1p/somatic_2p/dendrites/axons/widefield toggles."""
        method = _METHOD_KEYS[self.method_combo.currentText()]
        messages = []

        if self.state.somatic_2p and method == "cnmf_e":
            messages.append("CNMF-E is designed for 1P (microendoscopic) data, but 2P-Somatic is checked.")
        if self.state.somatic_1p and method in ("cnmf", "pca_ica"):
            messages.append(
                f"{self.method_combo.currentText()} is better suited to 2P data, but 1P-Somatic is checked "
                "-- consider CNMF-E instead."
            )
        dendritic_or_axonal_or_widefield = [
            name for name, on in (
                ("Dendrites", self.state.dendrites), ("Axons", self.state.axons), ("Widefield", self.state.widefield),
            ) if on
        ]
        if dendritic_or_axonal_or_widefield and method != "graft":
            messages.append(
                f"{', '.join(dendritic_or_axonal_or_widefield)} is checked -- GraFT is the only method here "
                "well suited to dendritic/axonal/widefield data."
            )

        self.modality_warning_label.setText("⚠ " + "  ".join(messages))
        self.modality_warning_label.setVisible(bool(messages))

    def _update_patch_rows_visibility(self, method: str, group: str, patch_check: QCheckBox) -> None:
        """Shows ``group``'s patch-only param rows only when both
        ``method`` is the currently-selected Method AND its own patch
        checkbox is checked -- shared by every batch method with a
        patch-based mode (CNMF, GraFT)."""
        visible = _METHOD_KEYS[self.method_combo.currentText()] == method and patch_check.isChecked()
        self.params_dialog.set_group_visible(group, visible)

    def on_modality_changed(self) -> None:
        """Reacts to the Load tab's Volumetric toggle (wired in app.py) --
        only GraFT has a 3D implementation, so every other batch method
        is disabled and the combo forced onto GraFT. Correlation-based
        click-to-add is always-active infrastructure, not gated by the
        Method combo, so it's disabled directly here too -- clicking a
        2D FOV image has no volumetric equivalent yet (self._corr_image
        also stays None for volumetric, see on_data_loaded, so
        auto-seeding would otherwise just show a confusing "no data"
        warning instead of being visibly unavailable)."""
        volumetric = self.state.volumetric
        model = self.method_combo.model()
        for i, (_label, key) in enumerate(_METHODS):
            if key == "graft":
                continue
            item = model.item(i)
            item.setEnabled(not volumetric)
            item.setToolTip("Not yet available for volumetric data." if volumetric else "")
        if volumetric and _METHOD_KEYS[self.method_combo.currentText()] != "graft":
            self.method_combo.setCurrentText("GraFT")
        self.auto_seed_btn.setEnabled(not volumetric)
        self._update_modality_warning()

    def on_data_loaded(self) -> None:
        movie = self.state.active_data()
        self._candidates = []
        self._clear_preview()
        self._last_batch_run = None  # a new/changed movie invalidates any prior "already run" state
        self.review_panel.set_candidates(self._candidates)
        if movie is None:
            self._corr_image = None
            self.status_label.setText("No data loaded.")
            return
        self.status_label.setText(f"Ready. shape={movie.shape}")
        if self.state.volumetric:
            # local_correlation_projection assumes a 3D (H, W, T) movie --
            # only GraFT has a volumetric implementation (see
            # on_modality_changed), so correlation-based click-to-add
            # stays inert here (self._corr_image stays None). The FOV
            # background still gets a depth-projected preview, same
            # display-only technique as Motion Correction/Denoising.
            self._corr_image = None
            projected = depth_project(preview_slice_volumetric(movie))
            self.review_panel.set_base_image(projected.mean(axis=2))
            return
        # preview_slice bounds this to the first 5000 frames for a
        # memmap movie -- local_correlation_projection materializes its
        # whole input, which would otherwise force a full read of an
        # arbitrarily large movie just to draw the FOV correlation image.
        self.worker = run_worker(
            self.busy_bar, "Computing local correlation image...",
            local_correlation_projection, preview_slice(movie),
            on_success=self._on_corr_image_ready, on_failure=self._on_failed,
        )

    def _on_corr_image_ready(self, corr_image: np.ndarray) -> None:
        self._corr_image = corr_image
        self.review_panel.set_base_image(corr_image)
        self.busy_bar.stop("")

    def _on_pixel_clicked(self, row: int, col: int) -> None:
        movie = self.state.active_data()
        if movie is None or self._corr_image is None:
            return
        height, width = movie.shape[:2]
        if not (0 <= row < height and 0 <= col < width):
            return
        self._grow_preview(movie, (row, col))

    def _on_auto_seed_clicked(self) -> None:
        movie = self.state.active_data()
        if movie is None or self._corr_image is None:
            QMessageBox.warning(self, "No data", "Load data on the Load tab first.")
            return
        seeds = find_seed_candidates(self._corr_image, n_seeds=self.n_auto_seeds_spin.value())
        if not seeds:
            QMessageBox.information(
                self, "Auto-select seeds", "No seed candidates found above the correlation threshold."
            )
            return
        self._grow_from_seeds(movie, seeds)

    def _corr_kwargs(self) -> dict:
        thresh = self.corr_thresh_spin.value()
        return dict(
            max_dist=self.max_dist_spin.value(),
            growth_method=self.growth_method_combo.currentText(),
            seed_block_radius=self.seed_block_radius_spin.value(),
            local_corr_image=self._corr_image,
            thresh=thresh if thresh > 0 else None,
        )

    def _grow_preview(self, movie: np.ndarray, seed: tuple[int, int]) -> None:
        """Interactive single click: grows a candidate but does NOT add it
        to the collection -- only _on_add_roi_clicked does that, mirroring
        roiapp's click-grows-a-preview / explicit-Select-ROI-adds-it split."""
        kwargs = self._corr_kwargs()
        self._pending_corr_params = _corr_shared_params(kwargs)
        self.worker = run_worker(
            self.busy_bar, "Growing ROI preview...",
            _grow_seeds, movie, [seed], kwargs,
            on_success=self._on_preview_grown, on_failure=self._on_failed,
        )

    def _on_preview_grown(self, results: list[dict]) -> None:
        r = results[0]
        self._preview_roi = ROI(
            id=-1, mask=r["mask"], trace=r["trace"], source_method="correlation", status="preview",
            seed_loc=r["seed_loc"], params={**self._pending_corr_params, "threshold": r["thresh"]},
        )
        self.review_panel.set_preview_roi(self._preview_roi)
        self.add_roi_btn.setEnabled(True)
        self.busy_bar.stop("")
        self.status_label.setText("Preview ready as 'ROI -1' -- click 'Add ROI' to keep it, or click elsewhere to replace it.")

    def _on_add_roi_clicked(self) -> None:
        if self._preview_roi is None:
            return
        self._preview_roi.id = self._next_id
        self._preview_roi.status = "pending"
        self._next_id += 1
        self._candidates.append(self._preview_roi)
        self._preview_roi = None
        self.add_roi_btn.setEnabled(False)
        self.review_panel.set_preview_roi(None)
        self._refresh_candidates_display()

    def _clear_preview(self) -> None:
        self._preview_roi = None
        self.add_roi_btn.setEnabled(False)
        self.review_panel.set_preview_roi(None)

    def _grow_from_seeds(self, movie: np.ndarray, seeds: list[tuple[int, int]]) -> None:
        """Batch path (auto-select-seeds): every seed's ROI is added
        directly to the collection -- reviewed afterward via the table
        rather than one at a time."""
        kwargs = self._corr_kwargs()
        self._pending_corr_params = _corr_shared_params(kwargs)
        self.worker = run_worker(
            self.busy_bar, f"Growing {len(seeds)} ROI(s) from seed(s)...",
            _grow_seeds, movie, seeds, kwargs,
            on_success=self._on_seeds_grown, on_failure=self._on_failed,
        )

    def _on_seeds_grown(self, results: list[dict]) -> None:
        rois = self._make_rois(
            [r["mask"] for r in results], [r["trace"] for r in results], "correlation",
            seed_locs=[r["seed_loc"] for r in results], thresholds=[r["thresh"] for r in results],
            params=self._pending_corr_params,
        )
        self._add_candidates(rois)

    def _run_batch_method(self, message: str, fn, on_success, worker_movie=None, **kwargs) -> None:
        """Launches a batch extraction algorithm (PCA-ICA, CNMF, ...) in
        the background -- shared by every such method since they only
        differ in the function/message/kwargs/result-handler. Skips
        redoing the work (after confirming) if this exact function+
        parameters already ran on this movie -- it would just reproduce
        the same candidates.

        ``worker_movie``, if given, is what actually gets passed to
        ``fn`` (e.g. a 5000-frame preview_slice of a memmap movie) --
        the "already ran" fingerprint still keys off the real active
        movie (``id(movie)``) regardless, so switching between a memmap
        and non-memmap load of the same path is still treated as a
        different dataset."""
        movie = self.state.active_data()
        if movie is None:
            QMessageBox.warning(self, "No data", "Load data on the Load tab first.")
            return

        fingerprint = (id(movie), fn, tuple(sorted(kwargs.items())))
        if fingerprint == self._last_batch_run:
            already_done = "This method was already run with these exact parameters on this data."
            if not confirm_recompute(self, already_done):
                return

        def _on_success(result) -> None:
            self._last_batch_run = fingerprint
            on_success(result)

        self.worker = run_worker(
            self.busy_bar, message, fn, worker_movie if worker_movie is not None else movie,
            on_success=_on_success, on_failure=self._on_failed, **kwargs,
        )

    def _on_run_pca_ica_clicked(self) -> None:
        movie = self.state.active_data()
        self._pending_batch_params = dict(
            n_pca_components=self.n_pca_components_spin.value(), n_ica_components=self.n_ica_components_spin.value(),
        )
        # PCA/ICA decomposes the whole (P, T) movie at once -- no
        # patch-based equivalent exists for it, so a memmap movie is
        # always capped to the same 5000-frame preview Apply uses
        # elsewhere in the app, rather than materializing the whole thing.
        worker_movie = preview_slice(movie) if movie is not None and is_memmap(movie) else None
        self._run_batch_method(
            "Running PCA-ICA...", pca_ica_source_extraction, self._on_pca_ica_finished,
            worker_movie=worker_movie, **self._pending_batch_params,
        )

    def _on_pca_ica_finished(self, result: PCAICAResult) -> None:
        rois = self._make_rois(result.masks, result.traces, "pca_ica", params=self._pending_batch_params)
        self._add_candidates(rois)

    def _refuse_if_memmap_without_patch(self, memmap_input: bool, patch_check: QCheckBox, method_name: str) -> bool:
        """True (after showing the standard warning) if ``memmap_input``
        but ``patch_check`` isn't checked -- shared by every batch method
        with both a whole-FOV and patch-based mode (CNMF, GraFT), since
        whole-FOV mode would otherwise need to read a memmap movie fully
        into RAM."""
        if not (memmap_input and not patch_check.isChecked()):
            return False
        QMessageBox.warning(
            self, "Patch-based extraction required",
            f"This movie is memory-mapped -- whole-FOV {method_name} would need to read the entire "
            f"movie into RAM. Check 'Use patch-based extraction' to run {method_name} on it, or turn "
            "off memory mapping on the Load tab.",
        )
        return True

    def _on_run_cnmf_clicked(self) -> None:
        movie = self.state.active_data()
        memmap_input = movie is not None and is_memmap(movie)
        if self._refuse_if_memmap_without_patch(memmap_input, self.cnmf_patch_check, "CNMF"):
            return
        # Both CNMF modes are capped to the same 5000-frame preview for a
        # memmap movie: patch-based CNMF is already bounded by patch
        # size regardless of frame count, but at the sizes memmap users
        # are dealing with it's still much faster to fit against a
        # representative sample than the whole recording. _commit()
        # re-extracts every accepted ROI's trace from the full movie
        # afterward, so this doesn't leave traces reflecting only the
        # preview in the committed result.
        worker_movie = preview_slice(movie) if memmap_input else None

        if self.cnmf_patch_check.isChecked():
            patch = self.cnmf_patch_size_spin.value()
            self._pending_batch_params = dict(
                patch_size=(patch, patch), overlap=self.cnmf_patch_overlap_spin.value(),
                n_components_per_patch=self.cnmf_components_per_patch_spin.value(),
                merge_thresh=self.cnmf_merge_thresh_spin.value(),
                search_radius=self.cnmf_search_radius_spin.value(),
            )
            self._run_batch_method(
                "Running patch-based CNMF (this can take a while)...", patch_cnmf_source_extraction,
                self._on_cnmf_finished, worker_movie=worker_movie, **self._pending_batch_params,
            )
            return

        self._pending_batch_params = dict(
            n_components=self.cnmf_n_components_spin.value(), search_radius=self.cnmf_search_radius_spin.value(),
            merge_thresh=self.cnmf_merge_thresh_spin.value(),
        )
        self._run_batch_method(
            "Running CNMF (this can take a while)...", cnmf_source_extraction, self._on_cnmf_finished,
            worker_movie=worker_movie, **self._pending_batch_params,
        )

    def _on_cnmf_finished(self, result: CNMFResult) -> None:
        rois = self._make_rois(
            result.masks, result.traces, "cnmf", spike_traces=result.spike_traces, params=self._pending_batch_params
        )
        self._add_candidates(rois)

    def _on_run_cnmf_e_clicked(self) -> None:
        movie = self.state.active_data()
        memmap_input = movie is not None and is_memmap(movie)
        if self._refuse_if_memmap_without_patch(memmap_input, self.cnmf_e_patch_check, "CNMF-E"):
            return
        # Same reasoning as _on_run_cnmf_clicked: capped to the 5000-frame
        # preview for a memmap movie either way; _commit() re-extracts
        # every accepted ROI's trace from the full movie afterward.
        worker_movie = preview_slice(movie) if memmap_input else None

        # search_radius/min_corr/min_pnr/ring_* are common to both modes
        # (whole-FOV takes them directly; patch mode forwards them as
        # **cnmf_e_kwargs to each patch's own cnmf_e_source_extraction
        # call) -- only n_components vs. n_components_per_patch differs.
        ring_params = dict(
            search_radius=self.cnmf_e_search_radius_spin.value(), min_corr=self.cnmf_e_min_corr_spin.value(),
            min_pnr=self.cnmf_e_min_pnr_spin.value(), ring_inner_radius=self.cnmf_e_ring_inner_radius_spin.value(),
            ring_outer_radius=self.cnmf_e_ring_outer_radius_spin.value(),
            ring_downsample=self.cnmf_e_ring_downsample_spin.value(),
            ring_max_fit_frames=self.cnmf_e_ring_max_fit_frames_spin.value(),
        )

        if self.cnmf_e_patch_check.isChecked():
            patch = self.cnmf_e_patch_size_spin.value()
            self._pending_batch_params = dict(
                patch_size=(patch, patch), overlap=self.cnmf_e_patch_overlap_spin.value(),
                n_components_per_patch=self.cnmf_e_components_per_patch_spin.value(),
                merge_thresh=self.cnmf_e_merge_thresh_spin.value(), **ring_params,
            )
            self._run_batch_method(
                "Running patch-based CNMF-E (this can take a while)...", patch_cnmf_e_source_extraction,
                self._on_cnmf_e_finished, worker_movie=worker_movie, **self._pending_batch_params,
            )
            return

        self._pending_batch_params = dict(
            n_components=self.cnmf_e_n_components_spin.value(), merge_thresh=self.cnmf_e_merge_thresh_spin.value(),
            **ring_params,
        )
        self._run_batch_method(
            "Running CNMF-E (this can take a while)...", cnmf_e_source_extraction, self._on_cnmf_e_finished,
            worker_movie=worker_movie, **self._pending_batch_params,
        )

    def _on_cnmf_e_finished(self, result: CNMFResult) -> None:
        rois = self._make_rois(
            result.masks, result.traces, "cnmf_e", spike_traces=result.spike_traces, params=self._pending_batch_params
        )
        self._add_candidates(rois)

    def _graft_shared_params(self) -> dict:
        """lambda/lamForb/lamCorr/lamCont/learn_eps -- same params dict
        keys graft_source_extraction/patch_graft_source_extraction both
        accept (forwarded straight into graft.graft's/graft.patch_graft's
        own ``params``), so one dict works for either mode. "lambda" is a
        dict *key* here (a plain string), not the Python keyword, so
        this has to be a literal rather than dict(lambda=...)."""
        return {
            "lambda": self.graft_lambda_spin.value(),
            "lamForb": self.graft_lam_forb_spin.value(),
            "lamCorr": self.graft_lam_corr_spin.value(),
            "lamCont": self.graft_lam_cont_spin.value(),
            "learn_eps": self.graft_learn_eps_spin.value(),
        }

    def _on_run_graft_clicked(self) -> None:
        if self.state.volumetric:
            self._on_run_graft_clicked_volumetric()
            return
        movie = self.state.active_data()
        memmap_input = movie is not None and is_memmap(movie)
        if self._refuse_if_memmap_without_patch(memmap_input, self.graft_patch_check, "GraFT"):
            return
        # Same reasoning as _on_run_cnmf_clicked: capped to the 5000-frame
        # preview for a memmap movie either way (patch-based GraFT is
        # already memmap-safe regardless, but fitting against the whole
        # recording is unnecessarily slow at the sizes memmap users deal
        # with) -- _commit() re-extracts traces from the full movie after.
        worker_movie = preview_slice(movie) if memmap_input else None

        if self.graft_patch_check.isChecked():
            patch = self.graft_patch_size_spin.value()
            overlap = self.graft_patch_overlap_spin.value()
            self._pending_batch_params = dict(
                patch_size=(patch, patch), overlap=(overlap, overlap),
                n_dict_per_patch=self.graft_n_dict_per_patch_spin.value(), **self._graft_shared_params(),
            )
            self._run_batch_method(
                "Running patch-based GraFT (this can take a while)...", patch_graft_source_extraction,
                self._on_graft_finished, worker_movie=worker_movie, **self._pending_batch_params,
            )
            return

        self._pending_batch_params = dict(n_dict=self.graft_n_dict_spin.value(), **self._graft_shared_params())
        self._run_batch_method(
            "Running GraFT (this can take a while)...", graft_source_extraction, self._on_graft_finished,
            worker_movie=worker_movie, **self._pending_batch_params,
        )

    def _on_run_graft_clicked_volumetric(self) -> None:
        # Must be a real, restricting mask -- not given, not empty, and
        # not all-True (an explicit Clear Mask, or an auto-threshold that
        # happened to keep everything, doesn't reduce the voxel count).
        # Mirrors orbit.roi_extraction_graft_3d._require_mask's own check,
        # just checked proactively here so a worker never even starts.
        mask = self.state.mask
        if mask is None or not mask.any() or mask.all():
            QMessageBox.warning(
                self, "Mask required",
                "GraFT on volumetric data requires a real, restricting mask -- run Mask tab "
                "(Auto-threshold) first; an empty or all-kept (Clear Mask) mask isn't enough.",
            )
            return
        movie = self.state.active_data()
        memmap_input = movie is not None and is_memmap(movie)
        if self._refuse_if_memmap_without_patch(memmap_input, self.graft_patch_check, "GraFT"):
            return
        worker_movie = preview_slice_volumetric(movie) if memmap_input else None

        # mask is bound into the callable itself (functools.partial) rather
        # than passed as a kwarg -- _run_batch_method's "already ran with
        # these exact params" fingerprint sorts+compares kwargs, and a raw
        # boolean array in that comparison would raise (ambiguous truth
        # value of an array) the moment two runs' masks were compared for
        # equality. self._pending_batch_params (used for each ROI's own
        # recorded provenance) stays JSON-serializable scalars/tuples only,
        # same convention as the 2D path above.
        if self.graft_patch_check.isChecked():
            patch = self.graft_patch_size_spin.value()
            overlap = self.graft_patch_overlap_spin.value()
            self._pending_batch_params = dict(
                patch_size=(patch, patch, patch), overlap=overlap,
                n_dict_per_patch=self.graft_n_dict_per_patch_spin.value(), **self._graft_shared_params(),
            )
            fn = functools.partial(patch_graft_source_extraction_3d, mask=self.state.mask)
            self._run_batch_method(
                "Running patch-based GraFT (this can take a while)...", fn,
                self._on_graft_finished, worker_movie=worker_movie, **self._pending_batch_params,
            )
            return

        self._pending_batch_params = dict(n_dict=self.graft_n_dict_spin.value(), **self._graft_shared_params())
        fn = functools.partial(graft_source_extraction_3d, mask=self.state.mask)
        self._run_batch_method(
            "Running GraFT (this can take a while)...", fn, self._on_graft_finished,
            worker_movie=worker_movie, **self._pending_batch_params,
        )

    def _on_graft_finished(self, result: GraFTResult | GraFTResult3D) -> None:
        rois = self._make_rois(result.masks, result.traces, "graft", params=self._pending_batch_params)
        self._add_candidates(rois)

    def _on_run_real_seudo_clicked(self) -> None:
        # Unlike every other batch method above, no memmap/patch handling
        # at all: real_seudo_source_extraction fits one frame at a time by
        # construction, so it's already memmap-safe -- no worker_movie=
        # override needed, _run_batch_method runs it directly against the
        # real active_data() regardless of whether that's memmap-backed.
        self._pending_batch_params = dict(
            sigma2=self.real_seudo_sigma2_spin.value(), lambda_blob=self.real_seudo_lambda_blob_spin.value(),
            blob_radius=self.real_seudo_blob_radius_spin.value(), pad_space=self.real_seudo_pad_space_spin.value(),
            lookahead_frames=self.real_seudo_lookahead_frames_spin.value(),
            cutoff_multiplier=self.real_seudo_cutoff_multiplier_spin.value(),
            min_roi_size=self.real_seudo_min_roi_size_spin.value(),
            min_avg_px=self.real_seudo_min_avg_px_spin.value(),
            mask_blur_rad=self.real_seudo_mask_blur_rad_spin.value(),
            exclude_radius_known_cells=self.real_seudo_exclude_radius_spin.value(),
            consecutive_frames_required=self.real_seudo_consecutive_frames_spin.value(),
            max_track_gap=self.real_seudo_max_track_gap_spin.value(),
            eq8_merge_threshold=self.real_seudo_eq8_merge_thresh_spin.value(),
            eq9_merge_threshold=self.real_seudo_eq9_merge_thresh_spin.value(),
        )
        self._run_batch_method(
            "Running Real-SEUDO (this can take a while)...", real_seudo_source_extraction,
            self._on_real_seudo_finished, **self._pending_batch_params,
        )

    def _on_real_seudo_finished(self, result: RealSeudoResult) -> None:
        rois = self._make_rois(result.masks, result.traces, "real_seudo", params=self._pending_batch_params)
        self._add_candidates(rois)

    def _make_rois(
        self, masks: list[np.ndarray], traces: list[np.ndarray], source_method: str,
        seed_locs: list[tuple[int, int] | None] | None = None,
        spike_traces: list[np.ndarray | None] | None = None,
        thresholds: list[float | None] | None = None,
        params: dict | None = None,
    ) -> list[ROI]:
        """One ROI per (mask, trace) pair, auto-incrementing self._next_id
        -- shared by every batch-producing method (correlation seeds,
        PCA-ICA, CNMF) so the id-assignment isn't re-derived per method.
        ``spike_traces`` is CNMF-only (its deconvolved output); every
        other method leaves ROI.spike_trace at its None default.
        ``params`` is this method's own parameters (shared across the
        whole batch); ``thresholds`` is correlation-only, each ROI's own
        resolved threshold, folded into its params dict (see
        session_io.py for how these get saved)."""
        seed_locs = seed_locs or [None] * len(masks)
        spike_traces = spike_traces or [None] * len(masks)
        thresholds = thresholds or [None] * len(masks)
        rois = []
        for mask, trace, seed_loc, spike_trace, threshold in zip(masks, traces, seed_locs, spike_traces, thresholds):
            roi_params = dict(params) if params else {}
            if threshold is not None:
                roi_params["threshold"] = threshold
            rois.append(
                ROI(
                    id=self._next_id, mask=mask, trace=trace, source_method=source_method,
                    seed_loc=seed_loc, spike_trace=spike_trace, params=roi_params or None,
                )
            )
            self._next_id += 1
        return rois

    def _add_candidates(self, new_rois: list[ROI]) -> None:
        self._candidates.extend(new_rois)
        self._refresh_candidates_display()

    def _on_roi_deleted(self, _roi_id: int) -> None:
        # Deleting an ROI frees its pixels for every other ROI's ring too.
        self._sync_candidates()

    def _sync_candidates(self, status_message: str | None = None) -> None:
        """Recomputes neuropil for the whole current set (any add/delete
        can change every other ROI's ring) and refreshes the review panel
        -- shared by every path that mutates self._candidates.
        compute_neuropil_traces is 2D-dilation-based (orbit.neuropil) --
        skipped for volumetric ROIs (ROI.neuropil_trace just stays None
        there); ROIReviewPanel itself needs no such guard, since it only
        ever computes its own display-only ring from a depth-projected
        2D silhouette (see roi_review_panel.py's _project_mask_2d)."""
        movie = self.state.active_data()
        if movie is not None and not self.state.volumetric:
            compute_neuropil_traces(movie, self._candidates)
        self.review_panel.set_candidates(self._candidates)
        if status_message is not None:
            self.status_label.setText(status_message)

    def _refresh_candidates_display(self) -> None:
        self._sync_candidates(f"{len(self._candidates)} ROI(s) in the collection -- review and accept before committing.")
        self.busy_bar.stop("")

    def _on_failed(self, message: str) -> None:
        self.busy_bar.stop("Failed.")
        self.status_label.setText(f"Failed: {message}")
        QMessageBox.critical(self, "Source Extraction failed", message)

    def _commit(self) -> None:
        accepted = [roi for roi in self._candidates if roi.status == "accepted"]
        if not accepted:
            QMessageBox.information(self, "Nothing to commit", "Accept at least one candidate ROI first.")
            return

        movie = self.state.active_data()
        if movie is not None and is_memmap(movie):
            self.commit_btn.setEnabled(False)
            reextract_fn = _reextract_traces_3d if self.state.volumetric else _reextract_traces
            self.worker = run_worker(
                self.busy_bar, "Re-extracting ROI traces from the full movie before committing...",
                reextract_fn, accepted, movie,
                on_success=self._on_traces_reextracted, on_failure=self._on_failed,
            )
            return
        self._finish_commit(accepted)

    def _on_traces_reextracted(self, accepted: list[ROI]) -> None:
        self.commit_btn.setEnabled(True)
        self.busy_bar.stop("")
        self._finish_commit(accepted)

    def _finish_commit(self, accepted: list[ROI]) -> None:
        methods = {roi.source_method for roi in accepted}
        label = _PIPELINE_LABELS[accepted[0].source_method] if len(methods) == 1 else "Source Extraction"
        self.state.commit_rois(accepted, label)

        self._candidates = [roi for roi in self._candidates if roi.status != "accepted"]
        self._sync_candidates(f"Committed {len(accepted)} ROI(s) as pipeline step '{label}'.")
        self.data_changed.emit()

    def _on_clear_all_clicked(self) -> None:
        """Wipes every pending candidate AND every already-committed ROI
        (state.rois) -- a full reset so Source Extraction can be redone
        from scratch without reloading the movie. Confirmed first since
        undoing committed ROIs is otherwise unrecoverable."""
        n_pending = len(self._candidates)
        n_committed = len(self.state.rois)
        if n_pending == 0 and n_committed == 0:
            return

        reply = QMessageBox.question(
            self, "Clear all traces",
            f"This clears {n_pending} pending candidate(s) and undoes {n_committed} already-committed ROI(s). "
            "This cannot be undone. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._candidates = []
        self._clear_preview()
        self._next_id = 0
        self.state.clear_rois()
        self.review_panel.set_candidates(self._candidates)
        self.status_label.setText("Cleared all candidate and committed ROIs.")
        self.data_changed.emit()
