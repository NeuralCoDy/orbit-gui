"""Shared application state, passed between tabs (mirrors pyGraFT's
graftapp/state.py). More fields (params, results, ...) get added as
later pipeline stages need them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ROI:
    """One region of interest found by a Source Extraction method --
    Correlation-based, PCA-ICA, or CNMF all produce these, so later stages
    (review UI, export, ...) can treat them uniformly regardless of source."""

    id: int
    mask: np.ndarray  # (H, W) bool
    trace: np.ndarray  # (T,) float -- masked-mean fluorescence
    source_method: str  # "correlation" | "pca_ica" | "cnmf"
    status: str = "pending"  # "pending" | "accepted" | "rejected"
    spike_trace: np.ndarray | None = None  # CNMF-only deconvolved spikes
    seed_loc: tuple[int, int] | None = None  # correlation method only -- the pixel clicked
    neuropil_trace: np.ndarray | None = None  # masked-mean of the dilated ring, other ROIs excluded
    params: dict | None = None  # the method's own parameters used to produce this ROI (see session_io.py)


@dataclass
class PipelineStep:
    """One committed pipeline stage, recorded in order -- what
    session_io.py's "pipeline" file is built from. ``params`` are the
    exact algorithm kwargs used (JSON-serializable plain types only);
    ``metrics`` are whatever headline QC numbers that stage computed
    (empty for stages that don't have any, e.g. Load)."""

    stage: str  # e.g. "motion_correction", "denoising", "normalization", "source_extraction"
    label: str  # human-readable, matches AppState.pipeline's breadcrumb strings
    params: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


@dataclass
class AppState:
    data_path: str | None = None
    original_data: np.ndarray | None = None  # (H, W, T), as loaded
    preprocessed_data: np.ndarray | None = None  # after the latest *committed* stage
    pipeline: list[str] = field(default_factory=list)  # committed stage names, e.g. ["Load", "Patch Warp"]
    rois: list[ROI] = field(default_factory=list)  # committed ROIs; accumulates across commits/methods
    steps: list[PipelineStep] = field(default_factory=list)  # parallel to pipeline, with params/metrics attached

    def active_data(self) -> np.ndarray | None:
        """The data later stages should operate on: the latest committed
        version if one exists, else the originally-loaded movie. A stage
        tab's own in-progress/uncommitted candidate result (see
        orbitapp.widgets.CommitControls) is never reflected here -- only
        commit() changes what this returns."""
        return self.preprocessed_data if self.preprocessed_data is not None else self.original_data

    def load(self, path: str, movie: np.ndarray) -> None:
        """Registers freshly loaded data, resetting any prior pipeline
        (a new file makes earlier committed stages meaningless)."""
        self.data_path = path
        self.original_data = movie
        self.preprocessed_data = None
        self.pipeline = ["Load"]
        self.rois = []
        self.steps = [PipelineStep(stage="load", label="Load", params={"data_path": path})]

    def commit(self, data: np.ndarray, step_name: str, stage: str = "", params: dict | None = None,
               metrics: dict | None = None) -> None:
        """Makes ``data`` the new active dataset and records ``step_name``
        in the pipeline -- the only way active_data() should change after
        the initial load, called when the user explicitly clicks Commit."""
        self.preprocessed_data = data
        self.pipeline.append(step_name)
        self.steps.append(PipelineStep(stage=stage, label=step_name, params=params or {}, metrics=metrics or {}))

    def commit_rois(self, new_rois: list[ROI], step_name: str, params: dict | None = None) -> None:
        """Adds newly-accepted ROIs to the committed set and records
        ``step_name`` in the pipeline -- parallel to commit(), but ROIs
        accumulate across multiple rounds/methods rather than replacing a
        single active movie (unlike commit(), which always replaces).
        ``params`` here is the batch method's own parameters (PCA-ICA/
        CNMF); correlation-based ROIs instead carry their own per-ROI
        seed_loc/params, set directly on each ROI before it reaches here."""
        self.rois.extend(new_rois)
        self.pipeline.append(step_name)
        self.steps.append(PipelineStep(stage="source_extraction", label=step_name, params=params or {}))

    def clear_rois(self) -> None:
        """Removes every committed ROI, undoing any number of prior
        commit_rois() calls -- lets Source Extraction be redone from
        scratch without reloading the movie. The pipeline breadcrumb is
        left as-is (a history log of what ran, not current state -- same
        as every other stage's commits)."""
        self.rois = []
