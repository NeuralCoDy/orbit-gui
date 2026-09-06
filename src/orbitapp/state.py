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
    mask: np.ndarray | None = None  # (H, W) bool, or (L, W, D) for volumetric -- set whenever Mask tab commits
    pipeline: list[str] = field(default_factory=list)  # committed stage names, e.g. ["Load", "Patch Warp"]
    rois: list[ROI] = field(default_factory=list)  # committed ROIs; accumulates across commits/methods
    steps: list[PipelineStep] = field(default_factory=list)  # parallel to pipeline, with params/metrics attached

    # Load tab's data-modality toggles -- describe the *kind* of dataset
    # being analyzed (independent of any one file, so NOT reset by
    # load() the way pipeline/rois/steps are). Purely informational for
    # now: they only label the header's pipeline diagram (see
    # modality_modifiers below) -- a later, larger change will make
    # these actually affect processing.
    dendrites: bool = False
    axons: bool = False
    widefield: bool = False
    volumetric: bool = False
    somatic_1p: bool = False
    somatic_2p: bool = False

    def modality_modifiers(self) -> list[str]:
        """Which of the Load tab's data-modality toggles are on, in a
        fixed display order -- used to label the header's pipeline
        diagram, e.g. "Current pipeline (widefield):"."""
        names = ("dendrites", "axons", "widefield", "volumetric", "somatic_1p", "somatic_2p")
        return [name for name in names if getattr(self, name)]

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
        self.mask = None
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

    def commit_rois(
        self, new_rois: list[ROI], step_name: str, params: dict | None = None, metrics: dict | None = None,
    ) -> None:
        """Adds newly-accepted ROIs to the committed set and records
        ``step_name`` in the pipeline -- parallel to commit(), but ROIs
        accumulate across multiple rounds/methods rather than replacing a
        single active movie (unlike commit(), which always replaces).
        ``params`` here is the batch method's own parameters (PCA-ICA/
        CNMF); correlation-based ROIs instead carry their own per-ROI
        seed_loc/params, set directly on each ROI before it reaches here.
        ``metrics`` is this commit's own headline numbers (e.g. how many
        of the run's candidates were committed vs. deleted before
        commit) -- see SourceExtractionTab._finish_commit."""
        self.rois.extend(new_rois)
        self.pipeline.append(step_name)
        self.steps.append(
            PipelineStep(stage="source_extraction", label=step_name, params=params or {}, metrics=metrics or {})
        )

    def clear_rois(self) -> None:
        """Removes every committed ROI, undoing any number of prior
        commit_rois() calls -- lets Source Extraction be redone from
        scratch without reloading the movie. Unlike every other stage's
        commits (a history log of what ran, left as-is even after later
        stages supersede their output), the source-extraction pipeline
        breadcrumbs are removed here too: with every committed ROI gone,
        those blocks would otherwise claim ROIs were produced/committed
        that no longer exist anywhere in the session. self.pipeline is
        always steps' own labels in order (see commit/commit_rois), so
        rebuilding it from the filtered steps keeps both in sync."""
        self.rois = []
        self.steps = [step for step in self.steps if step.stage != "source_extraction"]
        self.pipeline = [step.label for step in self.steps]
