"""Shared application state, passed between tabs (mirrors pyGraFT's
graftapp/state.py). More fields (params, results, ...) get added as
later pipeline stages need them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AppState:
    data_path: str | None = None
    original_data: np.ndarray | None = None  # (H, W, T), as loaded
    preprocessed_data: np.ndarray | None = None  # after the latest *committed* stage
    pipeline: list[str] = field(default_factory=list)  # committed stage names, e.g. ["Load", "Patch Warp"]

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

    def commit(self, data: np.ndarray, step_name: str) -> None:
        """Makes ``data`` the new active dataset and records ``step_name``
        in the pipeline -- the only way active_data() should change after
        the initial load, called when the user explicitly clicks Commit."""
        self.preprocessed_data = data
        self.pipeline.append(step_name)
