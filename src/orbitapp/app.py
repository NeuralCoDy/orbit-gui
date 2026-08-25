"""Main orbitapp window: a persistent header + one tab per pipeline
stage. More stage tabs arrive as their phases are implemented.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QSplashScreen, QTabWidget, QVBoxLayout, QWidget

from . import io as orbitapp_io
from .assets import LOGO_PATH, load_logo_on_black
from .format import format_header_summary
from .state import AppState
from .tabs import (
    DenoisingTab,
    LoadTab,
    MaskTab,
    MotionCorrectionTab,
    NormalizationTab,
    ProjectionsTab,
    ROIValidationTab,
    SaveTab,
    SourceExtractionTab,
)
from .theme import apply_dark_theme
from .widgets import HeaderBar

_SPLASH_MIN_DISPLAY_S = 1.2
_SPLASH_HEIGHT_PX = 420


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("orbit")
        self.resize(1100, 750)

        self.state = AppState()

        self.header = HeaderBar()
        self.load_tab = LoadTab(self.state)
        self.projections_tab = ProjectionsTab(self.state)
        self.motion_correction_tab = MotionCorrectionTab(self.state)
        self.mask_tab = MaskTab(self.state)
        self.denoising_tab = DenoisingTab(self.state)
        self.normalization_tab = NormalizationTab(self.state)
        self.source_extraction_tab = SourceExtractionTab(self.state)
        self.roi_validation_tab = ROIValidationTab(self.state)
        self.save_tab = SaveTab(self.state, self.roi_validation_tab)

        # Every tab that reads active_data() -- refreshed whenever new data
        # loads or any stage below commits a change (see the wiring loop).
        # ROI Validation also depends on state.rois, which only Source
        # Extraction's Commit changes -- wired separately below since it's
        # not one of _mutating_tabs (that set changes the active *movie*).
        self._stage_tabs = [
            self.projections_tab,
            self.motion_correction_tab,
            self.mask_tab,
            self.denoising_tab,
            self.normalization_tab,
            self.source_extraction_tab,
            self.roi_validation_tab,
        ]
        # Subset that can mutate the active *movie* (Commit). Source
        # Extraction's Commit only adds to state.rois -- active_data() is
        # unchanged, so there's nothing for other movie-consuming tabs to
        # refresh -- it's wired to the header breadcrumb separately below.
        self._mutating_tabs = [self.motion_correction_tab, self.mask_tab, self.denoising_tab, self.normalization_tab]

        self.tabs = QTabWidget()
        self.tabs.addTab(self.load_tab, "Load")
        self.tabs.addTab(self.projections_tab, "Data Projections")
        self.tabs.addTab(self.motion_correction_tab, "Motion Correction")
        self.tabs.addTab(self.mask_tab, "Mask")
        self.tabs.addTab(self.denoising_tab, "Denoising")
        self.tabs.addTab(self.normalization_tab, "Normalization")
        self.tabs.addTab(self.source_extraction_tab, "Source Extraction")
        self.tabs.addTab(self.roi_validation_tab, "ROI Validation")
        self.tabs.addTab(self.save_tab, "Save")

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.header)
        central_layout.addWidget(self.tabs)
        self.setCentralWidget(central)

        self.load_tab.data_loaded.connect(self._on_data_loaded)
        for tab in self._stage_tabs:
            self.load_tab.data_loaded.connect(tab.on_data_loaded)

        for tab in self._mutating_tabs:
            tab.data_changed.connect(self._on_data_committed)
            for other in self._stage_tabs:
                tab.data_changed.connect(other.on_data_loaded)

        self.source_extraction_tab.data_changed.connect(self._on_data_committed)
        self.source_extraction_tab.data_changed.connect(self.roi_validation_tab.on_data_loaded)

        self.load_tab.modality_changed.connect(self._on_modality_changed)
        self.load_tab.modality_changed.connect(self.motion_correction_tab.on_modality_changed)

        self.save_tab.session_loaded.connect(self._on_session_loaded)

    def _on_data_loaded(self) -> None:
        movie = self.state.original_data
        path = self.state.data_path
        if movie is None or path is None:
            summary = None
        elif self.state.volumetric:
            # format_header_summary hard-assumes a 3D (H, W, T) movie
            # (height, width, n_frames = movie.shape) -- a volumetric
            # (T, L, W, D) movie needs its own one-liner rather than a
            # change to that shared, 2D-only helper.
            n_frames, length, width, depth = movie.shape
            summary = f"{Path(path).name}  --  {length} x {width} x {depth}, {n_frames} time-steps"
        else:
            summary = format_header_summary(path, movie)
        self.header.set_data_info(summary)
        self.header.set_pipeline(self.state.pipeline, self.state.modality_modifiers())

    def _on_data_committed(self) -> None:
        """A stage tab committed a new active dataset (see
        orbitapp.widgets.CommitControls) -- reflect the updated pipeline
        breadcrumb."""
        self.header.set_pipeline(self.state.pipeline, self.state.modality_modifiers())

    def _on_modality_changed(self) -> None:
        """One of the Load tab's data-modality toggles flipped -- only
        the header caption reflects this today (see AppState.
        modality_modifiers)."""
        self.header.set_pipeline(self.state.pipeline, self.state.modality_modifiers())

    def _on_session_loaded(self, session: dict) -> None:
        """SaveTab only reads/writes files -- reconstructing AppState and
        refreshing every other tab from what it loaded is done here,
        same as every other cross-tab wiring in this class."""
        pipeline = session.get("pipeline")
        output = session.get("output")

        data_path = pipeline["data_path"] if pipeline else None
        if data_path:
            try:
                movie = orbitapp_io.load_movie(data_path)
                self.state.load(data_path, movie)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "Could not reload movie", f"{data_path}: {exc}")

        if pipeline is not None:
            self.state.pipeline = pipeline["pipeline"]
            self.state.steps = pipeline["steps"]
            tabs_by_stage = {
                self.motion_correction_tab._stage_key: self.motion_correction_tab,
                self.mask_tab._stage_key: self.mask_tab,
                self.denoising_tab._stage_key: self.denoising_tab,
                self.normalization_tab._stage_key: self.normalization_tab,
            }
            for step in pipeline["steps"]:
                tab = tabs_by_stage.get(step.stage)
                if tab is not None and step.params:
                    tab.restore_params(step.params)

        if output is not None:
            self.state.rois = output["rois"]
            # the pipeline file (if also loaded) is the only place
            # seed_loc/params survive -- the output file only has the
            # mask/trace/spike/neuropil arrays, matched back up by id.
            if pipeline is not None:
                meta_by_id = {meta["id"]: meta for meta in pipeline["source_extraction_rois"]}
                for roi in self.state.rois:
                    meta = meta_by_id.get(roi.id)
                    if meta is not None:
                        roi.seed_loc = meta["seed_loc"]
                        roi.params = meta["params"] or None
        elif pipeline is not None and pipeline["source_extraction_rois"]:
            QMessageBox.information(
                self, "Partial load",
                "Pipeline loaded, but no output file was found alongside it -- ROI mask/trace "
                "data wasn't restored, only its recorded parameters.",
            )

        for tab in self._stage_tabs:
            tab.on_data_loaded()
        self._on_data_loaded()

        if output is not None and output["roi_validation_results"] is not None and self.state.rois:
            self.roi_validation_tab._on_load_rois_clicked()
            self.roi_validation_tab.import_results(output["roi_validation_results"])


def run() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    apply_dark_theme(app)

    splash = None
    if LOGO_PATH.exists():
        splash = QSplashScreen(load_logo_on_black(_SPLASH_HEIGHT_PX))
        splash.show()
        app.processEvents()

    start = time.perf_counter()
    window = MainWindow()

    if splash is not None:
        remaining = _SPLASH_MIN_DISPLAY_S - (time.perf_counter() - start)
        end = time.perf_counter() + max(remaining, 0.0)
        while time.perf_counter() < end:
            app.processEvents()
            time.sleep(0.02)
        splash.finish(window)

    window.show()
    app.exec()
