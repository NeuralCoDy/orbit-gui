"""Main orbitapp window: a persistent header + one tab per pipeline
stage. More stage tabs arrive as their phases are implemented.
"""

from __future__ import annotations

import sys
import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow, QSplashScreen, QTabWidget, QVBoxLayout, QWidget

from .assets import LOGO_PATH
from .format import format_header_summary
from .state import AppState
from .tabs import DenoisingTab, LoadTab, MotionCorrectionTab, NormalizationTab, ProjectionsTab
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
        self.denoising_tab = DenoisingTab(self.state)
        self.normalization_tab = NormalizationTab(self.state)

        # Every tab that reads active_data() -- refreshed whenever new data
        # loads or any stage below commits a change (see the wiring loop).
        self._stage_tabs = [
            self.projections_tab,
            self.motion_correction_tab,
            self.denoising_tab,
            self.normalization_tab,
        ]
        # Subset that can actually mutate the active dataset (Commit).
        self._mutating_tabs = [self.motion_correction_tab, self.denoising_tab, self.normalization_tab]

        self.tabs = QTabWidget()
        self.tabs.addTab(self.load_tab, "Load")
        self.tabs.addTab(self.projections_tab, "Data Projections")
        self.tabs.addTab(self.motion_correction_tab, "Motion Correction")
        self.tabs.addTab(self.denoising_tab, "Denoising")
        self.tabs.addTab(self.normalization_tab, "Normalization")

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

        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._on_tab_changed(self.tabs.currentIndex())

    def _on_data_loaded(self) -> None:
        movie = self.state.original_data
        path = self.state.data_path
        summary = None if movie is None or path is None else format_header_summary(path, movie)
        self.header.set_data_info(summary)
        self.header.set_pipeline(self.state.pipeline)

    def _on_data_committed(self) -> None:
        """A stage tab committed a new active dataset (see
        orbitapp.widgets.CommitControls) -- reflect the updated pipeline
        breadcrumb."""
        self.header.set_pipeline(self.state.pipeline)

    def _on_tab_changed(self, index: int) -> None:
        self.header.set_active_stage(self.tabs.tabText(index))


def run() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    apply_dark_theme(app)

    splash = None
    if LOGO_PATH.exists():
        pixmap = QPixmap(str(LOGO_PATH)).scaledToHeight(_SPLASH_HEIGHT_PX, Qt.TransformationMode.SmoothTransformation)
        splash = QSplashScreen(pixmap)
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
