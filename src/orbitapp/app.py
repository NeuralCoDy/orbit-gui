"""Main orbitapp window: a persistent header + one tab per pipeline
stage. More stage tabs arrive as their phases are implemented.
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget

from .format import format_header_summary
from .state import AppState
from .tabs import LoadTab, MotionCorrectionTab, ProjectionsTab
from .theme import apply_dark_theme
from .widgets import HeaderBar


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

        self.tabs = QTabWidget()
        self.tabs.addTab(self.load_tab, "Load")
        self.tabs.addTab(self.projections_tab, "Data Projections")
        self.tabs.addTab(self.motion_correction_tab, "Motion Correction")

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.header)
        central_layout.addWidget(self.tabs)
        self.setCentralWidget(central)

        self.load_tab.data_loaded.connect(self._on_data_loaded)
        self.load_tab.data_loaded.connect(self.projections_tab.on_data_loaded)
        self.load_tab.data_loaded.connect(self.motion_correction_tab.on_data_loaded)
        self.motion_correction_tab.data_changed.connect(self._on_data_committed)
        self.motion_correction_tab.data_changed.connect(self.projections_tab.on_data_loaded)
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
    window = MainWindow()
    window.show()
    app.exec()
