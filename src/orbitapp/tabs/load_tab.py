"""Load tab: browse a movie file/folder; a sidebar shows load controls,
a busy indicator, and a human-readable summary, and a large embedded
MovieSliderWidget (bundled with the roiapp distribution) shows the
movie itself once loaded. Data projections live in their own "Data
Projections" tab (projections_tab.py).

Loading runs in a background FunctionWorker -- some formats (a folder of
many TIFFs, non-memmap-able stacks) take well over the 2-second bar for
showing a busy indicator, and the GUI must stay responsive regardless.
"""

from __future__ import annotations

import numpy as np
from movieslider.gui.movie_slider_widget import MovieSliderWidget
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import io as orbitapp_io
from ..format import format_movie_summary
from ..state import AppState
from ..widgets import BusyBar, confirm_recompute, show_movie_popout
from ..workers import FunctionWorker, run_worker


class LoadTab(QWidget):
    data_loaded = Signal()

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.worker: FunctionWorker | None = None
        self._pending_path: str | None = None
        self._movie_player: MovieSliderWidget | None = None

        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        self._browse_buttons: list[QPushButton] = []
        for label, slot in (
            ("Browse File...", self._browse_file),
            ("Browse Folder (TIFF sequence)...", self._browse_folder),
            ("Load Default Dataset", self._load_default_dataset),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            sidebar_layout.addWidget(btn)
            self._browse_buttons.append(btn)

        self.view_movie_btn = QPushButton("View Movie")
        self.view_movie_btn.setEnabled(False)
        self.view_movie_btn.clicked.connect(self._on_view_movie_clicked)
        sidebar_layout.addWidget(self.view_movie_btn)

        self.busy_bar = BusyBar()
        sidebar_layout.addWidget(self.busy_bar)

        self.info_label = QLabel("No data loaded.")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        sidebar_layout.addWidget(self.info_label)
        sidebar_layout.addStretch()

        self.movie_view = MovieSliderWidget()

        splitter = QSplitter()
        splitter.addWidget(sidebar)
        splitter.addWidget(self.movie_view)
        splitter.setSizes([260, 840])

        layout = QHBoxLayout(self)
        layout.addWidget(splitter)

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select movie file", "", "Movies (*.tif *.tiff *.npy *.h5 *.hdf5 *.mat);;All files (*)"
        )
        if path:
            self._load(path)

    def _browse_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select folder of TIFF frames")
        if path:
            self._load(path)

    def _load(self, path: str) -> None:
        self._begin_load(path, f"Loading {path}...", orbitapp_io.load_movie, path)

    def _load_default_dataset(self) -> None:
        path = str(orbitapp_io.DEFAULT_DATASET_PATH)
        self._begin_load(path, "Loading default dataset (downloading if needed)...", orbitapp_io.load_default_dataset)

    def _begin_load(self, path: str, message: str, fn, *args) -> None:
        """Shared by every "load a movie" action -- skips redoing the
        work (after confirming) if this exact path is already the active
        dataset, since re-downloading/re-reading it would just reproduce
        what's already loaded."""
        if self.state.data_path == path and not confirm_recompute(self, f"'{path}' is already loaded."):
            return
        self._pending_path = path
        self._start_load(message, fn, *args)

    def _start_load(self, message: str, fn, *args) -> None:
        for btn in self._browse_buttons:
            btn.setEnabled(False)
        self.worker = run_worker(
            self.busy_bar, message, fn, *args, on_success=self._on_loaded, on_failure=self._on_failed
        )

    def _on_loaded(self, movie: np.ndarray) -> None:
        path = self._pending_path
        self.state.load(path, movie)
        self.info_label.setText(format_movie_summary(path, movie))

        # show_movie() computes a full pixel histogram for its contrast
        # controls -- a genuine ~seconds-scale cost on a large movie, and
        # it has to run synchronously here (Qt widgets aren't safe to
        # touch off the GUI thread), so at least say what's happening
        # rather than freezing silently after "Loading" disappears.
        self.busy_bar.set_message(f"Rendering movie viewer for {path}...")
        QApplication.processEvents()
        self.movie_view.show_movie(movie)

        self.busy_bar.stop(f"Loaded {path}.")
        for btn in self._browse_buttons:
            btn.setEnabled(True)
        self.view_movie_btn.setEnabled(True)

        self.data_loaded.emit()

    def _on_view_movie_clicked(self) -> None:
        movie = self.state.active_data()
        if movie is None:
            return
        self._movie_player = show_movie_popout(self._movie_player, movie, "Movie Player - Loaded Movie")

    def _on_failed(self, message: str) -> None:
        self.busy_bar.stop("Load failed.")
        for btn in self._browse_buttons:
            btn.setEnabled(True)
        QMessageBox.critical(self, "Load failed", message)
