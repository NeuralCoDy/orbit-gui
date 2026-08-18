"""Reusable image-stack slideshow: one view, prev/next through a
(H, W, K) stack -- e.g. the top-K spatial PC maps in Motion Correction,
or any future stage that wants to browse a small set of per-component
images rather than show them all at once. "View Large" pops the whole
stack out into a MovieSliderWidget, treating components as "frames" so
they can be scrubbed/played at full size with its own controls.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from movieslider.gui.movie_slider_widget import MovieSliderWidget
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class ImageSlideshow(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._images: np.ndarray | None = None
        self._index = 0
        self._player: MovieSliderWidget | None = None

        layout = QVBoxLayout(self)

        self.view = pg.ImageView()
        self.view.ui.histogram.hide()
        self.view.ui.roiBtn.hide()
        self.view.ui.menuBtn.hide()
        layout.addWidget(self.view)

        nav_row = QHBoxLayout()
        self.view_large_btn = QPushButton("View Large")
        self.view_large_btn.clicked.connect(self._view_large)
        self.prev_btn = QPushButton("<")
        self.prev_btn.clicked.connect(self._prev)
        self.index_label = QLabel("0 / 0")
        self.next_btn = QPushButton(">")
        self.next_btn.clicked.connect(self._next)
        for w in (self.view_large_btn, self.prev_btn, self.index_label, self.next_btn):
            nav_row.addWidget(w)
        layout.addLayout(nav_row)

    def set_stack(self, images: np.ndarray | None) -> None:
        """``images`` is (H, W, K), or None to clear."""
        self._images = images
        self._index = 0
        self._refresh()

    def _count(self) -> int:
        return 0 if self._images is None else self._images.shape[2]

    def _refresh(self) -> None:
        k = self._count()
        if k == 0:
            self.index_label.setText("0 / 0")
            self.view.clear()
            return
        self.index_label.setText(f"{self._index + 1} / {k}")
        self.view.setImage(self._images[:, :, self._index])

    def _prev(self) -> None:
        if self._count() == 0:
            return
        self._index = (self._index - 1) % self._count()
        self._refresh()

    def _next(self) -> None:
        if self._count() == 0:
            return
        self._index = (self._index + 1) % self._count()
        self._refresh()

    def _view_large(self) -> None:
        if self._images is None:
            return
        if self._player is None:
            self._player = MovieSliderWidget()
        self._player.show_movie(self._images)
        self._player.setWindowTitle("PC Stack Viewer")
        self._player.show()
        self._player.raise_()
        self._player.activateWindow()
