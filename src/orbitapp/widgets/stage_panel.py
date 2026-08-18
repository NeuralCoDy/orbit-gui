"""Reusable layout for pipeline-stage tabs: before/after images on top,
a metrics row (numbers/plots) underneath -- metrics get the full window
width rather than being squeezed into a sidebar. Every stage tab should
embed a StagePanel rather than rolling its own arrangement.

Each image also gets a "Play Movie" button (alongside pg.ImageView's own
ROI/Menu buttons) that pops up the full (H, W, T) movie behind that
projection in a MovieSliderWidget -- set via set_before_movie/
set_after_movie, since the panel itself only ever displays a single 2D
projection (e.g. a mean image), not the movie.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from movieslider.gui.movie_slider_widget import MovieSliderWidget
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .movie_popout import show_movie_popout


class StagePanel(QWidget):
    def __init__(self, before_title: str = "Before", after_title: str = "After", parent=None) -> None:
        super().__init__(parent)
        self._movies: dict[str, np.ndarray | None] = {"before": None, "after": None}
        self._players: dict[str, MovieSliderWidget | None] = {"before": None, "after": None}
        self._titles = {"before": before_title, "after": after_title}

        layout = QVBoxLayout(self)

        images_row = QHBoxLayout()
        self.before_view = pg.ImageView()
        self.after_view = pg.ImageView()
        for key, view, title in (("before", self.before_view, before_title), ("after", self.after_view, after_title)):
            col = QVBoxLayout()
            col.addWidget(QLabel(title))
            col.addWidget(view)
            play_btn = QPushButton("Play Movie")
            play_btn.clicked.connect(lambda _checked=False, key=key: self._play_movie(key))
            col.addWidget(play_btn)
            images_row.addLayout(col)
        images_container = QWidget()
        images_container.setLayout(images_row)
        layout.addWidget(images_container, stretch=2)

        self.metrics_row = QHBoxLayout()
        metrics_container = QWidget()
        metrics_container.setLayout(self.metrics_row)
        layout.addWidget(metrics_container, stretch=1)

    def add_metric_widget(self, widget: QWidget) -> None:
        """Add one metric/plot widget to the row below the images."""
        self.metrics_row.addWidget(widget)

    def set_before_movie(self, movie: np.ndarray | None) -> None:
        self._movies["before"] = movie

    def set_after_movie(self, movie: np.ndarray | None) -> None:
        self._movies["after"] = movie

    def _play_movie(self, key: str) -> None:
        movie = self._movies[key]
        if movie is None:
            return
        self._players[key] = show_movie_popout(self._players[key], movie, f"Movie Player - {self._titles[key]}")
