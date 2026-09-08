"""Reusable layout for pipeline-stage tabs: before/after images on top,
a metrics row (numbers/plots) underneath -- metrics get the full window
width rather than being squeezed into a sidebar. Every stage tab should
embed a StagePanel rather than rolling its own arrangement.

Each image also gets a "Play Movie" button (alongside pg.ImageView's own
ROI/Menu buttons) that pops up the full (H, W, T) movie behind that
projection in a MovieSliderWidget -- set via set_before_movie/
set_after_movie, since the panel itself only ever displays a single 2D
projection (e.g. a mean image), not the movie. add_extra_movie_button
adds another such button/movie slot, placed next to (not below)
``column``'s existing Play Movie button, each splitting the row's width
evenly -- for a stage with a movie worth previewing besides its own
before/after image (e.g. Denoising's residual).

For volumetric (T, L, W, D) data each column swaps its 2D ImageView for
a drag-rotatable 3D VolumeView (see set_volumetric): the time-mean
volume in perspective rather than a flat depth projection. Play Movie
buttons are hidden in that mode -- the 3D view is the viewer.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from movieslider.gui.movie_slider_widget import MovieSliderWidget
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from ..io import preview_slice
from .movie_popout import show_movie_popout
from .volume_view import VolumeView


class StagePanel(QWidget):
    def __init__(self, before_title: str = "Before", after_title: str = "After", parent=None) -> None:
        super().__init__(parent)
        self._movies: dict[str, np.ndarray | None] = {"before": None, "after": None}
        self._players: dict[str, MovieSliderWidget | None] = {"before": None, "after": None}
        self._titles = {"before": before_title, "after": after_title}
        self._button_rows: dict[str, QHBoxLayout] = {}
        self._button_containers: dict[str, QWidget] = {}
        self._volumetric = False

        layout = QVBoxLayout(self)

        images_row = QHBoxLayout()
        self.before_view = pg.ImageView()
        self.after_view = pg.ImageView()
        self.before_volume = VolumeView()
        self.after_volume = VolumeView(placeholder="Run this stage to see the candidate volume in 3D.")
        self._stacks: dict[str, QStackedWidget] = {}
        for key, view, volume, title in (
            ("before", self.before_view, self.before_volume, before_title),
            ("after", self.after_view, self.after_volume, after_title),
        ):
            col = QVBoxLayout()
            col.addWidget(QLabel(title))
            stack = QStackedWidget()
            stack.addWidget(view)  # index 0: 2D projection
            stack.addWidget(volume)  # index 1: 3D volume (volumetric data)
            self._stacks[key] = stack
            col.addWidget(stack)
            button_row = QHBoxLayout()
            play_btn = QPushButton("Play Movie")
            play_btn.clicked.connect(lambda _checked=False, key=key: self._play_movie(key))
            button_row.addWidget(play_btn, 1)
            button_container = QWidget()
            button_container.setLayout(button_row)
            col.addWidget(button_container)
            self._button_rows[key] = button_row
            self._button_containers[key] = button_container
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

    def add_extra_movie_button(self, column: str, key: str, label: str) -> None:
        """Adds another independently-playable movie slot, with its own
        button placed next to ``column``'s ("before" or "after")
        existing Play Movie button (each splits the row's width evenly)
        -- for a stage with a movie worth previewing besides its own
        before/after image. Populate it via set_movie(key, ...); title
        bar of its popout is just ``label``."""
        self._movies[key] = None
        self._players[key] = None
        self._titles[key] = label
        play_btn = QPushButton(label)
        play_btn.clicked.connect(lambda _checked=False, key=key: self._play_movie(key))
        self._button_rows[column].addWidget(play_btn, 1)

    # -- volumetric mode --------------------------------------------------

    def set_volumetric(self, enabled: bool) -> None:
        """Switch both columns between the 2D ImageView (``False``) and
        the 3D VolumeView (``True``). Play Movie buttons are hidden in 3D
        mode. Safe to call repeatedly / on every data load."""
        self._volumetric = enabled
        index = 1 if enabled else 0
        for key in ("before", "after"):
            self._stacks[key].setCurrentIndex(index)
            self._button_containers[key].setVisible(not enabled)
            if not enabled:
                self._stacks[key].widget(1).clear()

    def set_before_volume(self, volume: np.ndarray | None) -> None:
        self._set_volume("before", volume)

    def set_after_volume(self, volume: np.ndarray | None) -> None:
        self._set_volume("after", volume)

    def _set_volume(self, key: str, volume: np.ndarray | None) -> None:
        view = self._stacks[key].widget(1)
        if volume is None:
            view.clear()
        else:
            view.set_volume(volume)

    def set_movie(self, key: str, movie: np.ndarray | None) -> None:
        self._movies[key] = movie

    def set_before_movie(self, movie: np.ndarray | None) -> None:
        self.set_movie("before", movie)

    def set_after_movie(self, movie: np.ndarray | None) -> None:
        self.set_movie("after", movie)

    def _play_movie(self, key: str) -> None:
        movie = self._movies[key]
        if movie is None:
            return
        # preview_slice caps a memmap-backed movie to its first 5000
        # frames -- show_movie's full-array histogram scan would
        # otherwise force a full read of an otherwise-still-lazy movie.
        self._players[key] = show_movie_popout(
            self._players[key], preview_slice(movie), f"Movie Player - {self._titles[key]}"
        )
