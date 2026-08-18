"""Shared "pop this stack/movie out into its own viewer" behavior used
by both StagePanel's Play Movie buttons and ImageSlideshow's View Large.
"""

from __future__ import annotations

import numpy as np
from movieslider.gui.movie_slider_widget import MovieSliderWidget


def show_movie_popout(player: MovieSliderWidget | None, data: np.ndarray, title: str) -> MovieSliderWidget:
    """Lazily creates (or reuses) a MovieSliderWidget showing ``data``,
    titled ``title``, and brings it to front. Returns the player so the
    caller can hold onto it for next time."""
    if player is None:
        player = MovieSliderWidget()
    player.show_movie(data)
    player.setWindowTitle(title)
    player.show()
    player.raise_()
    player.activateWindow()
    return player
