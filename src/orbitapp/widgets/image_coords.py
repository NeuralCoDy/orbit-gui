"""Scene-position <-> array-index conversion for an interactive image
click, robust to pyqtgraph's *global* imageAxisOrder config -- which
third-party widgets can silently flip as a side effect of being
constructed anywhere earlier in the process (confirmed: instantiating
movieslider's MovieSliderWidget, used by LoadTab/StagePanel/
ImageSlideshow, switches it from the default 'col-major' to
'row-major' application-wide). Every plain image *display* in this app
is unaffected by that (setImage/render both just follow whatever the
current convention is, consistently), but code that converts a click's
scene position into an array (row, col) -- or the reverse, placing a
marker at a given array index -- must ask the specific ImageItem what
convention it actually got constructed under, not assume one.
"""

from __future__ import annotations

import pyqtgraph as pg


def scene_pos_to_pixel(image_item: pg.ImageItem, view_box: pg.ViewBox, scene_pos) -> tuple[int, int]:
    """(row, col) array indices for a scene position (e.g. from a
    sigMouseClicked event's event.scenePos())."""
    view_pos = view_box.mapSceneToView(scene_pos)
    if image_item.axisOrder == "row-major":
        return int(round(view_pos.y())), int(round(view_pos.x()))
    return int(round(view_pos.x())), int(round(view_pos.y()))


def pixel_to_data_pos(image_item: pg.ImageItem, row: float, col: float) -> tuple[float, float]:
    """(x, y) data/view coordinates for array index (row, col) -- the
    inverse of scene_pos_to_pixel's axis assignment, for placing a
    marker/scatter point so it visually lands on that exact pixel."""
    if image_item.axisOrder == "row-major":
        return col, row
    return row, col


def pixels_to_data_pos(image_item: pg.ImageItem, rows: list[float], cols: list[float]) -> tuple[list[float], list[float]]:
    """Vectorized pixel_to_data_pos for several points at once -- (xs, ys)
    ready for ScatterPlotItem.setData(xs, ys)."""
    if image_item.axisOrder == "row-major":
        return list(cols), list(rows)
    return list(rows), list(cols)
