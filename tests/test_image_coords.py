"""scene_pos_to_pixel/pixel_to_data_pos must stay correct under BOTH
pyqtgraph imageAxisOrder conventions -- instantiating movieslider's
MovieSliderWidget anywhere earlier in the process silently flips the
global default from 'col-major' to 'row-major', which is exactly the
bug these functions exist to be robust against (see image_coords.py's
module docstring).
"""

import pytest

pytest.importorskip("PySide6")

import pyqtgraph as pg  # noqa: E402
from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from orbitapp.widgets.image_coords import pixel_to_data_pos, pixels_to_data_pos, scene_pos_to_pixel  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


class _FakeImageItem:
    def __init__(self, axis_order):
        self.axisOrder = axis_order


def _real_view_box() -> pg.ViewBox:
    """A ViewBox with actual widget geometry -- a bare pg.ViewBox() with no
    parent widget has degenerate scene<->view mapping (everything resolves
    to (0,0)), so scene_pos_to_pixel needs a real GraphicsView to test
    against, same as the app's own image views."""
    widget = pg.GraphicsLayoutWidget()
    view_box = widget.addViewBox()
    widget.resize(400, 300)
    widget.show()
    QApplication.processEvents()
    view_box.setRange(xRange=(0, 100), yRange=(0, 100), padding=0)
    view_box._test_widget_ref = widget  # keep the parent widget alive -- otherwise GC deletes the underlying C++ ViewBox
    return view_box


def test_scene_pos_to_pixel_col_major():
    view_box = _real_view_box()
    item = _FakeImageItem("col-major")
    row, col = scene_pos_to_pixel(item, view_box, view_box.mapViewToScene(QPointF(7, 12)))
    assert (row, col) == (7, 12)


def test_scene_pos_to_pixel_row_major():
    view_box = _real_view_box()
    item = _FakeImageItem("row-major")
    row, col = scene_pos_to_pixel(item, view_box, view_box.mapViewToScene(QPointF(12, 7)))
    assert (row, col) == (7, 12)


def test_pixel_to_data_pos_round_trips_through_scene_pos_to_pixel():
    view_box = _real_view_box()
    for axis_order in ("col-major", "row-major"):
        item = _FakeImageItem(axis_order)
        x, y = pixel_to_data_pos(item, 7, 12)
        row, col = scene_pos_to_pixel(item, view_box, view_box.mapViewToScene(QPointF(x, y)))
        assert (row, col) == (7, 12), f"round trip failed for axisOrder={axis_order}"


def test_pixels_to_data_pos_col_major_keeps_rows_as_x():
    item = _FakeImageItem("col-major")
    xs, ys = pixels_to_data_pos(item, [1, 2, 3], [4, 5, 6])
    assert xs == [1, 2, 3]
    assert ys == [4, 5, 6]


def test_pixels_to_data_pos_row_major_swaps_rows_and_cols():
    item = _FakeImageItem("row-major")
    xs, ys = pixels_to_data_pos(item, [1, 2, 3], [4, 5, 6])
    assert xs == [4, 5, 6]
    assert ys == [1, 2, 3]
