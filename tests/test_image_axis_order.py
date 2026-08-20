"""Regression check for the pyqtgraph image axis-order assumption that
SourceExtractionTab's click-to-pixel mapping depends on. orbit-gui never
calls pg.setConfigOptions(imageAxisOrder=...), so it stays at pyqtgraph's
default ('col-major'): array axis 0 maps to the image item's local
x-coordinate, axis 1 to y. Getting this backwards (e.g. by copying
roiapp's own click-mapping code, which explicitly sets
imageAxisOrder="row-major") would silently misalign every click -- see
SourceExtractionTab._on_scene_clicked's comment.
"""

import numpy as np
import pyqtgraph as pg
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


def test_default_image_axis_order_maps_array_axis0_to_x():
    QApplication.instance() or QApplication([])
    # Pinned explicitly rather than relying on nothing else in this
    # process having touched it -- any earlier-run test that constructs
    # a MovieSliderWidget (LoadTab, MainWindow, ...) silently flips this
    # *global* pyqtgraph config for the rest of the process (see
    # orbitapp.widgets.image_coords's docstring), which would otherwise
    # make this test's outcome depend on test collection order. "col-major"
    # is genuinely pyqtgraph's own default -- this isn't testing something
    # different, just making that fact immune to ambient state.
    pg.setConfigOptions(imageAxisOrder="col-major")
    image_item = pg.ImageItem()
    image_item.setImage(np.zeros((40, 60)))

    rect = image_item.boundingRect()

    assert rect.width() == 40  # array axis 0 (rows) -> local x
    assert rect.height() == 60  # array axis 1 (cols) -> local y
