import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from orbitapp.state import AppState  # noqa: E402
from orbitapp.tabs.projections_tab import _VOLUME_3D, ProjectionsTab  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def _drain(tab):
    if tab.worker is not None:
        tab.worker.wait(20000)
    for _ in range(40):
        QApplication.processEvents()


def _mouse(kind, x, y):
    return QMouseEvent(
        kind, QPointF(x, y), QPointF(x, y),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )


def _item(tab, name):
    model = tab.selector.model()
    for i in range(model.rowCount()):
        if model.item(i).text() == name:
            return model.item(i)
    raise AssertionError(name)


def test_2d_movie_projects_over_time_and_disables_volume_3d():
    state = AppState()
    state.original_data = np.random.default_rng(0).random((12, 10, 40)).astype(np.float32)
    tab = ProjectionsTab(state)
    tab.on_data_loaded()
    _drain(tab)

    assert tab._stack.currentIndex() == 0  # the 2D image view
    assert tab.view.image is not None and tab.view.image.shape == (12, 10)  # (H, W), reduced over T
    assert not _item(tab, _VOLUME_3D).isEnabled()
    assert _item(tab, "Local correlation").isEnabled()


def test_volumetric_projection_reduces_over_time_and_shows_a_depth_stack():
    state = AppState()
    state.volumetric = True
    state.original_data = (np.random.default_rng(1).random((5, 8, 12, 4)) * 200).astype(np.uint8)  # (T, L, W, D)
    tab = ProjectionsTab(state)
    tab.on_data_loaded()
    _drain(tab)

    assert tab._stack.currentIndex() == 0
    assert tab.view.image.shape == (4, 8, 12)  # (D, L, W): depth-first stack, not smeared over a space axis
    assert not _item(tab, "Local correlation").isEnabled()  # no volumetric form
    assert _item(tab, _VOLUME_3D).isEnabled()


def test_volume_3d_entry_renders_and_click_drag_rotates():
    state = AppState()
    state.volumetric = True
    state.original_data = (np.random.default_rng(2).random((4, 20, 24, 10)) * 255).astype(np.uint8)
    tab = ProjectionsTab(state)
    tab.show()  # prep is deferred until the VolumeView is visible
    tab.on_data_loaded()
    _drain(tab)

    tab.selector.setCurrentText(_VOLUME_3D)
    _drain(tab)

    vv = tab.volume_view
    assert tab._stack.currentWidget() is vv
    assert vv._full is not None
    assert vv._last_scalar is not None  # a frame was rendered

    az0, el0 = vv._az, vv._el
    vv.eventFilter(vv.image, _mouse(QEvent.Type.MouseButtonPress, 100, 100))
    vv.eventFilter(vv.image, _mouse(QEvent.Type.MouseMove, 150, 80))
    vv.eventFilter(vv.image, _mouse(QEvent.Type.MouseButtonRelease, 150, 80))
    for _ in range(10):
        QApplication.processEvents()

    assert vv._az == pytest.approx(az0 + 50 * 0.4)   # horizontal drag -> azimuth
    assert vv._el == pytest.approx(el0 + 20 * 0.4)   # vertical drag (up) -> elevation


def test_switching_back_from_volume_3d_restores_the_2d_view():
    state = AppState()
    state.volumetric = True
    state.original_data = (np.random.default_rng(3).random((4, 10, 12, 5)) * 200).astype(np.uint8)
    tab = ProjectionsTab(state)
    tab.on_data_loaded()
    _drain(tab)

    tab.selector.setCurrentText(_VOLUME_3D)
    _drain(tab)
    assert tab._stack.currentWidget() is tab.volume_view

    tab.selector.setCurrentText("Variance")
    _drain(tab)
    assert tab._stack.currentWidget() is tab.view
    assert tab.view.image.shape == (5, 10, 12)
