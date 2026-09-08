import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from orbitapp.widgets import VolumeView  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def _volume(t=4, l=16, w=20, d=8):
    return (np.random.default_rng(0).random((t, l, w, d)) * 255).astype(np.uint8)


def _mouse(kind, x, y):
    return QMouseEvent(
        kind, QPointF(x, y), QPointF(x, y),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )


def test_set_volume_is_deferred_until_visible():
    vv = VolumeView()  # never shown
    vv.set_volume(_volume())
    assert vv._full is None  # prep deferred
    assert vv._pending_movie is not None

    vv.show()
    QApplication.processEvents()
    assert vv._full is not None  # showEvent consumed the pending volume
    assert vv._last_scalar is not None
    vv.close()


def test_set_volume_preps_immediately_when_visible():
    vv = VolumeView()
    vv.show()
    QApplication.processEvents()
    vv.set_volume(_volume())
    assert vv._full is not None
    vv.close()


def test_accepts_an_already_3d_volume():
    vv = VolumeView()
    vv.show()
    QApplication.processEvents()
    vv.set_volume(np.random.default_rng(1).random((16, 20, 8)).astype(np.float32))  # (L, W, D)
    assert vv._full is not None
    vv.close()


def test_click_drag_rotates_and_release_renders():
    vv = VolumeView()
    vv.show()
    QApplication.processEvents()
    vv.set_volume(_volume())
    az0, el0 = vv._az, vv._el

    vv.eventFilter(vv.image, _mouse(QEvent.Type.MouseButtonPress, 100, 100))
    vv.eventFilter(vv.image, _mouse(QEvent.Type.MouseMove, 140, 70))
    vv.eventFilter(vv.image, _mouse(QEvent.Type.MouseButtonRelease, 140, 70))

    assert vv._az == pytest.approx(az0 + 40 * 0.4)
    assert vv._el == pytest.approx(el0 + 30 * 0.4)
    vv.close()


def test_mode_switch_disables_transfer_controls_and_rerenders():
    vv = VolumeView()
    vv.show()
    QApplication.processEvents()
    vv.set_volume(_volume())

    vv.mode.setCurrentText("MIP")
    assert not vv.gamma.isEnabled() and not vv.density.isEnabled()
    vv.mode.setCurrentText("emission-absorption")
    assert vv.gamma.isEnabled() and vv.density.isEnabled()
    vv.close()


def test_clear_drops_everything():
    vv = VolumeView()
    vv.show()
    QApplication.processEvents()
    vv.set_volume(_volume())
    vv.clear()
    assert vv._full is None and vv._preview is None and vv._pending_movie is None and vv._last_scalar is None
    vv.close()
