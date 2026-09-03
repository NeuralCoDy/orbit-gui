import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from orbitapp.widgets import StagePanel  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def test_set_volumetric_swaps_both_columns_and_hides_play_buttons():
    panel = StagePanel()
    panel.show()
    QApplication.processEvents()

    for key in ("before", "after"):
        assert panel._stacks[key].currentIndex() == 0  # ImageView by default
        assert panel._button_containers[key].isVisible()

    panel.set_volumetric(True)
    for key in ("before", "after"):
        assert panel._stacks[key].currentWidget() is panel._stacks[key].widget(1)  # VolumeView
        assert not panel._button_containers[key].isVisible()

    panel.set_volumetric(False)
    for key in ("before", "after"):
        assert panel._stacks[key].currentIndex() == 0
        assert panel._button_containers[key].isVisible()
    panel.close()


def test_before_and_after_volume_route_to_the_volume_views():
    panel = StagePanel()
    panel.set_volumetric(True)
    panel.show()
    QApplication.processEvents()

    movie = (np.random.default_rng(0).random((3, 12, 14, 6)) * 255).astype(np.uint8)
    panel.set_before_volume(movie)
    panel.set_after_volume(movie)
    assert panel.before_volume._full is not None
    assert panel.after_volume._full is not None

    panel.set_after_volume(None)
    assert panel.after_volume._full is None
    panel.close()


def test_leaving_volumetric_mode_clears_the_volume_views():
    panel = StagePanel()
    panel.set_volumetric(True)
    panel.show()
    QApplication.processEvents()
    panel.set_before_volume((np.random.default_rng(1).random((3, 10, 10, 4)) * 255).astype(np.uint8))
    assert panel.before_volume._full is not None

    panel.set_volumetric(False)
    assert panel.before_volume._full is None
    panel.close()
