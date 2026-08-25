import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from orbitapp.volumetric_io import INTERLEAVED, ONE_VOLUME_PER_STACK  # noqa: E402
from orbitapp.widgets.volumetric_load_dialog import VolumetricLoadDialog  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def test_defaults_to_one_volume_per_stack_with_no_depth():
    dialog = VolumetricLoadDialog()
    assert dialog.mode() == ONE_VOLUME_PER_STACK
    assert dialog.depth() is None
    assert not dialog.depth_spin.isEnabled()


def test_selecting_interleaved_enables_depth_and_reports_it():
    dialog = VolumetricLoadDialog()
    dialog.interleaved_radio.setChecked(True)
    dialog.depth_spin.setValue(12)

    assert dialog.mode() == INTERLEAVED
    assert dialog.depth() == 12
    assert dialog.depth_spin.isEnabled()


def test_switching_back_to_one_volume_per_stack_drops_depth():
    dialog = VolumetricLoadDialog()
    dialog.interleaved_radio.setChecked(True)
    dialog.depth_spin.setValue(5)
    dialog.one_per_stack_radio.setChecked(True)

    assert dialog.mode() == ONE_VOLUME_PER_STACK
    assert dialog.depth() is None
    assert not dialog.depth_spin.isEnabled()
