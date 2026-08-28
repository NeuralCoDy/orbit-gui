import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from orbitapp.widgets.options_dialog import OptionsDialog  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def test_font_size_spin_defaults_to_zero_with_a_zero_to_ten_range():
    dialog = OptionsDialog()
    assert dialog.font_size_spin.value() == 0
    assert dialog.font_size_spin.minimum() == 0
    assert dialog.font_size_spin.maximum() == 10


def test_close_button_accepts_the_dialog():
    dialog = OptionsDialog()
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(1))

    dialog.close_btn.click()

    assert accepted == [1]
