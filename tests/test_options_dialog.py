import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from orbitapp.widgets.options_dialog import OptionsDialog  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def test_font_size_slider_defaults_to_the_low_end_with_a_100_to_200_range():
    dialog = OptionsDialog()
    assert dialog.font_size_slider.value() == 100  # the low end == the current default size (1x)
    assert dialog.font_size_slider.minimum() == 100
    assert dialog.font_size_slider.maximum() == 200  # the high end == 2x the default size
    assert dialog.font_size_value_label.text() == "100%"


def test_font_size_slider_starts_at_a_given_current_scale():
    dialog = OptionsDialog(current_scale=1.5)
    assert dialog.font_size_slider.value() == 150
    assert dialog.font_size_value_label.text() == "150%"


def test_moving_the_slider_emits_font_scale_changed_and_updates_the_label():
    dialog = OptionsDialog()
    scales = []
    dialog.font_scale_changed.connect(scales.append)

    dialog.font_size_slider.setValue(175)

    assert scales == [1.75]
    assert dialog.font_size_value_label.text() == "175%"


def test_close_button_accepts_the_dialog():
    dialog = OptionsDialog()
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(1))

    dialog.close_btn.click()

    assert accepted == [1]
