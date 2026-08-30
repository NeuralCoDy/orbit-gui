import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from orbitapp.theme import ACCENT_COLORS  # noqa: E402
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


def test_color_checkboxes_present_one_per_accent_color_in_order():
    dialog = OptionsDialog()
    assert list(dialog._color_checks) == [key for key, _label, _hex in ACCENT_COLORS]
    for key, label, _hex in ACCENT_COLORS:
        assert dialog._color_checks[key].text() == label


def test_color_checkboxes_are_styled_in_their_own_color_not_the_current_accent():
    dialog = OptionsDialog(current_color="#61afef")
    for key, _label, hex_color in ACCENT_COLORS:
        assert hex_color in dialog._color_checks[key].styleSheet()


def test_default_selection_is_t_geco1_blue_when_no_current_color_given():
    dialog = OptionsDialog()
    assert dialog._color_checks["t_geco1_blue"].isChecked()
    assert not dialog._color_checks["gfp_green"].isChecked()


def test_dialog_opens_with_the_given_current_color_checked():
    dialog = OptionsDialog(current_color="#ff5c5c")
    assert dialog._color_checks["rcamp_red"].isChecked()
    assert not dialog._color_checks["t_geco1_blue"].isChecked()


def test_checking_a_color_unchecks_every_other_and_emits_its_hex():
    dialog = OptionsDialog()
    colors = []
    dialog.accent_color_changed.connect(colors.append)

    dialog._color_checks["gfp_green"].setChecked(True)

    assert colors == ["#3ddc71"]
    assert dialog._color_checks["gfp_green"].isChecked()
    assert not dialog._color_checks["t_geco1_blue"].isChecked()
    assert not dialog._color_checks["rcamp_red"].isChecked()
    assert not dialog._color_checks["iglu_snfr_yellow"].isChecked()


def test_switching_colors_only_emits_once_per_switch():
    dialog = OptionsDialog()
    colors = []
    dialog.accent_color_changed.connect(colors.append)

    dialog._color_checks["gfp_green"].setChecked(True)
    dialog._color_checks["rcamp_red"].setChecked(True)

    assert colors == ["#3ddc71", "#ff5c5c"]


def test_unchecking_the_only_active_color_re_checks_it():
    dialog = OptionsDialog()
    colors = []
    dialog.accent_color_changed.connect(colors.append)

    dialog._color_checks["t_geco1_blue"].setChecked(False)

    assert dialog._color_checks["t_geco1_blue"].isChecked()
    assert colors == []  # refusing the uncheck shouldn't emit a spurious change
