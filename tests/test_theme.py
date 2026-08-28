import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from orbitapp import theme  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def test_apply_dark_theme_captures_a_positive_base_font_size(qapp):
    theme.apply_dark_theme(qapp)
    assert theme._base_font_point_size is not None
    assert theme._base_font_point_size > 0


def test_set_font_size_delta_is_relative_to_the_original_base_not_cumulative(qapp):
    theme.apply_dark_theme(qapp)
    base = theme._base_font_point_size

    theme.set_font_size_delta(qapp, 10)
    assert qapp.font().pointSize() == base + 10

    theme.set_font_size_delta(qapp, 3)
    assert qapp.font().pointSize() == base + 3  # not base + 10 + 3

    theme.set_font_size_delta(qapp, 0)
    assert qapp.font().pointSize() == base


def test_set_font_size_delta_never_produces_a_non_positive_size(qapp):
    theme.apply_dark_theme(qapp)
    theme.set_font_size_delta(qapp, -1000)
    assert qapp.font().pointSize() >= 1
    theme.set_font_size_delta(qapp, 0)  # reset for other tests
