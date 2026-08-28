import pytest

pytest.importorskip("PySide6")
pg = pytest.importorskip("pyqtgraph")

from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from orbitapp import theme  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def test_apply_dark_theme_captures_a_positive_base_font_size(qapp):
    theme.apply_dark_theme(qapp)
    assert theme._base_font_point_size is not None
    assert theme._base_font_point_size > 0


def test_set_font_size_scale_is_relative_to_the_original_base_not_cumulative(qapp):
    theme.apply_dark_theme(qapp)
    base = theme._base_font_point_size

    theme.set_font_size_scale(qapp, 2.0)
    assert qapp.font().pointSize() == base * 2

    theme.set_font_size_scale(qapp, 1.5)
    assert qapp.font().pointSize() == round(base * 1.5)  # not (base * 2) * 1.5

    theme.set_font_size_scale(qapp, 1.0)
    assert qapp.font().pointSize() == base


def test_set_font_size_scale_never_produces_a_non_positive_size(qapp):
    theme.apply_dark_theme(qapp)
    theme.set_font_size_scale(qapp, 0.0)
    assert qapp.font().pointSize() >= 1
    theme.set_font_size_scale(qapp, 1.0)  # reset for other tests


def test_set_font_size_scale_updates_already_constructed_widgets(qapp):
    # Regression guard: QApplication.setFont() alone only changes the
    # default new widgets pick up -- with this app's QSS stylesheet
    # active, an already-open GUI's existing widgets don't reliably
    # re-derive their font from that alone (confirmed empirically before
    # this fix: an existing QLabel's font stayed put while a freshly
    # created one picked up the new size).
    theme.apply_dark_theme(qapp)
    base = theme._base_font_point_size
    label = QLabel("hello")
    assert label.font().pointSize() == base

    theme.set_font_size_scale(qapp, 2.0)

    assert label.font().pointSize() == base * 2
    theme.set_font_size_scale(qapp, 1.0)  # reset for other tests


def test_apply_dark_theme_styles_checkbox_indicators_as_a_visible_blue_square():
    # Regression guard: QCheckBox had no stylesheet rules at all, so its
    # indicator box had no defined colors -- easy to miss against the
    # black background.
    assert "QCheckBox" in theme._STYLESHEET
    assert "indicator" in theme._STYLESHEET
    assert theme.ACCENT in theme._STYLESHEET


def test_add_legend_uses_this_theme_s_colors_not_pyqtgraph_s_own_defaults(qapp):
    # Regression guard: pyqtgraph's own default legend box/text colors
    # have little contrast against this app's black background.
    plot = pg.PlotWidget()
    legend = theme.add_legend(plot)

    assert legend.opts["labelTextColor"] == pg.mkColor(theme.ACCENT)
    assert legend.opts["pen"] == pg.mkPen(theme.ACCENT)
    assert legend.opts["brush"] == pg.mkBrush(theme.PANEL_BACKGROUND)
