import pytest

pytest.importorskip("PySide6")
pg = pytest.importorskip("pyqtgraph")

from PySide6.QtWidgets import QApplication, QLabel, QMainWindow  # noqa: E402

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
    stylesheet = theme._build_stylesheet()
    assert "QCheckBox" in stylesheet
    assert "indicator" in stylesheet
    assert theme.ACCENT in stylesheet


def test_add_legend_uses_this_theme_s_colors_not_pyqtgraph_s_own_defaults(qapp):
    # Regression guard: pyqtgraph's own default legend box/text colors
    # have little contrast against this app's black background.
    plot = pg.PlotWidget()
    legend = theme.add_legend(plot)

    assert legend.opts["labelTextColor"] == pg.mkColor(theme.ACCENT)
    assert legend.opts["pen"] == pg.mkPen(theme.ACCENT)
    assert legend.opts["brush"] == pg.mkBrush(theme.PANEL_BACKGROUND)


@pytest.fixture(autouse=True)
def _restore_accent():
    # theme.ACCENT/DISABLED are genuine mutable global state (see
    # set_accent_color) shared across every test in the whole process --
    # restore them after each test here so a color-changing test can't
    # leak into an unrelated one, in this file or any other.
    original_accent, original_disabled = theme.ACCENT, theme.DISABLED
    yield
    theme.ACCENT = original_accent
    theme.DISABLED = original_disabled


def test_accent_colors_lists_all_four_fluorophore_options_with_t_geco1_blue_as_the_default():
    keys = [key for key, _label, _hex in theme.ACCENT_COLORS]
    assert keys == ["t_geco1_blue", "gfp_green", "rcamp_red", "iglu_snfr_yellow"]
    default_hex = dict((key, hexcolor) for key, _label, hexcolor in theme.ACCENT_COLORS)["t_geco1_blue"]
    assert default_hex == "#61afef" == theme.ACCENT


def test_set_accent_color_updates_the_module_global_and_app_stylesheet(qapp):
    theme.apply_dark_theme(qapp)
    theme.set_accent_color(qapp, "#3ddc71")

    assert theme.ACCENT == "#3ddc71"
    assert "#3ddc71" in qapp.styleSheet()


def test_set_accent_color_updates_pyqtgraph_s_global_foreground_config(qapp):
    theme.apply_dark_theme(qapp)
    theme.set_accent_color(qapp, "#ff5c5c")
    assert pg.getConfigOption("foreground") == "#ff5c5c"


def test_set_accent_color_updates_an_already_constructed_plot_s_axes(qapp):
    # Regression guard: pg.setConfigOption only affects NEWLY created
    # pyqtgraph objects -- an already-open GUI's existing plots need
    # their axis pens updated directly.
    theme.apply_dark_theme(qapp)
    plot = pg.PlotWidget()
    assert plot.getPlotItem().getAxis("bottom").pen().color().name() == theme.ACCENT

    theme.set_accent_color(qapp, "#d9e75c")

    assert plot.getPlotItem().getAxis("bottom").pen().color().name() == "#d9e75c"
    assert plot.getPlotItem().getAxis("left").pen().color().name() == "#d9e75c"


def test_set_accent_color_updates_an_already_added_legend(qapp):
    theme.apply_dark_theme(qapp)
    plot = pg.PlotWidget()
    legend = theme.add_legend(plot)

    theme.set_accent_color(qapp, "#ff5c5c")

    assert legend.opts["labelTextColor"] == pg.mkColor("#ff5c5c")
    assert legend.opts["pen"] == pg.mkPen("#ff5c5c")


def test_set_accent_color_affects_a_plot_created_after_the_change_via_add_legend(qapp):
    theme.apply_dark_theme(qapp)
    theme.set_accent_color(qapp, "#3ddc71")

    plot = pg.PlotWidget()
    legend = theme.add_legend(plot)

    assert legend.opts["labelTextColor"] == pg.mkColor("#3ddc71")


def test_set_accent_color_updates_an_already_constructed_plot_s_tick_number_color(qapp):
    # Regression guard: AxisItem tracks the axis LINE/tick pen and the
    # tick-NUMBER text pen separately -- updating only the former leaves
    # tick labels at whatever color was in effect when the axis was
    # first built.
    theme.apply_dark_theme(qapp)
    plot = pg.PlotWidget()
    axis = plot.getPlotItem().getAxis("left")
    assert axis.textPen().color().name() == theme.ACCENT

    theme.set_accent_color(qapp, "#d9e75c")

    assert axis.textPen().color().name() == "#d9e75c"


def test_set_accent_color_updates_an_already_constructed_image_view_histogram_axis(qapp):
    # Regression guard: pg.ImageView's histogram/LUT widget has its own
    # AxisItem (the intensity-scale ticks), reachable only via
    # getHistogramWidget().item.axis -- not a pg.PlotWidget at all, so
    # it's easy to miss when walking app.allWidgets() for PlotWidgets only.
    import numpy as np

    theme.apply_dark_theme(qapp)
    image_view = pg.ImageView()
    image_view.setImage(np.random.default_rng(0).random((10, 10)))
    axis = image_view.getHistogramWidget().item.axis
    assert axis.textPen().color().name() == theme.ACCENT

    theme.set_accent_color(qapp, "#ff5c5c")

    assert axis.textPen().color().name() == "#ff5c5c"
    assert axis.pen().color().name() == "#ff5c5c"


def test_set_accent_color_re_derives_disabled_from_the_new_accent(qapp):
    # Regression guard: DISABLED ("grayed out" disabled buttons/
    # checkboxes, e.g. while a stage's Apply/Commit is running) used to
    # be a fixed independent hex, so it stayed the same blue-gray
    # regardless of the chosen accent color.
    theme.apply_dark_theme(qapp)
    original_disabled = theme.DISABLED

    theme.set_accent_color(qapp, "#3ddc71")

    assert theme.DISABLED != original_disabled
    assert theme.DISABLED == theme._muted_color("#3ddc71")
    assert theme.DISABLED in qapp.styleSheet()


def test_muted_color_reproduces_the_original_accent_disabled_relationship():
    # This theme's original, hand-picked ACCENT/DISABLED pair -- pins
    # the 0.8/0.47 factors to that known-good relationship.
    assert theme._muted_color("#61afef") == "#3b5870"


def test_set_accent_color_updates_an_already_constructed_plot_s_title(qapp):
    # Regression guard: pyqtgraph's LabelItem (used for both a plot's
    # title and each legend entry -- see the next test) bakes its color
    # into a fixed HTML string at setText() time rather than tracking it
    # as a live, re-appliable property; re-setting pg.setConfigOption
    # alone (or even PlotItem/LegendItem's own color setters) doesn't
    # force the already-rendered HTML to update.
    theme.apply_dark_theme(qapp)
    plot = pg.PlotWidget(title="My Title")
    plot_item = plot.getPlotItem()
    assert theme.ACCENT in plot_item.titleLabel.item.toHtml().lower()

    theme.set_accent_color(qapp, "#ff5c5c")

    assert "ff5c5c" in plot_item.titleLabel.item.toHtml().lower()


def test_set_accent_color_leaves_an_untitled_plot_s_title_hidden(qapp):
    # titleLabel.text is "" when no title was ever set -- must not call
    # setText("", ...) in that case (that would make a previously-
    # hidden, empty title row visibly render/reserve space).
    theme.apply_dark_theme(qapp)
    plot = pg.PlotWidget()  # no title kwarg
    assert not plot.getPlotItem().titleLabel.isVisible()

    theme.set_accent_color(qapp, "#ff5c5c")

    assert not plot.getPlotItem().titleLabel.isVisible()


def test_set_accent_color_updates_an_already_added_legend_s_item_labels(qapp):
    # Regression guard: LegendItem.setLabelTextColor only updates each
    # item's stored *option* dict (via LabelItem.setAttr), never
    # actually re-rendering the item's already-displayed HTML.
    theme.apply_dark_theme(qapp)
    plot = pg.PlotWidget()
    legend = theme.add_legend(plot)
    plot.plot([1, 2, 3], pen="r", name="line a")
    label = legend.items[0][1]
    assert theme.ACCENT in label.item.toHtml().lower()

    theme.set_accent_color(qapp, "#3ddc71")

    assert "3ddc71" in label.item.toHtml().lower()


def test_reachable_widgets_finds_a_standalone_unparented_plot():
    # The existing-test convention (pg.PlotWidget() with no parent) IS
    # itself a top-level widget -- app.topLevelWidgets() alone wouldn't
    # find it as a CHILD of anything, so _reachable_widgets must also
    # check whether each top-level widget itself matches, not just its
    # descendants.
    plot = pg.PlotWidget()
    assert plot in theme._reachable_widgets(QApplication.instance(), pg.PlotWidget)


def test_reachable_widgets_finds_a_plot_nested_in_a_real_window():
    win = QMainWindow()
    plot = pg.PlotWidget()
    win.setCentralWidget(plot)
    assert plot in theme._reachable_widgets(QApplication.instance(), pg.PlotWidget)


def test_reachable_widgets_only_returns_the_requested_type():
    plot = pg.PlotWidget()
    label = QLabel()
    found = theme._reachable_widgets(QApplication.instance(), pg.PlotWidget)
    assert plot in found
    assert label not in found


def test_reachable_widgets_does_not_duplicate_a_widget_reachable_two_ways():
    win = QMainWindow()
    plot = pg.PlotWidget()
    win.setCentralWidget(plot)
    found = theme._reachable_widgets(QApplication.instance(), pg.PlotWidget)
    assert found.count(plot) == 1
