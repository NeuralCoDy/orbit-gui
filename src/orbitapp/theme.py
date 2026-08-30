"""Dark theme: black background, light-blue text and UI chrome --
applied globally via QApplication.setStyleSheet, plus pyqtgraph's own
config (pyqtgraph draws its own canvas and doesn't follow Qt
stylesheets, so its background/foreground need setting separately,
before any pg widget is constructed).
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QWidget

BACKGROUND = "#000000"
PANEL_BACKGROUND = "#0d0d0d"
ACCENT = "#61afef"  # current selection -- mutable at runtime, see set_accent_color; default is "T-GECO1 blue" below
WARNING = "#e5a94e"  # amber -- non-blocking inline warnings (e.g. a poorly-matched method/modality pairing)


def _muted_color(color: str) -> str:
    """Same hue as ``color``, darker and less saturated -- for "grayed
    out" disabled UI (see DISABLED below). Derived rather than a fixed
    independent hex so DISABLED shifts hue family along with ACCENT
    instead of staying a fixed blue-gray regardless of the chosen
    color. The 0.8/0.47 factors were reverse-engineered from this
    theme's original hand-picked pair (ACCENT "#61afef" -> DISABLED
    "#3a5a70") and reproduce it almost exactly (#3b5870)."""
    qcolor = QColor(color)
    hue, saturation, value, alpha = qcolor.getHsvF()
    muted = QColor.fromHsvF(hue, max(0.0, min(1.0, saturation * 0.8)), max(0.0, min(1.0, value * 0.47)), alpha)
    return muted.name()


DISABLED = _muted_color(ACCENT)  # mutable at runtime -- see set_accent_color

# (key, display label, hex) -- the Options panel's text/line color picker
# (widgets/options_dialog.py) shows one color-swatched checkbox per
# entry, in this order; "T-GECO1 blue" is this theme's original default
# (matches the ACCENT value above).
ACCENT_COLORS = (
    ("t_geco1_blue", "T-GECO1 blue", "#61afef"),
    ("gfp_green", "GFP green", "#3ddc71"),
    ("rcamp_red", "RCaMP red", "#ff5c5c"),
    ("iglu_snfr_yellow", "iGluSnFR yellow", "#d9e75c"),
)


def _build_stylesheet() -> str:
    """Rebuilt (rather than a module-level constant) so it always
    reflects the CURRENT ACCENT -- called both by apply_dark_theme and
    by set_accent_color, whenever ACCENT changes."""
    return f"""
QWidget {{
    background-color: {BACKGROUND};
    color: {ACCENT};
    selection-background-color: {ACCENT};
    selection-color: {BACKGROUND};
}}
QMainWindow, QTabWidget::pane {{
    background-color: {BACKGROUND};
    border: 1px solid {ACCENT};
}}
QTabBar::tab {{
    background-color: {PANEL_BACKGROUND};
    color: {ACCENT};
    padding: 6px 12px;
    border: 1px solid {ACCENT};
}}
QTabBar::tab:selected {{
    background-color: {ACCENT};
    color: {BACKGROUND};
}}
QPushButton {{
    background-color: {PANEL_BACKGROUND};
    color: {ACCENT};
    border: 1px solid {ACCENT};
    padding: 4px 10px;
}}
QPushButton:hover:!disabled {{
    background-color: {ACCENT};
    color: {BACKGROUND};
}}
QPushButton:disabled {{
    color: {DISABLED};
    border-color: {DISABLED};
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {{
    background-color: {PANEL_BACKGROUND};
    color: {ACCENT};
    border: 1px solid {ACCENT};
}}
QLabel {{
    color: {ACCENT};
    background-color: transparent;
}}
QSplitter::handle {{
    background-color: {ACCENT};
}}
QScrollBar {{
    background-color: {PANEL_BACKGROUND};
}}
QMenu, QComboBox QAbstractItemView {{
    background-color: {PANEL_BACKGROUND};
    color: {ACCENT};
    selection-background-color: {ACCENT};
    selection-color: {BACKGROUND};
}}
QSlider::groove:horizontal {{
    height: 6px;
    background-color: {PANEL_BACKGROUND};
    border: 1px solid {ACCENT};
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 14px;
    margin: -6px 0;
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
    border-radius: 7px;
}}
QCheckBox {{
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {ACCENT};
    border-radius: 2px;
    background-color: {PANEL_BACKGROUND};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
}}
QCheckBox::indicator:disabled {{
    border-color: {DISABLED};
}}
"""


_base_font_point_size: int | None = None  # captured once in apply_dark_theme -- see set_font_size_scale


def add_legend(plot: pg.PlotWidget, **kwargs) -> pg.LegendItem:
    """plot.addLegend() with this theme's own colors -- pyqtgraph's own
    legend defaults (a near-black, low-contrast box) are easy to miss
    against this app's black background otherwise. Every plot with more
    than one named line should call this instead of addLegend() directly
    (see qc_panel.py, motion_correction_tab.py, roi_review_panel.py,
    normalization_tab.py). Reads ACCENT/PANEL_BACKGROUND at call time
    (not import time), so a plot legended after set_accent_color already
    picks up the current color with no extra work."""
    return plot.addLegend(
        labelTextColor=ACCENT, brush=pg.mkBrush(PANEL_BACKGROUND), pen=pg.mkPen(ACCENT), **kwargs
    )


def apply_dark_theme(app: QApplication) -> None:
    """Applies the dark theme app-wide -- call once, before any widgets
    (especially pyqtgraph ones) are constructed."""
    global _base_font_point_size
    _base_font_point_size = app.font().pointSize()
    if _base_font_point_size <= 0:
        _base_font_point_size = 10  # platform reports pixel size instead of point size -- a sane fallback
    app.setStyleSheet(_build_stylesheet())
    pg.setConfigOption("background", BACKGROUND)
    pg.setConfigOption("foreground", ACCENT)


def _reachable_widgets(app: QApplication, widget_type: type) -> list:
    """Every widget of ``widget_type`` reachable from a CURRENT top-level
    window, rather than app.allWidgets() -- which returns literally
    every QWidget instance the process has ever constructed and not yet
    garbage-collected, including widgets with no live window ancestor
    (e.g. a test's throwaway pg.PlotWidget()). That distinction is
    mostly academic in a normal running GUI session (whose widget count
    stays bounded by what's actually on screen), but it matters a lot
    in a long test suite: confirmed empirically that after enough
    GUI-heavy tests accumulate in one process, app.allWidgets() can
    return tens of thousands of stale objects, making a naive walk over
    it (as set_font_size_scale/set_accent_color used to do) take seconds
    to minutes per call -- topLevelWidgets()+findChildren() instead
    stays proportional to what's actually reachable, regardless of how
    much unreferenced debris a long-lived process has piled up."""
    seen: set[int] = set()
    result = []
    for top in app.topLevelWidgets():
        candidates = list(top.findChildren(widget_type))
        if isinstance(top, widget_type):
            candidates.append(top)
        for widget in candidates:
            if id(widget) not in seen:
                seen.add(id(widget))
                result.append(widget)
    return result


def set_font_size_scale(app: QApplication, scale: float) -> None:
    """Sets the app-wide font size to ``scale`` times the original base
    size (captured once in apply_dark_theme) -- always relative to the
    true original, not cumulative across repeated calls, so moving the
    Options panel's text-size slider back and forth stays exact.

    QApplication.setFont() alone only changes the *default* font new
    widgets pick up -- with this app's QSS stylesheet active, already-
    constructed widgets don't reliably re-derive their own font from a
    later app-level change (confirmed empirically: an existing QLabel's
    font stayed put after setFont() while a freshly-created one picked up
    the new size immediately). Explicitly re-applying to every live
    widget makes an already-open GUI actually update, not just whatever
    gets built after this call.

    A few header elements (the title, pipeline-diagram arrows) have their
    own explicit, deliberately-larger fonts and need their own refresh on
    top of this -- see HeaderBar._on_font_scale_changed, which calls this
    first and then re-derives those from the new base."""
    base = _base_font_point_size if _base_font_point_size is not None else 10
    font = app.font()
    font.setPointSize(max(1, round(base * scale)))
    app.setFont(font)
    for widget in _reachable_widgets(app, QWidget):
        widget.setFont(font)


def _restyle_axis(axis: pg.AxisItem, color: str) -> None:
    """Recolors both the axis line/ticks and the tick-number text --
    AxisItem tracks these as separate pens (setPen alone leaves the
    tick numbers, drawn via a distinct _textPen, at whatever color was
    in effect when the axis was first built)."""
    axis.setPen(color)
    axis.setTextPen(color)


def _restyle_label(label: pg.LabelItem, color: str) -> None:
    """Re-renders a pyqtgraph LabelItem (a plot's title, or one legend
    entry's text) in ``color``. LabelItem bakes its color into a fixed
    HTML string at setText() time rather than tracking it as a live
    pen/property -- confirmed empirically that neither PlotItem.
    titleLabel nor LegendItem's own setLabelTextColor (which only
    updates each item's stored *option*, via LabelItem.setAttr, without
    ever re-rendering the HTML) actually change what's on screen;
    calling setText() again with the new color is what forces the
    re-render."""
    label.setText(label.text, color=color)


def set_accent_color(app: QApplication, color: str) -> None:
    """Changes ACCENT (the app-wide text/border/plot-axis/legend color)
    at runtime and re-applies it to an already-open GUI, not just
    whatever gets built after this call. DISABLED (the "grayed out"
    color used for disabled buttons/checkboxes, e.g. while a stage's
    Apply/Commit is running) is re-derived from the new ACCENT too, so
    it shifts hue family along with everything else instead of staying
    a fixed blue-gray.

    Two different re-application mechanisms are needed, matching the
    same "already-constructed widgets don't reliably pick up a later
    global change" gotcha set_font_size_scale documents:
    - Qt's own QSS-styled chrome (buttons, tabs, checkboxes, labels,
      borders, ...) IS automatically re-polished by a fresh
      app.setStyleSheet() call -- unlike fonts, this part needs no
      per-widget walk.
    - pyqtgraph draws its own canvas rather than following Qt's
      stylesheet, and pg.setConfigOption("foreground", ...) only
      affects pyqtgraph objects created AFTER the call -- already-built
      plots' axis pens/text and any already-added legend's colors need
      updating directly, via _reachable_widgets (see its own docstring
      for why that's used here rather than app.allWidgets()).
      pg.ImageView's own histogram/LUT widget has an AxisItem of its
      own (its intensity-scale ticks) that's easy to miss since it
      isn't reachable through a PlotWidget at all."""
    global ACCENT, DISABLED
    ACCENT = color
    DISABLED = _muted_color(color)
    app.setStyleSheet(_build_stylesheet())
    pg.setConfigOption("foreground", ACCENT)

    for widget in _reachable_widgets(app, pg.PlotWidget):
        plot_item = widget.getPlotItem()
        for axis_name in ("left", "bottom", "right", "top"):
            _restyle_axis(plot_item.getAxis(axis_name), ACCENT)
        if plot_item.titleLabel.text:
            _restyle_label(plot_item.titleLabel, ACCENT)
        legend = plot_item.legend
        if legend is not None:
            legend.setLabelTextColor(ACCENT)  # sets the legend's OWN default for any item added later
            legend.setPen(pg.mkPen(ACCENT))
            legend.setBrush(pg.mkBrush(PANEL_BACKGROUND))
            for _sample, label in legend.items:
                _restyle_label(label, ACCENT)

    for widget in _reachable_widgets(app, pg.ImageView):
        _restyle_axis(widget.getHistogramWidget().item.axis, ACCENT)
