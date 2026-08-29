"""Dark theme: black background, light-blue text and UI chrome --
applied globally via QApplication.setStyleSheet, plus pyqtgraph's own
config (pyqtgraph draws its own canvas and doesn't follow Qt
stylesheets, so its background/foreground need setting separately,
before any pg widget is constructed).
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtWidgets import QApplication

BACKGROUND = "#000000"
PANEL_BACKGROUND = "#0d0d0d"
ACCENT = "#61afef"  # the light blue used for highlights/links in this chat's terminal UI
DISABLED = "#3a5a70"
WARNING = "#e5a94e"  # amber -- non-blocking inline warnings (e.g. a poorly-matched method/modality pairing)

_STYLESHEET = f"""
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
    normalization_tab.py)."""
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
    app.setStyleSheet(_STYLESHEET)
    pg.setConfigOption("background", BACKGROUND)
    pg.setConfigOption("foreground", ACCENT)


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
    for widget in app.allWidgets():
        widget.setFont(font)
