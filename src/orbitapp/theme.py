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
"""


_base_font_point_size: int | None = None  # captured once in apply_dark_theme -- see set_font_size_delta


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


def set_font_size_delta(app: QApplication, delta: int) -> None:
    """Sets the app-wide font size to the original base size (captured
    once in apply_dark_theme) plus ``delta`` points -- always relative to
    the true original, not cumulative across repeated calls, so moving
    the Options panel's font-size control back and forth stays exact."""
    base = _base_font_point_size if _base_font_point_size is not None else 10
    font = app.font()
    font.setPointSize(max(1, base + delta))
    app.setFont(font)
