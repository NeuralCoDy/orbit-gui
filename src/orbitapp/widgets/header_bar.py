"""Persistent header strip shown above the tabs on every stage: what's
loaded and the committed processing pipeline so far -- labeled "Current
pipeline:" (or "Current pipeline (widefield):" etc. if any of the Load
tab's data-modality toggles are on -- see set_pipeline's modifiers) --
as a block diagram (one box per step, connected by arrows, e.g.
"[Load] -> [Patch Warp] -> [Normalize]") -- all stay visible no matter
which tab is open (unlike pyGraFT's per-tab-only status labels).

Two columns: everything text-based (Generate Report button, data-loaded
status, title, pipeline caption+diagram) stacks in its own two-row
column on the left, same relative positions as before; the options
(gear) button and the orbit logo share a row of their own on the
right, both vertically centered (gear first, roughly half the logo's
height; the logo itself sized to the header's own height -- ~60px by
default). Both are rescaled on every resize (see
resizeEvent/_refresh_logo_pixmap) so they track the header's actual
height -- driven by the text column -- rather than a fixed pixel size
baked in up front.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .. import theme
from ..assets import LOGO_PATH, load_logo
from ..theme import set_accent_color, set_font_size_scale
from .options_dialog import OptionsDialog


def _box_style() -> str:
    """Rebuilt (rather than a module-level constant) so it always
    reflects the CURRENT theme.ACCENT -- a plain ``from ..theme import
    ACCENT`` would instead bind the value once at import time, so a
    later theme.set_accent_color() call wouldn't reach it. Same colors
    as every other bordered widget in the app's dark theme (theme.py's
    own QPushButton/QTabBar/QMainWindow rules all use "border: 1px
    solid {ACCENT}" against a black background) -- not palette(...)
    roles, since this app's dark theme is a QSS stylesheet override
    rather than an actual QPalette change, so palette(...) would
    resolve to the default (unthemed) system colors instead."""
    return (
        f"QLabel {{ border: 1px solid {theme.ACCENT}; border-radius: 4px; "
        f"padding: 2px 10px; background-color: {theme.BACKGROUND}; }}"
    )


class _PipelineDiagram(QWidget):
    """A horizontal row of boxes, one per committed pipeline step, each
    connected to the next by an arrow. Rebuilt from scratch on every
    set_steps() call (matching how the plain-text breadcrumb this
    replaces was always given the FULL step list, never appended to)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._boxes: list[QLabel] = []

    def set_steps(self, steps: list[str]) -> None:
        while self._row.count():
            item = self._row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._boxes = []

        for i, step in enumerate(steps):
            if i > 0:
                self._row.addWidget(self._make_arrow())
            box = QLabel(step)
            box.setStyleSheet(_box_style())
            self._row.addWidget(box)
            self._boxes.append(box)
        self._row.addStretch()

    @staticmethod
    def _make_arrow() -> QLabel:
        """Bigger and bolder than plain body text -- a step transition
        should read clearly at a glance, not blend into the boxes on
        either side of it."""
        arrow = QLabel("→")
        font = arrow.font()
        font.setPointSize(font.pointSize() + 6)
        font.setBold(True)
        arrow.setFont(font)
        return arrow

    def step_labels(self) -> list[str]:
        """Ordered step text, one per box -- for tests and anything
        else that wants the steps back out without reaching into layout
        internals."""
        return [box.text() for box in self._boxes]


class HeaderBar(QWidget):
    generate_report_clicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._font_scale = 1.0  # 1.0 == the app's base text size -- see _on_font_scale_changed
        self._accent_color = theme.ACCENT  # see _on_accent_color_changed

        # Two columns: all text on the left (same two-row arrangement as
        # before, with its own vertical padding), gear+logo in a row of
        # their own on the right. Zero vertical margin here (the text
        # column supplies its own instead) so the logo can be stretched
        # to exactly fill the header's height -- see resizeEvent.
        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 0, 8, 0)
        outer.setSpacing(8)

        self._text_col = QVBoxLayout()
        text_col = self._text_col
        text_col.setContentsMargins(0, 4, 0, 4)
        text_col.setSpacing(2)
        outer.addLayout(text_col, 1)

        top_row = QHBoxLayout()

        # Report button + data-loaded status share the far-left column,
        # button above status, so "Generate Report..." sits at the very
        # top-left corner of the whole header.
        left_col = QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(0)

        self.report_btn = QPushButton("Generate Report...")
        self.report_btn.setToolTip(
            "Writes a LaTeX report of the committed pipeline (steps, equations, parameters, "
            "validation metrics) and compiles it to PDF."
        )
        self.report_btn.clicked.connect(self.generate_report_clicked)
        left_col.addWidget(self.report_btn, 0, Qt.AlignmentFlag.AlignLeft)

        self.data_label = QLabel("No data loaded.")
        left_col.addWidget(self.data_label, 0, Qt.AlignmentFlag.AlignLeft)

        top_row.addLayout(left_col)
        top_row.addStretch()

        self.title_label = QLabel("ORBIT GUI")
        self._style_title_label()
        top_row.addWidget(self.title_label, 0, Qt.AlignmentFlag.AlignVCenter)
        text_col.addLayout(top_row)

        pipeline_row = QHBoxLayout()
        self.pipeline_caption_label = QLabel("Current pipeline:")
        pipeline_row.addWidget(self.pipeline_caption_label, 0, Qt.AlignmentFlag.AlignVCenter)
        self.pipeline_diagram = _PipelineDiagram()
        pipeline_row.addWidget(self.pipeline_diagram, 0, Qt.AlignmentFlag.AlignVCenter)
        pipeline_row.addStretch()
        text_col.addLayout(pipeline_row)

        # Gear button + logo share a row of their own, to the right of
        # all the text: gear first (roughly half the logo's height,
        # vertically centered), then the logo (stretched to fill the
        # row's full height -- see resizeEvent/_refresh_logo_pixmap).
        side_row = QHBoxLayout()
        side_row.setContentsMargins(0, 0, 0, 0)

        self.options_btn = QPushButton("⚙")  # gear icon
        self.options_btn.setToolTip("Options...")
        self.options_btn.clicked.connect(self._on_options_clicked)
        side_row.addWidget(self.options_btn, 0, Qt.AlignmentFlag.AlignVCenter)

        self.logo_label = QLabel()
        side_row.addWidget(self.logo_label, 0, Qt.AlignmentFlag.AlignVCenter)

        outer.addLayout(side_row)
        self._refresh_logo_pixmap()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_logo_pixmap()

    def _refresh_logo_pixmap(self) -> None:
        """Re-renders the logo at the text column's natural height
        (preserving its aspect ratio -- see assets.load_logo), and the
        gear button at half of that -- called on every resize so both
        track the header's actual height as driven by the text content.

        Deliberately measures ``self._text_col.sizeHint()`` rather than
        ``self.height()``: the header's own height is itself partly
        determined by the logo's size (both share the same outer
        row), so sizing the logo off ``self.height()`` closes a
        feedback loop -- each resize pass re-requests a pixmap as big
        as the *previous* pass's (now-larger) header, snowballing the
        logo (and the whole header) far past the text column's actual
        size. The text column's sizeHint has no such dependency on the
        logo, so it's a stable target."""
        if not LOGO_PATH.exists():
            return
        logo_height = max(self._text_col.sizeHint().height(), 1)
        self.logo_label.setPixmap(load_logo(logo_height))
        gear_size = max(logo_height // 2, 1)
        self.options_btn.setFixedSize(gear_size, gear_size)

    def _style_title_label(self) -> None:
        """(Re-)derives the title's font from the app's current default
        -- called both at construction and whenever the text-size slider
        changes, since a widget's own explicit setFont() call (needed
        here for the permanent bold/+2 bump) stops it from automatically
        following later QApplication.setFont() changes."""
        app = QApplication.instance()
        base_font = app.font() if app is not None else self.title_label.font()
        title_font = QFont(base_font)
        title_font.setBold(True)
        title_font.setPointSize(title_font.pointSize() + 2)
        self.title_label.setFont(title_font)

    def _on_options_clicked(self) -> None:
        dialog = OptionsDialog(self, current_scale=self._font_scale, current_color=self._accent_color)
        dialog.font_scale_changed.connect(self._on_font_scale_changed)
        dialog.accent_color_changed.connect(self._on_accent_color_changed)
        dialog.exec()

    def _on_font_scale_changed(self, scale: float) -> None:
        self._font_scale = scale
        app = QApplication.instance()
        if app is None:
            return
        set_font_size_scale(app, scale)
        # The app-wide font change above doesn't reach widgets with their
        # own explicit font (title, pipeline-diagram arrows) -- refresh
        # those specifically so ALL text scales, not just the majority
        # that inherits the app default automatically.
        self._style_title_label()
        self.pipeline_diagram.set_steps(self.pipeline_diagram.step_labels())
        # The text column just grew/shrank -- resize the logo/gear to match
        # right away rather than waiting for the next resizeEvent.
        self._refresh_logo_pixmap()

    def _on_accent_color_changed(self, color: str) -> None:
        self._accent_color = color
        app = QApplication.instance()
        if app is None:
            return
        set_accent_color(app, color)
        # The pipeline-diagram boxes bake _box_style() into each QLabel's
        # own stylesheet at construction time -- refresh them explicitly,
        # same reasoning as _on_font_scale_changed's title/arrow refresh.
        self.pipeline_diagram.set_steps(self.pipeline_diagram.step_labels())

    def set_data_info(self, summary: str | None) -> None:
        self.data_label.setText(summary or "No data loaded.")

    def set_report_busy(self, busy: bool) -> None:
        """Disables and relabels the button while a report is being
        generated -- the button's own state doubles as the busy
        indicator, since this compact header has no BusyBar of its own."""
        self.report_btn.setEnabled(not busy)
        self.report_btn.setText("Generating..." if busy else "Generate Report...")

    def set_pipeline(self, steps: list[str], modifiers: list[str] | None = None) -> None:
        """``modifiers`` are the Load tab's active data-modality toggles
        (see AppState.modality_modifiers), shown parenthetically in the
        caption, e.g. "Current pipeline (widefield):" -- purely a label
        for now."""
        suffix = f" ({', '.join(modifiers)})" if modifiers else ""
        self.pipeline_caption_label.setText(f"Current pipeline{suffix}:")
        self.pipeline_diagram.set_steps(steps)
