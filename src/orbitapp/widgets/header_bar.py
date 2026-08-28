"""Persistent header strip shown above the tabs on every stage: what's
loaded and the committed processing pipeline so far -- labeled "Current
pipeline:" (or "Current pipeline (widefield):" etc. if any of the Load
tab's data-modality toggles are on -- see set_pipeline's modifiers) --
as a block diagram (one box per step, connected by arrows, e.g.
"[Load] -> [Patch Warp] -> [Normalize]"), plus the full orbit logo in
the top-right corner -- all stay visible no matter which tab is open
(unlike pyGraFT's per-tab-only status labels). Two rows: the "ORBIT
GUI" title sits above the data-loaded status text on the left; the
pipeline diagram shares the row directly under that with the logo, at
the same vertical level.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..assets import LOGO_PATH, load_logo
from ..theme import ACCENT, BACKGROUND, set_font_size_delta
from .options_dialog import OptionsDialog

_LOGO_HEIGHT_PX = 60

# Same colors as every other bordered widget in the app's dark theme
# (theme.py's own QPushButton/QTabBar/QMainWindow rules all use
# "border: 1px solid {ACCENT}" against a black background) -- not
# palette(...) roles, since this app's dark theme is a QSS stylesheet
# override rather than an actual QPalette change, so palette(...)
# would resolve to the default (unthemed) system colors instead.
_BOX_STYLE = (
    f"QLabel {{ border: 1px solid {ACCENT}; border-radius: 4px; "
    f"padding: 2px 10px; background-color: {BACKGROUND}; }}"
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
            box.setStyleSheet(_BOX_STYLE)
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

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 4, 8, 4)
        outer.setSpacing(2)

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

        title_label = QLabel("ORBIT GUI")
        title_font = title_label.font()
        title_font.setBold(True)
        title_font.setPointSize(title_font.pointSize() + 2)
        title_label.setFont(title_font)
        top_row.addWidget(title_label, 0, Qt.AlignmentFlag.AlignVCenter)
        outer.addLayout(top_row)

        pipeline_row = QHBoxLayout()
        self.pipeline_caption_label = QLabel("Current pipeline:")
        pipeline_row.addWidget(self.pipeline_caption_label, 0, Qt.AlignmentFlag.AlignVCenter)
        self.pipeline_diagram = _PipelineDiagram()
        pipeline_row.addWidget(self.pipeline_diagram, 0, Qt.AlignmentFlag.AlignVCenter)
        pipeline_row.addStretch()

        self.options_btn = QPushButton("⚙")  # gear icon
        self.options_btn.setToolTip("Options...")
        self.options_btn.setFixedWidth(32)
        self.options_btn.clicked.connect(self._on_options_clicked)
        pipeline_row.addWidget(self.options_btn, 0, Qt.AlignmentFlag.AlignVCenter)

        pipeline_row.addWidget(self._make_logo_label(), 0, Qt.AlignmentFlag.AlignVCenter)
        outer.addLayout(pipeline_row)

    def _make_logo_label(self) -> QLabel:
        logo_label = QLabel()
        if LOGO_PATH.exists():
            logo_label.setPixmap(load_logo(_LOGO_HEIGHT_PX))
        return logo_label

    def _on_options_clicked(self) -> None:
        dialog = OptionsDialog(self)
        dialog.font_size_spin.valueChanged.connect(self._on_font_size_changed)
        dialog.exec()

    def _on_font_size_changed(self, delta: int) -> None:
        app = QApplication.instance()
        if app is not None:
            set_font_size_delta(app, delta)

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
