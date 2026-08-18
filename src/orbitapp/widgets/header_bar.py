"""Persistent header strip shown above the tabs on every stage: what's
loaded, which stage tab is active, the committed processing pipeline so
far (e.g. "Load > Patch Warp > Normalize"), and the full orbit logo in
the top-right corner -- all stay visible no matter which tab is open
(unlike pyGraFT's per-tab-only status labels).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ..assets import LOGO_PATH, load_logo

_LOGO_HEIGHT_PX = 60


class HeaderBar(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 4, 8, 4)
        outer.setSpacing(2)

        top_row = QHBoxLayout()
        self.data_label = QLabel("No data loaded.")
        self.stage_label = QLabel("")
        top_row.addWidget(self.data_label, 0, Qt.AlignmentFlag.AlignVCenter)
        top_row.addStretch()
        top_row.addWidget(self.stage_label, 0, Qt.AlignmentFlag.AlignVCenter)
        top_row.addWidget(self._make_logo_label())
        outer.addLayout(top_row)

        self.pipeline_label = QLabel("")
        outer.addWidget(self.pipeline_label)

    def _make_logo_label(self) -> QLabel:
        logo_label = QLabel()
        if LOGO_PATH.exists():
            logo_label.setPixmap(load_logo(_LOGO_HEIGHT_PX))
        return logo_label

    def set_data_info(self, summary: str | None) -> None:
        self.data_label.setText(summary or "No data loaded.")

    def set_active_stage(self, name: str) -> None:
        self.stage_label.setText(f"Stage: {name}")

    def set_pipeline(self, steps: list[str]) -> None:
        self.pipeline_label.setText("Pipeline: " + " > ".join(steps) if steps else "")
