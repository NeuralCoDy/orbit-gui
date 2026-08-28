"""Global GUI options, opened via the header bar's gear-icon button --
distinct from any pipeline stage's own Parameters dialog, since these
settings are display/chrome preferences rather than algorithm parameters
and apply app-wide rather than to one tab.

Font size is the first option here; more global (non-pipeline) settings
belong in this same dialog as they're added, rather than each getting
its own button.
"""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from .spinbox import make_spinbox


class OptionsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Options")

        layout = QVBoxLayout(self)

        font_row = QHBoxLayout()
        font_row.addWidget(QLabel("Font size (+points over default):"))
        self.font_size_spin = make_spinbox(0, 10, 0)
        font_row.addWidget(self.font_size_spin)
        layout.addLayout(font_row)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        layout.addWidget(self.close_btn)
