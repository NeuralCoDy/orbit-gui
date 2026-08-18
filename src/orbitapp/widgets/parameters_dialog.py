"""Reusable "Parameters..." popup: a small modal dialog wrapping a
QFormLayout, opened from a single button rather than sprawling a stage
tab's full parameter set across the main panel. Any stage tab with
per-algorithm parameters (motion correction now; normalization, ...
later) should build one of these instead of embedding the form directly.
"""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QFormLayout, QPushButton, QVBoxLayout, QWidget


class ParametersDialog(QDialog):
    def __init__(self, title: str = "Parameters", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)

        outer = QVBoxLayout(self)
        self.form = QFormLayout()
        outer.addLayout(self.form)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        outer.addWidget(close_btn)

    def add_row(self, label: str, widget: QWidget) -> None:
        self.form.addRow(label, widget)
