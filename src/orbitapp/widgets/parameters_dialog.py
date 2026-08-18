"""Reusable "Parameters..." popup: a small modal dialog wrapping a
QFormLayout, opened from a single button rather than sprawling a stage
tab's full parameter set across the main panel. Any stage tab with
per-algorithm parameters (motion correction, denoising, ...) should
build one of these instead of embedding the form directly.

Rows can be tagged with a ``group`` (e.g. one per algorithm a method
dropdown can select) and shown/hidden together via set_group_visible --
so a tab with several algorithms, each with its own irrelevant-to-the-
others parameters, only shows the ones that currently apply.
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
        self._rows_by_group: dict[str, list[QWidget]] = {}

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        outer.addWidget(close_btn)

    def add_row(self, label: str, widget: QWidget, group: str | None = None) -> None:
        self.form.addRow(label, widget)
        if group is not None:
            self._rows_by_group.setdefault(group, []).append(widget)

    def set_group_visible(self, group: str, visible: bool) -> None:
        """Shows/hides every row previously added with ``group=group``."""
        for widget in self._rows_by_group.get(group, []):
            self.form.setRowVisible(widget, visible)

    def show_only_group(self, group: str) -> None:
        """Shows ``group``'s rows and hides every other known group's --
        the common case of "one algorithm selected, show just its params"."""
        for other in self._rows_by_group:
            self.set_group_visible(other, other == group)
