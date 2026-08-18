"""Reusable Apply/Commit button pair for any stage that transforms the
movie (motion correction now; normalization, demixing, ... later).
Running a stage ("Apply") never touches the shared active dataset by
itself -- it only produces a candidate result for that tab to preview.
The change only becomes part of the pipeline once the user explicitly
clicks "Commit", horizontally aligned next to Apply. This makes every
mutation of the working data an explicit, visible user action rather
than an automatic side effect of clicking Apply.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget


class CommitControls(QWidget):
    apply_clicked = Signal()
    commit_clicked = Signal()

    def __init__(self, apply_label: str, commit_label: str = "Commit to Active Dataset", parent=None) -> None:
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.apply_btn = QPushButton(apply_label)
        self.commit_btn = QPushButton(commit_label)
        self.commit_btn.setEnabled(False)

        layout.addWidget(self.apply_btn)
        layout.addWidget(self.commit_btn)

        self.apply_btn.clicked.connect(self.apply_clicked)
        self.commit_btn.clicked.connect(self.commit_clicked)

    def set_apply_enabled(self, enabled: bool) -> None:
        self.apply_btn.setEnabled(enabled)

    def set_commit_enabled(self, enabled: bool) -> None:
        self.commit_btn.setEnabled(enabled)
