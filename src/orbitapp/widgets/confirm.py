"""Shared "this was already done, redo it anyway?" confirmation -- used
wherever a tab detects it's about to repeat identical work (see
StageTab._apply and LoadTab._begin_load).
"""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox, QWidget


def confirm_recompute(parent: QWidget, message: str) -> bool:
    """Yes/No dialog defaulting to No -- skipping unchanged work is the
    whole point of the check calling this, so redoing it should be a
    deliberate override, not the path of least resistance."""
    return (
        QMessageBox.question(
            parent, "Already done", f"{message} Recompute anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        )
        == QMessageBox.StandardButton.Yes
    )
