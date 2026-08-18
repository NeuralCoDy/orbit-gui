"""Shared QSpinBox/QDoubleSpinBox construction -- every stage tab with
several numeric parameters (motion correction now; normalization, ...
later) needs the same range/step/value boilerplate repeated per field.
"""

from __future__ import annotations

from PySide6.QtWidgets import QAbstractSpinBox, QDoubleSpinBox, QSpinBox


def make_spinbox(
    minimum: float, maximum: float, value: float, step: float | None = None, decimal: bool = False
) -> QAbstractSpinBox:
    """A QSpinBox (or QDoubleSpinBox if ``decimal``) pre-configured with
    range, optional step, and initial value."""
    box = QDoubleSpinBox() if decimal else QSpinBox()
    box.setRange(minimum, maximum)
    if step is not None:
        box.setSingleStep(step)
    box.setValue(value)
    return box
