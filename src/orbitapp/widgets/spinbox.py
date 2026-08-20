"""Shared QSpinBox/QDoubleSpinBox construction -- every stage tab with
several numeric parameters needs the same range/step/value boilerplate
repeated per field.
"""

from __future__ import annotations

from PySide6.QtWidgets import QAbstractSpinBox, QDoubleSpinBox, QSpinBox


def make_spinbox(
    minimum: float, maximum: float, value: float, step: float | None = None, decimal: bool = False,
    decimals: int | None = None,
) -> QAbstractSpinBox:
    """A QSpinBox (or QDoubleSpinBox if ``decimal``) pre-configured with
    range, optional step, and initial value. ``decimals`` (only
    meaningful with ``decimal=True``) overrides Qt's own default of 2 --
    needed for a field whose meaningful values are much smaller than
    that, e.g. a convergence tolerance around 1e-4."""
    box = QDoubleSpinBox() if decimal else QSpinBox()
    box.setRange(minimum, maximum)
    if step is not None:
        box.setSingleStep(step)
    if decimals is not None:
        box.setDecimals(decimals)
    box.setValue(value)
    return box
