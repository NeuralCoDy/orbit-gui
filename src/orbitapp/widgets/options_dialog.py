"""Global GUI options, opened via the header bar's gear-icon button --
distinct from any pipeline stage's own Parameters dialog, since these
settings are display/chrome preferences rather than algorithm parameters
and apply app-wide rather than to one tab.

Text size was the first option here; text/line color is the second --
more global (non-pipeline) settings belong in this same dialog as
they're added, rather than each getting its own button.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QDialog, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout

from ..theme import ACCENT_COLORS

_MIN_PERCENT = 100  # slider's low end -- the current default text size, 1x
_MAX_PERCENT = 200  # slider's high end -- 2x the default text size

_COLORS_BY_KEY = {key: hex_color for key, _label, hex_color in ACCENT_COLORS}


class OptionsDialog(QDialog):
    font_scale_changed = Signal(float)  # e.g. 1.35 for 135% of the base size
    accent_color_changed = Signal(str)  # a hex color from theme.ACCENT_COLORS

    def __init__(self, parent=None, current_scale: float = 1.0, current_color: str | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Options")
        self.setMinimumWidth(360)  # the default dialog size left too little travel to see/grab the slider

        layout = QVBoxLayout(self)

        font_row = QHBoxLayout()
        font_row.addWidget(QLabel("Text size:"))
        self.font_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.font_size_slider.setRange(_MIN_PERCENT, _MAX_PERCENT)
        self.font_size_slider.setValue(round(current_scale * 100))
        self.font_size_slider.setMinimumWidth(220)
        self.font_size_slider.valueChanged.connect(self._on_slider_changed)
        font_row.addWidget(self.font_size_slider)
        self.font_size_value_label = QLabel(f"{self.font_size_slider.value()}%")
        font_row.addWidget(self.font_size_value_label)
        layout.addLayout(font_row)

        layout.addWidget(QLabel("Text/line color:"))
        # One checkbox per color, shown IN its own color (not the current
        # accent) so each option reads at a glance -- checking one
        # unchecks the others (see _on_color_toggled), a single-select
        # group implemented with checkboxes rather than radio buttons per
        # explicit request.
        self._color_checks: dict[str, QCheckBox] = {}
        for key, label, hex_color in ACCENT_COLORS:
            check = QCheckBox(label)
            check.setStyleSheet(f"QCheckBox {{ color: {hex_color}; font-weight: bold; }}")
            check.setChecked(hex_color == current_color)
            check.toggled.connect(lambda checked, key=key: self._on_color_toggled(key, checked))
            layout.addWidget(check)
            self._color_checks[key] = check
        if not any(check.isChecked() for check in self._color_checks.values()):
            # current_color didn't match any known entry (or wasn't given) -- default to the first (T-GECO1 blue).
            first_key = ACCENT_COLORS[0][0]
            self._color_checks[first_key].setChecked(True)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        layout.addWidget(self.close_btn)

    def _on_slider_changed(self, value: int) -> None:
        self.font_size_value_label.setText(f"{value}%")
        self.font_scale_changed.emit(value / 100.0)

    def _on_color_toggled(self, key: str, checked: bool) -> None:
        if not checked:
            # Not a single-select via QButtonGroup (explicit request was
            # checkboxes) -- enforce "exactly one always checked" by hand:
            # refuse to leave every box unchecked by re-checking this one
            # if it was the last one standing.
            if not any(check.isChecked() for check in self._color_checks.values()):
                self._color_checks[key].blockSignals(True)
                self._color_checks[key].setChecked(True)
                self._color_checks[key].blockSignals(False)
            return

        for other_key, check in self._color_checks.items():
            if other_key != key:
                check.blockSignals(True)
                check.setChecked(False)
                check.blockSignals(False)
        self.accent_color_changed.emit(_COLORS_BY_KEY[key])
