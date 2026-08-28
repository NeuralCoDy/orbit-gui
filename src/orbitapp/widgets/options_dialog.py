"""Global GUI options, opened via the header bar's gear-icon button --
distinct from any pipeline stage's own Parameters dialog, since these
settings are display/chrome preferences rather than algorithm parameters
and apply app-wide rather than to one tab.

Text size is the first option here; more global (non-pipeline) settings
belong in this same dialog as they're added, rather than each getting
its own button.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout

_MIN_PERCENT = 100  # slider's low end -- the current default text size, 1x
_MAX_PERCENT = 200  # slider's high end -- 2x the default text size


class OptionsDialog(QDialog):
    font_scale_changed = Signal(float)  # e.g. 1.35 for 135% of the base size

    def __init__(self, parent=None, current_scale: float = 1.0) -> None:
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

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        layout.addWidget(self.close_btn)

    def _on_slider_changed(self, value: int) -> None:
        self.font_size_value_label.setText(f"{value}%")
        self.font_scale_changed.emit(value / 100.0)
