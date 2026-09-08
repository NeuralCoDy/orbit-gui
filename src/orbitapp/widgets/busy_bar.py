"""Reusable "this is running" indicator: a status message plus a
progress bar, both hidden when idle. None of orbit's algorithms report
a real fractional progress, so this fakes one from elapsed time -- the
bar eases toward (but never quite reaches) a ceiling while running,
snapping to full and hiding once the operation actually finishes. A
smoothly filling bar reads as "working" far better than a bouncing
indeterminate one for anything that runs tens of seconds.

Every tab kicking off a FunctionWorker for anything that can take more
than a couple of seconds (loading, computing a projection, motion
correction, ...) should show one of these rather than just freezing.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QWidget

_TICK_MS = 150
_CEILING_PCT = 92  # asymptote while running; stop() jumps straight to 100
_TIME_CONSTANT_S = 6.0  # higher = slower approach to the ceiling


class BusyBar(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel("")
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setMaximumWidth(200)
        self.bar.setVisible(False)

        layout.addWidget(self.label, stretch=1)
        layout.addWidget(self.bar)

        self._elapsed_ms = 0
        self._base_message = ""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def start(self, message: str) -> None:
        self.label.setText(message)
        self.bar.setValue(0)
        self.bar.setVisible(True)
        self._elapsed_ms = 0
        self._base_message = message
        self._timer.start(_TICK_MS)

    def set_message(self, message: str) -> None:
        """Updates the status text only, leaving the progress bar's
        visibility/animation running -- for a multi-phase operation that
        should read as one continuous wait rather than restarting."""
        self.label.setText(message)
        self._base_message = message

    def set_progress(self, done: int, total: int) -> None:
        """Switches the bar from the fake elapsed-time animation to a
        real fraction, for callers that can actually count units of
        work (e.g. "N of M files loaded") -- stops the timer so it
        can't keep overwriting a real value with a guessed one."""
        if total <= 0:
            return
        self._timer.stop()
        self.bar.setValue(min(int(100 * done / total), 100))
        self.label.setText(f"{self._base_message} ({done}/{total})")

    def _tick(self) -> None:
        self._elapsed_ms += _TICK_MS
        elapsed_s = self._elapsed_ms / 1000.0
        progress = _CEILING_PCT * (1.0 - math.exp(-elapsed_s / _TIME_CONSTANT_S))
        self.bar.setValue(int(progress))

    def stop(self, message: str = "") -> None:
        self._timer.stop()
        self.bar.setValue(100)
        self.bar.setVisible(False)
        self.label.setText(message)
