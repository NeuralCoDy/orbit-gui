"""Background execution of long-running orbit operations, ported from
pyGraFT's graftapp/workers.py. Qt widgets must not block the event loop,
so anything that takes more than a beat (loading, a projection, motion
correction, ...) runs on a QThread and reports back via signals.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

from PySide6.QtCore import QThread, Signal

from .widgets import BusyBar


class FunctionWorker(QThread):
    """Runs an arbitrary callable off the GUI thread."""

    finished_ok = Signal(object)
    failed = Signal(str)
    warning = Signal(str)

    def __init__(self, fn: Callable, *args: Any, parent=None, **kwargs: Any) -> None:
        super().__init__(parent)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = self.fn(*self.args, **self.kwargs)
            for w in caught:
                self.warning.emit(str(w.message))
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def run_worker(
    busy_bar: BusyBar, message: str, fn: Callable, *args: Any, on_success: Callable, on_failure: Callable, **kwargs: Any
) -> FunctionWorker:
    """Starts ``busy_bar``, runs ``fn(*args, **kwargs)`` on a background
    FunctionWorker, and wires its result/failure signals -- the "launch
    a long-running orbit call" sequence every tab needs. Caller must
    keep the returned worker referenced (e.g. ``self.worker = run_worker(...)``)
    so it isn't garbage-collected mid-run.

    ``worker.wait()`` before invoking the caller's handler matters: our
    finished_ok/failed signals are emitted from inside run(), queued to
    the GUI thread, which can be delivered a moment before the OS thread
    has actually finished unwinding -- if the handler then replaces this
    reference (e.g. self.worker = run_worker(...) again), Python can
    garbage-collect a QThread Qt still considers running, which aborts
    the process. wait() blocks until it's genuinely done first (near-
    instant here, since run() has already returned by the time either
    signal fires).
    """
    busy_bar.start(message)
    worker = FunctionWorker(fn, *args, **kwargs)

    def _joined(handler: Callable, arg: Any) -> None:
        worker.wait()
        handler(arg)

    worker.finished_ok.connect(lambda result: _joined(on_success, result))
    worker.failed.connect(lambda message: _joined(on_failure, message))
    worker.start()
    return worker
