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
    """Runs an arbitrary callable off the GUI thread. self.fn/args/kwargs
    are cleared once run() has actually called fn (see _drop_call_args) --
    don't read them expecting to find "what this worker ran" after the
    fact; they exist only to get the call into run()'s own background
    thread."""

    finished_ok = Signal(object)
    failed = Signal(str)
    warning = Signal(str)
    progress = Signal(int, int)  # (done, total) -- see run_worker's on_progress

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
        except Exception as exc:  # noqa: BLE001
            self._drop_call_args()
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        # Drop our own copies of fn/args/kwargs now that fn has already
        # consumed them -- run_worker's own docstring says the caller
        # must keep the returned worker referenced (self.worker = ...)
        # so it isn't garbage-collected mid-run, but that means a
        # FINISHED worker stays referenced (and alive) for the rest of
        # that tab's lifetime too, pinning whatever was passed in right
        # alongside it -- typically a full movie-sized array. Confirmed
        # via a real 5-stage pipeline run on the real default dataset:
        # every StageTab's own self.worker was still holding a full
        # movie-sized array this way, long after its result had already
        # been delivered and rendered -- the single largest remaining
        # contributor to that pipeline's overall memory footprint even
        # after clearing every OTHER stale movie reference we'd found.
        self._drop_call_args()
        for w in caught:
            self.warning.emit(str(w.message))
        self.finished_ok.emit(result)

    def _drop_call_args(self) -> None:
        self.fn = None
        self.args = ()
        self.kwargs = {}


def run_worker(
    busy_bar: BusyBar, message: str, fn: Callable, *args: Any, on_success: Callable, on_failure: Callable,
    on_progress: bool = False, **kwargs: Any,
) -> FunctionWorker:
    """Starts ``busy_bar``, runs ``fn(*args, **kwargs)`` on a background
    FunctionWorker, and wires its result/failure signals -- the "launch
    a long-running orbit call" sequence every tab needs. Caller must
    keep the returned worker referenced (e.g. ``self.worker = run_worker(...)``)
    so it isn't garbage-collected mid-run.

    ``on_progress=True`` is for an ``fn`` that itself accepts a
    ``progress`` keyword (a ``(done, total) -> None`` callback, e.g.
    load_volumetric_tiff_folder) -- it's given ``worker.progress.emit``,
    which is safe to call from ``fn``'s background thread (Qt auto-
    queues cross-thread signal emits to the receiving/GUI thread), and
    the emitted values are wired straight to ``busy_bar.set_progress``
    so the bar shows real progress instead of guessing from elapsed
    time. Leave it off for an ``fn`` that has no such notion of progress.

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
    if on_progress:
        worker.kwargs["progress"] = worker.progress.emit
        worker.progress.connect(busy_bar.set_progress)

    def _joined(handler: Callable, arg: Any) -> None:
        worker.wait()
        handler(arg)

    worker.finished_ok.connect(lambda result: _joined(on_success, result))
    worker.failed.connect(lambda message: _joined(on_failure, message))
    worker.start()
    return worker
