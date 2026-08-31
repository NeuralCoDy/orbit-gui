"""CPU/RAM usage indicators for MainWindow's status bar (see app.py) --
both this app's OWN usage and the system-wide total, side by side, since
neither number alone tells the whole story:

- This app's own number would look near-idle during exactly the runs a
  user most wants to watch, if it only counted the main GUI process:
  patch-based CNMF/CNMF-E/GraFT run in separate worker processes via
  ProcessPoolExecutor (see cnmf.py's _run_patches_and_merge), not the
  main process itself. _AppUsageTracker below sums the main process
  with its current live children (recursively), so a patch run's worker
  processes are counted as "this app" while they're alive and drop back
  out once the pool shuts down.
- The system-wide number is what "is the machine about to run out of
  memory" really needs (following this session's own earlier close call
  chasing an OOM crash from an unbounded FFT allocation) -- this app
  might be well-behaved while something else on the machine isn't, or
  vice versa.

Qt's own QStatusBar left/right convention (addWidget vs
addPermanentWidget) places these rather than a custom layout -- see
add_resource_monitor below.
"""

from __future__ import annotations

import psutil
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QStatusBar

_POLL_INTERVAL_MS = 2000


def _format_bytes(n: float) -> str:
    """Human-readable byte count, e.g. "58.2 GB" -- binary (1024-based)
    units, matching psutil's own byte convention."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


class _AppUsageTracker:
    """This app's own CPU%/RSS: the main process plus every currently-
    live descendant (recursively -- ProcessPoolExecutor's worker
    processes are direct children of the main process while a patch-
    based run is in flight, gone again once its ``with`` block exits).

    Persists one psutil.Process object per pid across refreshes rather
    than constructing a fresh one each tick: Process.cpu_percent()
    reports elapsed CPU time since ITS OWN previous call on that same
    object, not since the process started, so a freshly-built wrapper
    always reports 0.0 on its first call regardless of how long the
    real process has actually been running -- newly-seen children are
    primed once (see _sync_children) the same way __init__ primes the
    main process.

    ``main_process`` is injectable (defaults to ``psutil.Process()``,
    i.e. this process) so tests can substitute a fake process tree
    instead of monkeypatching psutil globally."""

    def __init__(self, main_process: psutil.Process | None = None) -> None:
        self._main = main_process if main_process is not None else psutil.Process()
        self._main.cpu_percent(interval=None)  # prime
        self._children: dict[int, psutil.Process] = {}

    def _sync_children(self) -> None:
        try:
            live = self._main.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            live = []
        live_pids = set()
        for child in live:
            live_pids.add(child.pid)
            if child.pid not in self._children:
                try:
                    child.cpu_percent(interval=None)  # prime a newly-seen child
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                self._children[child.pid] = child
        for pid in list(self._children):
            if pid not in live_pids:
                del self._children[pid]  # child exited (e.g. its patch finished) since the last refresh

    def sample(self) -> tuple[float, int]:
        """Returns (cpu_percent, rss_bytes) summed across the main
        process and its current live children, in one shared child-list
        sync. cpu_percent follows psutil's own per-process convention
        (100% per fully-busy core -- several concurrent worker processes
        can push this well past 100%, unlike the system-wide figure)."""
        self._sync_children()
        cpu = self._main.cpu_percent(interval=None)
        rss = self._main.memory_info().rss
        for proc in list(self._children.values()):
            try:
                cpu += proc.cpu_percent(interval=None)
                rss += proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue  # exited between _sync_children and here -- a real, harmless race
        return cpu, rss


class CPULabel(QLabel):
    """"CPU: <this app>% (<system total>% total)" -- see module
    docstring for why both numbers matter."""

    def __init__(self, tracker: _AppUsageTracker | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setToolTip("This app's own CPU usage (all its processes) vs. the whole system's")
        self._tracker = tracker if tracker is not None else _AppUsageTracker()
        psutil.cpu_percent(interval=None)  # first call always returns 0.0 -- primes the system-wide baseline
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(_POLL_INTERVAL_MS)
        self._refresh()

    def _refresh(self) -> None:
        app_cpu, _app_rss = self._tracker.sample()
        system_cpu = psutil.cpu_percent(interval=None)
        self.setText(f"CPU: {app_cpu:.0f}% ({system_cpu:.0f}% total)")


class RAMLabel(QLabel):
    """"RAM: <this app> (<system used> total) / <system total>"."""

    def __init__(self, tracker: _AppUsageTracker | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setToolTip("This app's own RAM usage (all its processes) vs. the whole system's")
        self._tracker = tracker if tracker is not None else _AppUsageTracker()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(_POLL_INTERVAL_MS)
        self._refresh()

    def _refresh(self) -> None:
        _app_cpu, app_rss = self._tracker.sample()
        vm = psutil.virtual_memory()
        self.setText(f"RAM: {_format_bytes(app_rss)} ({_format_bytes(vm.used)} total) / {_format_bytes(vm.total)}")


def add_resource_monitor(status_bar: QStatusBar) -> tuple[CPULabel, RAMLabel]:
    """Adds a CPU indicator at the far left and a RAM indicator at the
    far right of ``status_bar`` -- addWidget (CPU) and addPermanentWidget
    (RAM) are Qt's own left/right status-bar placement, not a custom
    layout. Both labels share one _AppUsageTracker (rather than each
    building their own) so the app-side process-tree sync only happens
    once per refresh tick, not twice. Returns both labels (mainly for
    tests; app.py itself doesn't need to keep a reference -- QTimer(self)
    parents each label's own poll timer to it, so it's cleaned up
    automatically with the label)."""
    tracker = _AppUsageTracker()
    cpu_label = CPULabel(tracker)
    ram_label = RAMLabel(tracker)
    status_bar.addWidget(cpu_label)
    status_bar.addPermanentWidget(ram_label)
    return cpu_label, ram_label
