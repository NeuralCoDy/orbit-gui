import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QStatusBar  # noqa: E402

from orbitapp.widgets import resource_monitor  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


class _FakeProcess:
    """Stand-in for psutil.Process: fixed cpu_percent/rss, and a
    children() list that a test can mutate between sample() calls to
    simulate a worker process appearing/exiting."""

    def __init__(self, pid: int, cpu: float, rss: int, children: list | None = None) -> None:
        self.pid = pid
        self.cpu = cpu
        self.rss = rss
        self._children = children or []

    def cpu_percent(self, interval=None) -> float:
        return self.cpu

    def memory_info(self):
        return type("MemInfo", (), {"rss": self.rss})()

    def children(self, recursive=True) -> list:
        return self._children


def test_format_bytes_picks_a_sensible_unit():
    assert resource_monitor._format_bytes(500) == "500 B"
    assert resource_monitor._format_bytes(2048) == "2.0 KB"
    assert resource_monitor._format_bytes(3 * 1024**2) == "3.0 MB"
    assert resource_monitor._format_bytes(58.2 * 1024**3) == "58.2 GB"


def test_app_usage_tracker_reports_just_the_main_process_when_childless():
    main = _FakeProcess(pid=1, cpu=12.0, rss=100)
    tracker = resource_monitor._AppUsageTracker(main)

    cpu, rss = tracker.sample()

    assert cpu == 12.0
    assert rss == 100


def test_app_usage_tracker_sums_in_live_children():
    child = _FakeProcess(pid=2, cpu=30.0, rss=500)
    main = _FakeProcess(pid=1, cpu=5.0, rss=100, children=[child])
    tracker = resource_monitor._AppUsageTracker(main)

    cpu, rss = tracker.sample()

    assert cpu == 35.0
    assert rss == 600


def test_app_usage_tracker_drops_a_child_that_has_since_exited():
    # Simulates a patch-based run's worker process finishing between
    # two refresh ticks -- e.g. cnmf.py's ProcessPoolExecutor pool
    # shutting down once patch_cnmf_source_extraction returns.
    child = _FakeProcess(pid=2, cpu=30.0, rss=500)
    main = _FakeProcess(pid=1, cpu=5.0, rss=100, children=[child])
    tracker = resource_monitor._AppUsageTracker(main)
    tracker.sample()  # first tick: child is alive and gets tracked

    main._children = []  # the worker process has since exited
    cpu, rss = tracker.sample()

    assert cpu == 5.0
    assert rss == 100


def test_app_usage_tracker_picks_up_a_newly_spawned_child():
    main = _FakeProcess(pid=1, cpu=5.0, rss=100)
    tracker = resource_monitor._AppUsageTracker(main)
    tracker.sample()  # first tick: no children yet

    child = _FakeProcess(pid=2, cpu=40.0, rss=1000)
    main._children = [child]  # a patch-based run just started a worker
    cpu, rss = tracker.sample()

    assert cpu == 45.0
    assert rss == 1100


def test_cpu_label_shows_app_and_system_wide_percentages(monkeypatch):
    monkeypatch.setattr(resource_monitor.psutil, "cpu_percent", lambda interval=None: 42.0)
    tracker = resource_monitor._AppUsageTracker(_FakeProcess(pid=1, cpu=7.0, rss=0))

    label = resource_monitor.CPULabel(tracker)

    assert label.text() == "CPU: 7% (42% total)"


def test_ram_label_shows_app_usage_system_used_and_system_total(monkeypatch):
    class _VM:
        used = 4 * 1024**3
        total = 16 * 1024**3
        percent = 25.0

    monkeypatch.setattr(resource_monitor.psutil, "virtual_memory", lambda: _VM())
    tracker = resource_monitor._AppUsageTracker(_FakeProcess(pid=1, cpu=0.0, rss=int(0.5 * 1024**3)))

    label = resource_monitor.RAMLabel(tracker)

    assert label.text() == "RAM: 512.0 MB (4.0 GB total) / 16.0 GB"


def test_add_resource_monitor_places_cpu_left_and_ram_right(monkeypatch):
    monkeypatch.setattr(resource_monitor.psutil, "Process", lambda: _FakeProcess(pid=1, cpu=1.0, rss=1024))
    monkeypatch.setattr(resource_monitor.psutil, "cpu_percent", lambda interval=None: 10.0)
    monkeypatch.setattr(
        resource_monitor.psutil, "virtual_memory",
        lambda: type("VM", (), {"used": 1024**3, "total": 2 * 1024**3, "percent": 50.0})(),
    )
    status_bar = QStatusBar()
    status_bar.resize(400, 24)

    cpu_label, ram_label = resource_monitor.add_resource_monitor(status_bar)
    status_bar.show()
    QApplication.processEvents()

    assert isinstance(cpu_label, resource_monitor.CPULabel)
    assert isinstance(ram_label, resource_monitor.RAMLabel)
    assert cpu_label.text() == "CPU: 1% (10% total)"
    assert ram_label.text() == "RAM: 1.0 KB (1.0 GB total) / 2.0 GB"
    # addWidget (CPU) is Qt's own documented "left" placement,
    # addPermanentWidget (RAM) its "right" one.
    assert cpu_label.x() < ram_label.x()


def test_cpu_label_refreshes_on_its_own_timer(monkeypatch):
    calls = {"n": 0}

    def _fake_system_cpu(interval=None):
        calls["n"] += 1
        return float(calls["n"])

    monkeypatch.setattr(resource_monitor.psutil, "cpu_percent", _fake_system_cpu)
    tracker = resource_monitor._AppUsageTracker(_FakeProcess(pid=1, cpu=0.0, rss=0))
    label = resource_monitor.CPULabel(tracker)
    first_text = label.text()
    label._refresh()
    assert label.text() != first_text
