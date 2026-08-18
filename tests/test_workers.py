import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from orbitapp.widgets import BusyBar  # noqa: E402
from orbitapp.workers import run_worker  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def _run_until(*signals, timeout_ms=5000) -> None:
    loop = QEventLoop()
    for sig in signals:
        sig.connect(loop.quit)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()


def test_run_worker_starts_busy_bar_and_reports_success():
    bar = BusyBar()
    results = []

    worker = run_worker(bar, "Working...", lambda x: x * 2, 21, on_success=results.append, on_failure=lambda _m: None)

    assert bar.bar.isVisibleTo(bar) is True
    _run_until(worker.finished_ok)

    assert results == [42]


def test_run_worker_reports_failure():
    bar = BusyBar()
    failures = []

    def _boom():
        raise ValueError("nope")

    worker = run_worker(bar, "Working...", _boom, on_success=lambda _r: None, on_failure=failures.append)
    _run_until(worker.failed)

    assert len(failures) == 1
    assert "nope" in failures[0]
