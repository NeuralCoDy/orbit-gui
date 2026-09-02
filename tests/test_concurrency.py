import os
import subprocess
import sys
from pathlib import Path

import pytest

from orbit._concurrency import available_cpu_count

_SRC = str(Path(__file__).resolve().parent.parent / "src")


def test_available_cpu_count_returns_a_positive_int():
    n = available_cpu_count()
    assert isinstance(n, int)
    assert n >= 1


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="sched_getaffinity is Linux-only")
def test_available_cpu_count_respects_cpu_affinity_restriction():
    # os.cpu_count() ignores CPU affinity entirely -- it reports the
    # machine's full core count even when this process has been pinned
    # to a handful of cores (confirmed directly: still reports the full
    # physical core count under `taskset -c 0-3`, which really does
    # restrict the process to 4 cores). available_cpu_count() must NOT
    # have that blind spot, since every _DEFAULT_MAX_WORKERS in this
    # codebase is min()'d against it -- an affinity-restricted machine
    # (a container with a CPU limit, a shared HPC/cluster allocation,
    # ...) would otherwise oversubscribe its real core count exactly the
    # way an unbounded worker count already measurably hurt on an
    # unrestricted 80-core machine (see patchwarp.py's own
    # _DEFAULT_MAX_WORKERS comment), just starting from a much lower
    # true core count. Uses a real `taskset`-restricted subprocess
    # rather than mocking os.sched_getaffinity/os.process_cpu_count,
    # since the whole point is confirming those actually observe a real
    # OS-level restriction, not just that this function calls them.
    if subprocess.run(["taskset", "-c", "0-1", "true"], capture_output=True).returncode != 0:
        pytest.skip("taskset not available")
    result = subprocess.run(
        ["taskset", "-c", "0-1", sys.executable, "-c", "import sys; sys.path.insert(0, %r); "
         "from orbit._concurrency import available_cpu_count; print(available_cpu_count())" % _SRC],
        capture_output=True, text=True, check=True,
    )
    assert int(result.stdout.strip()) == 2


def test_available_cpu_count_falls_back_to_cpu_count_when_neither_newer_api_is_present(monkeypatch):
    # Python <3.13 without sched_getaffinity (non-Linux) -- must still
    # return something sane rather than raising.
    monkeypatch.delattr("os.process_cpu_count", raising=False)
    monkeypatch.delattr("os.sched_getaffinity", raising=False)
    monkeypatch.setattr("orbit._concurrency.os.cpu_count", lambda: 7)
    assert available_cpu_count() == 7
