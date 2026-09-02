"""Shared "how many worker threads/processes should we use" helper --
every module here that sizes a thread/process pool off the machine's
own core count needs this, not just os.cpu_count() directly.
"""

from __future__ import annotations

import os


def available_cpu_count() -> int:
    """CPUs actually usable by THIS process, not the system's total
    core count -- os.cpu_count() ignores CPU affinity restrictions
    entirely (confirmed directly: still reports the full physical core
    count even when the process itself has been pinned to a handful of
    cores via taskset/cgroups' cpuset controller -- a real deployment
    shape, e.g. a container with a CPU limit or a shared HPC/cluster job
    allocation). Sizing a worker pool off the wrong (too high) number
    reproduces the same "too many workers hurts, not just fails to
    help" pattern measured directly on an unrestricted 80-core machine
    (see patchwarp.py's own _DEFAULT_MAX_WORKERS comment) -- just
    starting from a much lower true core count, making the effective
    oversubscription worse.

    os.process_cpu_count() (Python 3.13+) is the correct, portable
    answer where available. Older Python falls back to
    os.sched_getaffinity (Linux-only, the same underlying mechanism);
    anywhere else (macOS, Windows, or a 3.10-3.12 install without
    sched_getaffinity) falls back to plain os.cpu_count(), matching
    this codebase's previous, affinity-blind behavior -- not worse than
    the status quo, just not improved on those platforms."""
    if hasattr(os, "process_cpu_count"):
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1
