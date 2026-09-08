"""Shared "how many worker threads/processes should we use" helpers --
every module here that sizes a thread/process pool off the machine's
own core count needs this, not just os.cpu_count() directly.
"""

from __future__ import annotations

import os

import threadpoolctl


def limited_native_threads(n: int) -> threadpoolctl.threadpool_limits:
    """Temporarily caps every native-threading library ALREADY LOADED in
    this process (OpenMP, OpenBLAS, MKL, ...) to ``n`` threads, restored
    on exit -- for a call into a compiled extension (graft's own
    OpenMP-parallel native solver) that has no worker-count parameter of
    its own to pass a limit through. Returns threadpoolctl's own context
    manager directly (usable as ``with limited_native_threads(n):``) --
    not a thin wrapper reimplementing it, since getting this wrong once
    already caused a real, reproducible crash (see below).

    threadpoolctl, NOT setting OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/...
    via os.environ directly: an env var is only read when a native
    library FIRST initializes its own thread pool, typically at first
    use -- by the time GraFT is called in a real orbit-gui session,
    NumPy/SciPy/graft have all already run plenty of BLAS/OpenMP calls
    of their own, so changing the env var afterward does nothing for
    already-initialized pools, silently failing to cap anything. Worse,
    confirmed directly: changing OPENBLAS_NUM_THREADS this way WHILE
    that library's thread pool is already live isn't just a no-op, it's
    unsafe -- reproduced a real heap corruption crash ("corrupted size
    vs. prev_size") this way on a 150x150x150 whole-FOV GraFT call.
    threadpoolctl instead finds each already-loaded native library's own
    runtime thread-count control function (via introspection, not env
    vars) and calls that directly -- confirmed both safe (no crash,
    repeated runs) and effective (correctly found and capped graft's own
    bundled libgomp, not just NumPy's BLAS) on that same repro.

    cnmf.py's own _single_threaded_blas_for_children is a DIFFERENT
    situation this function doesn't cover: it sets env vars for
    processes not yet spawned (a ProcessPoolExecutor's future workers),
    which inherit them at spawn/exec time before ever touching BLAS
    themselves -- safe for exactly the reason changing them for an
    already-running process's already-loaded libraries is not."""
    return threadpoolctl.threadpool_limits(limits=n)


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
