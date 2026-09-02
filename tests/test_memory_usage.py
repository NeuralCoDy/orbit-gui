"""Peak-memory regression tests for the two stages users run on the
largest movies: motion correction and source extraction (CNMF).

Each case launches tests/_memory_worker.py in a fresh subprocess and
polls its peak resident memory from the *parent* side, via
/proc/<pid>/status's VmHWM, rather than trusting the child's own
self-reported resource.getrusage(RUSAGE_SELF).ru_maxrss. That distinction
matters, and was tracked down empirically while writing these tests: a
subprocess spawned (fork()+exec() *or* posix_spawn(), both were tried)
from a pytest process that has already grown large (e.g. after a heavy
prior test builds up a lot of Qt/numpy state) reports an inflated
ru_maxrss that's already present at the child's very first instruction,
before any of its own imports or work -- a documented Linux quirk where
a freshly-spawned child's RSS high-water mark starts from whatever the
parent's RSS was at spawn time and isn't reset by exec(). Confirmed via
a diagnostic script printing ru_maxrss at multiple checkpoints: the
number was identical from process start through every import to the
final result, for a workload that should only need ~100MB by itself.
/proc/<pid>/status's VmHWM, read from outside the child while it runs,
does not have this problem (confirmed against the same repro: correctly
tiny for a trivial child regardless of how large the spawning parent was).

Peak RSS is summed across the worker process AND every live descendant
at each poll (see _live_descendant_pids) rather than just the worker
process's own VmHWM, since patch-based CNMF now runs its patches in a
ProcessPoolExecutor -- a single process's VmHWM wouldn't reflect
concurrently-running sibling worker processes at all.

Bounds are deliberately loose -- around 1.5-2x what was actually
measured when these were written -- since the point isn't to pin memory
usage exactly (that varies by machine/BLAS backend/allocator), it's to
catch a real regression, e.g. an accidentally-reintroduced extra
full-(H, W, T) copy roughly doubling peak use.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("resource")
if not Path("/proc").is_dir():
    pytest.skip("these tests poll /proc/<pid>/status for VmHWM -- Linux only", allow_module_level=True)

_WORKER = Path(__file__).resolve().parent / "_memory_worker.py"


def _live_descendant_pids(pid: int) -> list[int]:
    """Every currently-live process whose ancestry traces back to
    ``pid`` (children, grandchildren, ...) -- patch-based CNMF now
    spawns its own worker subprocesses (ProcessPoolExecutor), whose
    memory use wouldn't show up in ``pid``'s own VmHWM at all."""
    children_by_ppid: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            continue
        # PPid is the field right after the ")" closing the (comm) field,
        # which may itself contain spaces/parens.
        after_comm = stat.rsplit(")", 1)[1].split()
        ppid = int(after_comm[1])
        children_by_ppid.setdefault(ppid, []).append(int(entry.name))

    descendants: list[int] = []
    frontier = [pid]
    while frontier:
        next_frontier = []
        for parent in frontier:
            next_frontier.extend(children_by_ppid.get(parent, []))
        descendants.extend(next_frontier)
        frontier = next_frontier
    return descendants


def _vm_hwm_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except FileNotFoundError:
        pass  # process exited between discovery and this read
    return 0


def _peak_rss_mb(
    stage: str, height: int, width: int, n_frames: int, dtype: str, timeout: float = 90.0, n_stages: int | None = None,
) -> float:
    args = [sys.executable, str(_WORKER), stage, str(height), str(width), str(n_frames), dtype]
    if n_stages is not None:  # gui_pipeline-only -- see _memory_worker._run_gui_pipeline
        args.append(str(n_stages))
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Tracks the peak *sum* of VmHWM across the worker process and every
    # live descendant at each poll -- a single process's own VmHWM
    # doesn't capture concurrently-running sibling worker processes
    # (e.g. patch-based CNMF's ProcessPoolExecutor workers), and summing
    # only what's alive at each snapshot (rather than summing every
    # descendant's VmHWM ever seen, even after some have already exited
    # and been replaced) approximates concurrent peak rather than
    # overcounting a long sequence of small, non-overlapping workers.
    peak_kb = 0
    deadline = time.monotonic() + timeout
    try:
        while proc.poll() is None:
            if time.monotonic() > deadline:
                proc.kill()
                raise subprocess.TimeoutExpired(str(_WORKER), timeout)
            pids = [proc.pid, *_live_descendant_pids(proc.pid)]
            total_kb = sum(_vm_hwm_kb(p) for p in pids)
            peak_kb = max(peak_kb, total_kb)
            time.sleep(0.02)
    finally:
        returncode = proc.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, proc.args)
    return peak_kb / 1e3


def test_rigid_motion_correction_peak_memory_is_bounded():
    # A raw uint16 movie forces a float64 working copy internally (~4x);
    # peak should stay close to that one copy plus a fixed baseline, not
    # blow up with extra full-movie temporaries.
    peak_mb = _peak_rss_mb("motion_rigid", 300, 300, 400, "uint16")
    assert peak_mb < 1000, f"rigid motion correction peak RSS {peak_mb:.0f}MB exceeds bound"


def test_rigid_motion_correction_memmap_commit_scales_sublinearly_with_frame_count():
    # The chunked Commit path (real memmap input, FITS-backed memmap
    # output, chunk-at-a-time processing) should NOT scale peak RSS
    # linearly with total frame count the way the in-RAM path does --
    # 10x the frames should cost nowhere near 10x the peak RSS. Some
    # growth is still expected: on Linux, ru_maxrss counts every
    # touched (already-written) memmap page as resident, even though
    # those are "clean" file-backed pages the OS can reclaim under real
    # memory pressure -- this isn't a hard ceiling the way an actual
    # full-movie in-RAM copy would be, just how RSS accounting works for
    # memory-mapped I/O.
    small = _peak_rss_mb("motion_rigid_memmap", 150, 150, 300, "uint16")
    large = _peak_rss_mb("motion_rigid_memmap", 150, 150, 3000, "uint16", timeout=60.0)
    assert large < small * 5, (
        f"peak RSS grew {large / small:.1f}x for a 10x increase in frame count "
        f"({small:.0f}MB -> {large:.0f}MB) -- expected much less than proportional growth"
    )


def test_patch_motion_correction_peak_memory_is_bounded():
    # Bound is looser than the other cases here -- this stage's ThreadPoolExecutor
    # workers (per-patch phase correlation) show more run-to-run peak-RSS
    # variance under system load than the other stages did, even though
    # the underlying algorithm's memory use didn't change.
    peak_mb = _peak_rss_mb("motion_patch", 200, 200, 80, "uint16")
    assert peak_mb < 750, f"patch motion correction peak RSS {peak_mb:.0f}MB exceeds bound"


def test_cnmf_source_extraction_peak_memory_is_bounded():
    peak_mb = _peak_rss_mb("cnmf", 180, 180, 200, "float32")
    assert peak_mb < 700, f"CNMF peak RSS {peak_mb:.0f}MB exceeds bound"


def test_patch_cnmf_source_extraction_peak_memory_is_bounded():
    # The patch-based path exists specifically so peak memory stays
    # bounded by patch size rather than the whole field of view --
    # regressing that (e.g. accidentally materializing a full-FOV
    # intermediate) is exactly what this guards against. Patches now run
    # concurrently across worker processes (see cnmf.py's
    # ProcessPoolExecutor-based parallelism), so the bound accounts for
    # several workers' fixed per-process overhead (numpy/scipy/sklearn
    # imports, ~4 concurrent by default) on top of each one's own patch
    # data -- measured ~1070MB at the time this bound was set.
    peak_mb = _peak_rss_mb("patch_cnmf", 250, 250, 150, "float32")
    assert peak_mb < 1700, f"patch-based CNMF peak RSS {peak_mb:.0f}MB exceeds bound"


def test_patch_graft_source_extraction_peak_memory_is_bounded():
    # Same reasoning as patch-based CNMF above. GraFT's own compiled
    # solver has a higher fixed overhead per patch-worker than CNMF's
    # pure-Python/numpy path (confirmed empirically: ~1.0-1.1GB here vs
    # patch CNMF's much smaller footprint at a comparable size), hence
    # the wider bound -- still catches a real regression (e.g. a patch
    # accidentally spanning the whole FOV) without being sensitive to
    # that fixed cost.
    peak_mb = _peak_rss_mb("patch_graft", 250, 250, 150, "float32", timeout=60.0)
    assert peak_mb < 2000, f"patch-based GraFT peak RSS {peak_mb:.0f}MB exceeds bound"


def test_gui_pipeline_peak_memory_does_not_scale_with_pipeline_depth():
    # Regression test for StageTab/FunctionWorker holding full-size
    # movie references forever once Apply had been clicked on a tab even
    # once (self._input_movie, the panel's "after" movie, FunctionWorker's
    # own .args/.kwargs -- see their docstrings). A single committed
    # stage's own baseline (widget construction, one active movie, ...)
    # is expected to cost something; what must NOT happen is that cost
    # multiplying by pipeline depth. Compares a 1-stage vs a 5-stage
    # pipeline (Motion Correction -> Mask -> Denoising -> Normalization ->
    # Detrending, see _memory_worker._run_gui_pipeline) on the SAME movie
    # size, same style as the memmap frame-count scaling test above.
    #
    # Calibrated against a real before/after measurement at this exact
    # movie size: the delta was ~443MB pre-fix (~110MB/stage) vs ~169MB
    # post-fix (~42MB/stage) for these same 4 extra stages -- 300MB
    # cleanly separates the two while leaving real headroom for
    # legitimate per-tab overhead (each stage's own widgets, params
    # dialog, ...) that isn't itself a bug.
    one_stage = _peak_rss_mb("gui_pipeline", 250, 250, 300, "float32", timeout=60.0, n_stages=1)
    five_stages = _peak_rss_mb("gui_pipeline", 250, 250, 300, "float32", timeout=60.0, n_stages=5)
    delta = five_stages - one_stage
    assert delta < 300, (
        f"peak RSS grew {delta:.0f}MB from 1 to 5 committed pipeline stages "
        f"({one_stage:.0f}MB -> {five_stages:.0f}MB) -- expected roughly flat, not growing with pipeline depth"
    )
