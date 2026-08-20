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


def _peak_rss_mb(stage: str, height: int, width: int, n_frames: int, dtype: str, timeout: float = 90.0) -> float:
    proc = subprocess.Popen(
        [sys.executable, str(_WORKER), stage, str(height), str(width), str(n_frames), dtype],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    peak_kb = 0
    deadline = time.monotonic() + timeout
    try:
        while proc.poll() is None:
            if time.monotonic() > deadline:
                proc.kill()
                raise subprocess.TimeoutExpired(str(_WORKER), timeout)
            try:
                with open(f"/proc/{proc.pid}/status") as f:
                    for line in f:
                        if line.startswith("VmHWM:"):
                            peak_kb = max(peak_kb, int(line.split()[1]))
                            break
            except FileNotFoundError:
                pass  # process exited between poll() and the /proc read
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
    # intermediate) is exactly what this guards against.
    peak_mb = _peak_rss_mb("patch_cnmf", 250, 250, 150, "float32")
    assert peak_mb < 600, f"patch-based CNMF peak RSS {peak_mb:.0f}MB exceeds bound"
