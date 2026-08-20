"""Runs one pipeline stage on a synthetic movie in its own process and
prints its peak RSS in KB (Linux/macOS ru_maxrss) -- invoked via
subprocess from test_memory_usage.py so each measurement starts from a
clean process, not contaminated by whatever else has run earlier in the
same pytest session.
"""

from __future__ import annotations

import resource
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402


def _make_movie(height: int, width: int, n_frames: int, dtype: str) -> np.ndarray:
    rng = np.random.default_rng(0)
    if np.issubdtype(np.dtype(dtype), np.integer):
        return rng.integers(0, 4000, size=(height, width, n_frames), dtype=dtype)
    return (rng.standard_normal((height, width, n_frames)).astype(dtype) * 100 + 500)


def _run(stage: str, height: int, width: int, n_frames: int, dtype: str) -> None:
    movie = _make_movie(height, width, n_frames, dtype)

    if stage == "motion_rigid":
        from orbit.motion_correction import rigid_motion_correct

        rigid_motion_correct(movie, bin_width=200, n_iter=1)
    elif stage == "motion_rigid_memmap":
        # The Commit-time chunked path (StageTab._chunked_commit /
        # MotionCorrectionTab._chunked_commit): input is a real memmap,
        # output is a new FITS-backed memmap written chunk by chunk --
        # peak RSS should stay bounded by chunk size, not total frames.
        from orbitapp.fits_io import create_fits_memmap
        from orbitapp.io import load_movie
        from orbit.motion_correction import rigid_motion_correct

        with tempfile.TemporaryDirectory() as d:
            src_path = Path(d) / "movie.npy"
            np.save(src_path, movie)
            source = load_movie(src_path, mmap=True)
            output = create_fits_memmap(Path(d) / "out.fits", source.shape, np.float32)
            rigid_motion_correct(source, bin_width=50, n_iter=1, output=output)
            output.flush()
    elif stage == "motion_patch":
        from orbit.motion_correction import patch_motion_correct

        patch_motion_correct(movie, grid_size=32, bin_width=200, n_iter=1, min_patch_contrast=0)
    elif stage == "cnmf":
        from orbit.cnmf import cnmf_source_extraction

        cnmf_source_extraction(movie.astype(np.float64), n_components=8, n_iterations=1)
    elif stage == "patch_cnmf":
        from orbit.cnmf import patch_cnmf_source_extraction

        patch_cnmf_source_extraction(
            movie.astype(np.float64), patch_size=(80, 80), overlap=20, n_components_per_patch=6, n_iterations=1
        )
    else:
        raise ValueError(f"unknown stage: {stage!r}")


if __name__ == "__main__":
    _stage, _h, _w, _t, _dtype = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    _run(_stage, _h, _w, _t, _dtype)
    # ru_maxrss is KB on Linux, bytes on macOS -- this repo's CI/dev target is Linux.
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
