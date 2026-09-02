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


def _run(stage: str, height: int, width: int, n_frames: int, dtype: str, n_stages: int = 5) -> None:
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
    elif stage == "patch_graft":
        from orbit.roi_extraction_graft import patch_graft_source_extraction

        patch_graft_source_extraction(
            movie.astype(np.float64), patch_size=(80, 80), overlap=(20, 20), n_dict_per_patch=6,
        )
    elif stage == "gui_pipeline":
        _run_gui_pipeline(movie, n_stages)
    elif stage == "patchwarp":
        from orbit.patchwarp import patchwarp_motion_correct

        patchwarp_motion_correct(movie.astype(np.float64), grid_size=2, ecc_iterations=20, max_workers=4)
    else:
        raise ValueError(f"unknown stage: {stage!r}")


def _run_gui_pipeline(movie: np.ndarray, n_stages: int) -> None:
    """Drives ``n_stages`` real StageTab commits in sequence (Motion
    Correction -> Mask -> Denoising -> Normalization -> Detrending) --
    the GUI layer's own equivalent of the raw-algorithm stages above,
    catching a regression in StageTab/FunctionWorker's own reference-
    holding (self._input_movie, the panel's "after" movie,
    FunctionWorker.args/kwargs) rather than anything about a single
    algorithm's own internal memory use (already covered above). See
    StageTab._clear_stale_candidate's and FunctionWorker's own docstrings
    for the bug this guards against: before those existed, every
    StageTab-derived tab kept its own full-size input/candidate movie
    alive for the rest of the app's lifetime once Apply had been clicked
    on it even once -- confirmed via a real 5-stage pipeline run on a
    real ~1GB dataset, ~9.4GB of that kind of waste alone, on top of
    another ~5GB held by finished-but-still-referenced FunctionWorker
    instances. ``n_stages`` lets test_memory_usage.py compare a short vs
    long pipeline's peak RSS (see its own scaling-style test) rather than
    just pinning one absolute number, which is what actually catches
    "grows with pipeline depth" regressions specifically."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    # Qt's own QApplication singleton stays alive globally once
    # constructed (same as every other test fixture's
    # ``QApplication.instance() or QApplication([])`` -- see
    # tests/test_stage_tab.py's qapp fixture), so no local reference
    # needs to be kept around here.
    QApplication.instance() or QApplication([])

    from orbitapp.state import AppState
    from orbitapp.tabs.denoising_tab import DenoisingTab
    from orbitapp.tabs.detrending_tab import DetrendingTab
    from orbitapp.tabs.mask_tab import MaskTab
    from orbitapp.tabs.motion_correction_tab import MotionCorrectionTab
    from orbitapp.tabs.normalization_tab import NormalizationTab

    def _wait(tab) -> None:
        if tab.worker is not None:
            tab.worker.wait(30000)
        for _ in range(50):
            QApplication.processEvents()

    state = AppState()
    state.load("movie.npy", movie.astype(np.float64))
    all_tabs = [
        MotionCorrectionTab(state), MaskTab(state), DenoisingTab(state),
        NormalizationTab(state), DetrendingTab(state),
    ]
    tabs = all_tabs[:n_stages]
    for t in tabs:
        t.on_data_loaded()

    for tab in tabs:
        tab._apply()
        _wait(tab)
        if tab._pending_result is None:
            continue
        tab._commit()
        _wait(tab)
        for other in tabs:
            other.on_data_loaded()


if __name__ == "__main__":
    _stage, _h, _w, _t, _dtype = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    # optional trailing arg, "gui_pipeline"-only: how many StageTab
    # commits to drive (see _run_gui_pipeline) -- every other stage
    # ignores it.
    _n_stages = int(sys.argv[6]) if len(sys.argv) > 6 else 5
    _run(_stage, _h, _w, _t, _dtype, _n_stages)
    # ru_maxrss is KB on Linux, bytes on macOS -- this repo's CI/dev target is Linux.
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
