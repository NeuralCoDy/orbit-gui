"""Tests for StageTab's shared Apply/Commit skeleton, including the
memmap-aware chunked-commit path -- exercised through a minimal
concrete subclass (StageTab itself is abstract) rather than through any
specific stage, since this logic is shared by all of them.
"""

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QHBoxLayout  # noqa: E402

from orbitapp.fits_io import create_fits_memmap  # noqa: E402
from orbitapp.io import is_memmap, load_movie  # noqa: E402
from orbitapp.state import AppState  # noqa: E402
from orbitapp.tabs.stage_tab import StageTab  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


class _DoubleTab(StageTab):
    """Doubles every value -- simplest possible algorithm, just enough
    to exercise Apply/Commit/chunked-Commit plumbing."""

    _stage_name = "Double"
    _result_key = "doubled"
    _stage_key = "double"
    _chunk_frames = 4  # small, so a short test movie still spans multiple chunks

    def _build_controls_row(self) -> QHBoxLayout:
        return QHBoxLayout()

    def _build_metrics(self) -> None:
        pass

    def _current_fingerprint(self) -> dict:
        return {}

    def restore_params(self, params: dict) -> None:
        pass

    def _start_worker(self, movie: np.ndarray) -> None:
        self._pending_step_label = "Double"
        self._on_finished({self._result_key: movie * 2})

    def _render_result(self, result: dict) -> None:
        pass

    def _chunked_commit(self, source, output_path):
        output = create_fits_memmap(output_path, source.shape, np.float32)
        for t0 in range(0, source.shape[-1], self._chunk_frames):
            t1 = min(t0 + self._chunk_frames, source.shape[-1])
            output[:, :, t0:t1] = np.asarray(source[:, :, t0:t1], dtype=np.float32) * 2
        output.flush()
        return output


def _wait(tab):
    if tab.worker is not None:
        tab.worker.wait(10000)
    for _ in range(50):
        QApplication.processEvents()


def test_apply_and_commit_in_ram_movie_unchanged_from_before(tmp_path):
    state = AppState()
    movie = np.arange(2 * 2 * 6, dtype=float).reshape(2, 2, 6)
    state.load("movie.tif", movie)
    tab = _DoubleTab(state, apply_label="Apply")
    tab.on_data_loaded()

    tab._apply()
    assert np.array_equal(tab._pending_result["doubled"], movie * 2)

    tab._commit()
    assert np.array_equal(state.active_data(), movie * 2)
    assert state.pipeline == ["Load", "Double"]


def test_apply_previews_only_the_first_5000_frames_of_a_long_memmap(tmp_path):
    movie = np.arange(2 * 2 * 12000, dtype=float).reshape(2, 2, 12000)
    path = tmp_path / "movie.npy"
    np.save(path, movie)
    mmap_movie = load_movie(path, mmap=True)
    assert is_memmap(mmap_movie) and mmap_movie.shape[-1] == 12000

    state = AppState()
    state.load(str(path), mmap_movie)
    tab = _DoubleTab(state, apply_label="Apply")
    tab.on_data_loaded()

    tab._apply()

    assert tab._pending_result["doubled"].shape[-1] == 5000
    assert np.array_equal(tab._pending_result["doubled"], mmap_movie[:, :, :5000] * 2)
    # Apply's own preview stays bounded, but _input_movie (used by Commit) is the real, full movie
    assert tab._input_movie.shape[-1] == 12000
    assert is_memmap(tab._input_movie)


def test_commit_of_a_memmap_input_runs_the_chunked_path_over_the_whole_movie(tmp_path):
    movie = np.arange(2 * 2 * 30, dtype=float).reshape(2, 2, 30)
    path = tmp_path / "movie.npy"
    np.save(path, movie)
    mmap_movie = load_movie(path, mmap=True)

    state = AppState()
    state.load(str(path), mmap_movie)
    tab = _DoubleTab(state, apply_label="Apply")
    tab.on_data_loaded()

    tab._apply()
    tab._commit()
    _wait(tab)

    committed = state.active_data()
    assert is_memmap(committed)
    assert committed.shape == movie.shape  # whole movie, not just the 5000-frame preview
    assert np.allclose(np.asarray(committed), movie * 2)
    assert state.pipeline == ["Load", "Double"]


def test_commit_of_a_memmap_input_disables_buttons_during_the_chunked_worker_then_reenables(tmp_path):
    movie = np.arange(2 * 2 * 20, dtype=float).reshape(2, 2, 20)
    path = tmp_path / "movie.npy"
    np.save(path, movie)
    mmap_movie = load_movie(path, mmap=True)

    state = AppState()
    state.load(str(path), mmap_movie)
    tab = _DoubleTab(state, apply_label="Apply")
    tab.on_data_loaded()
    tab._apply()

    tab._commit()
    assert not tab.commit_controls.apply_btn.isEnabled()
    assert not tab.commit_controls.commit_btn.isEnabled()
    _wait(tab)

    assert tab.commit_controls.apply_btn.isEnabled()
    assert not tab.commit_controls.commit_btn.isEnabled()  # already committed, nothing pending


def test_chunked_commit_failure_shows_a_dialog_and_leaves_commit_retryable(tmp_path, monkeypatch):
    monkeypatch.setattr("orbitapp.tabs.stage_tab.QMessageBox.critical", lambda *a, **k: None)

    movie = np.arange(2 * 2 * 10, dtype=float).reshape(2, 2, 10)
    path = tmp_path / "movie.npy"
    np.save(path, movie)
    mmap_movie = load_movie(path, mmap=True)

    state = AppState()
    state.load(str(path), mmap_movie)

    class _BrokenTab(_DoubleTab):
        def _chunked_commit(self, source, output_path):
            raise RuntimeError("boom")

    tab = _BrokenTab(state, apply_label="Apply")
    tab.on_data_loaded()
    tab._apply()

    tab._commit()
    _wait(tab)

    assert state.active_data() is mmap_movie  # nothing got committed
    assert tab.commit_controls.apply_btn.isEnabled()
    assert tab.commit_controls.commit_btn.isEnabled()  # candidate is still there to retry


def test_stage_without_a_chunked_commit_override_fails_cleanly_for_memmap_input(tmp_path, monkeypatch):
    monkeypatch.setattr("orbitapp.tabs.stage_tab.QMessageBox.critical", lambda *a, **k: None)

    movie = np.arange(2 * 2 * 10, dtype=float).reshape(2, 2, 10)
    path = tmp_path / "movie.npy"
    np.save(path, movie)
    mmap_movie = load_movie(path, mmap=True)

    state = AppState()
    state.load(str(path), mmap_movie)

    class _NoChunkTab(_DoubleTab):
        _chunked_commit = StageTab._chunked_commit  # explicitly fall back to the NotImplementedError default

    tab = _NoChunkTab(state, apply_label="Apply")
    tab.on_data_loaded()
    tab._apply()

    tab._commit()
    _wait(tab)

    assert state.active_data() is mmap_movie  # nothing got committed
