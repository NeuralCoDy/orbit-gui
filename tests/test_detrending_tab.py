import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from orbitapp.io import is_memmap, load_movie  # noqa: E402
from orbitapp.state import AppState  # noqa: E402
from orbitapp.tabs.detrending_tab import DetrendingTab  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def _wait(tab, timeout_ms=15000):
    if tab.worker is not None:
        tab.worker.wait(timeout_ms)
    for _ in range(50):
        QApplication.processEvents()


def _drifting_movie(height=10, width=10, n_frames=300, seed=0):
    rng = np.random.default_rng(seed)
    decay = np.linspace(1.0, 0.4, n_frames)
    movie = (rng.standard_normal((height, width, n_frames)) * 0.01 + 1.0) * decay[None, None, :]
    return np.clip(movie, 0.01, None)


def _tab_with_loaded_movie(movie=None) -> tuple[AppState, DetrendingTab]:
    state = AppState()
    state.load("movie.tif", movie if movie is not None else _drifting_movie())
    tab = DetrendingTab(state)
    tab.on_data_loaded()
    return state, tab


def test_on_data_loaded_plots_the_raw_fov_average_trace():
    movie = _drifting_movie()
    _state, tab = _tab_with_loaded_movie(movie)

    items = tab.trace_plot.listDataItems()
    assert len(items) == 1
    plotted = items[0].yData
    assert np.allclose(plotted, movie.mean(axis=(0, 1)))


def test_params_dialog_has_percentile_and_window_rows():
    _state, tab = _tab_with_loaded_movie()
    assert tab.params_dialog.form.rowCount() == 2


def test_current_fingerprint_reports_percentile_and_window():
    _state, tab = _tab_with_loaded_movie()
    tab.percentile_spin.setValue(15.0)
    tab.window_spin.setValue(75)
    assert tab._current_fingerprint() == {"percentile": 15.0, "window": 75}


def test_restore_params_sets_widgets_from_a_saved_fingerprint():
    _state, tab = _tab_with_loaded_movie()
    saved = {"percentile": 20.0, "window": 100}

    tab.restore_params(saved)

    assert tab._current_fingerprint() == saved


def test_restore_params_tolerates_missing_keys():
    _state, tab = _tab_with_loaded_movie()
    tab.restore_params({"percentile": 30.0})
    assert tab.percentile_spin.value() == 30.0


def test_apply_produces_a_candidate_and_plots_trace_and_trend():
    movie = _drifting_movie()
    _state, tab = _tab_with_loaded_movie(movie)

    tab.percentile_spin.setValue(8.0)
    tab.window_spin.setValue(30)
    tab._apply()
    _wait(tab)

    assert tab.commit_controls.commit_btn.isEnabled()
    result = tab._pending_result
    assert result["corrected"].shape == movie.shape
    assert result["trace"].shape == (movie.shape[-1],)
    assert result["trend"].shape == (movie.shape[-1],)

    items = tab.trace_plot.listDataItems()
    assert len(items) == 2  # raw + trend
    legend = tab.trace_plot.plotItem.legend
    assert legend is not None
    labels = {item[1].text for item in legend.items}
    assert labels == {"Raw", "Trend (running percentile)"}


def test_apply_without_data_disables_commit(monkeypatch):
    monkeypatch.setattr("orbitapp.tabs.stage_tab.QMessageBox.warning", lambda *a, **k: None)
    state = AppState()
    tab = DetrendingTab(state)
    tab._apply()
    assert not tab.commit_controls.commit_btn.isEnabled()


def test_commit_updates_active_data_and_pipeline():
    movie = _drifting_movie()
    state, tab = _tab_with_loaded_movie(movie)

    tab.percentile_spin.setValue(8.0)
    tab.window_spin.setValue(30)
    tab._apply()
    _wait(tab)
    tab._commit()

    assert state.pipeline == ["Load", "Detrend (percentile=8, window=30)"]
    corrected = state.active_data()
    assert corrected.shape == movie.shape
    assert corrected is not movie

    # the drift should be substantially flattened
    raw_trace = movie.mean(axis=(0, 1))
    corrected_trace = np.asarray(corrected).mean(axis=(0, 1))
    raw_drop = raw_trace[0] - raw_trace[-1]
    corrected_drop = abs(corrected_trace[-30:].mean() - corrected_trace[:30].mean())
    assert corrected_drop < raw_drop * 0.2


def test_apply_with_unchanged_parameters_prompts_and_skips_if_declined(monkeypatch):
    movie = _drifting_movie()
    state, tab = _tab_with_loaded_movie(movie)
    tab._last_run = (id(state.active_data()), tuple(sorted(tab._current_fingerprint().items())))

    monkeypatch.setattr("orbitapp.tabs.stage_tab.confirm_recompute", lambda *a, **k: False)
    calls = []
    monkeypatch.setattr(tab, "_start_worker", lambda m: calls.append(m))

    tab._apply()

    assert calls == []


def test_apply_with_changed_parameters_does_not_prompt(monkeypatch):
    movie = _drifting_movie()
    state, tab = _tab_with_loaded_movie(movie)
    tab._last_run = (id(state.active_data()), tuple(sorted(tab._current_fingerprint().items())))
    tab.window_spin.setValue(tab.window_spin.value() + 10)

    prompted = []
    monkeypatch.setattr("orbitapp.tabs.stage_tab.confirm_recompute", lambda *a, **k: prompted.append(1) or False)
    calls = []
    monkeypatch.setattr(tab, "_start_worker", lambda m: calls.append(m))

    tab._apply()

    assert prompted == []
    assert calls == [movie]


def _memmapped_drifting_movie(tmp_path, height=8, width=8, n_frames=8000, seed=0):
    movie = _drifting_movie(height, width, n_frames, seed).astype(np.float32)
    path = tmp_path / "movie.npy"
    np.save(path, movie)
    return path, load_movie(path, mmap=True), movie


def test_commit_of_a_memmap_movie_covers_the_whole_movie_not_just_the_preview(tmp_path):
    path, mmap_movie, raw_movie = _memmapped_drifting_movie(tmp_path)
    state = AppState()
    state.load(str(path), mmap_movie)
    tab = DetrendingTab(state)
    tab.on_data_loaded()
    tab._chunk_frames = 1000
    tab.percentile_spin.setValue(8.0)
    tab.window_spin.setValue(200)

    tab._apply()
    _wait(tab)
    assert tab._pending_result["corrected"].shape[-1] == 5000  # capped preview

    tab._commit()
    _wait(tab)

    committed = state.active_data()
    assert is_memmap(committed)
    assert committed.shape == mmap_movie.shape  # the WHOLE movie, not just the 5000-frame preview
    assert np.all(np.isfinite(np.asarray(committed)))

    # trend covering the full movie should flatten the drift throughout,
    # including well past the 5000-frame preview cap.
    committed_trace = np.asarray(committed).mean(axis=(0, 1))
    raw_trace = raw_movie.mean(axis=(0, 1))
    raw_drop = raw_trace[0] - raw_trace[-1]
    corrected_drop = abs(committed_trace[-200:].mean() - committed_trace[:200].mean())
    assert corrected_drop < raw_drop * 0.2
