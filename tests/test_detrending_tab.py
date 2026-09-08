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


def test_on_data_loaded_plots_the_raw_fov_average_trace_normalized_to_frame_0():
    movie = _drifting_movie()
    _state, tab = _tab_with_loaded_movie(movie)

    items = tab.trace_plot.listDataItems()
    assert len(items) == 1
    plotted = items[0].yData
    raw = movie.mean(axis=(0, 1))
    assert np.allclose(plotted, raw / raw[0])
    assert plotted[0] == 1.0


def test_on_data_loaded_plots_the_masked_average_when_a_mask_is_set():
    movie = _drifting_movie(height=10, width=10)
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    movie_with_junk = movie.copy()
    movie_with_junk[~mask] = 1e6  # blank background at a wildly different, non-decaying scale

    state, tab = _tab_with_loaded_movie(movie_with_junk)
    state.mask = mask
    tab.on_data_loaded()  # re-trigger the preview now that a mask is set, as a real Mask-tab commit would

    plotted = tab.trace_plot.listDataItems()[0].yData
    expected = movie[mask].mean(axis=0)
    assert np.allclose(plotted, expected / expected[0])


def test_params_dialog_has_percentile_window_and_exponential_rows():
    # 2 percentile-method rows + 2 exponential-method rows -- see
    # _update_visible_params for hiding whichever isn't the active method.
    _state, tab = _tab_with_loaded_movie()
    assert tab.params_dialog.form.rowCount() == 4


def test_method_combo_shows_only_the_selected_methods_param_rows():
    _state, tab = _tab_with_loaded_movie()

    tab.method_combo.setCurrentText("Running percentile")
    assert tab.params_dialog.form.isRowVisible(tab.percentile_spin)
    assert tab.params_dialog.form.isRowVisible(tab.window_spin)
    assert not tab.params_dialog.form.isRowVisible(tab.delta_pos_sigma_spin)
    assert not tab.params_dialog.form.isRowVisible(tab.delta_neg_sigma_spin)

    tab.method_combo.setCurrentText("Exponential decay (Huber)")
    assert not tab.params_dialog.form.isRowVisible(tab.percentile_spin)
    assert not tab.params_dialog.form.isRowVisible(tab.window_spin)
    assert tab.params_dialog.form.isRowVisible(tab.delta_pos_sigma_spin)
    assert tab.params_dialog.form.isRowVisible(tab.delta_neg_sigma_spin)


def test_current_fingerprint_reports_percentile_and_window():
    _state, tab = _tab_with_loaded_movie()
    tab.method_combo.setCurrentText("Running percentile")
    tab.percentile_spin.setValue(15.0)
    tab.window_spin.setValue(75)
    assert tab._current_fingerprint() == {"method": "percentile", "percentile": 15.0, "window": 75}


def test_current_fingerprint_reports_exponential_params():
    _state, tab = _tab_with_loaded_movie()
    tab.method_combo.setCurrentText("Exponential decay (Huber)")
    tab.delta_pos_sigma_spin.setValue(2.0)
    tab.delta_neg_sigma_spin.setValue(5.0)
    assert tab._current_fingerprint() == {"method": "exponential", "delta_pos_sigma": 2.0, "delta_neg_sigma": 5.0}


def test_restore_params_sets_widgets_from_a_saved_fingerprint():
    _state, tab = _tab_with_loaded_movie()
    saved = {"method": "percentile", "percentile": 20.0, "window": 100}

    tab.restore_params(saved)

    assert tab._current_fingerprint() == saved
    assert tab.method_combo.currentText() == "Running percentile"


def test_restore_params_round_trips_the_exponential_method():
    _state, tab = _tab_with_loaded_movie()
    saved = {"method": "exponential", "delta_pos_sigma": 1.5, "delta_neg_sigma": 4.0}

    tab.restore_params(saved)

    assert tab._current_fingerprint() == saved
    assert tab.method_combo.currentText() == "Exponential decay (Huber)"


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


def _exponentially_drifting_movie(height=10, width=10, n_frames=600, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames, dtype=np.float64)
    # A true exponential decaying toward (near) zero, not _drifting_movie's
    # linear ramp -- timescale scaled to n_frames (b = n_frames/4, decaying
    # to ~e^-4 =~ 2% by the end) rather than a fixed b, since a fixed short
    # b against a much longer trace clips to the floor below for most of
    # the recording, corrupting fit_exponential_trend's own initial-guess
    # regression (see that function's own docstring for the real failure
    # this reproduced with n_frames=8000, b=150 fixed).
    decay = np.exp(-t / (n_frames / 4.0))
    movie = (1.0 + rng.standard_normal((height, width, n_frames)) * 0.01) * decay[None, None, :]
    return np.clip(movie, 1e-4, None)


def test_apply_produces_a_candidate_and_plots_trace_and_trend_exponential_method():
    movie = _exponentially_drifting_movie()
    _state, tab = _tab_with_loaded_movie(movie)
    tab.method_combo.setCurrentText("Exponential decay (Huber)")

    tab._apply()
    _wait(tab)

    assert tab.commit_controls.commit_btn.isEnabled()
    result = tab._pending_result
    assert result["corrected"].shape == movie.shape
    assert result["trace"].shape == (movie.shape[-1],)
    assert result["trend"].shape == (movie.shape[-1],)
    assert result["a"] > 0
    assert result["b"] > 0

    items = tab.trace_plot.listDataItems()
    assert len(items) == 2  # raw + trend
    legend = tab.trace_plot.plotItem.legend
    assert legend is not None
    labels = {item[1].text for item in legend.items}
    assert any(label.startswith("Trend (exponential decay:") for label in labels)


def test_commit_updates_active_data_and_pipeline_exponential_method():
    movie = _exponentially_drifting_movie()
    state, tab = _tab_with_loaded_movie(movie)
    tab.method_combo.setCurrentText("Exponential decay (Huber)")

    tab._apply()
    _wait(tab)
    tab._commit()

    assert len(state.pipeline) == 2
    assert state.pipeline[1].startswith("Detrend (exponential decay,")
    corrected = state.active_data()
    assert corrected.shape == movie.shape
    assert corrected is not movie

    raw_trace = movie.mean(axis=(0, 1))
    corrected_trace = np.asarray(corrected).mean(axis=(0, 1))
    raw_drop = raw_trace[0] - raw_trace[-1]
    corrected_drop = abs(corrected_trace[-30:].mean() - corrected_trace[:30].mean())
    assert corrected_drop < raw_drop * 0.2


def test_commit_of_a_memmap_movie_covers_the_whole_movie_exponential_method(tmp_path):
    movie = _exponentially_drifting_movie(height=8, width=8, n_frames=8000).astype(np.float32)
    path = tmp_path / "movie.npy"
    np.save(path, movie)
    mmap_movie = load_movie(path, mmap=True)

    state = AppState()
    state.load(str(path), mmap_movie)
    tab = DetrendingTab(state)
    tab.on_data_loaded()
    tab._chunk_frames = 1000
    tab.method_combo.setCurrentText("Exponential decay (Huber)")

    tab._apply()
    _wait(tab)
    assert tab._pending_result["corrected"].shape[-1] == 5000  # capped preview

    tab._commit()
    _wait(tab)

    committed = state.active_data()
    assert is_memmap(committed)
    assert committed.shape == mmap_movie.shape  # the WHOLE movie, not just the 5000-frame preview
    assert np.all(np.isfinite(np.asarray(committed)))

    committed_trace = np.asarray(committed).mean(axis=(0, 1))
    raw_trace = movie.mean(axis=(0, 1))
    raw_drop = raw_trace[0] - raw_trace[-1]
    corrected_drop = abs(committed_trace[-200:].mean() - committed_trace[:200].mean())
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


def _drifting_volume(n_frames=200, length=6, width=6, depth=4, seed=0):
    rng = np.random.default_rng(seed)
    decay = np.linspace(1.0, 0.4, n_frames)
    movie = (rng.standard_normal((n_frames, length, width, depth)) * 0.01 + 1.0) * decay[:, None, None, None]
    return np.clip(movie, 0.01, None)


def test_volumetric_on_data_loaded_plots_the_per_volume_average_normalized_to_volume_0():
    # Regression test: this used to crash (AttributeError, then a pyqtgraph
    # ValueError trying to plot a 2D depth-projected slice as a 1D trace) --
    # see the StageTab _supports_volumetric fix this accompanies.
    movie = _drifting_volume()
    state = AppState()
    state.volumetric = True
    state.load("movie.fits", movie)
    tab = DetrendingTab(state)
    tab.on_data_loaded()

    items = tab.trace_plot.listDataItems()
    assert len(items) == 1
    plotted = items[0].yData
    raw = movie.mean(axis=(1, 2, 3))
    assert np.allclose(plotted, raw / raw[0])
    assert plotted[0] == 1.0


def test_volumetric_on_data_loaded_plots_the_masked_average_when_a_mask_is_set():
    movie = _drifting_volume()
    mask = np.zeros(movie.shape[1:], dtype=bool)
    mask[2:5, 2:5, 1:3] = True
    movie_with_junk = movie.copy()
    movie_with_junk[:, ~mask] = 1e6

    state = AppState()
    state.volumetric = True
    state.load("movie.fits", movie_with_junk)
    state.mask = mask
    tab = DetrendingTab(state)
    tab.on_data_loaded()

    plotted = tab.trace_plot.listDataItems()[0].yData
    expected = movie[:, mask].mean(axis=1)
    assert np.allclose(plotted, expected / expected[0])


def test_volumetric_apply_and_commit_produces_a_true_4d_array():
    movie = _drifting_volume()
    state = AppState()
    state.volumetric = True
    state.load("movie.fits", movie)
    tab = DetrendingTab(state)
    tab.on_data_loaded()
    tab.percentile_spin.setValue(8.0)
    tab.window_spin.setValue(30)

    tab._apply()
    _wait(tab)
    assert tab.commit_controls.commit_btn.isEnabled()
    tab._commit()
    _wait(tab)

    committed = state.active_data()
    assert committed.shape == movie.shape  # true (T, L, W, D)

    committed_trace = np.asarray(committed).mean(axis=(1, 2, 3))
    raw_trace = movie.mean(axis=(1, 2, 3))
    raw_drop = raw_trace[0] - raw_trace[-1]
    corrected_drop = abs(committed_trace[-30:].mean() - committed_trace[:30].mean())
    assert corrected_drop < raw_drop * 0.2


def test_volumetric_commit_respects_the_mask():
    movie = _drifting_volume()
    mask = np.zeros(movie.shape[1:], dtype=bool)
    mask[2:5, 2:5, 1:3] = True
    movie_with_junk = movie.copy()
    movie_with_junk[:, ~mask] = 1e6  # never decays -- would swamp an unmasked average

    state = AppState()
    state.volumetric = True
    state.load("movie.fits", movie_with_junk)
    state.mask = mask
    tab = DetrendingTab(state)
    tab.on_data_loaded()
    tab.percentile_spin.setValue(8.0)
    tab.window_spin.setValue(30)

    tab._apply()
    _wait(tab)
    tab._commit()
    _wait(tab)

    committed = np.asarray(state.active_data())
    # The masked-in region should be corrected as if the junk voxels never
    # existed -- compare against detrending the clean movie directly.
    clean_state = AppState()
    clean_state.volumetric = True
    clean_state.load("movie.fits", movie)
    clean_state.mask = mask
    clean_tab = DetrendingTab(clean_state)
    clean_tab.on_data_loaded()
    clean_tab.percentile_spin.setValue(8.0)
    clean_tab.window_spin.setValue(30)
    clean_tab._apply()
    _wait(clean_tab)
    clean_tab._commit()
    _wait(clean_tab)
    expected = np.asarray(clean_state.active_data())

    assert np.allclose(committed[:, mask], expected[:, mask], rtol=0.05)


def test_volumetric_commit_of_a_memmap_movie_produces_a_4d_fits_memmap(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    from orbitapp.volumetric_io import load_volumetric_movie

    movie = _drifting_volume(n_frames=8000).astype(np.float32)
    path = tmp_path / "movie.fits"
    fits.PrimaryHDU(data=movie).writeto(path, overwrite=True)
    mmapped = load_volumetric_movie(path, mmap=True)

    state = AppState()
    state.volumetric = True
    state.load(str(path), mmapped)
    tab = DetrendingTab(state)
    tab.on_data_loaded()
    tab._chunk_frames = 1000
    tab.percentile_spin.setValue(8.0)
    tab.window_spin.setValue(200)

    tab._apply()
    _wait(tab)
    assert tab._pending_result["corrected"].shape[0] == 5000  # capped preview

    tab._commit()
    _wait(tab)

    committed = state.active_data()
    assert is_memmap(committed)
    assert committed.shape == mmapped.shape  # the WHOLE volume series, not just the 5000-frame preview
    assert np.all(np.isfinite(np.asarray(committed)))

    committed_trace = np.asarray(committed).mean(axis=(1, 2, 3))
    raw_trace = np.asarray(mmapped).mean(axis=(1, 2, 3))
    raw_drop = raw_trace[0] - raw_trace[-1]
    corrected_drop = abs(committed_trace[-200:].mean() - committed_trace[:200].mean())
    assert corrected_drop < raw_drop * 0.2


def test_turning_volumetric_off_leaves_the_2d_detrending_path_unaffected():
    state = AppState()
    movie = _drifting_movie()
    state.load("movie.tif", movie)
    tab = DetrendingTab(state)
    tab.on_data_loaded()
    tab.percentile_spin.setValue(8.0)
    tab.window_spin.setValue(30)

    tab._apply()
    _wait(tab)
    tab._commit()
    _wait(tab)

    committed = state.active_data()
    assert committed.shape == movie.shape
    raw = movie.mean(axis=(0, 1))
    assert np.allclose(tab._pending_result["trace"], raw / raw[0])
