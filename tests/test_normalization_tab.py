import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from orbitapp.state import AppState  # noqa: E402
from orbitapp.tabs.normalization_tab import NormalizationTab  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def _x_link_target(plot):
    linked = plot.getViewBox().state["linkedViews"][0]
    return None if linked is None else linked()


def test_before_histograms_share_one_x_axis_and_after_histograms_share_another():
    tab = NormalizationTab(AppState())
    grid = tab.hist_grid
    rows = grid.left_rows + grid.right_rows

    before_plots = [row[0] for row in rows]
    after_plots = [row[1] for row in rows]

    # every "Before" plot (past the first, which is the link target) follows
    # the first Before plot's x-axis -- not any "After" plot's.
    for plot in before_plots[1:]:
        assert _x_link_target(plot) is before_plots[0].getViewBox()

    for plot in after_plots[1:]:
        assert _x_link_target(plot) is after_plots[0].getViewBox()


def _tab_with_loaded_movie() -> tuple[AppState, NormalizationTab]:
    state = AppState()
    state.load("movie.tif", np.zeros((4, 4, 5)))
    tab = NormalizationTab(state)
    tab.on_data_loaded()
    return state, tab


def _fingerprint(tab, movie):
    return (id(movie), tuple(sorted(tab._current_fingerprint().items())))


def test_apply_with_unchanged_parameters_prompts_and_skips_if_declined(monkeypatch):
    state, tab = _tab_with_loaded_movie()
    tab._last_run = _fingerprint(tab, state.active_data())  # simulate a prior successful run

    monkeypatch.setattr("orbitapp.tabs.stage_tab.confirm_recompute", lambda *a, **k: False)
    calls = []
    monkeypatch.setattr(tab, "_start_worker", lambda movie: calls.append(movie))

    tab._apply()

    assert calls == []


def test_apply_with_unchanged_parameters_reruns_if_confirmed(monkeypatch):
    state, tab = _tab_with_loaded_movie()
    movie = state.active_data()
    tab._last_run = _fingerprint(tab, movie)

    monkeypatch.setattr("orbitapp.tabs.stage_tab.confirm_recompute", lambda *a, **k: True)
    calls = []
    monkeypatch.setattr(tab, "_start_worker", lambda m: calls.append(m))

    tab._apply()

    assert calls == [movie]


def test_apply_with_changed_parameters_does_not_prompt(monkeypatch):
    state, tab = _tab_with_loaded_movie()
    movie = state.active_data()
    tab._last_run = _fingerprint(tab, movie)
    tab.norm_baseline_combo.setCurrentText("max")  # was "robuststd" by default -- now different

    prompted = []
    monkeypatch.setattr("orbitapp.tabs.stage_tab.confirm_recompute", lambda *a, **k: prompted.append(1) or False)
    calls = []
    monkeypatch.setattr(tab, "_start_worker", lambda m: calls.append(m))

    tab._apply()

    assert prompted == []
    assert calls == [movie]


def test_restore_params_sets_widgets_from_a_saved_fingerprint():
    _state, tab = _tab_with_loaded_movie()
    saved = dict(
        center=False, center_baseline="median", pixel_center=False,
        normalize=False, norm_baseline="max", pixel_norm=False,
    )

    tab.restore_params(saved)

    assert tab._current_fingerprint() == saved


def test_extract_metrics_returns_before_after_stats():
    _state, tab = _tab_with_loaded_movie()
    result = {"stats_before": {"min": 0.0}, "stats_after": {"min": 1.0}, "normalized": None, "qc_traces": []}

    metrics = tab._extract_metrics(result)

    assert metrics == {"stats_before": {"min": 0.0}, "stats_after": {"min": 1.0}}
