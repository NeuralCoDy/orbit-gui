import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from orbitapp.io import is_memmap  # noqa: E402
from orbitapp.state import AppState  # noqa: E402
from orbitapp.tabs.load_tab import LoadTab  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def test_view_movie_button_disabled_until_data_loaded():
    state = AppState()
    tab = LoadTab(state)
    assert not tab.view_movie_btn.isEnabled()


def test_view_movie_button_enabled_and_pops_out_a_player_after_loading(tmp_path):
    state = AppState()
    tab = LoadTab(state)
    movie_path = tmp_path / "test.tif"
    movie_path.touch()
    tab._pending_path = str(movie_path)
    movie = np.zeros((4, 4, 5))
    tab._on_loaded(movie)

    assert tab.view_movie_btn.isEnabled()

    tab._on_view_movie_clicked()
    assert tab._movie_player is not None
    assert "Loaded Movie" in tab._movie_player.windowTitle()


def test_view_movie_click_before_loading_is_a_noop():
    state = AppState()
    tab = LoadTab(state)
    tab._on_view_movie_clicked()  # no data loaded yet -- must not raise
    assert tab._movie_player is None


def test_loading_the_same_path_again_prompts_and_skips_if_declined(tmp_path, monkeypatch):
    state = AppState()
    movie_path = tmp_path / "test.tif"
    movie_path.touch()
    state.load(str(movie_path), np.zeros((4, 4, 5)))
    tab = LoadTab(state)

    monkeypatch.setattr("orbitapp.tabs.load_tab.confirm_recompute", lambda *a, **k: False)
    calls = []
    monkeypatch.setattr(tab, "_start_load", lambda *a, **k: calls.append(a))

    tab._load(str(movie_path))

    assert calls == []


def test_loading_the_same_path_again_proceeds_if_confirmed(tmp_path, monkeypatch):
    state = AppState()
    movie_path = tmp_path / "test.tif"
    movie_path.touch()
    state.load(str(movie_path), np.zeros((4, 4, 5)))
    tab = LoadTab(state)

    monkeypatch.setattr("orbitapp.tabs.load_tab.confirm_recompute", lambda *a, **k: True)
    calls = []
    monkeypatch.setattr(tab, "_start_load", lambda *a, **k: calls.append(a))

    tab._load(str(movie_path))

    assert len(calls) == 1


def test_mmap_checkbox_off_by_default():
    state = AppState()
    tab = LoadTab(state)
    assert not tab.mmap_check.isChecked()


def test_mmap_checkbox_threads_through_to_load_movie(tmp_path, monkeypatch):
    state = AppState()
    tab = LoadTab(state)
    tab.mmap_check.setChecked(True)

    calls = []
    monkeypatch.setattr(tab, "_start_load", lambda *a, **k: calls.append((a, k)))

    tab._load(str(tmp_path / "movie.tif"))

    assert len(calls) == 1
    _args, kwargs = calls[0]
    assert kwargs == {"mmap": True}


def test_loaded_memmap_movie_is_capped_to_5000_frames_in_the_viewer(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    from orbitapp.io import load_movie

    movie_path = tmp_path / "movie.tif"
    frames = np.zeros((5200, 4, 4), dtype="uint16")  # (T, H, W), more than the 5000-frame preview cap
    tifffile.imwrite(movie_path, frames)

    state = AppState()
    tab = LoadTab(state)
    tab._pending_path = str(movie_path)
    movie = load_movie(movie_path, mmap=True)
    assert is_memmap(movie) and movie.shape[-1] == 5200

    tab._on_loaded(movie)

    assert tab.movie_view.state.movie.shape[-1] == 5000
    # AppState itself still holds the full (uncapped) memmap -- only the viewer is capped
    assert state.original_data.shape[-1] == 5200


def test_loading_a_different_path_does_not_prompt(tmp_path, monkeypatch):
    state = AppState()
    old_path = tmp_path / "old.tif"
    old_path.touch()
    state.load(str(old_path), np.zeros((4, 4, 5)))
    tab = LoadTab(state)

    prompted = []
    monkeypatch.setattr("orbitapp.tabs.load_tab.confirm_recompute", lambda *a, **k: prompted.append(1) or False)
    calls = []
    monkeypatch.setattr(tab, "_start_load", lambda *a, **k: calls.append(a))

    new_path = tmp_path / "new.tif"
    new_path.touch()
    tab._load(str(new_path))

    assert prompted == []
    assert len(calls) == 1
