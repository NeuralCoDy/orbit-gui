import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

from orbitapp.io import is_memmap  # noqa: E402
from orbitapp.state import AppState  # noqa: E402
from orbitapp.tabs.load_tab import LoadTab  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def _wait(tab):
    if tab.worker is not None:
        tab.worker.wait(15000)
    for _ in range(50):
        QApplication.processEvents()


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


def test_modality_toggles_off_by_default():
    state = AppState()
    tab = LoadTab(state)
    assert not any(check.isChecked() for check in tab._modality_checks.values())
    assert state.modality_modifiers() == []


def test_modality_toggle_updates_state_and_emits_modality_changed():
    state = AppState()
    tab = LoadTab(state)

    emitted = []
    tab.modality_changed.connect(lambda: emitted.append(1))

    tab._modality_checks["widefield"].setChecked(True)

    assert state.widefield
    assert state.modality_modifiers() == ["widefield"]
    assert len(emitted) == 1


def test_modality_modifiers_reflects_fixed_display_order_regardless_of_toggle_order():
    state = AppState()
    tab = LoadTab(state)

    tab._modality_checks["volumetric"].setChecked(True)
    tab._modality_checks["dendrites"].setChecked(True)

    assert state.modality_modifiers() == ["dendrites", "volumetric"]


def test_untoggling_a_modality_removes_it():
    state = AppState()
    tab = LoadTab(state)
    tab._modality_checks["axons"].setChecked(True)
    assert state.modality_modifiers() == ["axons"]

    tab._modality_checks["axons"].setChecked(False)
    assert state.modality_modifiers() == []


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


def test_browse_folder_uses_the_regular_path_when_volumetric_is_off(tmp_path, monkeypatch):
    state = AppState()
    tab = LoadTab(state)
    monkeypatch.setattr("orbitapp.tabs.load_tab.QFileDialog.getExistingDirectory", lambda *a, **k: str(tmp_path))
    calls = []
    monkeypatch.setattr(tab, "_load", lambda path: calls.append(("regular", path)))
    monkeypatch.setattr(tab, "_load_volumetric_folder", lambda path: calls.append(("volumetric", path)))

    tab._browse_folder()

    assert calls == [("regular", str(tmp_path))]


def test_browse_folder_uses_the_volumetric_path_when_volumetric_is_on(tmp_path, monkeypatch):
    state = AppState()
    tab = LoadTab(state)
    tab._modality_checks["volumetric"].setChecked(True)
    monkeypatch.setattr("orbitapp.tabs.load_tab.QFileDialog.getExistingDirectory", lambda *a, **k: str(tmp_path))
    calls = []
    monkeypatch.setattr(tab, "_load", lambda path: calls.append(("regular", path)))
    monkeypatch.setattr(tab, "_load_volumetric_folder", lambda path: calls.append(("volumetric", path)))

    tab._browse_folder()

    assert calls == [("volumetric", str(tmp_path))]


def test_browse_folder_no_selection_does_nothing(monkeypatch):
    state = AppState()
    tab = LoadTab(state)
    monkeypatch.setattr("orbitapp.tabs.load_tab.QFileDialog.getExistingDirectory", lambda *a, **k: "")
    calls = []
    monkeypatch.setattr(tab, "_load", lambda path: calls.append(path))
    monkeypatch.setattr(tab, "_load_volumetric_folder", lambda path: calls.append(path))

    tab._browse_folder()

    assert calls == []


def test_volumetric_folder_load_cancelled_dialog_does_nothing(tmp_path, monkeypatch):
    state = AppState()
    tab = LoadTab(state)
    monkeypatch.setattr(
        "orbitapp.tabs.load_tab.VolumetricLoadDialog.exec", lambda self: QDialog.DialogCode.Rejected
    )

    tab._load_volumetric_folder(str(tmp_path))

    assert tab.worker is None
    assert state.original_data is None


def test_volumetric_folder_load_end_to_end(tmp_path, monkeypatch):
    tifffile = pytest.importorskip("tifffile")
    T, D, L, W = 2, 3, 4, 5
    volumes = np.arange(T * D * L * W, dtype=np.float32).reshape(T, D, L, W)
    for t in range(T):
        tifffile.imwrite(tmp_path / f"vol_{t:03d}.tif", volumes[t])

    state = AppState()
    tab = LoadTab(state)
    monkeypatch.setattr(
        "orbitapp.tabs.load_tab.VolumetricLoadDialog.exec", lambda self: QDialog.DialogCode.Accepted
    )
    # default dialog selection is "one volume per stack" -- no need to touch mode/depth

    loaded = []
    tab.data_loaded.connect(lambda: loaded.append(1))

    tab._load_volumetric_folder(str(tmp_path))
    _wait(tab)

    assert loaded == [1]
    assert state.original_data.shape == (T, L, W, D)
    assert not tab.view_movie_btn.isEnabled()  # no volumetric viewer yet
    assert "volumetric" in tab.info_label.text().lower()


def test_volumetric_folder_load_of_the_same_path_prompts_and_skips_if_declined(tmp_path, monkeypatch):
    tifffile = pytest.importorskip("tifffile")
    tifffile.imwrite(tmp_path / "vol_000.tif", np.zeros((3, 4, 5), dtype=np.float32))

    state = AppState()
    state.load(str(tmp_path), np.zeros((1, 4, 5, 3)))
    tab = LoadTab(state)
    monkeypatch.setattr(
        "orbitapp.tabs.load_tab.VolumetricLoadDialog.exec", lambda self: QDialog.DialogCode.Accepted
    )
    monkeypatch.setattr("orbitapp.tabs.load_tab.confirm_recompute", lambda *a, **k: False)

    tab._load_volumetric_folder(str(tmp_path))

    assert tab.worker is None
