"""Memmap-specific NormalizationTab coverage -- separate from
test_normalization_tab.py since these need real memmapped movies on
disk rather than plain in-RAM arrays.
"""

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from orbit.normalization import apply_baselines, compute_baselines  # noqa: E402
from orbitapp.io import is_memmap, load_movie, preview_slice  # noqa: E402
from orbitapp.state import AppState  # noqa: E402
from orbitapp.tabs.normalization_tab import NormalizationTab  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def _wait(tab):
    if tab.worker is not None:
        tab.worker.wait(15000)
    for _ in range(50):
        QApplication.processEvents()


def _memmapped_movie(tmp_path, height=10, width=10, n_frames=4000, seed=0):
    rng = np.random.default_rng(seed)
    movie = (rng.standard_normal((height, width, n_frames)).astype(np.float32) * 5 + 100)
    path = tmp_path / "movie.npy"
    np.save(path, movie)
    return path, load_movie(path, mmap=True)


def test_commit_applies_the_preview_fitted_baseline_to_the_whole_movie(tmp_path):
    # n_frames > 5000 so the baseline used for commit is fit from a
    # genuinely different (smaller) sample than the full movie.
    path, movie = _memmapped_movie(tmp_path, n_frames=6000)
    state = AppState()
    state.load(str(path), movie)
    tab = NormalizationTab(state)
    tab.on_data_loaded()

    tab.center_baseline_combo.setCurrentText("mode")
    tab.norm_baseline_combo.setCurrentText("robuststd")
    tab._chunk_frames = 700
    tab._apply()
    _wait(tab)
    tab._commit()
    _wait(tab)

    committed = state.active_data()
    assert is_memmap(committed)
    assert committed.shape == movie.shape

    expected_baselines = compute_baselines(
        np.asarray(preview_slice(movie)), center=True, normalize=True,
        center_baseline="mode", norm_baseline="robuststd", pixel_center=True, pixel_norm=True,
    )
    expected = apply_baselines(np.asarray(movie), expected_baselines)
    np.testing.assert_allclose(np.asarray(committed), expected, atol=1e-4)


def test_commit_with_only_normalize_enabled_matches_preview_fitted_baseline(tmp_path):
    path, movie = _memmapped_movie(tmp_path, n_frames=6000)
    state = AppState()
    state.load(str(path), movie)
    tab = NormalizationTab(state)
    tab.on_data_loaded()

    tab.center_check.setChecked(False)
    tab.norm_baseline_combo.setCurrentText("max")
    tab.pixel_norm_check.setChecked(False)  # a single global scalar baseline this time
    tab._chunk_frames = 900
    tab._apply()
    _wait(tab)
    tab._commit()
    _wait(tab)

    committed = state.active_data()
    expected_baselines = compute_baselines(
        np.asarray(preview_slice(movie)), center=False, normalize=True,
        norm_baseline="max", pixel_norm=False,
    )
    expected = apply_baselines(np.asarray(movie), expected_baselines, center=False, normalize=True)
    np.testing.assert_allclose(np.asarray(committed), expected, atol=1e-4)


def test_short_memmap_movie_commit_baseline_matches_whole_movie_exactly(tmp_path):
    # When the movie is shorter than the 5000-frame preview cap, the
    # "fitted from preview" baseline IS the true whole-movie baseline --
    # no approximation gap at all in this case.
    path, movie = _memmapped_movie(tmp_path, n_frames=200)
    state = AppState()
    state.load(str(path), movie)
    tab = NormalizationTab(state)
    tab.on_data_loaded()
    tab._chunk_frames = 40
    tab._apply()
    _wait(tab)
    tab._commit()
    _wait(tab)

    committed = state.active_data()
    from orbit.normalization import normalize_movie

    expected = normalize_movie(np.asarray(movie), **tab._kwargs())
    np.testing.assert_allclose(np.asarray(committed), expected, atol=1e-4)
