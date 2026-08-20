import numpy as np
import pytest

pytest.importorskip("graft")

from orbitapp.io import is_memmap, load_movie  # noqa: E402
from orbit.roi_extraction_graft import graft_source_extraction, patch_graft_source_extraction  # noqa: E402


def _synthetic_cell_movie(height=30, width=30, n_frames=150, seed=0):
    """A movie with two well-separated "cell" blobs against noisy
    background -- same shape/contrast this session's CNMF/PCA-ICA tests
    already use, for comparable convergence difficulty."""
    rng = np.random.default_rng(seed)
    movie = rng.standard_normal((height, width, n_frames)).astype(np.float64) * 0.1 + 1.0
    movie[5:10, 5:10, :] += 3 * np.clip(rng.standard_normal(n_frames), 0, None)
    movie[20:25, 20:25, :] += 3 * np.clip(rng.standard_normal(n_frames), 0, None)
    return np.clip(movie, 0, None)


def _near_both_blobs(masks) -> bool:
    centers = [np.argwhere(mask).mean(axis=0) for mask in masks if mask.any()]
    near_a = any(np.hypot(cy - 7.5, cx - 7.5) < 4 for cy, cx in centers)
    near_b = any(np.hypot(cy - 22.5, cx - 22.5) < 4 for cy, cx in centers)
    return near_a and near_b


def test_graft_source_extraction_recovers_both_synthetic_blobs():
    movie = _synthetic_cell_movie()

    # n_dict/max_learn chosen generously (not GraFT's own defaults) --
    # confirmed empirically to converge reliably (5/5 trials) on this
    # small synthetic movie; GraFT's compiled OpenMP solver has some
    # run-to-run floating-point non-determinism even with a fixed rng
    # seed, so this needs real headroom, not just-barely-enough iterations.
    result = graft_source_extraction(movie, n_dict=10, max_learn=400, rng=np.random.default_rng(0))

    assert len(result.masks) == len(result.traces)
    assert len(result.masks) >= 2
    assert _near_both_blobs(result.masks)


def test_graft_source_extraction_traces_are_measured_not_the_model_reconstruction():
    # ROI.trace should be the actual masked-mean signal from the movie
    # (matching every other extraction method's convention), not GraFT's
    # own learned temporal dictionary column -- a model fit, not the
    # measurement.
    movie = _synthetic_cell_movie()
    result = graft_source_extraction(movie, n_dict=10, max_learn=400, rng=np.random.default_rng(0))

    for mask, trace in zip(result.masks, result.traces):
        rows, cols = np.nonzero(mask)
        expected = movie[rows, cols, :].mean(axis=0)
        assert np.allclose(trace, expected)


def test_graft_source_extraction_masks_are_boolean_not_continuous():
    movie = _synthetic_cell_movie()
    result = graft_source_extraction(movie, n_dict=10, max_learn=400, rng=np.random.default_rng(0))
    for mask in result.masks:
        assert mask.dtype == bool


def test_patch_graft_source_extraction_recovers_both_synthetic_blobs():
    movie = _synthetic_cell_movie()

    result = patch_graft_source_extraction(
        movie, patch_size=(20, 20), overlap=(6, 6), n_dict_per_patch=5, max_learn=300, rng=np.random.default_rng(0),
    )

    assert len(result.masks) == len(result.traces)
    assert _near_both_blobs(result.masks)


def test_patch_graft_source_extraction_traces_are_measured(tmp_path):
    movie = _synthetic_cell_movie()
    result = patch_graft_source_extraction(
        movie, patch_size=(20, 20), overlap=(6, 6), n_dict_per_patch=5, max_learn=300, rng=np.random.default_rng(0),
    )
    for mask, trace in zip(result.masks, result.traces):
        rows, cols = np.nonzero(mask)
        expected = movie[rows, cols, :].mean(axis=0)
        assert np.allclose(trace, expected)


def test_patch_graft_source_extraction_stays_correct_against_a_real_memmap(tmp_path):
    movie = _synthetic_cell_movie(height=40, width=40)
    path = tmp_path / "movie.npy"
    np.save(path, movie.astype(np.float32))
    mmap_movie = load_movie(path, mmap=True)
    assert is_memmap(mmap_movie)

    result = patch_graft_source_extraction(
        mmap_movie, patch_size=(20, 20), overlap=(6, 6), n_dict_per_patch=5, max_learn=300,
        rng=np.random.default_rng(0),
    )

    assert len(result.masks) > 0
    assert _near_both_blobs(result.masks)
