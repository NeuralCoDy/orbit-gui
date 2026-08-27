import numpy as np
import pytest

pytest.importorskip("graft")

from orbit.roi_extraction_graft_3d import (  # noqa: E402
    GraFTResult3D,
    _merge_overlapping_masks_3d,
    graft_source_extraction_3d,
    patch_graft_source_extraction_3d,
)


def _synthetic_cell_volume(length=20, width=20, depth=10, n_frames=200, seed=0):
    """A volumetric movie with two well-separated "cell" blobs against
    noisy background, each with its own independent activity trace --
    same synthetic-movie idea tests_roi_extraction_graft.py already uses,
    generalized to (T, L, W, D)."""
    rng = np.random.default_rng(seed)
    movie = rng.standard_normal((n_frames, length, width, depth)).astype(np.float64) * 0.1 + 1.0

    blob1 = np.zeros((length, width, depth))
    blob1[3:7, 3:7, 2:5] = 1.0
    blob2 = np.zeros((length, width, depth))
    blob2[12:16, 12:16, 5:8] = 1.0
    act1 = np.clip(rng.standard_normal(n_frames), 0, None) * 3
    act2 = np.clip(rng.standard_normal(n_frames), 0, None) * 3
    for t in range(n_frames):
        movie[t] += blob1 * act1[t] + blob2 * act2[t]

    mask = np.zeros((length, width, depth), dtype=bool)
    mask[0 : length // 2, 0 : width // 2, :] = True
    mask[length // 2 :, width // 2 :, :] = True
    return movie, mask, act1, act2


def _near_both_blobs(masks, act1, act2, traces) -> bool:
    near_1 = any(np.corrcoef(trace, act1)[0, 1] > 0.9 for trace in traces)
    near_2 = any(np.corrcoef(trace, act2)[0, 1] > 0.9 for trace in traces)
    return near_1 and near_2


def test_graft_source_extraction_3d_recovers_both_synthetic_blobs():
    movie, mask, act1, act2 = _synthetic_cell_volume()

    result = graft_source_extraction_3d(movie, mask, n_dict=6, max_learn=400, rng=np.random.default_rng(1))

    assert len(result.masks) == len(result.traces)
    assert _near_both_blobs(result.masks, act1, act2, result.traces)


def test_graft_source_extraction_3d_masks_are_boolean_and_3d():
    movie, mask, _act1, _act2 = _synthetic_cell_volume()
    result = graft_source_extraction_3d(movie, mask, n_dict=6, max_learn=400, rng=np.random.default_rng(1))
    for roi_mask in result.masks:
        assert roi_mask.dtype == bool
        assert roi_mask.shape == mask.shape


def test_graft_source_extraction_3d_traces_are_measured_not_the_model_reconstruction():
    movie, mask, _act1, _act2 = _synthetic_cell_volume()
    result = graft_source_extraction_3d(movie, mask, n_dict=6, max_learn=400, rng=np.random.default_rng(1))
    for roi_mask, trace in zip(result.masks, result.traces):
        expected = movie[:, roi_mask].mean(axis=1)
        assert np.allclose(trace, expected)


def test_graft_source_extraction_3d_raises_without_a_mask():
    movie, _mask, _act1, _act2 = _synthetic_cell_volume(n_frames=20)
    with pytest.raises(ValueError, match="requires a real"):
        graft_source_extraction_3d(movie, None)


def test_graft_source_extraction_3d_raises_for_an_all_false_mask():
    movie, mask, _act1, _act2 = _synthetic_cell_volume(n_frames=20)
    with pytest.raises(ValueError, match="requires a real"):
        graft_source_extraction_3d(movie, np.zeros_like(mask))


def test_graft_source_extraction_3d_raises_for_an_all_true_mask():
    # An explicit Clear Mask (or an auto-threshold that happened to keep
    # everything) doesn't reduce the voxel count -- the whole reason a
    # mask is required here -- so it's rejected too, not just an empty one.
    movie, mask, _act1, _act2 = _synthetic_cell_volume(n_frames=20)
    with pytest.raises(ValueError, match="requires a real"):
        graft_source_extraction_3d(movie, np.ones_like(mask))


def test_patch_graft_source_extraction_3d_recovers_both_synthetic_blobs():
    movie, mask, act1, act2 = _synthetic_cell_volume()

    result = patch_graft_source_extraction_3d(
        movie, mask, patch_size=(12, 12, 10), overlap=3, n_dict_per_patch=4, max_learn=300,
        rng=np.random.default_rng(2),
    )

    assert len(result.masks) == len(result.traces)
    assert _near_both_blobs(result.masks, act1, act2, result.traces)


def test_patch_graft_source_extraction_3d_traces_are_measured(tmp_path):
    movie, mask, _act1, _act2 = _synthetic_cell_volume()
    result = patch_graft_source_extraction_3d(
        movie, mask, patch_size=(12, 12, 10), overlap=3, n_dict_per_patch=4, max_learn=300,
        rng=np.random.default_rng(2),
    )
    for roi_mask, trace in zip(result.masks, result.traces):
        expected = movie[:, roi_mask].mean(axis=1)
        assert np.allclose(trace, expected)


def test_patch_graft_source_extraction_3d_raises_without_a_mask():
    movie, _mask, _act1, _act2 = _synthetic_cell_volume(n_frames=20)
    with pytest.raises(ValueError, match="requires a real"):
        patch_graft_source_extraction_3d(movie, None)


def test_patch_graft_source_extraction_3d_skips_empty_regions_without_calling_graft(monkeypatch):
    # A mask confined to one small corner -- every other patch has no
    # masked voxels and should never even reach graft.graft.
    length, width, depth, n_frames = 20, 20, 10, 30
    rng = np.random.default_rng(3)
    movie = rng.standard_normal((n_frames, length, width, depth)).astype(np.float64)
    mask = np.zeros((length, width, depth), dtype=bool)
    mask[0:4, 0:4, 0:4] = True

    calls = []
    import orbit.roi_extraction_graft_3d as mod

    real_graft = mod.graft.graft

    def _spy(*args, **kwargs):
        calls.append(1)
        return real_graft(*args, **kwargs)

    monkeypatch.setattr(mod.graft, "graft", _spy)

    result = patch_graft_source_extraction_3d(
        movie, mask, patch_size=(6, 6, 6), overlap=1, n_dict_per_patch=2, rng=np.random.default_rng(4),
    )

    assert len(calls) == 1  # only the one patch overlapping the mask ever called graft.graft
    assert isinstance(result, GraFTResult3D)


def test_merge_overlapping_masks_3d_merges_a_split_component():
    movie = np.random.default_rng(0).standard_normal((10, 6, 6, 6))
    mask_a = np.zeros((6, 6, 6), dtype=bool)
    mask_a[1:4, 1:4, 1:4] = True
    mask_b = np.zeros((6, 6, 6), dtype=bool)
    mask_b[2:5, 2:5, 2:5] = True  # overlaps mask_a
    trace = np.arange(10, dtype=float)

    merged_masks, merged_traces = _merge_overlapping_masks_3d(movie, [mask_a, mask_b], [trace, trace], merge_thresh=0.8)

    assert len(merged_masks) == 1
    np.testing.assert_array_equal(merged_masks[0], mask_a | mask_b)


def test_merge_overlapping_masks_3d_keeps_non_overlapping_masks_separate():
    movie = np.random.default_rng(0).standard_normal((10, 6, 6, 6))
    mask_a = np.zeros((6, 6, 6), dtype=bool)
    mask_a[0:2, 0:2, 0:2] = True
    mask_b = np.zeros((6, 6, 6), dtype=bool)
    mask_b[4:6, 4:6, 4:6] = True
    trace = np.arange(10, dtype=float)

    merged_masks, _merged_traces = _merge_overlapping_masks_3d(
        movie, [mask_a, mask_b], [trace, trace], merge_thresh=0.8
    )

    assert len(merged_masks) == 2
