import numpy as np

from orbit.roi_extraction_pca_ica import (
    extract_masks_from_components,
    pca_ica_source_extraction,
    spatiotemporal_ica_components,
)


def _synthetic_cell_movie(height=40, width=40, n_frames=200, seed=0):
    """A movie with two well-separated, internally-correlated "cell"
    blobs against uncorrelated background noise."""
    rng = np.random.default_rng(seed)
    movie = rng.standard_normal((height, width, n_frames)).astype(np.float64) * 0.1 + 5.0
    movie[5:12, 5:12, :] += 3 * rng.standard_normal(n_frames)
    movie[25:32, 25:32, :] += 3 * rng.standard_normal(n_frames)
    return movie


def test_spatiotemporal_ica_components_shape():
    movie = _synthetic_cell_movie()
    components = spatiotemporal_ica_components(movie, n_pca_components=15, n_ica_components=10)
    assert components.shape == (10, 40, 40)


def test_extract_masks_from_components_finds_the_blob():
    height, width = 30, 30
    comp = np.zeros((height, width))
    comp[10:15, 10:15] = 5.0  # a single, well-separated bright blob

    masks = extract_masks_from_components(comp[None, :, :], num_std=2.0)

    assert len(masks) >= 1
    combined = np.logical_or.reduce(masks)
    assert combined[12, 12]  # blob center is covered by at least one mask


def test_extract_masks_from_components_respects_explicit_threshold():
    comp = np.zeros((20, 20))
    comp[5:10, 5:10] = 1.0
    masks_low_thresh = extract_masks_from_components(comp[None, :, :], thresh=0.1)
    masks_high_thresh = extract_masks_from_components(comp[None, :, :], thresh=0.9)
    assert sum(m.sum() for m in masks_low_thresh) >= sum(m.sum() for m in masks_high_thresh)


def test_pca_ica_source_extraction_recovers_both_synthetic_blobs():
    movie = _synthetic_cell_movie()
    result = pca_ica_source_extraction(movie, n_pca_components=15, n_ica_components=10)

    assert len(result.masks) == len(result.traces)
    assert len(result.masks) >= 2

    centers = []
    for mask in result.masks:
        ys, xs = np.nonzero(mask)
        centers.append((ys.mean(), xs.mean()))

    # one mask centered near each blob (8.5,8.5) and (28.5,28.5)
    near_a = any(np.hypot(cy - 8.5, cx - 8.5) < 3 for cy, cx in centers)
    near_b = any(np.hypot(cy - 28.5, cx - 28.5) < 3 for cy, cx in centers)
    assert near_a and near_b


def test_pca_ica_source_extraction_traces_match_movie_at_mask_pixels():
    movie = _synthetic_cell_movie()
    result = pca_ica_source_extraction(movie, n_pca_components=15, n_ica_components=10)

    for mask, trace in zip(result.masks, result.traces):
        rows, cols = np.nonzero(mask)
        expected = movie[rows, cols, :].mean(axis=0)
        assert np.allclose(trace, expected)
