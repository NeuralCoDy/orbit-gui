import numpy as np

from orbit.projections import local_correlation_projection
from orbit.roi_extraction_corr import find_seed_candidates, roi_from_seed


def _synthetic_cell_movie(height=30, width=30, n_frames=80, seed=0):
    """A movie with two well-separated, internally-correlated "cell" blobs
    against uncorrelated background noise."""
    rng = np.random.default_rng(seed)
    movie = rng.standard_normal((height, width, n_frames)).astype(np.float64) * 0.1 + 5.0
    movie[5:10, 5:10, :] += rng.standard_normal(n_frames)
    movie[20:25, 20:25, :] += rng.standard_normal(n_frames)
    return movie


def test_roi_from_seed_local_corr_threshold_recovers_the_seeded_blob():
    movie = _synthetic_cell_movie()
    corr_image = local_correlation_projection(movie)

    result = roi_from_seed(movie, pix_loc=(7, 7), max_dist=10, growth_method="local_corr_threshold",
                            local_corr_image=corr_image)

    assert result.mask[7, 7]  # the clicked pixel is included
    assert result.mask.sum() > 0
    assert result.trace.shape == (movie.shape[2],)
    # the mask should stay within (or very near) the blob's own footprint
    assert result.mask[5:10, 5:10].sum() >= result.mask.sum() * 0.5


def test_roi_from_seed_fixed_seed_growth_method_runs():
    movie = _synthetic_cell_movie()
    result = roi_from_seed(movie, pix_loc=(22, 22), max_dist=10, growth_method="fixed_seed", thresh=0.5)
    assert result.mask[22, 22]
    assert 0 < result.mask.sum() < movie.shape[0] * movie.shape[1]


def test_roi_from_seed_flood_fill_growth_method_runs():
    movie = _synthetic_cell_movie()
    result = roi_from_seed(movie, pix_loc=(7, 7), max_dist=10, growth_method="flood_fill", thresh=0.5)
    assert result.mask.sum() > 0


def test_roi_from_seed_auto_threshold_search_picks_a_valid_threshold():
    movie = _synthetic_cell_movie()
    corr_image = local_correlation_projection(movie)
    result = roi_from_seed(movie, pix_loc=(7, 7), max_dist=10, thresh=None,
                            growth_method="local_corr_threshold", local_corr_image=corr_image)
    assert 0.6 <= result.thresh <= 0.9


def test_roi_from_seed_rejects_threshold_above_one():
    movie = _synthetic_cell_movie(height=10, width=10, n_frames=10)
    corr_image = local_correlation_projection(movie)
    try:
        roi_from_seed(movie, pix_loc=(5, 5), thresh=1.5, growth_method="local_corr_threshold",
                       local_corr_image=corr_image)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_find_seed_candidates_finds_both_blobs_separated():
    movie = _synthetic_cell_movie()
    corr_image = local_correlation_projection(movie)

    seeds = find_seed_candidates(corr_image, n_seeds=2, min_separation_frac=0.2, min_corr=0.3)

    assert len(seeds) == 2
    (r1, c1), (r2, c2) = seeds
    assert np.hypot(r1 - r2, c1 - c2) >= 0.2 * max(movie.shape[:2])


def test_find_seed_candidates_respects_min_corr():
    movie = _synthetic_cell_movie()
    corr_image = local_correlation_projection(movie)
    seeds = find_seed_candidates(corr_image, n_seeds=10, min_corr=0.99)
    assert all(corr_image[r, c] >= 0.99 for r, c in seeds)
