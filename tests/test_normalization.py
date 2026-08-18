import numpy as np

from orbit.normalization import normalize_movie, robust_std, summary_stats


def test_robust_std_matches_std_for_normal_data():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000)
    assert np.isclose(robust_std(x), x.std(), rtol=0.1)


def test_robust_std_resistant_to_outliers():
    x = np.concatenate([np.zeros(99), [1000.0]])
    assert robust_std(x) < 1.0  # a plain std would be blown up by the outlier


def test_normalize_movie_min_centering_makes_minimum_zero():
    movie = np.array([[[1.0, 2.0, 3.0]]])
    result = normalize_movie(movie, center=True, normalize=False, center_baseline="min")
    assert np.isclose(result.min(), 0.0)


def test_normalize_movie_pixelwise_min_centering():
    movie = np.zeros((2, 1, 3))
    movie[0, 0, :] = [1, 2, 3]
    movie[1, 0, :] = [10, 20, 30]
    result = normalize_movie(movie, center=True, normalize=False, center_baseline="min", pixel_center=True)
    assert np.allclose(result[0, 0, :], [0, 1, 2])
    assert np.allclose(result[1, 0, :], [0, 10, 20])


def test_normalize_movie_mode_baseline_matches_mode_projection():
    from orbit.projections import mode_projection

    rng = np.random.default_rng(1)
    movie = rng.random((3, 3, 20))
    result = normalize_movie(movie, center=True, normalize=False, center_baseline="mode", pixel_center=True)
    expected = movie - mode_projection(movie)[:, :, None]
    assert np.allclose(result, expected)


def test_normalize_movie_normalization_divides_by_baseline():
    movie = np.full((1, 1, 4), 2.0)
    result = normalize_movie(movie, center=False, normalize=True, norm_baseline="mean")
    assert np.allclose(result, 1.0)


def test_normalize_movie_zero_baseline_is_a_no_op_not_a_divide_by_zero():
    movie = np.zeros((2, 2, 5))
    result = normalize_movie(movie, center=False, normalize=True, norm_baseline="max")
    assert np.all(np.isfinite(result))
    assert np.allclose(result, 0.0)


def test_normalize_movie_no_nans_in_output():
    rng = np.random.default_rng(2)
    movie = rng.random((4, 4, 10))
    result = normalize_movie(movie)
    assert not np.any(np.isnan(result))


def test_summary_stats_reports_expected_keys_and_values():
    movie = np.array([[[1.0, 2.0, 3.0, 4.0]]])
    stats = summary_stats(movie)
    assert stats["min"] == 1.0
    assert stats["max"] == 4.0
    assert stats["mean"] == 2.5
    assert np.isclose(stats["std"], movie.std())
