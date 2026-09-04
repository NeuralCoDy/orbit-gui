import numpy as np

from orbit.normalization import describe_normalization, normalize_movie, pixel_value_histogram, robust_std, summary_stats


def test_robust_std_matches_std_for_normal_data():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000)
    assert np.isclose(robust_std(x), x.std(), rtol=0.1)


def test_robust_std_resistant_to_outliers():
    x = np.concatenate([np.zeros(99), [1000.0]])
    assert robust_std(x) < 1.0  # a plain std would be blown up by the outlier


def test_robust_std_accepts_a_multi_axis_tuple():
    # orbit.denoising's wavelet shrinkage calls robust_std with a
    # multi-axis tuple (e.g. axis=(-2, -1), reducing one subband's two
    # spatial axes at once, keeping the batch/frame axis) -- the chunked
    # single-axis fast path must not be taken here, since np.moveaxis
    # can't map a 2-element axis tuple onto a single destination axis.
    rng = np.random.default_rng(0)
    x = rng.random((5, 8, 6)) * 100

    got = robust_std(x, axis=(-2, -1), keepdims=True)

    centered = x - np.median(x, axis=(-2, -1), keepdims=True)
    want = np.median(np.abs(centered), axis=(-2, -1), keepdims=True) / 0.6741891400433162
    assert got.shape == (5, 1, 1)
    np.testing.assert_allclose(got, want, rtol=1e-10)


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


def test_normalize_movie_pixelwise_median_centering_matches_plain_numpy_across_a_chunk_boundary():
    # H=340 spans multiple chunked_median blocks (150-pixel cap) -- confirms
    # the chunked pixel-wise median baseline matches an unchunked np.median.
    rng = np.random.default_rng(2)
    movie = rng.random((340, 3, 11)) * 100
    result = normalize_movie(movie, center=True, normalize=False, center_baseline="median", pixel_center=True)
    expected = movie - np.median(movie, axis=2, keepdims=True)
    np.testing.assert_allclose(result, expected)


def test_robust_std_pixelwise_matches_unchunked_formula_across_a_chunk_boundary():
    rng = np.random.default_rng(3)
    movie = rng.random((340, 3, 11)) * 100

    got = robust_std(movie, axis=2, keepdims=True)

    centered = movie - np.median(movie, axis=2, keepdims=True)
    want = np.median(np.abs(centered), axis=2, keepdims=True) / 0.6741891400433162
    np.testing.assert_allclose(got, want, rtol=1e-10)


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


def test_pixel_value_histogram_reports_correct_mean_and_median():
    trace = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    hist = pixel_value_histogram(trace)
    assert np.isclose(hist["mean"], trace.mean())
    assert np.isclose(hist["median"], np.median(trace))
    assert len(hist["edges"]) == len(hist["counts"]) + 1
    assert hist["counts"].sum() == trace.size


def test_pixel_value_histogram_constant_trace_is_a_single_bin():
    trace = np.full(10, 5.0)
    hist = pixel_value_histogram(trace)
    assert hist["mean"] == 5.0
    assert hist["median"] == 5.0
    assert hist["mode"] == 5.0
    assert hist["counts"].sum() == trace.size


def test_pixel_value_histogram_bin_count_matches_n_bins():
    rng = np.random.default_rng(9)
    trace = rng.standard_normal(500)
    hist = pixel_value_histogram(trace, n_bins=15)
    assert len(hist["counts"]) == 15


def test_describe_normalization_both_pixel_wise():
    desc = describe_normalization(
        center=True, center_baseline="mode", pixel_center=True,
        normalize=True, norm_baseline="median", pixel_norm=True,
    )
    assert desc == "(pixel-mode, pixel-median)"


def test_describe_normalization_center_disabled_normalize_global():
    desc = describe_normalization(
        center=False, center_baseline="min", pixel_center=False,
        normalize=True, norm_baseline="median", pixel_norm=False,
    )
    assert desc == "([], all-median)"


def test_describe_normalization_both_disabled():
    desc = describe_normalization(
        center=False, center_baseline="min", pixel_center=False,
        normalize=False, norm_baseline="median", pixel_norm=False,
    )
    assert desc == "([], [])"
