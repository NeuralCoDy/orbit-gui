import numpy as np

from orbit.motion_metrics import (
    enhanced_correlation_coefficient,
    mean_correlation_to_reference,
    mean_max_intensity_difference,
    spatiotemporal_svd,
)


def test_mean_max_intensity_difference_is_negative_when_after_is_dimmer():
    rng = np.random.default_rng(0)
    before = rng.random((10, 10, 100)) + 1.0  # includes a bright-motion-artifact-like spread
    after = before * 0.5  # uniformly dimmer max projection

    mmd = mean_max_intensity_difference(before, after, bin_size=10)

    assert mmd < 0


def test_mean_max_intensity_difference_zero_for_identical_movies():
    rng = np.random.default_rng(1)
    movie = rng.random((8, 8, 50))
    assert mean_max_intensity_difference(movie, movie, bin_size=10) == 0.0


def test_enhanced_correlation_coefficient_is_one_for_identical_images():
    rng = np.random.default_rng(2)
    image = rng.random((12, 12))
    assert np.isclose(enhanced_correlation_coefficient(image, image), 1.0)


def test_enhanced_correlation_coefficient_invariant_to_bias_and_gain():
    rng = np.random.default_rng(3)
    image = rng.random((12, 12))
    scaled_biased = image * 3.7 + 5.0

    assert np.isclose(enhanced_correlation_coefficient(image, scaled_biased), 1.0)


def test_enhanced_correlation_coefficient_low_for_unrelated_images():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((30, 30))
    b = rng.standard_normal((30, 30))
    assert abs(enhanced_correlation_coefficient(a, b)) < 0.3


def test_mean_correlation_to_reference_defaults_to_self_mean():
    rng = np.random.default_rng(5)
    movie = rng.random((10, 10, 40))
    score = mean_correlation_to_reference(movie)
    assert -1.0 <= score <= 1.0


def test_mean_correlation_to_reference_high_for_static_movie():
    rng = np.random.default_rng(6)
    frame = rng.random((10, 10))
    movie = np.stack([frame + 0.01 * rng.standard_normal((10, 10)) for _ in range(20)], axis=-1)

    score = mean_correlation_to_reference(movie)

    assert score > 0.9


def test_mean_correlation_to_reference_with_explicit_reference():
    rng = np.random.default_rng(7)
    reference = rng.random((10, 10))
    movie = np.stack([reference for _ in range(15)], axis=-1)

    score = mean_correlation_to_reference(movie, reference=reference)

    assert np.isclose(score, 1.0)


def test_spatiotemporal_svd_shapes():
    rng = np.random.default_rng(8)
    movie = rng.random((16, 12, 25))

    singular_values, pc_maps = spatiotemporal_svd(movie, n_components=5)

    assert singular_values.shape == (5,)
    assert pc_maps.shape == (16, 12, 5)


def test_spatiotemporal_svd_singular_values_descending():
    rng = np.random.default_rng(9)
    movie = rng.random((20, 20, 40))

    singular_values, _ = spatiotemporal_svd(movie, n_components=10)

    assert np.all(np.diff(singular_values) <= 0)


def test_spatiotemporal_svd_concentrates_for_low_rank_movie():
    # A movie built from a single spatial pattern modulated over time is
    # exactly rank-1 -- the spectrum should be maximally "tight" (one
    # dominant singular value).
    rng = np.random.default_rng(10)
    spatial = rng.random((15, 15))
    time_course = rng.random(30)
    movie = spatial[:, :, None] * time_course[None, None, :]

    singular_values, _ = spatiotemporal_svd(movie, n_components=5)

    assert singular_values[0] > 0
    assert np.allclose(singular_values[1:], 0.0, atol=1e-8)
