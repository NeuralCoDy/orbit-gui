import numpy as np
import pytest

import orbit._native as _native
from orbit.projections import (
    fano_factor_projection,
    local_correlation_projection,
    mean_projection,
    median_projection,
    mode_projection,
    variance_projection,
)


def _toy_movie() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((4, 5, 20))


def test_mean_projection_shape_and_value():
    movie = np.zeros((3, 3, 10))
    movie[1, 1, :] = np.arange(10)
    proj = mean_projection(movie)
    assert proj.shape == (3, 3)
    assert proj[1, 1] == np.arange(10).mean()
    assert proj[0, 0] == 0


def test_median_projection_robust_to_outlier():
    movie = np.zeros((2, 2, 5))
    movie[0, 0, :] = [1, 1, 1, 1, 100]
    proj = median_projection(movie)
    assert proj[0, 0] == 1.0


def test_variance_projection_constant_pixel_is_zero():
    movie = np.full((2, 2, 10), 5.0)
    proj = variance_projection(movie)
    assert np.allclose(proj, 0.0)


def test_fano_factor_zero_mean_pixel_is_zero_not_nan():
    movie = np.zeros((2, 2, 6))
    proj = fano_factor_projection(movie)
    assert np.all(proj == 0.0)
    assert not np.any(np.isnan(proj))


def test_fano_factor_matches_manual_ratio():
    movie = np.zeros((1, 1, 5))
    movie[0, 0, :] = [1, 2, 3, 4, 5]
    proj = fano_factor_projection(movie)
    expected = movie.var(axis=2) / movie.mean(axis=2)
    assert np.isclose(proj[0, 0], expected[0, 0])


def test_shapes_consistent_across_all_projections():
    movie = _toy_movie()
    for fn in (
        mean_projection,
        median_projection,
        variance_projection,
        fano_factor_projection,
        local_correlation_projection,
        mode_projection,
    ):
        assert fn(movie).shape == movie.shape[:2]


def test_mode_projection_constant_pixel_returns_that_value():
    movie = np.full((2, 2, 9), 3.5)
    proj = mode_projection(movie)
    assert np.allclose(proj, 3.5)


def test_mode_projection_robust_to_a_single_outlier():
    movie = np.zeros((1, 1, 7))
    movie[0, 0, :] = [1, 1, 1, 1, 1, 1, 1000]
    proj = mode_projection(movie)
    assert proj[0, 0] == 1.0


def test_mode_projection_finds_the_denser_cluster():
    # Densely clustered around 0, one sparse outlier group around 50 --
    # the mode should land in the dense cluster, unlike the mean.
    rng = np.random.default_rng(5)
    cluster = rng.normal(0, 0.1, size=17)
    outliers = np.array([50.0, 51.0, 52.0])
    trace = np.concatenate([cluster, outliers])
    movie = trace.reshape(1, 1, -1)

    proj = mode_projection(movie)

    assert abs(proj[0, 0]) < 1.0


def test_local_correlation_is_one_when_all_pixels_share_the_same_trace():
    rng = np.random.default_rng(1)
    trace = rng.random(15)
    movie = np.broadcast_to(trace, (4, 4, 15)).copy()

    proj = local_correlation_projection(movie)

    assert np.allclose(proj, 1.0, atol=1e-5)  # float32 internally, see projections.py


def test_local_correlation_is_low_for_independent_pixel_noise():
    rng = np.random.default_rng(2)
    movie = rng.standard_normal((20, 20, 50))

    proj = local_correlation_projection(movie)

    assert np.all(proj >= -1.0 - 1e-8) and np.all(proj <= 1.0 + 1e-8)
    assert np.abs(proj.mean()) < 0.2


def test_local_correlation_finite_at_image_border():
    rng = np.random.default_rng(3)
    movie = rng.random((5, 5, 10))

    proj = local_correlation_projection(movie)

    assert np.all(np.isfinite(proj))


@pytest.mark.skipif(not _native.NATIVE_AVAILABLE, reason="native extension not built -- run orbit/_native/build_native.sh")
def test_local_correlation_native_matches_numpy_fallback(monkeypatch):
    rng = np.random.default_rng(4)
    movie = rng.standard_normal((12, 10, 30)).astype(np.float32)

    native_result = local_correlation_projection(movie)

    monkeypatch.setattr(_native, "NATIVE_AVAILABLE", False)
    numpy_result = local_correlation_projection(movie)

    assert np.allclose(native_result, numpy_result, atol=1e-4)


@pytest.mark.skipif(not _native.NATIVE_AVAILABLE, reason="native extension not built -- run orbit/_native/build_native.sh")
def test_mode_native_matches_numpy_fallback(monkeypatch):
    rng = np.random.default_rng(6)
    movie = rng.standard_normal((10, 8, 25)).astype(np.float32)

    native_result = mode_projection(movie)

    monkeypatch.setattr(_native, "NATIVE_AVAILABLE", False)
    numpy_result = mode_projection(movie)

    assert np.allclose(native_result, numpy_result, atol=1e-4)
