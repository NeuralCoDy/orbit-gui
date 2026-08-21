import numpy as np
import pytest

from orbit.ljung_box import _default_max_lag, ljung_box_test, ljung_box_test_movie


def test_default_max_lag_follows_the_10log10_n_rule():
    assert _default_max_lag(100) == 20  # 10*log10(100) = 20
    assert _default_max_lag(1000) == 30  # 10*log10(1000) = 30
    assert _default_max_lag(3) == 2  # capped at n-1


def test_ljung_box_test_white_noise_passes():
    rng = np.random.default_rng(0)
    trace = rng.standard_normal(300)
    passed, alpha = ljung_box_test(trace)
    assert passed
    assert alpha > 0.05


def test_ljung_box_test_strong_long_range_autocorrelation_fails():
    # A slow sine wave has structure spanning many lags, well beyond
    # any small number of excluded near-zero ones.
    n = 300
    trace = np.sin(np.linspace(0, 6 * np.pi, n)) + np.random.default_rng(1).standard_normal(n) * 0.05
    passed, alpha = ljung_box_test(trace, n_exclude=5)
    assert not passed
    assert alpha < 0.05


def test_ljung_box_test_excluding_central_lags_passes_a_short_decay_only_trace():
    # AR(1) with a fast decay (like a calcium indicator's own kinetics)
    # only creates autocorrelation at the smallest few lags -- without
    # exclusion that shows up as significant, but excluding those lags
    # should reveal nothing structured beyond them.
    rng = np.random.default_rng(2)
    n = 400
    g = 0.3  # fast decay -> correlation dies off after ~1-2 lags
    trace = rng.standard_normal(n)
    for t in range(1, n):
        trace[t] += g * trace[t - 1]

    passed_unexcluded, _ = ljung_box_test(trace, n_exclude=0)
    passed_excluded, alpha_excluded = ljung_box_test(trace, n_exclude=3)

    assert not passed_unexcluded
    assert passed_excluded
    assert alpha_excluded > 0.05


def test_ljung_box_test_constant_trace_passes_trivially():
    passed, alpha = ljung_box_test(np.full(200, 5.0))
    assert passed
    assert alpha == 1.0


def test_ljung_box_test_raises_when_n_exclude_leaves_no_lags():
    with pytest.raises(ValueError):
        ljung_box_test(np.random.default_rng(0).standard_normal(50), n_exclude=100)


def test_ljung_box_test_movie_matches_per_pixel_single_trace_calls():
    rng = np.random.default_rng(3)
    height, width, n_frames = 4, 5, 250
    movie = rng.standard_normal((height, width, n_frames))
    for t in range(1, n_frames):
        movie[:, :, t] += 0.5 * movie[:, :, t - 1]
    movie[1, 2, :] = 3.0  # a constant pixel, exercised alongside real ones

    passed_map, alpha_map = ljung_box_test_movie(movie, n_exclude=2)

    assert passed_map.shape == (height, width)
    assert alpha_map.shape == (height, width)
    for i in range(height):
        for j in range(width):
            passed, alpha = ljung_box_test(movie[i, j, :], n_exclude=2)
            assert passed == passed_map[i, j]
            assert alpha == pytest.approx(alpha_map[i, j], abs=1e-10)
