import numpy as np

from orbit.detrending import detrend_movie, fov_average_trace, running_percentile_trend


def _naive_running_percentile(trace, percentile, window):
    n = len(trace)
    out = np.empty(n)
    for t in range(n):
        lo = max(0, t - window + 1)
        out[t] = np.percentile(trace[lo : t + 1], percentile)
    return out


def test_fov_average_trace_is_mean_over_space_per_frame():
    movie = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    trace = fov_average_trace(movie)
    assert trace.shape == (4,)
    for t in range(4):
        assert np.isclose(trace[t], movie[:, :, t].mean())


def test_running_percentile_trend_matches_naive_reference_across_window_sizes():
    rng = np.random.default_rng(0)
    trace = rng.standard_normal(60)
    for window in (1, 3, 10, 30, 60, 200):  # 200 > len(trace) -- window larger than the trace
        fast = running_percentile_trend(trace, 25.0, window)
        ref = _naive_running_percentile(trace, 25.0, window)
        assert np.allclose(fast, ref)


def test_running_percentile_trend_is_trailing_not_centered():
    # A single spike at t=10 in an otherwise-zero trace should only
    # raise the trend at and after t=10 (a trailing/causal window never
    # looks ahead), not before it.
    trace = np.zeros(30)
    trace[10] = 100.0
    trend = running_percentile_trend(trace, 99.0, window=5)

    assert np.all(trend[:10] == 0.0)
    assert trend[10] > 0.0


def test_running_percentile_trend_window_one_returns_the_trace_itself():
    rng = np.random.default_rng(0)
    trace = rng.standard_normal(20)
    trend = running_percentile_trend(trace, 50.0, window=1)
    assert np.allclose(trend, trace)


def test_detrend_movie_flattens_a_synthetic_multiplicative_drift():
    rng = np.random.default_rng(0)
    height, width, n_frames = 10, 10, 400
    decay = np.linspace(1.0, 0.4, n_frames)  # synthetic photobleaching
    movie = (rng.standard_normal((height, width, n_frames)) * 0.01 + 1.0) * decay[None, None, :]
    movie = np.clip(movie, 0.01, None)

    corrected, trace, trend = detrend_movie(movie, percentile=8.0, window=30)

    assert corrected.shape == movie.shape
    assert trace.shape == (n_frames,)
    assert trend.shape == (n_frames,)

    corrected_trace = corrected.mean(axis=(0, 1))
    # The raw trace drops by more than half; the corrected trace should
    # stay much flatter across the same span.
    raw_drop = trace[0] - trace[-1]
    corrected_drop = abs(corrected_trace[-50:].mean() - corrected_trace[:50].mean())
    assert corrected_drop < raw_drop * 0.2


def test_detrend_movie_preserves_overall_scale():
    # Dividing by trend/mean(trend) (not just trend) should keep the
    # movie's overall intensity level, not shrink/inflate it -- only the
    # slow drift is removed.
    rng = np.random.default_rng(0)
    height, width, n_frames = 8, 8, 200
    decay = np.linspace(1.0, 0.6, n_frames)
    movie = np.ones((height, width, n_frames)) * decay[None, None, :] + rng.standard_normal((height, width, n_frames)) * 0.01

    corrected, _trace, _trend = detrend_movie(movie, percentile=50.0, window=20)

    assert np.isclose(corrected.mean(), movie.mean(), rtol=0.05)


def test_detrend_movie_with_no_drift_leaves_the_movie_almost_unchanged():
    rng = np.random.default_rng(0)
    height, width, n_frames = 8, 8, 100
    movie = np.ones((height, width, n_frames)) * 5.0 + rng.standard_normal((height, width, n_frames)) * 0.001

    corrected, _trace, _trend = detrend_movie(movie, percentile=50.0, window=10)
    assert np.allclose(corrected, movie, rtol=0.01)
