import numpy as np

from orbit.detrending import (
    detrend_movie,
    detrend_movie_3d,
    fov_average_trace,
    fov_average_trace_3d,
    running_percentile_trend,
)


def _naive_running_percentile(trace, percentile, window):
    n = len(trace)
    out = np.empty(n)
    for t in range(n):
        lo = max(0, t - window + 1)
        out[t] = np.percentile(trace[lo : t + 1], percentile)
    return out


def test_fov_average_trace_is_mean_over_space_per_frame_normalized_to_the_first_frame():
    movie = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4) + 1.0  # +1 so frame 0's mean isn't 0
    trace = fov_average_trace(movie)
    assert trace.shape == (4,)
    assert trace[0] == 1.0
    for t in range(4):
        assert np.isclose(trace[t], movie[:, :, t].mean() / movie[:, :, 0].mean())


def test_fov_average_trace_ignores_pixels_outside_the_mask():
    rng = np.random.default_rng(0)
    movie = rng.standard_normal((5, 5, 10)) + 10.0
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:3, 1:3] = True  # a small real region -- the rest is "blank background"
    movie[~mask] = 1e6  # a wildly different blank-space value that must NOT leak into the average

    trace = fov_average_trace(movie, mask)
    expected = movie[mask].mean(axis=0)
    expected = expected / expected[0]
    assert np.allclose(trace, expected)


def test_fov_average_trace_returns_the_raw_trace_when_the_first_frame_s_mean_is_zero():
    # A degenerate all-zero first frame can't be normalized against --
    # falls back to the unnormalized trace rather than producing inf/nan.
    movie = np.zeros((2, 2, 3))
    movie[:, :, 1:] = 5.0
    trace = fov_average_trace(movie)
    assert np.all(np.isfinite(trace))
    assert trace[0] == 0.0


def test_fov_average_trace_3d_is_mean_over_all_spatial_axes_per_volume():
    rng = np.random.default_rng(1)
    movie = rng.standard_normal((6, 4, 4, 3)) + 10.0  # (T, L, W, D)
    trace = fov_average_trace_3d(movie)
    assert trace.shape == (6,)
    assert trace[0] == 1.0
    for t in range(6):
        assert np.isclose(trace[t], movie[t].mean() / movie[0].mean())


def test_fov_average_trace_3d_ignores_voxels_outside_the_mask():
    rng = np.random.default_rng(2)
    movie = rng.standard_normal((6, 5, 5, 5)) + 10.0
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    movie[:, ~mask] = 1e6

    trace = fov_average_trace_3d(movie, mask)
    expected = movie[:, mask].mean(axis=1)
    expected = expected / expected[0]
    assert np.allclose(trace, expected)


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


def test_detrend_movie_correction_is_unaffected_by_the_trace_s_own_normalization():
    # fov_average_trace's first-frame normalization only rescales trace/trend
    # by a positive constant -- running_percentile_trend's percentile and the
    # scale = trend/trend.mean() ratio both commute with (i.e. are invariant
    # to) that rescaling, so the actual `corrected` movie must come out
    # identical whether or not the trace happens to be normalized. Verified
    # directly here against a manually-unnormalized trace/trend, rather than
    # just asserted in the docstring.
    rng = np.random.default_rng(3)
    height, width, n_frames = 8, 8, 150
    decay = np.linspace(1.0, 0.5, n_frames)
    movie = (np.ones((height, width, n_frames)) * 100.0 + rng.standard_normal((height, width, n_frames))) * decay[
        None, None, :
    ]

    corrected, trace, trend = detrend_movie(movie, percentile=8.0, window=25)

    raw_trace = movie.mean(axis=(0, 1))  # deliberately NOT normalized
    raw_trend = running_percentile_trend(raw_trace, 8.0, 25)
    raw_scale = raw_trend / raw_trend.mean()
    raw_corrected = movie / raw_scale[None, None, :]

    assert np.allclose(trace, raw_trace / raw_trace[0])
    assert np.allclose(trend, raw_trend / raw_trace[0])
    assert np.allclose(corrected, raw_corrected)


def test_detrend_movie_accepts_a_mask_and_ignores_pixels_outside_it():
    rng = np.random.default_rng(4)
    height, width, n_frames = 8, 8, 150
    decay = np.linspace(1.0, 0.5, n_frames)
    movie = (np.ones((height, width, n_frames)) * 10.0 + rng.standard_normal((height, width, n_frames)) * 0.1) * decay[
        None, None, :
    ]
    mask = np.zeros((height, width), dtype=bool)
    mask[2:6, 2:6] = True
    movie_with_junk = movie.copy()
    movie_with_junk[~mask] = 1e6  # blank background at a wildly different, non-decaying scale

    corrected, trace, _trend = detrend_movie(movie_with_junk, percentile=8.0, window=25, mask=mask)
    expected_corrected, expected_trace, _expected_trend = detrend_movie(movie, percentile=8.0, window=25, mask=mask)

    assert np.allclose(trace, expected_trace)
    # The masked-in region should be corrected the same way regardless of
    # what the (excluded) background pixels were doing.
    assert np.allclose(corrected[mask], expected_corrected[mask], rtol=0.05)


def test_detrend_movie_3d_flattens_a_synthetic_multiplicative_drift():
    rng = np.random.default_rng(5)
    n_frames, length, width, depth = 200, 6, 6, 4
    decay = np.linspace(1.0, 0.4, n_frames)
    movie = (rng.standard_normal((n_frames, length, width, depth)) * 0.01 + 1.0) * decay[:, None, None, None]
    movie = np.clip(movie, 0.01, None)

    corrected, trace, trend = detrend_movie_3d(movie, percentile=8.0, window=30)

    assert corrected.shape == movie.shape
    assert trace.shape == (n_frames,)
    assert trend.shape == (n_frames,)

    corrected_trace = corrected.mean(axis=(1, 2, 3))
    raw_drop = trace[0] - trace[-1]
    corrected_drop = abs(corrected_trace[-50:].mean() - corrected_trace[:50].mean())
    assert corrected_drop < raw_drop * 0.2


def test_detrend_movie_3d_accepts_a_mask_and_ignores_voxels_outside_it():
    rng = np.random.default_rng(6)
    n_frames, length, width, depth = 150, 8, 8, 6
    decay = np.linspace(1.0, 0.5, n_frames)
    movie = (np.ones((n_frames, length, width, depth)) * 10.0 + rng.standard_normal(
        (n_frames, length, width, depth)
    ) * 0.1) * decay[:, None, None, None]
    mask = np.zeros((length, width, depth), dtype=bool)
    mask[2:6, 2:6, 1:4] = True
    movie_with_junk = movie.copy()
    movie_with_junk[:, ~mask] = 1e6

    corrected, trace, _trend = detrend_movie_3d(movie_with_junk, percentile=8.0, window=25, mask=mask)
    expected_corrected, expected_trace, _ = detrend_movie_3d(movie, percentile=8.0, window=25, mask=mask)

    assert np.allclose(trace, expected_trace)
    assert np.allclose(corrected[:, mask], expected_corrected[:, mask], rtol=0.05)
