import tracemalloc

import numpy as np
import pytest

import orbit._native as _native
from orbit.projections import (
    _BLOCK_PIXELS,
    fano_factor_projection,
    fano_factor_projection_volumetric,
    local_correlation_projection,
    mean_projection,
    mean_projection_volumetric,
    median_projection,
    median_projection_volumetric,
    mode_projection,
    mode_projection_volumetric,
    variance_projection,
    variance_projection_volumetric,
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


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_median_projection_matches_plain_numpy_across_a_chunk_boundary(dtype):
    # H well past _BLOCK_PIXELS so the chunked loop actually runs
    # more than one block, and per numpy's own dtype rule (int/float32-
    # under-float64 -> float64, float64 stays float64) the dtype must
    # match too.
    rng = np.random.default_rng(0)
    H = _BLOCK_PIXELS * 2 + 7
    movie = (rng.random((H, 3, 11)) * 200).astype(dtype)

    got = median_projection(movie)
    want = np.median(movie, axis=2)

    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got, want)


def test_median_projection_volumetric_matches_plain_numpy_across_a_chunk_boundary():
    rng = np.random.default_rng(1)
    L = _BLOCK_PIXELS * 2 + 7
    vol = (rng.random((6, L, 3, 2)) * 200).astype(np.uint16)

    got = median_projection_volumetric(vol)
    want = np.median(vol, axis=0)

    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got, want)


def test_median_projection_peak_memory_stays_near_one_block_not_the_whole_movie():
    # A large-enough H (many multiples of _BLOCK_PIXELS) that a
    # whole-array np.median partition copy would dwarf one block's --
    # confirms the chunking is actually bounding the transient, not
    # just matching output by accident.
    rng = np.random.default_rng(2)
    H, W, T = _BLOCK_PIXELS * 40, 8, 25
    movie = (rng.random((H, W, T)) * 200).astype(np.float32)
    whole_array_partition_bytes = movie.nbytes  # np.median's internal copy, unchunked

    tracemalloc.start()
    median_projection(movie)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < whole_array_partition_bytes / 10


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_variance_projection_matches_plain_numpy_across_a_chunk_boundary(dtype):
    rng = np.random.default_rng(6)
    H = _BLOCK_PIXELS * 2 + 7
    movie = (rng.random((H, 3, 11)) * 200).astype(dtype)

    got = variance_projection(movie)
    want = movie.var(axis=2)

    assert got.dtype == want.dtype
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_variance_projection_volumetric_matches_plain_numpy_across_a_chunk_boundary():
    rng = np.random.default_rng(7)
    L = _BLOCK_PIXELS * 2 + 7
    vol = (rng.random((6, L, 3, 2)) * 200).astype(np.uint16)

    got = variance_projection_volumetric(vol)
    want = vol.var(axis=0)

    assert got.dtype == want.dtype
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_fano_factor_projection_matches_plain_numpy_across_a_chunk_boundary():
    rng = np.random.default_rng(8)
    H = _BLOCK_PIXELS * 2 + 7
    movie = (rng.random((H, 3, 11)) * 200).astype(np.uint16)

    got = fano_factor_projection(movie)

    mean = movie.mean(axis=2)
    var = movie.var(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        want = np.where(mean > 0, var / mean, 0.0)
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_variance_projection_peak_memory_stays_near_one_block_not_the_whole_movie():
    # ndarray.var() builds a full centered `arr - mean` array before
    # squaring/summing it -- unchunked, that's a second, float64-sized
    # (often larger than the input) transient. Confirms chunking bounds
    # it to one block instead.
    rng = np.random.default_rng(9)
    H, W, T = _BLOCK_PIXELS * 40, 8, 25
    movie = (rng.random((H, W, T)) * 200).astype(np.float32)
    whole_array_centered_bytes = movie.nbytes  # what an unchunked .var() would copy, at minimum

    tracemalloc.start()
    variance_projection(movie)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < whole_array_centered_bytes / 10


def test_fano_factor_projection_combine_step_avoids_extra_full_size_temporaries():
    # A large enough (H, W) OUTPUT (not just a large movie) that
    # `var / mean` + `np.where`'s two extra full-size temporaries would
    # show up clearly -- confirms fano_factor_projection's in-place
    # divide-into-var doesn't allocate them. H*W matters here, not T.
    rng = np.random.default_rng(10)
    H, W, T = 4000, 4000, 3
    movie = (rng.random((H, W, T)) * 200).astype(np.float32)
    one_output_array_bytes = H * W * 4  # movie is float32, so mean/var stay float32 (np.mean/np.var's own promotion rule)

    tracemalloc.start()
    fano_factor_projection(movie)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # mean + var (both float32, (H, W)) must coexist to compute their
    # ratio -- that's the necessary floor, plus a couple of small bool
    # masks (~0.25x each) for the positive-mean selection -- measured
    # ~2.5x with the in-place fix. A reintroduced `var / mean` +
    # `np.where` would add two more full-size buffers (~2x more, ~4.5x
    # total) -- clearly over this bound.
    assert peak < one_output_array_bytes * 3.5


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


def test_volumetric_projections_reduce_over_time_not_space():
    # (T, L, W, D): a voxel whose value ramps over time, everything else
    # constant. Reducing over time must give an (L, W, D) volume where
    # only that voxel differs -- reducing over a space axis (the bug this
    # guards against) would smear it across a whole row/column.
    T, L, W, D = 8, 3, 4, 2
    vol = np.zeros((T, L, W, D))
    vol[:, 1, 2, 0] = np.arange(T)

    for fn in (
        mean_projection_volumetric,
        median_projection_volumetric,
        variance_projection_volumetric,
        fano_factor_projection_volumetric,
        mode_projection_volumetric,
    ):
        proj = fn(vol)
        assert proj.shape == (L, W, D)
        nonzero = np.argwhere(proj != 0)
        assert nonzero.tolist() == [[1, 2, 0]], f"{fn.__name__} smeared beyond the one active voxel"

    np.testing.assert_allclose(mean_projection_volumetric(vol), vol.mean(axis=0))
    np.testing.assert_allclose(variance_projection_volumetric(vol), vol.var(axis=0))


def test_mode_projection_volumetric_matches_per_voxel_2d_mode():
    rng = np.random.default_rng(7)
    vol = rng.standard_normal((20, 3, 5, 4)).astype(np.float32)

    got = mode_projection_volumetric(vol)

    expected = np.empty((3, 5, 4), dtype=np.float32)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                expected[i, j, k] = mode_projection(vol[:, i, j, k].reshape(1, 1, -1))[0, 0]
    np.testing.assert_allclose(got, expected, atol=1e-5)


@pytest.mark.skipif(not _native.NATIVE_AVAILABLE, reason="native extension not built -- run orbit/_native/build_native.sh")
def test_mode_native_matches_numpy_fallback(monkeypatch):
    rng = np.random.default_rng(6)
    movie = rng.standard_normal((10, 8, 25)).astype(np.float32)

    native_result = mode_projection(movie)

    monkeypatch.setattr(_native, "NATIVE_AVAILABLE", False)
    numpy_result = mode_projection(movie)

    assert np.allclose(native_result, numpy_result, atol=1e-4)
