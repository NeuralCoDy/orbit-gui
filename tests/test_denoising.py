import numpy as np

from orbit import denoising as _denoising
from orbit.denoising import (
    denoise_gaussian,
    denoise_median,
    denoise_wavelet_space,
    denoise_wavelet_time,
    residual_autocorrelation_failures,
    residual_energy_fraction,
)


def _noisy_step_movie(height=16, width=16, n_frames=64, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_frames)
    clean_trace = (t > 0.5).astype(float)  # a simple step, same for every pixel
    movie = np.broadcast_to(clean_trace, (height, width, n_frames)).copy()
    movie += 0.3 * rng.standard_normal(movie.shape)
    return movie, np.broadcast_to(clean_trace, (height, width, n_frames))


def test_denoise_wavelet_time_reduces_noise_relative_to_clean_signal():
    movie, clean = _noisy_step_movie()
    denoised = denoise_wavelet_time(movie, wavelet="sym4", level=3, method="bayes")

    assert denoised.shape == movie.shape
    noisy_error = np.mean((movie - clean) ** 2)
    denoised_error = np.mean((denoised - clean) ** 2)
    assert denoised_error < noisy_error


def test_denoise_wavelet_time_universal_method_runs():
    movie, _clean = _noisy_step_movie(n_frames=32)
    denoised = denoise_wavelet_time(movie, wavelet="db4", level=2, method="universal")
    assert denoised.shape == movie.shape
    assert np.all(np.isfinite(denoised))


def test_denoise_wavelet_space_reduces_noise():
    rng = np.random.default_rng(1)
    height, width, n_frames = 32, 32, 5
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    clean_frame = ((yy > height / 2)).astype(float)
    clean = np.broadcast_to(clean_frame[:, :, None], (height, width, n_frames))
    movie = clean + 0.3 * rng.standard_normal((height, width, n_frames))

    denoised = denoise_wavelet_space(movie, wavelet="sym4", level=2, method="bayes")

    assert denoised.shape == movie.shape
    noisy_error = np.mean((movie - clean) ** 2)
    denoised_error = np.mean((denoised - clean) ** 2)
    assert denoised_error < noisy_error


def test_denoise_gaussian_temporal_only_reduces_noise_and_preserves_shape():
    movie, clean = _noisy_step_movie(n_frames=64)
    denoised = denoise_gaussian(movie, spatial_sigma=0.0, temporal_sigma=2.0)

    assert denoised.shape == movie.shape
    assert np.mean((denoised - clean) ** 2) < np.mean((movie - clean) ** 2)


def test_denoise_gaussian_temporal_only_does_not_blur_across_pixels():
    # Two pixels with very different, noise-free constant traces --
    # temporal-only smoothing (spatial_sigma=0) must not average them together.
    movie = np.zeros((2, 1, 20))
    movie[0, 0, :] = 1.0
    movie[1, 0, :] = 100.0

    denoised = denoise_gaussian(movie, spatial_sigma=0.0, temporal_sigma=3.0)

    assert np.allclose(denoised[0, 0, :], 1.0)
    assert np.allclose(denoised[1, 0, :], 100.0)


def test_denoise_gaussian_spatial_only_reduces_noise():
    rng = np.random.default_rng(4)
    height, width, n_frames = 32, 32, 5
    yy, _xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    clean_frame = (yy > height / 2).astype(float)
    clean = np.broadcast_to(clean_frame[:, :, None], (height, width, n_frames))
    movie = clean + 0.3 * rng.standard_normal((height, width, n_frames))

    denoised = denoise_gaussian(movie, spatial_sigma=1.5, temporal_sigma=0.0)

    assert denoised.shape == movie.shape
    assert np.mean((denoised - clean) ** 2) < np.mean((movie - clean) ** 2)


def test_denoise_gaussian_spatial_only_does_not_blur_across_time():
    # Two frames with very different, noise-free constant values --
    # spatial-only smoothing (temporal_sigma=0) must not average them together.
    movie = np.zeros((10, 10, 2))
    movie[:, :, 0] = 1.0
    movie[:, :, 1] = 100.0

    denoised = denoise_gaussian(movie, spatial_sigma=2.0, temporal_sigma=0.0)

    assert np.allclose(denoised[:, :, 0], 1.0)
    assert np.allclose(denoised[:, :, 1], 100.0)


def test_denoise_gaussian_both_axes_smooths_space_and_time():
    rng = np.random.default_rng(6)
    movie = rng.standard_normal((16, 16, 32))
    denoised = denoise_gaussian(movie, spatial_sigma=1.0, temporal_sigma=1.0)

    assert denoised.shape == movie.shape
    assert denoised.var() < movie.var()


def test_denoise_median_removes_salt_and_pepper_spikes():
    rng = np.random.default_rng(5)
    movie = np.full((10, 10, 10), 5.0)
    # a handful of isolated extreme spikes -- classic case a median
    # filter should remove but a Gaussian blur would only smear around
    spike_idx = rng.integers(0, 10, size=(20, 3))
    for i, j, t in spike_idx:
        movie[i, j, t] = 1000.0

    denoised = denoise_median(movie, space_window=3, time_window=3)

    assert denoised.shape == movie.shape
    assert denoised.max() < 1000.0
    assert np.count_nonzero(denoised != 5.0) < np.count_nonzero(movie != 5.0)


def test_denoise_median_time_window_one_is_spatial_only():
    # time_window=1 means no temporal reach -- a single-frame spike must
    # survive as long as it's not surrounded spatially, since only that
    # frame's own neighborhood is ever considered.
    movie = np.full((10, 10, 3), 5.0)
    movie[5, 5, 1] = 1000.0

    denoised = denoise_median(movie, space_window=3, time_window=1)

    assert denoised[5, 5, 1] == 5.0  # the spike is a spatial outlier within its own frame
    assert np.all(denoised[:, :, 0] == 5.0)  # untouched frames stay untouched
    assert np.all(denoised[:, :, 2] == 5.0)


def test_residual_energy_fraction_zero_for_identical_movies():
    rng = np.random.default_rng(2)
    movie = rng.random((5, 5, 10))
    assert residual_energy_fraction(movie, movie) == 0.0


def test_residual_energy_fraction_one_when_everything_is_removed():
    rng = np.random.default_rng(3)
    movie = rng.random((5, 5, 10)) + 1.0  # avoid a zero-energy "before"
    assert residual_energy_fraction(movie, np.zeros_like(movie)) == 1.0


def test_residual_energy_fraction_partial_removal():
    movie = np.ones((2, 2, 5))
    half = movie * 0.5
    assert np.isclose(residual_energy_fraction(movie, half), 0.25)


def test_residual_energy_fraction_is_block_size_independent(monkeypatch):
    # Both sums are accumulated a slab at a time along axis 0 (see
    # orbit._blocks); the ratio must not depend on the slab size.
    rng = np.random.default_rng(9)
    before = (rng.standard_normal((30, 8, 12)) * 40 + 100).astype(np.float32)
    after = before + 0.2 * rng.standard_normal(before.shape)

    whole = residual_energy_fraction(before, after)

    def _rows_of_5(shape, axis, **_):
        for start in range(0, shape[axis], 5):
            yield slice(start, min(start + 5, shape[axis]))

    monkeypatch.setattr(_denoising, "iter_axis_slices", _rows_of_5)
    chunked = residual_energy_fraction(before, after)

    assert np.isclose(whole, chunked, rtol=1e-6, atol=1e-10)


def test_residual_autocorrelation_failures_zero_for_a_perfect_denoise():
    # An exactly-zero residual (denoised == original) is constant --
    # ljung_box_test_movie always passes a constant trace trivially.
    rng = np.random.default_rng(0)
    movie = rng.standard_normal((4, 4, 200))
    n_failed, n_total = residual_autocorrelation_failures(movie, movie)
    assert n_failed == 0
    assert n_total == 16


def test_residual_autocorrelation_failures_flags_removed_signal():
    # Denoising that also subtracts a genuine slow signal (not just
    # noise) leaves that signal's structure behind in the residual --
    # every pixel's residual should then fail the whiteness test.
    rng = np.random.default_rng(1)
    height, width, n_frames = 4, 4, 200
    movie = rng.standard_normal((height, width, n_frames))
    removed_signal = np.sin(np.linspace(0, 6 * np.pi, n_frames))
    over_denoised = movie - np.broadcast_to(removed_signal, (height, width, n_frames))

    n_failed, n_total = residual_autocorrelation_failures(movie, over_denoised)
    assert n_failed == n_total == height * width
