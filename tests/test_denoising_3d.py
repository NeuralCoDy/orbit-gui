import numpy as np

from orbit.denoising_3d import denoise_gaussian_3d, denoise_median_3d


def _noisy_step_volume(seed=0, n_frames=64, shape=(6, 6, 5)):
    rng = np.random.default_rng(seed)
    clean_step = np.concatenate([np.zeros(n_frames // 2), np.ones(n_frames - n_frames // 2)])
    clean = np.broadcast_to(clean_step[:, None, None, None], (n_frames, *shape)).copy()
    noisy = clean + 0.3 * rng.standard_normal((n_frames, *shape))
    return noisy, clean


def test_denoise_gaussian_3d_temporal_only_reduces_noise_and_preserves_shape():
    movie, clean = _noisy_step_volume()
    denoised = denoise_gaussian_3d(movie, spatial_sigma=0.0, temporal_sigma=2.0)

    assert denoised.shape == movie.shape
    assert np.mean((denoised - clean) ** 2) < np.mean((movie - clean) ** 2)


def test_denoise_gaussian_3d_temporal_only_does_not_blur_across_voxels():
    # Two voxels with very different, noise-free constant traces --
    # temporal-only smoothing (spatial_sigma=0) must not average them.
    movie = np.zeros((20, 2, 1, 1))
    movie[:, 0, 0, 0] = 1.0
    movie[:, 1, 0, 0] = 100.0

    denoised = denoise_gaussian_3d(movie, spatial_sigma=0.0, temporal_sigma=3.0)

    assert np.allclose(denoised[:, 0, 0, 0], 1.0)
    assert np.allclose(denoised[:, 1, 0, 0], 100.0)


def test_denoise_gaussian_3d_spatial_only_reduces_noise_across_all_three_spatial_axes():
    rng = np.random.default_rng(4)
    n_frames, length, width, depth = 5, 16, 16, 16
    zz, _yy, _xx = np.meshgrid(np.arange(length), np.arange(width), np.arange(depth), indexing="ij")
    clean_vol = (zz > length / 2).astype(float)
    clean = np.broadcast_to(clean_vol[None, :, :, :], (n_frames, length, width, depth))
    movie = clean + 0.3 * rng.standard_normal((n_frames, length, width, depth))

    denoised = denoise_gaussian_3d(movie, spatial_sigma=1.5, temporal_sigma=0.0)

    assert denoised.shape == movie.shape
    assert np.mean((denoised - clean) ** 2) < np.mean((movie - clean) ** 2)


def test_denoise_gaussian_3d_spatial_only_does_not_blur_across_time():
    movie = np.zeros((2, 10, 10, 10))
    movie[0] = 1.0
    movie[1] = 100.0

    denoised = denoise_gaussian_3d(movie, spatial_sigma=2.0, temporal_sigma=0.0)

    assert np.allclose(denoised[0], 1.0)
    assert np.allclose(denoised[1], 100.0)


def test_denoise_median_3d_removes_salt_and_pepper_spikes():
    rng = np.random.default_rng(5)
    movie = np.full((10, 10, 10, 10), 5.0)
    spike_idx = rng.integers(0, 10, size=(20, 4))
    for t, i, j, k in spike_idx:
        movie[t, i, j, k] = 1000.0

    denoised = denoise_median_3d(movie, space_window=3, time_window=3)

    assert denoised.shape == movie.shape
    assert denoised.max() < 1000.0
    assert np.count_nonzero(denoised != 5.0) < np.count_nonzero(movie != 5.0)


def test_denoise_median_3d_time_window_one_is_spatial_only():
    # time_window=1 means no temporal reach -- a single-frame spike gets
    # removed as a spatial outlier within its own frame, but neighboring
    # frames are left completely untouched (only that frame's own
    # neighborhood is ever considered).
    movie = np.full((3, 9, 9, 9), 5.0)
    movie[1, 4, 4, 4] = 1000.0

    denoised = denoise_median_3d(movie, space_window=3, time_window=1)

    assert denoised[1, 4, 4, 4] == 5.0  # the spike is a spatial outlier within its own frame
    assert np.all(denoised[0] == 5.0)  # untouched frames stay untouched
    assert np.all(denoised[2] == 5.0)
