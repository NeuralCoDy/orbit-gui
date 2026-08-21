import numpy as np

from orbit.pca_denoise import _svd_reconstruct, pca_denoise, pca_denoise_block


def _synthetic_low_rank_movie(height, width, n_frames, rank=3, noise_std=0.5, seed=0):
    rng = np.random.default_rng(seed)
    spatial = rng.standard_normal((height * width, rank))
    temporal = rng.standard_normal((rank, n_frames))
    signal = (spatial @ temporal).reshape(height, width, n_frames)
    noisy = signal + rng.standard_normal(signal.shape) * noise_std
    return signal, noisy


def test_svd_reconstruct_full_rank_recovers_the_original_matrix():
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((20, 15))
    reconstructed = _svd_reconstruct(matrix, n_components=15)  # full rank of a (20, 15) matrix
    assert np.allclose(reconstructed, matrix, atol=1e-8)


def test_svd_reconstruct_low_rank_reduces_noise_error():
    height, width, n_frames = 30, 30, 100
    signal, noisy = _synthetic_low_rank_movie(height, width, n_frames, rank=3, noise_std=0.5)
    flat_signal = signal.reshape(-1, n_frames)
    flat_noisy = noisy.reshape(-1, n_frames)

    reconstructed = _svd_reconstruct(flat_noisy, n_components=3)

    err_before = np.linalg.norm(flat_noisy - flat_signal)
    err_after = np.linalg.norm(reconstructed - flat_signal)
    assert err_after < err_before


def test_pca_denoise_block_shape_and_low_rank_recovery():
    height, width, n_frames = 20, 20, 80
    signal, noisy = _synthetic_low_rank_movie(height, width, n_frames, rank=2, noise_std=0.4)

    denoised = pca_denoise_block(noisy, n_components=2)

    assert denoised.shape == noisy.shape
    assert np.linalg.norm(denoised - signal) < np.linalg.norm(noisy - signal)


def test_pca_denoise_single_block_matches_pca_denoise_block_directly():
    # A movie smaller than block_size/block_frames collapses to exactly
    # one block (see test_patches.py) -- pca_denoise should then give
    # the identical result to calling pca_denoise_block directly, with
    # no blending involved.
    height, width, n_frames = 20, 20, 80
    _signal, noisy = _synthetic_low_rank_movie(height, width, n_frames, rank=2, noise_std=0.4)

    direct = pca_denoise_block(noisy, n_components=2)
    via_pca_denoise = pca_denoise(
        noisy, n_components=2, block_size=(250, 250), block_frames=5000, spatial_overlap=30, temporal_overlap=500
    )

    assert np.allclose(direct, via_pca_denoise, atol=1e-8)


def test_pca_denoise_tiles_and_blends_a_movie_larger_than_one_block():
    height, width, n_frames = 60, 60, 200
    signal, noisy = _synthetic_low_rank_movie(height, width, n_frames, rank=3, noise_std=0.4)

    denoised = pca_denoise(
        noisy, n_components=3, block_size=(35, 35), block_frames=120, spatial_overlap=10, temporal_overlap=30
    )

    assert denoised.shape == noisy.shape
    assert np.isfinite(denoised).all()
    # Denoising (even split across multiple blended blocks) should still
    # recover the underlying low-rank signal better than doing nothing.
    assert np.linalg.norm(denoised - signal) < np.linalg.norm(noisy - signal)


def test_pca_denoise_n_components_zero_returns_all_zeros():
    _signal, noisy = _synthetic_low_rank_movie(15, 15, 50, rank=2, noise_std=0.4)
    denoised = pca_denoise_block(noisy, n_components=0)
    assert np.allclose(denoised, 0.0)
