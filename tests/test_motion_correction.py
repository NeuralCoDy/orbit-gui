import warnings

import numpy as np
import pytest
from scipy.ndimage import fourier_shift, gaussian_filter, map_coordinates

from orbit.motion_correction import motion_correct, patch_motion_correct, rigid_motion_correct


def _mean_corr(mov: np.ndarray, ref: np.ndarray) -> float:
    T = mov.shape[-1]
    return float(np.mean([np.corrcoef(mov[:, :, t].ravel(), ref.ravel())[0, 1] for t in range(T)]))


def test_rigid_motion_correct_recovers_known_shifts_and_improves_correlation():
    rng = np.random.default_rng(0)
    H, W, T = 50, 50, 20
    base = np.zeros((H, W))
    base[15:35, 15:35] = 1.0
    base = gaussian_filter(base, 3)

    true_shifts = rng.uniform(-5, 5, size=(T, 2))
    movie = np.zeros((H, W, T))
    for t in range(T):
        frame = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(base), true_shifts[t])))
        movie[:, :, t] = frame + 0.02 * rng.standard_normal((H, W))

    registered, shifts, template, initial_template = rigid_motion_correct(
        movie, template=base, max_shift=10, upsample_factor=20, bin_width=200, n_iter=1
    )

    assert registered.shape == movie.shape
    assert shifts.shape == (T, 2)
    assert np.all(np.isfinite(registered))
    np.testing.assert_array_equal(initial_template, base)
    np.testing.assert_allclose(shifts, -true_shifts, atol=0.5)
    assert _mean_corr(registered, base) > _mean_corr(movie, base)
    assert _mean_corr(registered, base) > 0.95


def test_rigid_motion_correct_with_no_motion_is_near_identity():
    rng = np.random.default_rng(1)
    H, W, T = 30, 30, 10
    base = gaussian_filter(rng.standard_normal((H, W)), 2)
    movie = np.stack([base + 0.01 * rng.standard_normal((H, W)) for _ in range(T)], axis=-1)

    registered, shifts, _, _ = rigid_motion_correct(movie, max_shift=5, upsample_factor=20, init_batch=T)
    np.testing.assert_allclose(shifts, 0.0, atol=0.3)


def test_rigid_motion_correct_shifts_are_cumulative_across_iterations():
    rng = np.random.default_rng(3)
    H, W, T = 40, 40, 8
    base = gaussian_filter(rng.standard_normal((H, W)), 2)

    true_shifts = rng.uniform(-4, 4, size=(T, 2))
    movie = np.zeros((H, W, T))
    for t in range(T):
        frame = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(base), true_shifts[t])))
        movie[:, :, t] = frame

    registered, shifts, template, initial_template = rigid_motion_correct(
        movie, template=base, max_shift=10, upsample_factor=20, n_iter=3
    )

    for t in range(T):
        reconstructed = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(movie[:, :, t]), shifts[t])))
        np.testing.assert_allclose(reconstructed, registered[:, :, t], atol=1e-6)

    np.testing.assert_array_equal(initial_template, base)
    assert not np.array_equal(initial_template, template)


def test_patch_motion_correct_outperforms_rigid_on_local_shear():
    rng = np.random.default_rng(0)
    H, W, T = 60, 60, 15

    base = gaussian_filter(rng.standard_normal((H, W)), 2)
    base = (base - base.min()) / (base.max() - base.min())

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    def make_frame(global_shift, shear_amt):
        dy = np.full((H, W), global_shift[0])
        dx = global_shift[1] + shear_amt * (xx - W / 2) / W
        coords = [yy - dy, xx - dx]
        return map_coordinates(base, coords, order=3, mode="nearest")

    movie = np.zeros((H, W, T))
    global_shifts = rng.uniform(-3, 3, size=(T, 2))
    shear_amts = rng.uniform(-8, 8, size=T)
    for t in range(T):
        frame = make_frame(global_shifts[t], shear_amts[t])
        movie[:, :, t] = frame + 0.02 * rng.standard_normal((H, W))

    def region_corr(mov, c0, c1):
        return np.mean([np.corrcoef(mov[:, c0:c1, t].ravel(), base[:, c0:c1].ravel())[0, 1] for t in range(T)])

    reg_rigid, _, _, _ = rigid_motion_correct(movie, max_shift=10, upsample_factor=20, init_batch=T, n_iter=2)
    reg_patch, shift_fields, _, _ = patch_motion_correct(
        movie, grid_size=20, max_shift=10, max_dev=8, upsample_factor=20, init_batch=T, n_iter=2
    )

    assert reg_patch.shape == movie.shape
    assert shift_fields.shape[0] == T
    assert np.all(np.isfinite(reg_patch))

    for c0, c1 in [(0, 20), (20, 40), (40, 60)]:
        assert region_corr(reg_patch, c0, c1) > region_corr(reg_rigid, c0, c1)


def test_patch_motion_correct_single_patch_matches_rigid_shape():
    rng = np.random.default_rng(2)
    H, W, T = 20, 20, 5
    movie = gaussian_filter(rng.standard_normal((H, W, T)), (2, 2, 0))

    registered, shift_fields, template, initial_template = patch_motion_correct(
        movie, grid_size=64, max_shift=5, max_dev=2, upsample_factor=10, init_batch=T
    )
    assert registered.shape == movie.shape
    assert shift_fields.shape == (T, 1, 1, 2)
    assert template.shape == (H, W)
    assert initial_template.shape == (H, W)


def test_patch_motion_correct_warns_on_low_contrast_patches():
    rng = np.random.default_rng(3)
    H, W, T = 64, 64, 6
    base = np.zeros((H, W))
    base[28:36, 28:36] = 1.0
    base = gaussian_filter(base, 2)
    movie = np.stack([base + 0.01 * rng.standard_normal((H, W)) for _ in range(T)], axis=-1)

    with pytest.warns(UserWarning, match="local contrast"):
        patch_motion_correct(movie, template=base, grid_size=16, max_shift=5, max_dev=2, upsample_factor=10)


def test_motion_correct_dispatches_by_method_name():
    rng = np.random.default_rng(4)
    movie = gaussian_filter(rng.standard_normal((20, 20, 5)), (2, 2, 0))

    rigid_result = motion_correct(movie, method="rigid", max_shift=5, upsample_factor=10, init_batch=5)
    patch_result = motion_correct(
        movie, method="patch", grid_size=64, max_shift=5, max_dev=2, upsample_factor=10, init_batch=5
    )

    assert rigid_result[1].shape == (5, 2)  # rigid shifts: (T, 2)
    assert patch_result[1].shape == (5, 1, 1, 2)  # patch shift_fields: (T, ny, nx, 2)


def test_motion_correct_rejects_unknown_method():
    movie = np.zeros((10, 10, 3))
    with pytest.raises(ValueError, match="Unknown motion correction method"):
        motion_correct(movie, method="bogus")
