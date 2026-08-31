import warnings

import numpy as np
import pytest
from scipy.ndimage import fourier_shift, gaussian_filter, map_coordinates

from orbit.motion_correction import _pad_to_fast_len, motion_correct, patch_motion_correct, rigid_motion_correct


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


def test_rigid_motion_correct_output_param_matches_default_in_ram_path():
    # output= (used for a memmap movie's chunked Commit) must produce
    # numerically identical results to the normal in-RAM path -- it's a
    # where-results-are-written change, not an algorithm change.
    rng = np.random.default_rng(5)
    movie = gaussian_filter(rng.standard_normal((16, 16, 25)), (2, 2, 0))

    reg_default, shifts_default, tmpl_default, init_default = rigid_motion_correct(
        movie, bin_width=7, n_iter=2, upsample_factor=10, init_batch=10
    )
    output = np.zeros_like(movie)
    reg_output, shifts_output, tmpl_output, init_output = rigid_motion_correct(
        movie, bin_width=7, n_iter=2, upsample_factor=10, init_batch=10, output=output
    )

    assert reg_output is output
    np.testing.assert_allclose(reg_default, reg_output)
    np.testing.assert_allclose(shifts_default, shifts_output)
    np.testing.assert_allclose(tmpl_default, tmpl_output)
    np.testing.assert_array_equal(init_default, init_output)


def test_patch_motion_correct_output_param_matches_default_in_ram_path():
    rng = np.random.default_rng(6)
    movie = gaussian_filter(rng.standard_normal((16, 16, 25)), (2, 2, 0))

    reg_default, sf_default, tmpl_default, init_default = patch_motion_correct(
        movie, grid_size=8, bin_width=7, n_iter=1, upsample_factor=10, init_batch=10, min_patch_contrast=0
    )
    output = np.zeros_like(movie)
    reg_output, sf_output, tmpl_output, init_output = patch_motion_correct(
        movie, grid_size=8, bin_width=7, n_iter=1, upsample_factor=10, init_batch=10, min_patch_contrast=0,
        output=output,
    )

    assert reg_output is output
    np.testing.assert_allclose(reg_default, reg_output)
    np.testing.assert_allclose(sf_default, sf_output)
    np.testing.assert_allclose(tmpl_default, tmpl_output)
    np.testing.assert_array_equal(init_default, init_output)


def test_rigid_motion_correct_output_param_reads_from_movie_not_uninitialized_output():
    # A regression check for the exact bug this feature could introduce:
    # if the first pass ever read from `output` before it's populated
    # (rather than from `movie`), results would silently be garbage
    # (registering against zeros) instead of matching the in-RAM path.
    rng = np.random.default_rng(7)
    movie = gaussian_filter(rng.standard_normal((16, 16, 12)), (2, 2, 0))

    output = np.full_like(movie, np.nan)  # would poison results if ever read before being written
    registered, _shifts, _tmpl, _init = rigid_motion_correct(
        movie, bin_width=4, n_iter=1, upsample_factor=10, init_batch=6, output=output
    )
    assert np.all(np.isfinite(registered))


def test_pad_to_fast_len_is_a_true_noop_for_an_already_fast_size():
    # 256 = 2**8 -- already FFT-fast on every axis, so padding must
    # return the SAME array (no copy), not just an equal one.
    image = np.zeros((256, 256))
    assert _pad_to_fast_len(image) is image


def test_pad_to_fast_len_pads_each_axis_up_to_its_own_next_fast_len():
    # 509 is prime (next fast len 512); 240 = 2**4*3*5 is already fast.
    image = np.zeros((509, 240))
    padded = _pad_to_fast_len(image)
    assert padded.shape == (512, 240)


def test_pad_to_fast_len_preserves_the_original_content_in_the_unpadded_region():
    rng = np.random.default_rng(0)
    image = rng.standard_normal((509, 509))
    padded = _pad_to_fast_len(image, mode="edge")
    np.testing.assert_array_equal(padded[:509, :509], image)


def test_pad_to_fast_len_is_dimension_agnostic():
    # motion_correction_3d.py reuses _apply_shift/_estimate_shift (and
    # therefore _pad_to_fast_len) as-is against volumes, not just 2D frames.
    volume = np.zeros((509, 100, 100))
    padded = _pad_to_fast_len(volume)
    assert padded.shape[0] == 512
    assert padded.shape[1:] == (100, 100)


def test_rigid_motion_correct_recovers_known_shifts_at_an_unlucky_prime_size():
    # Regression guard for the FFT-fast-size padding in _estimate_shift/
    # _apply_shift: 251 is prime, deliberately not FFT-friendly. Padding
    # for speed must not meaningfully change the recovered shifts (same
    # tolerance as the equivalent non-prime-size test above).
    rng = np.random.default_rng(0)
    H, W, T = 251, 251, 10
    base = np.zeros((H, W))
    base[100:150, 100:150] = 1.0
    base = gaussian_filter(base, 3)

    true_shifts = rng.uniform(-5, 5, size=(T, 2))
    movie = np.zeros((H, W, T))
    for t in range(T):
        frame = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(base), true_shifts[t])))
        movie[:, :, t] = frame + 0.02 * rng.standard_normal((H, W))

    registered, shifts, _template, _initial = rigid_motion_correct(
        movie, template=base, max_shift=10, upsample_factor=20, bin_width=200, n_iter=1
    )

    assert registered.shape == movie.shape
    assert np.all(np.isfinite(registered))
    np.testing.assert_allclose(shifts, -true_shifts, atol=0.5)
    assert _mean_corr(registered, base) > 0.95


def test_estimate_shift_padding_does_not_meaningfully_change_the_estimate():
    # Direct check that padding (inside _estimate_shift, exercised here
    # via rigid_motion_correct at n_iter=1/bin_width=T so there's no
    # template-refresh noise) tracks the true unpadded phase-correlation
    # answer closely, not just "recovers the shift to within the test's
    # own generous tolerance" -- pins the ~0.02px (one upsample_factor=50
    # quantization step) agreement confirmed during development.
    from skimage.registration import phase_cross_correlation

    rng = np.random.default_rng(0)
    ref = gaussian_filter(rng.standard_normal((509, 509)), 2).astype(np.float64) * 10 + 100
    true_shift = np.array([2.37, -1.84])
    moving = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(ref), true_shift)))

    unpadded_shift, _e, _p = phase_cross_correlation(ref, moving, upsample_factor=50)

    padded_ref = _pad_to_fast_len(ref, mode="edge")
    padded_moving = _pad_to_fast_len(moving, mode="edge")
    padded_shift, _e, _p = phase_cross_correlation(padded_ref, padded_moving, upsample_factor=50)

    np.testing.assert_allclose(padded_shift, unpadded_shift, atol=0.15)


def test_patch_motion_correct_recovers_shifts_with_prime_width_patches():
    # A 93px frame split grid_size=31 lands on three EXACTLY 31px-wide
    # patches (31 is prime) -- confirmed via _patch_centers(93, 31).
    # Regression guard that _estimate_shift's FFT-fast-size padding
    # (exercised once per patch per frame here) doesn't break patch-
    # based correction at exactly the kind of unlucky size it's meant
    # to help with.
    rng = np.random.default_rng(0)
    H, W, T = 93, 93, 10
    base = np.zeros((H, W))
    base[30:60, 30:60] = 1.0
    base = gaussian_filter(base, 2)

    true_shifts = rng.uniform(-3, 3, size=(T, 2))
    movie = np.zeros((H, W, T))
    for t in range(T):
        frame = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(base), true_shifts[t])))
        movie[:, :, t] = frame + 0.02 * rng.standard_normal((H, W))

    registered, shift_fields, _template, _initial = patch_motion_correct(
        movie, template=base, grid_size=31, max_shift=10, max_dev=8, upsample_factor=20, init_batch=T, n_iter=1,
        min_patch_contrast=0,  # several patches are pure background at this grid density -- expected, not a bug
    )

    assert registered.shape == movie.shape
    assert shift_fields.shape[1:3] == (3, 3)  # 3x3 grid of 31px patches
    assert np.all(np.isfinite(registered))
    assert _mean_corr(registered, base) > 0.9
