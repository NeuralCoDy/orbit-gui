import numpy as np
import pytest
from scipy.ndimage import fourier_shift, gaussian_filter

from orbit._volumetric import depth_project
from orbit.motion_correction_3d import rigid_motion_correct_3d


def _mean_corr_3d(mov: np.ndarray, ref: np.ndarray) -> float:
    T = mov.shape[0]
    return float(np.mean([np.corrcoef(mov[t].ravel(), ref.ravel())[0, 1] for t in range(T)]))


def test_rigid_motion_correct_3d_recovers_known_shifts_and_improves_correlation():
    rng = np.random.default_rng(0)
    L, W, D, T = 30, 30, 12, 15
    base = np.zeros((L, W, D))
    base[10:20, 10:20, 4:8] = 1.0
    base = gaussian_filter(base, 2)

    true_shifts = rng.uniform(-4, 4, size=(T, 3))
    movie = np.zeros((T, L, W, D))
    for t in range(T):
        vol = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(base), true_shifts[t])))
        movie[t] = vol + 0.02 * rng.standard_normal((L, W, D))

    registered, shifts, template, initial_template = rigid_motion_correct_3d(
        movie, template=base, max_shift=8, upsample_factor=20, bin_width=200, n_iter=1
    )

    assert registered.shape == movie.shape
    assert shifts.shape == (T, 3)
    assert np.all(np.isfinite(registered))
    np.testing.assert_array_equal(initial_template, base)
    np.testing.assert_allclose(shifts, -true_shifts, atol=0.5)
    assert _mean_corr_3d(registered, base) > _mean_corr_3d(movie, base)
    assert _mean_corr_3d(registered, base) > 0.95


def test_rigid_motion_correct_3d_with_no_motion_is_near_identity():
    rng = np.random.default_rng(1)
    L, W, D, T = 16, 16, 8, 10
    base = gaussian_filter(rng.standard_normal((L, W, D)), 2)
    movie = np.stack([base + 0.01 * rng.standard_normal((L, W, D)) for _ in range(T)], axis=0)

    registered, shifts, _, _ = rigid_motion_correct_3d(movie, max_shift=5, upsample_factor=20, init_batch=T)
    assert registered.shape == movie.shape
    np.testing.assert_allclose(shifts, 0.0, atol=0.3)


def test_rigid_motion_correct_3d_shifts_are_cumulative_across_iterations():
    rng = np.random.default_rng(3)
    L, W, D, T = 20, 20, 10, 8
    base = gaussian_filter(rng.standard_normal((L, W, D)), 2)

    true_shifts = rng.uniform(-3, 3, size=(T, 3))
    movie = np.zeros((T, L, W, D))
    for t in range(T):
        movie[t] = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(base), true_shifts[t])))

    registered, shifts, template, initial_template = rigid_motion_correct_3d(
        movie, template=base, max_shift=8, upsample_factor=20, n_iter=3
    )

    for t in range(T):
        reconstructed = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(movie[t]), shifts[t])))
        # Chaining several subpixel Fourier shifts (one per iteration) isn't
        # bit-identical to a single shift by their sum -- how close depends
        # on how much each iteration's residual correction actually was,
        # which is data/RNG-dependent (confirmed: the 2D version of this
        # same check ranges from ~1e-15 to ~9e-4 across different seeds).
        # 1e-3 comfortably covers that composition error while still
        # catching a genuinely broken accumulation.
        np.testing.assert_allclose(reconstructed, registered[t], atol=1e-3)

    np.testing.assert_array_equal(initial_template, base)
    assert not np.array_equal(initial_template, template)


def test_rigid_motion_correct_3d_output_param_matches_default_in_ram_path():
    rng = np.random.default_rng(5)
    movie = gaussian_filter(rng.standard_normal((25, 12, 12, 6)), (0, 2, 2, 0))

    reg_default, shifts_default, tmpl_default, init_default = rigid_motion_correct_3d(
        movie, bin_width=7, n_iter=2, upsample_factor=10, init_batch=10
    )
    output = np.zeros(movie.shape, dtype=np.float32)  # matches the real FITS memmap sink
    reg_output, shifts_output, tmpl_output, init_output = rigid_motion_correct_3d(
        movie, bin_width=7, n_iter=2, upsample_factor=10, init_batch=10, output=output
    )

    assert reg_output is output
    # float32-scale, not exact: registration works in float32 and the
    # default vs output= paths round intermediates slightly differently
    # -- see the 2D test of the same name in test_motion_correction.py.
    np.testing.assert_allclose(reg_default, reg_output, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(shifts_default, shifts_output, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(tmpl_default, tmpl_output, rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(init_default, init_output)


def test_rigid_motion_correct_3d_output_param_reads_from_movie_not_uninitialized_output():
    rng = np.random.default_rng(7)
    movie = gaussian_filter(rng.standard_normal((12, 12, 12, 6)), (0, 2, 2, 0))

    output = np.full_like(movie, np.nan)  # would poison results if ever read before being written
    registered, _shifts, _tmpl, _init = rigid_motion_correct_3d(
        movie, bin_width=4, n_iter=1, upsample_factor=10, init_batch=6, output=output
    )
    assert np.all(np.isfinite(registered))


def test_depth_project_max_matches_direct_max_over_depth_axis():
    rng = np.random.default_rng(0)
    T, L, W, D = 4, 5, 6, 7
    movie = rng.standard_normal((T, L, W, D))

    projected = depth_project(movie, mode="max")

    assert projected.shape == (L, W, T)
    expected = np.moveaxis(movie.max(axis=3), 0, -1)
    np.testing.assert_array_equal(projected, expected)


def test_depth_project_mean_matches_direct_mean_over_depth_axis():
    rng = np.random.default_rng(1)
    T, L, W, D = 3, 4, 5, 6
    movie = rng.standard_normal((T, L, W, D))

    projected = depth_project(movie, mode="mean")

    assert projected.shape == (L, W, T)
    expected = np.moveaxis(movie.mean(axis=3), 0, -1)
    np.testing.assert_array_equal(projected, expected)


def test_depth_project_rejects_unknown_mode():
    movie = np.zeros((2, 3, 3, 3))
    with pytest.raises(ValueError, match="Unknown mode"):
        depth_project(movie, mode="bogus")
