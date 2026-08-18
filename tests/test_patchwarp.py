import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from orbit.motion_correction import motion_correct
from orbit.patchwarp import patchwarp_motion_correct


def _mean_corr(mov: np.ndarray, ref: np.ndarray) -> float:
    T = mov.shape[-1]
    return float(np.mean([np.corrcoef(mov[:, :, t].ravel(), ref.ravel())[0, 1] for t in range(T)]))


def test_patchwarp_shapes_and_finite():
    rng = np.random.default_rng(0)
    H, W, T = 40, 40, 6
    movie = gaussian_filter(rng.standard_normal((H, W, T)), (2, 2, 0))

    registered, affine_matrices, template, initial_template = patchwarp_motion_correct(
        movie, grid_size=2, overlap_frac=0.1, ecc_iterations=20, pyramid_levels=1
    )

    assert registered.shape == movie.shape
    assert affine_matrices.shape == (T, 2, 2, 2, 3)
    assert template.shape == (H, W)
    assert initial_template.shape == (H, W)
    assert np.all(np.isfinite(registered))


def test_patchwarp_corrects_local_shear_better_than_rigid_alone():
    # Same local-shear scenario as the piecewise-rigid test: rigid alone
    # can only fit the frame-average shift, patchwarp's per-patch affine
    # fit should measurably outperform it in every spatial region.
    rng = np.random.default_rng(1)
    H, W, T = 60, 60, 10

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
    shear_amts = rng.uniform(-6, 6, size=T)
    for t in range(T):
        movie[:, :, t] = make_frame(global_shifts[t], shear_amts[t]) + 0.01 * rng.standard_normal((H, W))

    reg_rigid, _, _, _ = motion_correct(movie, method="rigid", max_shift=10, upsample_factor=20, init_batch=T)
    reg_patchwarp, _, _, _ = patchwarp_motion_correct(
        movie, grid_size=3, overlap_frac=0.15, rigid_max_shift=10, ecc_iterations=40, pyramid_levels=1
    )

    assert _mean_corr(reg_patchwarp, base) > _mean_corr(reg_rigid, base)


def test_motion_correct_dispatches_to_patchwarp():
    rng = np.random.default_rng(2)
    movie = gaussian_filter(rng.standard_normal((30, 30, 4)), (2, 2, 0))

    registered, affine_matrices, template, initial_template = motion_correct(
        movie, method="patchwarp", grid_size=2, ecc_iterations=10
    )

    assert registered.shape == movie.shape
    assert affine_matrices.shape[0] == 4
