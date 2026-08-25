import numpy as np

from orbit.masking import apply_mask, triangle_mask


def _bright_blob_on_dim_background(height=40, width=40, seed=0):
    rng = np.random.default_rng(seed)
    proj = np.full((height, width), 0.1)
    proj[10:20, 10:20] = 5.0
    proj += rng.standard_normal((height, width)) * 0.02
    return proj


def test_triangle_mask_keeps_the_bright_blob_and_excludes_the_background():
    proj = _bright_blob_on_dim_background()
    mask = triangle_mask(proj)

    assert mask.dtype == bool
    assert mask[15, 15]  # inside the bright blob
    assert not mask[0, 0]  # dim background corner
    assert not mask[39, 39]


def test_apply_mask_zeroes_pixels_outside_the_mask_and_preserves_the_rest():
    rng = np.random.default_rng(1)
    height, width, n_frames = 10, 10, 20
    movie = rng.standard_normal((height, width, n_frames)) + 1.0
    mask = np.zeros((height, width), dtype=bool)
    mask[2:5, 2:5] = True

    masked = apply_mask(movie, mask)

    assert masked.shape == movie.shape
    assert np.allclose(masked[0, 0, :], 0.0)
    assert np.allclose(masked[3, 3, :], movie[3, 3, :])


def test_apply_mask_all_true_is_a_no_op():
    rng = np.random.default_rng(2)
    movie = rng.standard_normal((5, 5, 10))
    mask = np.ones((5, 5), dtype=bool)
    assert np.array_equal(apply_mask(movie, mask), movie)
