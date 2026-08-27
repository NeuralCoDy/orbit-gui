import numpy as np

from orbit.masking import apply_mask_3d, triangle_mask


def _bright_blob_volume(length=30, width=30, depth=10, seed=0):
    rng = np.random.default_rng(seed)
    volume = np.full((length, width, depth), 0.1)
    volume[10:20, 10:20, 3:7] = 5.0
    volume += rng.standard_normal((length, width, depth)) * 0.02
    return volume


def test_triangle_mask_on_a_genuine_3d_volume_keeps_the_blob_and_excludes_background():
    # triangle_mask itself needs no volumetric variant -- threshold_triangle
    # has no shape assumptions, so it already works unchanged here.
    volume = _bright_blob_volume()
    mask = triangle_mask(volume)

    assert mask.dtype == bool
    assert mask.shape == volume.shape
    assert mask[15, 15, 5]  # inside the bright blob
    assert not mask[0, 0, 0]  # dim background corner
    assert not mask[29, 29, 9]


def test_apply_mask_3d_zeroes_voxels_outside_the_mask_and_preserves_the_rest():
    rng = np.random.default_rng(1)
    length, width, depth, n_frames = 10, 10, 6, 20
    movie = rng.standard_normal((n_frames, length, width, depth)) + 1.0
    mask = np.zeros((length, width, depth), dtype=bool)
    mask[2:5, 2:5, 2:4] = True

    masked = apply_mask_3d(movie, mask)

    assert masked.shape == movie.shape
    assert np.allclose(masked[:, 0, 0, 0], 0.0)
    assert np.allclose(masked[:, 3, 3, 3], movie[:, 3, 3, 3])


def test_apply_mask_3d_all_true_is_a_no_op():
    rng = np.random.default_rng(2)
    movie = rng.standard_normal((5, 5, 5, 4))
    mask = np.ones((5, 5, 4), dtype=bool)
    assert np.array_equal(apply_mask_3d(movie, mask), movie)
