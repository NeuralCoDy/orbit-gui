import numpy as np

from orbit._patches import make_blocks_3d, make_patches_2d, patch_bounds_1d


def test_patch_bounds_1d_covers_frame_without_a_near_duplicate_final_patch():
    # 256 % 75 == 31, so the naive stride sequence [0, 75, 150] lands only
    # 6px short of the true edge (156) -- appending a *new* patch there
    # (rather than shifting the last one) would nearly double coverage for
    # 6 extra pixels of frame.
    starts = patch_bounds_1d(size=256, patch_extent=100, overlap=25)
    assert starts == [0, 75, 156]
    assert starts[-1] + 100 == 256  # last patch flush with the far edge
    assert starts[0] == 0

    # a size that fits inside one patch collapses to a single patch
    assert patch_bounds_1d(size=50, patch_extent=100, overlap=25) == [0]


def test_make_patches_2d_bounds_tile_the_full_frame():
    patches = make_patches_2d(height=120, width=90, patch_size=(70, 70), overlap=20)
    covered = np.zeros((120, 90), dtype=bool)
    for r0, r1, c0, c1 in patches:
        assert 0 <= r0 < r1 <= 120
        assert 0 <= c0 < c1 <= 90
        covered[r0:r1, c0:c1] = True
    assert covered.all()  # every pixel is inside at least one patch


def test_make_blocks_3d_bounds_tile_the_full_movie():
    blocks = make_blocks_3d(
        height=120, width=90, n_frames=300, block_size=(70, 70), block_frames=200,
        spatial_overlap=20, temporal_overlap=50,
    )
    covered = np.zeros((120, 90, 300), dtype=bool)
    for r0, r1, c0, c1, t0, t1 in blocks:
        assert 0 <= r0 < r1 <= 120
        assert 0 <= c0 < c1 <= 90
        assert 0 <= t0 < t1 <= 300
        covered[r0:r1, c0:c1, t0:t1] = True
    assert covered.all()  # every voxel is inside at least one block


def test_make_blocks_3d_collapses_to_one_block_when_movie_fits():
    blocks = make_blocks_3d(
        height=50, width=50, n_frames=100, block_size=(250, 250), block_frames=5000,
        spatial_overlap=30, temporal_overlap=500,
    )
    assert blocks == [(0, 50, 0, 50, 0, 100)]
