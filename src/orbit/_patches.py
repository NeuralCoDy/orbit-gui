"""Overlapping spatial(/temporal) patch tiling, shared by every method
that splits a movie into bounded chunks and reassembles the result
afterward (patch-based CNMF, PCA denoising -- GraFT uses its own
``graft.construct_patches`` instead, since that's part of its own
package's public API).
"""

from __future__ import annotations


def patch_bounds_1d(size: int, patch_extent: int, overlap: int) -> list[int]:
    """Start offsets of overlapping patches of length ``patch_extent``
    tiling ``[0, size)`` -- the last one is pulled back to end exactly at
    ``size`` (rather than running past it) so every pixel is covered by
    at least one patch, matching however unevenly `size` divides. When
    the regular stride already lands within half a stride of that edge
    position, the last regular start is shifted to the edge instead of
    an extra patch being appended there -- otherwise a small remainder
    (e.g. tiling 256px with a 100px/25px-overlap patch leaves a 6px
    remainder) adds a near-duplicate patch just a few pixels over from
    the previous one, close to doubling total patch coverage for
    negligible extra frame coverage."""
    if size <= patch_extent:
        return [0]
    stride = max(1, patch_extent - overlap)
    starts = list(range(0, size - patch_extent + 1, stride))
    edge = size - patch_extent
    if starts[-1] != edge:
        # Shifting (rather than appending) only when there's already a
        # second-to-last start to keep the near-0 edge covered -- if
        # starts is just [0], shifting it away from 0 would leave [0,
        # edge) uncovered entirely, since nothing else covers that end.
        if len(starts) > 1 and edge - starts[-1] < stride / 2:
            starts[-1] = edge
        else:
            starts.append(edge)
    return starts


def make_patches_2d(
    height: int, width: int, patch_size: tuple[int, int], overlap: int
) -> list[tuple[int, int, int, int]]:
    """(row0, row1, col0, col1) bounds of every patch tiling (height, width)."""
    patch_h, patch_w = min(patch_size[0], height), min(patch_size[1], width)
    row_starts = patch_bounds_1d(height, patch_h, overlap)
    col_starts = patch_bounds_1d(width, patch_w, overlap)
    return [(r0, r0 + patch_h, c0, c0 + patch_w) for r0 in row_starts for c0 in col_starts]


def make_blocks_3d(
    height: int,
    width: int,
    n_frames: int,
    block_size: tuple[int, int],
    block_frames: int,
    spatial_overlap: int,
    temporal_overlap: int,
) -> list[tuple[int, int, int, int, int, int]]:
    """(row0, row1, col0, col1, t0, t1) bounds of every block tiling
    (height, width, n_frames), overlapping by ``spatial_overlap`` in row
    and column and ``temporal_overlap`` in time -- the 3D generalization
    of make_patches_2d, for methods (PCA denoising) that also need to
    bound how many frames one chunk covers, not just its spatial
    extent."""
    block_h, block_w = min(block_size[0], height), min(block_size[1], width)
    block_t = min(block_frames, n_frames)
    row_starts = patch_bounds_1d(height, block_h, spatial_overlap)
    col_starts = patch_bounds_1d(width, block_w, spatial_overlap)
    t_starts = patch_bounds_1d(n_frames, block_t, temporal_overlap)
    return [
        (r0, r0 + block_h, c0, c0 + block_w, t0, t0 + block_t)
        for r0 in row_starts
        for c0 in col_starts
        for t0 in t_starts
    ]
