"""PCA (SVD) denoising for a (H, W, T) movie: split into overlapping
spatiotemporal blocks (each bounded by ``block_size``/``block_frames``,
since a full SVD of a whole large movie is intractable -- confirmed
directly: a full economy SVD of one 250x250x2000 block alone took 27s,
already slower than a truncated SVD of a full 250x250x5000 block),
reshape each block to (pixels, frames), keep only its top
``n_components`` singular components, and reconstruct. Overlapping
blocks' reconstructions are averaged back together to avoid visible
seams at block boundaries -- the same reasoning patch-based CNMF/GraFT
already use overlap for, just blending pixel values here instead of
resolving duplicate components.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD

from ._patches import make_blocks_3d


def _svd_reconstruct(matrix: np.ndarray, n_components: int) -> np.ndarray:
    """Low-rank reconstruction of ``matrix`` (rows, cols), keeping only
    its top ``n_components`` singular components. Uses a randomized
    truncated SVD (sklearn's TruncatedSVD) rather than a full
    ``numpy.linalg.svd``, since only a small number of components are
    ever needed but a full economy SVD would compute every one of
    ``min(rows, cols)`` regardless -- confirmed far slower at the block
    sizes this is used at (see module docstring). Falls back to a
    direct SVD only at the boundary case where ``n_components`` covers
    the whole rank, since TruncatedSVD requires strictly fewer
    components than ``min(matrix.shape)``."""
    k = min(n_components, min(matrix.shape))
    if k <= 0:
        return np.zeros_like(matrix)
    if k >= min(matrix.shape):
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        return (u[:, :k] * s[:k]) @ vt[:k, :]
    svd = TruncatedSVD(n_components=k, random_state=0)
    projected = svd.fit_transform(matrix)  # (rows, k), already U*S
    return projected @ svd.components_


def pca_denoise_block(block: np.ndarray, n_components: int) -> np.ndarray:
    """PCA-denoises one (h, w, t) block: reshape to (h*w, t), keep the
    top ``n_components`` singular components, reshape back."""
    height, width, n_frames = block.shape
    flat = np.asarray(block, dtype=np.float64).reshape(height * width, n_frames)
    denoised_flat = _svd_reconstruct(flat, n_components)
    return denoised_flat.reshape(height, width, n_frames)


def pca_denoise(
    movie: np.ndarray,
    n_components: int,
    block_size: tuple[int, int] = (250, 250),
    block_frames: int = 5000,
    spatial_overlap: int = 30,
    temporal_overlap: int = 500,
) -> np.ndarray:
    """PCA-denoises a whole (H, W, T) movie by tiling it into overlapping
    blocks (each no larger than ``block_size`` x ``block_frames``),
    running pca_denoise_block on each, and averaging overlapping
    reconstructions back into the full movie -- a movie already smaller
    than one block collapses to a single block, so this handles any
    movie size uniformly, whole-FOV or chunked, without a separate
    entry point for each."""
    height, width, n_frames = movie.shape
    blocks = make_blocks_3d(height, width, n_frames, block_size, block_frames, spatial_overlap, temporal_overlap)

    accum = np.zeros((height, width, n_frames))
    weight = np.zeros((height, width, n_frames))
    for r0, r1, c0, c1, t0, t1 in blocks:
        denoised_block = pca_denoise_block(movie[r0:r1, c0:c1, t0:t1], n_components)
        accum[r0:r1, c0:c1, t0:t1] += denoised_block
        weight[r0:r1, c0:c1, t0:t1] += 1.0

    return accum / np.maximum(weight, 1.0)
