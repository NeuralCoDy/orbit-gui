"""PatchWarp-style piecewise-affine motion correction (Hattori & Komiyama
2022, Cell Reports Methods, github.com/ryhattori/PatchWarp) -- a second
stage after rigid correction that fits an independent affine transform
per spatial patch per frame, correcting the slow non-uniform distortions
(tissue drift, dilation, shear) that a single whole-frame rigid shift
can't.

The per-patch affine fit (PatchWarp/utils/ecc_patchwarp/ecc_patchwarp.m)
is a direct application of the ECC algorithm (Evangelidis & Psarakis,
IEEE PAMI 2008) -- confirmed against the upstream source rather than
assumed. OpenCV implements the same algorithm natively as
cv2.findTransformECC, used here instead of hand-porting the MATLAB
Lucas-Kanade-style iteration; also gives a compiled, well-tested inner
loop for what would otherwise be the most expensive part of this stage.

Simplifications vs. upstream: each patch's affine transform is applied
only to its own non-overlapping "core" region (a wider, overlapping
"read" region is used just for the ECC fit itself, for more context/
texture), rather than PatchWarp's overlap-blended stitching; PatchWarp's
multi-session-split iterative template warm-starting and temporal
outlier cleanup (median filtering across frames, sudden-jump rejection)
aren't ported -- each (patch, frame) is fit independently, since the
rigid pre-pass here already removes most of the large motion that
warm-starting exists to help with.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .motion_correction import rigid_motion_correct

_DEFAULT_MAX_WORKERS = 8


def _patch_ranges(size: int, n_patches: int, overlap_frac: float) -> list[tuple[int, int, int, int]]:
    """(core_start, core_end, read_start, read_end) per patch along one
    axis. ``core`` is this patch's own non-overlapping territory (used
    to stitch the output back together); ``read`` is ``core`` expanded
    by the overlap fraction (used for the ECC fit, for more context),
    clamped to [0, size)."""
    core_size = int(np.ceil(size / n_patches))
    overlap = int(np.ceil(overlap_frac * core_size))
    ranges = []
    for i in range(n_patches):
        core_start = i * core_size
        core_end = min(size, core_start + core_size)
        read_start = max(0, core_start - overlap)
        read_end = min(size, core_end + overlap)
        ranges.append((core_start, core_end, read_start, read_end))
    return ranges


def _fit_affine_ecc(
    template: np.ndarray, frame: np.ndarray, n_iterations: int, pyramid_levels: int
) -> tuple[np.ndarray, float]:
    """Affine warp aligning ``frame`` to ``template`` via ECC
    (cv2.findTransformECC), coarse-to-fine over ``pyramid_levels``
    (1 = no pyramid). Returns (2x3 warp matrix, final rho); identity +
    rho=0.0 if ECC fails to converge (e.g. a low-texture patch) --
    mirrors ecc_patchwarp.m's warp_success handling, without its
    separate cross-frame temporal cleanup pass."""
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iterations, 1e-4)

    pyramid_t = [template]
    pyramid_f = [frame]
    for _ in range(pyramid_levels - 1):
        pyramid_t.append(cv2.pyrDown(pyramid_t[-1]))
        pyramid_f.append(cv2.pyrDown(pyramid_f[-1]))

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    rho = 0.0
    for level in range(pyramid_levels - 1, -1, -1):
        try:
            rho, warp_matrix = cv2.findTransformECC(
                pyramid_t[level], pyramid_f[level], warp_matrix, cv2.MOTION_AFFINE, criteria
            )
        except cv2.error:
            return np.eye(2, 3, dtype=np.float32), 0.0
        if level > 0:
            warp_matrix[:, 2] *= 2.0  # translation doubles moving to the next-finer level

    return warp_matrix, float(rho)


def _process_one_frame(
    frame: np.ndarray,
    template: np.ndarray,
    y_ranges: list[tuple[int, int, int, int]],
    x_ranges: list[tuple[int, int, int, int]],
    ecc_iterations: int,
    pyramid_levels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One frame's full patch grid: returns (warped_frame, affine_matrices)
    where affine_matrices is (ny, nx, 2, 3)."""
    ny, nx = len(y_ranges), len(x_ranges)
    out_frame = frame.copy()
    matrices = np.zeros((ny, nx, 2, 3), dtype=np.float32)
    matrices[:, :, 0, 0] = 1.0
    matrices[:, :, 1, 1] = 1.0

    for i, (cy0, cy1, ry0, ry1) in enumerate(y_ranges):
        for j, (cx0, cx1, rx0, rx1) in enumerate(x_ranges):
            template_patch = template[ry0:ry1, rx0:rx1]
            frame_patch = frame[ry0:ry1, rx0:rx1]
            warp_matrix, _rho = _fit_affine_ecc(template_patch, frame_patch, ecc_iterations, pyramid_levels)
            matrices[i, j] = warp_matrix

            warped_patch = cv2.warpAffine(
                frame_patch,
                warp_matrix,
                (rx1 - rx0, ry1 - ry0),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
            core_y0, core_y1 = cy0 - ry0, cy0 - ry0 + (cy1 - cy0)
            core_x0, core_x1 = cx0 - rx0, cx0 - rx0 + (cx1 - cx0)
            out_frame[cy0:cy1, cx0:cx1] = warped_patch[core_y0:core_y1, core_x0:core_x1]

    return out_frame, matrices


def patchwarp_motion_correct(
    movie: np.ndarray,
    template: np.ndarray | None = None,
    grid_size: int = 4,
    overlap_frac: float = 0.1,
    rigid_max_shift: float = 15.0,
    rigid_n_iter: int = 1,
    ecc_iterations: int = 50,
    pyramid_levels: int = 1,
    max_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """PatchWarp-style two-stage correction: whole-frame rigid
    registration (rigid_motion_correct), then an independent per-patch
    affine fit (ECC) on a ``grid_size`` x ``grid_size`` grid of
    overlapping patches, correcting local distortion the rigid stage
    can't.

    ``movie`` is (H, W, T). Returns ``(registered_movie, affine_matrices,
    template, initial_template)`` -- ``affine_matrices`` is (T,
    n_patches_y, n_patches_x, 2, 3), the per-patch-per-frame ECC warp
    (identity where a patch's fit didn't converge).

    Frames are registered against the same (rigid-corrected) template
    independently, so they run concurrently in a thread pool (OpenCV's
    C++ routines release the GIL); ``max_workers`` defaults to
    ``min(8, os.cpu_count())``.
    """
    rigid_registered, _shifts, rigid_template, initial_template = rigid_motion_correct(
        movie, template=template, max_shift=rigid_max_shift, n_iter=rigid_n_iter, init_batch=movie.shape[-1]
    )

    H, W, T = rigid_registered.shape
    y_ranges = _patch_ranges(H, grid_size, overlap_frac)
    x_ranges = _patch_ranges(W, grid_size, overlap_frac)
    ny, nx = len(y_ranges), len(x_ranges)

    frames_f32 = rigid_registered.astype(np.float32)
    template_f32 = rigid_template.astype(np.float32)
    workers = max_workers if max_workers is not None else min(_DEFAULT_MAX_WORKERS, os.cpu_count() or 1)

    registered = np.empty_like(rigid_registered)
    affine_matrices = np.zeros((T, ny, nx, 2, 3), dtype=np.float32)

    def _worker(t: int) -> tuple[int, np.ndarray, np.ndarray]:
        out_frame, matrices = _process_one_frame(
            frames_f32[:, :, t], template_f32, y_ranges, x_ranges, ecc_iterations, pyramid_levels
        )
        return t, out_frame, matrices

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for t, out_frame, matrices in pool.map(_worker, range(T)):
            registered[:, :, t] = out_frame
            affine_matrices[t] = matrices

    return registered, affine_matrices, rigid_template, initial_template
