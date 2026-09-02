"""Motion-correction quality metrics -- baked-in assessment for the
Motion Correction stage, not a separate QC pass. Two families:

- mMD / mean-correlation-to-reference / ECC, from PatchWarp (Hattori &
  Komiyama 2022, Cell Reports Methods): correlation- and intensity-based
  before/after scores.
- spatiotemporal_svd: singular-value-spectrum tightening and spatial PC
  maps -- a well-registered movie concentrates variance into fewer
  components (steeper spectrum decay); halo/crescent shapes in the PC
  maps are a visual sign of residual motion.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import svds


def _downsampled_max_projection(movie: np.ndarray, bin_size: int = 50) -> np.ndarray:
    """Max-intensity projection after averaging consecutive frames in
    groups of ``bin_size`` -- PatchWarp's noise-suppression step before
    computing mMD."""
    height, width, n_frames = movie.shape
    n_bins = max(1, n_frames // bin_size)
    trimmed = movie[:, :, : n_bins * bin_size]
    binned = trimmed.reshape(height, width, n_bins, -1).mean(axis=3)
    return binned.max(axis=2)


def mean_max_intensity_difference(before: np.ndarray, after: np.ndarray, bin_size: int = 50) -> float:
    """PatchWarp's mMD: difference in the mean of (noise-suppressed)
    max-intensity-projection images, post minus pre. More negative means
    registration reduced spurious bright-pixel spread from motion."""
    pre = _downsampled_max_projection(before, bin_size)
    post = _downsampled_max_projection(after, bin_size)
    return float(post.mean() - pre.mean())


def _pearson_correlations(reference: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Pearson correlation of a flat ``reference`` vector against each
    column of a (len(reference), n) matrix -- the one correlation
    computation enhanced_correlation_coefficient (n=1) and
    mean_correlation_to_reference (n=n_frames) both reduce to."""
    ref_c = reference - reference.mean()
    cols_c = columns - columns.mean(axis=0, keepdims=True)
    numer = ref_c @ cols_c
    denom = np.sqrt((ref_c**2).sum() * (cols_c**2).sum(axis=0))
    return np.divide(numer, denom, out=np.zeros(columns.shape[1]), where=denom > 0)


def enhanced_correlation_coefficient(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """PatchWarp's ECC: normalized cross-correlation between two images,
    invariant to bias/gain/contrast differences (mathematically Pearson
    correlation between the flattened, mean-centered images)."""
    a = image_a.ravel().astype(np.float64)
    b = image_b.ravel().astype(np.float64)
    return float(_pearson_correlations(a, b[:, None])[0])


def mean_correlation_to_reference(movie: np.ndarray, reference: np.ndarray | None = None) -> float:
    """PatchWarp's mCM: average per-frame ECC against a reference image.
    ``reference`` defaults to the movie's own mean projection ("self-mCM");
    pass e.g. a registration ``template`` for "cross-mCM". Higher means
    frames more consistently resemble the reference (better registration).
    """
    if reference is None:
        reference = movie.mean(axis=2)

    flat = movie.reshape(-1, movie.shape[-1]).astype(np.float64)
    ref = reference.ravel().astype(np.float64)
    return float(_pearson_correlations(ref, flat).mean())


def spatiotemporal_svd(movie: np.ndarray, n_components: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Truncated SVD of the (pixels x time) movie, mean-centered per
    pixel. Returns ``(singular_values, spatial_pc_maps)``:

    - ``singular_values``: descending, length <= ``n_components``. A
      tighter (faster-decaying) spectrum after motion correction means
      variance is concentrated in fewer components -- a sign correction
      worked; residual motion spreads variance across many components.
    - ``spatial_pc_maps``: (H, W, k) -- the left singular vectors
      reshaped to images. Crescent/halo shapes ringing cell bodies are a
      visual sign of residual (uncorrected) motion.

    Uses ``scipy.sparse.linalg.svds`` (ARPACK) rather than a full SVD,
    since only the top few components are needed and a full SVD of a
    (H*W, T) matrix is prohibitively expensive for realistic movie sizes.
    Stays in float64: ARPACK's iteration can fail to converge in float32
    when ``k`` approaches ``min(shape)`` (a short movie asking for nearly
    every component), so the narrower dtype isn't safe here -- the
    in-place mean-centering below already halves this function's own
    scratch (one (H*W, T) copy, not two).
    """
    height, width, n_frames = movie.shape
    flat = movie.reshape(-1, n_frames).astype(np.float64)  # .astype always copies -- safe to mutate below
    flat -= flat.mean(axis=1, keepdims=True)

    k = min(n_components, min(flat.shape) - 1)
    u, s, _vt = svds(flat, k=k)

    order = np.argsort(s)[::-1]
    singular_values = s[order]
    spatial_pc_maps = u[:, order].reshape(height, width, k)
    return singular_values, spatial_pc_maps
