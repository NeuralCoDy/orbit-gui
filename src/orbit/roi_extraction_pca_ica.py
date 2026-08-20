"""PCA-ICA source extraction (Mukamel et al. 2009), ported from CaImAn's
movies.IPCA_stICA + rois.extractROIsFromPCAICA as closely as possible.
Adapted from CaImAn's (T, H, W) movie convention and bespoke Movie
subclass to orbit's plain (H, W, T) ndarray.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, label
from sklearn.decomposition import FastICA, IncrementalPCA

from ._masks import EIGHT_CONNECTED, masked_mean_trace


@dataclass
class PCAICAResult:
    masks: list[np.ndarray]  # each (H, W) bool
    traces: list[np.ndarray]  # each (T,) float, plain masked-mean of the fluorescence movie


def spatiotemporal_ica_components(
    movie: np.ndarray,
    n_pca_components: int = 50,
    n_ica_components: int = 40,
    batch_size: int = 1000,
    mu: float = 0.5,
    ica_fun: str = "logcosh",
) -> np.ndarray:
    """PCA followed by spatiotemporal ICA: PCA on every pixel's time trace
    gives a temporal basis (``eigenseries``, T x n_pca_components) and a
    spatial basis (``eigenframes``, H*W x n_pca_components); stacking both
    together before one shared ICA unmixing (rather than ICA on either
    alone) is what makes the resulting components respect both spatial
    and temporal structure. ``mu`` in [0, 1] trades weight between the
    spatial (mu -> 1) and temporal (mu -> 0) halves of that stack.
    Returns (n_ica_components, H, W)."""
    height, width, n_frames = movie.shape
    pixel_traces = movie.reshape(-1, n_frames).astype(np.float64)  # (H*W, T)

    ipca = IncrementalPCA(n_components=n_pca_components, batch_size=batch_size)
    ipca.fit(pixel_traces)
    projected = ipca.inverse_transform(ipca.transform(pixel_traces))  # (H*W, T)
    eigenseries = ipca.components_.T  # (T, n_pca_components) -- temporal basis
    eigenframes = projected @ eigenseries  # (H*W, n_pca_components) -- spatial basis

    # series_scale intentionally normalizes by eigenframes' own max (not
    # eigenseries') -- a quirk in the upstream CaImAn source, preserved
    # here rather than silently "fixed", per this project's usual ported-
    # code policy.
    frame_scale = mu / eigenframes.max()
    n_eigenframes = frame_scale * (eigenframes - eigenframes.mean(axis=0))
    series_scale = (1 - mu) / eigenframes.max()
    n_eigenseries = series_scale * (eigenseries - eigenseries.mean(axis=0))

    joint = np.concatenate([n_eigenframes, n_eigenseries])  # (H*W + T, n_pca_components)
    joint_ics = FastICA(n_components=n_ica_components, fun=ica_fun).fit_transform(joint)

    spatial_ics = joint_ics[: height * width, :]  # (H*W, n_ica_components)
    return spatial_ics.T.reshape(n_ica_components, height, width)


def extract_masks_from_components(
    components: np.ndarray, num_std: float = 4.0, gaussian_sigma: float = 2.0, thresh: float | None = None
) -> list[np.ndarray]:
    """Turns spatial ICA components into candidate ROI masks: Gaussian-
    smooth each component, threshold by a robust (IQR-based) outlier
    cutoff -- or an explicit ``thresh`` if given -- and take every
    connected component of the thresholded blob as its own mask."""
    masks = []
    for comp in components:
        smoothed = gaussian_filter(comp, gaussian_sigma)
        median = np.median(smoothed)
        iqr = np.subtract(*np.percentile(smoothed, [75, 25]))
        pos_thresh = median + num_std * iqr / 1.35
        neg_thresh = median - num_std * iqr / 1.35
        if thresh is None:
            signed = smoothed * (smoothed > pos_thresh) - smoothed * (smoothed < neg_thresh)
        else:
            signed = smoothed * (smoothed > thresh) - smoothed * (smoothed < -thresh)
        labeled, n_labels = label(signed > 0, structure=EIGHT_CONNECTED)
        masks.extend(labeled == i for i in range(1, n_labels + 1))
    return masks


def pca_ica_source_extraction(
    movie: np.ndarray,
    n_pca_components: int = 50,
    n_ica_components: int = 40,
    mu: float = 0.5,
    num_std: float = 4.0,
    batch_size: int = 1000,
) -> PCAICAResult:
    """Runs the full pipeline: spatiotemporal ICA components -> ROI masks
    -> masked-mean traces (the same trace convention every extraction
    method uses, for a consistent review-panel look)."""
    components = spatiotemporal_ica_components(
        movie, n_pca_components=n_pca_components, n_ica_components=n_ica_components, batch_size=batch_size, mu=mu
    )
    masks = extract_masks_from_components(components, num_std=num_std)
    traces = [masked_mean_trace(movie, mask) for mask in masks]
    return PCAICAResult(masks=masks, traces=traces)
