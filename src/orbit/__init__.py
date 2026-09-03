"""orbit: pure-Python algorithm and validation-metric core for the
assessment-forward functional optical imaging pipeline.

No Qt/GUI dependencies live here (see orbitapp for the PySide6 shell) --
every stage and every metric is a standalone, independently testable
function, called by the GUI rather than implemented inside it.
"""

from .cnmf import CNMFResult, cnmf_source_extraction
from .cnmf_e import cnmf_e_source_extraction
from .denoising import (
    denoise_gaussian,
    denoise_median,
    denoise_wavelet_space,
    denoise_wavelet_time,
    residual_energy_fraction,
)
from .motion_correction import motion_correct, patch_motion_correct, rigid_motion_correct
from .motion_metrics import (
    enhanced_correlation_coefficient,
    mean_correlation_to_reference,
    mean_max_intensity_difference,
    spatiotemporal_svd,
)
from .neuropil import compute_neuropil_traces, neuropil_ring_mask
from .normalization import describe_normalization, normalize_movie, pixel_value_histogram, robust_std, summary_stats
from .patchwarp import patchwarp_motion_correct
from .projections import (
    fano_factor_projection,
    fano_factor_projection_volumetric,
    local_correlation_projection,
    mean_projection,
    mean_projection_volumetric,
    median_projection,
    median_projection_volumetric,
    mode_projection,
    mode_projection_volumetric,
    variance_projection,
    variance_projection_volumetric,
)
from .qc_traces import qc_trace_samples
from .roi_extraction_corr import CorrMaskResult, find_seed_candidates, roi_from_seed
from .roi_extraction_pca_ica import PCAICAResult, pca_ica_source_extraction

__all__ = [
    "mean_projection",
    "median_projection",
    "variance_projection",
    "fano_factor_projection",
    "local_correlation_projection",
    "mode_projection",
    "mean_projection_volumetric",
    "median_projection_volumetric",
    "variance_projection_volumetric",
    "fano_factor_projection_volumetric",
    "mode_projection_volumetric",
    "rigid_motion_correct",
    "patch_motion_correct",
    "patchwarp_motion_correct",
    "motion_correct",
    "mean_max_intensity_difference",
    "enhanced_correlation_coefficient",
    "mean_correlation_to_reference",
    "spatiotemporal_svd",
    "denoise_wavelet_time",
    "denoise_wavelet_space",
    "denoise_gaussian",
    "denoise_median",
    "qc_trace_samples",
    "residual_energy_fraction",
    "normalize_movie",
    "describe_normalization",
    "pixel_value_histogram",
    "robust_std",
    "summary_stats",
    "CorrMaskResult",
    "roi_from_seed",
    "find_seed_candidates",
    "compute_neuropil_traces",
    "neuropil_ring_mask",
    "PCAICAResult",
    "pca_ica_source_extraction",
    "CNMFResult",
    "cnmf_source_extraction",
    "cnmf_e_source_extraction",
]
