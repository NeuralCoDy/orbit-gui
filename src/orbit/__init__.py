"""orbit: pure-Python algorithm and validation-metric core for the
assessment-forward functional optical imaging pipeline.

No Qt/GUI dependencies live here (see orbitapp for the PySide6 shell) --
every stage and every metric is a standalone, independently testable
function, called by the GUI rather than implemented inside it.
"""

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
from .normalization import normalize_movie, pixel_value_histogram, robust_std, summary_stats
from .patchwarp import patchwarp_motion_correct
from .projections import (
    fano_factor_projection,
    local_correlation_projection,
    mean_projection,
    median_projection,
    mode_projection,
    variance_projection,
)
from .qc_traces import qc_trace_samples

__all__ = [
    "mean_projection",
    "median_projection",
    "variance_projection",
    "fano_factor_projection",
    "local_correlation_projection",
    "mode_projection",
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
    "pixel_value_histogram",
    "robust_std",
    "summary_stats",
]
