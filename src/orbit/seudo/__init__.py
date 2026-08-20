"""SEUDO (Source Extraction Used to Distinguish Overlap) -- ported from the
`seudo` PyPI package (itself a Python port of the MATLAB SEUDO toolbox,
Gauthier & Charles, eLife 2021) for validating candidate ROI transients:
distinguishing a real calcium transient from one misattributed from an
overlapping/unmodeled nearby source. See core.SeudoData for the entry
point every other function in this package operates on.
"""

from __future__ import annotations

from .auto_classify import auto_classify_transients
from .classification_io import load_classification, load_classification_into, save_classification
from .constants import VAL_FALSE, VAL_MIX, VAL_TRUE, VAL_UNC, classification_color, cycle_classification, pick_color
from .core import SeudoData
from .estimate import estimate_time_courses_with_seudo
from .run_on_transients import run_seudo_restricted_to_transients
from .seudo_residual import compute_seudo_residual_fractions
from .transients import compute_transient_info, identify_transients

__all__ = [
    "SeudoData",
    "estimate_time_courses_with_seudo",
    "compute_transient_info",
    "identify_transients",
    "auto_classify_transients",
    "compute_seudo_residual_fractions",
    "run_seudo_restricted_to_transients",
    "save_classification",
    "load_classification",
    "load_classification_into",
    "VAL_TRUE",
    "VAL_FALSE",
    "VAL_MIX",
    "VAL_UNC",
    "pick_color",
    "classification_color",
    "cycle_classification",
]
