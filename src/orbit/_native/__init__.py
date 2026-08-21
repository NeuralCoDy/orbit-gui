"""Optional native (C++) accelerators. Functions in orbit fall back to
pure Python/numpy if the extension hasn't been built -- run
build_native.sh (requires a C++ compiler and `pip install pybind11`).
"""

from __future__ import annotations

try:
    from ._orbit_native import half_sample_mode as half_sample_mode_native
    from ._orbit_native import local_correlation as local_correlation_native
    from ._orbit_native import oasis_ar1 as oasis_ar1_native

    NATIVE_AVAILABLE = True
except ImportError:
    local_correlation_native = None
    half_sample_mode_native = None
    oasis_ar1_native = None
    NATIVE_AVAILABLE = False

__all__ = ["local_correlation_native", "half_sample_mode_native", "oasis_ar1_native", "NATIVE_AVAILABLE"]
