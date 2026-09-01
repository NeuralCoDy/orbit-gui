"""Optional compiled accelerator for the per-cell FISTA solve -- see
fista_native.cpp's own module docstring for what it is and why it exists
(a from-scratch C++ port of this project's own fista_nonneg_weighted_l1,
NOT the upstream `seudo` package's different, measured-slower native
solver). Not required: estimate.py's _setup_cell_window/_solve_one_frame_
cell fall back to the pure-Python solver whenever NATIVE_AVAILABLE is
False here, which is the case until someone runs `bash build_native.sh`
in this directory (needs a C++14 compiler, pybind11, and FFTW3 -- see
that script's own header comment)."""

from __future__ import annotations

try:
    from . import _fista_native
    NATIVE_AVAILABLE = True
except ImportError:
    _fista_native = None
    NATIVE_AVAILABLE = False


def make_native_blob_conv(kernel, n_y: int, n_x: int):
    """A BlobConv handle for an (n_y, n_x) cell window -- build once per
    cell setup, reuse across every subsequent frame/iteration solve
    against that same window (see fista_native.cpp's BlobConv docstring).
    Only call this when NATIVE_AVAILABLE is True."""
    return _fista_native.BlobConv(kernel, n_y, n_x)


def fista_native(conv, rois, b, lam, tol: float, max_iter: int, l0: float = 1.0):
    """Solve minimize_{x>=0} 0.5*||Ax-b||^2 + lam.x given a pre-built
    BlobConv (see make_native_blob_conv) -- same problem/algorithm as
    fista_nonneg_weighted_l1 (solver.py), just compiled. Returns
    (weights, n_iter, L) -- pass L back in as the next call's l0 whenever
    the operator doesn't change between calls (e.g. the same known cell's
    window, frame to frame) to skip rediscovering the same backtracking
    step size from scratch every time (see StreamingState's own per-cell
    L cache). Only call this when NATIVE_AVAILABLE is True."""
    return _fista_native.fista_native(conv, rois, b, lam, tol, max_iter, l0)
