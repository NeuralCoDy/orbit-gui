"""Tests for the optional compiled FISTA accelerator (orbit.seudo._native).
Not required for Real-SEUDO to work -- estimate.py falls back to the pure-
Python solver whenever it hasn't been built (`bash
src/orbit/seudo/_native/build_native.sh`), so this whole suite must pass
identically whether or not that build has happened here: correctness/
matching-output tests are skipped (not failed) when NATIVE_AVAILABLE is
False, while the fallback-safety tests always run.
"""
import numpy as np
import pytest

from orbit.seudo import _native
from orbit.seudo.blob import make_seudo_blob
from orbit.seudo.estimate import make_cached_blob_conv
from orbit.seudo.solver import fista_nonneg_weighted_l1
from orbit.seudo.streaming import DetectionParams, FitParams, PromotionParams, StreamingState, realSEUDOfit

requires_native = pytest.mark.skipif(
    not _native.NATIVE_AVAILABLE,
    reason="native FISTA accelerator not built -- see src/orbit/seudo/_native/build_native.sh",
)


def _single_blob_movie(mov_y=30, mov_x=30, n_frames=60, onset=10, amplitude=5.0, center=(15, 15), seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:mov_y, 0:mov_x]
    blob = np.exp(-((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / (2 * 2.0 ** 2))
    blob /= blob.max()
    activity = np.zeros(n_frames)
    activity[onset:] = amplitude
    movie = np.zeros((mov_y, mov_x, n_frames))
    for t in range(n_frames):
        movie[:, :, t] = blob * activity[t] + rng.normal(scale=0.05, size=(mov_y, mov_x))
    return movie, blob


def test_native_available_is_a_bool():
    # Whichever way it goes in this environment, it must be a plain bool
    # (not None, not an exception) -- the rest of the codebase branches on
    # it directly (e.g. StreamingState.use_native = fit.use_native and
    # _native.NATIVE_AVAILABLE).
    assert isinstance(_native.NATIVE_AVAILABLE, bool)


def test_streaming_with_use_native_true_never_errors_regardless_of_availability():
    # FitParams.use_native defaults to True -- this must be safe (silent
    # fallback to pure Python) whether or not the extension is built here,
    # per its own docstring's promise.
    movie, _blob = _single_blob_movie(n_frames=20)
    state = StreamingState(
        movie.shape[:2],
        fit=FitParams(sigma2=0.01, lambda_blob=5.0, blob_radius=2.0, pad_space=5, lookahead_frames=1),
        detection=DetectionParams(min_roi_size=5, cutoff_multiplier=3.0),
        promotion=PromotionParams(consecutive_frames_required=3),
    )
    assert state.use_native == _native.NATIVE_AVAILABLE  # use_native=True and NATIVE_AVAILABLE
    for t in range(movie.shape[-1]):
        realSEUDOfit(movie[:, :, t], state)  # must not raise either way


def test_streaming_with_use_native_false_never_uses_native_even_if_available():
    movie, _blob = _single_blob_movie(n_frames=20)
    state = StreamingState(
        movie.shape[:2],
        fit=FitParams(sigma2=0.01, lambda_blob=5.0, blob_radius=2.0, pad_space=5, lookahead_frames=1,
                      use_native=False),
        detection=DetectionParams(min_roi_size=5, cutoff_multiplier=3.0),
        promotion=PromotionParams(consecutive_frames_required=3),
    )
    assert state.use_native is False
    for t in range(movie.shape[-1]):
        realSEUDOfit(movie[:, :, t], state)


@requires_native
def test_native_blob_conv_constructs_for_a_real_kernel_and_window_size():
    # BlobConv doesn't expose its convolution directly to Python (only used
    # internally by fista_native) -- the actual convolution math is
    # exercised indirectly via the full-solve comparison tests below. This
    # just confirms construction succeeds for a real kernel/window size.
    kernel = make_seudo_blob(3.0)
    native_conv = _native.make_native_blob_conv(kernel, 33, 33)
    assert native_conv is not None


@requires_native
def test_native_fista_matches_python_fista_on_synthetic_problem():
    rng = np.random.default_rng(0)
    n_y, n_x, n_cells = 33, 33, 4
    kernel = make_seudo_blob(3.0)
    rois = rng.random((n_y * n_x, n_cells))
    lam = np.concatenate([np.full(n_cells, 0.0), np.full(n_y * n_x, 10.0 * 2 * 0.0020)])

    conv = make_cached_blob_conv(kernel, (n_y, n_x))

    def A(z):
        return rois @ z[:n_cells] + conv(z[n_cells:].reshape(n_y, n_x)).ravel()

    def At(v):
        return np.concatenate([rois.T @ v, conv(v.reshape(n_y, n_x)).ravel()])

    true_weights = np.concatenate([rng.uniform(1, 5, n_cells), rng.random(n_y * n_x) * 0.1])
    b = A(true_weights) + rng.normal(scale=0.02, size=n_y * n_x)

    x0 = np.zeros(n_cells + n_y * n_x)
    w_py, _n_iter, _L = fista_nonneg_weighted_l1(A, At, b, lam, x0, tol=0.01, max_iter=1000)

    native_conv = _native.make_native_blob_conv(kernel, n_y, n_x)
    w_native, _n_iter, _L_native = _native.fista_native(native_conv, rois, b, lam, 0.01, 1000, 1.0)

    r_py = A(w_py) - b
    r_native = A(w_native) - b
    obj_py = 0.5 * np.dot(r_py, r_py) + np.dot(lam, w_py)
    obj_native = 0.5 * np.dot(r_native, r_native) + np.dot(lam, w_native)
    # Same algorithm, same stopping criterion, ported line-for-line -- not
    # just "close", bit-identical to float64 precision.
    assert np.allclose(w_py, w_native, atol=1e-9)
    assert abs(obj_py - obj_native) < 1e-9


@requires_native
def test_realseudofit_with_native_finds_the_same_cell_as_pure_python():
    movie, _blob = _single_blob_movie(n_frames=30)
    fit_kwargs = dict(sigma2=0.01, lambda_blob=5.0, blob_radius=2.0, pad_space=5, lookahead_frames=1)
    detection = DetectionParams(min_roi_size=5, cutoff_multiplier=3.0)
    promotion = PromotionParams(consecutive_frames_required=3)

    state_native = StreamingState(
        movie.shape[:2], fit=FitParams(use_native=True, **fit_kwargs), detection=detection, promotion=promotion,
    )
    for t in range(movie.shape[-1]):
        realSEUDOfit(movie[:, :, t], state_native)

    state_python = StreamingState(
        movie.shape[:2], fit=FitParams(use_native=False, **fit_kwargs), detection=detection, promotion=promotion,
    )
    for t in range(movie.shape[-1]):
        realSEUDOfit(movie[:, :, t], state_python)

    assert state_native.profiles.shape[2] == state_python.profiles.shape[2] >= 1
    for i in range(state_native.profiles.shape[2]):
        assert np.allclose(state_native.profiles[:, :, i], state_python.profiles[:, :, i], atol=1e-6)
