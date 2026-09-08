"""Photobleaching/intensity-drift correction: divides the movie by a
slowly-varying trend of its own field-of-view-average intensity, so a
monotonic (or otherwise slow) drift in overall brightness doesn't get
mistaken for real signal by later stages (Normalization, Source
Extraction). Multiplicative (divide), matching how photobleaching
actually behaves -- fluorophore depletion scales brightness down over
time, it doesn't just shift it by a constant -- not an additive
subtraction.

Two methods build that trend, both operating on the same (T,) FOV-average
trace and producing a (T,) trend the rest of the pipeline (detrend_movie/
detrend_movie_3d -- shared unmodified by both methods, and by
DetrendingTab's own chunked-Commit path) treats identically:

- running_percentile_trend: a *trailing* (causal) running percentile --
  trend[t] only depends on frames up to and including t, never frames
  after it, the same convention as a causal rolling-baseline filter
  (e.g. Suite2p's rolling-percentile baseline). Nonparametric and local
  -- tracks whatever shape the drift actually has, at the cost of
  needing a window-size parameter and having no single "decay rate" to
  report.

- exponential_trend: a single global exponential a*exp(-t/b) fit to the
  WHOLE trace at once (not causal/windowed -- every frame's fit sees
  every other frame), via fit_exponential_trend's robust asymmetric-
  Huber optimization (see that function's own docstring). Parametric --
  assumes photobleaching's own textbook shape rather than tracking
  arbitrary drift, in exchange for two directly-interpretable numbers
  (initial brightness a, decay time constant b) instead of a window
  size, and no locality artifacts from a moving window.

fov_average_trace/detrend_movie are the (H, W, T) 2D versions;
fov_average_trace_3d/detrend_movie_3d are the (T, L, W, D) volumetric
counterparts (mean over all three spatial axes per volume instead of
the two per frame) -- both trend functions are dimension-agnostic (just
a 1D trace in, 1D trend out), so they're shared unmodified.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .cnmf_deconvolution import estimate_noise_std


def fov_average_trace(movie: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """(T,) mean pixel intensity per frame across the field of view,
    normalized to the FIRST frame's own mean (so the trace and trend
    both start at 1.0 and read directly as "fraction of initial
    brightness" -- comparable across movies of very different absolute
    intensity scale). Purely a display/inspection convenience: rescaling
    a trace by a positive constant changes neither running_percentile_trend's
    trend shape nor detrend_movie's actual `scale = trend / trend.mean()`
    correction (both percentile and mean commute with positive scalar
    multiplication, so the constant cancels out of that ratio) --
    confirmed via test_detrend_movie_correction_is_unaffected_by_the_
    trace_s_own_normalization.

    ``mask`` (H, W) bool, e.g. from a committed Mask stage -- restricts
    the average to True pixels only, so blank background/empty FOV space
    doesn't dilute a real intensity change happening only in the imaged
    tissue. None (the default) averages every pixel, unmasked."""
    movie = np.asarray(movie, dtype=np.float64)
    trace = movie.mean(axis=(0, 1)) if mask is None else movie[mask].mean(axis=0)
    baseline = trace[0]
    return trace / baseline if baseline != 0 else trace


def fov_average_trace_3d(movie: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """(T,) mean voxel intensity per volume -- (T, L, W, D) counterpart of
    fov_average_trace; ``mask`` is (L, W, D). See that function's
    docstring for the first-volume normalization and mask semantics,
    identical here."""
    movie = np.asarray(movie, dtype=np.float64)
    trace = movie.mean(axis=(1, 2, 3)) if mask is None else movie[:, mask].mean(axis=1)
    baseline = trace[0]
    return trace / baseline if baseline != 0 else trace


def running_percentile_trend(trace: np.ndarray, percentile: float, window: int) -> np.ndarray:
    """Trailing-window Xth-percentile trend of a 1D trace: trend[t] is
    the ``percentile``-th percentile of trace[max(0, t-window+1) : t+1].
    The interior (once a full window is available, t >= window-1) is
    computed in one vectorized call via a sliding-window view rather
    than a per-frame Python loop; only the short growing-window prefix
    (t < window-1, at most window-1 frames) is looped, since that
    region can't share the same fixed-size window shape."""
    trace = np.asarray(trace, dtype=np.float64)
    n = len(trace)
    window = int(np.clip(window, 1, max(n, 1)))
    trend = np.empty(n)

    for t in range(min(window - 1, n)):
        trend[t] = np.percentile(trace[: t + 1], percentile)

    if n >= window:
        windows = np.lib.stride_tricks.sliding_window_view(trace, window)  # (n-window+1, window)
        trend[window - 1 :] = np.percentile(windows, percentile, axis=-1)

    return trend


def _asymmetric_huber(r: np.ndarray, delta_pos: float, delta_neg: float) -> np.ndarray:
    """Elementwise asymmetric Huber loss of residual ``r`` -- standard
    (symmetric) Huber generalized to its own independent transition
    threshold on each side of zero: ``delta_pos`` where ``r > 0`` (the
    observed trace sits ABOVE the fit -- e.g. a real calcium transient),
    ``delta_neg`` where ``r < 0`` (the fit sits above the trace -- the
    fitted baseline overshooting the true signal).

    Ordinary Huber has the well-known identity H'(r) = clip(r, -delta,
    delta), i.e. its derivative is just the residual clipped to
    [-delta, delta]; this generalizes cleanly to independent per-side
    thresholds:

        clipped = clip(r, -delta_neg, delta_pos)
        J(r)  = 0.5*clipped**2 + clipped*(r - clipped)
        J'(r) = clipped

    (the ``clipped*(r - clipped)`` term is exactly zero inside the
    quadratic region, since clipped == r there, and reduces to the
    standard linear-beyond-delta Huber term once r is clipped -- see
    fit_exponential_trend's docstring for why this asymmetry, and
    _asymmetric_huber_grad below for the matching derivative).

    A SMALL delta_pos keeps the positive side's quadratic region narrow,
    so it transitions to a shallow-sloped linear regime quickly -- a
    large positive deviation (transient) barely moves the objective,
    i.e. is "allowed". A LARGE delta_neg keeps the negative side's
    quadratic region wide, so even a moderately negative residual is
    still penalized quadratically (steeply) -- "only small negative
    deviations" stay cheap; anything more costs increasingly more,
    pulling the fit down to track the trace's lower envelope rather than
    its mean."""
    clipped = np.clip(r, -delta_neg, delta_pos)
    return 0.5 * clipped**2 + clipped * (r - clipped)


def _asymmetric_huber_grad(r: np.ndarray, delta_pos: float, delta_neg: float) -> np.ndarray:
    """d/dr of _asymmetric_huber -- see its own docstring for the
    clip-based identity this is just clip(r, -delta_neg, delta_pos)."""
    return np.clip(r, -delta_neg, delta_pos)


def _initial_exponential_guess(trace: np.ndarray, t: np.ndarray, noise_std: float) -> tuple[float, float]:
    """Ordinary-least-squares log-linear fit (log(trace) ~ log(a) - t/b)
    as a starting point for fit_exponential_trend's own nonlinear,
    robust optimization below -- fast and closed-form, even though it's
    itself biased by any positive transients in ``trace`` (unlike the
    final fit, which is robust to them by design). Good enough as JUST a
    starting point: the actual asymmetric-Huber optimization corrects
    for that bias.

    Restricted to frames where ``trace`` is still clearly above the
    noise floor (> 3 noise-stds) -- confirmed via a real failure: once a
    fast decay (b much smaller than the recording length) has dropped
    the trace down into the noise for most of the recording, an
    unweighted OLS fit over the WHOLE trace lets that long, uninformative
    near-zero/floor-clipped tail (with thousands of points, all carrying
    the same degenerate "no more signal here" non-information) dominate
    the fitted slope, corrupting the starting point badly enough that
    the following optimization gets stuck there too (a real case: an
    8000-frame trace with a true b=150 fit an initial b in the millions
    without this restriction). Falls back to the whole trace if fewer
    than 2 points clear the floor (e.g. an already near-fully-decayed or
    very noisy trace has no informative segment left to restrict to).
    Falls back to a flat-decay guess (the whole trace span as the time
    constant) if the log-linear fit doesn't come out decaying at all
    (e.g. a trace with no photobleaching, or dominated by upward
    transients) -- b must stay positive regardless."""
    floor = max(3.0 * noise_std, max(trace.max(), 1.0) * 1e-6)
    informative = trace > floor
    if informative.sum() < 2:
        informative = np.ones_like(trace, dtype=bool)
    log_trace = np.log(np.clip(trace[informative], floor, None))
    design = np.stack([np.ones(informative.sum()), t[informative]], axis=1)
    (log_a0, slope), *_ = np.linalg.lstsq(design, log_trace, rcond=None)
    a0 = max(np.exp(log_a0), floor)
    b0 = -1.0 / slope if slope < 0 else max(t[-1], 1.0)
    return float(a0), float(b0)


def fit_exponential_trend(
    trace: np.ndarray, delta_pos_sigma: float = 1.0, delta_neg_sigma: float = 3.0, max_iter: int = 200,
) -> tuple[float, float]:
    """Fits ``a, b`` (both constrained positive) minimizing
    ``sum_t J(trace[t] - a*exp(-t/b))``, ``J`` the asymmetric Huber loss
    above -- a single global exponential decay robustly tracking the
    trace's own lower envelope rather than its mean, so real activity
    (positive-going transients riding on top of the photobleaching decay)
    doesn't pull the fitted decay upward the way an ordinary (symmetric)
    least-squares fit would.

    ``delta_pos_sigma``/``delta_neg_sigma`` are the Huber transition
    thresholds, each a multiple of the trace's own estimated noise std
    (orbit.cnmf_deconvolution.estimate_noise_std, a Welch-PSD high-
    frequency-band estimate already used for exactly this purpose
    elsewhere in this codebase) rather than a raw intensity value --
    keeps the defaults meaningful regardless of a movie's own absolute
    brightness scale or normalization. The default asymmetry
    (delta_neg_sigma=3 vs delta_pos_sigma=1) means residuals up to 1
    noise-std above the fit are cheap AND residuals up to 3 noise-stds
    below the fit are cheap, but beyond those thresholds the fit is
    pulled toward the trace much more strongly from above (a transient
    pushing the trace up) than from below (a dip pushing it down) --
    see _asymmetric_huber's own docstring for the mechanics.

    ``a`` and ``b`` are optimized in log-space (unconstrained ``log_a``,
    ``log_b``) so the ``a > 0, b > 0`` constraints are enforced exactly,
    with no bounded/constrained-optimizer machinery needed. Uses
    scipy.optimize.minimize (L-BFGS-B) with an analytic gradient (the
    Huber loss's own kink makes finite-difference gradients right at the
    transition thresholds less reliable, and an analytic gradient is
    cheap here regardless -- see _exponential_trend_objective_and_grad).

    Returns ``(a, b)``; ``b`` is in frame units (the number of frames
    for the fit to decay by a factor of ``1/e``, i.e. its own scale is
    always relative to the trace's own frame rate, not physical time)."""
    trace = np.asarray(trace, dtype=np.float64)
    n = len(trace)
    t = np.arange(n, dtype=np.float64)

    noise_std = estimate_noise_std(trace) or (trace.std() or 1.0)  # a constant trace has zero noise std either way
    delta_pos = delta_pos_sigma * noise_std
    delta_neg = delta_neg_sigma * noise_std

    a0, b0 = _initial_exponential_guess(trace, t, noise_std)
    x0 = np.array([np.log(a0), np.log(b0)])

    result = minimize(
        _exponential_trend_objective_and_grad, x0, args=(t, trace, delta_pos, delta_neg),
        jac=True, method="L-BFGS-B", options={"maxiter": max_iter},
    )
    log_a, log_b = result.x
    return float(np.exp(log_a)), float(np.exp(log_b))


def _exponential_trend_objective_and_grad(
    log_params: np.ndarray, t: np.ndarray, trace: np.ndarray, delta_pos: float, delta_neg: float,
) -> tuple[float, np.ndarray]:
    """(loss, gradient) w.r.t. (log_a, log_b) for fit_exponential_trend's
    own optimization -- computed together (scipy.optimize.minimize's own
    ``jac=True`` convention) since both need the same model/residual."""
    log_a, log_b = log_params
    a, b = np.exp(log_a), np.exp(log_b)
    model = a * np.exp(-t / b)
    r = trace - model

    loss = _asymmetric_huber(r, delta_pos, delta_neg).sum()

    d_loss_d_r = _asymmetric_huber_grad(r, delta_pos, delta_neg)
    # d(model)/da = exp(-t/b) = model/a ; d(model)/db = model * t / b**2
    # r = trace - model, so d(r)/d(.) = -d(model)/d(.)
    d_loss_d_a = np.sum(d_loss_d_r * -(model / a))
    d_loss_d_b = np.sum(d_loss_d_r * -(model * t / b**2))
    # chain rule to log-space: da/d(log_a) = a, db/d(log_b) = b
    grad = np.array([d_loss_d_a * a, d_loss_d_b * b])
    return loss, grad


def exponential_trend(
    trace: np.ndarray, delta_pos_sigma: float = 1.0, delta_neg_sigma: float = 3.0, max_iter: int = 200,
) -> np.ndarray:
    """fit_exponential_trend, evaluated as a (T,) trend array -- the
    exponential-decay counterpart of running_percentile_trend, same
    (T,) trace in/(T,) trend out convention."""
    trace = np.asarray(trace, dtype=np.float64)
    a, b = fit_exponential_trend(trace, delta_pos_sigma, delta_neg_sigma, max_iter)
    t = np.arange(len(trace), dtype=np.float64)
    return a * np.exp(-t / b)


def detrend_movie(
    movie: np.ndarray, percentile: float = 8.0, window: int = 200, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full pipeline: fov_average_trace -> running_percentile_trend ->
    divide every pixel's trace by the trend, rescaled to the trend's own
    mean so the movie's overall intensity scale is preserved (only the
    slow drift is removed, not the average brightness level). Returns
    (corrected movie, fov trace, trend) -- the trace/trend are what the
    GUI plots. ``mask`` (H, W) bool restricts fov_average_trace to True
    pixels only -- see that function's docstring."""
    trace = fov_average_trace(movie, mask)
    trend = running_percentile_trend(trace, percentile, window)
    scale = trend / trend.mean()
    corrected = movie / scale[None, None, :]
    return corrected, trace, trend


def detrend_movie_3d(
    movie: np.ndarray, percentile: float = 8.0, window: int = 200, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Volumetric (T, L, W, D) counterpart of detrend_movie -- ``mask``
    is (L, W, D). See fov_average_trace_3d for the per-volume average."""
    trace = fov_average_trace_3d(movie, mask)
    trend = running_percentile_trend(trace, percentile, window)
    scale = trend / trend.mean()
    corrected = movie / scale[:, None, None, None]
    return corrected, trace, trend


def detrend_movie_exponential(
    movie: np.ndarray, delta_pos_sigma: float = 1.0, delta_neg_sigma: float = 3.0, max_iter: int = 200,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Full pipeline, exponential-decay counterpart of detrend_movie:
    fov_average_trace -> fit_exponential_trend -> divide every pixel's
    trace by the trend, rescaled to the trend's own mean (same
    scale-preserving correction as detrend_movie -- see its own
    docstring). Returns (corrected movie, fov trace, trend, a, b) -- the
    two extra values (the fitted initial brightness and decay time
    constant) aren't needed to apply the correction, only to report
    them (e.g. a GUI label showing the fitted decay rate); every other
    return value matches detrend_movie's own convention. ``mask`` (H, W)
    bool restricts fov_average_trace to True pixels only -- see that
    function's docstring."""
    trace = fov_average_trace(movie, mask)
    a, b = fit_exponential_trend(trace, delta_pos_sigma, delta_neg_sigma, max_iter)
    t = np.arange(len(trace), dtype=np.float64)
    trend = a * np.exp(-t / b)
    scale = trend / trend.mean()
    corrected = movie / scale[None, None, :]
    return corrected, trace, trend, a, b


def detrend_movie_3d_exponential(
    movie: np.ndarray, delta_pos_sigma: float = 1.0, delta_neg_sigma: float = 3.0, max_iter: int = 200,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Volumetric (T, L, W, D) counterpart of detrend_movie_exponential --
    ``mask`` is (L, W, D). See fov_average_trace_3d for the per-volume
    average."""
    trace = fov_average_trace_3d(movie, mask)
    a, b = fit_exponential_trend(trace, delta_pos_sigma, delta_neg_sigma, max_iter)
    t = np.arange(len(trace), dtype=np.float64)
    trend = a * np.exp(-t / b)
    scale = trend / trend.mean()
    corrected = movie / scale[:, None, None, None]
    return corrected, trace, trend, a, b
