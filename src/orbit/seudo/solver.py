"""FISTA solver for the nonnegative weighted-L1 problem, ported from the
`seudo` package's solver.py:

    minimize_{x >= 0}  0.5 * ||A(x) - b||^2 + lambda^T x

Since x is constrained nonnegative, the L1 penalty is linear on the
feasible set, so the proximal step is just a shifted ReLU.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def fista_nonneg_weighted_l1(
    A: Callable[[np.ndarray], np.ndarray],
    At: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    lam: np.ndarray,
    x0: np.ndarray,
    tol: float = 0.01,
    max_iter: int = 1000,
    l0: float = 1.0,
) -> tuple[np.ndarray, int, float]:
    """Minimize 0.5*||A(x) - b||^2 + lam^T x over x >= 0. A/At: forward and
    adjoint linear operators. lam: per-coordinate L1 weight, >= 0. Returns
    (x, n_iter, L) -- L (the final backtracking step-size/Lipschitz
    estimate) is worth passing back in as the NEXT call's l0 whenever A
    doesn't change between calls (e.g. the same known cell's window,
    frame to frame): L only ever grows within one solve, and the true
    Lipschitz constant of a fixed quadratic operator is the same every
    time, so restarting from l0=1.0 every call forces the backtracking
    search to rediscover the SAME L from scratch every time -- confirmed
    on real data this was over half of all forward-operator evaluations
    in the streaming (short, ~2.4-iteration) regime. See fista_native.cpp
    (seudo/_native) for the same optimization on the compiled path, and
    StreamingState's own per-cell L cache for how it's threaded through."""
    x = np.array(x0, dtype=float, copy=True)
    y = x.copy()
    t = 1.0
    L = l0
    n_iter = 0

    Ax = A(x)
    f_prev = 0.5 * np.dot(Ax - b, Ax - b) + np.dot(lam, x)

    for _ in range(max_iter):
        n_iter += 1
        Ay = A(y)
        resid = Ay - b
        grad = At(resid)
        fy = 0.5 * np.dot(resid, resid)

        while True:
            x_new = np.maximum(0.0, y - (grad + lam) / L)
            diff = x_new - y
            Ax_new = A(x_new)
            lhs = 0.5 * np.dot(Ax_new - b, Ax_new - b)
            rhs = fy + np.dot(grad, diff) + 0.5 * L * np.dot(diff, diff)
            if lhs <= rhs + 1e-12 or L > 1e10:
                break
            L *= 2.0

        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)

        f_new = lhs + np.dot(lam, x_new)
        rel_change = abs(f_new - f_prev) / max(1.0, abs(f_prev))

        x, t, f_prev = x_new, t_new, f_new

        if rel_change < tol:
            break

    return x, n_iter, L
