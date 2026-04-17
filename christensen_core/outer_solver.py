"""Outer maximization: find θ* ∈ QClass maximizing the inner objective value.

For each QClass, we solve:

    max_θ  L(M*(θ), m*(θ); θ)

where (M*(θ), m*(θ)) = solve_inner(b_n, W_n, r_n(q_θ)) with q_θ derived from
the class-specific parameterization.

## Structure

The inner objective is NOT convex in θ in general. However, for specific
QClass structures it often has a tractable outer problem:

- `ConstantQ` (1D): golden-section search on [q_min, q_max].
- `Parametric2ParamForBinary` (2D): grid + polish, or direct bang-bang
  analysis (the inner objective for binary-y Christensen often has its
  maximum at a corner of the box; verify empirically).
- `MonotoneInY` (K-dim with ordering constraint): constrained optimization
  via scipy.optimize.minimize with SLSQP or an inequality-transformed space.

The solver dispatch is simple: each QClass has a companion optimizer. For v1
we implement the first two and defer MonotoneInY.

## Important guardrail: verify the reduction

Before any outer optimization, we verify:

1. For ConstantQ with θ = empirical observation rate q̂, the estimator
   equals the classical MAR-corrected OLS.
2. For Parametric2ParamForBinary with θ = [q̂, q̂], the estimator equals
   the ConstantQ solution at q̂.

These are tests in `tests/test_reduction_to_ols.py` and must pass before
trusting any outer-max output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .inner_solver import inner_objective_value, solve_inner
from .moments import compute_b_n, compute_r_n, compute_W_n
from .q_classes import ConstantQ, Parametric2ParamForBinary, QClass


@dataclass(frozen=True)
class OuterResult:
    """Result of the outer max-over-Q."""
    theta_star: np.ndarray
    M_star: np.ndarray
    m_star: np.ndarray
    beta_hat: np.ndarray
    inner_value_at_star: float
    n_inner_evaluations: int


def solve_outer(
    q_class: QClass,
    X: np.ndarray,
    Y_tilde: np.ndarray,
    response_mask: np.ndarray,
) -> OuterResult:
    """Dispatch to the class-specific outer optimizer. Returns the complete
    estimator artifact.

    Args:
        q_class: an instance of a QClass subclass.
        X: (n, d) feature matrix (intercept column already prepended if desired).
        Y_tilde: (n,) observed labels (0 for non-respondents).
        response_mask: (n,) bool array.

    Returns:
        OuterResult with θ*, (M*, m*), β̂, and bookkeeping.
    """
    if isinstance(q_class, ConstantQ):
        return _solve_outer_constant(q_class, X, Y_tilde, response_mask)
    if isinstance(q_class, Parametric2ParamForBinary):
        return _solve_outer_2param_binary(q_class, X, Y_tilde, response_mask)
    raise NotImplementedError(
        f"No outer solver for QClass={type(q_class).__name__}. Implement one and register."
    )


def _solve_outer_constant(
    q_class: ConstantQ,
    X: np.ndarray,
    Y_tilde: np.ndarray,
    response_mask: np.ndarray,
) -> OuterResult:
    """1D outer max for ConstantQ.

    Implementation sketch:
        1. Define inner_value(q_scalar) that calls moments.compute_moments,
           inner_solver.solve_inner, inner_solver.inner_objective_value.
        2. Run scipy.optimize.minimize_scalar(lambda q: -inner_value(q),
           bounds=(q_min, q_max), method='bounded').
        3. Recompute (M*, m*) and β̂ at the maximizer.
    """
    # b_n and W_n do NOT depend on q; compute once. Cache pinv(W_n) for the
    # fast inner objective evaluation (see _solve_outer_2param_binary for derivation).
    b_n = compute_b_n(X, Y_tilde)
    W_n = compute_W_n(X)
    W_n_pinv = np.linalg.pinv(W_n)
    low, high = q_class.theta_bounds()
    q_min = float(low[0])
    q_max = float(high[0])

    def neg_inner_value(q_scalar: float) -> float:
        theta = np.array([q_scalar], dtype=float)
        q_vec = q_class.q_values(theta, X, Y_tilde)
        r_n = compute_r_n(X, Y_tilde, q_vec, response_mask)
        beta_hat = W_n_pinv @ r_n
        return -(float(beta_hat @ W_n @ beta_hat - 2.0 * beta_hat @ r_n))

    result = minimize_scalar(
        neg_inner_value,
        bounds=(q_min, q_max),
        method="bounded",
        options={"xatol": 1e-6},
    )

    q_star = float(result.x)
    theta_star = np.array([q_star], dtype=float)
    q_vec = q_class.q_values(theta_star, X, Y_tilde)
    r_n = compute_r_n(X, Y_tilde, q_vec, response_mask)
    M_star, m_star = solve_inner(b_n, W_n, r_n)
    beta_hat = M_star @ b_n + m_star
    value = inner_objective_value(M_star, m_star, b_n, W_n, r_n)

    return OuterResult(
        theta_star=theta_star,
        M_star=M_star,
        m_star=m_star,
        beta_hat=beta_hat,
        inner_value_at_star=float(value),
        n_inner_evaluations=int(getattr(result, "nfev", 0)),
    )


def _solve_outer_2param_binary(
    q_class: Parametric2ParamForBinary,
    X: np.ndarray,
    Y_tilde: np.ndarray,
    response_mask: np.ndarray,
) -> OuterResult:
    """2D outer max for Parametric2ParamForBinary.

    Implementation sketch:
        1. Start with a coarse grid over (q_0, q_1) ∈ [q_min, q_max]² subject
           to the monotone constraint. A 20×20 grid with monotone filter gives
           ~200 candidates.
        2. Evaluate inner_value at each grid point; select top k (e.g., k=5).
        3. Polish each with scipy.optimize.minimize (method='L-BFGS-B') with
           bounds and (optionally) a linear inequality constraint for monotonicity.
        4. Return the best polished point.

    Subtle point: when q_class.monotone is 'increasing', we require q_0 <= q_1;
    'decreasing' requires q_0 >= q_1. None gives the full box. Grid + polish
    handles all three.
    """
    b_n = compute_b_n(X, Y_tilde)
    W_n = compute_W_n(X)
    low, high = q_class.theta_bounds()
    q_min = float(low[0])
    q_max = float(high[0])
    monotone = q_class.monotone

    n_evals = [0]

    # FAST PATH: the inner objective at its minimizer is equivalent to
    # -r_n' W_n^{-1} r_n (substituting β̂ = W_n^{-1} r_n into the quadratic).
    # This avoids the O(d^6) Kronecker vec-trick inside solve_inner; we still
    # recover the full (M*, m*) at the end via solve_inner for bookkeeping.
    # Correctness: PDF §1.5 states all valid (M, m) solutions give identical β̂.
    # For fast EVALUATION during the outer loop we only need the objective value,
    # not (M, m) themselves. Precompute a cached pinv(W_n) for repeated solves.
    W_n_pinv = np.linalg.pinv(W_n)

    def inner_value_at(theta: np.ndarray) -> float:
        q_vec = q_class.q_values(theta, X, Y_tilde)
        r_n = compute_r_n(X, Y_tilde, q_vec, response_mask)
        beta_hat = W_n_pinv @ r_n
        n_evals[0] += 1
        # inner objective at its minimizer: β̂' W β̂ - 2 β̂' r = -β̂' r = -r' W^{-1} r
        return float(beta_hat @ W_n @ beta_hat - 2.0 * beta_hat @ r_n)

    def feasible(theta: np.ndarray) -> bool:
        if monotone == "increasing":
            return theta[0] <= theta[1] + 1e-12
        if monotone == "decreasing":
            return theta[0] + 1e-12 >= theta[1]
        return True

    # Step 1: coarse grid search.
    qs = np.linspace(q_min, q_max, 15)
    candidates: list[tuple[float, np.ndarray]] = []
    for q0 in qs:
        for q1 in qs:
            theta = np.array([q0, q1], dtype=float)
            if not feasible(theta):
                continue
            val = inner_value_at(theta)
            candidates.append((val, theta))

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    top = candidates[:5]

    # Step 2: polish each of the top candidates.
    def neg_inner_value(theta_array: np.ndarray) -> float:
        return -inner_value_at(np.asarray(theta_array, dtype=float))

    bounds = [(q_min, q_max), (q_min, q_max)]
    best_val = -np.inf
    best_theta: np.ndarray | None = None

    for grid_val, theta0 in top:
        if monotone is None:
            res = minimize(
                neg_inner_value,
                x0=theta0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-9, "gtol": 1e-7},
            )
        else:
            # SLSQP with inequality constraint. For 'increasing' we need
            # q_1 - q_0 >= 0; for 'decreasing' we need q_0 - q_1 >= 0.
            if monotone == "increasing":
                constraint = {"type": "ineq", "fun": lambda t: t[1] - t[0]}
            else:
                constraint = {"type": "ineq", "fun": lambda t: t[0] - t[1]}
            res = minimize(
                neg_inner_value,
                x0=theta0,
                method="SLSQP",
                bounds=bounds,
                constraints=[constraint],
                options={"ftol": 1e-9},
            )

        theta_polished = np.asarray(res.x, dtype=float)
        # Guard: clamp into bounds and enforce monotone via projection if
        # the solver nudged just outside (tiny numerical slack).
        theta_polished = np.clip(theta_polished, q_min, q_max)
        if not feasible(theta_polished):
            # Fall back to the unpolished grid point, which is feasible by
            # construction.
            theta_polished = theta0
            polished_val = grid_val
        else:
            polished_val = -float(res.fun)

        # Compare polished vs grid-point value and take the better feasible option.
        if polished_val < grid_val:
            theta_polished = theta0
            polished_val = grid_val

        if polished_val > best_val:
            best_val = polished_val
            best_theta = theta_polished

    assert best_theta is not None, "grid search produced no feasible candidates"

    # Recompute artifacts at the best θ.
    q_vec = q_class.q_values(best_theta, X, Y_tilde)
    r_n = compute_r_n(X, Y_tilde, q_vec, response_mask)
    M_star, m_star = solve_inner(b_n, W_n, r_n)
    beta_hat = M_star @ b_n + m_star
    value = inner_objective_value(M_star, m_star, b_n, W_n, r_n)

    return OuterResult(
        theta_star=best_theta,
        M_star=M_star,
        m_star=m_star,
        beta_hat=beta_hat,
        inner_value_at_star=float(value),
        n_inner_evaluations=int(n_evals[0]),
    )
