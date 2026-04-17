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
    raise NotImplementedError("See docstring for implementation outline")


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
    raise NotImplementedError("See docstring for implementation outline")
