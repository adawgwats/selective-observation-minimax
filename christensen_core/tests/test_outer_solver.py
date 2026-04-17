"""Tests for christensen_core.outer_solver.

Acceptance criteria:
    - ConstantQ outer max returns q* near the empirical non-observation rate
      complement on pure MAR data (where the adversary has no bite).
    - Parametric2ParamForBinary outer max returns q_0 < q_1 on data generated
      with MBOV_Lower-style selection.
    - Objective value at θ* is at least as high as at any initial grid point.
"""

from __future__ import annotations

import numpy as np

from christensen_core.inner_solver import inner_objective_value, solve_inner
from christensen_core.moments import compute_b_n, compute_r_n, compute_W_n
from christensen_core.outer_solver import solve_outer
from christensen_core.q_classes import (
    ConstantQ,
    Parametric2ParamForBinary,
    QClassConfig,
)


def _make_mbov_lower_dataset(
    n: int = 500,
    q0: float = 0.4,
    q1: float = 0.85,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a binary-Y dataset with MBOV_Lower-style selection:
    respondents with Y=0 have response probability q0, with Y=1 have q1.
    Returns (X_with_intercept, Y_tilde, response_mask).
    """
    rng = np.random.default_rng(seed)
    X_raw = rng.standard_normal((n, 3))
    X = np.concatenate([np.ones((n, 1)), X_raw], axis=1)
    beta_true = np.array([0.3, 0.8, -0.4, 0.2])
    y_continuous = X @ beta_true + 0.1 * rng.standard_normal(n)
    Y = (y_continuous > 0).astype(float)
    q_true = np.where(Y == 0, q0, q1)
    response = rng.random(n) < q_true
    Y_tilde = np.where(response, Y, 0.0)
    return X, Y_tilde, response.astype(bool)


def _inner_value_at(
    q_class,
    theta: np.ndarray,
    X: np.ndarray,
    Y_tilde: np.ndarray,
    response_mask: np.ndarray,
) -> float:
    b_n = compute_b_n(X, Y_tilde)
    W_n = compute_W_n(X)
    q_vec = q_class.q_values(theta, X, Y_tilde)
    r_n = compute_r_n(X, Y_tilde, q_vec, response_mask)
    M, m = solve_inner(b_n, W_n, r_n)
    return float(inner_objective_value(M, m, b_n, W_n, r_n))


def test_outer_respects_monotone_constraint() -> None:
    """For Parametric2ParamForBinary(monotone='increasing') on MBOV_Lower data,
    the returned θ* should satisfy θ_0 <= θ_1 up to tolerance."""
    X, Y_tilde, response_mask = _make_mbov_lower_dataset(n=500, q0=0.4, q1=0.85, seed=0)
    q_class = Parametric2ParamForBinary(
        monotone="increasing",
        config=QClassConfig(q_min=0.05, q_max=1.0),
    )

    result = solve_outer(q_class, X, Y_tilde, response_mask)

    assert result.theta_star.shape == (2,)
    tol = 1e-6
    assert result.theta_star[0] <= result.theta_star[1] + tol, (
        f"monotone constraint violated: theta_star={result.theta_star}"
    )


def test_outer_returns_nontrivial_theta_under_MNAR() -> None:
    """On data with strong MBOV_Lower selection, θ* should be far from the
    midpoint of the box (the neutral starting guess), reflecting the
    adversary's freedom to bias low observation rates at Y=0."""
    X, Y_tilde, response_mask = _make_mbov_lower_dataset(
        n=800, q0=0.3, q1=0.9, seed=1
    )
    config = QClassConfig(q_min=0.05, q_max=1.0)
    q_class = Parametric2ParamForBinary(monotone="increasing", config=config)

    result = solve_outer(q_class, X, Y_tilde, response_mask)

    midpoint = 0.5 * (config.q_min + config.q_max)
    dist = np.max(np.abs(result.theta_star - midpoint))
    assert dist >= 0.05, (
        f"theta_star={result.theta_star} too close to midpoint={midpoint};"
        f" max component distance={dist}"
    )


def test_outer_objective_improves_over_initial() -> None:
    """Final inner value at θ* >= inner value at the midpoint of the box."""
    X, Y_tilde, response_mask = _make_mbov_lower_dataset(
        n=500, q0=0.4, q1=0.85, seed=2
    )

    # Parametric2ParamForBinary case.
    config = QClassConfig(q_min=0.05, q_max=1.0)
    q_class_2p = Parametric2ParamForBinary(monotone="increasing", config=config)
    result_2p = solve_outer(q_class_2p, X, Y_tilde, response_mask)
    mid = 0.5 * (config.q_min + config.q_max)
    val_mid_2p = _inner_value_at(
        q_class_2p, np.array([mid, mid]), X, Y_tilde, response_mask
    )
    assert result_2p.inner_value_at_star >= val_mid_2p - 1e-9, (
        f"2param: value at theta*={result_2p.inner_value_at_star} < midpoint value={val_mid_2p}"
    )

    # ConstantQ case.
    q_class_c = ConstantQ(config=config)
    result_c = solve_outer(q_class_c, X, Y_tilde, response_mask)
    val_mid_c = _inner_value_at(
        q_class_c, np.array([mid]), X, Y_tilde, response_mask
    )
    assert result_c.inner_value_at_star >= val_mid_c - 1e-9, (
        f"constant: value at theta*={result_c.inner_value_at_star} < midpoint value={val_mid_c}"
    )
