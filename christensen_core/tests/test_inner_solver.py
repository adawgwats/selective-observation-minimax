"""Tests for christensen_core.inner_solver.

Acceptance criteria:
    - solve_inner on a tiny hand-computable example returns the expected (M, m)
      up to the non-uniqueness caveat from PDF §1.5.
    - For any output (M, m), plugging into inner_objective_value gives a value
      at least as low as any perturbation (verify via random perturbations).
    - The prediction β̂ = M·b + m is invariant to which member of the solution
      set the solver returned (also from §1.5).

Invariance test: if (M*, m*) and (M', m') are both solutions, they should give
the same β̂ = M·b + m (even though M and m themselves may differ).
"""

from __future__ import annotations

import numpy as np

# from christensen_core.inner_solver import solve_inner, inner_objective_value, predict_from_M_m


def test_solve_inner_produces_valid_solution() -> None:
    """For random (b, W, r), the solution should satisfy the FOC."""
    np.random.seed(0)
    d = 4
    b = np.random.randn(d)
    W = np.eye(d) + 0.1 * np.random.randn(d, d) @ np.random.randn(d, d).T
    r = np.random.randn(d)

    # M_star, m_star = solve_inner(b, W, r)
    # beta_hat = M_star @ b + m_star
    # # FOC: W beta_hat = r  (derived from the unconstrained quadratic min)
    # # Actually the FOC is stated in vec form; the reduced-form FOC for beta_hat is
    # # W beta_hat = r (see PDF page 7 bottom). Verify this.
    # np.testing.assert_allclose(W @ beta_hat, r, atol=1e-8)
    raise NotImplementedError


def test_minimum_at_solution_not_perturbation() -> None:
    """Objective at (M*, m*) should be less than at perturbations of (M*, m*)."""
    np.random.seed(1)
    d = 3
    b = np.random.randn(d)
    W = np.eye(d) * 2.0
    r = np.random.randn(d)

    # M_star, m_star = solve_inner(b, W, r)
    # obj_star = inner_objective_value(M_star, m_star, b, W, r)
    # for _ in range(10):
    #     M_pert = M_star + 0.01 * np.random.randn(*M_star.shape)
    #     m_pert = m_star + 0.01 * np.random.randn(*m_star.shape)
    #     assert inner_objective_value(M_pert, m_pert, b, W, r) >= obj_star - 1e-10
    raise NotImplementedError


def test_reduction_to_OLS_when_q_is_constant_and_full_response() -> None:
    """Smoke test: when every observation responds (response_mask all True) and
    q is 1.0 everywhere, the Christensen estimator should equal OLS on full data.

    Implementation will require moments.compute_r_n to be done first.
    """
    np.random.seed(2)
    n, d = 200, 3
    X = np.random.randn(n, d)
    X = np.concatenate([np.ones((n, 1)), X], axis=1)
    beta_true = np.array([1.0, 0.5, -1.0, 0.2])
    Y = X @ beta_true + 0.1 * np.random.randn(n)

    # q = np.ones(n)
    # mask = np.ones(n, dtype=bool)
    # b = compute_b_n(X, Y); W = compute_W_n(X); r = compute_r_n(X, Y, q, mask)
    # M_star, m_star = solve_inner(b, W, r)
    # beta_hat = M_star @ b + m_star
    # beta_ols = np.linalg.lstsq(X, Y, rcond=None)[0]
    # np.testing.assert_allclose(beta_hat, beta_ols, atol=1e-6)
    raise NotImplementedError
