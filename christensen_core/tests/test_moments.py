"""Tests for christensen_core.moments.

These tests establish that the moment quantities computed from corrupted data
match the definitions in Christensen's PDF §1.3 Problem 2 (page 3) and the
IPW trick introduced on page 4.

Acceptance criteria:
    - compute_b_n matches hand-computed value on small example
    - compute_W_n matches X.T @ X / n exactly
    - compute_r_n excludes non-respondents (via response_mask)
    - compute_r_n matches hand-computed value under known q
    - IPW identity holds in expectation: E[(1/q) X Ỹ] == E[X Y] on large sample
"""

from __future__ import annotations

import numpy as np

from christensen_core.moments import compute_b_n, compute_W_n, compute_r_n


def test_b_n_manual() -> None:
    """Hand-computed small example for b_n."""
    X = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])  # n=3, d=2
    Y_tilde = np.array([10.0, 0.0, 20.0])  # second row is non-respondent
    # b_n = (1/3) * (X.T @ Y_tilde) = (1/3) * [1*10+1*0+1*20, 2*10+3*0+4*20]
    #      = (1/3) * [30, 100]
    expected = np.array([10.0, 100.0 / 3.0])
    actual = compute_b_n(X, Y_tilde)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_W_n_matches_XtX_over_n() -> None:
    """W_n = (1/n) X.T @ X, computed by numpy directly as the reference."""
    np.random.seed(0)
    X = np.random.randn(50, 4)
    expected = X.T @ X / X.shape[0]
    actual = compute_W_n(X)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_r_n_sums_respondents_only() -> None:
    """r_n(q) should sum only over indices where response_mask is True."""
    X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    Y_tilde = np.array([2.0, 0.0, 4.0])  # second is non-respondent
    q = np.array([0.5, 0.5, 0.5])  # arbitrary for non-respondent; ignored
    mask = np.array([True, False, True])
    # (1/n) * [(1/0.5)*X0*Y_tilde[0] + (1/0.5)*X2*Y_tilde[2]]
    # = (1/3) * [2*[1,1]*2 + 2*[1,3]*4]
    # = (1/3) * [4 + 8, 4 + 24] = (1/3) * [12, 28]
    expected = np.array([4.0, 28.0 / 3.0])
    actual = compute_r_n(X, Y_tilde, q, mask)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_r_n_ipw_identity_large_sample() -> None:
    """E[(1/q) X Ỹ] should equal E[X Y] when q is the TRUE response probability.

    Generate large IID sample, induce MNAR with known q, verify the sample
    r_n(q_true) converges to E[X Y] (computed on the uncorrupted data).
    """
    # NOTE ON SAMPLE SIZE: with tail-heavy IPW weights (q bounded at 0.2 → weights up
    # to 5) the Monte Carlo variance of r_n requires n ≳ 100k for rtol=0.05 to hold
    # reliably at every seed. At n=20k the test fails ~85% of seeds even though the
    # implementation is correct. We use n=100k and a tighter q bound to keep the test
    # a sharp correctness check (not a flakiness trap) — see
    # https://github.com/adawgwats/selective-observation-minimax/... for discussion.
    np.random.seed(42)
    n = 100_000
    X = np.random.randn(n, 3)
    X = np.concatenate([np.ones((n, 1)), X], axis=1)  # intercept
    beta_true = np.array([0.5, 1.0, -0.5, 0.3])
    Y = X @ beta_true + 0.1 * np.random.randn(n)

    # Response probability depends on y: q(y) = sigmoid(-y) bounded [0.2, 0.9]
    # (tighter lower bound vs [0.1, 0.9] halves the max IPW weight to 5 and drops
    # MC variance enough that rtol=0.05 is comfortably satisfied.)
    q_true_func = lambda y: np.clip(1.0 / (1.0 + np.exp(y)), 0.2, 0.9)
    q_true_per_example = q_true_func(Y)
    response = np.random.rand(n) < q_true_per_example

    Y_tilde = np.where(response, Y, 0.0)
    EXY_uncorrupted = X.T @ Y / n
    r_n_with_true_q = compute_r_n(X, Y_tilde, q_true_per_example, response)
    np.testing.assert_allclose(r_n_with_true_q, EXY_uncorrupted, rtol=0.05)
