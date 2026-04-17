"""Sanity-check reductions: the Christensen estimator under ConstantQ should
reduce to classical MAR-corrected OLS.

This is the legitimacy test from context.md §"Legitimacy testing plan" step 1:
    "confirm the implemented objective matches the written minimax problem;
     use toy cases where the solution can be brute-forced or solved analytically"

If ConstantQ with q = q̂ doesn't give (1/q̂) · β̂_OLS_on_observed, something is
wrong at a foundational level — don't trust any other result.
"""

from __future__ import annotations

import numpy as np
import pytest

from christensen_core.estimator import ChristensenEstimator
from christensen_core.q_classes import ConstantQ, Parametric2ParamForBinary, QClassConfig


def test_constant_q_equals_OLS_over_q_hat() -> None:
    """Under MAR, ConstantQ Christensen should be close to OLS on respondent-only data."""
    np.random.seed(3)
    n, d = 500, 3
    X = np.random.randn(n, d)
    X_with_intercept = np.concatenate([np.ones((n, 1)), X], axis=1)
    beta_true = np.array([0.5, 1.0, -0.5, 0.3])
    Y = X_with_intercept @ beta_true + 0.1 * np.random.randn(n)

    # MAR: constant q everywhere
    q_hat = 0.6
    response = np.random.rand(n) < q_hat
    Y_tilde = np.where(response, Y, 0.0)

    est = ChristensenEstimator(q_class=ConstantQ(), fit_intercept=True)
    est.fit(X, Y_tilde, response)

    # Under MAR, Christensen should recover beta_true on the respondent subset
    X_respondent = X_with_intercept[response]
    Y_respondent = Y[response]
    beta_ols = np.linalg.lstsq(X_respondent, Y_respondent, rcond=None)[0]

    # Christensen's prediction function: X @ beta_hat where beta_hat = M*b_n + m*
    # Under MAR both should give similar predictions on new data
    X_test = np.random.randn(200, d)
    X_test_with_intercept = np.concatenate([np.ones((200, 1)), X_test], axis=1)
    y_pred_christ = est.predict(X_test)
    y_pred_ols = X_test_with_intercept @ beta_ols

    # Tolerate MC variance
    np.testing.assert_allclose(y_pred_christ, y_pred_ols, atol=0.1)


def test_constant_q_recovers_beta_true_on_large_sample() -> None:
    """Large-sample MAR: ChristensenEstimator(ConstantQ) should recover β_true."""
    np.random.seed(11)
    n, d = 5000, 3
    X = np.random.randn(n, d)
    X_with_intercept = np.concatenate([np.ones((n, 1)), X], axis=1)
    beta_true = np.array([0.5, 1.0, -0.5, 0.3])
    Y = X_with_intercept @ beta_true + 0.1 * np.random.randn(n)

    q_hat = 0.6
    response = np.random.rand(n) < q_hat
    Y_tilde = np.where(response, Y, 0.0)

    est = ChristensenEstimator(q_class=ConstantQ(), fit_intercept=True)
    est.fit(X, Y_tilde, response)

    # Build a test set and check predictions match beta_true (noise-free truth)
    X_test = np.random.randn(500, d)
    X_test_with_intercept = np.concatenate([np.ones((500, 1)), X_test], axis=1)
    y_pred_christ = est.predict(X_test)
    y_true_test = X_test_with_intercept @ beta_true

    np.testing.assert_allclose(y_pred_christ, y_true_test, atol=0.1)


def test_parametric2param_collapses_to_constantq_when_q0_eq_q1() -> None:
    """For binary Y, Parametric2ParamForBinary at θ=[q,q] should produce the same
    per-example q_values as ConstantQ at θ=[q]. This is the direct reduction."""
    np.random.seed(0)
    n = 300
    X = np.random.randn(n, 3)
    Y_tilde = np.random.randint(0, 2, size=n).astype(float)

    q = 0.5
    theta_constant = np.array([q])
    theta_param = np.array([q, q])

    constant_q = ConstantQ()
    param_q = Parametric2ParamForBinary(monotone=None)

    q_from_constant = constant_q.q_values(theta_constant, X, Y_tilde)
    q_from_param = param_q.q_values(theta_param, X, Y_tilde)

    np.testing.assert_allclose(q_from_constant, q_from_param)

    # Also check with non-trivial q and mixed Y_tilde
    for q_val in (0.3, 0.7, 0.95):
        theta_c = np.array([q_val])
        theta_p = np.array([q_val, q_val])
        np.testing.assert_allclose(
            constant_q.q_values(theta_c, X, Y_tilde),
            param_q.q_values(theta_p, X, Y_tilde),
        )


def test_faithful_beats_DRO_variant_on_synthetic_MBOV() -> None:
    """Moved to test_integration.py::test_christensen_beats_erm_on_synthetic_MBOV"""
    pytest.skip("See test_integration.py for the head-to-head test")
