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


def test_constant_q_equals_OLS_over_q_hat() -> None:
    """ConstantQ at q = q̂ (empirical observation rate) should yield the MAR-corrected OLS.

    Under MAR with q constant:
        β̂_Christensen = E[XX']⁻¹ · E[(1/q) X Ỹ]
                      = E[XX']⁻¹ · (1/q) E[X Y]
                      = (1/q) · β_true                  (approximately, for large n)

    And the sample analog:
        β̂_Christensen ≈ (1/q̂) · β̂_OLS_on_observed_rows (wait: actually the
        observed-rows OLS estimates β̂ = E[XX']⁻¹ E[XY|responded], which under MAR
        equals β_true. So ACTUALLY: β̂_Christensen ≈ β̂_OLS_on_all_rows_with_true_y
        ≈ β̂_OLS_on_responded_rows under MAR.)

    Simplest formulation: under MAR with q constant, ChristensenEstimator(ConstantQ)
    should be close to OLS on the respondents-only subset.

    Deferred until ChristensenEstimator is implemented.
    """
    raise NotImplementedError


def test_constant_q_recovers_beta_true_on_large_sample() -> None:
    """Large-sample MAR: ChristensenEstimator(ConstantQ) should recover β_true."""
    raise NotImplementedError


def test_parametric2param_collapses_to_constantq_when_q0_eq_q1() -> None:
    """Setting q_0 = q_1 should give identical β̂ to ConstantQ with the same value."""
    raise NotImplementedError


def test_faithful_beats_DRO_variant_on_synthetic_MBOV() -> None:
    """Synthetic MBOV_Lower with known Q: faithful Christensen should outperform
    the DRO variant from minimax_core.

    This is the core claim of the package. Generate binary-y data with a known
    q(y) = decreasing function, fit both:
        (a) ChristensenEstimator(Parametric2ParamForBinary("increasing"))
        (b) ScoreMinimaxRegressor from phase1_pereira_benchmark/minimax_adapter.py

    Evaluate both on held-out test data. Faithful should have lower MSE on
    average over repeated seeds.
    """
    raise NotImplementedError
