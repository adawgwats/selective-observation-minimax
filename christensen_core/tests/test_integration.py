"""End-to-end integration test: ChristensenEstimator on synthetic MNAR data.

This is the "synthetic empirical validation" from context.md §"Legitimacy testing
plan" step 3. Generate data with a known q(x,y), fit the estimator with the
correct Q class, verify test MSE is lower than:
    - ERM (pure OLS on observed rows only)
    - MAR OLS correction (OLS_observed / q̂)

Acceptance threshold: ChristensenEstimator beats both on the MEAN of 30 seeds,
with 95% CI separation.
"""

from __future__ import annotations

import numpy as np


def test_christensen_beats_erm_on_synthetic_MBOV() -> None:
    """30-seed synthetic experiment:
        - n=500 training, n=500 test
        - X ~ N(0, I_3); β_true = [1, 0.5, -1, 0.3] (with intercept)
        - Y = X β_true + 0.2·noise
        - q(x, y) = increasing in Y (MBOV_Lower equivalent on LPM-like data)
        - Fit ChristensenEstimator(Parametric2ParamForBinary("increasing")) and OLS on observed
        - Report both test MSEs with 95% CIs.
        - Assert Christensen CI upper < OLS CI lower.
    """
    raise NotImplementedError


def test_christensen_ties_OLS_under_MAR() -> None:
    """Under MAR (q constant), ChristensenEstimator(ConstantQ) should tie OLS/q̂.
    No significant MSE difference across seeds."""
    raise NotImplementedError


def test_christensen_with_wrong_Q_degrades_gracefully() -> None:
    """If we fit Parametric2ParamForBinary("increasing") to MBOV_Higher data
    (where the truth is decreasing), MSE should be at least as good as OLS.
    The worst-case guarantee means we can't be catastrophically hurt by the
    class being too permissive, though we may lose the MNAR-correction benefit."""
    raise NotImplementedError
