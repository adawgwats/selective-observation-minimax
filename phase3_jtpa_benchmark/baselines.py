"""Baseline treatment effect estimators for the JTPA benchmark.

Each estimator implements a `fit_estimate` method that returns an ATT or ITT
estimate plus a standard error / confidence interval.

Methods:
    CompleteCaseOLS      — naive; fit OLS on responders only
    ExperimentalITT      — treatment group means on responders only
    HeckmanTwoStep       — two-step with selection correction
    IPWHorvitzThompson   — inverse probability weighting
    AIPWDoublyRobust     — augmented IPW (Robins-Rotnitzky-Zhao)
    DoubleML             — Chernozhukov et al. 2018 framework
    AbadieImbensMatching — nearest-neighbor matching on propensity score

All estimators take `(Y, D, X, response_mask)` and return an `EstimationResult`.

This is scaffold. Real implementations require:
    - statsmodels (Heckman, Probit/Logit)
    - sklearn (for propensity scores, outcome regression)
    - econml or DoubleML (for DML)
    - A numerically-stable matching routine (sklearn.neighbors + bias correction)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EstimationResult:
    """Standardized output from any treatment effect estimator."""
    att_estimate: float          # point estimate of ATT or ITT
    std_error: float             # standard error
    ci_lower: float              # 95% CI lower bound
    ci_upper: float              # 95% CI upper bound
    method: str                  # method name
    notes: str = ""


class BaseEstimator(ABC):
    """Abstract base: fit and return a treatment effect estimate."""

    @abstractmethod
    def fit_estimate(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
        response_mask: np.ndarray,
    ) -> EstimationResult:
        """Returns the ATT (or ITT) estimate.

        Args:
            Y: (n,) outcome (earnings). NaN or zero for non-respondents.
            D: (n,) treatment indicator.
            X: (n, d) covariates.
            response_mask: (n,) bool; True where Y is observed.
        """


class CompleteCaseOLS(BaseEstimator):
    """Naive: regress Y on [1, D, X] using only respondents.

    This is the simplest baseline and the most biased when non-response is
    outcome-correlated. Included for contrast.
    """

    def fit_estimate(self, Y, D, X, response_mask):
        raise NotImplementedError("Stub: implement OLS on observed rows + extract D coefficient")


class ExperimentalITT(BaseEstimator):
    """Group-means difference on responders; this is the 'naive ITT'."""

    def fit_estimate(self, Y, D, X, response_mask):
        raise NotImplementedError("Stub")


class HeckmanTwoStep(BaseEstimator):
    """Heckman (1979) two-step selection correction.

    Step 1: Probit of response_mask on (X, Z) where Z is an exclusion restriction.
            Compute inverse Mills ratio λ.
    Step 2: OLS of Y on (1, D, X, λ) on responders.

    Treatment effect is the D coefficient. This requires a valid exclusion
    restriction (a variable that predicts response but not outcome). For JTPA,
    common choices include site dummies (questionable) or interviewer fixed
    effects (not in public data). Document the choice in notes.
    """

    def fit_estimate(self, Y, D, X, response_mask):
        raise NotImplementedError("Stub: implement two-step with clear exclusion restriction")


class IPWHorvitzThompson(BaseEstimator):
    """Inverse-probability-weighting estimator.

    Fit response propensity p(R=1 | D, X) via logistic regression. Weight
    responders by 1/p̂. Report weighted mean of Y on each treatment arm;
    difference is the ATT.
    """

    def fit_estimate(self, Y, D, X, response_mask):
        raise NotImplementedError("Stub")


class AIPWDoublyRobust(BaseEstimator):
    """Augmented IPW (Robins-Rotnitzky-Zhao 1994).

    Combines outcome regression m(D, X) with response propensity p(D, X) in a
    way that's consistent if EITHER model is correct.
    """

    def fit_estimate(self, Y, D, X, response_mask):
        raise NotImplementedError("Stub")


class DoubleML(BaseEstimator):
    """Double/debiased machine learning (Chernozhukov et al. 2018).

    Uses cross-fitting to estimate nuisance functions (outcome, propensity)
    with arbitrary ML methods; recovers asymptotically efficient ATT.

    Implementation: use `econml` or `doubleml` Python packages.
    """

    def fit_estimate(self, Y, D, X, response_mask):
        raise NotImplementedError("Stub")


class AbadieImbensMatching(BaseEstimator):
    """Nearest-neighbor matching with bias correction (Abadie & Imbens 2006).

    Implementation: sklearn.neighbors + manual bias-correction step.
    """

    def fit_estimate(self, Y, D, X, response_mask):
        raise NotImplementedError("Stub")


REGISTRY = {
    "complete_case": CompleteCaseOLS,
    "experimental_itt": ExperimentalITT,
    "heckman": HeckmanTwoStep,
    "ipw": IPWHorvitzThompson,
    "aipw": AIPWDoublyRobust,
    "double_ml": DoubleML,
    "matching": AbadieImbensMatching,
}
