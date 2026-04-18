"""Christensen minimax adapter for JTPA treatment effect estimation.

The JTPA task is different from Phases 1 and 2: instead of predicting Y, we
estimate a treatment effect (scalar coefficient on D in a regression of
Y = α + τD + Xβ + ε where Y is MNAR). The Christensen framework gives us the
β̂ = Mb + m estimator, with minimax-robust choice of (M, m) over a Q class.

The key adaptations for treatment-effect estimation:

1. **Build the regressor matrix as [1, D, X]** — treatment becomes a regressor
   alongside covariates.
2. **Extract the treatment coefficient** — τ̂ is the second entry of β̂.
3. **Compute CIs via bootstrap** — Christensen doesn't have closed-form SEs
   for the treatment coefficient the way Heckman does. We resample respondents
   and refit. (Alternative: influence function + delta method; more work.)
4. **Sensitivity analysis across Q families** — report the [min, max] of τ̂
   across plausible Q classes instead of a single point estimate.

This gives us the "Christensen bound" on the treatment effect that's
interpretable as "the range of τ̂ that's minimax-robust under assumptions
about the non-response mechanism."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass(frozen=True)
class SensitivityBound:
    """Christensen sensitivity bounds on a treatment effect."""
    tau_min: float
    tau_max: float
    tau_midpoint: float
    ci_lower: float  # bootstrap CI (accounts for sampling + minimax-over-Q variance)
    ci_upper: float
    q_classes_explored: List[str]


@dataclass
class ChristensenTreatmentEffect:
    """Adapter: Christensen minimax for treatment effect estimation on JTPA.

    Reports sensitivity bounds rather than point estimates, following
    Christensen & Connault (2023) Econometrica:
        Range of τ̂ across a plausible Q family quantifies how much the
        treatment effect depends on unverifiable non-response assumptions.

    Recommended Q family for JTPA:
        - ConstantQ (baseline: MAR assumption)
        - MonotoneInY("decreasing") (Heckman-style: low earners less likely
          to respond)
        - Custom Q class reflecting the selection direction empirically
          documented in the JTPA literature (response rate decreasing in
          earnings-distress indicators)
    """

    delta: float = 0.30  # Reference ball radius around q̂
    n_bootstrap: int = 500  # For CI estimation
    q_classes: List[str] = field(default_factory=lambda: [
        "ConstantQ",
        "MonotoneInY_decreasing",
        "MonotoneInY_decreasing_tight",  # δ=0.10 variant
    ])

    def estimate(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
        response_mask: np.ndarray,
    ) -> SensitivityBound:
        """Fit Christensen estimator across Q family and return sensitivity bounds.

        Implementation notes (for later):
            1. Build regressor matrix R = [1, D, X] (treatment in column 1)
            2. For each Q class in self.q_classes:
                 est = ChristensenEstimator(q_class=q_class, fit_intercept=False)
                 est.fit(R, Y, response_mask)
                 tau_this = est._beta[1]  # coefficient on D
               Collect all tau estimates.
            3. tau_min = min(tau_estimates), tau_max = max(tau_estimates)
            4. Bootstrap: resample with replacement from respondents; recompute
               min and max of tau across Q classes in each bootstrap; report
               95% percentile CI for each of tau_min and tau_max.
            5. Return SensitivityBound(tau_min, tau_max, midpoint, CI, q_classes)

        Honest framing: this is a sensitivity analysis, not a point estimate.
        A narrow [tau_min, tau_max] means the treatment effect is insensitive
        to non-response assumptions. A wide range means assumptions matter a
        lot — and the user should seek an exclusion restriction or accept
        that the treatment effect is only weakly identified.
        """
        raise NotImplementedError(
            "Implement after data is loaded and baselines are implemented. "
            "See module docstring for algorithm."
        )
