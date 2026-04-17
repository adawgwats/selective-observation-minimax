"""Mapping from Pereira et al. 2024 MNAR mechanism names to Christensen QClass instances.

This module encodes which Q class is theoretically appropriate for each MNAR
mechanism in the Pereira benchmark. The mapping matters because Christensen's
guarantees are WORST-CASE-OVER-Q — picking the wrong Q gives a worst-case
guarantee against the wrong nature.

## Mapping

| Pereira mechanism    | Best-matching QClass                           | Fidelity |
|----------------------|-----------------------------------------------|----------|
| MBOV_Lower           | Parametric2ParamForBinary(monotone="increasing")  | High — MBOV_Lower preferentially hides Y=0, i.e., q(0) < q(1), which is Christensen's "g increasing" (equivalently, "g decreasing" with opposite sign convention). |
| MBOV_Higher          | Parametric2ParamForBinary(monotone="decreasing") | High — hides Y=1 preferentially. |
| MBOV_Stochastic      | Parametric2ParamForBinary(monotone="increasing")  | Medium — 25% of missingness is MCAR noise, 75% is MBOV_Lower. The monotone constraint is still right but the theoretical best Q is a MIXTURE; a 2-parameter monotone class is a conservative approximation. |
| MBOV_Centered        | (no good match; deferred)                     | Low — non-monotone; requires a richer Q class. For binary y, "centered" is approximately MCAR (median is 0.5, ties broken arbitrarily) so `ConstantQ` is a reasonable stand-in. |
| MBUV                 | ConstantQ                                      | Medium — on binary label, MBUV generates missingness based on an unobserved N(0,1) feature independent of Y; so q(x,y) ≈ constant w.r.t. y. Treating as MAR is approximately correct. |
| MBIR_Frequentist     | (deferred; needs DependentOnUnobservedScore class) | Low — MBIR identifies an observable feature, correlates missingness with it, then HIDES that feature. The resulting MNAR structure is "q depends on an unobserved covariate." This needs a Q class we haven't scaffolded yet. |
| MBIR_Bayesian        | same as MBIR_Frequentist                      | Low |

## Implementation

A single function `q_class_for(mechanism_name, config=None)` returns the
appropriate QClass instance, or raises NotImplementedError for unsupported
mechanisms.

For v1, we implement the high-fidelity mappings (MBOV_Lower, MBOV_Higher) and
the MBUV=ConstantQ approximation. MBIR and MBOV_Centered are explicitly
deferred and their benchmark cells should be labeled as "Q-mismatch" in the
report rather than being treated as a test of Christensen's framework.
"""

from __future__ import annotations

from .q_classes import (
    ConstantQ,
    Parametric2ParamForBinary,
    QClass,
    QClassConfig,
)


FIDELITY = {
    "MBOV_Lower": "high",
    "MBOV_Higher": "high",
    "MBOV_Stochastic": "medium",
    "MBOV_Centered": "low",
    "MBUV": "medium",
    "MBIR_Frequentist": "low",
    "MBIR_Bayesian": "low",
}

# When the adaptive Q-specification pattern from reference_based_q.adaptive_centered_q_for
# is used (a Christensen-idiomatic neighborhood around q_hat, with radius calibrated to
# the mechanism's expected MNAR magnitude), fidelity changes for several mechanisms.
# The diagnostic at christensen_core/tests/diagnostic_centered_vs_wide.py shows MBUV-like
# MAR data is handled cleanly under adaptive delta=0.05 (MSE 2.6x OLS vs wide-box 26x OLS).
FIDELITY_ADAPTIVE = {
    "MBOV_Lower": "high",       # delta=0.30 ball captures true q-spread; monotone class correct
    "MBOV_Higher": "high",      # symmetric of above
    "MBOV_Stochastic": "high",  # delta=0.25 centered; MBOV-like signal w/ MCAR mix captured
    "MBOV_Centered": "medium",  # delta=0.05 gives near-MAR behavior; non-monotone truth still a mismatch but small
    "MBUV": "high",             # delta=0.05 centered ball ~ MAR solution; near-OLS/q_hat behavior
    "MBIR_Frequentist": "low",  # still requires DependentOnUnobservedScore class (v2)
    "MBIR_Bayesian": "low",
}


def q_class_for(mechanism: str, config: QClassConfig | None = None) -> QClass:
    """Return the Christensen QClass appropriate for the given Pereira mechanism.

    Args:
        mechanism: one of Pereira's mechanism names.
        config: optional QClassConfig; defaults to QClassConfig() (q_min=0.05).

    Returns:
        QClass instance configured for the mechanism.

    Raises:
        NotImplementedError: for mechanisms whose Q class is deferred to v2.
    """
    cfg = config  # pass through; QClasses default if None
    if mechanism in ("MBOV_Lower", "MBOV_Stochastic"):
        return Parametric2ParamForBinary(monotone="increasing", config=cfg)
    if mechanism == "MBOV_Higher":
        return Parametric2ParamForBinary(monotone="decreasing", config=cfg)
    if mechanism == "MBUV":
        return ConstantQ(config=cfg)
    if mechanism == "MBOV_Centered":
        # Non-monotone truth; ConstantQ is a MAR approximation (low fidelity).
        # Returning it allows the benchmark to run; REPORT.md must flag as low-fidelity.
        return ConstantQ(config=cfg)
    if mechanism in ("MBIR_Frequentist", "MBIR_Bayesian"):
        raise NotImplementedError(
            f"Mechanism {mechanism!r} requires a DependentOnUnobservedScore QClass "
            "that is not yet implemented (deferred to v2). See christensen_core/"
            "pereira_q.py docstring and IMPLEMENTATION_PLAN.md §'What we are "
            "deliberately NOT building in v1'."
        )
    raise ValueError(f"Unknown Pereira mechanism {mechanism!r}")
