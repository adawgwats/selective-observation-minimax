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
    raise NotImplementedError(
        "Implement dispatch:\n"
        "  MBOV_Lower / MBOV_Stochastic → Parametric2ParamForBinary(monotone='increasing')\n"
        "  MBOV_Higher → Parametric2ParamForBinary(monotone='decreasing')\n"
        "  MBUV / MBOV_Centered → ConstantQ (note the Q-mismatch for MBOV_Centered)\n"
        "  MBIR_* → raise NotImplementedError until a DependentOnUnobservedScore class lands"
    )
