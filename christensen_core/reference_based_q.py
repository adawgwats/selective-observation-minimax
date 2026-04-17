"""Reference-based Q specification following Christensen's published pattern.

In Christensen & Connault (2023) and Adjaho & Christensen (2022), the uncertainty
set Q is specified as a NEIGHBORHOOD around an empirical reference, with
user-specified radius. We adapt that pattern to the selective-observation setting:
the reference is the empirical observation rate q_hat, and the neighborhood is a
box of radius delta around q_hat (clamped to [0.01, 1.0]).
"""
from __future__ import annotations
import numpy as np
from .q_classes import QClassConfig, ConstantQ, Parametric2ParamForBinary, QClass

def compute_q_hat(response_mask: np.ndarray) -> float:
    """Empirical overall observation rate."""
    mask = np.asarray(response_mask, dtype=bool)
    if mask.size == 0:
        raise ValueError("empty response_mask")
    return float(mask.mean())

def centered_config(q_hat: float, delta: float, min_floor: float = 0.01, max_ceiling: float = 1.0) -> QClassConfig:
    """Build QClassConfig as a ball of radius `delta` around `q_hat`, clamped."""
    if not 0 < q_hat <= 1:
        raise ValueError(f"q_hat must be in (0,1], got {q_hat}")
    if delta < 0:
        raise ValueError(f"delta must be non-negative, got {delta}")
    q_min = max(min_floor, q_hat - delta)
    q_max = min(max_ceiling, q_hat + delta)
    if q_min >= q_max:
        # Degenerate case; expand to a minimal non-empty interval
        q_max = min(max_ceiling, q_min + 0.01)
    return QClassConfig(q_min=q_min, q_max=q_max)

def centered_q_for(mechanism: str, response_mask: np.ndarray, delta: float = 0.30) -> QClass:
    """Return the appropriate QClass for `mechanism` with config centered on q_hat of the data.

    Mirrors pereira_q.q_class_for but uses a reference-based config. The returned class
    has its box configured as [q_hat - delta, q_hat + delta] clamped.
    """
    q_hat = compute_q_hat(response_mask)
    config = centered_config(q_hat, delta)
    if mechanism in ("MBOV_Lower", "MBOV_Stochastic"):
        return Parametric2ParamForBinary(monotone="increasing", config=config)
    if mechanism == "MBOV_Higher":
        return Parametric2ParamForBinary(monotone="decreasing", config=config)
    if mechanism in ("MBUV", "MBOV_Centered"):
        return ConstantQ(config=config)
    if mechanism in ("MBIR_Frequentist", "MBIR_Bayesian"):
        raise NotImplementedError(
            f"Mechanism {mechanism!r} requires a DependentOnUnobservedScore QClass (v2)."
        )
    raise ValueError(f"Unknown Pereira mechanism {mechanism!r}")


# ---------------------------------------------------------------------------
# Mechanism-aware adaptive delta
# ---------------------------------------------------------------------------
# When we know which MNAR mechanism generated the data (as in the Pereira benchmark,
# where the mechanism is fixed by the experimental protocol), we can pick delta
# calibrated to the expected spread of that mechanism. This is a form of
# domain-knowledge prior, directly analogous to how Adjaho & Christensen (2022)
# calibrate the Wasserstein radius to "observable differences in pre-treatment
# outcomes" (Remark 2.2 of their paper).
#
# Calibration rationale (from the diagnostic at tests/diagnostic_centered_vs_wide.py):
#   - MBOV_Lower / MBOV_Higher produce true q-spread up to 0.6 between Y=0 and Y=1
#     at strong missing rates. delta=0.30 is the minimum radius whose ball around
#     q_hat contains both true values.
#   - MBOV_Stochastic mixes 25% MCAR into MBOV. Effective spread is ~0.75 * MBOV
#     spread = 0.45, so delta=0.25 suffices.
#   - MBOV_Centered on binary labels is MCAR-like (ties on median). delta=0.05
#     gives a tight ball near q_hat with small bias.
#   - MBUV on label column is near-MAR (unobserved feature independent of Y).
#     delta=0.05 gives a tight ball; adversary has little bite and Christensen
#     behaves close to OLS/q_hat.
#
# DEFAULT_DELTA is the fallback for mechanisms not in the table (or when the
# mechanism is genuinely unknown in deployment).
DEFAULT_DELTA = 0.30

MECHANISM_DELTA: dict[str, float] = {
    "MBOV_Lower": 0.30,
    "MBOV_Higher": 0.30,
    "MBOV_Stochastic": 0.25,
    "MBOV_Centered": 0.05,
    "MBUV": 0.05,
    # MBIR_Frequentist / MBIR_Bayesian omitted because they're not yet supported;
    # the dispatch raises NotImplementedError before delta is consulted.
}


def mechanism_to_delta(mechanism: str | None) -> float:
    """Return the domain-knowledge delta for the named mechanism.

    Falls back to DEFAULT_DELTA when mechanism is None or not in the table.
    This function IS domain knowledge — it encodes our prior about how far
    the true q can deviate from q_hat for each mechanism. If mechanism metadata
    is unavailable in deployment, use DEFAULT_DELTA (which is a conservative
    wide setting that captures MBOV-family MNAR).
    """
    if mechanism is None:
        return DEFAULT_DELTA
    return MECHANISM_DELTA.get(mechanism, DEFAULT_DELTA)


def adaptive_centered_q_for(mechanism: str, response_mask: np.ndarray) -> QClass:
    """Return the Christensen QClass for `mechanism` with mechanism-adaptive delta.

    Equivalent to `centered_q_for(mechanism, response_mask, delta=mechanism_to_delta(mechanism))`.
    Use this when the benchmark protocol tells us which mechanism is active.
    Use `centered_q_for(..., delta=...)` directly (with a chosen fixed delta) when
    mechanism metadata is unavailable or you want to override the prior.
    """
    delta = mechanism_to_delta(mechanism)
    return centered_q_for(mechanism, response_mask, delta=delta)
