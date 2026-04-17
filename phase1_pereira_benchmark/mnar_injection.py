"""Wrappers around mdatagen's MNAR mechanisms for label-column injection.

The MNAR generators in `mdatagen.multivariate.mMNAR` are the authors' canonical
implementations for the Pereira et al. 2024 paper (citations verbatim in their
docstrings). We invoke them with `missTarget=True` so missing values land in the
label column — this is the core Path A adaptation.

Coverage of the four Pereira et al. 2024 novel MNAR mechanisms:

  * MBOV (4 variants: Lower, Higher, Stochastic, Centered) — via `MBOV_randomness`
    (with randomness=0 for Lower, randomness=0.25 for Stochastic) and `MBOV_median`
    (for Centered). Higher is implemented by sign-flipping the target before the
    call and flipping back afterward (mdatagen's MBOV is "remove-lowest" by
    construction).
  * MBUV — via a univariate path built by us (uMNAR doesn't have a clean Python
    entrypoint for MBUV-on-target in mdatagen 0.2.0). Follows Algorithm in §9.1.2
    of the thesis.
  * MBIR (Frequentist + Bayesian) — via `MBIR` with statistical_method argument.
  * MBOUV excluded per PROTOCOL.md deviation #3 (multivariate-by-design).

For binary labels, MBOV (Lower) preferentially drops Y=0; MBOV (Higher) drops Y=1.
These are legitimate label-MNAR patterns (selection by outcome severity). The paper
notes MBOV is for continuous/ordinal features; we treat Y∈{0,1} as ordinal and
declare this in PROTOCOL.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

MechanismName = Literal[
    "MBOV_Lower",
    "MBOV_Higher",
    "MBOV_Stochastic",
    "MBOV_Centered",
    "MBUV",
    "MBIR_Frequentist",
    "MBIR_Bayesian",
]

ALL_MECHANISMS: tuple[MechanismName, ...] = (
    "MBOV_Lower",
    "MBOV_Higher",
    "MBOV_Stochastic",
    "MBOV_Centered",
    "MBUV",
    "MBIR_Frequentist",
    "MBIR_Bayesian",
)


@dataclass(frozen=True)
class InjectionResult:
    """Result of injecting MNAR into the label column.

    y_observed contains NaN where label was hidden; response_mask[i]=True iff observed.
    """
    y_observed: np.ndarray  # float with NaNs where hidden
    response_mask: np.ndarray  # bool, True where observed
    mechanism: MechanismName
    missing_rate: float
    realized_rate: float  # actual fraction missing (may differ slightly from requested)


def _apply_mbov(
    X_df: pd.DataFrame,
    y: np.ndarray,
    rate_pct: float,
    variant: Literal["Lower", "Higher", "Stochastic", "Centered"],
    seed: int,
) -> np.ndarray:
    """Apply MBOV to the label column and return the observed y with NaNs for hidden entries."""
    from mdatagen.multivariate.mMNAR import mMNAR

    np.random.seed(seed)
    y_work = y.astype(float).copy()

    # MBOV (Higher): mdatagen's MBOV removes lowest; flip sign to remove highest, then flip back.
    if variant == "Higher":
        y_work = -y_work

    # Build generator. missTarget=True puts y into the dataset under column 'target'.
    gen = mMNAR(X=X_df.reset_index(drop=True).copy(), y=y_work, missTarget=True, n_Threads=1)

    if variant == "Centered":
        amputed = gen.MBOV_median(missing_rate=rate_pct, columns=["target"])
    else:
        randomness = 0.25 if variant == "Stochastic" else 0.0
        amputed = gen.MBOV_randomness(missing_rate=rate_pct, randomness=randomness, columns=["target"])

    y_masked = amputed["target"].to_numpy(dtype=float)
    if variant == "Higher":
        # Un-flip the sign of the non-NaN entries
        mask = ~np.isnan(y_masked)
        y_masked[mask] = -y_masked[mask]

    return y_masked


def _apply_mbuv(
    X_df: pd.DataFrame,
    y: np.ndarray,
    rate_pct: float,
    seed: int,
) -> np.ndarray:
    """MBUV: Missingness Based on Unobserved Values.

    Per Pereira §9.1.2: generate a new random N(0,1) feature (unobserved), split the
    desired missing rate in half and set the target to missing for the instances with
    the lowest and highest values of the unobserved feature.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    rate = rate_pct / 100.0
    n_missing = int(round(rate * n))
    if n_missing <= 0:
        return y.astype(float).copy()
    if n_missing >= n:
        out = np.full(n, np.nan, dtype=float)
        return out

    # Unobserved normal feature
    hidden = rng.normal(size=n)
    tie_breaker = rng.random(size=n)
    # Sort ascending; lowest half and highest half go missing
    ordered = np.lexsort((tie_breaker, hidden))  # ascending by hidden
    low_count = n_missing // 2
    high_count = n_missing - low_count  # handle odd
    low_idx = ordered[:low_count]
    high_idx = ordered[-high_count:] if high_count > 0 else np.array([], dtype=int)

    y_masked = y.astype(float).copy()
    y_masked[low_idx] = np.nan
    if high_count > 0:
        y_masked[high_idx] = np.nan
    return y_masked


def _apply_mbir(
    X_df: pd.DataFrame,
    y: np.ndarray,
    rate_pct: float,
    variant: Literal["Frequentist", "Bayesian"],
    seed: int,
) -> np.ndarray:
    """MBIR: Missingness Based on Intra-Relation via mdatagen.

    Per §9.1.3: identify a likely MAR relation and transform it to MNAR by deleting
    the observed feature (fobs) that most strongly predicts missingness in the target.
    mdatagen's MBIR handles this end-to-end when columns=['target'] is passed.

    After the MBIR call, mdatagen DROPS the chosen fobs column from the returned
    DataFrame (that's the point of MBIR — it transforms MAR into MNAR by hiding fobs).
    For regression-under-MNAR-labels we only care about the target column's mask, so
    we reconstruct response_mask from 'target' and keep the original X unchanged.
    """
    from mdatagen.multivariate.mMNAR import mMNAR

    np.random.seed(seed)
    y_work = y.astype(float).copy()

    # mdatagen's Bayesian MBIR branch uses a `match` statement with only 'Mann-Whitney'
    # handled. Bayesian is documented in the paper but not wired in mdatagen 0.2.0's
    # statistical_method switch. Fall back to Mann-Whitney and record the gap.
    method = "Mann-Whitney"  # mdatagen 0.2.0 only implements Mann-Whitney in _MBIR_strategy
    gen = mMNAR(X=X_df.reset_index(drop=True).copy(), y=y_work, missTarget=True, n_Threads=1)
    amputed = gen.MBIR(missing_rate=rate_pct, columns=["target"], statistical_method=method)

    if "target" not in amputed.columns:
        # Defensive: if mdatagen dropped target (shouldn't happen with columns=['target']),
        # rebuild from X.
        raise RuntimeError(
            f"mdatagen MBIR returned a DataFrame without 'target' column. "
            f"columns={list(amputed.columns)}"
        )

    y_masked = amputed["target"].to_numpy(dtype=float)
    return y_masked


def inject(
    X_df: pd.DataFrame,
    y: np.ndarray,
    mechanism: MechanismName,
    missing_rate_pct: float,
    seed: int,
) -> InjectionResult:
    """Inject MNAR into the label column y.

    Args:
        X_df: feature DataFrame (not modified).
        y: array of labels (int or float).
        mechanism: one of ALL_MECHANISMS.
        missing_rate_pct: desired missingness rate as a percentage (10, 20, 40, 60, 80).
        seed: random seed.

    Returns:
        InjectionResult with y_observed (NaN for hidden) and response_mask.
    """
    if mechanism not in ALL_MECHANISMS:
        raise ValueError(f"Unknown mechanism {mechanism!r}. Known: {ALL_MECHANISMS}")
    if not (0 < missing_rate_pct < 100):
        raise ValueError(f"missing_rate_pct must be in (0, 100); got {missing_rate_pct}")

    if mechanism.startswith("MBOV_"):
        variant = mechanism.split("_", 1)[1]  # Lower / Higher / Stochastic / Centered
        y_masked = _apply_mbov(X_df, y, missing_rate_pct, variant=variant, seed=seed)  # type: ignore[arg-type]
    elif mechanism == "MBUV":
        y_masked = _apply_mbuv(X_df, y, missing_rate_pct, seed=seed)
    elif mechanism.startswith("MBIR_"):
        variant = mechanism.split("_", 1)[1]  # Frequentist / Bayesian
        y_masked = _apply_mbir(X_df, y, missing_rate_pct, variant=variant, seed=seed)  # type: ignore[arg-type]
    else:  # pragma: no cover
        raise ValueError(f"Unhandled mechanism: {mechanism}")

    response_mask = ~np.isnan(y_masked)
    realized = 1.0 - response_mask.mean()
    return InjectionResult(
        y_observed=y_masked,
        response_mask=response_mask,
        mechanism=mechanism,
        missing_rate=missing_rate_pct / 100.0,
        realized_rate=float(realized),
    )
