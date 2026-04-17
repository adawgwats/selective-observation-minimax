"""Sample moments from the corrupted training data.

Corresponds to PDF §1.3 Problem 2 and page 5. Given corrupted data (Xᵢ, Ỹᵢ) where
Ỹᵢ = εᵢ·Yᵢ (with εᵢ the response indicator) and a response function q(X, Ỹ),
compute:

    b_n    = (1/n) Σᵢ Xᵢ Ỹᵢ                     # (d-vector)
    W_n    = (1/n) Σᵢ Xᵢ Xᵢᵀ                    # (d × d)
    r_n(q) = (1/n) Σᵢ∈Rₙ (1/q(Xᵢ, Ỹᵢ)) Xᵢ Ỹᵢ    # (d-vector, sum over respondents only)

Rₙ is the set of indices where εᵢ = 1 (responders). For non-respondents, Ỹᵢ = 0
in Christensen's formulation, so their terms drop from both b_n (X·0 = 0) and
r_n (they are excluded from the sum by definition of Rₙ).

All three quantities are computable from (X, Ỹ) and the response_mask. Wₙ does
not depend on q or Ỹ. bₙ depends on Ỹ but not q. rₙ depends on q(·, Ỹ) for
respondents.

The `d` in the PDF is the dimension of Xᵢ. If we include an intercept, we
prepend a column of 1s to X in the adapter, and d becomes d_features + 1.

Implementation is numpy-only. No SGD, no iterative procedures — these are
closed-form sample moments.
"""

from __future__ import annotations

import numpy as np


def compute_b_n(X: np.ndarray, Y_tilde: np.ndarray) -> np.ndarray:
    """b_n = (1/n) Σᵢ Xᵢ Ỹᵢ.

    PDF page 3, Problem 2.

    Args:
        X: (n, d) feature matrix including any intercept column.
        Y_tilde: (n,) observed labels with Y_tilde[i] = 0 for non-respondents.

    Returns:
        (d,) vector.
    """
    n = X.shape[0]
    return (X.T @ Y_tilde) / n


def compute_W_n(X: np.ndarray) -> np.ndarray:
    """W_n = (1/n) Σᵢ Xᵢ Xᵢᵀ.

    PDF page 3, Problem 2. Sample analog of E[X X'].

    Args:
        X: (n, d) feature matrix.

    Returns:
        (d, d) symmetric positive semidefinite matrix.
    """
    n = X.shape[0]
    return (X.T @ X) / n


def compute_r_n(
    X: np.ndarray,
    Y_tilde: np.ndarray,
    q_values: np.ndarray,
    response_mask: np.ndarray,
) -> np.ndarray:
    """r_n(q) = (1/n) Σᵢ∈Rₙ (1/q(Xᵢ, Ỹᵢ)) Xᵢ Ỹᵢ.

    PDF page 4–5. Sum is over respondents only (Rₙ = {i : εᵢ = 1}); non-respondents
    are excluded because Christensen sets Ỹᵢ = 0 for them (which also makes the
    IPW identity E[XY] = E[(1/q)XỸ] work).

    Args:
        X: (n, d) feature matrix.
        Y_tilde: (n,) observed labels (0 for non-respondents).
        q_values: (n,) per-example response probabilities for the current q ∈ Q.
            Only q_values[i] for i in Rₙ is used. Callers may pass arbitrary values
            for non-respondent indices; they are skipped.
        response_mask: (n,) bool array, True where εᵢ = 1.

    Returns:
        (d,) vector.

    Raises:
        ValueError: if any q_values[i] ≤ 0 for i in Rₙ.
    """
    n = X.shape[0]
    mask = np.asarray(response_mask, dtype=bool)
    q_resp = q_values[mask]
    if np.any(q_resp <= 0):
        raise ValueError("q_values must be strictly positive for all respondents")
    X_resp = X[mask]
    Y_resp = Y_tilde[mask]
    weights = Y_resp / q_resp
    return (X_resp.T @ weights) / n


def compute_moments(
    X: np.ndarray,
    Y_tilde: np.ndarray,
    response_mask: np.ndarray,
    q_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience: compute all three moments in one call.

    Returns (b_n, W_n, r_n). See individual functions for details.
    """
    return (
        compute_b_n(X, Y_tilde),
        compute_W_n(X),
        compute_r_n(X, Y_tilde, q_values, response_mask),
    )
