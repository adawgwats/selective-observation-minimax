"""Closed-form inner solver: given fixed q, compute (M*, m*) minimizing the
inner quadratic objective.

Corresponds directly to PDF §1.5 (pages 6–7).

## The inner problem (PDF Problem 2 with q fixed)

    min_{M, m}  (M·b + m)ᵀ W (M·b + m) - 2(M·b + m)ᵀ r(q)

where b = b_n (d-vector), W = W_n (d×d), r = r_n(q) (d-vector).

## The vec reformulation (PDF page 6)

Define:
    A = [M | m]        # d × (d+1)
    bb = [b; 1]        # (d+1)-vector  (b stacked with scalar 1)
    a = vec(A)         # stack columns of A into a (d·(d+1))-vector

Using vec(AB) = (Bᵀ ⊗ I) · vec(A):
    A · bb = (bbᵀ ⊗ I_d) · a

So the objective becomes:
    aᵀ (bb ⊗ I_d) W (bbᵀ ⊗ I_d) a  -  2 aᵀ (bb ⊗ I_d) r

The FOC with respect to `a`:
    (bb ⊗ I_d) W (bbᵀ ⊗ I_d) a  =  (bb ⊗ I_d) r           # (Eq. FOC)

The matrix on the LHS is singular (rank at most d), so Eq. FOC has infinitely
many solutions. The PDF notes: *"any such solution gives the same minimum
squared prediction error (conditional on q)"* — so pseudoinverse or
least-squares via backslash both yield a valid `a*`.

## Recovering (M, m) and β̂

Given a* = vec(A*), unstack into A* (d × (d+1)) and then split:
    M* = A*[:, :d]     # d × d
    m* = A*[:, d]      # d

The final estimator:
    β̂ = M* · b + m*     # d-vector

Used for prediction on new X: ŷ = X_test · β̂.

## Numerical notes

- `scipy.linalg.pinv` or `numpy.linalg.lstsq` are both acceptable solvers for
  Eq. FOC. Christensen's PDF explicitly suggests pinv/backslash.
- Condition number of `(bb ⊗ I_d) W (bbᵀ ⊗ I_d)` can be poor; tolerance
  parameters may need tuning on some datasets.
- For numerical stability, solve the NORMAL FORM equation rather than forming
  the Kronecker products explicitly where d is large (memory cost is
  d²(d+1)² for the LHS matrix — fine for d ≤ 100, grows fast above).
"""

from __future__ import annotations

import numpy as np


def solve_inner(
    b: np.ndarray,
    W: np.ndarray,
    r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the inner problem for fixed q, returning (M*, m*).

    Args:
        b: (d,) = b_n from moments.py
        W: (d, d) = W_n from moments.py
        r: (d,) = r_n(q) from moments.py, for the current q

    Returns:
        M: (d, d) matrix
        m: (d,) vector

    Notes:
        Implementation should (a) build bb = concatenate([b, [1.0]]), (b) form
        the LHS matrix L = (bb ⊗ I_d) W (bbᵀ ⊗ I_d) and RHS vector y = (bb ⊗ I_d) r,
        (c) solve L a = y via pinv or lstsq, (d) reshape a into A (d, d+1) using
        column-major (Fortran) order to invert vec, (e) split A into M and m.
    """
    b = np.asarray(b, dtype=float).reshape(-1)
    W = np.asarray(W, dtype=float)
    r = np.asarray(r, dtype=float).reshape(-1)
    d = b.shape[0]
    if W.shape != (d, d):
        raise ValueError(f"W must be (d, d) = ({d}, {d}); got {W.shape}")
    if r.shape != (d,):
        raise ValueError(f"r must be (d,) = ({d},); got {r.shape}")

    # Build bb = [b; 1]  (PDF's "b" in §1.5 is our `bb`).
    bb = np.concatenate([b, np.array([1.0])])  # (d+1,)
    I_d = np.eye(d)

    # (bb ⊗ I_d) is shape (d*(d+1), d), (bb' ⊗ I_d) is shape (d, d*(d+1)).
    # Note: bb is a 1-D array, so np.kron(bb, I_d) treats bb as a row.
    # We want the column-Kronecker (bb ⊗ I_d) shaped (d(d+1), d):
    bb_col = bb.reshape(-1, 1)  # (d+1, 1)
    bb_row = bb.reshape(1, -1)  # (1, d+1)
    K_left = np.kron(bb_col, I_d)   # (d*(d+1), d)        == (bb ⊗ I_d)
    K_right = np.kron(bb_row, I_d)  # (d,        d*(d+1)) == (bb' ⊗ I_d)

    # FOC: (bb ⊗ I_d) W (bb' ⊗ I_d) a = (bb ⊗ I_d) r
    L = K_left @ W @ K_right        # (d(d+1), d(d+1)), rank ≤ d
    y = K_left @ r                  # (d(d+1),)

    # Rank-deficient solve: lstsq is the clean choice.
    a, *_ = np.linalg.lstsq(L, y, rcond=None)

    # Invert vec: a (length d*(d+1)) -> A (d, d+1) column-major.
    A = a.reshape((d, d + 1), order="F")
    M = A[:, :d]
    m = A[:, d]
    return M, m


def predict_from_M_m(
    M: np.ndarray,
    m: np.ndarray,
    b: np.ndarray,
    X_new: np.ndarray,
) -> np.ndarray:
    """Compute β̂ = M·b + m, then return predictions X_new · β̂.

    Args:
        M: (d, d)
        m: (d,)
        b: (d,) — the same b_n used to compute (M, m) in the inner solve.
        X_new: (n_new, d) test features (same intercept convention as training).

    Returns:
        (n_new,) predictions.
    """
    M = np.asarray(M, dtype=float)
    m = np.asarray(m, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    X_new = np.asarray(X_new, dtype=float)
    beta_hat = M @ b + m
    return X_new @ beta_hat


def inner_objective_value(
    M: np.ndarray,
    m: np.ndarray,
    b: np.ndarray,
    W: np.ndarray,
    r: np.ndarray,
) -> float:
    """Evaluate the inner objective (M·b+m)ᵀ W (M·b+m) - 2(M·b+m)ᵀ r.

    Useful for (a) sanity-checking that the solver is at the minimum, and
    (b) passing to the outer max routine.

    Returns:
        scalar objective value.
    """
    M = np.asarray(M, dtype=float)
    m = np.asarray(m, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    W = np.asarray(W, dtype=float)
    r = np.asarray(r, dtype=float).reshape(-1)
    beta_hat = M @ b + m
    return float(beta_hat @ W @ beta_hat - 2.0 * beta_hat @ r)
