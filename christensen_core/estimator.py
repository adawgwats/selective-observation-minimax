"""ChristensenEstimator — sklearn-style fit/predict wrapping the faithful pipeline.

Combines `moments`, `inner_solver`, `q_classes`, and `outer_solver` into a
single estimator object. One instance per (dataset, Q-class) combination.

Usage:
    from christensen_core.estimator import ChristensenEstimator
    from christensen_core.q_classes import Parametric2ParamForBinary

    q = Parametric2ParamForBinary(monotone="increasing")
    est = ChristensenEstimator(q_class=q, fit_intercept=True)
    est.fit(X_train, Y_tilde_train, response_mask_train)
    y_pred = est.predict(X_test)

The `X_train, X_test` passed here are the RAW feature matrices (n×d_features);
the estimator prepends the intercept column internally if fit_intercept=True.

Two top-level invariants this class enforces:

1. Y_tilde must be 0 where response_mask is False (Christensen's convention;
   non-respondents contribute 0 to b_n). The estimator defensively overwrites
   Y_tilde[~response_mask] = 0 in fit() to guard against callers who pass in
   NaN or other sentinels.

2. After fit(), both self._beta (the estimated β̂ = M* b_n + m*) AND the
   training b_n are stored. predict(X) uses only β̂ (the standard linear
   prediction); b_n is kept for reproducibility/inspection only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .q_classes import QClass
from .outer_solver import OuterResult


@dataclass
class ChristensenEstimator:
    q_class: QClass
    fit_intercept: bool = True

    # Fit-time artifacts (populated after fit())
    _beta: np.ndarray | None = field(default=None, repr=False)
    _M_star: np.ndarray | None = field(default=None, repr=False)
    _m_star: np.ndarray | None = field(default=None, repr=False)
    _theta_star: np.ndarray | None = field(default=None, repr=False)
    _b_n: np.ndarray | None = field(default=None, repr=False)
    _d: int = field(default=0, repr=False)  # feature dim INCLUDING intercept

    def fit(
        self,
        X: np.ndarray,
        Y_tilde: np.ndarray,
        response_mask: np.ndarray,
    ) -> "ChristensenEstimator":
        """Fit the faithful Christensen estimator.

        Implementation outline:
            1. Defensive: set Y_tilde = np.where(response_mask, Y_tilde, 0.0).
            2. If fit_intercept, prepend ones column to X.
            3. Call outer_solver.solve_outer(self.q_class, X_aug, Y_tilde, response_mask).
            4. Store β̂ = M* @ b_n + m*, along with (M*, m*, θ*, b_n).
        """
        from .moments import compute_b_n
        from .outer_solver import solve_outer

        X = np.asarray(X, dtype=float)
        Y_tilde = np.asarray(Y_tilde, dtype=float).copy()
        response_mask = np.asarray(response_mask, dtype=bool)

        # Defensive: enforce Y_tilde = 0 at non-respondents (Christensen's convention)
        Y_tilde[~response_mask] = 0.0

        # Prepend intercept column if requested
        if self.fit_intercept:
            X_aug = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        else:
            X_aug = X

        self._d = X_aug.shape[1]

        # Delegate to outer_solver
        result = solve_outer(self.q_class, X_aug, Y_tilde, response_mask)

        # Store artifacts
        self._M_star = result.M_star
        self._m_star = result.m_star
        self._theta_star = result.theta_star
        self._b_n = compute_b_n(X_aug, Y_tilde)
        self._beta = result.beta_hat  # M* @ b_n + m*
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return ŷ = X_aug @ β̂. Prepends intercept column if fit_intercept."""
        if self._beta is None:
            raise RuntimeError("Estimator not fit.")
        X = np.asarray(X, dtype=float)
        if self.fit_intercept:
            X_aug = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        else:
            X_aug = X
        return X_aug @ self._beta

    @property
    def beta(self) -> np.ndarray:
        if self._beta is None:
            raise RuntimeError("Estimator not fit.")
        return self._beta
