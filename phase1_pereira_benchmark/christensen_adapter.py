"""sklearn-style fit/predict wrapper around christensen_core.ChristensenEstimator.

Mirrors the shape of phase1_pereira_benchmark.minimax_adapter.ScoreMinimaxRegressor
so the benchmark harness can treat both methods uniformly.

Usage pattern in the harness:
    model = ChristensenRegressor(mechanism_name=mech)
    model.fit(X_train_arr, y_train_float, response_mask=mask)
    y_pred = model.predict(X_test_arr)

The mechanism name is needed because the right Q class depends on the
Pereira mechanism (see christensen_core.pereira_q). Passing it via the
constructor keeps the harness signature unchanged while giving this adapter
what it needs.

Uncertainty set Q follows Christensen's reference-based pattern (Christensen &
Connault 2023; Adjaho & Christensen 2022): a neighborhood of user-specified
radius `delta` around the empirical observation rate q_hat computed from the
training response_mask. See christensen_core.reference_based_q.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ChristensenRegressor:
    """Adapter: Pereira mechanism name + data q_hat -> reference-based QClass -> ChristensenEstimator.

    Q-specification policy:
      - If `delta` is None (default), use `adaptive_centered_q_for` which looks up a
        mechanism-calibrated delta from the domain-knowledge table in
        `christensen_core.reference_based_q.MECHANISM_DELTA`. Mechanisms not in the
        table fall back to `DEFAULT_DELTA=0.30`. This is the right choice when the
        benchmark protocol fixes the mechanism (as in Pereira).
      - If `delta` is a float, it overrides the mechanism prior and uses that fixed
        radius for the ball around q_hat. Use this in deployment where mechanism
        metadata is unavailable, or for ablations that probe the delta tradeoff.
    """
    mechanism_name: str
    fit_intercept: bool = True
    delta: float | None = None  # None = adaptive; float = fixed override

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        response_mask: np.ndarray,
    ) -> "ChristensenRegressor":
        """Dispatch on mechanism_name to obtain reference-based Q class, then fit ChristensenEstimator."""
        from christensen_core.reference_based_q import (
            adaptive_centered_q_for,
            centered_q_for,
        )
        from christensen_core.estimator import ChristensenEstimator

        mask = np.asarray(response_mask, dtype=bool)
        if self.delta is None:
            q_cls = adaptive_centered_q_for(self.mechanism_name, mask)
        else:
            q_cls = centered_q_for(self.mechanism_name, mask, delta=self.delta)
        self._inner = ChristensenEstimator(q_class=q_cls, fit_intercept=self.fit_intercept)

        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        Y_tilde = np.where(mask, np.asarray(y, dtype=float), 0.0)
        self._inner.fit(X_arr, Y_tilde, mask)
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Delegate to the inner estimator's predict."""
        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        return self._inner.predict(X_arr)
