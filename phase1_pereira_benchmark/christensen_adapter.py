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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ChristensenRegressor:
    """Adapter: Pereira mechanism name → appropriate Christensen QClass → ChristensenEstimator."""
    mechanism_name: str
    fit_intercept: bool = True

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        response_mask: np.ndarray,
    ) -> "ChristensenRegressor":
        """Dispatch on mechanism_name to obtain Q class, then fit ChristensenEstimator.

        Implementation outline:
            from christensen_core.pereira_q import q_class_for
            from christensen_core.estimator import ChristensenEstimator
            q_cls = q_class_for(self.mechanism_name)
            self._inner = ChristensenEstimator(q_class=q_cls, fit_intercept=self.fit_intercept)
            X_arr = X.to_numpy(float) if isinstance(X, pd.DataFrame) else np.asarray(X, float)
            Y_tilde = np.where(np.asarray(response_mask, bool), np.asarray(y, float), 0.0)
            self._inner.fit(X_arr, Y_tilde, np.asarray(response_mask, bool))
            return self
        """
        raise NotImplementedError(
            "Wire up christensen_core.pereira_q.q_class_for + ChristensenEstimator.fit."
        )

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Delegate to the inner estimator's predict."""
        raise NotImplementedError("Delegate to self._inner.predict.")
