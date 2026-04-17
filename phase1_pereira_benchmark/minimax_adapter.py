"""fit/predict shim around `minimax_core.gradient_validation.train_robust_score`.

The existing minimax_core training is functional: it takes a `LinearDataset` and a
`GradientValidationConfig`, runs SGD with an online score-based adversary over
`config.epochs` iterations, and returns parameters as a `list[float]`. This adapter
wraps that into a sklearn-like estimator (fit/predict) so the benchmark harness can
treat it uniformly with MICE+OLS, Heckman, etc.

Important context from PROTOCOL.md:

  * The estimator trained here is SGD-with-online-adversary, NOT the closed-form
    β̂ = M·(1/n Σ XᵢỸᵢ) + m from Christensen's 2020 write-up. Any report comparing
    against MICE must state which algorithm is being compared.

  * The model is linear: predictions are computed as X @ parameters (no separate
    intercept — we prepend a constant column in `fit`).

  * For rows with missing labels (unobserved), train_robust_score uses a
    proxy_label. We set proxy_labels = mean of observed y. This mirrors the
    mean-imputation baseline at the unobserved rows, which is a reasonable default
    given no auxiliary signal is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from minimax_core.config import Q1ObjectiveConfig
from minimax_core.gradient_validation import (
    GradientValidationConfig,
    LinearDataset,
    train_robust_score,
    train_erm,
    train_oracle,
)


@dataclass(frozen=True)
class MinimaxConfig:
    q_min: float = 0.25
    q_max: float = 1.0
    adversary_step_size: float = 0.05
    learning_rate: float = 0.05
    epochs: int = 180
    assumed_observation_rate: float | None = None
    # Scale-adaptive learning rate: effective_lr = learning_rate / sqrt(n / lr_reference_n).
    # minimax_core's default tests use ~200 training rows; larger datasets need proportionally
    # smaller steps to avoid SGD divergence with the reweighted gradient.
    lr_reference_n: int = 200
    adaptive_lr: bool = True


class ScoreMinimaxRegressor:
    """Christensen-style score-based minimax regressor, sklearn-like.

    Usage:
        model = ScoreMinimaxRegressor()
        model.fit(X_train, y_train, response_mask=mask)
        y_pred = model.predict(X_test)
    """

    def __init__(self, config: MinimaxConfig | None = None):
        self.config = config or MinimaxConfig()
        self._parameters: np.ndarray | None = None

    def _build_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        response_mask: np.ndarray,
    ) -> LinearDataset:
        n, d = X.shape
        if response_mask.sum() == 0:
            raise ValueError("All observations are hidden; cannot fit.")

        # Prepend constant column for intercept
        X_aug = np.concatenate([np.ones((n, 1)), X], axis=1)

        # Fill missing y with 0 for actual_loss computation; the score-based estimator
        # uses proxy_labels for hidden rows anyway, but the loss function still reads
        # train_labels[i] — keeping it 0 for unobserved is safe because those rows
        # are score-weighted via proxy_labels.
        y_filled = y.copy()
        y_filled[~response_mask] = 0.0

        # Proxy labels = mean of observed y (acts like mean-imputation at unobserved rows)
        observed_mean = float(y[response_mask].mean())
        proxy_labels = np.where(response_mask, y_filled, observed_mean)

        # No test set needed for training — test_features/test_labels unused by train_robust_score
        # but the dataclass is frozen so we populate with placeholders
        test_features = X_aug[:1].tolist()
        test_labels = [0.0]

        return LinearDataset(
            train_features=X_aug.tolist(),
            train_labels=y_filled.tolist(),
            train_proxy_labels=proxy_labels.tolist(),
            train_group_ids=["g0"] * n,  # unused for score mode
            train_observed_mask=response_mask.tolist(),
            train_time_indices=list(range(n)),  # unused for score mode
            train_history_scores=[0.0] * n,  # unused
            train_path_ids=[0] * n,  # unused
            test_features=test_features,
            test_labels=test_labels,
            stable_observation_probability=1.0,
            distressed_observation_probability=1.0,
        )

    def _build_config(self, n_train: int) -> GradientValidationConfig:
        q1 = Q1ObjectiveConfig(
            q_min=self.config.q_min,
            q_max=self.config.q_max,
            adversary_step_size=self.config.adversary_step_size,
        )
        if self.config.adaptive_lr and n_train > self.config.lr_reference_n:
            lr = self.config.learning_rate / (n_train / self.config.lr_reference_n) ** 0.5
        else:
            lr = self.config.learning_rate
        return GradientValidationConfig(
            adversary_mode="score",
            learning_rate=lr,
            epochs=self.config.epochs,
            assumed_observation_rate=self.config.assumed_observation_rate,
            q1=q1,
        )

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        response_mask: np.ndarray,
    ) -> "ScoreMinimaxRegressor":
        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        mask = np.asarray(response_mask, dtype=bool)
        dataset = self._build_dataset(X_arr, y_arr, mask)
        config = self._build_config(n_train=X_arr.shape[0])
        params = train_robust_score(dataset, config)
        self._parameters = np.asarray(params, dtype=float)
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self._parameters is None:
            raise RuntimeError("Model not fit.")
        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        X_aug = np.concatenate([np.ones((X_arr.shape[0], 1)), X_arr], axis=1)
        return X_aug @ self._parameters


class ErmRegressor:
    """SGD-based ERM baseline from minimax_core.train_erm (for apples-to-apples
    comparison with ScoreMinimaxRegressor — same SGD engine, no adversary).
    """

    def __init__(self, config: MinimaxConfig | None = None):
        self.config = config or MinimaxConfig()
        self._parameters: np.ndarray | None = None

    def fit(self, X, y, response_mask):
        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        mask = np.asarray(response_mask, dtype=bool)
        n = X_arr.shape[0]
        X_aug = np.concatenate([np.ones((n, 1)), X_arr], axis=1)
        y_filled = y_arr.copy()
        y_filled[~mask] = 0.0
        dataset = LinearDataset(
            train_features=X_aug.tolist(),
            train_labels=y_filled.tolist(),
            train_proxy_labels=y_filled.tolist(),
            train_group_ids=["g0"] * n,
            train_observed_mask=mask.tolist(),
            train_time_indices=list(range(n)),
            train_history_scores=[0.0] * n,
            train_path_ids=[0] * n,
            test_features=X_aug[:1].tolist(),
            test_labels=[0.0],
            stable_observation_probability=1.0,
            distressed_observation_probability=1.0,
        )
        if self.config.adaptive_lr and n > self.config.lr_reference_n:
            lr = self.config.learning_rate / (n / self.config.lr_reference_n) ** 0.5
        else:
            lr = self.config.learning_rate
        config = GradientValidationConfig(
            adversary_mode="score",
            learning_rate=lr,
            epochs=self.config.epochs,
        )
        params = train_erm(dataset, config)
        self._parameters = np.asarray(params, dtype=float)
        return self

    def predict(self, X):
        if self._parameters is None:
            raise RuntimeError("Model not fit.")
        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        X_aug = np.concatenate([np.ones((X_arr.shape[0], 1)), X_arr], axis=1)
        return X_aug @ self._parameters
