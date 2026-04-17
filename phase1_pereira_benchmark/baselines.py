"""Baseline regression-under-MNAR-labels methods.

All baselines expose the same interface as minimax_adapter.ScoreMinimaxRegressor:
    model.fit(X, y, response_mask) -> self
    model.predict(X) -> np.ndarray

Baselines included:
    Oracle           — OLS on full pre-injection data. Upper bound on MSE.
    CompleteCase     — drop rows where y missing, OLS on rest.
    MeanImpute       — impute missing y with mean(observed y), OLS on all.
    MICEImpute       — sklearn IterativeImputer on [X, y], OLS on imputed. Primary baseline.
    KNNImpute        — sklearn KNNImputer(k=5) on [X, y], OLS on imputed.
    IPWEstimated     — estimate response prob q(X) via logistic reg; weighted OLS on observed.
    Heckman          — two-step selection correction via statsmodels.

All use numpy-linalg OLS (X'X)^-1 X'y; no regularization. We prepend a bias column.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _augment(X: np.ndarray) -> np.ndarray:
    """Prepend a constant column for intercept."""
    X = np.asarray(X, dtype=float)
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)


def _ols(X_aug: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Ridge-regularized OLS fallback for rank-deficient matrices (ε=1e-8 * I).

    Returns parameter vector of length X_aug.shape[1].
    """
    if weights is None:
        XtX = X_aug.T @ X_aug
        Xty = X_aug.T @ y
    else:
        W = np.diag(weights)
        XtX = X_aug.T @ W @ X_aug
        Xty = X_aug.T @ W @ y
    # Small ridge for numerical stability
    ridge = 1e-8 * np.eye(XtX.shape[0])
    return np.linalg.solve(XtX + ridge, Xty)


class _BaseEstimator:
    """Shared fit/predict shell."""
    def __init__(self):
        self._parameters: np.ndarray | None = None

    def predict(self, X) -> np.ndarray:
        if self._parameters is None:
            raise RuntimeError("Model not fit.")
        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        X_aug = _augment(X_arr)
        return X_aug @ self._parameters


class OracleRegressor(_BaseEstimator):
    """OLS on the full (un-injected) training data. Upper-bound baseline."""
    def fit(self, X, y, response_mask):
        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        self._parameters = _ols(_augment(X_arr), y_arr)
        return self


class CompleteCaseRegressor(_BaseEstimator):
    """OLS on observed rows only."""
    def fit(self, X, y, response_mask):
        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        mask = np.asarray(response_mask, dtype=bool)
        if mask.sum() == 0:
            raise ValueError("All observations hidden.")
        self._parameters = _ols(_augment(X_arr[mask]), y_arr[mask])
        return self


class MeanImputeRegressor(_BaseEstimator):
    """Impute missing y with mean of observed y, then OLS on all rows."""
    def fit(self, X, y, response_mask):
        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).copy()
        mask = np.asarray(response_mask, dtype=bool)
        y_mean = float(y_arr[mask].mean())
        y_arr[~mask] = y_mean
        self._parameters = _ols(_augment(X_arr), y_arr)
        return self


class MICERegressor(_BaseEstimator):
    """sklearn IterativeImputer on [X | y], then OLS using imputed y for all rows.

    This mirrors Pereira's MICE baseline, which is the method to beat per Table 9.4.
    """
    def __init__(self, max_iter: int = 10, random_state: int = 0):
        super().__init__()
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y, response_mask):
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).copy()
        mask = np.asarray(response_mask, dtype=bool)
        # Build combined matrix with NaN for missing y
        y_with_nan = y_arr.copy()
        y_with_nan[~mask] = np.nan
        combined = np.concatenate([X_arr, y_with_nan.reshape(-1, 1)], axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imputer = IterativeImputer(max_iter=self.max_iter, random_state=self.random_state)
            imputed = imputer.fit_transform(combined)
        y_imputed = imputed[:, -1]
        self._parameters = _ols(_augment(X_arr), y_imputed)
        return self


class KNNImputeRegressor(_BaseEstimator):
    """sklearn KNNImputer(k=5) on [X | y], then OLS using imputed y for all rows.

    Mirrors Pereira's kNN baseline.
    """
    def __init__(self, n_neighbors: int = 5):
        super().__init__()
        self.n_neighbors = n_neighbors

    def fit(self, X, y, response_mask):
        from sklearn.impute import KNNImputer

        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).copy()
        mask = np.asarray(response_mask, dtype=bool)
        y_with_nan = y_arr.copy()
        y_with_nan[~mask] = np.nan
        combined = np.concatenate([X_arr, y_with_nan.reshape(-1, 1)], axis=1)
        imputer = KNNImputer(n_neighbors=self.n_neighbors)
        imputed = imputer.fit_transform(combined)
        y_imputed = imputed[:, -1]
        self._parameters = _ols(_augment(X_arr), y_imputed)
        return self


class IPWEstimatedRegressor(_BaseEstimator):
    """Inverse Probability Weighting with q(X) estimated from a logistic regression.

    Fit logistic regression: response_mask ~ X → predicted response probability q̂(X).
    Then weighted OLS on observed rows with weights 1/q̂(Xᵢ), clipped.
    """
    def __init__(self, q_clip: tuple[float, float] = (0.05, 1.0)):
        super().__init__()
        self.q_clip = q_clip

    def fit(self, X, y, response_mask):
        from sklearn.linear_model import LogisticRegression

        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        mask = np.asarray(response_mask, dtype=bool).astype(int)

        # Estimate q(X) = P(observed | X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logit = LogisticRegression(max_iter=1000, solver="lbfgs")
            logit.fit(X_arr, mask)
        q_hat = logit.predict_proba(X_arr)[:, 1]
        q_hat = np.clip(q_hat, self.q_clip[0], self.q_clip[1])

        observed_bool = mask.astype(bool)
        X_obs = X_arr[observed_bool]
        y_obs = y_arr[observed_bool]
        weights = 1.0 / q_hat[observed_bool]
        # Normalize weights to sum to n_observed to keep scale comparable
        weights = weights * (observed_bool.sum() / weights.sum())
        self._parameters = _ols(_augment(X_obs), y_obs, weights=weights)
        return self


class HeckmanRegressor(_BaseEstimator):
    """Heckman two-step selection correction.

    Step 1: probit(response_mask ~ X) → compute inverse Mills ratio λ(X).
    Step 2: OLS(y ~ X + λ) on observed rows.

    Implementation uses statsmodels Probit. The inverse Mills ratio λ(x'β) = φ(x'β)/Φ(x'β).
    """
    def fit(self, X, y, response_mask):
        import statsmodels.api as sm
        from scipy.stats import norm

        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        mask = np.asarray(response_mask, dtype=bool)

        X_aug = _augment(X_arr)
        # Step 1: probit for selection
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                probit = sm.Probit(mask.astype(int), X_aug)
                probit_res = probit.fit(disp=0, method="bfgs", maxiter=200)
            xb = X_aug @ probit_res.params
        except Exception:
            # Fallback to logistic link if Probit diverges
            from sklearn.linear_model import LogisticRegression
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lr = LogisticRegression(max_iter=1000).fit(X_arr, mask.astype(int))
            xb = X_arr @ lr.coef_[0] + lr.intercept_[0]

        # Inverse Mills ratio for observed rows
        lam = norm.pdf(xb) / np.clip(norm.cdf(xb), 1e-12, 1.0)

        # Step 2: OLS with lambda appended
        X_obs = X_arr[mask]
        lam_obs = lam[mask].reshape(-1, 1)
        X_obs_lam = np.concatenate([X_obs, lam_obs], axis=1)
        y_obs = y_arr[mask]

        params_with_lam = _ols(_augment(X_obs_lam), y_obs)
        # Strip the lambda coefficient; keep [intercept, original X coefs]
        self._parameters = params_with_lam[:-1]
        return self


REGISTRY = {
    "oracle": OracleRegressor,
    "complete_case": CompleteCaseRegressor,
    "mean_impute": MeanImputeRegressor,
    "mice": MICERegressor,
    "knn_impute": KNNImputeRegressor,
    "ipw_estimated": IPWEstimatedRegressor,
    "heckman": HeckmanRegressor,
}
