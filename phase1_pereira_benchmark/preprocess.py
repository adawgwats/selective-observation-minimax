"""Preprocessing helpers per Pereira §9.2: [0,1] normalization, stratified 50/20/30 split,
one-hot encoding after missing-data injection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .datasets import LoadedDataset


@dataclass
class Split:
    X_train: pd.DataFrame  # pre-onehot (categoricals still native dtype)
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray  # binary {0, 1}
    y_val: np.ndarray
    y_test: np.ndarray
    categorical_cols: tuple[str, ...]
    continuous_cols: tuple[str, ...]


def minmax_normalize(X: pd.DataFrame, continuous_cols: tuple[str, ...]) -> pd.DataFrame:
    """Normalize continuous columns to [0,1] using train-set min/max.

    Categorical columns are left unchanged. Caller should pass the full dataset pre-split;
    better practice is to fit on train and apply to val/test, but Pereira doesn't specify —
    to match their protocol most literally, min/max is over the full dataset.
    """
    out = X.copy()
    if not continuous_cols:
        return out
    cont_df = out[list(continuous_cols)].astype(float)
    mins = cont_df.min(axis=0)
    maxs = cont_df.max(axis=0)
    ranges = (maxs - mins).replace(0, 1.0)
    out[list(continuous_cols)] = (cont_df - mins) / ranges
    return out


def stratified_split(
    ds: LoadedDataset,
    seed: int,
    train_frac: float = 0.5,
    val_frac: float = 0.2,
    test_frac: float = 0.3,
) -> Split:
    """Stratified train/val/test split per Pereira §9.2."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    X_norm = minmax_normalize(ds.X, ds.continuous_cols)

    # First split: train vs (val+test)
    rest_frac = val_frac + test_frac
    X_train, X_rest, y_train, y_rest = train_test_split(
        X_norm, ds.y,
        test_size=rest_frac,
        stratify=ds.y,
        random_state=seed,
    )
    # Second split: val vs test within rest
    val_size_of_rest = val_frac / rest_frac
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest,
        test_size=1 - val_size_of_rest,
        stratify=y_rest,
        random_state=seed,
    )
    return Split(
        X_train=X_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        categorical_cols=ds.categorical_cols,
        continuous_cols=ds.continuous_cols,
    )


def onehot_encode(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_cols: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-hot encode categorical columns, fit on train, transform all.

    Returns (X_train_arr, X_val_arr, X_test_arr) as float numpy arrays.
    """
    if not categorical_cols:
        return (
            X_train.to_numpy(dtype=float),
            X_val.to_numpy(dtype=float),
            X_test.to_numpy(dtype=float),
        )

    # Union categories across train/val/test to ensure consistent encoding
    # (simpler than sklearn's handle_unknown='ignore' for this small-data case)
    cat_list = list(categorical_cols)
    cont_cols = [c for c in X_train.columns if c not in cat_list]

    combined = pd.concat(
        [X_train[cat_list], X_val[cat_list], X_test[cat_list]],
        axis=0, ignore_index=True,
    )
    combined_encoded = pd.get_dummies(combined.astype(str), drop_first=False, dtype=float)

    n_train, n_val = len(X_train), len(X_val)
    encoded_train = combined_encoded.iloc[:n_train].reset_index(drop=True)
    encoded_val = combined_encoded.iloc[n_train : n_train + n_val].reset_index(drop=True)
    encoded_test = combined_encoded.iloc[n_train + n_val :].reset_index(drop=True)

    X_train_out = pd.concat([X_train[cont_cols].reset_index(drop=True).astype(float), encoded_train], axis=1)
    X_val_out = pd.concat([X_val[cont_cols].reset_index(drop=True).astype(float), encoded_val], axis=1)
    X_test_out = pd.concat([X_test[cont_cols].reset_index(drop=True).astype(float), encoded_test], axis=1)

    return (
        X_train_out.to_numpy(dtype=float),
        X_val_out.to_numpy(dtype=float),
        X_test_out.to_numpy(dtype=float),
    )
