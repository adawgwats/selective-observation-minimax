"""Benchmark harness: run all (dataset, mechanism, rate, seed, method) cells and collect MSE.

Per PROTOCOL.md:
  - 30 seeds per cell
  - Stratified 50/20/30 split
  - MNAR injection on the training split only (val + test remain complete)
  - Evaluation metric: test-set MSE (float y vs predicted float y)
  - 95% CI: μ ± 1.96·σ/√n_seeds (approximation ok for n≥30)

The harness is designed to be both runnable as a script and importable so that the
vertical slice can exercise a smaller subset (e.g., 1 dataset × 2 mechanisms × 3 seeds)
without copy-pasting loop logic.
"""

from __future__ import annotations

import itertools
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import REGISTRY as DATASET_REGISTRY, load as load_dataset
from .mnar_injection import ALL_MECHANISMS, inject
from .preprocess import stratified_split, onehot_encode
from .minimax_adapter import ScoreMinimaxRegressor, ErmRegressor
from .baselines import REGISTRY as BASELINE_REGISTRY

# Pereira §9.2
MISSING_RATES: tuple[float, ...] = (10.0, 20.0, 40.0, 60.0, 80.0)


METHOD_FACTORIES = {
    # Baselines (impute-then-OLS and selection-bias-specific)
    "oracle": lambda: BASELINE_REGISTRY["oracle"](),
    "complete_case": lambda: BASELINE_REGISTRY["complete_case"](),
    "mean_impute": lambda: BASELINE_REGISTRY["mean_impute"](),
    "mice": lambda: BASELINE_REGISTRY["mice"](),
    "knn_impute": lambda: BASELINE_REGISTRY["knn_impute"](),
    "ipw_estimated": lambda: BASELINE_REGISTRY["ipw_estimated"](),
    "heckman": lambda: BASELINE_REGISTRY["heckman"](),
    # SGD-based methods from minimax_core
    "erm_sgd": lambda: ErmRegressor(),
    "minimax_score": lambda: ScoreMinimaxRegressor(),
}

METHOD_ORDER = (
    "oracle",
    "complete_case",
    "mean_impute",
    "mice",
    "knn_impute",
    "ipw_estimated",
    "heckman",
    "erm_sgd",
    "minimax_score",
)


@dataclass
class CellResult:
    dataset: str
    mechanism: str
    missing_rate_pct: float
    seed: int
    method: str
    test_mse: float
    fit_seconds: float
    response_rate: float
    observed_y_positive_rate: float


def run_cell(
    dataset_name: str,
    mechanism: str,
    missing_rate_pct: float,
    seed: int,
    methods: tuple[str, ...] = METHOD_ORDER,
) -> list[CellResult]:
    """Run one (dataset, mechanism, rate, seed) cell across all methods, return per-method results."""
    ds = load_dataset(dataset_name)
    split = stratified_split(ds, seed=seed)

    # Inject MNAR into train labels only
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inj = inject(split.X_train, split.y_train, mechanism=mechanism, missing_rate_pct=missing_rate_pct, seed=seed)

    # One-hot encode train/val/test
    X_tr_arr, _X_val_arr, X_te_arr = onehot_encode(
        split.X_train, split.X_val, split.X_test, split.categorical_cols,
    )

    y_tr_float = split.y_train.astype(float)
    y_te_float = split.y_test.astype(float)
    mask = inj.response_mask

    obs_y_pos = float(y_tr_float[mask].mean()) if mask.any() else float("nan")
    response_rate = float(mask.mean())

    results: list[CellResult] = []
    for method_name in methods:
        factory = METHOD_FACTORIES[method_name]
        model = factory()
        start = time.perf_counter()
        try:
            if method_name == "oracle":
                # Oracle ignores response_mask — fits on full y_tr_float
                model.fit(X_tr_arr, y_tr_float, response_mask=np.ones_like(mask))
            else:
                model.fit(X_tr_arr, y_tr_float, response_mask=mask)
            y_pred = model.predict(X_te_arr)
            mse = float(np.mean((y_pred - y_te_float) ** 2))
        except Exception as exc:
            mse = float("nan")
            elapsed = time.perf_counter() - start
            results.append(CellResult(
                dataset=dataset_name, mechanism=mechanism,
                missing_rate_pct=missing_rate_pct, seed=seed, method=method_name,
                test_mse=mse, fit_seconds=elapsed,
                response_rate=response_rate, observed_y_positive_rate=obs_y_pos,
            ))
            print(f"  WARN {dataset_name}/{mechanism}/{missing_rate_pct}%/seed{seed}/{method_name}: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        elapsed = time.perf_counter() - start
        results.append(CellResult(
            dataset=dataset_name, mechanism=mechanism,
            missing_rate_pct=missing_rate_pct, seed=seed, method=method_name,
            test_mse=mse, fit_seconds=elapsed,
            response_rate=response_rate, observed_y_positive_rate=obs_y_pos,
        ))
    return results


def run_benchmark(
    datasets: tuple[str, ...] = tuple(DATASET_REGISTRY.keys()),
    mechanisms: tuple[str, ...] = ALL_MECHANISMS,
    rates: tuple[float, ...] = MISSING_RATES,
    seeds: tuple[int, ...] = tuple(range(30)),
    methods: tuple[str, ...] = METHOD_ORDER,
    out_csv: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the full grid. Returns a DataFrame of per-cell-per-method results."""
    rows: list[CellResult] = []
    total = len(datasets) * len(mechanisms) * len(rates) * len(seeds)
    count = 0
    t_start = time.perf_counter()
    for ds_name, mech, rate, seed in itertools.product(datasets, mechanisms, rates, seeds):
        cell_results = run_cell(ds_name, mech, rate, seed, methods=methods)
        rows.extend(cell_results)
        count += 1
        if verbose and count % max(1, total // 50) == 0:
            elapsed = time.perf_counter() - t_start
            eta = elapsed * (total / count - 1) if count > 0 else 0
            print(f"  [{count}/{total}] elapsed {elapsed:.0f}s  ETA {eta:.0f}s")
        if out_csv is not None and count % 100 == 0:
            # Incremental checkpoint
            df_partial = pd.DataFrame([vars(r) for r in rows])
            df_partial.to_csv(out_csv, index=False)

    df = pd.DataFrame([vars(r) for r in rows])
    if out_csv is not None:
        df.to_csv(out_csv, index=False)
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, std, 95% CI of test_mse per (dataset, mechanism, rate, method)."""
    grouped = df.groupby(["dataset", "mechanism", "missing_rate_pct", "method"])["test_mse"]
    n_seeds = grouped.count()
    mean = grouped.mean()
    std = grouped.std(ddof=1)
    se = std / np.sqrt(n_seeds)
    ci_lower = mean - 1.96 * se
    ci_upper = mean + 1.96 * se
    agg = pd.DataFrame({
        "mean_mse": mean, "std_mse": std, "n_seeds": n_seeds,
        "ci_lower": ci_lower, "ci_upper": ci_upper,
    }).reset_index()
    return agg
