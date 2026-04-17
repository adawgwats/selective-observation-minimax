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
import math
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
from .christensen_adapter import ChristensenRegressor

# Pereira §9.2
MISSING_RATES: tuple[float, ...] = (10.0, 20.0, 40.0, 60.0, 80.0)


METHOD_FACTORIES = {
    # Baselines (impute-then-OLS and selection-bias-specific)
    "oracle": lambda mech=None: BASELINE_REGISTRY["oracle"](),
    "complete_case": lambda mech=None: BASELINE_REGISTRY["complete_case"](),
    "mean_impute": lambda mech=None: BASELINE_REGISTRY["mean_impute"](),
    "mice": lambda mech=None: BASELINE_REGISTRY["mice"](),
    "knn_impute": lambda mech=None: BASELINE_REGISTRY["knn_impute"](),
    "ipw_estimated": lambda mech=None: BASELINE_REGISTRY["ipw_estimated"](),
    "heckman": lambda mech=None: BASELINE_REGISTRY["heckman"](),
    # SGD-based methods from minimax_core (DRO variant)
    "erm_sgd": lambda mech=None: ErmRegressor(),
    "minimax_score": lambda mech=None: ScoreMinimaxRegressor(),
    # Faithful Christensen estimator (christensen_core). Needs mechanism_name to pick
    # the right QClass (see christensen_adapter.ChristensenRegressor).
    "christensen_faithful": lambda mech: ChristensenRegressor(mechanism_name=mech),
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
    "christensen_faithful",
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
        # Pass mechanism name so methods that need it (e.g., christensen_faithful)
        # can pick the right QClass; others accept it via default None.
        model = factory(mechanism)
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
    failures_csv = Path(out_csv).with_name("failed_cells.csv") if out_csv is not None else None

    # Resume: load existing results and skip completed cells
    completed: set[tuple[str, str, float, int]] = set()
    rows: list[CellResult] = []
    failed_rows: list[CellResult] = []
    if out_csv is not None and Path(out_csv).exists():
        existing = pd.read_csv(out_csv)
        if not existing.empty:
            rows = [CellResult(**row) for row in existing.to_dict("records")]
            completed = {
                (r["dataset"], r["mechanism"], float(r["missing_rate_pct"]), int(r["seed"]))
                for r in existing.to_dict("records")
            }
            if verbose:
                print(f"  Resuming: {len(completed)} cells already completed, loaded {len(rows)} rows.")
    if failures_csv is not None and failures_csv.exists():
        existing_failures = pd.read_csv(failures_csv)
        if not existing_failures.empty:
            failed_rows = [CellResult(**row) for row in existing_failures.to_dict("records")]

    def _checkpoint() -> None:
        df_partial = pd.DataFrame([vars(r) for r in rows])
        df_partial.to_csv(out_csv, index=False)
        if failed_rows:
            pd.DataFrame([vars(r) for r in failed_rows]).to_csv(failures_csv, index=False)

    total = len(datasets) * len(mechanisms) * len(rates) * len(seeds)
    count = 0
    t_start = time.perf_counter()
    for ds_name, mech, rate, seed in itertools.product(datasets, mechanisms, rates, seeds):
        count += 1
        if (ds_name, mech, float(rate), int(seed)) in completed:
            continue
        try:
            cell_results = run_cell(ds_name, mech, rate, seed, methods=methods)
        except Exception as exc:
            print(f"  ERROR {ds_name}/{mech}/{rate}%/seed{seed}: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        good = [r for r in cell_results if not math.isnan(r.test_mse)]
        bad = [r for r in cell_results if math.isnan(r.test_mse)]
        rows.extend(good)
        failed_rows.extend(bad)
        if verbose and count % max(1, total // 50) == 0:
            elapsed = time.perf_counter() - t_start
            eta = elapsed * (total / count - 1) if count > 0 else 0
            print(f"  [{count}/{total}] elapsed {elapsed:.0f}s  ETA {eta:.0f}s")
        if out_csv is not None and count % 100 == 0:
            _checkpoint()

    df = pd.DataFrame([vars(r) for r in rows])
    if out_csv is not None:
        _checkpoint()
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
