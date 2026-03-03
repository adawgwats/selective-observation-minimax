from __future__ import annotations

from dataclasses import dataclass

import pytest

from minimax_core.ag_benchmark import (
    AgricultureBenchmarkConfig,
    _build_proxy_label,
    _featurize_example,
    run_agriculture_benchmark,
)


@dataclass(frozen=True)
class _Example:
    cash: float
    debt: float
    credit_limit: float
    acres: float
    input_level: str
    weather_regime: str
    farm_alive_next_year: bool
    group_id: str
    observed_yield_per_acre: float | None = None
    observed_net_income: float | None = None
    latent_yield_per_acre: float = 0.0
    latent_net_income: float = 0.0


def test_featurize_example_encodes_finance_action_and_weather() -> None:
    features = _featurize_example(
        _Example(
            cash=300_000.0,
            debt=100_000.0,
            credit_limit=175_000.0,
            acres=500.0,
            input_level="medium",
            weather_regime="drought",
            farm_alive_next_year=False,
            group_id="distressed",
        )
    )

    assert features == [1.0, 1.0, 0.5, 0.875, 1.0, 1.0, 0.0, 1.0, 0.0]


def test_missing_proxy_uses_group_mean_before_global_mean() -> None:
    proxy = _build_proxy_label(
        label=None,
        group_id="distressed",
        observed_by_group={"stable": [0.2], "distressed": [-0.4, -0.2]},
        global_proxy=0.1,
        label_scale=1.0,
    )
    assert proxy == pytest.approx(-0.3)


@pytest.mark.integration
def test_real_agriculture_benchmark_runs_with_installed_dependencies() -> None:
    try:
        import ag_survival_sim  # noqa: F401
    except ImportError:
        pytest.skip("ag-survival-sim is not installed")

    try:
        _trial_metrics, summary = run_agriculture_benchmark(
            AgricultureBenchmarkConfig(
                trials=1,
                train_paths=2,
                test_paths=1,
                horizon_years=2,
                epochs=80,
                workspace_root="dssat_runs/test_minimax_ag",
            )
        )
    except FileNotFoundError:
        pytest.skip("real DSSAT installation not available")

    assert summary.train_count > 0
    assert summary.test_count > 0
    assert summary.methods["robust_group"].mean_test_rmse >= 0.0
