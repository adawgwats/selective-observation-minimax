from __future__ import annotations

from dataclasses import dataclass

import pytest

from minimax_core.ag_benchmark import (
    AgricultureBenchmarkConfig,
    AgricultureBenchmarkSummary,
    AgricultureBenchmarkSuiteSummary,
    _build_proxy_label,
    _featurize_example,
    format_agriculture_benchmark_suite_summary,
    run_agriculture_benchmark,
    run_agriculture_benchmark_suite,
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


def test_suite_runner_invokes_requested_benchmarks(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_run_agriculture_benchmark(config: AgricultureBenchmarkConfig):
        calls.append(config.benchmark_name)
        return [], AgricultureBenchmarkSummary(
            benchmark_name=config.benchmark_name,
            target="net_income",
            label_unit="USD",
            trials=1,
            train_count=10,
            test_count=5,
            mean_observation_rate=0.5,
            mean_stable_observation_rate=0.9,
            mean_distressed_observation_rate=0.2,
            methods={},
        )

    monkeypatch.setattr(
        "minimax_core.ag_benchmark.run_agriculture_benchmark",
        fake_run_agriculture_benchmark,
    )

    summary = run_agriculture_benchmark_suite(
        AgricultureBenchmarkConfig(),
        benchmark_names=("iowa_maize", "georgia_soybean"),
    )

    assert calls == ["iowa_maize", "georgia_soybean"]
    assert tuple(summary.benchmarks) == ("iowa_maize", "georgia_soybean")


def test_format_suite_summary_includes_benchmark_headers() -> None:
    summary = AgricultureBenchmarkSuiteSummary(
        benchmarks={
            "georgia_soybean": AgricultureBenchmarkSummary(
                benchmark_name="georgia_soybean",
                target="net_income",
                label_unit="USD",
                trials=1,
                train_count=10,
                test_count=5,
                mean_observation_rate=0.5,
                mean_stable_observation_rate=0.9,
                mean_distressed_observation_rate=0.2,
                methods={},
            )
        }
    )

    rendered = format_agriculture_benchmark_suite_summary(summary)

    assert "[georgia_soybean]" in rendered
    assert "benchmark: georgia_soybean" in rendered


@pytest.mark.integration
def test_real_agriculture_benchmark_runs_for_nondefault_benchmark(tmp_path) -> None:
    try:
        import ag_survival_sim  # noqa: F401
    except ImportError:
        pytest.skip("ag-survival-sim is not installed")

    try:
        _trial_metrics, summary = run_agriculture_benchmark(
            AgricultureBenchmarkConfig(
                benchmark_name="georgia_soybean",
                trials=1,
                train_paths=2,
                test_paths=1,
                horizon_years=2,
                epochs=80,
                workspace_root=str(tmp_path / "test_minimax_ag"),
            )
        )
    except FileNotFoundError:
        pytest.skip("real DSSAT installation not available")

    assert summary.train_count > 0
    assert summary.test_count > 0
    assert summary.benchmark_name == "georgia_soybean"
    assert summary.methods["robust_group"].mean_test_rmse >= 0.0
