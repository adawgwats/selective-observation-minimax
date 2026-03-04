from __future__ import annotations

from dataclasses import dataclass

import pytest

from minimax_core.ag_benchmark import (
    AgricultureBenchmarkConfig,
    AgricultureReferencePolicySummary,
    AgricultureBenchmarkSummary,
    AgricultureBenchmarkSuiteSummary,
    _outlast_rate,
    _build_proxy_label,
    _build_policy_targets,
    _featurize_example,
    format_agriculture_benchmark_suite_summary,
    run_agriculture_benchmark,
    run_agriculture_benchmark_suite,
)


@dataclass(frozen=True)
class _Example:
    path_index: int
    step_index: int
    year: int
    crop: str
    cash: float
    debt: float
    credit_limit: float
    acres: float
    land_mortgage_balance: float
    land_mortgage_years_remaining: int
    land_mortgage_grace_years_remaining: int
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
            path_index=0,
            step_index=0,
            year=3,
            crop="corn",
            cash=300_000.0,
            debt=100_000.0,
            credit_limit=175_000.0,
            acres=500.0,
            land_mortgage_balance=2_000_000.0,
            land_mortgage_years_remaining=30,
            land_mortgage_grace_years_remaining=2,
            input_level="medium",
            weather_regime="drought",
            farm_alive_next_year=False,
            group_id="distressed",
        ),
        action_index_by_key={("corn", "low"): 0, ("corn", "medium"): 1},
    )

    assert features == [1.0, 1.0, 0.5, 0.875, 1.0, 1.0, 1.0, 1.0, 0.3, 0.0, 1.0]


def test_missing_proxy_uses_group_mean_before_global_mean() -> None:
    proxy = _build_proxy_label(
        label=None,
        group_id="distressed",
        observed_by_group={"stable": [0.2], "distressed": [-0.4, -0.2]},
        global_proxy=0.1,
        label_scale=1.0,
    )
    assert proxy == pytest.approx(-0.3)


def test_build_policy_targets_supports_survival_years() -> None:
    examples = [
        _Example(
            path_index=0,
            step_index=0,
            year=0,
            crop="corn",
            cash=0.0,
            debt=0.0,
            credit_limit=0.0,
            acres=500.0,
            land_mortgage_balance=0.0,
            land_mortgage_years_remaining=0,
            land_mortgage_grace_years_remaining=0,
            input_level="low",
            weather_regime="normal",
            farm_alive_next_year=True,
            group_id="stable",
        ),
        _Example(
            path_index=0,
            step_index=1,
            year=1,
            crop="corn",
            cash=0.0,
            debt=0.0,
            credit_limit=0.0,
            acres=500.0,
            land_mortgage_balance=0.0,
            land_mortgage_years_remaining=0,
            land_mortgage_grace_years_remaining=0,
            input_level="low",
            weather_regime="normal",
            farm_alive_next_year=False,
            group_id="distressed",
        ),
    ]

    targets = _build_policy_targets(examples, "survival_years")

    assert targets == [2.0, 1.0]


def test_build_policy_targets_supports_cumulative_profit_to_go() -> None:
    examples = [
        _Example(
            path_index=0,
            step_index=0,
            year=0,
            crop="corn",
            cash=0.0,
            debt=0.0,
            credit_limit=0.0,
            acres=500.0,
            land_mortgage_balance=0.0,
            land_mortgage_years_remaining=0,
            land_mortgage_grace_years_remaining=0,
            input_level="low",
            weather_regime="normal",
            farm_alive_next_year=True,
            group_id="stable",
            latent_net_income=100.0,
        ),
        _Example(
            path_index=0,
            step_index=1,
            year=1,
            crop="corn",
            cash=0.0,
            debt=0.0,
            credit_limit=0.0,
            acres=500.0,
            land_mortgage_balance=0.0,
            land_mortgage_years_remaining=0,
            land_mortgage_grace_years_remaining=0,
            input_level="low",
            weather_regime="normal",
            farm_alive_next_year=False,
            group_id="distressed",
            latent_net_income=-40.0,
        ),
    ]

    targets = _build_policy_targets(examples, "cumulative_profit_to_go")

    assert targets == [60.0, -40.0]


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
            initial_cash=300000.0,
            initial_operating_debt=100000.0,
            initial_credit_limit=175000.0,
            initial_acres=500.0,
            initial_land_value_per_acre=4000.0,
            initial_land_mortgage_balance=1000000.0,
            initial_land_mortgage_rate=0.045,
            initial_land_mortgage_years=30,
            initial_land_mortgage_grace_years=2,
            best_reference_policy_name="static_corn_medium",
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
                initial_cash=300000.0,
                initial_operating_debt=100000.0,
                initial_credit_limit=175000.0,
                initial_acres=500.0,
                initial_land_value_per_acre=4000.0,
                initial_land_mortgage_balance=1000000.0,
                initial_land_mortgage_rate=0.045,
                initial_land_mortgage_years=30,
                initial_land_mortgage_grace_years=2,
                best_reference_policy_name="static_soy_medium",
                methods={},
                reference_policies={},
            )
        }
    )

    rendered = format_agriculture_benchmark_suite_summary(summary)

    assert "[georgia_soybean]" in rendered
    assert "benchmark: georgia_soybean" in rendered


def test_format_summary_includes_reference_static_policies() -> None:
    summary = AgricultureBenchmarkSummary(
        benchmark_name="iowa_maize",
        target="net_income",
        label_unit="USD",
        trials=1,
        train_count=10,
        test_count=5,
        mean_observation_rate=0.5,
        mean_stable_observation_rate=0.9,
        mean_distressed_observation_rate=0.2,
        initial_cash=300000.0,
        initial_operating_debt=100000.0,
        initial_credit_limit=175000.0,
        initial_acres=500.0,
        initial_land_value_per_acre=4000.0,
        initial_land_mortgage_balance=1000000.0,
        initial_land_mortgage_rate=0.045,
        initial_land_mortgage_years=30,
        initial_land_mortgage_grace_years=2,
        best_reference_policy_name="static_corn_medium",
        methods={},
        reference_policies={
            "static_corn_medium": AgricultureReferencePolicySummary(
                name="static_corn_medium",
                mean_survival_years=3.0,
                mean_bankruptcy_rate=0.25,
                mean_terminal_wealth=250000.0,
                mean_fifth_percentile_terminal_wealth=100000.0,
                mean_cumulative_profit=20000.0,
            )
        },
    )

    rendered = format_agriculture_benchmark_suite_summary(
        AgricultureBenchmarkSuiteSummary(benchmarks={"iowa_maize": summary})
    )

    assert "reference static policies" in rendered
    assert "static_corn_medium" in rendered
    assert "action profile" in rendered


def test_format_summary_includes_best_static_competitor() -> None:
    summary = AgricultureBenchmarkSummary(
        benchmark_name="georgia_maize_management",
        target="survival_years",
        label_unit="years",
        trials=1,
        train_count=10,
        test_count=5,
        mean_observation_rate=0.5,
        mean_stable_observation_rate=0.9,
        mean_distressed_observation_rate=0.2,
        initial_cash=300000.0,
        initial_operating_debt=100000.0,
        initial_credit_limit=175000.0,
        initial_acres=500.0,
        initial_land_value_per_acre=4000.0,
        initial_land_mortgage_balance=1000000.0,
        initial_land_mortgage_rate=0.045,
        initial_land_mortgage_years=30,
        initial_land_mortgage_grace_years=2,
        best_reference_policy_name="static_corn_irrigated_high",
        methods={},
        reference_policies={},
    )

    rendered = format_agriculture_benchmark_suite_summary(
        AgricultureBenchmarkSuiteSummary(benchmarks={"georgia_maize_management": summary})
    )

    assert "best static competitor: static_corn_irrigated_high" in rendered
    assert "land finance:" in rendered
    assert "grace_years=2" in rendered


def test_outlast_rate_counts_only_strict_survival_wins() -> None:
    @dataclass(frozen=True)
    class _PathResult:
        survival_years: int

    @dataclass(frozen=True)
    class _PolicyEvaluation:
        path_results: list[_PathResult]

    rate = _outlast_rate(
        _PolicyEvaluation(path_results=[_PathResult(3), _PathResult(2), _PathResult(4)]),
        _PolicyEvaluation(path_results=[_PathResult(2), _PathResult(2), _PathResult(5)]),
    )

    assert rate == pytest.approx(1 / 3)


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
    except OSError as error:
        if getattr(error, "errno", None) == 28:
            pytest.skip("insufficient disk space for DSSAT integration test")
        raise
    except Exception as error:
        if "not enough space on the disk" in str(error).lower():
            pytest.skip("insufficient disk space for DSSAT integration test")
        raise

    assert summary.train_count > 0
    assert summary.test_count > 0
    assert summary.benchmark_name == "georgia_soybean"
    assert summary.methods["robust_group"].mean_test_rmse >= 0.0
    assert summary.methods["robust_group_online"].mean_test_rmse >= 0.0
    assert summary.methods["robust_structural_break"].mean_test_rmse >= 0.0
    assert summary.methods["robust_surprise"].mean_test_rmse >= 0.0
    assert summary.methods["robust_group"].mean_survival_years >= 0.0
    assert 0.0 <= summary.methods["robust_group"].mean_bankruptcy_rate <= 1.0
