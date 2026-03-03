from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from .comparison import (
    train_erm_baseline,
    train_focal_baseline,
    train_group_balanced_baseline,
    train_group_dro_baseline,
    train_group_prior_baseline,
    train_oracle_baseline,
)
from .config import Q1ObjectiveConfig
from .gradient_validation import (
    GradientValidationConfig,
    LinearDataset,
    _mse,
    _predict,
    train_robust_group,
    train_robust_group_online,
    train_robust_score,
)
from .mnar import MNAR_VIEW_MODES, SyntheticMNARConfig, apply_synthetic_mnar, build_proxy_labels


AG_METHOD_ORDER = (
    "erm",
    "group_balanced",
    "group_prior",
    "focal",
    "group_dro",
    "robust_group",
    "robust_group_online",
    "robust_score",
    "oracle",
)


@dataclass(frozen=True)
class AgricultureBenchmarkConfig:
    benchmark_name: str = "iowa_maize"
    all_benchmarks: bool = False
    seed: int = 31
    trials: int = 3
    train_paths: int = 6
    test_paths: int = 4
    horizon_years: int = 4
    observation_seed: int = 7
    distressed_penalty: float = 0.6
    mnar_mode: str = "explicit_missing"
    base_observation_probability: float = 0.95
    drought_penalty: float = 0.10
    exit_penalty: float = 0.15
    learning_rate: float = 0.05
    epochs: int = 140
    workspace_root: str = "dssat_runs/minimax_ag"
    dssat_root: str | None = None
    initial_cash: float = 300_000.0
    initial_debt: float = 100_000.0
    initial_credit_limit: float = 175_000.0
    acres: float = 500.0
    target: str = "net_income"
    include_score_baseline: bool = True
    include_online_mnar_baseline: bool = True
    assumed_observation_rate: float | None = None
    q1: Q1ObjectiveConfig = field(default_factory=Q1ObjectiveConfig)

    def __post_init__(self) -> None:
        if self.trials <= 0:
            raise ValueError("trials must be positive.")
        if self.train_paths <= 0 or self.test_paths <= 0:
            raise ValueError("train_paths and test_paths must be positive.")
        if self.horizon_years <= 0:
            raise ValueError("horizon_years must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.target not in {"yield", "net_income"}:
            raise ValueError("target must be 'yield' or 'net_income'.")
        if self.mnar_mode not in MNAR_VIEW_MODES:
            raise ValueError(f"mnar_mode must be one of {MNAR_VIEW_MODES}.")
        if self.assumed_observation_rate is not None and not 0.0 < self.assumed_observation_rate <= 1.0:
            raise ValueError("assumed_observation_rate must be in (0, 1].")


@dataclass(frozen=True)
class AgricultureDataset:
    linear: LinearDataset
    test_group_ids: list[str]
    label_scale: float
    label_unit: str
    observation_rate: float
    stable_observation_rate: float
    distressed_observation_rate: float
    train_count: int
    test_count: int


@dataclass(frozen=True)
class AgricultureMethodMetrics:
    test_rmse: float
    distressed_group_rmse: float
    stable_group_rmse: float
    mean_survival_years: float
    median_survival_years: float
    bankruptcy_rate: float
    mean_terminal_wealth: float
    fifth_percentile_terminal_wealth: float
    mean_cumulative_profit: float


@dataclass(frozen=True)
class AgricultureMethodSummary:
    name: str
    mean_test_rmse: float
    mean_distressed_group_rmse: float
    mean_stable_group_rmse: float
    mean_survival_years: float
    mean_median_survival_years: float
    mean_bankruptcy_rate: float
    mean_terminal_wealth: float
    mean_fifth_percentile_terminal_wealth: float
    mean_cumulative_profit: float
    mean_improvement_vs_erm: float
    mean_distressed_improvement_vs_erm: float
    mean_survival_improvement_vs_erm: float
    mean_bankruptcy_reduction_vs_erm: float
    win_rate_vs_erm: float


@dataclass(frozen=True)
class AgricultureReferencePolicySummary:
    name: str
    mean_survival_years: float
    mean_bankruptcy_rate: float
    mean_terminal_wealth: float
    mean_fifth_percentile_terminal_wealth: float
    mean_cumulative_profit: float


@dataclass(frozen=True)
class AgricultureBenchmarkSummary:
    benchmark_name: str
    target: str
    label_unit: str
    trials: int
    train_count: int
    test_count: int
    mean_observation_rate: float
    mean_stable_observation_rate: float
    mean_distressed_observation_rate: float
    methods: dict[str, AgricultureMethodSummary]
    reference_policies: dict[str, AgricultureReferencePolicySummary] = field(default_factory=dict)


@dataclass(frozen=True)
class AgricultureBenchmarkSuiteSummary:
    benchmarks: dict[str, AgricultureBenchmarkSummary]


class _MemoizedCropModel:
    def __init__(self, base_model: Any) -> None:
        self._base_model = base_model
        self._cache: dict[tuple[str, str, str], float] = {}

    def yield_per_acre(self, *, state: Any, action: Any, scenario: Any) -> float:
        key = (action.crop, action.input_level, scenario.weather_regime)
        if key not in self._cache:
            self._cache[key] = self._base_model.yield_per_acre(
                state=state,
                action=action,
                scenario=scenario,
            )
        return self._cache[key]


@dataclass(frozen=True)
class _LinearPredictivePolicy:
    parameters: list[float]
    actions: tuple[Any, ...]

    def choose_action(self, state: Any, scenario: Any) -> Any:
        best_action = self.actions[0]
        best_score = float("-inf")
        for action in self.actions:
            score = _dot_product(
                self.parameters,
                _featurize_decision(state=state, action=action, scenario=scenario),
            )
            if score > best_score:
                best_score = score
                best_action = action
        return best_action


@dataclass(frozen=True)
class _FullObservationRecord:
    observed_net_income: float
    observed_yield_per_acre: float
    observed_price: float
    fully_observed: bool = True


class _FullObservationProcess:
    def apply(
        self,
        records: list[Any],
        *,
        path_index: int,
    ) -> list[_FullObservationRecord]:
        del path_index
        return [
            _FullObservationRecord(
                observed_net_income=record.net_income,
                observed_yield_per_acre=record.realized_yield_per_acre,
                observed_price=record.realized_price,
                fully_observed=True,
            )
            for record in records
        ]


def _require_ag_survival_sim() -> dict[str, Any]:
    try:
        from ag_survival_sim import (  # type: ignore[import-not-found]
            Action,
            FarmState,
            ScenarioGenerator,
            StaticPolicy,
            build_benchmark_crop_model,
            evaluate_policies,
            generate_training_examples,
            get_benchmark_definition,
            list_benchmark_definitions,
        )
        from ag_survival_sim.simulator import FarmSimulator  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError(
            "Agriculture benchmark requires ag-survival-sim. "
            "Install it first, for example: pip install "
            "'ag-survival-sim @ git+https://github.com/adawgwats/ag-survival-sim.git'"
        ) from error

    return {
        "Action": Action,
        "FarmState": FarmState,
        "ScenarioGenerator": ScenarioGenerator,
        "StaticPolicy": StaticPolicy,
        "FarmSimulator": FarmSimulator,
        "build_benchmark_crop_model": build_benchmark_crop_model,
        "evaluate_policies": evaluate_policies,
        "generate_training_examples": generate_training_examples,
        "get_benchmark_definition": get_benchmark_definition,
        "list_benchmark_definitions": list_benchmark_definitions,
    }


def _available_benchmark_names() -> tuple[str, ...]:
    try:
        ag = _require_ag_survival_sim()
    except ImportError:
        return ("iowa_maize",)

    return tuple(definition.name for definition in ag["list_benchmark_definitions"]())


def _build_agriculture_dataset(config: AgricultureBenchmarkConfig, *, trial_index: int) -> AgricultureDataset:
    ag = _require_ag_survival_sim()
    Action = ag["Action"]
    FarmState = ag["FarmState"]
    ScenarioGenerator = ag["ScenarioGenerator"]
    StaticPolicy = ag["StaticPolicy"]
    FarmSimulator = ag["FarmSimulator"]
    build_benchmark_crop_model = ag["build_benchmark_crop_model"]
    generate_training_examples = ag["generate_training_examples"]
    get_benchmark_definition = ag["get_benchmark_definition"]

    benchmark = get_benchmark_definition(config.benchmark_name)
    crop_model = build_benchmark_crop_model(
        config.benchmark_name,
        dssat_root=config.dssat_root,
        workspace_root=str(
            Path(config.workspace_root) / f"{config.benchmark_name}_trial{trial_index}"
        ),
    )
    simulator = FarmSimulator(crop_model=_MemoizedCropModel(crop_model))
    initial_state = FarmState.initial(
        cash=config.initial_cash,
        debt=config.initial_debt,
        credit_limit=config.initial_credit_limit,
        acres=config.acres,
    )
    full_observation_process = _FullObservationProcess()

    policies = tuple(
        StaticPolicy(Action(action.crop, action.input_level))
        for action in benchmark.actions
    )

    train_examples = []
    test_examples = []
    for policy in policies:
        train_examples.extend(
            generate_training_examples(
                simulator=simulator,
                scenario_generator=ScenarioGenerator(seed=config.seed + trial_index),
                policy=policy,
                observation_process=full_observation_process,
                initial_state=initial_state,
                horizon_years=config.horizon_years,
                num_paths=config.train_paths,
            )
        )
        test_examples.extend(
            generate_training_examples(
                simulator=simulator,
                scenario_generator=ScenarioGenerator(seed=config.seed + 10_000 + trial_index),
                policy=policy,
                observation_process=full_observation_process,
                initial_state=initial_state,
                horizon_years=config.horizon_years,
                num_paths=config.test_paths,
            )
        )

    label_scale = 100_000.0 if config.target == "net_income" else 1.0
    label_unit = "USD" if config.target == "net_income" else "bu/ac"
    latent_train_features = [_featurize_example(example) for example in train_examples]
    latent_train_labels = [_extract_label(example, config.target) / label_scale for example in train_examples]
    latent_train_group_ids = [example.group_id for example in train_examples]
    latent_train_path_indices = [int(example.path_index) for example in train_examples]
    latent_train_step_indices = [int(example.step_index) for example in train_examples]
    latent_train_weather_regimes = [str(example.weather_regime) for example in train_examples]
    latent_train_alive_next = [bool(example.farm_alive_next_year) for example in train_examples]

    mnar_result = apply_synthetic_mnar(
        labels=latent_train_labels,
        group_ids=latent_train_group_ids,
        path_indices=latent_train_path_indices,
        step_indices=latent_train_step_indices,
        weather_regimes=latent_train_weather_regimes,
        farm_alive_next_year=latent_train_alive_next,
        config=SyntheticMNARConfig(
            seed=config.observation_seed + trial_index,
            view_mode=config.mnar_mode,
            base_observation_probability=config.base_observation_probability,
            distressed_penalty=config.distressed_penalty,
            drought_penalty=config.drought_penalty,
            exit_penalty=config.exit_penalty,
        ),
    )

    retained_indices = [index for index, keep in enumerate(mnar_result.keep_mask) if keep]
    train_features = [latent_train_features[index] for index in retained_indices]
    train_labels = [latent_train_labels[index] for index in retained_indices]
    train_group_ids = [latent_train_group_ids[index] for index in retained_indices]
    train_observed_mask = [mnar_result.observed_mask[index] for index in retained_indices]
    train_observed_values = [mnar_result.observed_values[index] for index in retained_indices]
    train_proxy_labels = build_proxy_labels(
        observed_values=train_observed_values,
        group_ids=train_group_ids,
        observed_mask=train_observed_mask,
        label_scale=1.0,
    )

    test_features = [_featurize_example(example) for example in test_examples]
    test_labels = [_extract_label(example, config.target) / label_scale for example in test_examples]
    test_group_ids = [example.group_id for example in test_examples]

    linear = LinearDataset(
        train_features=train_features,
        train_labels=train_labels,
        train_proxy_labels=train_proxy_labels,
        train_group_ids=train_group_ids,
        train_observed_mask=train_observed_mask,
        test_features=test_features,
        test_labels=test_labels,
        stable_observation_probability=mnar_result.stable_observation_rate,
        distressed_observation_probability=mnar_result.distressed_observation_rate,
    )
    return AgricultureDataset(
        linear=linear,
        test_group_ids=test_group_ids,
        label_scale=label_scale,
        label_unit=label_unit,
        observation_rate=mnar_result.observation_rate,
        stable_observation_rate=mnar_result.stable_observation_rate,
        distressed_observation_rate=mnar_result.distressed_observation_rate,
        train_count=len(train_features),
        test_count=len(test_features),
    )


def _extract_label(example: Any, target: str) -> float:
    if target == "yield":
        return float(example.latent_yield_per_acre)
    return float(example.latent_net_income)


def _extract_observed_label(example: Any, target: str) -> float | None:
    if target == "yield":
        return None if example.observed_yield_per_acre is None else float(example.observed_yield_per_acre)
    return None if example.observed_net_income is None else float(example.observed_net_income)


def _build_proxy_label(
    *,
    label: float | None,
    group_id: str,
    observed_by_group: dict[str, list[float]],
    global_proxy: float,
    label_scale: float,
) -> float:
    if label is not None:
        return label / label_scale
    group_values = observed_by_group.get(group_id, [])
    if group_values:
        return mean(group_values)
    return global_proxy


def _featurize_fields(
    *,
    cash: float,
    debt: float,
    credit_limit: float,
    acres: float,
    input_level: str,
    weather_regime: str,
) -> list[float]:
    return [
        1.0,
        cash / 300_000.0,
        debt / 200_000.0,
        credit_limit / 200_000.0,
        acres / 500.0,
        1.0 if input_level == "medium" else 0.0,
        1.0 if weather_regime == "good" else 0.0,
        1.0 if weather_regime == "drought" else 0.0,
    ]


def _featurize_example(example: Any) -> list[float]:
    return _featurize_fields(
        cash=example.cash,
        debt=example.debt,
        credit_limit=example.credit_limit,
        acres=example.acres,
        input_level=example.input_level,
        weather_regime=example.weather_regime,
    )


def _featurize_decision(*, state: Any, action: Any, scenario: Any) -> list[float]:
    return _featurize_fields(
        cash=state.cash,
        debt=state.debt,
        credit_limit=state.credit_limit,
        acres=state.acres,
        input_level=action.input_level,
        weather_regime=scenario.weather_regime,
    )


def _dot_product(parameters: list[float], features: list[float]) -> float:
    return sum(parameter * feature for parameter, feature in zip(parameters, features))


def _evaluate_method(
    parameters: list[float],
    dataset: AgricultureDataset,
    *,
    policy_metrics: Any,
) -> AgricultureMethodMetrics:
    predictions = _predict(parameters, dataset.linear.test_features)
    labels = dataset.linear.test_labels
    groups = dataset.test_group_ids
    scale = dataset.label_scale
    overall_rmse = math.sqrt(_mse(predictions, labels)) * scale

    distressed_labels = [label for label, group in zip(labels, groups) if group == "distressed"]
    distressed_predictions = [
        prediction for prediction, group in zip(predictions, groups) if group == "distressed"
    ]
    stable_labels = [label for label, group in zip(labels, groups) if group == "stable"]
    stable_predictions = [
        prediction for prediction, group in zip(predictions, groups) if group == "stable"
    ]

    return AgricultureMethodMetrics(
        test_rmse=overall_rmse,
        distressed_group_rmse=(
            math.sqrt(_mse(distressed_predictions, distressed_labels)) * scale
            if distressed_labels
            else overall_rmse
        ),
        stable_group_rmse=(
            math.sqrt(_mse(stable_predictions, stable_labels)) * scale
            if stable_labels
            else overall_rmse
        ),
        mean_survival_years=policy_metrics.mean_survival_years,
        median_survival_years=policy_metrics.median_survival_years,
        bankruptcy_rate=policy_metrics.bankruptcy_rate,
        mean_terminal_wealth=policy_metrics.mean_terminal_wealth,
        fifth_percentile_terminal_wealth=policy_metrics.fifth_percentile_terminal_wealth,
        mean_cumulative_profit=policy_metrics.mean_cumulative_profit,
    )


def _run_policy_evaluation(
    config: AgricultureBenchmarkConfig,
    *,
    trial_index: int,
    method_parameters: dict[str, list[float]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ag = _require_ag_survival_sim()
    FarmState = ag["FarmState"]
    ScenarioGenerator = ag["ScenarioGenerator"]
    StaticPolicy = ag["StaticPolicy"]
    FarmSimulator = ag["FarmSimulator"]
    build_benchmark_crop_model = ag["build_benchmark_crop_model"]
    evaluate_policies = ag["evaluate_policies"]
    get_benchmark_definition = ag["get_benchmark_definition"]

    benchmark = get_benchmark_definition(config.benchmark_name)
    crop_model = build_benchmark_crop_model(
        config.benchmark_name,
        dssat_root=config.dssat_root,
        workspace_root=str(
            Path(config.workspace_root)
            / f"{config.benchmark_name}_trial{trial_index}_policy_eval"
        ),
    )
    simulator = FarmSimulator(crop_model=_MemoizedCropModel(crop_model))
    initial_state = FarmState.initial(
        cash=config.initial_cash,
        debt=config.initial_debt,
        credit_limit=config.initial_credit_limit,
        acres=config.acres,
    )
    policies = {
        method_name: _LinearPredictivePolicy(parameters=parameters, actions=benchmark.actions)
        for method_name, parameters in method_parameters.items()
    }
    learned_summary = evaluate_policies(
        simulator=simulator,
        scenario_generator=ScenarioGenerator(seed=config.seed + 20_000 + trial_index),
        policies=policies,
        initial_state=initial_state,
        horizon_years=config.horizon_years,
        num_paths=config.test_paths,
    )
    reference_policies = {
        f"static_{action.crop}_{action.input_level}": StaticPolicy(action)
        for action in benchmark.actions
    }
    reference_summary = evaluate_policies(
        simulator=simulator,
        scenario_generator=ScenarioGenerator(seed=config.seed + 20_000 + trial_index),
        policies=reference_policies,
        initial_state=initial_state,
        horizon_years=config.horizon_years,
        num_paths=config.test_paths,
    )
    return learned_summary.metrics, reference_summary.metrics


def run_agriculture_benchmark(
    config: AgricultureBenchmarkConfig,
) -> tuple[list[dict[str, AgricultureMethodMetrics]], AgricultureBenchmarkSummary]:
    per_trial_results: list[dict[str, AgricultureMethodMetrics]] = []
    reference_policy_trials: dict[str, list[Any]] = {}
    observation_rates: list[float] = []
    stable_rates: list[float] = []
    distressed_rates: list[float] = []
    train_counts: list[int] = []
    test_counts: list[int] = []
    label_unit = "scaled"

    for trial_index in range(config.trials):
        dataset = _build_agriculture_dataset(config, trial_index=trial_index)
        label_unit = dataset.label_unit
        train_counts.append(dataset.train_count)
        test_counts.append(dataset.test_count)
        observation_rates.append(dataset.observation_rate)
        stable_rates.append(dataset.stable_observation_rate)
        distressed_rates.append(dataset.distressed_observation_rate)

        baseline_config = _baseline_config_for_ag(config)
        robust_group_config = _robust_config_for_ag(
            config,
            adversary_mode="group",
            dataset_observation_rate=dataset.observation_rate,
        )
        robust_score_config = _robust_config_for_ag(config, adversary_mode="score")

        method_parameters: dict[str, list[float]] = {
            "erm": train_erm_baseline(dataset.linear, baseline_config),
            "group_balanced": train_group_balanced_baseline(dataset.linear, baseline_config),
            "group_prior": train_group_prior_baseline(dataset.linear, baseline_config),
            "focal": train_focal_baseline(dataset.linear, baseline_config),
            "group_dro": train_group_dro_baseline(dataset.linear, baseline_config),
            "robust_group": train_robust_group(dataset.linear, robust_group_config),
            "oracle": train_oracle_baseline(dataset.linear, baseline_config),
        }
        if config.include_online_mnar_baseline:
            method_parameters["robust_group_online"] = train_robust_group_online(
                dataset.linear,
                robust_group_config,
            )
        if config.include_score_baseline:
            method_parameters["robust_score"] = train_robust_score(
                dataset.linear,
                robust_score_config,
            )

        policy_metrics, reference_policy_metrics = _run_policy_evaluation(
            config,
            trial_index=trial_index,
            method_parameters=method_parameters,
        )
        for policy_name, metrics in reference_policy_metrics.items():
            reference_policy_trials.setdefault(policy_name, []).append(metrics)

        per_trial_results.append(
            {
                name: _evaluate_method(
                    parameters,
                    dataset,
                    policy_metrics=policy_metrics[name],
                )
                for name, parameters in method_parameters.items()
            }
        )

    summaries: dict[str, AgricultureMethodSummary] = {}
    reference_summaries: dict[str, AgricultureReferencePolicySummary] = {}
    available_methods = [
        name for name in AG_METHOD_ORDER if any(name in trial for trial in per_trial_results)
    ]
    for method_name in available_methods:
        method_trials = [trial[method_name] for trial in per_trial_results]
        erm_trials = [trial["erm"] for trial in per_trial_results]
        summaries[method_name] = AgricultureMethodSummary(
            name=method_name,
            mean_test_rmse=mean(metric.test_rmse for metric in method_trials),
            mean_distressed_group_rmse=mean(metric.distressed_group_rmse for metric in method_trials),
            mean_stable_group_rmse=mean(metric.stable_group_rmse for metric in method_trials),
            mean_survival_years=mean(metric.mean_survival_years for metric in method_trials),
            mean_median_survival_years=mean(metric.median_survival_years for metric in method_trials),
            mean_bankruptcy_rate=mean(metric.bankruptcy_rate for metric in method_trials),
            mean_terminal_wealth=mean(metric.mean_terminal_wealth for metric in method_trials),
            mean_fifth_percentile_terminal_wealth=mean(
                metric.fifth_percentile_terminal_wealth for metric in method_trials
            ),
            mean_cumulative_profit=mean(metric.mean_cumulative_profit for metric in method_trials),
            mean_improvement_vs_erm=mean(
                erm_metric.test_rmse - metric.test_rmse
                for erm_metric, metric in zip(erm_trials, method_trials)
            ),
            mean_distressed_improvement_vs_erm=mean(
                erm_metric.distressed_group_rmse - metric.distressed_group_rmse
                for erm_metric, metric in zip(erm_trials, method_trials)
            ),
            mean_survival_improvement_vs_erm=mean(
                metric.mean_survival_years - erm_metric.mean_survival_years
                for erm_metric, metric in zip(erm_trials, method_trials)
            ),
            mean_bankruptcy_reduction_vs_erm=mean(
                erm_metric.bankruptcy_rate - metric.bankruptcy_rate
                for erm_metric, metric in zip(erm_trials, method_trials)
            ),
            win_rate_vs_erm=sum(
                1 for erm_metric, metric in zip(erm_trials, method_trials) if metric.test_rmse < erm_metric.test_rmse
            )
            / len(method_trials),
        )
    for policy_name, metrics_list in reference_policy_trials.items():
        reference_summaries[policy_name] = AgricultureReferencePolicySummary(
            name=policy_name,
            mean_survival_years=mean(metric.mean_survival_years for metric in metrics_list),
            mean_bankruptcy_rate=mean(metric.bankruptcy_rate for metric in metrics_list),
            mean_terminal_wealth=mean(metric.mean_terminal_wealth for metric in metrics_list),
            mean_fifth_percentile_terminal_wealth=mean(
                metric.fifth_percentile_terminal_wealth for metric in metrics_list
            ),
            mean_cumulative_profit=mean(metric.mean_cumulative_profit for metric in metrics_list),
        )

    return per_trial_results, AgricultureBenchmarkSummary(
        benchmark_name=config.benchmark_name,
        target=config.target,
        label_unit=label_unit,
        trials=config.trials,
        train_count=round(mean(train_counts)),
        test_count=round(mean(test_counts)),
        mean_observation_rate=mean(observation_rates),
        mean_stable_observation_rate=mean(stable_rates),
        mean_distressed_observation_rate=mean(distressed_rates),
        methods=summaries,
        reference_policies=reference_summaries,
    )


def run_agriculture_benchmark_suite(
    config: AgricultureBenchmarkConfig,
    *,
    benchmark_names: list[str] | tuple[str, ...] | None = None,
) -> AgricultureBenchmarkSuiteSummary:
    selected_names = tuple(benchmark_names or _available_benchmark_names())
    summaries: dict[str, AgricultureBenchmarkSummary] = {}

    for benchmark_name in selected_names:
        benchmark_config = replace(config, benchmark_name=benchmark_name)
        _trial_results, summary = run_agriculture_benchmark(benchmark_config)
        summaries[benchmark_name] = summary

    return AgricultureBenchmarkSuiteSummary(benchmarks=summaries)


def _baseline_config_for_ag(config: AgricultureBenchmarkConfig):
    from .comparison import BaselineComparisonConfig

    return BaselineComparisonConfig(
        seed=config.seed,
        trials=1,
        scenario="aligned_selective",
        learning_rate=config.learning_rate,
        epochs=config.epochs,
    )


def _robust_config_for_ag(
    config: AgricultureBenchmarkConfig,
    *,
    adversary_mode: str,
    dataset_observation_rate: float | None = None,
) -> GradientValidationConfig:
    return GradientValidationConfig(
        seed=config.seed,
        trials=1,
        scenario="aligned_selective",
        adversary_mode=adversary_mode,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        assumed_observation_rate=(
            config.assumed_observation_rate
            if config.assumed_observation_rate is not None
            else dataset_observation_rate
        ),
        q1=config.q1,
    )


def format_agriculture_benchmark_summary(summary: AgricultureBenchmarkSummary) -> str:
    lines = [
        "DSSAT-backed agriculture benchmark summary",
        f"benchmark: {summary.benchmark_name}",
        f"target: {summary.target} ({summary.label_unit})",
        f"trials: {summary.trials}",
        f"mean train examples: {summary.train_count}",
        f"mean test examples: {summary.test_count}",
        f"mean observation rate: {summary.mean_observation_rate:.3f}",
        f"mean stable observation rate: {summary.mean_stable_observation_rate:.3f}",
        f"mean distressed observation rate: {summary.mean_distressed_observation_rate:.3f}",
        "",
        "predictive fit",
        "method         overall_rmse  distressed_rmse  stable_rmse  improve_vs_erm  distressed_improve  win_rate",
        "-------------  ------------  ---------------  -----------  --------------  ------------------  --------",
    ]

    for method_name in AG_METHOD_ORDER:
        method = summary.methods.get(method_name)
        if method is None:
            continue
        lines.append(
            f"{method_name:<13}"
            f"  {method.mean_test_rmse:>12.2f}"
            f"  {method.mean_distressed_group_rmse:>15.2f}"
            f"  {method.mean_stable_group_rmse:>11.2f}"
            f"  {method.mean_improvement_vs_erm:>14.2f}"
                f"  {method.mean_distressed_improvement_vs_erm:>18.2f}"
                f"  {method.win_rate_vs_erm:>8.2f}"
        )
    lines.extend(
        [
            "",
            "downstream policy",
            "method         mean_survival  median_survival  bankruptcy  mean_terminal_wealth  p05_terminal_wealth  mean_cum_profit  surv_vs_erm  bankr_reduct",
            "-------------  -------------  ---------------  ----------  --------------------  -------------------  ---------------  -----------  ------------",
        ]
    )
    for method_name in AG_METHOD_ORDER:
        method = summary.methods.get(method_name)
        if method is None:
            continue
        lines.append(
            f"{method_name:<13}"
            f"  {method.mean_survival_years:>13.2f}"
            f"  {method.mean_median_survival_years:>15.2f}"
            f"  {method.mean_bankruptcy_rate:>10.2%}"
            f"  {method.mean_terminal_wealth:>20.2f}"
            f"  {method.mean_fifth_percentile_terminal_wealth:>19.2f}"
            f"  {method.mean_cumulative_profit:>15.2f}"
            f"  {method.mean_survival_improvement_vs_erm:>11.2f}"
            f"  {method.mean_bankruptcy_reduction_vs_erm:>12.2%}"
        )
    if summary.reference_policies:
        lines.extend(
            [
                "",
                "reference static policies",
                "policy                    mean_survival  bankruptcy  mean_terminal_wealth  p05_terminal_wealth  mean_cum_profit",
                "------------------------  -------------  ----------  --------------------  -------------------  ---------------",
            ]
        )
        for policy_name in sorted(summary.reference_policies):
            policy = summary.reference_policies[policy_name]
            lines.append(
                f"{policy_name:<24}"
                f"  {policy.mean_survival_years:>13.2f}"
                f"  {policy.mean_bankruptcy_rate:>10.2%}"
                f"  {policy.mean_terminal_wealth:>20.2f}"
                f"  {policy.mean_fifth_percentile_terminal_wealth:>19.2f}"
                f"  {policy.mean_cumulative_profit:>15.2f}"
            )
    return "\n".join(lines)


def format_agriculture_benchmark_suite_summary(
    summary: AgricultureBenchmarkSuiteSummary,
) -> str:
    lines = ["DSSAT-backed agriculture benchmark suite"]
    for benchmark_name, benchmark_summary in summary.benchmarks.items():
        lines.append("")
        lines.append(f"[{benchmark_name}]")
        lines.append(format_agriculture_benchmark_summary(benchmark_summary))
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> AgricultureBenchmarkConfig:
    parser = argparse.ArgumentParser(
        description="Run a DSSAT-backed agriculture benchmark against minimax baselines."
    )
    parser.add_argument(
        "--benchmark",
        choices=_available_benchmark_names(),
        default=AgricultureBenchmarkConfig.benchmark_name,
    )
    parser.add_argument("--seed", type=int, default=AgricultureBenchmarkConfig.seed)
    parser.add_argument("--trials", type=int, default=AgricultureBenchmarkConfig.trials)
    parser.add_argument("--train-paths", type=int, default=AgricultureBenchmarkConfig.train_paths)
    parser.add_argument("--test-paths", type=int, default=AgricultureBenchmarkConfig.test_paths)
    parser.add_argument("--horizon-years", type=int, default=AgricultureBenchmarkConfig.horizon_years)
    parser.add_argument("--epochs", type=int, default=AgricultureBenchmarkConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=AgricultureBenchmarkConfig.learning_rate)
    parser.add_argument("--workspace-root", type=str, default=AgricultureBenchmarkConfig.workspace_root)
    parser.add_argument("--dssat-root", type=str, default=None)
    parser.add_argument("--target", choices=["yield", "net_income"], default=AgricultureBenchmarkConfig.target)
    parser.add_argument("--mnar-mode", choices=MNAR_VIEW_MODES, default=AgricultureBenchmarkConfig.mnar_mode)
    parser.add_argument(
        "--base-observation-probability",
        type=float,
        default=AgricultureBenchmarkConfig.base_observation_probability,
    )
    parser.add_argument("--distressed-penalty", type=float, default=AgricultureBenchmarkConfig.distressed_penalty)
    parser.add_argument("--drought-penalty", type=float, default=AgricultureBenchmarkConfig.drought_penalty)
    parser.add_argument("--exit-penalty", type=float, default=AgricultureBenchmarkConfig.exit_penalty)
    parser.add_argument("--q-min", type=float, default=Q1ObjectiveConfig.q_min)
    parser.add_argument("--q-max", type=float, default=Q1ObjectiveConfig.q_max)
    parser.add_argument("--adversary-step-size", type=float, default=Q1ObjectiveConfig.adversary_step_size)
    parser.add_argument("--exclude-score-baseline", action="store_true")
    parser.add_argument("--exclude-online-mnar-baseline", action="store_true")
    parser.add_argument("--assumed-observation-rate", type=float, default=None)
    parser.add_argument("--all-benchmarks", action="store_true")
    args = parser.parse_args(argv)
    return AgricultureBenchmarkConfig(
        benchmark_name=args.benchmark,
        all_benchmarks=args.all_benchmarks,
        seed=args.seed,
        trials=args.trials,
        train_paths=args.train_paths,
        test_paths=args.test_paths,
        horizon_years=args.horizon_years,
        mnar_mode=args.mnar_mode,
        base_observation_probability=args.base_observation_probability,
        distressed_penalty=args.distressed_penalty,
        drought_penalty=args.drought_penalty,
        exit_penalty=args.exit_penalty,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        workspace_root=args.workspace_root,
        dssat_root=args.dssat_root,
        target=args.target,
        include_score_baseline=not args.exclude_score_baseline,
        include_online_mnar_baseline=not args.exclude_online_mnar_baseline,
        assumed_observation_rate=args.assumed_observation_rate,
        q1=Q1ObjectiveConfig(
            q_min=args.q_min,
            q_max=args.q_max,
            adversary_step_size=args.adversary_step_size,
        ),
    )


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    if config.all_benchmarks:
        suite_summary = run_agriculture_benchmark_suite(config)
        print(format_agriculture_benchmark_suite_summary(suite_summary))
        return

    _trials, summary = run_agriculture_benchmark(config)
    print(format_agriculture_benchmark_summary(summary))


if __name__ == "__main__":
    main()
