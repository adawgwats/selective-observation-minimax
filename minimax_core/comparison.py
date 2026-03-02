from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from statistics import mean

from .gradient_validation import (
    GradientValidationConfig,
    LinearDataset,
    _mse,
    _normalize,
    _parameter_error,
    _predict,
    _weighted_gradient,
    generate_linear_dataset,
    train_robust_group,
    train_robust_score,
)


METHOD_ORDER = (
    "erm",
    "group_balanced",
    "group_prior",
    "focal",
    "group_dro",
    "robust_group",
    "robust_score",
    "oracle",
)


@dataclass(frozen=True)
class BaselineComparisonConfig:
    seed: int = 23
    trials: int = 16
    scenario: str = "aligned_selective"
    learning_rate: float = 0.05
    epochs: int = 140
    focal_gamma: float = 2.0
    group_dro_step_size: float = 0.1
    gradient: GradientValidationConfig = field(default_factory=GradientValidationConfig)

    def __post_init__(self) -> None:
        if self.trials <= 0:
            raise ValueError("trials must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.focal_gamma < 0.0:
            raise ValueError("focal_gamma must be non-negative.")
        if self.group_dro_step_size <= 0.0:
            raise ValueError("group_dro_step_size must be positive.")


@dataclass(frozen=True)
class MethodMetrics:
    test_mse: float
    parameter_error: float


@dataclass(frozen=True)
class MethodSummary:
    name: str
    mean_test_mse: float
    mean_parameter_error: float
    mean_improvement_vs_erm: float
    win_rate_vs_erm: float


@dataclass(frozen=True)
class ScenarioComparisonSummary:
    scenario: str
    trials: int
    methods: dict[str, MethodSummary]


def _observed_uniform_weights(dataset: LinearDataset) -> list[float]:
    return _normalize([1.0 if observed else 0.0 for observed in dataset.train_observed_mask])


def _group_balanced_weights(dataset: LinearDataset) -> list[float]:
    observed_counts: dict[str, int] = defaultdict(int)
    for group_id, observed in zip(dataset.train_group_ids, dataset.train_observed_mask):
        if observed:
            observed_counts[group_id] += 1

    groups = sorted(observed_counts)
    if not groups:
        raise ValueError("at least one observed example is required.")

    raw_weights: list[float] = []
    group_mass = 1.0 / len(groups)
    for group_id, observed in zip(dataset.train_group_ids, dataset.train_observed_mask):
        if not observed:
            raw_weights.append(0.0)
            continue
        raw_weights.append(group_mass / observed_counts[group_id])
    return raw_weights


def _group_prior_weights(dataset: LinearDataset) -> list[float]:
    total_counts: dict[str, int] = defaultdict(int)
    observed_counts: dict[str, int] = defaultdict(int)
    for group_id, observed in zip(dataset.train_group_ids, dataset.train_observed_mask):
        total_counts[group_id] += 1
        if observed:
            observed_counts[group_id] += 1

    total_examples = len(dataset.train_group_ids)
    if total_examples <= 0:
        raise ValueError("at least one training example is required.")

    raw_weights: list[float] = []
    for group_id, observed in zip(dataset.train_group_ids, dataset.train_observed_mask):
        if not observed:
            raw_weights.append(0.0)
            continue
        group_prior = total_counts[group_id] / total_examples
        raw_weights.append(group_prior / observed_counts[group_id])
    return raw_weights


def _fit_with_static_weights(
    dataset: LinearDataset,
    *,
    weights: list[float],
    learning_rate: float,
    epochs: int,
) -> list[float]:
    parameters = [0.0, 0.0, 0.0]
    normalized_weights = _normalize(weights)
    for _ in range(epochs):
        gradients = _weighted_gradient(
            parameters,
            dataset.train_features,
            dataset.train_labels,
            normalized_weights,
        )
        parameters = [
            parameter - learning_rate * gradient
            for parameter, gradient in zip(parameters, gradients)
        ]
    return parameters


def train_erm_baseline(dataset: LinearDataset, config: BaselineComparisonConfig) -> list[float]:
    return _fit_with_static_weights(
        dataset,
        weights=_observed_uniform_weights(dataset),
        learning_rate=config.learning_rate,
        epochs=config.epochs,
    )


def train_group_balanced_baseline(
    dataset: LinearDataset,
    config: BaselineComparisonConfig,
) -> list[float]:
    return _fit_with_static_weights(
        dataset,
        weights=_group_balanced_weights(dataset),
        learning_rate=config.learning_rate,
        epochs=config.epochs,
    )


def train_group_prior_baseline(
    dataset: LinearDataset,
    config: BaselineComparisonConfig,
) -> list[float]:
    return _fit_with_static_weights(
        dataset,
        weights=_group_prior_weights(dataset),
        learning_rate=config.learning_rate,
        epochs=config.epochs,
    )


def train_focal_baseline(dataset: LinearDataset, config: BaselineComparisonConfig) -> list[float]:
    parameters = [0.0, 0.0, 0.0]
    base_mask = [1.0 if observed else 0.0 for observed in dataset.train_observed_mask]

    for _ in range(config.epochs):
        predictions = _predict(parameters, dataset.train_features)
        losses = [
            (prediction - label) ** 2
            for prediction, label in zip(predictions, dataset.train_labels)
        ]
        raw_weights = [
            mask * (loss + 1e-8) ** config.focal_gamma
            for mask, loss in zip(base_mask, losses)
        ]
        weights = _normalize(raw_weights)
        gradients = _weighted_gradient(
            parameters,
            dataset.train_features,
            dataset.train_labels,
            weights,
        )
        parameters = [
            parameter - config.learning_rate * gradient
            for parameter, gradient in zip(parameters, gradients)
        ]
    return parameters


def train_group_dro_baseline(
    dataset: LinearDataset,
    config: BaselineComparisonConfig,
) -> list[float]:
    parameters = [0.0, 0.0, 0.0]
    observed_groups = sorted(
        {
            group_id
            for group_id, observed in zip(dataset.train_group_ids, dataset.train_observed_mask)
            if observed
        }
    )
    if not observed_groups:
        raise ValueError("at least one observed group is required.")

    group_weights = {group_id: 1.0 / len(observed_groups) for group_id in observed_groups}

    for _ in range(config.epochs):
        predictions = _predict(parameters, dataset.train_features)
        losses = [
            (prediction - label) ** 2
            for prediction, label in zip(predictions, dataset.train_labels)
        ]

        group_loss_sums: dict[str, float] = defaultdict(float)
        group_counts: dict[str, int] = defaultdict(int)
        for loss, group_id, observed in zip(
            losses,
            dataset.train_group_ids,
            dataset.train_observed_mask,
        ):
            if not observed:
                continue
            group_loss_sums[group_id] += loss
            group_counts[group_id] += 1

        for group_id in observed_groups:
            if group_counts[group_id] <= 0:
                continue
            average_loss = group_loss_sums[group_id] / group_counts[group_id]
            group_weights[group_id] *= math.exp(config.group_dro_step_size * average_loss)

        weight_total = sum(group_weights.values())
        group_weights = {
            group_id: weight / weight_total
            for group_id, weight in group_weights.items()
        }

        example_weights: list[float] = []
        for group_id, observed in zip(dataset.train_group_ids, dataset.train_observed_mask):
            if not observed:
                example_weights.append(0.0)
                continue
            example_weights.append(group_weights[group_id] / group_counts[group_id])

        gradients = _weighted_gradient(
            parameters,
            dataset.train_features,
            dataset.train_labels,
            example_weights,
        )
        parameters = [
            parameter - config.learning_rate * gradient
            for parameter, gradient in zip(parameters, gradients)
        ]

    return parameters


def train_oracle_baseline(dataset: LinearDataset, config: BaselineComparisonConfig) -> list[float]:
    return _fit_with_static_weights(
        dataset,
        weights=[1.0 for _ in dataset.train_labels],
        learning_rate=config.learning_rate,
        epochs=config.epochs,
    )


def _evaluate_parameters(
    parameters: list[float],
    dataset: LinearDataset,
    true_parameters: list[float],
) -> MethodMetrics:
    predictions = _predict(parameters, dataset.test_features)
    return MethodMetrics(
        test_mse=_mse(predictions, dataset.test_labels),
        parameter_error=_parameter_error(parameters, true_parameters),
    )


def _build_gradient_config(config: BaselineComparisonConfig) -> GradientValidationConfig:
    return replace(
        config.gradient,
        seed=config.seed,
        trials=config.trials,
        scenario=config.scenario,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
    )


def run_baseline_comparison(
    config: BaselineComparisonConfig,
) -> tuple[list[dict[str, MethodMetrics]], ScenarioComparisonSummary]:
    gradient_config = _build_gradient_config(config)
    import random

    rng = random.Random(config.seed)
    per_trial_results: list[dict[str, MethodMetrics]] = []

    for _ in range(config.trials):
        dataset, true_parameters = generate_linear_dataset(rng, gradient_config)

        method_parameters = {
            "erm": train_erm_baseline(dataset, config),
            "group_balanced": train_group_balanced_baseline(dataset, config),
            "group_prior": train_group_prior_baseline(dataset, config),
            "focal": train_focal_baseline(dataset, config),
            "group_dro": train_group_dro_baseline(dataset, config),
            "robust_group": train_robust_group(dataset, gradient_config),
            "robust_score": train_robust_score(dataset, gradient_config),
            "oracle": train_oracle_baseline(dataset, config),
        }

        method_metrics = {
            method_name: _evaluate_parameters(parameters, dataset, true_parameters)
            for method_name, parameters in method_parameters.items()
        }
        per_trial_results.append(method_metrics)

    summaries: dict[str, MethodSummary] = {}
    erm_mses = [trial["erm"].test_mse for trial in per_trial_results]

    for method_name in METHOD_ORDER:
        method_mses = [trial[method_name].test_mse for trial in per_trial_results]
        method_parameter_errors = [
            trial[method_name].parameter_error for trial in per_trial_results
        ]
        win_rate_vs_erm = mean(
            1.0 if trial[method_name].test_mse < trial["erm"].test_mse else 0.0
            for trial in per_trial_results
        )
        summaries[method_name] = MethodSummary(
            name=method_name,
            mean_test_mse=mean(method_mses),
            mean_parameter_error=mean(method_parameter_errors),
            mean_improvement_vs_erm=mean(
                erm_mse - method_mse for erm_mse, method_mse in zip(erm_mses, method_mses)
            ),
            win_rate_vs_erm=win_rate_vs_erm,
        )

    return per_trial_results, ScenarioComparisonSummary(
        scenario=config.scenario,
        trials=config.trials,
        methods=summaries,
    )


def run_baseline_comparison_suite(
    config: BaselineComparisonConfig,
    scenarios: tuple[str, ...] = ("aligned_selective", "group_agnostic", "label_dependent"),
) -> dict[str, ScenarioComparisonSummary]:
    return {
        scenario: run_baseline_comparison(replace(config, scenario=scenario))[1]
        for scenario in scenarios
    }


def _format_summary(summary: ScenarioComparisonSummary) -> str:
    lines = [
        "Baseline comparison summary",
        f"scenario: {summary.scenario}",
        f"trials: {summary.trials}",
    ]
    for method_name in METHOD_ORDER:
        method = summary.methods[method_name]
        lines.extend(
            [
                f"[{method_name}] mean test MSE: {method.mean_test_mse:.6f}",
                f"[{method_name}] mean parameter error: {method.mean_parameter_error:.6f}",
                f"[{method_name}] mean improvement vs ERM: {method.mean_improvement_vs_erm:.6f}",
                f"[{method_name}] win rate vs ERM: {method.win_rate_vs_erm:.3f}",
            ]
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> BaselineComparisonConfig:
    parser = argparse.ArgumentParser(
        description="Compare minimax estimators against standard ML baselines."
    )
    parser.add_argument(
        "--scenario",
        choices=["aligned_selective", "group_agnostic", "label_dependent", "suite"],
        default=BaselineComparisonConfig.scenario,
    )
    parser.add_argument("--seed", type=int, default=BaselineComparisonConfig.seed)
    parser.add_argument("--trials", type=int, default=BaselineComparisonConfig.trials)
    parser.add_argument("--learning-rate", type=float, default=BaselineComparisonConfig.learning_rate)
    parser.add_argument("--epochs", type=int, default=BaselineComparisonConfig.epochs)
    parser.add_argument("--focal-gamma", type=float, default=BaselineComparisonConfig.focal_gamma)
    parser.add_argument(
        "--group-dro-step-size",
        type=float,
        default=BaselineComparisonConfig.group_dro_step_size,
    )
    args = parser.parse_args(argv)
    return BaselineComparisonConfig(
        seed=args.seed,
        trials=args.trials,
        scenario="aligned_selective" if args.scenario == "suite" else args.scenario,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        focal_gamma=args.focal_gamma,
        group_dro_step_size=args.group_dro_step_size,
    )


def main(argv: list[str] | None = None) -> None:
    raw_args = argv
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scenario", default=BaselineComparisonConfig.scenario)
    parsed, _unknown = parser.parse_known_args(raw_args)
    config = parse_args(raw_args)
    if parsed.scenario == "suite":
        summaries = run_baseline_comparison_suite(config)
        print("\n\n".join(_format_summary(summary) for summary in summaries.values()))
        return
    _trial_results, summary = run_baseline_comparison(config)
    print(_format_summary(summary))


if __name__ == "__main__":
    main()
