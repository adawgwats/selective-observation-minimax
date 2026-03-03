from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field, replace
from statistics import mean

from .adversary import ScoreBasedObservationAdversary, SelectiveObservationAdversary
from .config import Q1ObjectiveConfig
from .monte_carlo import VALIDATION_SCENARIOS
from .objectives import (
    compute_example_weights,
    compute_score_based_weights,
    estimate_group_snapshot,
)


@dataclass(frozen=True)
class GradientValidationConfig:
    seed: int = 17
    trials: int = 24
    scenario: str = "aligned_selective"
    adversary_mode: str = "group"
    train_count_range: tuple[int, int] = (160, 240)
    test_count_range: tuple[int, int] = (180, 260)
    intercept_range: tuple[float, float] = (0.10, 0.35)
    slope_range: tuple[float, float] = (0.30, 0.70)
    group_effect_range: tuple[float, float] = (0.45, 0.90)
    noise_std_range: tuple[float, float] = (0.03, 0.12)
    proxy_noise_std_range: tuple[float, float] = (0.08, 0.20)
    stable_observation_range: tuple[float, float] = (0.92, 1.0)
    distressed_observation_range: tuple[float, float] = (0.15, 0.45)
    agnostic_observation_range: tuple[float, float] = (0.45, 0.75)
    label_penalty_range: tuple[float, float] = (0.10, 0.30)
    learning_rate: float = 0.05
    epochs: int = 180
    min_observed_per_group: int = 1
    online_mnar: bool = False
    assumed_observation_rate: float | None = None
    q1: Q1ObjectiveConfig = field(default_factory=Q1ObjectiveConfig)

    def __post_init__(self) -> None:
        if self.trials <= 0:
            raise ValueError("trials must be positive.")
        if self.scenario not in VALIDATION_SCENARIOS:
            raise ValueError(f"scenario must be one of {VALIDATION_SCENARIOS}.")
        if self.adversary_mode not in {"group", "score"}:
            raise ValueError("adversary_mode must be 'group' or 'score'.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.min_observed_per_group <= 0:
            raise ValueError("min_observed_per_group must be positive.")
        if self.assumed_observation_rate is not None and not 0.0 < self.assumed_observation_rate <= 1.0:
            raise ValueError("assumed_observation_rate must be in (0, 1].")
        for name, bounds in (
            ("train_count_range", self.train_count_range),
            ("test_count_range", self.test_count_range),
            ("intercept_range", self.intercept_range),
            ("slope_range", self.slope_range),
            ("group_effect_range", self.group_effect_range),
            ("noise_std_range", self.noise_std_range),
            ("proxy_noise_std_range", self.proxy_noise_std_range),
            ("stable_observation_range", self.stable_observation_range),
            ("distressed_observation_range", self.distressed_observation_range),
            ("agnostic_observation_range", self.agnostic_observation_range),
            ("label_penalty_range", self.label_penalty_range),
        ):
            if bounds[0] > bounds[1]:
                raise ValueError(f"{name} lower bound cannot exceed upper bound.")


@dataclass(frozen=True)
class LinearDataset:
    train_features: list[list[float]]
    train_labels: list[float]
    train_proxy_labels: list[float]
    train_group_ids: list[str]
    train_observed_mask: list[bool]
    test_features: list[list[float]]
    test_labels: list[float]
    stable_observation_probability: float
    distressed_observation_probability: float


@dataclass(frozen=True)
class GradientTrialResult:
    trial_index: int
    erm_test_mse: float
    robust_test_mse: float
    oracle_test_mse: float
    erm_parameter_error: float
    robust_parameter_error: float
    stable_observation_probability: float
    distressed_observation_probability: float
    observation_rate: float


@dataclass(frozen=True)
class GradientValidationSummary:
    scenario: str
    adversary_mode: str
    online_mnar: bool
    trials: int
    robust_beats_erm_rate: float
    mean_erm_test_mse: float
    mean_robust_test_mse: float
    mean_oracle_test_mse: float
    mean_test_mse_improvement: float
    mean_erm_parameter_error: float
    mean_robust_parameter_error: float
    mean_parameter_error_improvement: float
    mean_observation_rate: float
    mean_stable_observation_probability: float
    mean_distressed_observation_probability: float


def _uniform_int(rng: random.Random, bounds: tuple[int, int]) -> int:
    return rng.randint(bounds[0], bounds[1])


def _uniform_float(rng: random.Random, bounds: tuple[float, float]) -> float:
    if bounds[0] == bounds[1]:
        return bounds[0]
    return rng.uniform(bounds[0], bounds[1])


def _aligned_group_probability(
    rng: random.Random,
    config: GradientValidationConfig,
) -> tuple[float, float]:
    return (
        _uniform_float(rng, config.stable_observation_range),
        _uniform_float(rng, config.distressed_observation_range),
    )


def _agnostic_group_probability(
    rng: random.Random,
    config: GradientValidationConfig,
) -> tuple[float, float]:
    shared = _uniform_float(rng, config.agnostic_observation_range)
    return shared, shared


def _label_dependent_probability(
    label: float,
    base_probability: float,
    intercept: float,
    group_effect: float,
    penalty: float,
) -> float:
    normalized = max((label - intercept) / max(group_effect, 1e-9), 0.0)
    return min(max(base_probability - penalty * normalized, 0.05), 1.0)


def _dot(parameters: list[float], features: list[float]) -> float:
    return sum(weight * feature for weight, feature in zip(parameters, features))


def _predict(parameters: list[float], features: list[list[float]]) -> list[float]:
    return [_dot(parameters, feature_row) for feature_row in features]


def _mse(predictions: list[float], labels: list[float]) -> float:
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length.")
    return mean((prediction - label) ** 2 for prediction, label in zip(predictions, labels))


def _parameter_error(parameters: list[float], true_parameters: list[float]) -> float:
    return sum((estimate - truth) ** 2 for estimate, truth in zip(parameters, true_parameters)) ** 0.5


def _weighted_gradient(
    parameters: list[float],
    features: list[list[float]],
    labels: list[float],
    weights: list[float],
) -> list[float]:
    gradients = [0.0 for _ in parameters]
    for feature_row, label, weight in zip(features, labels, weights):
        if weight == 0.0:
            continue
        error = _dot(parameters, feature_row) - label
        scale = 2.0 * weight * error
        for index, feature in enumerate(feature_row):
            gradients[index] += scale * feature
    return gradients


def _normalize(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value.")
    return [weight / total for weight in weights]


def _clip_observation_rate(rate: float, config: GradientValidationConfig) -> float:
    return min(max(rate, config.q1.q_min), config.q1.q_max)


def _generate_split(
    rng: random.Random,
    count: int,
    intercept: float,
    slope: float,
    group_effect: float,
    noise_std: float,
    proxy_noise_std: float,
    stable_obs_prob: float,
    distressed_obs_prob: float,
    scenario: str,
    label_penalty: float,
    observed: bool,
) -> tuple[
    list[list[float]],
    list[float],
    list[float],
    list[str],
    list[bool],
    list[float],
    list[float],
]:
    features: list[list[float]] = []
    labels: list[float] = []
    proxy_labels: list[float] = []
    group_ids: list[str] = []
    observed_mask: list[bool] = []
    stable_probabilities: list[float] = []
    distressed_probabilities: list[float] = []

    for _ in range(count):
        is_distressed = rng.random() < 0.5
        x_value = rng.gauss(0.0, 1.0)
        group_indicator = 1.0 if is_distressed else 0.0
        label = intercept + slope * x_value + group_effect * group_indicator + rng.gauss(0.0, noise_std)
        features.append([1.0, x_value, group_indicator])
        labels.append(label)
        proxy_labels.append(label + rng.gauss(0.0, proxy_noise_std))
        group_ids.append("distressed" if is_distressed else "stable")

        if not observed:
            observed_mask.append(True)
            if is_distressed:
                distressed_probabilities.append(1.0)
            else:
                stable_probabilities.append(1.0)
            continue

        if scenario == "label_dependent":
            base_probability = distressed_obs_prob if is_distressed else stable_obs_prob
            observation_probability = _label_dependent_probability(
                label,
                base_probability,
                intercept,
                group_effect,
                label_penalty,
            )
        else:
            observation_probability = distressed_obs_prob if is_distressed else stable_obs_prob
        observed_mask.append(rng.random() < observation_probability)
        if is_distressed:
            distressed_probabilities.append(observation_probability)
        else:
            stable_probabilities.append(observation_probability)

    return (
        features,
        labels,
        proxy_labels,
        group_ids,
        observed_mask,
        stable_probabilities,
        distressed_probabilities,
    )


def generate_linear_dataset(
    rng: random.Random,
    config: GradientValidationConfig,
) -> tuple[LinearDataset, list[float]]:
    train_count = _uniform_int(rng, config.train_count_range)
    test_count = _uniform_int(rng, config.test_count_range)

    intercept = _uniform_float(rng, config.intercept_range)
    slope = _uniform_float(rng, config.slope_range)
    group_effect = _uniform_float(rng, config.group_effect_range)
    noise_std = _uniform_float(rng, config.noise_std_range)
    proxy_noise_std = _uniform_float(rng, config.proxy_noise_std_range)

    if config.scenario == "aligned_selective":
        stable_obs_prob, distressed_obs_prob = _aligned_group_probability(rng, config)
        label_penalty = 0.0
    elif config.scenario == "group_agnostic":
        stable_obs_prob, distressed_obs_prob = _agnostic_group_probability(rng, config)
        label_penalty = 0.0
    else:
        stable_obs_prob, distressed_obs_prob = _aligned_group_probability(rng, config)
        label_penalty = _uniform_float(rng, config.label_penalty_range)

    (
        train_features,
        train_labels,
        train_proxy_labels,
        train_group_ids,
        train_observed_mask,
        stable_probabilities,
        distressed_probabilities,
    ) = _generate_split(
        rng,
        train_count,
        intercept,
        slope,
        group_effect,
        noise_std,
        proxy_noise_std,
        stable_obs_prob,
        distressed_obs_prob,
        config.scenario,
        label_penalty,
        observed=True,
    )

    (
        test_features,
        test_labels,
        _test_proxy_labels,
        _test_group_ids,
        _test_observed_mask,
        _stable_test_probs,
        _distressed_test_probs,
    ) = _generate_split(
        rng,
        test_count,
        intercept,
        slope,
        group_effect,
        noise_std,
        proxy_noise_std,
        stable_obs_prob,
        distressed_obs_prob,
        config.scenario,
        label_penalty,
        observed=False,
    )

    if sum(
        1 for group, observed in zip(train_group_ids, train_observed_mask) if group == "stable" and observed
    ) < config.min_observed_per_group:
        for index, group in enumerate(train_group_ids):
            if group == "stable":
                train_observed_mask[index] = True
                break
    if sum(
        1 for group, observed in zip(train_group_ids, train_observed_mask) if group == "distressed" and observed
    ) < config.min_observed_per_group:
        for index, group in enumerate(train_group_ids):
            if group == "distressed":
                train_observed_mask[index] = True
                break

    dataset = LinearDataset(
        train_features=train_features,
        train_labels=train_labels,
        train_proxy_labels=train_proxy_labels,
        train_group_ids=train_group_ids,
        train_observed_mask=train_observed_mask,
        test_features=test_features,
        test_labels=test_labels,
        stable_observation_probability=mean(stable_probabilities),
        distressed_observation_probability=mean(distressed_probabilities),
    )
    true_parameters = [intercept, slope, group_effect]
    return dataset, true_parameters


def train_erm(dataset: LinearDataset, config: GradientValidationConfig) -> list[float]:
    parameters = [0.0 for _ in dataset.train_features[0]]
    weights = _normalize([1.0 if observed else 0.0 for observed in dataset.train_observed_mask])
    for _ in range(config.epochs):
        gradients = _weighted_gradient(parameters, dataset.train_features, dataset.train_labels, weights)
        parameters = [
            parameter - config.learning_rate * gradient
            for parameter, gradient in zip(parameters, gradients)
        ]
    return parameters


def train_robust(dataset: LinearDataset, config: GradientValidationConfig) -> list[float]:
    if config.adversary_mode == "group":
        return train_robust_group(dataset, config)
    return train_robust_score(dataset, config)


def train_robust_group(dataset: LinearDataset, config: GradientValidationConfig) -> list[float]:
    parameters = [0.0 for _ in dataset.train_features[0]]
    adversary = SelectiveObservationAdversary(config.q1)
    empirical_observation_rate = (
        sum(1 for observed in dataset.train_observed_mask if observed) / len(dataset.train_observed_mask)
    )
    assumed_observation_rate = _clip_observation_rate(
        config.assumed_observation_rate or empirical_observation_rate,
        config,
    )

    for _ in range(config.epochs):
        predictions = _predict(parameters, dataset.train_features)
        losses = [(prediction - label) ** 2 for prediction, label in zip(predictions, dataset.train_labels)]
        snapshot = estimate_group_snapshot(
            losses=losses,
            group_ids=dataset.train_group_ids,
            observed_mask=dataset.train_observed_mask,
        )
        if config.online_mnar:
            snapshot = replace(snapshot, observation_rate=assumed_observation_rate)
        q_values = adversary.update(snapshot)
        weights = compute_example_weights(
            snapshot,
            dataset.train_group_ids,
            dataset.train_observed_mask,
            q_values,
        )
        gradients = _weighted_gradient(parameters, dataset.train_features, dataset.train_labels, weights)
        parameters = [
            parameter - config.learning_rate * gradient
            for parameter, gradient in zip(parameters, gradients)
        ]
    return parameters


def train_robust_score(dataset: LinearDataset, config: GradientValidationConfig) -> list[float]:
    parameters = [0.0 for _ in dataset.train_features[0]]
    adversary = ScoreBasedObservationAdversary(config.q1)
    observation_rate = sum(1 for observed in dataset.train_observed_mask if observed) / len(dataset.train_observed_mask)
    if config.assumed_observation_rate is not None:
        observation_rate = config.assumed_observation_rate
    observation_rate = _clip_observation_rate(observation_rate, config)

    for _ in range(config.epochs):
        predictions = _predict(parameters, dataset.train_features)
        losses = [(prediction - label) ** 2 for prediction, label in zip(predictions, dataset.train_labels)]
        proxy_losses = [
            (prediction - proxy_label) ** 2
            for prediction, proxy_label in zip(predictions, dataset.train_proxy_labels)
        ]
        effective_scores = [
            actual_loss if observed else proxy_loss
            for actual_loss, proxy_loss, observed in zip(
                losses,
                proxy_losses,
                dataset.train_observed_mask,
            )
        ]
        q_values = adversary.update(effective_scores, observation_rate)
        weights = compute_score_based_weights(dataset.train_observed_mask, q_values)
        gradients = _weighted_gradient(parameters, dataset.train_features, dataset.train_labels, weights)
        parameters = [
            parameter - config.learning_rate * gradient
            for parameter, gradient in zip(parameters, gradients)
        ]
    return parameters


def train_robust_group_online(dataset: LinearDataset, config: GradientValidationConfig) -> list[float]:
    return train_robust_group(dataset, replace(config, online_mnar=True))


def train_robust_score_online(dataset: LinearDataset, config: GradientValidationConfig) -> list[float]:
    return train_robust_score(dataset, replace(config, online_mnar=True))


def train_oracle(dataset: LinearDataset, config: GradientValidationConfig) -> list[float]:
    parameters = [0.0 for _ in dataset.train_features[0]]
    weights = _normalize([1.0 for _ in dataset.train_labels])
    for _ in range(config.epochs):
        gradients = _weighted_gradient(parameters, dataset.train_features, dataset.train_labels, weights)
        parameters = [
            parameter - config.learning_rate * gradient
            for parameter, gradient in zip(parameters, gradients)
        ]
    return parameters


def run_gradient_trial(
    trial_index: int,
    rng: random.Random,
    config: GradientValidationConfig,
) -> GradientTrialResult:
    dataset, true_parameters = generate_linear_dataset(rng, config)
    erm_parameters = train_erm(dataset, config)
    robust_parameters = train_robust(dataset, config)
    oracle_parameters = train_oracle(dataset, config)

    erm_predictions = _predict(erm_parameters, dataset.test_features)
    robust_predictions = _predict(robust_parameters, dataset.test_features)
    oracle_predictions = _predict(oracle_parameters, dataset.test_features)
    observation_rate = sum(1 for observed in dataset.train_observed_mask if observed) / len(dataset.train_observed_mask)

    return GradientTrialResult(
        trial_index=trial_index,
        erm_test_mse=_mse(erm_predictions, dataset.test_labels),
        robust_test_mse=_mse(robust_predictions, dataset.test_labels),
        oracle_test_mse=_mse(oracle_predictions, dataset.test_labels),
        erm_parameter_error=_parameter_error(erm_parameters, true_parameters),
        robust_parameter_error=_parameter_error(robust_parameters, true_parameters),
        stable_observation_probability=dataset.stable_observation_probability,
        distressed_observation_probability=dataset.distressed_observation_probability,
        observation_rate=observation_rate,
    )


def summarize_gradient_trials(
    trials: list[GradientTrialResult],
    scenario: str,
    adversary_mode: str,
    online_mnar: bool,
) -> GradientValidationSummary:
    if not trials:
        raise ValueError("summarize_gradient_trials requires at least one trial.")
    robust_beats_erm = sum(1 for trial in trials if trial.robust_test_mse < trial.erm_test_mse)
    return GradientValidationSummary(
        scenario=scenario,
        adversary_mode=adversary_mode,
        online_mnar=online_mnar,
        trials=len(trials),
        robust_beats_erm_rate=robust_beats_erm / len(trials),
        mean_erm_test_mse=mean(trial.erm_test_mse for trial in trials),
        mean_robust_test_mse=mean(trial.robust_test_mse for trial in trials),
        mean_oracle_test_mse=mean(trial.oracle_test_mse for trial in trials),
        mean_test_mse_improvement=mean(trial.erm_test_mse - trial.robust_test_mse for trial in trials),
        mean_erm_parameter_error=mean(trial.erm_parameter_error for trial in trials),
        mean_robust_parameter_error=mean(trial.robust_parameter_error for trial in trials),
        mean_parameter_error_improvement=mean(
            trial.erm_parameter_error - trial.robust_parameter_error for trial in trials
        ),
        mean_observation_rate=mean(trial.observation_rate for trial in trials),
        mean_stable_observation_probability=mean(
            trial.stable_observation_probability for trial in trials
        ),
        mean_distressed_observation_probability=mean(
            trial.distressed_observation_probability for trial in trials
        ),
    )


def run_gradient_validation(
    config: GradientValidationConfig,
) -> tuple[list[GradientTrialResult], GradientValidationSummary]:
    rng = random.Random(config.seed)
    trials = [run_gradient_trial(index, rng, config) for index in range(config.trials)]
    return trials, summarize_gradient_trials(trials, config.scenario, config.adversary_mode, config.online_mnar)


def run_gradient_validation_suite(
    config: GradientValidationConfig,
    scenarios: tuple[str, ...] = VALIDATION_SCENARIOS,
) -> dict[str, GradientValidationSummary]:
    summaries: dict[str, GradientValidationSummary] = {}
    for scenario in scenarios:
        _, summary = run_gradient_validation(replace(config, scenario=scenario))
        summaries[scenario] = summary
    return summaries


def _format_summary(summary: GradientValidationSummary) -> str:
    lines = [
        "Gradient-based selective-observation validation summary",
        f"scenario: {summary.scenario}",
        f"adversary mode: {summary.adversary_mode}",
        f"online MNAR: {summary.online_mnar}",
        f"trials: {summary.trials}",
        f"robust beats ERM by test MSE: {summary.robust_beats_erm_rate:.3f}",
        f"mean ERM test MSE: {summary.mean_erm_test_mse:.6f}",
        f"mean robust test MSE: {summary.mean_robust_test_mse:.6f}",
        f"mean oracle test MSE: {summary.mean_oracle_test_mse:.6f}",
        f"mean test MSE improvement: {summary.mean_test_mse_improvement:.6f}",
        f"mean ERM parameter error: {summary.mean_erm_parameter_error:.6f}",
        f"mean robust parameter error: {summary.mean_robust_parameter_error:.6f}",
        f"mean parameter error improvement: {summary.mean_parameter_error_improvement:.6f}",
        f"mean observation rate: {summary.mean_observation_rate:.3f}",
        f"mean stable observation probability: {summary.mean_stable_observation_probability:.3f}",
        f"mean distressed observation probability: {summary.mean_distressed_observation_probability:.3f}",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> GradientValidationConfig:
    parser = argparse.ArgumentParser(
        description="Run gradient-based validation for selective-observation minimax training."
    )
    parser.add_argument(
        "--scenario",
        choices=[*VALIDATION_SCENARIOS, "suite"],
        default=GradientValidationConfig.scenario,
    )
    parser.add_argument(
        "--adversary-mode",
        choices=["group", "score"],
        default=GradientValidationConfig.adversary_mode,
    )
    parser.add_argument("--seed", type=int, default=GradientValidationConfig.seed)
    parser.add_argument("--trials", type=int, default=GradientValidationConfig.trials)
    parser.add_argument("--learning-rate", type=float, default=GradientValidationConfig.learning_rate)
    parser.add_argument("--epochs", type=int, default=GradientValidationConfig.epochs)
    parser.add_argument("--q-min", type=float, default=Q1ObjectiveConfig.q_min)
    parser.add_argument("--q-max", type=float, default=Q1ObjectiveConfig.q_max)
    parser.add_argument("--adversary-step-size", type=float, default=Q1ObjectiveConfig.adversary_step_size)
    parser.add_argument("--online-mnar", action="store_true")
    parser.add_argument("--assumed-observation-rate", type=float, default=None)
    args = parser.parse_args(argv)
    return GradientValidationConfig(
        seed=args.seed,
        trials=args.trials,
        scenario="aligned_selective" if args.scenario == "suite" else args.scenario,
        adversary_mode=args.adversary_mode,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        online_mnar=args.online_mnar,
        assumed_observation_rate=args.assumed_observation_rate,
        q1=Q1ObjectiveConfig(
            q_min=args.q_min,
            q_max=args.q_max,
            adversary_step_size=args.adversary_step_size,
        ),
    )


def main(argv: list[str] | None = None) -> None:
    raw_args = argv
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scenario", default=GradientValidationConfig.scenario)
    parsed, _unknown = parser.parse_known_args(raw_args)
    config = parse_args(raw_args)
    if parsed.scenario == "suite":
        summaries = run_gradient_validation_suite(config)
        print("\n\n".join(_format_summary(summary) for summary in summaries.values()))
        return
    _trials, summary = run_gradient_validation(config)
    print(_format_summary(summary))


if __name__ == "__main__":
    main()
