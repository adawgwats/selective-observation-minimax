from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field, replace
from statistics import mean

from .adversary import ScoreBasedObservationAdversary, SelectiveObservationAdversary
from .config import Q1ObjectiveConfig
from .objectives import (
    compute_score_based_weights,
    empirical_risk,
    estimate_group_snapshot,
    robust_risk,
    score_based_risk,
)


VALIDATION_SCENARIOS = (
    "aligned_selective",
    "group_agnostic",
    "label_dependent",
)


@dataclass(frozen=True)
class MonteCarloConfig:
    seed: int = 7
    trials: int = 100
    scenario: str = "aligned_selective"
    adversary_mode: str = "group"
    stable_count_range: tuple[int, int] = (80, 140)
    distressed_count_range: tuple[int, int] = (80, 140)
    stable_mean_range: tuple[float, float] = (0.05, 0.35)
    distressed_gap_range: tuple[float, float] = (0.45, 0.95)
    label_noise_std_range: tuple[float, float] = (0.02, 0.10)
    proxy_noise_std_range: tuple[float, float] = (0.08, 0.20)
    stable_observation_range: tuple[float, float] = (0.92, 1.0)
    distressed_observation_range: tuple[float, float] = (0.15, 0.45)
    agnostic_observation_range: tuple[float, float] = (0.45, 0.75)
    label_penalty_range: tuple[float, float] = (0.10, 0.30)
    theta_min: float = 0.0
    theta_max: float = 1.4
    theta_step: float = 0.02
    adversary_iterations: int = 100
    min_observed_per_group: int = 1
    q1: Q1ObjectiveConfig = field(default_factory=Q1ObjectiveConfig)

    def __post_init__(self) -> None:
        if self.trials <= 0:
            raise ValueError("trials must be positive.")
        if self.scenario not in VALIDATION_SCENARIOS:
            raise ValueError(f"scenario must be one of {VALIDATION_SCENARIOS}.")
        if self.adversary_mode not in {"group", "score"}:
            raise ValueError("adversary_mode must be 'group' or 'score'.")
        if self.theta_step <= 0.0:
            raise ValueError("theta_step must be positive.")
        if self.theta_max <= self.theta_min:
            raise ValueError("theta_max must be greater than theta_min.")
        if self.adversary_iterations <= 0:
            raise ValueError("adversary_iterations must be positive.")
        if self.min_observed_per_group <= 0:
            raise ValueError("min_observed_per_group must be positive.")
        for name, bounds in (
            ("stable_count_range", self.stable_count_range),
            ("distressed_count_range", self.distressed_count_range),
            ("stable_mean_range", self.stable_mean_range),
            ("distressed_gap_range", self.distressed_gap_range),
            ("label_noise_std_range", self.label_noise_std_range),
            ("proxy_noise_std_range", self.proxy_noise_std_range),
            ("stable_observation_range", self.stable_observation_range),
            ("distressed_observation_range", self.distressed_observation_range),
            ("agnostic_observation_range", self.agnostic_observation_range),
            ("label_penalty_range", self.label_penalty_range),
        ):
            if bounds[0] > bounds[1]:
                raise ValueError(f"{name} lower bound cannot exceed upper bound.")


@dataclass(frozen=True)
class TrialDataset:
    latent_labels: list[float]
    observed_labels: list[float]
    proxy_labels: list[float]
    group_ids: list[str]
    observed_mask: list[bool]
    stable_observation_probability: float
    distressed_observation_probability: float


@dataclass(frozen=True)
class TrialResult:
    trial_index: int
    erm_theta: float
    robust_theta: float
    latent_theta: float
    erm_latent_risk: float
    robust_latent_risk: float
    latent_optimal_risk: float
    stable_observation_probability: float
    distressed_observation_probability: float
    observation_rate: float


@dataclass(frozen=True)
class MonteCarloSummary:
    scenario: str
    adversary_mode: str
    trials: int
    robust_beats_erm_rate: float
    robust_closer_to_latent_theta_rate: float
    mean_erm_latent_risk: float
    mean_robust_latent_risk: float
    mean_latent_optimal_risk: float
    mean_latent_risk_improvement: float
    mean_erm_theta_error: float
    mean_robust_theta_error: float
    mean_theta_error_improvement: float
    mean_observation_rate: float
    mean_stable_observation_probability: float
    mean_distressed_observation_probability: float


def _uniform_int(rng: random.Random, bounds: tuple[int, int]) -> int:
    return rng.randint(bounds[0], bounds[1])


def _uniform_float(rng: random.Random, bounds: tuple[float, float]) -> float:
    if bounds[0] == bounds[1]:
        return bounds[0]
    return rng.uniform(bounds[0], bounds[1])


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _aligned_group_probability(
    rng: random.Random,
    config: MonteCarloConfig,
) -> tuple[float, float]:
    return (
        _uniform_float(rng, config.stable_observation_range),
        _uniform_float(rng, config.distressed_observation_range),
    )


def _agnostic_group_probability(
    rng: random.Random,
    config: MonteCarloConfig,
) -> tuple[float, float]:
    shared = _uniform_float(rng, config.agnostic_observation_range)
    return shared, shared


def _label_dependent_probability(
    label: float,
    base_probability: float,
    stable_mean: float,
    distressed_mean: float,
    penalty: float,
) -> float:
    scale = max(distressed_mean - stable_mean, 1e-9)
    normalized_label = (label - stable_mean) / scale
    return _clip(base_probability - penalty * normalized_label, 0.05, 1.0)


def generate_trial_dataset(rng: random.Random, config: MonteCarloConfig) -> TrialDataset:
    stable_count = _uniform_int(rng, config.stable_count_range)
    distressed_count = _uniform_int(rng, config.distressed_count_range)

    stable_mean = _uniform_float(rng, config.stable_mean_range)
    distressed_mean = stable_mean + _uniform_float(rng, config.distressed_gap_range)
    noise_std = _uniform_float(rng, config.label_noise_std_range)
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

    group_ids: list[str] = []
    latent_labels: list[float] = []
    observed_labels: list[float] = []
    proxy_labels: list[float] = []
    observed_mask: list[bool] = []
    stable_probabilities: list[float] = []
    distressed_probabilities: list[float] = []

    for _ in range(stable_count):
        label = _clip(rng.gauss(stable_mean, noise_std), config.theta_min, config.theta_max)
        if config.scenario == "label_dependent":
            observation_probability = _label_dependent_probability(
                label,
                stable_obs_prob,
                stable_mean,
                distressed_mean,
                label_penalty,
            )
        else:
            observation_probability = stable_obs_prob
        group_ids.append("stable")
        latent_labels.append(label)
        observed_labels.append(label)
        proxy_labels.append(_clip(label + rng.gauss(0.0, proxy_noise_std), config.theta_min, config.theta_max))
        observed_mask.append(rng.random() < observation_probability)
        stable_probabilities.append(observation_probability)

    for _ in range(distressed_count):
        label = _clip(rng.gauss(distressed_mean, noise_std), config.theta_min, config.theta_max)
        if config.scenario == "label_dependent":
            observation_probability = _label_dependent_probability(
                label,
                distressed_obs_prob,
                stable_mean,
                distressed_mean,
                label_penalty,
            )
        else:
            observation_probability = distressed_obs_prob
        group_ids.append("distressed")
        latent_labels.append(label)
        observed_labels.append(label)
        proxy_labels.append(_clip(label + rng.gauss(0.0, proxy_noise_std), config.theta_min, config.theta_max))
        observed_mask.append(rng.random() < observation_probability)
        distressed_probabilities.append(observation_probability)

    # Keep the synthetic study identifiable by ensuring at least one observed label per group.
    if sum(1 for group, observed in zip(group_ids, observed_mask) if group == "stable" and observed) < config.min_observed_per_group:
        for index, group in enumerate(group_ids):
            if group == "stable":
                observed_mask[index] = True
                break
    if sum(1 for group, observed in zip(group_ids, observed_mask) if group == "distressed" and observed) < config.min_observed_per_group:
        for index, group in enumerate(group_ids):
            if group == "distressed":
                observed_mask[index] = True
                break

    return TrialDataset(
        latent_labels=latent_labels,
        observed_labels=observed_labels,
        proxy_labels=proxy_labels,
        group_ids=group_ids,
        observed_mask=observed_mask,
        stable_observation_probability=mean(stable_probabilities),
        distressed_observation_probability=mean(distressed_probabilities),
    )


def losses_for_theta(theta: float, labels: list[float]) -> list[float]:
    return [(theta - label) ** 2 for label in labels]


def latent_risk(theta: float, dataset: TrialDataset) -> float:
    snapshot = estimate_group_snapshot(
        losses=losses_for_theta(theta, dataset.latent_labels),
        group_ids=dataset.group_ids,
        observed_mask=[True] * len(dataset.group_ids),
    )
    return empirical_risk(snapshot)


def observed_mean_theta(dataset: TrialDataset) -> float:
    total = 0.0
    count = 0
    for label, observed in zip(dataset.observed_labels, dataset.observed_mask):
        if observed:
            total += label
            count += 1
    if count == 0:
        raise ValueError("observed_mean_theta requires at least one observed example.")
    return total / count


def latent_mean_theta(dataset: TrialDataset) -> float:
    return sum(dataset.latent_labels) / len(dataset.latent_labels)


def robust_objective_for_theta_group(
    theta: float,
    dataset: TrialDataset,
    config: MonteCarloConfig,
) -> float:
    snapshot = estimate_group_snapshot(
        losses=losses_for_theta(theta, dataset.observed_labels),
        group_ids=dataset.group_ids,
        observed_mask=dataset.observed_mask,
    )
    adversary = SelectiveObservationAdversary(config.q1)
    q_values = adversary.current_q(snapshot)
    for _ in range(config.adversary_iterations):
        q_values = adversary.update(snapshot)
    return robust_risk(snapshot, q_values)


def robust_objective_for_theta_score(
    theta: float,
    dataset: TrialDataset,
    config: MonteCarloConfig,
) -> float:
    losses = losses_for_theta(theta, dataset.observed_labels)
    proxy_losses = losses_for_theta(theta, dataset.proxy_labels)
    effective_scores = [
        actual_loss if observed else proxy_loss
        for actual_loss, proxy_loss, observed in zip(
            losses,
            proxy_losses,
            dataset.observed_mask,
        )
    ]
    observation_rate = sum(1 for observed in dataset.observed_mask if observed) / len(dataset.observed_mask)
    adversary = ScoreBasedObservationAdversary(config.q1)
    q_values = adversary.current_q(effective_scores, observation_rate)
    for _ in range(config.adversary_iterations):
        q_values = adversary.update(effective_scores, observation_rate)
    return score_based_risk(losses, dataset.observed_mask, q_values)


def robust_objective_for_theta(
    theta: float,
    dataset: TrialDataset,
    config: MonteCarloConfig,
) -> float:
    if config.adversary_mode == "group":
        return robust_objective_for_theta_group(theta, dataset, config)
    return robust_objective_for_theta_score(theta, dataset, config)


def grid_argmin(config: MonteCarloConfig, objective) -> tuple[float, float]:
    best_theta = config.theta_min
    best_value = float("inf")
    steps = int(round((config.theta_max - config.theta_min) / config.theta_step))
    for step in range(steps + 1):
        theta = config.theta_min + step * config.theta_step
        value = objective(theta)
        if value < best_value:
            best_theta = theta
            best_value = value
    return best_theta, best_value


def run_trial(trial_index: int, rng: random.Random, config: MonteCarloConfig) -> TrialResult:
    dataset = generate_trial_dataset(rng, config)
    erm_theta = observed_mean_theta(dataset)
    latent_theta = latent_mean_theta(dataset)
    robust_theta, _ = grid_argmin(
        config,
        lambda theta: robust_objective_for_theta(theta, dataset, config),
    )

    observation_rate = sum(1 for observed in dataset.observed_mask if observed) / len(dataset.observed_mask)

    return TrialResult(
        trial_index=trial_index,
        erm_theta=erm_theta,
        robust_theta=robust_theta,
        latent_theta=latent_theta,
        erm_latent_risk=latent_risk(erm_theta, dataset),
        robust_latent_risk=latent_risk(robust_theta, dataset),
        latent_optimal_risk=latent_risk(latent_theta, dataset),
        stable_observation_probability=dataset.stable_observation_probability,
        distressed_observation_probability=dataset.distressed_observation_probability,
        observation_rate=observation_rate,
    )


def summarize_trials(trials: list[TrialResult]) -> MonteCarloSummary:
    if not trials:
        raise ValueError("summarize_trials requires at least one trial.")

    robust_beats_erm = sum(
        1 for trial in trials if trial.robust_latent_risk < trial.erm_latent_risk
    )
    robust_closer = sum(
        1
        for trial in trials
        if abs(trial.robust_theta - trial.latent_theta) < abs(trial.erm_theta - trial.latent_theta)
    )

    return MonteCarloSummary(
        scenario="unknown",
        adversary_mode="unknown",
        trials=len(trials),
        robust_beats_erm_rate=robust_beats_erm / len(trials),
        robust_closer_to_latent_theta_rate=robust_closer / len(trials),
        mean_erm_latent_risk=mean(trial.erm_latent_risk for trial in trials),
        mean_robust_latent_risk=mean(trial.robust_latent_risk for trial in trials),
        mean_latent_optimal_risk=mean(trial.latent_optimal_risk for trial in trials),
        mean_latent_risk_improvement=mean(
            trial.erm_latent_risk - trial.robust_latent_risk for trial in trials
        ),
        mean_erm_theta_error=mean(abs(trial.erm_theta - trial.latent_theta) for trial in trials),
        mean_robust_theta_error=mean(abs(trial.robust_theta - trial.latent_theta) for trial in trials),
        mean_theta_error_improvement=mean(
            abs(trial.erm_theta - trial.latent_theta) - abs(trial.robust_theta - trial.latent_theta)
            for trial in trials
        ),
        mean_observation_rate=mean(trial.observation_rate for trial in trials),
        mean_stable_observation_probability=mean(
            trial.stable_observation_probability for trial in trials
        ),
        mean_distressed_observation_probability=mean(
            trial.distressed_observation_probability for trial in trials
        ),
    )


def run_monte_carlo(config: MonteCarloConfig) -> tuple[list[TrialResult], MonteCarloSummary]:
    rng = random.Random(config.seed)
    trials = [run_trial(index, rng, config) for index in range(config.trials)]
    summary = summarize_trials(trials)
    return trials, replace(
        summary,
        scenario=config.scenario,
        adversary_mode=config.adversary_mode,
    )


def run_validation_suite(
    config: MonteCarloConfig,
    scenarios: tuple[str, ...] = VALIDATION_SCENARIOS,
) -> dict[str, MonteCarloSummary]:
    summaries: dict[str, MonteCarloSummary] = {}
    for scenario in scenarios:
        _, summary = run_monte_carlo(replace(config, scenario=scenario))
        summaries[scenario] = summary
    return summaries


def _format_summary(summary: MonteCarloSummary) -> str:
    lines = [
        "Selective-observation Monte Carlo summary",
        f"scenario: {summary.scenario}",
        f"adversary mode: {summary.adversary_mode}",
        f"trials: {summary.trials}",
        f"robust beats ERM by latent risk: {summary.robust_beats_erm_rate:.3f}",
        f"robust closer to latent theta: {summary.robust_closer_to_latent_theta_rate:.3f}",
        f"mean ERM latent risk: {summary.mean_erm_latent_risk:.6f}",
        f"mean robust latent risk: {summary.mean_robust_latent_risk:.6f}",
        f"mean latent optimal risk: {summary.mean_latent_optimal_risk:.6f}",
        f"mean latent risk improvement: {summary.mean_latent_risk_improvement:.6f}",
        f"mean ERM theta error: {summary.mean_erm_theta_error:.6f}",
        f"mean robust theta error: {summary.mean_robust_theta_error:.6f}",
        f"mean theta error improvement: {summary.mean_theta_error_improvement:.6f}",
        f"mean observation rate: {summary.mean_observation_rate:.3f}",
        f"mean stable observation probability: {summary.mean_stable_observation_probability:.3f}",
        f"mean distressed observation probability: {summary.mean_distressed_observation_probability:.3f}",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> MonteCarloConfig:
    parser = argparse.ArgumentParser(
        description="Run selective-observation Monte Carlo validation for minimax_core."
    )
    parser.add_argument(
        "--scenario",
        choices=[*VALIDATION_SCENARIOS, "suite"],
        default=MonteCarloConfig.scenario,
    )
    parser.add_argument(
        "--adversary-mode",
        choices=["group", "score"],
        default=MonteCarloConfig.adversary_mode,
    )
    parser.add_argument("--seed", type=int, default=MonteCarloConfig.seed)
    parser.add_argument("--trials", type=int, default=MonteCarloConfig.trials)
    parser.add_argument("--theta-min", type=float, default=MonteCarloConfig.theta_min)
    parser.add_argument("--theta-max", type=float, default=MonteCarloConfig.theta_max)
    parser.add_argument("--theta-step", type=float, default=MonteCarloConfig.theta_step)
    parser.add_argument("--adversary-iterations", type=int, default=MonteCarloConfig.adversary_iterations)
    parser.add_argument("--q-min", type=float, default=Q1ObjectiveConfig.q_min)
    parser.add_argument("--q-max", type=float, default=Q1ObjectiveConfig.q_max)
    parser.add_argument("--adversary-step-size", type=float, default=Q1ObjectiveConfig.adversary_step_size)
    args = parser.parse_args(argv)
    return MonteCarloConfig(
        seed=args.seed,
        trials=args.trials,
        scenario="aligned_selective" if args.scenario == "suite" else args.scenario,
        adversary_mode=args.adversary_mode,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        theta_step=args.theta_step,
        adversary_iterations=args.adversary_iterations,
        q1=Q1ObjectiveConfig(
            q_min=args.q_min,
            q_max=args.q_max,
            adversary_step_size=args.adversary_step_size,
        ),
    )


def main(argv: list[str] | None = None) -> None:
    raw_args = argv
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scenario", default=MonteCarloConfig.scenario)
    parsed, _unknown = parser.parse_known_args(raw_args)
    config = parse_args(raw_args)
    if parsed.scenario == "suite":
        summaries = run_validation_suite(config)
        print("\n\n".join(_format_summary(summary) for summary in summaries.values()))
        return
    _trials, summary = run_monte_carlo(config)
    print(_format_summary(summary))


if __name__ == "__main__":
    main()
