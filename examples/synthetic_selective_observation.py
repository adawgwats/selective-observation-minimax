from __future__ import annotations

from minimax_core import (
    Q1ObjectiveConfig,
    SelectiveObservationAdversary,
    estimate_group_snapshot,
    observed_empirical_risk,
)
from minimax_core.objectives import empirical_risk, robust_risk


def build_dataset() -> tuple[list[float], list[str], list[bool], list[float]]:
    observed_labels: list[float] = []
    group_ids: list[str] = []
    observed_mask: list[bool] = []
    latent_labels: list[float] = []

    for _ in range(100):
        group_ids.append("stable")
        latent_labels.append(0.2)
        observed_labels.append(0.2)
        observed_mask.append(True)

    for index in range(100):
        group_ids.append("distressed")
        latent_labels.append(1.0)
        observed_labels.append(1.0)
        observed_mask.append(index < 30)

    return observed_labels, group_ids, observed_mask, latent_labels


def losses_for_theta(theta: float, labels: list[float]) -> list[float]:
    return [(theta - label) ** 2 for label in labels]


def robust_objective_for_theta(theta: float) -> float:
    labels, group_ids, observed_mask, _latent = build_dataset()
    snapshot = estimate_group_snapshot(
        losses=losses_for_theta(theta, labels),
        group_ids=group_ids,
        observed_mask=observed_mask,
    )
    adversary = SelectiveObservationAdversary(Q1ObjectiveConfig(q_min=0.25, q_max=1.0, adversary_step_size=0.05))
    q_values = adversary.current_q(snapshot)
    for _ in range(200):
        q_values = adversary.update(snapshot)
    return robust_risk(snapshot, q_values)


def empirical_objective_for_theta(theta: float) -> float:
    labels, group_ids, observed_mask, _latent = build_dataset()
    snapshot = estimate_group_snapshot(
        losses=losses_for_theta(theta, labels),
        group_ids=group_ids,
        observed_mask=observed_mask,
    )
    return observed_empirical_risk(snapshot)


def latent_objective_for_theta(theta: float) -> float:
    _labels, group_ids, _observed_mask, latent = build_dataset()
    snapshot = estimate_group_snapshot(
        losses=losses_for_theta(theta, latent),
        group_ids=group_ids,
        observed_mask=[True] * len(group_ids),
    )
    return empirical_risk(snapshot)


def argmin(objective) -> tuple[float, float]:
    best_theta = 0.0
    best_value = float("inf")
    for step in range(0, 121):
        theta = step / 100.0
        value = objective(theta)
        if value < best_value:
            best_theta = theta
            best_value = value
    return best_theta, best_value


def main() -> None:
    erm_theta, _ = argmin(empirical_objective_for_theta)
    robust_theta, _ = argmin(robust_objective_for_theta)
    latent_theta, _ = argmin(latent_objective_for_theta)

    print("ERM theta:", erm_theta)
    print("Robust theta:", robust_theta)
    print("Latent full-data theta:", latent_theta)


if __name__ == "__main__":
    main()
