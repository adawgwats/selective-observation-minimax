from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping, Sequence


GroupId = Hashable


@dataclass(frozen=True)
class GroupSnapshot:
    group_order: tuple[GroupId, ...]
    total_counts: dict[GroupId, int]
    observed_counts: dict[GroupId, int]
    group_priors: dict[GroupId, float]
    group_losses: dict[GroupId, float]
    observation_rate: float

    def ordered_priors(self) -> list[float]:
        return [self.group_priors[group_id] for group_id in self.group_order]

    def ordered_losses(self) -> list[float]:
        return [self.group_losses[group_id] for group_id in self.group_order]


def estimate_group_snapshot(
    losses: Sequence[float],
    group_ids: Sequence[GroupId],
    observed_mask: Sequence[bool] | None = None,
    known_groups: Iterable[GroupId] | None = None,
) -> GroupSnapshot:
    if len(losses) != len(group_ids):
        raise ValueError("losses and group_ids must have the same length.")

    if observed_mask is None:
        observed_mask = [True] * len(group_ids)
    if len(observed_mask) != len(group_ids):
        raise ValueError("observed_mask and group_ids must have the same length.")

    seen_order: list[GroupId] = []
    if known_groups is not None:
        seen_order.extend(list(known_groups))

    for group_id in group_ids:
        if group_id not in seen_order:
            seen_order.append(group_id)

    total_counts = {group_id: 0 for group_id in seen_order}
    observed_counts = {group_id: 0 for group_id in seen_order}
    loss_sums = {group_id: 0.0 for group_id in seen_order}

    total_examples = len(group_ids)
    observed_examples = 0

    for loss, group_id, observed in zip(losses, group_ids, observed_mask):
        total_counts[group_id] += 1
        if observed:
            observed_counts[group_id] += 1
            observed_examples += 1
            loss_sums[group_id] += float(loss)

    if total_examples == 0:
        raise ValueError("at least one example is required.")

    group_priors = {
        group_id: total_counts[group_id] / total_examples for group_id in seen_order
    }
    group_losses = {
        group_id: (
            loss_sums[group_id] / observed_counts[group_id]
            if observed_counts[group_id] > 0
            else 0.0
        )
        for group_id in seen_order
    }
    observation_rate = observed_examples / total_examples

    return GroupSnapshot(
        group_order=tuple(seen_order),
        total_counts=total_counts,
        observed_counts=observed_counts,
        group_priors=group_priors,
        group_losses=group_losses,
        observation_rate=observation_rate,
    )


def empirical_risk(snapshot: GroupSnapshot) -> float:
    return sum(
        snapshot.group_priors[group_id] * snapshot.group_losses[group_id]
        for group_id in snapshot.group_order
    )


def observed_empirical_risk(snapshot: GroupSnapshot) -> float:
    total_observed = sum(snapshot.observed_counts.values())
    if total_observed <= 0:
        return 0.0

    return sum(
        (snapshot.observed_counts[group_id] / total_observed) * snapshot.group_losses[group_id]
        for group_id in snapshot.group_order
        if snapshot.observed_counts[group_id] > 0
    )


def robust_risk(snapshot: GroupSnapshot, q_values: Mapping[GroupId, float]) -> float:
    value = 0.0
    for group_id in snapshot.group_order:
        q_value = float(q_values[group_id])
        if q_value <= 0.0:
            raise ValueError("q_values must be strictly positive.")
        value += snapshot.group_priors[group_id] * snapshot.group_losses[group_id] / q_value
    return value


def compute_example_weights(
    snapshot: GroupSnapshot,
    group_ids: Sequence[GroupId],
    observed_mask: Sequence[bool],
    q_values: Mapping[GroupId, float],
) -> list[float]:
    if len(group_ids) != len(observed_mask):
        raise ValueError("group_ids and observed_mask must have the same length.")

    weights: list[float] = []
    for group_id, observed in zip(group_ids, observed_mask):
        if not observed:
            weights.append(0.0)
            continue

        observed_count = snapshot.observed_counts.get(group_id, 0)
        if observed_count <= 0:
            weights.append(0.0)
            continue

        q_value = float(q_values[group_id])
        group_prior = snapshot.group_priors[group_id]
        weights.append(group_prior / (q_value * observed_count))
    return weights


def score_based_risk(
    losses: Sequence[float],
    observed_mask: Sequence[bool],
    q_values: Sequence[float],
) -> float:
    if not (len(losses) == len(observed_mask) == len(q_values)):
        raise ValueError("losses, observed_mask, and q_values must have the same length.")
    example_count = len(losses)
    if example_count <= 0:
        raise ValueError("at least one example is required.")

    total = 0.0
    for loss, observed, q_value in zip(losses, observed_mask, q_values):
        if not observed:
            continue
        q_float = float(q_value)
        if q_float <= 0.0:
            raise ValueError("q_values must be strictly positive.")
        total += float(loss) / (example_count * q_float)
    return total


def compute_score_based_weights(
    observed_mask: Sequence[bool],
    q_values: Sequence[float],
) -> list[float]:
    if len(observed_mask) != len(q_values):
        raise ValueError("observed_mask and q_values must have the same length.")
    example_count = len(q_values)
    if example_count <= 0:
        raise ValueError("at least one q value is required.")

    weights: list[float] = []
    for observed, q_value in zip(observed_mask, q_values):
        if not observed:
            weights.append(0.0)
            continue
        q_float = float(q_value)
        if q_float <= 0.0:
            raise ValueError("q_values must be strictly positive.")
        weights.append(1.0 / (example_count * q_float))
    return weights
