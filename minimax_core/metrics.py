from __future__ import annotations

from typing import Hashable, Mapping

from .objectives import GroupSnapshot


GroupId = Hashable


def worst_group_loss(snapshot: GroupSnapshot) -> tuple[GroupId, float]:
    return max(
        ((group_id, snapshot.group_losses[group_id]) for group_id in snapshot.group_order),
        key=lambda item: item[1],
    )


def normalized_group_weights(
    snapshot: GroupSnapshot,
    q_values: Mapping[GroupId, float],
) -> dict[GroupId, float]:
    raw_weights = {
        group_id: snapshot.group_priors[group_id] / float(q_values[group_id])
        for group_id in snapshot.group_order
    }
    total = sum(raw_weights.values())
    if total <= 0.0:
        raise ValueError("normalized_group_weights requires positive raw weight sum.")
    return {group_id: weight / total for group_id, weight in raw_weights.items()}
