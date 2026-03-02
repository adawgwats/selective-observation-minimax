from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

from .config import Q1ObjectiveConfig
from .objectives import GroupSnapshot
from .uncertainty import ScoreBasedObservationSet, SelectiveObservationSet


GroupId = Hashable


@dataclass
class SelectiveObservationAdversary:
    config: Q1ObjectiveConfig
    uncertainty_set: SelectiveObservationSet | None = None
    _q_values: dict[GroupId, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.uncertainty_set is None:
            self.uncertainty_set = SelectiveObservationSet(self.config)

    def current_q(self, snapshot: GroupSnapshot) -> dict[GroupId, float]:
        if self._needs_initialization(snapshot):
            self._q_values = self.uncertainty_set.initialize(
                snapshot.group_order,
                snapshot.group_priors,
                snapshot.observation_rate,
            )
        return dict(self._q_values)

    def update(self, snapshot: GroupSnapshot) -> dict[GroupId, float]:
        current_q = self.current_q(snapshot)
        proposed_q: list[float] = []

        for group_id in snapshot.group_order:
            q_value = current_q[group_id]
            loss_value = snapshot.group_losses[group_id]
            prior = snapshot.group_priors[group_id]
            gradient = -prior * loss_value / max(q_value, self.config.epsilon) ** 2
            proposed_q.append(q_value + self.config.adversary_step_size * gradient)

        projected_q = self.uncertainty_set.project(
            snapshot.group_order,
            snapshot.group_priors,
            proposed_q,
            snapshot.observation_rate,
        )
        self._q_values = {
            group_id: value
            for group_id, value in zip(snapshot.group_order, projected_q)
        }
        return dict(self._q_values)

    def _needs_initialization(self, snapshot: GroupSnapshot) -> bool:
        if not self._q_values:
            return True
        if tuple(self._q_values.keys()) != snapshot.group_order:
            return True
        return False


@dataclass
class ScoreBasedObservationAdversary:
    config: Q1ObjectiveConfig
    uncertainty_set: ScoreBasedObservationSet | None = None
    _q_values: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.uncertainty_set is None:
            self.uncertainty_set = ScoreBasedObservationSet(self.config)

    def current_q(
        self,
        scores: list[float],
        observation_rate: float,
    ) -> list[float]:
        if self._needs_initialization(scores):
            self._q_values = self.uncertainty_set.initialize(len(scores), observation_rate)
        return list(self._q_values)

    def update(
        self,
        scores: list[float],
        observation_rate: float,
    ) -> list[float]:
        current_q = self.current_q(scores, observation_rate)
        scaled_scores = self._normalize_scores(scores)
        proposed_q: list[float] = []

        for q_value, score in zip(current_q, scaled_scores):
            gradient = -score / max(q_value, self.config.epsilon) ** 2
            proposed_q.append(q_value + self.config.adversary_step_size * gradient)

        self._q_values = self.uncertainty_set.project(proposed_q, observation_rate)
        return list(self._q_values)

    def _needs_initialization(self, scores: list[float]) -> bool:
        return len(self._q_values) != len(scores)

    @staticmethod
    def _normalize_scores(scores: list[float]) -> list[float]:
        if not scores:
            raise ValueError("scores must contain at least one value.")
        scale = sum(abs(score) for score in scores) / len(scores)
        if scale <= 0.0:
            return [0.0 for _ in scores]
        return [score / scale for score in scores]
