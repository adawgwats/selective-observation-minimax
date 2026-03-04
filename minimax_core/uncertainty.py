from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

from .config import Q1ObjectiveConfig


GroupId = Hashable


@dataclass(frozen=True)
class ObservationUncertaintySet(ABC):
    config: Q1ObjectiveConfig


@dataclass(frozen=True)
class GroupedObservationUncertaintySet(ObservationUncertaintySet, ABC):
    @abstractmethod
    def initialize(
        self,
        group_order: Sequence[GroupId],
        group_priors: Mapping[GroupId, float],
        observation_rate: float,
    ) -> dict[GroupId, float]:
        raise NotImplementedError

    @abstractmethod
    def project(
        self,
        group_order: Sequence[GroupId],
        group_priors: Mapping[GroupId, float],
        proposed_q: Sequence[float],
        observation_rate: float,
    ) -> list[float]:
        raise NotImplementedError


@dataclass(frozen=True)
class ScoreObservationUncertaintySet(ObservationUncertaintySet, ABC):
    @abstractmethod
    def initialize(self, count: int, observation_rate: float) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def project(
        self,
        proposed_q: Sequence[float],
        observation_rate: float,
    ) -> list[float]:
        raise NotImplementedError


@dataclass(frozen=True)
class TimeVaryingObservationUncertaintySet(ObservationUncertaintySet, ABC):
    @abstractmethod
    def initialize(
        self,
        count: int,
        observation_rate: float,
        time_indices: Sequence[int],
    ) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def project(
        self,
        proposed_q: Sequence[float],
        observation_rate: float,
        time_indices: Sequence[int],
    ) -> list[float]:
        raise NotImplementedError


@dataclass(frozen=True)
class HistoryAwareObservationUncertaintySet(ObservationUncertaintySet, ABC):
    @abstractmethod
    def initialize(
        self,
        count: int,
        observation_rate: float,
        time_indices: Sequence[int],
        history_scores: Sequence[float],
    ) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def project(
        self,
        proposed_q: Sequence[float],
        observation_rate: float,
        time_indices: Sequence[int],
        history_scores: Sequence[float],
    ) -> list[float]:
        raise NotImplementedError


@dataclass(frozen=True)
class SurpriseAwareObservationUncertaintySet(ObservationUncertaintySet, ABC):
    @abstractmethod
    def initialize(
        self,
        count: int,
        observation_rate: float,
        time_indices: Sequence[int],
        history_scores: Sequence[float],
        surprise_scores: Sequence[float],
    ) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def project(
        self,
        proposed_q: Sequence[float],
        observation_rate: float,
        time_indices: Sequence[int],
        history_scores: Sequence[float],
        surprise_scores: Sequence[float],
    ) -> list[float]:
        raise NotImplementedError


@dataclass(frozen=True)
class StructuralBreakAwareObservationUncertaintySet(ObservationUncertaintySet, ABC):
    @abstractmethod
    def initialize(
        self,
        count: int,
        observation_rate: float,
        time_indices: Sequence[int],
        history_scores: Sequence[float],
        break_scores: Sequence[float],
    ) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def project(
        self,
        proposed_q: Sequence[float],
        observation_rate: float,
        time_indices: Sequence[int],
        history_scores: Sequence[float],
        break_scores: Sequence[float],
    ) -> list[float]:
        raise NotImplementedError


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length.")
    total_weight = sum(weights)
    if total_weight <= 0.0:
        raise ValueError("weights must sum to a positive value.")
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


def project_to_boxed_weighted_mean(
    values: Sequence[float],
    weights: Sequence[float],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    target_mean: float,
    tolerance: float = 1e-9,
    max_iterations: int = 256,
) -> list[float]:
    if not (len(values) == len(weights) == len(lower_bounds) == len(upper_bounds)):
        raise ValueError("all sequences must have the same length.")
    if not values:
        raise ValueError("at least one value is required.")

    normalized_weights = list(weights)
    weight_sum = sum(normalized_weights)
    if weight_sum <= 0.0:
        raise ValueError("weights must sum to a positive value.")
    normalized_weights = [weight / weight_sum for weight in normalized_weights]

    for lower, upper in zip(lower_bounds, upper_bounds):
        if lower > upper:
            raise ValueError("lower bound cannot exceed upper bound.")

    min_feasible = weighted_mean(lower_bounds, normalized_weights)
    max_feasible = weighted_mean(upper_bounds, normalized_weights)
    if target_mean < min_feasible - tolerance or target_mean > max_feasible + tolerance:
        raise ValueError("target_mean is infeasible for the provided bounds.")

    if abs(weighted_mean(values, normalized_weights) - target_mean) <= tolerance:
        return [
            min(max(value, lower), upper)
            for value, lower, upper in zip(values, lower_bounds, upper_bounds)
        ]

    lo = min(
        (value - upper) / weight
        for value, upper, weight in zip(values, upper_bounds, normalized_weights)
    )
    hi = max(
        (value - lower) / weight
        for value, lower, weight in zip(values, lower_bounds, normalized_weights)
    )

    projected = list(values)
    for _ in range(max_iterations):
        mid = (lo + hi) / 2.0
        projected = [
            min(max(value - mid * weight, lower), upper)
            for value, weight, lower, upper in zip(
                values,
                normalized_weights,
                lower_bounds,
                upper_bounds,
            )
        ]
        current_mean = weighted_mean(projected, normalized_weights)
        if abs(current_mean - target_mean) <= tolerance:
            break
        if current_mean > target_mean:
            lo = mid
        else:
            hi = mid
    return projected


@dataclass(frozen=True)
class SelectiveObservationSet(GroupedObservationUncertaintySet):

    def _lower_bounds(self, group_order: Sequence[GroupId]) -> list[float]:
        return [self.config.q_min for _ in group_order]

    def _upper_bounds(self, group_order: Sequence[GroupId]) -> list[float]:
        return [self.config.q_max for _ in group_order]

    def initialize(
        self,
        group_order: Sequence[GroupId],
        group_priors: Mapping[GroupId, float],
        observation_rate: float,
    ) -> dict[GroupId, float]:
        initial = [observation_rate for _ in group_order]
        projected = self.project(group_order, group_priors, initial, observation_rate)
        return {group_id: value for group_id, value in zip(group_order, projected)}

    def project(
        self,
        group_order: Sequence[GroupId],
        group_priors: Mapping[GroupId, float],
        proposed_q: Sequence[float],
        observation_rate: float,
    ) -> list[float]:
        priors = [float(group_priors[group_id]) for group_id in group_order]
        lower_bounds = self._lower_bounds(group_order)
        upper_bounds = self._upper_bounds(group_order)
        return project_to_boxed_weighted_mean(
            values=proposed_q,
            weights=priors,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            target_mean=observation_rate,
            tolerance=self.config.projection_tolerance,
            max_iterations=self.config.projection_max_iterations,
        )


@dataclass(frozen=True)
class ScoreBasedObservationSet(ScoreObservationUncertaintySet):

    def initialize(self, count: int, observation_rate: float) -> list[float]:
        if count <= 0:
            raise ValueError("count must be positive.")
        initial = [observation_rate for _ in range(count)]
        return self.project(initial, observation_rate)

    def project(
        self,
        proposed_q: Sequence[float],
        observation_rate: float,
    ) -> list[float]:
        count = len(proposed_q)
        if count <= 0:
            raise ValueError("at least one proposed q value is required.")
        uniform_weights = [1.0 for _ in range(count)]
        lower_bounds = [self.config.q_min for _ in range(count)]
        upper_bounds = [self.config.q_max for _ in range(count)]
        return project_to_boxed_weighted_mean(
            values=proposed_q,
            weights=uniform_weights,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            target_mean=observation_rate,
            tolerance=self.config.projection_tolerance,
            max_iterations=self.config.projection_max_iterations,
        )


@dataclass(frozen=True)
class TimeVaryingObservationSet(TimeVaryingObservationUncertaintySet):
    time_strength: float = 0.5
    min_projection_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.time_strength < 0.0:
            raise ValueError("time_strength must be nonnegative.")
        if not 0.0 < self.min_projection_weight <= 1.0:
            raise ValueError("min_projection_weight must be in (0, 1].")

    def initialize(
        self,
        count: int,
        observation_rate: float,
        time_indices: Sequence[int],
    ) -> list[float]:
        if count <= 0:
            raise ValueError("count must be positive.")
        if len(time_indices) != count:
            raise ValueError("time_indices must match count.")
        initial = [observation_rate for _ in range(count)]
        return self.project(initial, observation_rate, time_indices)

    def project(
        self,
        proposed_q: Sequence[float],
        observation_rate: float,
        time_indices: Sequence[int],
    ) -> list[float]:
        count = len(proposed_q)
        if count <= 0:
            raise ValueError("at least one proposed q value is required.")
        if len(time_indices) != count:
            raise ValueError("time_indices must have the same length as proposed_q.")
        projection_weights = self.projection_weights(time_indices)
        lower_bounds = [self.config.q_min for _ in range(count)]
        upper_bounds = [self.config.q_max for _ in range(count)]
        return project_to_boxed_weighted_mean(
            values=proposed_q,
            weights=projection_weights,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            target_mean=observation_rate,
            tolerance=self.config.projection_tolerance,
            max_iterations=self.config.projection_max_iterations,
        )

    def time_factors(self, time_indices: Sequence[int]) -> list[float]:
        normalized = self._normalize_time_indices(time_indices)
        return [1.0 + self.time_strength * position for position in normalized]

    def projection_weights(self, time_indices: Sequence[int]) -> list[float]:
        return [
            max(1.0 / factor, self.min_projection_weight)
            for factor in self.time_factors(time_indices)
        ]

    @staticmethod
    def _normalize_time_indices(time_indices: Sequence[int]) -> list[float]:
        if not time_indices:
            raise ValueError("time_indices must contain at least one value.")
        min_time = min(time_indices)
        max_time = max(time_indices)
        if min_time == max_time:
            return [0.0 for _ in time_indices]
        scale = max_time - min_time
        return [(time_index - min_time) / scale for time_index in time_indices]


@dataclass(frozen=True)
class KnightianObservationSet(HistoryAwareObservationUncertaintySet):
    time_strength: float = 0.35
    history_strength: float = 1.0
    min_projection_weight: float = 0.2

    def __post_init__(self) -> None:
        if self.time_strength < 0.0:
            raise ValueError("time_strength must be nonnegative.")
        if self.history_strength < 0.0:
            raise ValueError("history_strength must be nonnegative.")
        if not 0.0 < self.min_projection_weight <= 1.0:
            raise ValueError("min_projection_weight must be in (0, 1].")

    def initialize(
        self,
        count: int,
        observation_rate: float,
        time_indices: Sequence[int],
        history_scores: Sequence[float],
    ) -> list[float]:
        if count <= 0:
            raise ValueError("count must be positive.")
        if len(time_indices) != count or len(history_scores) != count:
            raise ValueError("time_indices and history_scores must match count.")
        initial = [observation_rate for _ in range(count)]
        return self.project(initial, observation_rate, time_indices, history_scores)

    def project(
        self,
        proposed_q: Sequence[float],
        observation_rate: float,
        time_indices: Sequence[int],
        history_scores: Sequence[float],
    ) -> list[float]:
        count = len(proposed_q)
        if count <= 0:
            raise ValueError("at least one proposed q value is required.")
        if len(time_indices) != count or len(history_scores) != count:
            raise ValueError("time_indices and history_scores must match proposed_q.")
        projection_weights = self.projection_weights(time_indices, history_scores)
        lower_bounds = [self.config.q_min for _ in range(count)]
        upper_bounds = [self.config.q_max for _ in range(count)]
        return project_to_boxed_weighted_mean(
            values=proposed_q,
            weights=projection_weights,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            target_mean=observation_rate,
            tolerance=self.config.projection_tolerance,
            max_iterations=self.config.projection_max_iterations,
        )

    def ambiguity_factors(
        self,
        time_indices: Sequence[int],
        history_scores: Sequence[float],
    ) -> list[float]:
        normalized_time = TimeVaryingObservationSet._normalize_time_indices(time_indices)
        normalized_history = self._normalize_history_scores(history_scores)
        return [
            1.0 + self.time_strength * time_score + self.history_strength * history_score
            for time_score, history_score in zip(normalized_time, normalized_history)
        ]

    def projection_weights(
        self,
        time_indices: Sequence[int],
        history_scores: Sequence[float],
    ) -> list[float]:
        return [
            max(1.0 / factor, self.min_projection_weight)
            for factor in self.ambiguity_factors(time_indices, history_scores)
        ]

    @staticmethod
    def _normalize_history_scores(history_scores: Sequence[float]) -> list[float]:
        return normalize_context_scores(history_scores, label="history_scores")


@dataclass(frozen=True)
class SurpriseDrivenObservationSet(SurpriseAwareObservationUncertaintySet):
    time_strength: float = 0.25
    history_strength: float = 0.6
    surprise_strength: float = 1.35
    min_projection_weight: float = 0.15

    def __post_init__(self) -> None:
        if self.time_strength < 0.0:
            raise ValueError("time_strength must be nonnegative.")
        if self.history_strength < 0.0:
            raise ValueError("history_strength must be nonnegative.")
        if self.surprise_strength < 0.0:
            raise ValueError("surprise_strength must be nonnegative.")
        if not 0.0 < self.min_projection_weight <= 1.0:
            raise ValueError("min_projection_weight must be in (0, 1].")

    def initialize(
        self,
        count: int,
        observation_rate: float,
        time_indices: Sequence[int],
        history_scores: Sequence[float],
        surprise_scores: Sequence[float],
    ) -> list[float]:
        if count <= 0:
            raise ValueError("count must be positive.")
        if len(time_indices) != count or len(history_scores) != count or len(surprise_scores) != count:
            raise ValueError("time_indices, history_scores, and surprise_scores must match count.")
        initial = [observation_rate for _ in range(count)]
        return self.project(
            initial,
            observation_rate,
            time_indices,
            history_scores,
            surprise_scores,
        )

    def project(
        self,
        proposed_q: Sequence[float],
        observation_rate: float,
        time_indices: Sequence[int],
        history_scores: Sequence[float],
        surprise_scores: Sequence[float],
    ) -> list[float]:
        count = len(proposed_q)
        if count <= 0:
            raise ValueError("at least one proposed q value is required.")
        if len(time_indices) != count or len(history_scores) != count or len(surprise_scores) != count:
            raise ValueError("time_indices, history_scores, and surprise_scores must match proposed_q.")
        projection_weights = self.projection_weights(time_indices, history_scores, surprise_scores)
        lower_bounds = [self.config.q_min for _ in range(count)]
        upper_bounds = [self.config.q_max for _ in range(count)]
        return project_to_boxed_weighted_mean(
            values=proposed_q,
            weights=projection_weights,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            target_mean=observation_rate,
            tolerance=self.config.projection_tolerance,
            max_iterations=self.config.projection_max_iterations,
        )

    def ambiguity_factors(
        self,
        time_indices: Sequence[int],
        history_scores: Sequence[float],
        surprise_scores: Sequence[float],
    ) -> list[float]:
        normalized_time = TimeVaryingObservationSet._normalize_time_indices(time_indices)
        normalized_history = normalize_context_scores(history_scores, label="history_scores")
        normalized_surprise = normalize_context_scores(surprise_scores, label="surprise_scores")
        return [
            1.0
            + self.time_strength * time_score
            + self.history_strength * history_score
            + self.surprise_strength * surprise_score
            for time_score, history_score, surprise_score in zip(
                normalized_time,
                normalized_history,
                normalized_surprise,
            )
        ]

    def projection_weights(
        self,
        time_indices: Sequence[int],
        history_scores: Sequence[float],
        surprise_scores: Sequence[float],
    ) -> list[float]:
        return [
            max(1.0 / factor, self.min_projection_weight)
            for factor in self.ambiguity_factors(time_indices, history_scores, surprise_scores)
        ]


@dataclass(frozen=True)
class StructuralBreakObservationSet(StructuralBreakAwareObservationUncertaintySet):
    time_strength: float = 0.2
    history_strength: float = 0.45
    break_strength: float = 1.8
    min_projection_weight: float = 0.12

    def __post_init__(self) -> None:
        if self.time_strength < 0.0:
            raise ValueError("time_strength must be nonnegative.")
        if self.history_strength < 0.0:
            raise ValueError("history_strength must be nonnegative.")
        if self.break_strength < 0.0:
            raise ValueError("break_strength must be nonnegative.")
        if not 0.0 < self.min_projection_weight <= 1.0:
            raise ValueError("min_projection_weight must be in (0, 1].")

    def initialize(
        self,
        count: int,
        observation_rate: float,
        time_indices: Sequence[int],
        history_scores: Sequence[float],
        break_scores: Sequence[float],
    ) -> list[float]:
        if count <= 0:
            raise ValueError("count must be positive.")
        if len(time_indices) != count or len(history_scores) != count or len(break_scores) != count:
            raise ValueError("time_indices, history_scores, and break_scores must match count.")
        initial = [observation_rate for _ in range(count)]
        return self.project(
            initial,
            observation_rate,
            time_indices,
            history_scores,
            break_scores,
        )

    def project(
        self,
        proposed_q: Sequence[float],
        observation_rate: float,
        time_indices: Sequence[int],
        history_scores: Sequence[float],
        break_scores: Sequence[float],
    ) -> list[float]:
        count = len(proposed_q)
        if count <= 0:
            raise ValueError("at least one proposed q value is required.")
        if len(time_indices) != count or len(history_scores) != count or len(break_scores) != count:
            raise ValueError("time_indices, history_scores, and break_scores must match proposed_q.")
        projection_weights = self.projection_weights(time_indices, history_scores, break_scores)
        lower_bounds = [self.config.q_min for _ in range(count)]
        upper_bounds = [self.config.q_max for _ in range(count)]
        return project_to_boxed_weighted_mean(
            values=proposed_q,
            weights=projection_weights,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            target_mean=observation_rate,
            tolerance=self.config.projection_tolerance,
            max_iterations=self.config.projection_max_iterations,
        )

    def ambiguity_factors(
        self,
        time_indices: Sequence[int],
        history_scores: Sequence[float],
        break_scores: Sequence[float],
    ) -> list[float]:
        normalized_time = TimeVaryingObservationSet._normalize_time_indices(time_indices)
        normalized_history = normalize_context_scores(history_scores, label="history_scores")
        normalized_breaks = normalize_context_scores(break_scores, label="break_scores")
        return [
            1.0
            + self.time_strength * time_score
            + self.history_strength * history_score
            + self.break_strength * break_score
            for time_score, history_score, break_score in zip(
                normalized_time,
                normalized_history,
                normalized_breaks,
            )
        ]

    def projection_weights(
        self,
        time_indices: Sequence[int],
        history_scores: Sequence[float],
        break_scores: Sequence[float],
    ) -> list[float]:
        return [
            max(1.0 / factor, self.min_projection_weight)
            for factor in self.ambiguity_factors(time_indices, history_scores, break_scores)
        ]


def normalize_context_scores(scores: Sequence[float], *, label: str) -> list[float]:
    if not scores:
        raise ValueError(f"{label} must contain at least one value.")
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [0.0 for _ in scores]
    scale = max_score - min_score
    return [(score - min_score) / scale for score in scores]
