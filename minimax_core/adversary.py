from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Hashable

from .config import Q1ObjectiveConfig
from .objectives import GroupSnapshot
from .uncertainty import (
    GroupedObservationUncertaintySet,
    HistoryAwareObservationUncertaintySet,
    KnightianObservationSet,
    StructuralBreakAwareObservationUncertaintySet,
    StructuralBreakObservationSet,
    SurpriseAwareObservationUncertaintySet,
    SurpriseDrivenObservationSet,
    ScoreBasedObservationSet,
    ScoreObservationUncertaintySet,
    SelectiveObservationSet,
    TimeVaryingObservationSet,
    TimeVaryingObservationUncertaintySet,
)
from .structural_breaks import RupturesStructuralBreakDetector


GroupId = Hashable


@dataclass
class ObservationAdversary(ABC):
    config: Q1ObjectiveConfig


@dataclass
class SelectiveObservationAdversary(ObservationAdversary):
    config: Q1ObjectiveConfig
    uncertainty_set: GroupedObservationUncertaintySet | None = None
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
class ScoreBasedObservationAdversary(ObservationAdversary):
    config: Q1ObjectiveConfig
    uncertainty_set: ScoreObservationUncertaintySet | None = None
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


@dataclass
class TimeVaryingObservationAdversary(ObservationAdversary):
    config: Q1ObjectiveConfig
    uncertainty_set: TimeVaryingObservationUncertaintySet | None = None
    _q_values: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.uncertainty_set is None:
            self.uncertainty_set = TimeVaryingObservationSet(self.config)

    def current_q(
        self,
        scores: list[float],
        observation_rate: float,
        time_indices: list[int],
    ) -> list[float]:
        if self._needs_initialization(scores, time_indices):
            self._q_values = self.uncertainty_set.initialize(
                len(scores),
                observation_rate,
                time_indices,
            )
        return list(self._q_values)

    def update(
        self,
        scores: list[float],
        observation_rate: float,
        time_indices: list[int],
    ) -> list[float]:
        current_q = self.current_q(scores, observation_rate, time_indices)
        scaled_scores = ScoreBasedObservationAdversary._normalize_scores(scores)
        time_factors = self.uncertainty_set.time_factors(time_indices)
        proposed_q: list[float] = []

        for q_value, score, time_factor in zip(current_q, scaled_scores, time_factors):
            gradient = -(score * time_factor) / max(q_value, self.config.epsilon) ** 2
            proposed_q.append(q_value + self.config.adversary_step_size * gradient)

        self._q_values = self.uncertainty_set.project(
            proposed_q,
            observation_rate,
            time_indices,
        )
        return list(self._q_values)

    def _needs_initialization(self, scores: list[float], time_indices: list[int]) -> bool:
        return len(self._q_values) != len(scores) or len(time_indices) != len(scores)


@dataclass
class KnightianObservationAdversary(ObservationAdversary):
    config: Q1ObjectiveConfig
    uncertainty_set: HistoryAwareObservationUncertaintySet | None = None
    _q_values: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.uncertainty_set is None:
            self.uncertainty_set = KnightianObservationSet(self.config)

    def current_q(
        self,
        scores: list[float],
        observation_rate: float,
        time_indices: list[int],
        history_scores: list[float],
    ) -> list[float]:
        if self._needs_initialization(scores, time_indices, history_scores):
            self._q_values = self.uncertainty_set.initialize(
                len(scores),
                observation_rate,
                time_indices,
                history_scores,
            )
        return list(self._q_values)

    def update(
        self,
        scores: list[float],
        observation_rate: float,
        time_indices: list[int],
        history_scores: list[float],
    ) -> list[float]:
        current_q = self.current_q(scores, observation_rate, time_indices, history_scores)
        scaled_scores = ScoreBasedObservationAdversary._normalize_scores(scores)
        ambiguity_factors = self.uncertainty_set.ambiguity_factors(time_indices, history_scores)
        proposed_q: list[float] = []

        for q_value, score, ambiguity_factor in zip(current_q, scaled_scores, ambiguity_factors):
            gradient = -(score * ambiguity_factor) / max(q_value, self.config.epsilon) ** 2
            proposed_q.append(q_value + self.config.adversary_step_size * gradient)

        self._q_values = self.uncertainty_set.project(
            proposed_q,
            observation_rate,
            time_indices,
            history_scores,
        )
        return list(self._q_values)

    def _needs_initialization(
        self,
        scores: list[float],
        time_indices: list[int],
        history_scores: list[float],
    ) -> bool:
        return (
            len(self._q_values) != len(scores)
            or len(time_indices) != len(scores)
            or len(history_scores) != len(scores)
        )


@dataclass
class SurpriseDrivenObservationAdversary(ObservationAdversary):
    config: Q1ObjectiveConfig
    uncertainty_set: SurpriseAwareObservationUncertaintySet | None = None
    surprise_decay: float = 0.85
    _q_values: list[float] = field(default_factory=list)
    _expected_scores: list[float] = field(default_factory=list)
    _surprise_scores: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.uncertainty_set is None:
            self.uncertainty_set = SurpriseDrivenObservationSet(self.config)
        if not 0.0 <= self.surprise_decay < 1.0:
            raise ValueError("surprise_decay must be in [0, 1).")

    def current_q(
        self,
        scores: list[float],
        observation_rate: float,
        time_indices: list[int],
        history_scores: list[float],
    ) -> list[float]:
        if self._needs_initialization(scores, time_indices, history_scores):
            zeros = [0.0 for _ in scores]
            self._q_values = self.uncertainty_set.initialize(
                len(scores),
                observation_rate,
                time_indices,
                history_scores,
                zeros,
            )
            self._expected_scores = zeros
            self._surprise_scores = zeros
        return list(self._q_values)

    def update(
        self,
        scores: list[float],
        observation_rate: float,
        time_indices: list[int],
        history_scores: list[float],
    ) -> list[float]:
        current_q = self.current_q(scores, observation_rate, time_indices, history_scores)
        scaled_scores = ScoreBasedObservationAdversary._normalize_scores(scores)
        surprise_scores = self._update_surprise_state(scaled_scores)
        ambiguity_factors = self.uncertainty_set.ambiguity_factors(
            time_indices,
            history_scores,
            surprise_scores,
        )
        proposed_q: list[float] = []

        for q_value, score, ambiguity_factor in zip(current_q, scaled_scores, ambiguity_factors):
            gradient = -(score * ambiguity_factor) / max(q_value, self.config.epsilon) ** 2
            proposed_q.append(q_value + self.config.adversary_step_size * gradient)

        self._q_values = self.uncertainty_set.project(
            proposed_q,
            observation_rate,
            time_indices,
            history_scores,
            surprise_scores,
        )
        return list(self._q_values)

    def current_surprise_scores(self) -> list[float]:
        return list(self._surprise_scores)

    def _update_surprise_state(self, scaled_scores: list[float]) -> list[float]:
        if not self._expected_scores:
            self._expected_scores = [0.0 for _ in scaled_scores]
        if not self._surprise_scores:
            self._surprise_scores = [0.0 for _ in scaled_scores]

        updated_expected: list[float] = []
        updated_surprise: list[float] = []
        for expected, state, score in zip(self._expected_scores, self._surprise_scores, scaled_scores):
            innovation = abs(score - expected)
            updated_surprise.append(self.surprise_decay * state + (1.0 - self.surprise_decay) * innovation)
            updated_expected.append(self.surprise_decay * expected + (1.0 - self.surprise_decay) * score)

        self._expected_scores = updated_expected
        self._surprise_scores = updated_surprise
        return list(self._surprise_scores)

    def _needs_initialization(
        self,
        scores: list[float],
        time_indices: list[int],
        history_scores: list[float],
    ) -> bool:
        return (
            len(self._q_values) != len(scores)
            or len(time_indices) != len(scores)
            or len(history_scores) != len(scores)
            or len(self._expected_scores) != len(scores)
            or len(self._surprise_scores) != len(scores)
        )


@dataclass
class AutoDiscoveryObservationAdversary(ObservationAdversary):
    config: Q1ObjectiveConfig
    uncertainty_set: SurpriseAwareObservationUncertaintySet | None = None
    score_decay: float = 0.85
    history_decay: float = 0.92
    _q_values: list[float] = field(default_factory=list)
    _seen_examples: int = 0
    _expected_score: float = 0.0
    _surprise_state: float = 0.0
    _history_state: float = 0.0

    def __post_init__(self) -> None:
        if self.uncertainty_set is None:
            self.uncertainty_set = SurpriseDrivenObservationSet(self.config)
        if not 0.0 <= self.score_decay < 1.0:
            raise ValueError("score_decay must be in [0, 1).")
        if not 0.0 <= self.history_decay < 1.0:
            raise ValueError("history_decay must be in [0, 1).")

    def current_q(
        self,
        scores: list[float],
        observation_rate: float,
        observed_mask: list[bool],
    ) -> list[float]:
        time_indices, history_scores, surprise_scores = self._build_context(
            scores,
            observed_mask,
            mutate=False,
        )
        return self._current_q_with_context(
            scores,
            observation_rate,
            time_indices,
            history_scores,
            surprise_scores,
        )

    def update(
        self,
        scores: list[float],
        observation_rate: float,
        observed_mask: list[bool],
    ) -> list[float]:
        time_indices, history_scores, surprise_scores = self._build_context(
            scores,
            observed_mask,
            mutate=False,
        )
        current_q = self._current_q_with_context(
            scores,
            observation_rate,
            time_indices,
            history_scores,
            surprise_scores,
        )
        scaled_scores = ScoreBasedObservationAdversary._normalize_scores(scores)
        ambiguity_factors = self.uncertainty_set.ambiguity_factors(
            time_indices,
            history_scores,
            surprise_scores,
        )
        proposed_q: list[float] = []

        for q_value, score, ambiguity_factor in zip(current_q, scaled_scores, ambiguity_factors):
            gradient = -(score * ambiguity_factor) / max(q_value, self.config.epsilon) ** 2
            proposed_q.append(q_value + self.config.adversary_step_size * gradient)

        self._q_values = self.uncertainty_set.project(
            proposed_q,
            observation_rate,
            time_indices,
            history_scores,
            surprise_scores,
        )
        self._build_context(scores, observed_mask, mutate=True)
        return list(self._q_values)

    def _current_q_with_context(
        self,
        scores: list[float],
        observation_rate: float,
        time_indices: list[int],
        history_scores: list[float],
        surprise_scores: list[float],
    ) -> list[float]:
        if self._needs_initialization(scores):
            self._q_values = self.uncertainty_set.initialize(
                len(scores),
                observation_rate,
                time_indices,
                history_scores,
                surprise_scores,
            )
        return list(self._q_values)

    def _build_context(
        self,
        scores: list[float],
        observed_mask: list[bool],
        *,
        mutate: bool,
    ) -> tuple[list[int], list[float], list[float]]:
        if len(scores) != len(observed_mask):
            raise ValueError("scores and observed_mask must have the same length.")
        scaled_scores = ScoreBasedObservationAdversary._normalize_scores(scores)
        time_indices = list(range(self._seen_examples, self._seen_examples + len(scores)))

        expected_score = self._expected_score
        surprise_state = self._surprise_state
        history_state = self._history_state
        history_scores: list[float] = []
        surprise_scores: list[float] = []

        for score, observed in zip(scaled_scores, observed_mask):
            history_scores.append(history_state)
            innovation = abs(score - expected_score)
            surprise_state = self.score_decay * surprise_state + (1.0 - self.score_decay) * innovation
            surprise_scores.append(surprise_state)
            hidden_signal = 0.0 if observed else 1.0
            history_signal = innovation + hidden_signal
            history_state = self.history_decay * history_state + (1.0 - self.history_decay) * history_signal
            expected_score = self.score_decay * expected_score + (1.0 - self.score_decay) * score

        if mutate:
            self._seen_examples += len(scores)
            self._expected_score = expected_score
            self._surprise_state = surprise_state
            self._history_state = history_state
        return time_indices, history_scores, surprise_scores

    def _needs_initialization(self, scores: list[float]) -> bool:
        return len(self._q_values) != len(scores)


@dataclass
class StructuralBreakObservationAdversary(ObservationAdversary):
    config: Q1ObjectiveConfig
    uncertainty_set: StructuralBreakAwareObservationUncertaintySet | None = None
    detector: RupturesStructuralBreakDetector | None = None
    break_persistence: float = 0.8
    _q_values: list[float] = field(default_factory=list)
    _break_scores: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.uncertainty_set is None:
            self.uncertainty_set = StructuralBreakObservationSet(self.config)
        if self.detector is None:
            self.detector = RupturesStructuralBreakDetector()
        if not 0.0 <= self.break_persistence < 1.0:
            raise ValueError("break_persistence must be in [0, 1).")

    def current_q(
        self,
        scores: list[float],
        observation_rate: float,
        time_indices: list[int],
        history_scores: list[float],
        path_ids: list[Hashable],
    ) -> list[float]:
        if self._needs_initialization(scores, time_indices, history_scores, path_ids):
            zeros = [0.0 for _ in scores]
            self._q_values = self.uncertainty_set.initialize(
                len(scores),
                observation_rate,
                time_indices,
                history_scores,
                zeros,
            )
            self._break_scores = zeros
        return list(self._q_values)

    def update(
        self,
        scores: list[float],
        observation_rate: float,
        time_indices: list[int],
        history_scores: list[float],
        path_ids: list[Hashable],
    ) -> list[float]:
        current_q = self.current_q(scores, observation_rate, time_indices, history_scores, path_ids)
        scaled_scores = ScoreBasedObservationAdversary._normalize_scores(scores)
        detected_breaks = list(
            self.detector.detect(scaled_scores, time_indices, path_ids).break_scores
        )
        break_scores = self._update_break_state(detected_breaks)
        ambiguity_factors = self.uncertainty_set.ambiguity_factors(
            time_indices,
            history_scores,
            break_scores,
        )
        proposed_q: list[float] = []
        for q_value, score, ambiguity_factor in zip(current_q, scaled_scores, ambiguity_factors):
            gradient = -(score * ambiguity_factor) / max(q_value, self.config.epsilon) ** 2
            proposed_q.append(q_value + self.config.adversary_step_size * gradient)

        self._q_values = self.uncertainty_set.project(
            proposed_q,
            observation_rate,
            time_indices,
            history_scores,
            break_scores,
        )
        return list(self._q_values)

    def current_break_scores(self) -> list[float]:
        return list(self._break_scores)

    def _update_break_state(self, detected_breaks: list[float]) -> list[float]:
        if not self._break_scores:
            self._break_scores = [0.0 for _ in detected_breaks]
        self._break_scores = [
            self.break_persistence * prior + (1.0 - self.break_persistence) * current
            for prior, current in zip(self._break_scores, detected_breaks)
        ]
        return list(self._break_scores)

    def _needs_initialization(
        self,
        scores: list[float],
        time_indices: list[int],
        history_scores: list[float],
        path_ids: list[Hashable],
    ) -> bool:
        return (
            len(self._q_values) != len(scores)
            or len(time_indices) != len(scores)
            or len(history_scores) != len(scores)
            or len(path_ids) != len(scores)
            or len(self._break_scores) != len(scores)
        )
