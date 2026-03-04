from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from minimax_core.config import Q1ObjectiveConfig

TaskType = Literal["sequence_classification", "regression", "token_classification"]
UncertaintyMode = Literal["group", "score", "time_varying", "knightian", "surprise", "structural_break", "adaptive_v1"]


@dataclass(frozen=True)
class MinimaxHFConfig:
    group_key: str = "group_id"
    observed_key: str = "label_observed"
    time_key: str = "time_index"
    history_key: str = "history_score"
    path_key: str = "path_index"
    require_observed_key: bool = False
    task_type: TaskType = "sequence_classification"
    uncertainty_mode: UncertaintyMode = "group"
    token_ignore_index: int = -100
    online_mnar: bool = False
    assumed_observation_rate: float | None = None
    q1: Q1ObjectiveConfig = Q1ObjectiveConfig()

    def __post_init__(self) -> None:
        valid_task_types = {"sequence_classification", "regression", "token_classification"}
        if self.task_type not in valid_task_types:
            raise ValueError(f"task_type must be one of {sorted(valid_task_types)}.")
        valid_uncertainty_modes = {"group", "score", "time_varying", "knightian", "surprise", "structural_break", "adaptive_v1"}
        if self.uncertainty_mode not in valid_uncertainty_modes:
            raise ValueError(
                f"uncertainty_mode must be one of {sorted(valid_uncertainty_modes)}."
            )
        if self.assumed_observation_rate is not None and not 0.0 < self.assumed_observation_rate <= 1.0:
            raise ValueError("assumed_observation_rate must be in (0, 1].")
