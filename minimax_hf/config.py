from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from minimax_core.config import Q1ObjectiveConfig

TaskType = Literal["sequence_classification", "regression", "token_classification"]


@dataclass(frozen=True)
class MinimaxHFConfig:
    group_key: str = "group_id"
    observed_key: str = "label_observed"
    require_observed_key: bool = False
    task_type: TaskType = "sequence_classification"
    token_ignore_index: int = -100
    q1: Q1ObjectiveConfig = Q1ObjectiveConfig()

    def __post_init__(self) -> None:
        valid_task_types = {"sequence_classification", "regression", "token_classification"}
        if self.task_type not in valid_task_types:
            raise ValueError(f"task_type must be one of {sorted(valid_task_types)}.")
