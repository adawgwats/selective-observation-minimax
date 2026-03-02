from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Q1ObjectiveConfig:
    q_min: float = 0.25
    q_max: float = 1.0
    adversary_step_size: float = 0.05
    projection_tolerance: float = 1e-9
    projection_max_iterations: int = 256
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 < self.q_min <= 1.0:
            raise ValueError("q_min must be in (0, 1].")
        if not 0.0 < self.q_max <= 1.0:
            raise ValueError("q_max must be in (0, 1].")
        if self.q_min > self.q_max:
            raise ValueError("q_min must be less than or equal to q_max.")
        if self.adversary_step_size <= 0.0:
            raise ValueError("adversary_step_size must be positive.")
        if self.projection_tolerance <= 0.0:
            raise ValueError("projection_tolerance must be positive.")
        if self.projection_max_iterations <= 0:
            raise ValueError("projection_max_iterations must be positive.")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
