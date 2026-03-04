from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Hashable, Sequence

import numpy as np

try:  # pragma: no cover - exercised indirectly in benchmarks and unit tests.
    import ruptures as rpt
except ImportError:  # pragma: no cover - fallback exists for environments without ruptures.
    rpt = None


PathId = Hashable


@dataclass(frozen=True)
class StructuralBreakDetectionResult:
    break_scores: tuple[float, ...]
    breakpoints: dict[PathId, int]
    normalized_shifts: dict[PathId, float]


@dataclass(frozen=True)
class RupturesStructuralBreakDetector:
    min_size: int = 3
    jump: int = 1
    max_breakpoints: int = 1
    min_normalized_shift: float = 0.6
    break_decay: float = 0.88
    fallback_window: int = 3

    def __post_init__(self) -> None:
        if self.min_size <= 0:
            raise ValueError("min_size must be positive.")
        if self.jump <= 0:
            raise ValueError("jump must be positive.")
        if self.max_breakpoints <= 0:
            raise ValueError("max_breakpoints must be positive.")
        if self.min_normalized_shift < 0.0:
            raise ValueError("min_normalized_shift must be nonnegative.")
        if not 0.0 < self.break_decay <= 1.0:
            raise ValueError("break_decay must be in (0, 1].")
        if self.fallback_window <= 0:
            raise ValueError("fallback_window must be positive.")

    def detect(
        self,
        scores: Sequence[float],
        time_indices: Sequence[int],
        path_ids: Sequence[PathId],
    ) -> StructuralBreakDetectionResult:
        count = len(scores)
        if count == 0:
            raise ValueError("scores must contain at least one value.")
        if len(time_indices) != count or len(path_ids) != count:
            raise ValueError("scores, time_indices, and path_ids must have the same length.")

        break_scores = [0.0 for _ in range(count)]
        breakpoints: dict[PathId, int] = {}
        normalized_shifts: dict[PathId, float] = {}

        grouped_indices: dict[PathId, list[int]] = defaultdict(list)
        for index, path_id in enumerate(path_ids):
            grouped_indices[path_id].append(index)

        for path_id, indices in grouped_indices.items():
            ordered_indices = sorted(indices, key=lambda index: (time_indices[index], index))
            ordered_scores = [float(scores[index]) for index in ordered_indices]
            local_breakpoint, normalized_shift = self._detect_single_breakpoint(ordered_scores)
            if local_breakpoint is None or normalized_shift < self.min_normalized_shift:
                continue

            breakpoints[path_id] = int(time_indices[ordered_indices[local_breakpoint]])
            normalized_shifts[path_id] = normalized_shift
            for local_index, original_index in enumerate(ordered_indices):
                if local_index < local_breakpoint:
                    continue
                distance = local_index - local_breakpoint
                score = normalized_shift * (self.break_decay ** distance)
                break_scores[original_index] = max(break_scores[original_index], score)

        return StructuralBreakDetectionResult(
            break_scores=tuple(break_scores),
            breakpoints=breakpoints,
            normalized_shifts=normalized_shifts,
        )

    def _detect_single_breakpoint(self, signal: Sequence[float]) -> tuple[int | None, float]:
        if len(signal) < max(2 * self.min_size, 4):
            return None, 0.0

        breakpoint_index = self._detect_with_ruptures(signal)
        if breakpoint_index is None:
            breakpoint_index = self._detect_with_fallback(signal)
        if breakpoint_index is None or breakpoint_index <= 0 or breakpoint_index >= len(signal):
            return None, 0.0

        before = signal[:breakpoint_index]
        after = signal[breakpoint_index:]
        shift = abs(mean(after) - mean(before))
        scale = max(pstdev(signal), 1e-6)
        normalized_shift = shift / scale
        return breakpoint_index, normalized_shift

    def _detect_with_ruptures(self, signal: Sequence[float]) -> int | None:
        if rpt is None:
            return None
        min_size = min(self.min_size, max(2, len(signal) // 2 - 1))
        if len(signal) < max(2 * min_size, 4):
            return None

        try:
            algo = rpt.Binseg(model="l2", min_size=min_size, jump=self.jump)
            signal_array = np.asarray(signal, dtype=float).reshape(-1, 1)
            breakpoints = algo.fit(signal_array).predict(n_bkps=self.max_breakpoints)
        except Exception:
            return None

        candidates = [breakpoint for breakpoint in breakpoints if 0 < breakpoint < len(signal)]
        return min(candidates) if candidates else None

    def _detect_with_fallback(self, signal: Sequence[float]) -> int | None:
        window = min(self.fallback_window, max(2, len(signal) // 3))
        if len(signal) < 2 * window:
            return None

        best_index: int | None = None
        best_shift = 0.0
        for index in range(window, len(signal) - window + 1):
            before = signal[index - window:index]
            after = signal[index:index + window]
            shift = abs(mean(after) - mean(before))
            if shift > best_shift:
                best_shift = shift
                best_index = index
        return best_index
