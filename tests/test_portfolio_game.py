from __future__ import annotations

from dataclasses import dataclass

import pytest

from minimax_ag_game.portfolio_game import (
    allocation_l1_distance,
    allocation_share_map,
    allocation_to_map,
    validate_allocation,
)


@dataclass(frozen=True)
class _Action:
    crop: str
    input_level: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.crop, self.input_level)


@dataclass(frozen=True)
class _Slice:
    action: _Action
    acres: float


@dataclass(frozen=True)
class _Allocation:
    slices: tuple[_Slice, ...]

    def nonzero_slices(self) -> tuple[_Slice, ...]:
        return tuple(s for s in self.slices if s.acres > 0.0)


@dataclass(frozen=True)
class _State:
    acres: float
    cash: float
    debt: float
    credit_limit: float

    @property
    def remaining_credit(self) -> float:
        return max(self.credit_limit - max(self.debt, 0.0), 0.0)


def test_allocation_to_map_uses_action_names() -> None:
    allocation = _Allocation(
        slices=(
            _Slice(_Action("corn", "high"), 60.0),
            _Slice(_Action("soy", "medium"), 40.0),
        )
    )

    mapped = allocation_to_map(allocation)

    assert mapped == {"corn_high": 60.0, "soy_medium": 40.0}


def test_allocation_share_map_normalizes_by_total_acres() -> None:
    allocation = _Allocation(
        slices=(
            _Slice(_Action("corn", "high"), 60.0),
            _Slice(_Action("soy", "medium"), 40.0),
        )
    )

    shares = allocation_share_map(allocation, total_acres=200.0)

    assert shares["corn_high"] == pytest.approx(0.3)
    assert shares["soy_medium"] == pytest.approx(0.2)


def test_allocation_l1_distance_tracks_share_difference() -> None:
    left = _Allocation(
        slices=(
            _Slice(_Action("corn", "high"), 100.0),
        )
    )
    right = _Allocation(
        slices=(
            _Slice(_Action("soy", "medium"), 100.0),
        )
    )

    distance = allocation_l1_distance(left, right, total_acres=100.0)

    assert distance == pytest.approx(2.0)


def test_validate_allocation_rejects_excess_acres() -> None:
    allocation = _Allocation(
        slices=(
            _Slice(_Action("corn", "high"), 120.0),
        )
    )
    state = _State(acres=100.0, cash=100_000.0, debt=0.0, credit_limit=0.0)

    with pytest.raises(ValueError, match="acreage limit"):
        validate_allocation(
            allocation,
            state=state,
            planned_operating_cost=lambda _action, acres: 400.0 * acres,
        )


def test_validate_allocation_rejects_excess_financing() -> None:
    allocation = _Allocation(
        slices=(
            _Slice(_Action("corn", "high"), 100.0),
        )
    )
    state = _State(acres=100.0, cash=20_000.0, debt=0.0, credit_limit=0.0)

    with pytest.raises(ValueError, match="financing limit"):
        validate_allocation(
            allocation,
            state=state,
            planned_operating_cost=lambda _action, acres: 500.0 * acres,
        )
