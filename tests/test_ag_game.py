from __future__ import annotations

from dataclasses import dataclass

import pytest

from minimax_ag_game.game import (
    action_name,
    affordable_actions,
    best_action_for_method,
    score_actions,
)


@dataclass(frozen=True)
class _Action:
    crop: str
    input_level: str


@dataclass(frozen=True)
class _State:
    cash: float
    debt: float
    credit_limit: float
    acres: float
    land_mortgage_balance: float
    land_mortgage_years_remaining: int
    land_mortgage_grace_years_remaining: int
    year: int

    @property
    def remaining_credit(self) -> float:
        return max(self.credit_limit - max(self.debt, 0.0), 0.0)


def test_action_name_combines_crop_and_input_level() -> None:
    assert action_name(_Action(crop="corn", input_level="medium")) == "corn_medium"


def test_affordable_actions_filters_by_available_capital() -> None:
    state = _State(
        cash=100.0,
        debt=20.0,
        credit_limit=50.0,
        acres=100.0,
        land_mortgage_balance=0.0,
        land_mortgage_years_remaining=0,
        land_mortgage_grace_years_remaining=0,
        year=0,
    )
    low = _Action("corn", "low")
    high = _Action("corn", "high")

    feasible = affordable_actions(
        actions=(low, high),
        state=state,
        planned_operating_cost=lambda action, _acres: 110.0 if action.input_level == "low" else 150.0,
    )

    assert feasible == (low,)


def test_affordable_actions_falls_back_to_all_actions_if_none_fit() -> None:
    state = _State(
        cash=10.0,
        debt=5.0,
        credit_limit=5.0,
        acres=100.0,
        land_mortgage_balance=0.0,
        land_mortgage_years_remaining=0,
        land_mortgage_grace_years_remaining=0,
        year=0,
    )
    actions = (_Action("corn", "low"), _Action("corn", "high"))

    feasible = affordable_actions(
        actions=actions,
        state=state,
        planned_operating_cost=lambda _action, _acres: 1000.0,
    )

    assert feasible == actions


def test_score_actions_respects_action_one_hot_features() -> None:
    state = _State(
        cash=300_000.0,
        debt=0.0,
        credit_limit=100_000.0,
        acres=200.0,
        land_mortgage_balance=100_000.0,
        land_mortgage_years_remaining=20,
        land_mortgage_grace_years_remaining=1,
        year=2,
    )
    actions = (_Action("corn", "low"), _Action("corn", "high"))
    action_index_by_key = {("corn", "low"): 0, ("corn", "high"): 1}

    # 1 bias + 8 state fields + 2 action one-hot features.
    parameters = [0.0] * 11
    parameters[-2] = 0.5
    parameters[-1] = 1.25

    scores = score_actions(
        parameters=parameters,
        state=state,
        actions=actions,
        action_index_by_key=action_index_by_key,
    )

    assert scores["corn_high"] > scores["corn_low"]


def test_best_action_for_method_returns_highest_score() -> None:
    actions = (_Action("corn", "low"), _Action("corn", "high"))
    chosen = best_action_for_method(
        method_scores={"corn_low": 0.2, "corn_high": 0.4},
        candidate_actions=actions,
    )

    assert chosen == "corn_high"


def test_best_action_for_method_requires_candidates() -> None:
    with pytest.raises(ValueError):
        best_action_for_method(method_scores={}, candidate_actions=())
