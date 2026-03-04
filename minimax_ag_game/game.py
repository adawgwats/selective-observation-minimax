from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from minimax_core.ag_benchmark import (
    AgricultureBenchmarkConfig,
    _LinearPredictivePolicy,
    _MemoizedCropModel,
    _initial_action_price_histories,
    _build_price_feature_context,
    _build_agriculture_dataset,
    _build_initial_state,
    _dot_product,
    _featurize_decision,
    _require_ag_survival_sim,
    _train_agriculture_methods,
)

DEFAULT_MODEL_METHODS = ("erm", "robust_group", "robust_group_online")


@dataclass(frozen=True)
class TurnContext:
    turn_index: int
    scenario_year: int
    weather_regime: str
    scenario: Any
    human_state: Any
    feasible_actions: tuple[Any, ...]
    all_actions: tuple[Any, ...]
    model_scores_on_human_state: dict[str, dict[str, float]]
    model_actions_on_model_state: dict[str, str]


@dataclass(frozen=True)
class ActorStepOutcome:
    action: str
    net_income: float
    debt_payment: float
    dscr: float
    ending_cash: float
    ending_debt: float
    ending_wealth: float
    cumulative_profit: float
    alive: bool


@dataclass(frozen=True)
class TurnRecord:
    turn_index: int
    scenario_year: int
    weather_regime: str
    human_action: str
    model_actions_on_model_state: dict[str, str]
    outcomes: dict[str, ActorStepOutcome]


@dataclass(frozen=True)
class GameSummary:
    benchmark_name: str
    target: str
    trial_index: int
    path_index: int
    scenario_seed: int
    model_methods: tuple[str, ...]
    turns_available: int
    turns_played: int
    human_survived_full_horizon: bool
    decision_match_counts: dict[str, int]
    decision_match_rates: dict[str, float]
    final_states: dict[str, dict[str, float | int | bool]]
    turns: list[TurnRecord]


@dataclass(frozen=True)
class GameContext:
    config: AgricultureBenchmarkConfig
    trial_index: int
    path_index: int
    scenario_seed: int
    actions: tuple[Any, ...]
    action_index_by_key: dict[tuple[str, str], int]
    method_names: tuple[str, ...]
    method_parameters: dict[str, list[float]]
    policies: dict[str, Any]
    initial_state: Any
    scenario_path: list[Any]
    simulator: Any
    planned_operating_cost: Callable[[Any, float], float]
    include_price_features: bool
    price_history_lags: int
    price_dynamics: Any
    realized_price_fn: Callable[[Any, Any], float] | None
    action_base_price_by_key: dict[tuple[str, str], float]
    initial_price_history_by_action: dict[tuple[str, str], list[float]]


HumanPolicy = Callable[[TurnContext], Any]
TurnCallback = Callable[[TurnContext, TurnRecord], None]


def action_name(action: Any) -> str:
    return f"{action.crop}_{action.input_level}"


def state_snapshot(state: Any) -> dict[str, float | int | bool]:
    return {
        "year": int(state.year),
        "cash": float(state.cash),
        "debt": float(state.debt),
        "remaining_credit": float(state.remaining_credit),
        "land_mortgage_balance": float(state.land_mortgage_balance),
        "land_mortgage_years_remaining": int(state.land_mortgage_years_remaining),
        "land_mortgage_grace_years_remaining": int(state.land_mortgage_grace_years_remaining),
        "cumulative_profit": float(state.cumulative_profit),
        "alive": bool(state.alive),
    }


def affordable_actions(
    *,
    actions: Sequence[Any],
    state: Any,
    planned_operating_cost: Callable[[Any, float], float],
) -> tuple[Any, ...]:
    available_capital = state.cash + state.remaining_credit
    feasible = [
        action
        for action in actions
        if planned_operating_cost(action, state.acres) <= available_capital
    ]
    # Keep the game playable even if all actions are unaffordable in this step.
    return tuple(feasible or list(actions))


def score_actions(
    *,
    parameters: list[float],
    state: Any,
    actions: Sequence[Any],
    action_index_by_key: dict[tuple[str, str], int],
    scenario: Any | None = None,
    price_history_by_action: Mapping[tuple[str, str], Sequence[float]] | None = None,
    include_price_features: bool = False,
    price_history_lags: int = 0,
    price_dynamics: Any = None,
    realized_price_fn: Callable[[Any, Any], float] | None = None,
    action_base_price_by_key: Mapping[tuple[str, str], float] | None = None,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for action in actions:
        price_context = None
        if (
            include_price_features
            and scenario is not None
            and realized_price_fn is not None
            and price_dynamics is not None
        ):
            action_key = (str(action.crop), str(action.input_level))
            history = list(price_history_by_action.get(action_key, [])) if price_history_by_action else []
            spot_price = float(realized_price_fn(action, scenario))
            base_price = float(
                (action_base_price_by_key or {}).get(action_key, max(spot_price, 1.0))
            )
            price_context = _build_price_feature_context(
                history=history,
                spot_price=spot_price,
                base_price=base_price,
                lags=price_history_lags,
                price_dynamics=price_dynamics,
            )
        scores[action_name(action)] = _dot_product(
            parameters,
            _featurize_decision(
                state=state,
                action=action,
                action_index_by_key=action_index_by_key,
                price_context=price_context,
            ),
        )
    return scores


def prepare_game_context(
    config: AgricultureBenchmarkConfig,
    *,
    trial_index: int = 0,
    path_index: int = 0,
    method_names: Sequence[str] | None = None,
) -> GameContext:
    ag = _require_ag_survival_sim()
    FarmState = ag["FarmState"]
    ScenarioGenerator = ag["ScenarioGenerator"]
    FarmSimulator = ag["FarmSimulator"]
    build_benchmark_crop_model = ag["build_benchmark_crop_model"]
    economics_by_action = ag["economics_by_action"]
    get_benchmark_definition = ag["get_benchmark_definition"]
    planned_operating_cost = ag["planned_operating_cost"]
    realized_price = ag["realized_price"]

    dataset = _build_agriculture_dataset(config, trial_index=trial_index)
    trained_parameters = _train_agriculture_methods(config, dataset=dataset)
    selected_methods = tuple(method_names or DEFAULT_MODEL_METHODS)
    missing_methods = sorted(name for name in selected_methods if name not in trained_parameters)
    if missing_methods:
        available = ", ".join(sorted(trained_parameters))
        missing = ", ".join(missing_methods)
        raise ValueError(f"requested model methods not available: {missing}. Available methods: {available}")

    benchmark = get_benchmark_definition(config.benchmark_name)
    action_base_price_by_key = {
        (str(action.crop), str(action.input_level)): float(
            getattr(economics_by_action.get((str(action.crop), str(action.input_level))), "base_price", 1.0)
        )
        for action in benchmark.actions
    }
    initial_fred_bundle = _initial_action_price_histories(
        config=config,
        actions=benchmark.actions,
        action_base_price_by_key=action_base_price_by_key,
    )
    initial_price_history_by_action = (
        initial_fred_bundle.price_history_by_action
        if initial_fred_bundle is not None
        else {}
    )
    scenario_seed = config.seed + 20_000 + trial_index
    scenario_path = ScenarioGenerator(seed=scenario_seed).generate_path(
        config.horizon_years,
        path_index=path_index,
    )
    crop_model = build_benchmark_crop_model(
        config.benchmark_name,
        dssat_root=config.dssat_root,
        workspace_root=str(
            Path(config.workspace_root) / f"{config.benchmark_name}_trial{trial_index}_game"
        ),
    )
    simulator = FarmSimulator(crop_model=_MemoizedCropModel(crop_model))
    initial_state = _build_initial_state(config, FarmState=FarmState)
    policies = {
        method_name: _LinearPredictivePolicy(
            parameters=trained_parameters[method_name],
            actions=benchmark.actions,
            action_index_by_key=dataset.action_index_by_key,
            planned_operating_cost=planned_operating_cost,
            include_price_features=config.include_price_features,
            price_history_lags=config.price_history_lags,
            price_dynamics=config.price_dynamics_config(),
            realized_price_fn=realized_price,
            action_base_price_by_key=action_base_price_by_key,
            initial_price_history_by_action=initial_price_history_by_action,
        )
        for method_name in selected_methods
    }

    return GameContext(
        config=config,
        trial_index=trial_index,
        path_index=path_index,
        scenario_seed=scenario_seed,
        actions=benchmark.actions,
        action_index_by_key=dataset.action_index_by_key,
        method_names=selected_methods,
        method_parameters={name: trained_parameters[name] for name in selected_methods},
        policies=policies,
        initial_state=initial_state,
        scenario_path=scenario_path,
        simulator=simulator,
        planned_operating_cost=planned_operating_cost,
        include_price_features=config.include_price_features,
        price_history_lags=config.price_history_lags,
        price_dynamics=config.price_dynamics_config(),
        realized_price_fn=realized_price,
        action_base_price_by_key=action_base_price_by_key,
        initial_price_history_by_action=initial_price_history_by_action,
    )


def run_turn_based_game(
    context: GameContext,
    *,
    human_policy: HumanPolicy,
    turn_callback: TurnCallback | None = None,
    stop_when_human_bankrupt: bool = True,
) -> GameSummary:
    actor_states: dict[str, Any] = {
        "you": context.initial_state,
        **{method_name: context.initial_state for method_name in context.method_names},
    }
    decision_match_counts = {method_name: 0 for method_name in context.method_names}
    decision_turn_counts = {method_name: 0 for method_name in context.method_names}
    turns: list[TurnRecord] = []
    market_price_history_by_action: dict[tuple[str, str], list[float]] = {
        (str(action.crop), str(action.input_level)): list(
            context.initial_price_history_by_action.get((str(action.crop), str(action.input_level)), [])
        )
        for action in context.actions
    }

    for turn_index, scenario in enumerate(context.scenario_path, start=1):
        human_state = actor_states["you"]
        if not human_state.alive:
            break

        feasible_actions = affordable_actions(
            actions=context.actions,
            state=human_state,
            planned_operating_cost=context.planned_operating_cost,
        )
        model_scores_on_human_state = {
            method_name: score_actions(
                parameters=context.method_parameters[method_name],
                state=human_state,
                actions=context.actions,
                action_index_by_key=context.action_index_by_key,
                scenario=scenario,
                price_history_by_action=market_price_history_by_action,
                include_price_features=context.include_price_features,
                price_history_lags=context.price_history_lags,
                price_dynamics=context.price_dynamics,
                realized_price_fn=context.realized_price_fn,
                action_base_price_by_key=context.action_base_price_by_key,
            )
            for method_name in context.method_names
        }

        model_action_objects: dict[str, Any] = {}
        model_actions_on_model_state: dict[str, str] = {}
        for method_name in context.method_names:
            model_state = actor_states[method_name]
            if not model_state.alive:
                continue
            action = context.policies[method_name].choose_action(model_state, scenario)
            model_action_objects[method_name] = action
            model_actions_on_model_state[method_name] = action_name(action)

        turn_context = TurnContext(
            turn_index=turn_index,
            scenario_year=int(scenario.year_index),
            weather_regime=str(scenario.weather_regime),
            scenario=scenario,
            human_state=human_state,
            feasible_actions=feasible_actions,
            all_actions=context.actions,
            model_scores_on_human_state=model_scores_on_human_state,
            model_actions_on_model_state=model_actions_on_model_state,
        )

        human_action = human_policy(turn_context)
        if action_name(human_action) not in {action_name(action) for action in feasible_actions}:
            valid_actions = ", ".join(action_name(action) for action in feasible_actions)
            raise ValueError(f"human policy selected an infeasible action. Valid choices: {valid_actions}")

        outcomes: dict[str, ActorStepOutcome] = {}
        for actor_name, state in actor_states.items():
            if not state.alive:
                continue
            selected_action = (
                human_action if actor_name == "you" else model_action_objects.get(actor_name)
            )
            if selected_action is None:
                continue
            step_record = context.simulator.step(
                state=state,
                action=selected_action,
                scenario=scenario,
            )
            actor_states[actor_name] = step_record.ending_state
            outcomes[actor_name] = ActorStepOutcome(
                action=action_name(selected_action),
                net_income=float(step_record.net_income),
                debt_payment=float(step_record.debt_payment),
                dscr=float(step_record.dscr),
                ending_cash=float(step_record.ending_state.cash),
                ending_debt=float(step_record.ending_state.debt),
                ending_wealth=float(step_record.ending_state.cash - step_record.ending_state.debt),
                cumulative_profit=float(step_record.ending_state.cumulative_profit),
                alive=bool(step_record.ending_state.alive),
            )

        human_action_key = action_name(human_action)
        for method_name, model_action in model_actions_on_model_state.items():
            decision_turn_counts[method_name] += 1
            if model_action == human_action_key:
                decision_match_counts[method_name] += 1

        record = TurnRecord(
            turn_index=turn_index,
            scenario_year=int(scenario.year_index),
            weather_regime=str(scenario.weather_regime),
            human_action=human_action_key,
            model_actions_on_model_state=model_actions_on_model_state,
            outcomes=outcomes,
        )
        turns.append(record)

        if context.include_price_features and context.realized_price_fn is not None:
            for action in context.actions:
                action_key = (str(action.crop), str(action.input_level))
                history = market_price_history_by_action.setdefault(action_key, [])
                history.append(float(context.realized_price_fn(action, scenario)))
        if turn_callback is not None:
            turn_callback(turn_context, record)

        if stop_when_human_bankrupt and not actor_states["you"].alive:
            break

    decision_match_rates = {
        method_name: (
            decision_match_counts[method_name] / decision_turn_counts[method_name]
            if decision_turn_counts[method_name] > 0
            else 0.0
        )
        for method_name in context.method_names
    }
    final_states = {
        actor_name: state_snapshot(state)
        for actor_name, state in actor_states.items()
    }
    return GameSummary(
        benchmark_name=context.config.benchmark_name,
        target=context.config.target,
        trial_index=context.trial_index,
        path_index=context.path_index,
        scenario_seed=context.scenario_seed,
        model_methods=context.method_names,
        turns_available=len(context.scenario_path),
        turns_played=len(turns),
        human_survived_full_horizon=bool(actor_states["you"].alive) and len(turns) == len(context.scenario_path),
        decision_match_counts=decision_match_counts,
        decision_match_rates=decision_match_rates,
        final_states=final_states,
        turns=turns,
    )


def write_game_summary(summary: GameSummary, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = asdict(summary)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(serialized, handle, indent=2, sort_keys=True)
    return output


def best_action_for_method(
    *,
    method_scores: Mapping[str, float],
    candidate_actions: Sequence[Any],
) -> str:
    if not candidate_actions:
        raise ValueError("candidate_actions must not be empty.")
    ordered_action_names = [action_name(action) for action in candidate_actions]
    return max(ordered_action_names, key=lambda name: method_scores.get(name, float("-inf")))
