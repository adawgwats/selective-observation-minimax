from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from minimax_core.ag_benchmark import AgricultureBenchmarkConfig

DEFAULT_PORTFOLIO_POLICIES = ("greedy_margin", "christensen_knightian")


@dataclass(frozen=True)
class PortfolioTurnContext:
    turn_index: int
    scenario_year: int
    weather_regime: str
    scenario: Any
    human_state: Any
    action_options: tuple[Any, ...]
    model_allocations_on_model_state: dict[str, dict[str, float]]
    model_planned_costs_on_model_state: dict[str, float]


@dataclass(frozen=True)
class PortfolioActorStepOutcome:
    allocation: dict[str, float]
    planned_cost: float
    net_income: float
    debt_payment: float
    dscr: float
    ending_cash: float
    ending_debt: float
    ending_wealth: float
    cumulative_profit: float
    alive: bool
    components: tuple[dict[str, float | str], ...]


@dataclass(frozen=True)
class PortfolioTurnRecord:
    turn_index: int
    scenario_year: int
    weather_regime: str
    human_allocation: dict[str, float]
    model_allocations_on_model_state: dict[str, dict[str, float]]
    allocation_l1_distance_vs_models: dict[str, float]
    outcomes: dict[str, PortfolioActorStepOutcome]


@dataclass(frozen=True)
class PortfolioGameSummary:
    benchmark_name: str
    portfolio_benchmark_name: str
    target: str
    trial_index: int
    path_index: int
    scenario_seed: int
    model_policies: tuple[str, ...]
    turns_available: int
    turns_played: int
    human_survived_full_horizon: bool
    mean_allocation_l1_distance: dict[str, float]
    final_states: dict[str, dict[str, float | int | bool]]
    turns: list[PortfolioTurnRecord]


@dataclass(frozen=True)
class PortfolioGameContext:
    config: AgricultureBenchmarkConfig
    trial_index: int
    path_index: int
    scenario_seed: int
    portfolio_benchmark_name: str
    action_options: tuple[Any, ...]
    initial_state: Any
    scenario_path: list[Any]
    simulator: Any
    policies: dict[str, Any]
    policy_names: tuple[str, ...]
    planned_operating_cost: Callable[[Any, float], float]
    AllocationSlice: Any
    PortfolioAllocation: Any


PortfolioHumanPolicy = Callable[[PortfolioTurnContext], Any]
PortfolioTurnCallback = Callable[[PortfolioTurnContext, PortfolioTurnRecord], None]


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


def allocation_to_map(allocation: Any) -> dict[str, float]:
    return {
        action_name(allocation_slice.action): float(allocation_slice.acres)
        for allocation_slice in allocation.nonzero_slices()
    }


def allocation_total_cost(
    allocation: Any,
    *,
    planned_operating_cost: Callable[[Any, float], float],
) -> float:
    return sum(
        planned_operating_cost(allocation_slice.action, allocation_slice.acres)
        for allocation_slice in allocation.nonzero_slices()
    )


def allocation_share_map(
    allocation: Any,
    *,
    total_acres: float,
) -> dict[str, float]:
    if total_acres <= 0.0:
        return {}
    return {
        key: acres / total_acres
        for key, acres in allocation_to_map(allocation).items()
    }


def allocation_l1_distance(
    left: Any,
    right: Any,
    *,
    total_acres: float,
) -> float:
    left_share = allocation_share_map(left, total_acres=total_acres)
    right_share = allocation_share_map(right, total_acres=total_acres)
    keys = set(left_share) | set(right_share)
    return sum(abs(left_share.get(key, 0.0) - right_share.get(key, 0.0)) for key in keys)


def validate_allocation(
    allocation: Any,
    *,
    state: Any,
    planned_operating_cost: Callable[[Any, float], float],
) -> None:
    total_acres = sum(allocation_slice.acres for allocation_slice in allocation.nonzero_slices())
    if total_acres > state.acres + 1e-9:
        raise ValueError(f"allocation exceeds acreage limit ({total_acres:.2f} > {state.acres:.2f}).")
    planned_cost = allocation_total_cost(
        allocation,
        planned_operating_cost=planned_operating_cost,
    )
    available_capital = state.cash + state.remaining_credit
    if planned_cost > available_capital + 1e-9:
        raise ValueError(
            f"allocation exceeds financing limit ({planned_cost:,.0f} > {available_capital:,.0f})."
        )


def available_portfolio_benchmark_names() -> tuple[str, ...]:
    try:
        from ag_survival_sim import list_portfolio_benchmark_definitions  # type: ignore[import-not-found]
    except ImportError:
        return ("georgia_diversified_portfolio",)
    return tuple(definition.name for definition in list_portfolio_benchmark_definitions())


def prepare_portfolio_game_context(
    config: AgricultureBenchmarkConfig,
    *,
    portfolio_benchmark_name: str = "georgia_diversified_portfolio",
    trial_index: int = 0,
    path_index: int = 0,
    policy_names: Sequence[str] | None = None,
    include_learned_policy: bool = False,
) -> PortfolioGameContext:
    try:
        from ag_survival_sim import (  # type: ignore[import-not-found]
            AllocationSlice,
            FarmState,
            PortfolioAllocation,
            ScenarioGenerator,
            build_portfolio_benchmark_crop_model,
            build_portfolio_demo_policies,
            get_portfolio_benchmark_definition,
            planned_operating_cost,
        )
        from ag_survival_sim.portfolio_simulator import (  # type: ignore[import-not-found]
            PortfolioFarmSimulator,
        )
    except ImportError as error:
        raise ImportError(
            "Portfolio game mode requires ag-survival-sim with portfolio modules available."
        ) from error

    benchmark = get_portfolio_benchmark_definition(portfolio_benchmark_name)
    scenario_seed = config.seed + 20_000 + trial_index
    scenario_path = ScenarioGenerator(seed=scenario_seed).generate_path(
        config.horizon_years,
        path_index=path_index,
    )
    crop_model = build_portfolio_benchmark_crop_model(
        portfolio_benchmark_name,
        dssat_root=config.dssat_root,
        workspace_root=str(
            Path(config.workspace_root) / f"{portfolio_benchmark_name}_trial{trial_index}_portfolio_game"
        ),
    )
    simulator = PortfolioFarmSimulator(crop_model=crop_model)
    initial_state = FarmState.initial(
        cash=config.initial_cash,
        debt=config.initial_debt,
        credit_limit=config.initial_credit_limit,
        acres=config.acres,
        land_value_per_acre=config.land_value_per_acre,
        land_financed_fraction=config.land_financed_fraction,
        land_mortgage_rate=config.land_mortgage_rate,
        land_mortgage_years=config.land_mortgage_years,
        land_mortgage_grace_years=config.land_mortgage_grace_years,
    )
    all_policies = build_portfolio_demo_policies(
        portfolio_benchmark_name,
        crop_model=crop_model,
        include_learned_policy=include_learned_policy,
        initial_state=initial_state,
    )
    selected_policy_names = tuple(policy_names or DEFAULT_PORTFOLIO_POLICIES)
    missing = sorted(name for name in selected_policy_names if name not in all_policies)
    if missing:
        available = ", ".join(sorted(all_policies))
        raise ValueError(
            f"requested portfolio policies not available: {', '.join(missing)}. "
            f"Available policies: {available}"
        )

    return PortfolioGameContext(
        config=config,
        trial_index=trial_index,
        path_index=path_index,
        scenario_seed=scenario_seed,
        portfolio_benchmark_name=portfolio_benchmark_name,
        action_options=tuple(option.action for option in benchmark.options),
        initial_state=initial_state,
        scenario_path=scenario_path,
        simulator=simulator,
        policies={name: all_policies[name] for name in selected_policy_names},
        policy_names=selected_policy_names,
        planned_operating_cost=planned_operating_cost,
        AllocationSlice=AllocationSlice,
        PortfolioAllocation=PortfolioAllocation,
    )


def build_allocation(
    context: PortfolioGameContext,
    *,
    acres_by_action_name: Mapping[str, float],
) -> Any:
    by_key = {action_name(action): action for action in context.action_options}
    slices = []
    for key, acres in acres_by_action_name.items():
        if key not in by_key:
            raise ValueError(f"unknown action key in allocation: {key}")
        if acres < 0.0:
            raise ValueError(f"allocation acres must be nonnegative for action {key}.")
        if acres <= 1e-9:
            continue
        slices.append(context.AllocationSlice(action=by_key[key], acres=float(acres)))
    return context.PortfolioAllocation(tuple(slices))


def run_turn_based_portfolio_game(
    context: PortfolioGameContext,
    *,
    human_policy: PortfolioHumanPolicy,
    turn_callback: PortfolioTurnCallback | None = None,
    stop_when_human_bankrupt: bool = True,
) -> PortfolioGameSummary:
    actor_states: dict[str, Any] = {
        "you": context.initial_state,
        **{policy_name: context.initial_state for policy_name in context.policy_names},
    }
    turns: list[PortfolioTurnRecord] = []
    per_model_distances: dict[str, list[float]] = {name: [] for name in context.policy_names}

    for turn_index, scenario in enumerate(context.scenario_path, start=1):
        human_state = actor_states["you"]
        if not human_state.alive:
            break

        model_allocations: dict[str, Any] = {}
        model_allocation_maps: dict[str, dict[str, float]] = {}
        model_planned_costs: dict[str, float] = {}
        for policy_name in context.policy_names:
            model_state = actor_states[policy_name]
            if not model_state.alive:
                continue
            allocation = context.policies[policy_name].choose_allocation(model_state, scenario)
            model_allocations[policy_name] = allocation
            model_allocation_maps[policy_name] = allocation_to_map(allocation)
            model_planned_costs[policy_name] = allocation_total_cost(
                allocation,
                planned_operating_cost=context.planned_operating_cost,
            )

        turn_context = PortfolioTurnContext(
            turn_index=turn_index,
            scenario_year=int(scenario.year_index),
            weather_regime=str(scenario.weather_regime),
            scenario=scenario,
            human_state=human_state,
            action_options=context.action_options,
            model_allocations_on_model_state=model_allocation_maps,
            model_planned_costs_on_model_state=model_planned_costs,
        )
        human_allocation = human_policy(turn_context)
        validate_allocation(
            human_allocation,
            state=human_state,
            planned_operating_cost=context.planned_operating_cost,
        )

        outcomes: dict[str, PortfolioActorStepOutcome] = {}
        for actor_name, state in actor_states.items():
            if not state.alive:
                continue
            allocation = human_allocation if actor_name == "you" else model_allocations.get(actor_name)
            if allocation is None:
                continue
            step = context.simulator.step(
                state=state,
                allocation=allocation,
                scenario=scenario,
            )
            actor_states[actor_name] = step.ending_state
            outcomes[actor_name] = PortfolioActorStepOutcome(
                allocation=allocation_to_map(allocation),
                planned_cost=allocation_total_cost(
                    allocation,
                    planned_operating_cost=context.planned_operating_cost,
                ),
                net_income=float(step.net_income),
                debt_payment=float(step.debt_payment),
                dscr=float(step.dscr),
                ending_cash=float(step.ending_state.cash),
                ending_debt=float(step.ending_state.debt),
                ending_wealth=float(step.ending_state.cash - step.ending_state.debt),
                cumulative_profit=float(step.ending_state.cumulative_profit),
                alive=bool(step.ending_state.alive),
                components=tuple(
                    {
                        "action": f"{component.action_crop}_{component.action_input_level}",
                        "acres": float(component.acres),
                        "realized_yield_per_acre": float(component.realized_yield_per_acre),
                        "realized_price": float(component.realized_price),
                        "gross_revenue": float(component.gross_revenue),
                        "operating_cost": float(component.operating_cost),
                    }
                    for component in step.components
                ),
            )

        allocation_l1_distance_vs_models: dict[str, float] = {}
        for policy_name, model_allocation in model_allocations.items():
            distance = allocation_l1_distance(
                human_allocation,
                model_allocation,
                total_acres=max(float(human_state.acres), 1.0),
            )
            allocation_l1_distance_vs_models[policy_name] = distance
            per_model_distances[policy_name].append(distance)

        record = PortfolioTurnRecord(
            turn_index=turn_index,
            scenario_year=int(scenario.year_index),
            weather_regime=str(scenario.weather_regime),
            human_allocation=allocation_to_map(human_allocation),
            model_allocations_on_model_state=model_allocation_maps,
            allocation_l1_distance_vs_models=allocation_l1_distance_vs_models,
            outcomes=outcomes,
        )
        turns.append(record)
        if turn_callback is not None:
            turn_callback(turn_context, record)

        if stop_when_human_bankrupt and not actor_states["you"].alive:
            break

    mean_allocation_l1_distance = {
        policy_name: (
            sum(values) / len(values)
            if values
            else 0.0
        )
        for policy_name, values in per_model_distances.items()
    }
    final_states = {
        actor_name: state_snapshot(state)
        for actor_name, state in actor_states.items()
    }
    return PortfolioGameSummary(
        benchmark_name=context.config.benchmark_name,
        portfolio_benchmark_name=context.portfolio_benchmark_name,
        target=context.config.target,
        trial_index=context.trial_index,
        path_index=context.path_index,
        scenario_seed=context.scenario_seed,
        model_policies=context.policy_names,
        turns_available=len(context.scenario_path),
        turns_played=len(turns),
        human_survived_full_horizon=bool(actor_states["you"].alive) and len(turns) == len(context.scenario_path),
        mean_allocation_l1_distance=mean_allocation_l1_distance,
        final_states=final_states,
        turns=turns,
    )


def write_portfolio_game_summary(summary: PortfolioGameSummary, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, indent=2, sort_keys=True)
    return output
