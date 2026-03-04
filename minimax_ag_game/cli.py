from __future__ import annotations

import argparse
import math
from typing import Any, Mapping, Sequence

from minimax_core.ag_benchmark import _add_common_ag_benchmark_args, _config_from_namespace

from .game import (
    DEFAULT_MODEL_METHODS,
    GameSummary,
    TurnContext,
    TurnRecord,
    action_name,
    prepare_game_context,
    run_turn_based_game,
    write_game_summary,
)
from .portfolio_game import (
    DEFAULT_PORTFOLIO_POLICIES,
    PortfolioGameSummary,
    PortfolioTurnContext,
    PortfolioTurnRecord,
    available_portfolio_benchmark_names,
    build_allocation,
    prepare_portfolio_game_context,
    run_turn_based_portfolio_game,
    validate_allocation,
    write_portfolio_game_summary,
)

_REGIME_PARAMS = {
    "good": {
        "weather_yield_multiplier": 1.08,
        "market_price_multiplier": 0.95,
        "operating_cost_multiplier": 0.98,
        "basis_penalty": 0.02,
    },
    "normal": {
        "weather_yield_multiplier": 1.00,
        "market_price_multiplier": 1.00,
        "operating_cost_multiplier": 1.00,
        "basis_penalty": 0.04,
    },
    "drought": {
        "weather_yield_multiplier": 0.72,
        "market_price_multiplier": 1.15,
        "operating_cost_multiplier": 1.06,
        "basis_penalty": 0.08,
    },
}

try:
    from rich.box import SIMPLE_HEAVY
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback path.
    Console = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Prompt = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    SIMPLE_HEAVY = None  # type: ignore[assignment]
    _RICH_AVAILABLE = False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Play the agriculture simulator as a turn-based game and compare your decisions "
            "against model policy variants."
        )
    )
    _add_common_ag_benchmark_args(parser)
    parser.add_argument(
        "--mode",
        choices=["single", "portfolio"],
        default="single",
        help="Game mode: single action per turn or multi-crop portfolio allocation per turn.",
    )
    parser.add_argument("--trial-index", type=int, default=0)
    parser.add_argument("--path-index", type=int, default=0)
    parser.add_argument(
        "--method",
        dest="methods",
        action="append",
        default=None,
        help=(
            "Single-action mode only. Model method to compare against. Repeat for multiple methods. "
            f"Default: {', '.join(DEFAULT_MODEL_METHODS)}"
        ),
    )
    parser.add_argument(
        "--portfolio-benchmark",
        choices=available_portfolio_benchmark_names(),
        default="georgia_diversified_portfolio",
        help="Portfolio mode only. Portfolio benchmark containing multi-crop options.",
    )
    parser.add_argument(
        "--portfolio-policy",
        dest="portfolio_policies",
        action="append",
        default=None,
        help=(
            "Portfolio mode only. Model policy to compare against. Repeat for multiple policies. "
            f"Default: {', '.join(DEFAULT_PORTFOLIO_POLICIES)}"
        ),
    )
    parser.add_argument(
        "--include-learned-portfolio-policy",
        action="store_true",
        help="Portfolio mode only. Include the learned rollout policy in available portfolio policies.",
    )
    parser.add_argument(
        "--allocation-step-acres",
        type=float,
        default=10.0,
        help="Portfolio mode only. Human acreage input granularity.",
    )
    parser.add_argument(
        "--trace-output",
        type=str,
        default="outputs/ag_game_trace.json",
        help="Where to write the full turn-by-turn game trace as JSON.",
    )
    parser.add_argument(
        "--continue-after-bankruptcy",
        action="store_true",
        help="Keep simulating model actors after your farm fails.",
    )
    return parser.parse_args(argv)


def _print_header(
    console: Any,
    *,
    args: argparse.Namespace,
    config: Any,
    mode_label: str,
    model_label: str,
    model_names: Sequence[str],
) -> None:
    if _RICH_AVAILABLE:
        console.print(
            Panel.fit(
                (
                    f"[bold]Minimax AG Game ({mode_label})[/bold]\n"
                    f"benchmark={config.benchmark_name}  target={config.target}  mnar={config.mnar_mode}\n"
                    f"trial={args.trial_index}  path={args.path_index}  {model_label}={', '.join(model_names)}"
                ),
                border_style="cyan",
            )
        )
        return
    console.print(f"Minimax AG Game ({mode_label})")
    console.print(
        f"benchmark={config.benchmark_name}, target={config.target}, mnar={config.mnar_mode}, "
        f"trial={args.trial_index}, path={args.path_index}, {model_label}={', '.join(model_names)}"
    )


def _render_state(console: Any, turn: TurnContext | PortfolioTurnContext) -> None:
    state = turn.human_state
    wealth = state.cash - state.debt
    if _RICH_AVAILABLE:
        table = Table(title=f"Turn {turn.turn_index}: Year {turn.scenario_year} ({turn.weather_regime})", box=SIMPLE_HEAVY)
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Cash", f"{state.cash:,.0f}")
        table.add_row("Debt", f"{state.debt:,.0f}")
        table.add_row("Remaining Credit", f"{state.remaining_credit:,.0f}")
        table.add_row("Terminal Wealth (Now)", f"{wealth:,.0f}")
        table.add_row("Cumulative Profit", f"{state.cumulative_profit:,.0f}")
        table.add_row("Alive", str(state.alive))
        console.print(table)
        return
    console.print(f"Turn {turn.turn_index} Year {turn.scenario_year} Weather={turn.weather_regime}")
    console.print(
        f"cash={state.cash:,.0f}, debt={state.debt:,.0f}, remaining_credit={state.remaining_credit:,.0f}, "
        f"wealth={wealth:,.0f}, cumulative_profit={state.cumulative_profit:,.0f}, alive={state.alive}"
    )


def _scenario_with_regime(base_scenario: Any, regime: str) -> Any:
    from ag_survival_sim.scenario import AnnualScenario  # type: ignore[import-not-found]

    params = _REGIME_PARAMS[regime]
    return AnnualScenario(
        year_index=int(base_scenario.year_index),
        weather_regime=regime,
        weather_yield_multiplier=float(params["weather_yield_multiplier"]),
        market_price_multiplier=float(params["market_price_multiplier"]),
        operating_cost_multiplier=float(params["operating_cost_multiplier"]),
        basis_penalty=float(params["basis_penalty"]),
    )


def _risk_label(*, mean_margin: float, worst_margin: float) -> str:
    if worst_margin < -250.0:
        return "high"
    if worst_margin < 0.0:
        return "medium"
    if mean_margin < 100.0:
        return "medium"
    return "low"


def _compute_action_decision_metrics(
    *,
    state: Any,
    scenario: Any,
    action: Any,
    crop_model: Any,
    planned_operating_cost: Any,
) -> dict[str, float | str | bool]:
    from ag_survival_sim.finance import operating_cost, realized_price  # type: ignore[import-not-found]

    planned_cost_per_acre = float(planned_operating_cost(action, 1.0))
    available_capital = float(state.cash + state.remaining_credit)
    max_affordable_acres = (
        (available_capital / planned_cost_per_acre)
        if planned_cost_per_acre > 0.0
        else float(state.acres)
    )
    current_yield = float(
        crop_model.yield_per_acre(
            state=state,
            action=action,
            scenario=scenario,
        )
    )
    current_price = float(realized_price(action, scenario))
    current_operating_cost = float(operating_cost(action, 1.0, scenario))
    current_margin = current_yield * current_price - current_operating_cost

    per_regime_margin: dict[str, float] = {}
    for regime in ("good", "normal", "drought"):
        regime_scenario = _scenario_with_regime(scenario, regime)
        regime_yield = float(
            crop_model.yield_per_acre(
                state=state,
                action=action,
                scenario=regime_scenario,
            )
        )
        regime_price = float(realized_price(action, regime_scenario))
        regime_cost = float(operating_cost(action, 1.0, regime_scenario))
        per_regime_margin[regime] = regime_yield * regime_price - regime_cost

    margins = tuple(per_regime_margin.values())
    worst_margin = min(margins)
    mean_margin = sum(margins) / len(margins)
    margin_range = max(margins) - min(margins)
    risk = _risk_label(mean_margin=mean_margin, worst_margin=worst_margin)
    feasible = planned_cost_per_acre <= available_capital + 1e-9

    return {
        "planned_cost_per_acre": planned_cost_per_acre,
        "current_price": current_price,
        "current_yield": current_yield,
        "current_margin": current_margin,
        "worst_margin": worst_margin,
        "mean_margin": mean_margin,
        "drought_margin": per_regime_margin["drought"],
        "margin_range": margin_range,
        "max_affordable_acres": max_affordable_acres,
        "risk_label": risk,
        "feasible_now": feasible,
    }


def _render_action_intelligence(
    console: Any,
    *,
    state: Any,
    scenario: Any,
    actions: Sequence[Any],
    crop_model: Any,
    planned_operating_cost: Any,
) -> None:
    metrics_by_action = {
        action_name(action): _compute_action_decision_metrics(
            state=state,
            scenario=scenario,
            action=action,
            crop_model=crop_model,
            planned_operating_cost=planned_operating_cost,
        )
        for action in actions
    }
    if _RICH_AVAILABLE:
        table = Table(title="Decision Intelligence (Per Acre Estimates)", box=SIMPLE_HEAVY)
        table.add_column("Action")
        table.add_column("Cost/ac", justify="right")
        table.add_column("Price", justify="right")
        table.add_column("Yield/ac", justify="right")
        table.add_column("Margin/ac", justify="right")
        table.add_column("Worst/ac", justify="right")
        table.add_column("Drought/ac", justify="right")
        table.add_column("Range/ac", justify="right")
        table.add_column("Max Acres", justify="right")
        table.add_column("Risk")
        for action in actions:
            key = action_name(action)
            metrics = metrics_by_action[key]
            feasible_tag = "" if bool(metrics["feasible_now"]) else " (cash-limited)"
            table.add_row(
                key + feasible_tag,
                f"{float(metrics['planned_cost_per_acre']):,.0f}",
                f"{float(metrics['current_price']):.2f}",
                f"{float(metrics['current_yield']):.1f}",
                f"{float(metrics['current_margin']):,.0f}",
                f"{float(metrics['worst_margin']):,.0f}",
                f"{float(metrics['drought_margin']):,.0f}",
                f"{float(metrics['margin_range']):,.0f}",
                f"{float(metrics['max_affordable_acres']):.1f}",
                str(metrics["risk_label"]),
            )
        console.print(table)
        console.print(
            "Risk columns are scenario-regime stress estimates: "
            "`Worst/ac` = worst of good/normal/drought, `Range/ac` = spread across regimes."
        )
        return

    console.print("Decision intelligence (per acre):")
    for action in actions:
        key = action_name(action)
        m = metrics_by_action[key]
        console.print(
            f"  {key}: margin={float(m['current_margin']):,.0f}, worst={float(m['worst_margin']):,.0f}, "
            f"drought={float(m['drought_margin']):,.0f}, risk={m['risk_label']}"
        )


def _render_single_model_views(console: Any, turn: TurnContext) -> None:
    if _RICH_AVAILABLE:
        score_table = Table(title="Model Scores On Your Current State", box=SIMPLE_HEAVY)
        score_table.add_column("Action")
        for method_name in turn.model_scores_on_human_state:
            score_table.add_column(method_name, justify="right")
        for action in turn.all_actions:
            key = action_name(action)
            score_table.add_row(
                key,
                *[
                    f"{turn.model_scores_on_human_state[method_name].get(key, float('-inf')):.4f}"
                    for method_name in turn.model_scores_on_human_state
                ],
            )
        console.print(score_table)

        action_table = Table(title="Model Choices This Turn (Using Their Own States)", box=SIMPLE_HEAVY)
        action_table.add_column("Method")
        action_table.add_column("Action")
        for method_name, chosen_action in turn.model_actions_on_model_state.items():
            action_table.add_row(method_name, chosen_action)
        if not turn.model_actions_on_model_state:
            action_table.add_row("(none)", "all model farms are inactive")
        console.print(action_table)
        return

    console.print("Model choices this turn:")
    for method_name, chosen_action in turn.model_actions_on_model_state.items():
        console.print(f"  {method_name}: {chosen_action}")


def _prompt_human_action(console: Any, turn: TurnContext, planned_cost: Any) -> Any:
    if _RICH_AVAILABLE:
        choice_table = Table(title="Choose Your Action", box=SIMPLE_HEAVY)
        choice_table.add_column("#", justify="right")
        choice_table.add_column("Action")
        choice_table.add_column("Planned Cost", justify="right")
        for index, action in enumerate(turn.feasible_actions, start=1):
            choice_table.add_row(
                str(index),
                action_name(action),
                f"{planned_cost(action, turn.human_state.acres):,.0f}",
            )
        console.print(choice_table)
    else:
        console.print("Choose your action:")
        for index, action in enumerate(turn.feasible_actions, start=1):
            console.print(f"  {index}. {action_name(action)}")

    choices = [str(index) for index in range(1, len(turn.feasible_actions) + 1)]
    while True:
        if _RICH_AVAILABLE:
            selected = Prompt.ask("Action number", choices=choices, default="1")
        else:
            selected = input(f"Action number [{'/'.join(choices)}]: ").strip() or "1"
        if selected in choices:
            return turn.feasible_actions[int(selected) - 1]
        console.print(f"Invalid choice: {selected}")


def _render_single_turn_outcome(console: Any, record: TurnRecord) -> None:
    if _RICH_AVAILABLE:
        table = Table(title=f"Turn {record.turn_index} Outcomes", box=SIMPLE_HEAVY)
        table.add_column("Actor")
        table.add_column("Action")
        table.add_column("Net Income", justify="right")
        table.add_column("DSCR", justify="right")
        table.add_column("Ending Wealth", justify="right")
        table.add_column("Cum Profit", justify="right")
        table.add_column("Alive")
        for actor_name, outcome in record.outcomes.items():
            table.add_row(
                actor_name,
                outcome.action,
                f"{outcome.net_income:,.0f}",
                f"{outcome.dscr:.2f}",
                f"{outcome.ending_wealth:,.0f}",
                f"{outcome.cumulative_profit:,.0f}",
                str(outcome.alive),
            )
        console.print(table)
        return

    console.print(f"Turn {record.turn_index} outcomes:")
    for actor_name, outcome in record.outcomes.items():
        console.print(
            f"  {actor_name}: action={outcome.action}, net_income={outcome.net_income:,.0f}, "
            f"dscr={outcome.dscr:.2f}, wealth={outcome.ending_wealth:,.0f}, alive={outcome.alive}"
        )


def _render_single_final_summary(console: Any, summary: GameSummary) -> None:
    if _RICH_AVAILABLE:
        summary_table = Table(title="Final Summary", box=SIMPLE_HEAVY)
        summary_table.add_column("Actor")
        summary_table.add_column("Alive")
        summary_table.add_column("Cash", justify="right")
        summary_table.add_column("Debt", justify="right")
        summary_table.add_column("Wealth", justify="right")
        summary_table.add_column("Cum Profit", justify="right")
        for actor_name, state in summary.final_states.items():
            summary_table.add_row(
                actor_name,
                str(state["alive"]),
                f"{state['cash']:,.0f}",
                f"{state['debt']:,.0f}",
                f"{float(state['cash']) - float(state['debt']):,.0f}",
                f"{state['cumulative_profit']:,.0f}",
            )
        console.print(summary_table)

        match_table = Table(title="Decision Alignment", box=SIMPLE_HEAVY)
        match_table.add_column("Model")
        match_table.add_column("Match Count", justify="right")
        match_table.add_column("Match Rate", justify="right")
        for method_name in summary.model_methods:
            match_table.add_row(
                method_name,
                str(summary.decision_match_counts.get(method_name, 0)),
                f"{summary.decision_match_rates.get(method_name, 0.0):.2%}",
            )
        console.print(match_table)
        return

    console.print("Final summary:")
    for actor_name, state in summary.final_states.items():
        wealth = float(state["cash"]) - float(state["debt"])
        console.print(
            f"  {actor_name}: alive={state['alive']}, cash={state['cash']:,.0f}, "
            f"debt={state['debt']:,.0f}, wealth={wealth:,.0f}, cum_profit={state['cumulative_profit']:,.0f}"
        )
    console.print("Decision alignment:")
    for method_name in summary.model_methods:
        console.print(
            f"  {method_name}: {summary.decision_match_counts.get(method_name, 0)} "
            f"({summary.decision_match_rates.get(method_name, 0.0):.2%})"
        )


def _format_allocation(allocation: Mapping[str, float]) -> str:
    if not allocation:
        return "none"
    return ", ".join(f"{key}:{value:.1f}ac" for key, value in sorted(allocation.items()))


def _render_portfolio_model_views(console: Any, turn: PortfolioTurnContext) -> None:
    if _RICH_AVAILABLE:
        table = Table(title="Model Portfolio Choices This Turn (Using Their Own States)", box=SIMPLE_HEAVY)
        table.add_column("Policy")
        table.add_column("Planned Cost", justify="right")
        table.add_column("Allocation")
        for policy_name, allocation in turn.model_allocations_on_model_state.items():
            table.add_row(
                policy_name,
                f"{turn.model_planned_costs_on_model_state.get(policy_name, 0.0):,.0f}",
                _format_allocation(allocation),
            )
        if not turn.model_allocations_on_model_state:
            table.add_row("(none)", "0", "all model farms are inactive")
        console.print(table)
        return

    console.print("Model portfolio choices this turn:")
    for policy_name, allocation in turn.model_allocations_on_model_state.items():
        cost = turn.model_planned_costs_on_model_state.get(policy_name, 0.0)
        console.print(f"  {policy_name}: cost={cost:,.0f}, allocation={_format_allocation(allocation)}")


def _prompt_float_value(console: Any, prompt_label: str, *, default: float) -> float:
    while True:
        if _RICH_AVAILABLE:
            raw = Prompt.ask(prompt_label, default=f"{default:g}")
        else:
            raw = input(f"{prompt_label} [{default:g}]: ").strip() or f"{default:g}"
        try:
            return float(raw)
        except ValueError:
            console.print(f"Invalid numeric value: {raw}")


def _is_valid_step_multiple(value: float, step: float) -> bool:
    if step <= 0.0:
        return True
    return math.isclose(value / step, round(value / step), rel_tol=1e-9, abs_tol=1e-9)


def _prompt_portfolio_human_allocation(
    console: Any,
    turn: PortfolioTurnContext,
    *,
    game_context: Any,
    allocation_step_acres: float,
) -> Any:
    while True:
        remaining_acres = float(turn.human_state.acres)
        remaining_capital = float(turn.human_state.cash + turn.human_state.remaining_credit)
        acres_by_action_name: dict[str, float] = {}

        _render_action_intelligence(
            console,
            state=turn.human_state,
            scenario=turn.scenario,
            actions=turn.action_options,
            crop_model=game_context.simulator.crop_model,
            planned_operating_cost=game_context.planned_operating_cost,
        )

        if _RICH_AVAILABLE:
            table = Table(title="Choose Portfolio Allocation (Acres)", box=SIMPLE_HEAVY)
            table.add_column("Action")
            table.add_column("Planned Cost / Acre", justify="right")
            table.add_column("Current Max Acres", justify="right")
            for action in turn.action_options:
                cost_per_acre = float(game_context.planned_operating_cost(action, 1.0))
                capital_max = remaining_capital / cost_per_acre if cost_per_acre > 0.0 else remaining_acres
                max_acres = min(remaining_acres, capital_max)
                if allocation_step_acres > 0.0:
                    max_acres = math.floor(max_acres / allocation_step_acres) * allocation_step_acres
                table.add_row(
                    action_name(action),
                    f"{cost_per_acre:,.0f}",
                    f"{max(max_acres, 0.0):.1f}",
                )
            console.print(table)
            console.print(
                f"Input acres in steps of {allocation_step_acres:g}. "
                f"Remaining acres={remaining_acres:.1f}, remaining capital={remaining_capital:,.0f}."
            )
        else:
            console.print(
                f"Choose allocation in steps of {allocation_step_acres:g} acres "
                f"(remaining acres={remaining_acres:.1f}, remaining capital={remaining_capital:,.0f})."
            )

        valid = True
        for action in turn.action_options:
            key = action_name(action)
            cost_per_acre = float(game_context.planned_operating_cost(action, 1.0))
            capital_max = remaining_capital / cost_per_acre if cost_per_acre > 0.0 else remaining_acres
            max_acres = min(remaining_acres, capital_max)
            if allocation_step_acres > 0.0:
                max_acres = math.floor(max_acres / allocation_step_acres) * allocation_step_acres
                max_acres = max(max_acres, 0.0)

            if max_acres <= 0.0:
                acres_by_action_name[key] = 0.0
                continue

            label = f"{key} acres (0 to {max_acres:.1f})"
            acres = _prompt_float_value(console, label, default=0.0)
            if acres < 0.0 or acres > max_acres + 1e-9:
                console.print(f"Invalid acres for {key}. Must be within [0, {max_acres:.1f}]. Restarting entry.")
                valid = False
                break
            if allocation_step_acres > 0.0 and not _is_valid_step_multiple(acres, allocation_step_acres):
                console.print(
                    f"Invalid acres for {key}. Must be a multiple of {allocation_step_acres:g}. Restarting entry."
                )
                valid = False
                break
            acres_by_action_name[key] = acres
            remaining_acres -= acres
            remaining_capital -= cost_per_acre * acres

        if not valid:
            continue

        try:
            allocation = build_allocation(
                game_context,
                acres_by_action_name=acres_by_action_name,
            )
            validate_allocation(
                allocation,
                state=turn.human_state,
                planned_operating_cost=game_context.planned_operating_cost,
            )
        except ValueError as error:
            console.print(f"Allocation error: {error}. Restarting entry.")
            continue
        return allocation


def _render_portfolio_turn_outcome(console: Any, record: PortfolioTurnRecord) -> None:
    if _RICH_AVAILABLE:
        table = Table(title=f"Turn {record.turn_index} Portfolio Outcomes", box=SIMPLE_HEAVY)
        table.add_column("Actor")
        table.add_column("Planned Cost", justify="right")
        table.add_column("Net Income", justify="right")
        table.add_column("DSCR", justify="right")
        table.add_column("Ending Wealth", justify="right")
        table.add_column("Cum Profit", justify="right")
        table.add_column("Alive")
        table.add_column("Allocation")
        for actor_name, outcome in record.outcomes.items():
            table.add_row(
                actor_name,
                f"{outcome.planned_cost:,.0f}",
                f"{outcome.net_income:,.0f}",
                f"{outcome.dscr:.2f}",
                f"{outcome.ending_wealth:,.0f}",
                f"{outcome.cumulative_profit:,.0f}",
                str(outcome.alive),
                _format_allocation(outcome.allocation),
            )
        console.print(table)
        return

    console.print(f"Turn {record.turn_index} outcomes:")
    for actor_name, outcome in record.outcomes.items():
        console.print(
            f"  {actor_name}: cost={outcome.planned_cost:,.0f}, net_income={outcome.net_income:,.0f}, "
            f"wealth={outcome.ending_wealth:,.0f}, alive={outcome.alive}, allocation={_format_allocation(outcome.allocation)}"
        )


def _render_portfolio_final_summary(console: Any, summary: PortfolioGameSummary) -> None:
    if _RICH_AVAILABLE:
        summary_table = Table(title="Final Summary", box=SIMPLE_HEAVY)
        summary_table.add_column("Actor")
        summary_table.add_column("Alive")
        summary_table.add_column("Cash", justify="right")
        summary_table.add_column("Debt", justify="right")
        summary_table.add_column("Wealth", justify="right")
        summary_table.add_column("Cum Profit", justify="right")
        for actor_name, state in summary.final_states.items():
            summary_table.add_row(
                actor_name,
                str(state["alive"]),
                f"{state['cash']:,.0f}",
                f"{state['debt']:,.0f}",
                f"{float(state['cash']) - float(state['debt']):,.0f}",
                f"{state['cumulative_profit']:,.0f}",
            )
        console.print(summary_table)

        distance_table = Table(title="Allocation Distance Vs Your Choices (L1 Share Distance)", box=SIMPLE_HEAVY)
        distance_table.add_column("Policy")
        distance_table.add_column("Mean L1 Distance", justify="right")
        for policy_name in summary.model_policies:
            distance_table.add_row(
                policy_name,
                f"{summary.mean_allocation_l1_distance.get(policy_name, 0.0):.3f}",
            )
        console.print(distance_table)
        return

    console.print("Final summary:")
    for actor_name, state in summary.final_states.items():
        wealth = float(state["cash"]) - float(state["debt"])
        console.print(
            f"  {actor_name}: alive={state['alive']}, cash={state['cash']:,.0f}, "
            f"debt={state['debt']:,.0f}, wealth={wealth:,.0f}, cum_profit={state['cumulative_profit']:,.0f}"
        )
    console.print("Mean allocation L1 distance vs your choices:")
    for policy_name in summary.model_policies:
        console.print(f"  {policy_name}: {summary.mean_allocation_l1_distance.get(policy_name, 0.0):.3f}")


def _run_single_mode(console: Any, args: argparse.Namespace, config: Any) -> None:
    selected_methods = tuple(args.methods or DEFAULT_MODEL_METHODS)
    _print_header(
        console,
        args=args,
        config=config,
        mode_label="single",
        model_label="methods",
        model_names=selected_methods,
    )
    console.print("Preparing dataset and training selected model policies...")
    game_context = prepare_game_context(
        config,
        trial_index=args.trial_index,
        path_index=args.path_index,
        method_names=selected_methods,
    )
    console.print(
        f"Scenario seed={game_context.scenario_seed}; horizon={len(game_context.scenario_path)} years."
    )

    def _human_policy(turn: TurnContext) -> Any:
        _render_state(console, turn)
        _render_action_intelligence(
            console,
            state=turn.human_state,
            scenario=turn.scenario,
            actions=turn.all_actions,
            crop_model=game_context.simulator.crop_model,
            planned_operating_cost=game_context.planned_operating_cost,
        )
        _render_single_model_views(console, turn)
        return _prompt_human_action(console, turn, game_context.planned_operating_cost)

    def _turn_callback(_turn_context: TurnContext, record: TurnRecord) -> None:
        _render_single_turn_outcome(console, record)

    summary = run_turn_based_game(
        game_context,
        human_policy=_human_policy,
        turn_callback=_turn_callback,
        stop_when_human_bankrupt=not args.continue_after_bankruptcy,
    )
    _render_single_final_summary(console, summary)

    if args.trace_output:
        output = write_game_summary(summary, args.trace_output)
        console.print(f"Wrote trace: {output}")


def _run_portfolio_mode(console: Any, args: argparse.Namespace, config: Any) -> None:
    if args.allocation_step_acres <= 0.0:
        raise ValueError("--allocation-step-acres must be positive.")

    selected_policies = tuple(args.portfolio_policies or DEFAULT_PORTFOLIO_POLICIES)
    _print_header(
        console,
        args=args,
        config=config,
        mode_label="portfolio",
        model_label="policies",
        model_names=selected_policies,
    )
    console.print("Preparing portfolio simulator and selected model policies...")
    game_context = prepare_portfolio_game_context(
        config,
        portfolio_benchmark_name=args.portfolio_benchmark,
        trial_index=args.trial_index,
        path_index=args.path_index,
        policy_names=selected_policies,
        include_learned_policy=args.include_learned_portfolio_policy,
    )
    console.print(
        f"Portfolio benchmark={game_context.portfolio_benchmark_name}; "
        f"scenario seed={game_context.scenario_seed}; "
        f"horizon={len(game_context.scenario_path)} years."
    )
    console.print(
        "Action options: "
        + ", ".join(action_name(action) for action in game_context.action_options)
    )

    def _human_policy(turn: PortfolioTurnContext) -> Any:
        _render_state(console, turn)
        _render_portfolio_model_views(console, turn)
        return _prompt_portfolio_human_allocation(
            console,
            turn,
            game_context=game_context,
            allocation_step_acres=args.allocation_step_acres,
        )

    def _turn_callback(_turn_context: PortfolioTurnContext, record: PortfolioTurnRecord) -> None:
        _render_portfolio_turn_outcome(console, record)

    summary = run_turn_based_portfolio_game(
        game_context,
        human_policy=_human_policy,
        turn_callback=_turn_callback,
        stop_when_human_bankrupt=not args.continue_after_bankruptcy,
    )
    _render_portfolio_final_summary(console, summary)

    if args.trace_output:
        output = write_portfolio_game_summary(summary, args.trace_output)
        console.print(f"Wrote trace: {output}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = _config_from_namespace(args)
    console = Console() if _RICH_AVAILABLE else _FallbackConsole()
    if args.mode == "portfolio":
        _run_portfolio_mode(console, args, config)
        return
    _run_single_mode(console, args, config)


class _FallbackConsole:
    def print(self, message: Any) -> None:
        print(message)


if __name__ == "__main__":
    main()
