from __future__ import annotations

import argparse
import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from minimax_core.ag_benchmark import (
    _add_common_ag_benchmark_args,
    _config_from_namespace,
)

from .game import (
    DEFAULT_MODEL_METHODS,
    ActorStepOutcome,
    GameContext,
    GameSummary,
    TurnContext,
    TurnRecord,
    action_name,
    affordable_actions,
    best_action_for_method,
    prepare_game_context,
    score_actions,
    state_snapshot,
    write_game_summary,
)

try:
    from flask import Flask, redirect, render_template_string, request, url_for
except ImportError:  # pragma: no cover - optional dependency.
    Flask = None  # type: ignore[assignment]
    redirect = None  # type: ignore[assignment]
    render_template_string = None  # type: ignore[assignment]
    request = None  # type: ignore[assignment]
    url_for = None  # type: ignore[assignment]


_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Minimax AG Game UI</title>
  <style>
    :root {
      --bg0: #f4f7ed;
      --bg1: #e0ebd3;
      --card: #fffffe;
      --ink: #172015;
      --ink-soft: #536055;
      --accent: #1f6b3b;
      --accent-2: #ad7a2f;
      --danger: #8f2d25;
      --muted: #d8e1d0;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(1200px 700px at 0% 0%, #ffffff 0%, var(--bg0) 50%, var(--bg1) 100%);
    }
    .wrap {
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }
    .hero {
      background: linear-gradient(120deg, #234d2f 0%, #3b6f4a 45%, #8ea96f 100%);
      color: #f4f8ef;
      border-radius: 18px;
      padding: 20px 24px;
      box-shadow: 0 10px 24px rgba(24, 42, 26, 0.18);
    }
    .hero h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: 0.2px; }
    .hero p { margin: 4px 0; opacity: 0.93; }
    .row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-top: 16px;
    }
    .card {
      background: var(--card);
      border: 1px solid rgba(23, 32, 21, 0.08);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 4px 14px rgba(24, 42, 26, 0.08);
    }
    .label { font-size: 12px; color: var(--ink-soft); text-transform: uppercase; letter-spacing: 0.7px; }
    .value { font-size: 22px; font-weight: 700; margin-top: 4px; }
    .panel {
      margin-top: 16px;
      background: var(--card);
      border: 1px solid rgba(23, 32, 21, 0.08);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 4px 14px rgba(24, 42, 26, 0.08);
    }
    h2 {
      margin: 0 0 12px;
      font-size: 18px;
      color: #1a2b1d;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      border-bottom: 1px solid #e4eadf;
      padding: 8px 6px;
      text-align: left;
      vertical-align: top;
    }
    th { color: var(--ink-soft); font-weight: 600; }
    .pill {
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      background: var(--muted);
      color: #2d432f;
    }
    .pill.bad { background: #f4d9d4; color: var(--danger); }
    .btns {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    button {
      border: none;
      border-radius: 10px;
      padding: 9px 12px;
      font-weight: 600;
      cursor: pointer;
      background: var(--accent);
      color: #eff8ef;
    }
    button:hover { filter: brightness(1.06); }
    button.secondary { background: #5f6e63; }
    button.warn { background: var(--accent-2); }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.45;
    }
    .notice {
      margin-top: 12px;
      padding: 10px 12px;
      border-radius: 10px;
      background: #edf3e8;
      color: #243126;
      border: 1px solid #d8e4d3;
    }
    .error {
      background: #f7e6e4;
      border-color: #ebc9c5;
      color: #6f231d;
    }
    .kicker {
      margin-top: 14px;
      color: var(--ink-soft);
      font-size: 13px;
    }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
    .spaced { margin-top: 10px; }
    .flex {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .meta {
      color: var(--ink-soft);
      font-size: 13px;
    }
    @media (max-width: 980px) {
      .row { grid-template-columns: 1fr; }
      .wrap { padding: 14px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>Minimax AG Game UI</h1>
      <p>Benchmark <strong>{{ benchmark_name }}</strong> • Trial {{ trial_index }} • Path {{ path_index }}</p>
      <p>Models: {{ method_names|join(", ") }}</p>
      <p class="meta">Turn {{ turn_number }} / {{ turns_available }} • Scenario year {{ scenario_year }} • Weather {{ weather_regime }}</p>
    </div>

    {% if message %}
      <div class="notice">{{ message }}</div>
    {% endif %}
    {% if error %}
      <div class="notice error">{{ error }}</div>
    {% endif %}

    {% if finished %}
      <div class="panel">
        <h2>Game Complete</h2>
        <p class="kicker">Turns played: {{ summary.turns_played }} / {{ summary.turns_available }}</p>
        <p class="kicker">Trace written to <span class="mono">{{ trace_output }}</span></p>
        <table>
          <thead><tr><th>Actor</th><th>Cash</th><th>Debt</th><th>Wealth</th><th>Alive</th></tr></thead>
          <tbody>
            {% for actor_name, snap in summary.final_states.items() %}
              <tr>
                <td>{{ actor_name }}</td>
                <td>{{ "{:,.0f}".format(snap["cash"]) }}</td>
                <td>{{ "{:,.0f}".format(snap["debt"]) }}</td>
                <td>{{ "{:,.0f}".format(snap["cash"] - snap["debt"]) }}</td>
                <td>{{ "yes" if snap["alive"] else "no" }}</td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
        <div class="spaced btns">
          <form method="post" action="{{ url_for('reset_game') }}">
            <button type="submit" class="warn">Reset Game</button>
          </form>
        </div>
      </div>
    {% else %}
      <div class="row">
        <div class="card"><div class="label">Cash</div><div class="value">{{ "{:,.0f}".format(human_state.cash) }}</div></div>
        <div class="card"><div class="label">Debt</div><div class="value">{{ "{:,.0f}".format(human_state.debt) }}</div></div>
        <div class="card"><div class="label">Remaining Credit</div><div class="value">{{ "{:,.0f}".format(human_state.remaining_credit) }}</div></div>
      </div>
      <div class="row">
        <div class="card"><div class="label">Current Wealth</div><div class="value">{{ "{:,.0f}".format(human_state.cash - human_state.debt) }}</div></div>
        <div class="card"><div class="label">Cumulative Profit</div><div class="value">{{ "{:,.0f}".format(human_state.cumulative_profit) }}</div></div>
        <div class="card"><div class="label">Alive</div><div class="value">{{ "yes" if human_state.alive else "no" }}</div></div>
      </div>

      <div class="panel">
        <h2>Choose Your Action</h2>
        <div class="btns">
          {% for row in action_rows %}
            <form method="post" action="{{ url_for('play_turn') }}">
              <input type="hidden" name="action_name" value="{{ row.name }}" />
              <input type="hidden" name="lookback" value="{{ lookback }}" />
              <button type="submit" {% if not row.feasible %}disabled{% endif %}>{{ row.name }}</button>
            </form>
          {% endfor %}
          <form method="post" action="{{ url_for('reset_game') }}">
            <button type="submit" class="secondary">Reset Game</button>
          </form>
        </div>
      </div>

      <div class="panel">
        <h2>Decision Intelligence</h2>
        <table>
          <thead>
            <tr>
              <th>Action</th>
              <th>Planned Cost (Total)</th>
              <th>Current Price</th>
              <th>Model Price Estimate</th>
              <th>History Tail ({{ lookback }})</th>
              <th>Feasible</th>
            </tr>
          </thead>
          <tbody>
            {% for row in action_rows %}
              <tr>
                <td>{{ row.name }}</td>
                <td>{{ "{:,.0f}".format(row.planned_cost_total) }}</td>
                <td>{{ "{:.3f}".format(row.spot_price) }}</td>
                <td>{{ "{:.3f}".format(row.estimated_price) }}</td>
                <td class="mono">{{ row.history_tail }}</td>
                <td>
                  {% if row.feasible %}
                    <span class="pill">yes</span>
                  {% else %}
                    <span class="pill bad">no</span>
                  {% endif %}
                </td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
        <p class="kicker">Decision estimate is what policies use at time t: current spot price blended with historical lags (t-1, t-2, ...).</p>
      </div>

      <div class="panel">
        <div class="flex">
          <h2 style="margin:0;">Crop Price Lookback</h2>
          <form method="get" action="{{ url_for('home') }}" class="flex">
            <label for="lookback" class="meta">Window</label>
            <input id="lookback" name="lookback" type="number" min="1" max="300" value="{{ lookback }}" />
            <button type="submit" class="secondary">Update</button>
          </form>
        </div>
        <table>
          <thead><tr><th>Crop</th><th>Actions Included</th><th>Latest Price</th><th>History Tail ({{ lookback }})</th></tr></thead>
          <tbody>
            {% for row in crop_rows %}
              <tr>
                <td>{{ row.crop }}</td>
                <td>{{ row.actions }}</td>
                <td>{{ "{:.3f}".format(row.latest_price) }}</td>
                <td class="mono">{{ row.history_tail }}</td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>

      <div class="panel">
        <h2>Model Choices This Turn</h2>
        <table>
          <thead><tr><th>Model</th><th>Action (on model state)</th><th>Match Count</th></tr></thead>
          <tbody>
            {% for method_name in method_names %}
              <tr>
                <td>{{ method_name }}</td>
                <td>{{ model_actions.get(method_name, "(inactive)") }}</td>
                <td>{{ decision_match_counts.get(method_name, 0) }}</td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    {% endif %}
  </div>
</body>
</html>"""


@dataclass
class _WebGameState:
    context: GameContext
    trace_output: str
    actor_states: dict[str, Any] = field(default_factory=dict)
    turns: list[TurnRecord] = field(default_factory=list)
    decision_match_counts: dict[str, int] = field(default_factory=dict)
    decision_turn_counts: dict[str, int] = field(default_factory=dict)
    market_price_history_by_action: dict[tuple[str, str], list[float]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Minimax AG game in a browser UI."
    )
    _add_common_ag_benchmark_args(parser)
    parser.add_argument("--trial-index", type=int, default=0)
    parser.add_argument("--path-index", type=int, default=0)
    parser.add_argument(
        "--method",
        dest="methods",
        action="append",
        default=None,
        help=(
            "Model method to compare against. Repeat for multiple methods. "
            f"Default: {', '.join(DEFAULT_MODEL_METHODS)}"
        ),
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--lookback-default", type=int, default=24)
    parser.add_argument("--trace-output", type=str, default="outputs/ag_game_trace_ui.json")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def _empty_game_state(*, context: GameContext, trace_output: str) -> _WebGameState:
    state = _WebGameState(context=context, trace_output=trace_output)
    _reset_game_state(state)
    return state


def _reset_game_state(state: _WebGameState) -> None:
    state.actor_states = {
        "you": state.context.initial_state,
        **{method_name: state.context.initial_state for method_name in state.context.method_names},
    }
    state.turns = []
    state.decision_match_counts = {method_name: 0 for method_name in state.context.method_names}
    state.decision_turn_counts = {method_name: 0 for method_name in state.context.method_names}
    state.market_price_history_by_action = {
        (str(action.crop), str(action.input_level)): list(
            state.context.initial_price_history_by_action.get((str(action.crop), str(action.input_level)), [])
        )
        for action in state.context.actions
    }


def _find_action(actions: Sequence[Any], action_key: str) -> Any:
    for action in actions:
        if action_name(action) == action_key:
            return action
    raise KeyError(f"unknown action: {action_key}")


def _compute_model_actions(
    *,
    state: _WebGameState,
    scenario: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    action_objects: dict[str, Any] = {}
    action_names: dict[str, str] = {}
    for method_name in state.context.method_names:
        model_state = state.actor_states[method_name]
        if not model_state.alive:
            continue
        model_feasible = affordable_actions(
            actions=state.context.actions,
            state=model_state,
            planned_operating_cost=state.context.planned_operating_cost,
        )
        method_scores = score_actions(
            parameters=state.context.method_parameters[method_name],
            state=model_state,
            actions=state.context.actions,
            action_index_by_key=state.context.action_index_by_key,
            scenario=scenario,
            price_history_by_action=state.market_price_history_by_action,
            include_price_features=state.context.include_price_features,
            price_history_lags=state.context.price_history_lags,
            price_dynamics=state.context.price_dynamics,
            realized_price_fn=state.context.realized_price_fn,
            action_base_price_by_key=state.context.action_base_price_by_key,
        )
        chosen_name = best_action_for_method(
            method_scores=method_scores,
            candidate_actions=model_feasible,
        )
        chosen_action = _find_action(model_feasible, chosen_name)
        action_objects[method_name] = chosen_action
        action_names[method_name] = chosen_name
    return action_objects, action_names


def _is_finished(state: _WebGameState) -> bool:
    if len(state.turns) >= len(state.context.scenario_path):
        return True
    return not bool(state.actor_states["you"].alive)


def _build_summary(state: _WebGameState) -> GameSummary:
    decision_match_rates = {
        method_name: (
            state.decision_match_counts[method_name] / state.decision_turn_counts[method_name]
            if state.decision_turn_counts[method_name]
            else 0.0
        )
        for method_name in state.context.method_names
    }
    final_states = {
        actor_name: state_snapshot(actor_state)
        for actor_name, actor_state in state.actor_states.items()
    }
    return GameSummary(
        benchmark_name=state.context.config.benchmark_name,
        target=state.context.config.target,
        trial_index=state.context.trial_index,
        path_index=state.context.path_index,
        scenario_seed=state.context.scenario_seed,
        model_methods=state.context.method_names,
        turns_available=len(state.context.scenario_path),
        turns_played=len(state.turns),
        human_survived_full_horizon=bool(state.actor_states["you"].alive)
        and len(state.turns) == len(state.context.scenario_path),
        decision_match_counts=dict(state.decision_match_counts),
        decision_match_rates=decision_match_rates,
        final_states=final_states,
        turns=list(state.turns),
    )


def _history_tail(history: Sequence[float], lookback: int) -> str:
    if not history:
        return "-"
    return ", ".join(f"{value:.3f}" for value in history[-lookback:])


def _crop_rows(
    *,
    context: GameContext,
    market_price_history_by_action: Mapping[tuple[str, str], Sequence[float]],
    lookback: int,
) -> list[dict[str, Any]]:
    by_crop_actions: dict[str, list[tuple[str, str]]] = {}
    for action in context.actions:
        action_key = (str(action.crop), str(action.input_level))
        by_crop_actions.setdefault(action_key[0], []).append(action_key)

    rows: list[dict[str, Any]] = []
    for crop, action_keys in sorted(by_crop_actions.items()):
        action_histories = [list(market_price_history_by_action.get(key, [])) for key in action_keys]
        max_len = max((len(values) for values in action_histories), default=0)
        crop_history: list[float] = []
        for index in range(max_len):
            slice_values = [
                values[index]
                for values in action_histories
                if index < len(values)
            ]
            if slice_values:
                crop_history.append(float(mean(slice_values)))
        latest = crop_history[-1] if crop_history else 0.0
        rows.append(
            {
                "crop": crop,
                "actions": ", ".join(f"{name}_{level}" for name, level in action_keys),
                "latest_price": latest,
                "history_tail": _history_tail(crop_history, lookback),
            }
        )
    return rows


def _current_view(state: _WebGameState, *, lookback: int) -> dict[str, Any]:
    turn_number = len(state.turns) + 1
    summary = _build_summary(state)
    if _is_finished(state):
        return {
            "finished": True,
            "summary": summary,
            "turn_number": min(turn_number, len(state.context.scenario_path)),
            "turns_available": len(state.context.scenario_path),
            "scenario_year": len(state.turns),
            "weather_regime": "-",
            "human_state": state.actor_states["you"],
            "action_rows": [],
            "crop_rows": _crop_rows(
                context=state.context,
                market_price_history_by_action=state.market_price_history_by_action,
                lookback=lookback,
            ),
            "model_actions": {},
        }

    scenario = state.context.scenario_path[len(state.turns)]
    human_state = state.actor_states["you"]
    feasible_action_keys = {
        action_name(action)
        for action in affordable_actions(
            actions=state.context.actions,
            state=human_state,
            planned_operating_cost=state.context.planned_operating_cost,
        )
    }
    model_action_objects, model_actions = _compute_model_actions(
        state=state,
        scenario=scenario,
    )
    del model_action_objects

    action_rows: list[dict[str, Any]] = []
    for action in state.context.actions:
        action_key = (str(action.crop), str(action.input_level))
        history = state.market_price_history_by_action.get(action_key, [])
        spot_price = (
            float(state.context.realized_price_fn(action, scenario))
            if state.context.realized_price_fn is not None
            else 0.0
        )
        estimated_price = spot_price
        if state.context.include_price_features:
            from minimax_core.price_dynamics import estimate_decision_price

            estimated_price = estimate_decision_price(
                history=history,
                spot_price=spot_price,
                config=state.context.price_dynamics,
            )
        action_rows.append(
            {
                "name": action_name(action),
                "planned_cost_total": float(state.context.planned_operating_cost(action, human_state.acres)),
                "spot_price": spot_price,
                "estimated_price": float(estimated_price),
                "history_tail": _history_tail(history, lookback),
                "feasible": action_name(action) in feasible_action_keys,
            }
        )

    return {
        "finished": False,
        "summary": summary,
        "turn_number": turn_number,
        "turns_available": len(state.context.scenario_path),
        "scenario_year": int(scenario.year_index),
        "weather_regime": str(scenario.weather_regime),
        "human_state": human_state,
        "action_rows": action_rows,
        "crop_rows": _crop_rows(
            context=state.context,
            market_price_history_by_action=state.market_price_history_by_action,
            lookback=lookback,
        ),
        "model_actions": model_actions,
    }


def create_app(
    *,
    context: GameContext,
    trace_output: str,
    lookback_default: int,
) -> Any:
    if Flask is None:
        raise ImportError(
            "Web UI requires Flask. Install with: pip install \"minimax-optimization[ag_game]\""
        )

    app = Flask(__name__)
    state = _empty_game_state(context=context, trace_output=trace_output)

    @app.get("/")
    def home() -> Any:
        lookback = max(int(request.args.get("lookback", lookback_default)), 1)
        with state.lock:
            view = _current_view(state, lookback=lookback)
            summary = view["summary"]
            if view["finished"]:
                write_game_summary(summary, state.trace_output)
            render_payload = {
                "benchmark_name": context.config.benchmark_name,
                "trial_index": context.trial_index,
                "path_index": context.path_index,
                "method_names": list(context.method_names),
                "message": request.args.get("message"),
                "error": request.args.get("error"),
                "trace_output": state.trace_output,
                "lookback": lookback,
                "decision_match_counts": dict(state.decision_match_counts),
                **view,
            }
        return render_template_string(_PAGE_TEMPLATE, **render_payload)

    @app.post("/play")
    def play_turn() -> Any:
        lookback = max(int(request.form.get("lookback", lookback_default)), 1)
        selected_action_name = str(request.form.get("action_name", "")).strip()
        with state.lock:
            if _is_finished(state):
                return redirect(url_for("home", lookback=lookback, message="Game already complete."))

            scenario = context.scenario_path[len(state.turns)]
            human_state = state.actor_states["you"]
            feasible_actions = affordable_actions(
                actions=context.actions,
                state=human_state,
                planned_operating_cost=context.planned_operating_cost,
            )
            feasible_lookup = {action_name(action): action for action in feasible_actions}
            if selected_action_name not in feasible_lookup:
                return redirect(
                    url_for(
                        "home",
                        lookback=lookback,
                        error="Selected action is not feasible for current cash/credit constraints.",
                    )
                )

            model_action_objects, model_actions_on_model_state = _compute_model_actions(
                state=state,
                scenario=scenario,
            )
            human_action = feasible_lookup[selected_action_name]

            outcomes: dict[str, ActorStepOutcome] = {}
            for actor_name, actor_state in list(state.actor_states.items()):
                if not actor_state.alive:
                    continue
                selected_action = human_action if actor_name == "you" else model_action_objects.get(actor_name)
                if selected_action is None:
                    continue
                step_record = context.simulator.step(
                    state=actor_state,
                    action=selected_action,
                    scenario=scenario,
                )
                state.actor_states[actor_name] = step_record.ending_state
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

            for method_name, model_action in model_actions_on_model_state.items():
                state.decision_turn_counts[method_name] += 1
                if model_action == selected_action_name:
                    state.decision_match_counts[method_name] += 1

            state.turns.append(
                TurnRecord(
                    turn_index=len(state.turns) + 1,
                    scenario_year=int(scenario.year_index),
                    weather_regime=str(scenario.weather_regime),
                    human_action=selected_action_name,
                    model_actions_on_model_state=model_actions_on_model_state,
                    outcomes=outcomes,
                )
            )

            if context.realized_price_fn is not None:
                for action in context.actions:
                    action_key = (str(action.crop), str(action.input_level))
                    history = state.market_price_history_by_action.setdefault(action_key, [])
                    history.append(float(context.realized_price_fn(action, scenario)))

            if _is_finished(state):
                summary = _build_summary(state)
                write_game_summary(summary, state.trace_output)
                return redirect(url_for("home", lookback=lookback, message="Game complete."))

        return redirect(url_for("home", lookback=lookback))

    @app.post("/reset")
    def reset_game() -> Any:
        with state.lock:
            _reset_game_state(state)
        return redirect(url_for("home", message="Game reset."))

    @app.get("/trace")
    def trace() -> Any:
        trace_path = Path(state.trace_output)
        with state.lock:
            summary = _build_summary(state)
            payload = asdict(summary)
            payload["trace_path"] = str(trace_path)
        return app.response_class(
            response=json.dumps(payload, indent=2),
            status=200,
            mimetype="application/json",
        )

    return app


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = _config_from_namespace(args)
    context = prepare_game_context(
        config,
        trial_index=args.trial_index,
        path_index=args.path_index,
        method_names=args.methods or DEFAULT_MODEL_METHODS,
    )
    app = create_app(
        context=context,
        trace_output=args.trace_output,
        lookback_default=args.lookback_default,
    )
    print(f"Starting AG Game UI at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
