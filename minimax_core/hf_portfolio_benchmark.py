from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minimax_hf import MinimaxHFConfig, MinimaxTrainer, build_synthetic_mnar_view
from .mnar import SyntheticMNARConfig


@dataclass(frozen=True)
class HFPortfolioBenchmarkConfig:
    benchmark_name: str = "georgia_diversified_portfolio"
    seed: int = 13
    training_seed: int = 101
    train_paths: int = 12
    test_paths: int = 30
    horizon_years: int = 100
    workspace_root: str = "dssat_runs/hf_portfolio_benchmark"
    output_dir: str = "outputs/hf_portfolio_trainer"
    initial_cash: float = 500_000.0
    initial_debt: float = 0.0
    initial_credit_limit: float = 300_000.0
    acres: float = 200.0
    land_value_per_acre: float = 4_000.0
    land_financed_fraction: float = 0.5
    land_mortgage_rate: float = 0.045
    land_mortgage_years: int = 30
    land_mortgage_grace_years: int = 2
    learning_rate: float = 1.0e-3
    hidden_dim: int = 64
    num_train_epochs: int = 16
    train_batch_size: int = 64
    random_exploration_policies: int = 4
    top_action_count: int = 4
    candidate_random_samples: int = 18
    max_active_actions: int = 3
    bankruptcy_penalty_per_acre: float = 15_000.0
    view_mode: str = "drop_unobserved"
    base_observation_probability: float = 0.95
    distressed_penalty: float = 0.60
    drought_penalty: float = 0.10
    exit_penalty: float = 0.15
    weight_decay: float = 1.0e-4


@dataclass(frozen=True)
class HFPortfolioBenchmarkResult:
    config: HFPortfolioBenchmarkConfig
    train_examples: int
    observation_rate: float
    stable_observation_rate: float
    distressed_observation_rate: float
    policy_metrics: dict[str, Any]
    training_loss: float | None
    learned_policy_name: str


class _TabularCollator:
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        return {
            "features": torch.tensor([feature["features"] for feature in features], dtype=torch.float32),
            "labels": torch.tensor([feature["labels"] for feature in features], dtype=torch.float32),
            "time_index": torch.tensor([feature["time_index"] for feature in features], dtype=torch.long),
            "history_score": torch.tensor([feature["history_score"] for feature in features], dtype=torch.float32),
        }


def _require_ag_survival_sim() -> dict[str, Any]:
    try:
        from ag_survival_sim import (
            FarmState,
            PortfolioFarmSimulator,
            ScenarioGenerator,
            build_portfolio_benchmark_crop_model,
            build_portfolio_demo_policies,
            evaluate_portfolio_policies,
            get_portfolio_benchmark_definition,
        )
        from ag_survival_sim.portfolio_learning import (
            LearnedPortfolioConfig,
            PortfolioCandidateGenerator,
            RandomPortfolioPolicy,
            _featurize_state_allocation,
            _targets_from_steps,
            build_learning_exploration_policies,
        )
    except ImportError as error:
        raise ImportError(
            "HF portfolio benchmark requires ag-survival-sim to be installed."
        ) from error

    return {
        "FarmState": FarmState,
        "PortfolioFarmSimulator": PortfolioFarmSimulator,
        "ScenarioGenerator": ScenarioGenerator,
        "build_portfolio_benchmark_crop_model": build_portfolio_benchmark_crop_model,
        "build_portfolio_demo_policies": build_portfolio_demo_policies,
        "evaluate_portfolio_policies": evaluate_portfolio_policies,
        "get_portfolio_benchmark_definition": get_portfolio_benchmark_definition,
        "PortfolioCandidateGenerator": PortfolioCandidateGenerator,
        "RandomPortfolioPolicy": RandomPortfolioPolicy,
        "LearnedPortfolioConfig": LearnedPortfolioConfig,
        "_featurize_state_allocation": _featurize_state_allocation,
        "_targets_from_steps": _targets_from_steps,
        "build_learning_exploration_policies": build_learning_exploration_policies,
    }


def _build_initial_state(config: HFPortfolioBenchmarkConfig, *, FarmState: Any) -> Any:
    return FarmState.initial(
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


def _build_history_scores(steps: list[Any]) -> list[float]:
    history_scores: list[float] = []
    cumulative_distress = 0.0
    for step in steps:
        history_scores.append(cumulative_distress)
        if step.net_income < 0.0 or step.dscr < 1.0 or step.weather_regime == "drought":
            cumulative_distress += 1.0
    return history_scores


def _build_training_rows(config: HFPortfolioBenchmarkConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ag = _require_ag_survival_sim()
    FarmState = ag["FarmState"]
    ScenarioGenerator = ag["ScenarioGenerator"]
    PortfolioFarmSimulator = ag["PortfolioFarmSimulator"]
    build_portfolio_benchmark_crop_model = ag["build_portfolio_benchmark_crop_model"]
    get_portfolio_benchmark_definition = ag["get_portfolio_benchmark_definition"]
    PortfolioCandidateGenerator = ag["PortfolioCandidateGenerator"]
    RandomPortfolioPolicy = ag["RandomPortfolioPolicy"]
    _featurize_state_allocation = ag["_featurize_state_allocation"]
    _targets_from_steps = ag["_targets_from_steps"]
    build_learning_exploration_policies = ag["build_learning_exploration_policies"]

    benchmark = get_portfolio_benchmark_definition(config.benchmark_name)
    actions = tuple(option.action for option in benchmark.options)
    crop_model = build_portfolio_benchmark_crop_model(
        config.benchmark_name,
        workspace_root=str(Path(config.workspace_root) / "train"),
    )
    simulator = PortfolioFarmSimulator(crop_model=crop_model)
    initial_state = _build_initial_state(config, FarmState=FarmState)

    candidate_generator = PortfolioCandidateGenerator(
        actions=actions,
        crop_model=crop_model,
        top_action_count=config.top_action_count,
        random_samples=config.candidate_random_samples,
        max_active_actions=config.max_active_actions,
    )
    exploration_policies = build_learning_exploration_policies(
        actions=actions,
        crop_model=crop_model,
    )
    for random_index in range(config.random_exploration_policies):
        exploration_policies[f"random_{random_index}"] = RandomPortfolioPolicy(
            candidate_generator=candidate_generator,
            seed=config.training_seed + random_index + 100,
        )

    scenario_generator = ScenarioGenerator(seed=config.training_seed)
    rows: list[dict[str, Any]] = []
    path_offset = 0
    for policy in exploration_policies.values():
        for _ in range(config.train_paths):
            path = scenario_generator.generate_path(config.horizon_years, path_index=path_offset)
            path_offset += 1
            state = initial_state
            steps: list[Any] = []
            states: list[Any] = []
            scenarios: list[Any] = []
            allocations: list[Any] = []
            for scenario in path:
                if not state.alive:
                    break
                allocation = policy.choose_allocation(state, scenario)
                record = simulator.step(state=state, allocation=allocation, scenario=scenario)
                steps.append(record)
                states.append(state)
                scenarios.append(scenario)
                allocations.append(allocation)
                state = record.ending_state
            if not steps:
                continue

            targets = _targets_from_steps(
                steps,
                acres=max(config.acres, 1.0),
                bankruptcy_penalty_per_acre=config.bankruptcy_penalty_per_acre,
            )
            history_scores = _build_history_scores(steps)
            for step, state_row, scenario_row, allocation_row, target, history_score in zip(
                steps,
                states,
                scenarios,
                allocations,
                targets,
                history_scores,
            ):
                group_id = (
                    "distressed"
                    if step.net_income < 0.0 or step.dscr < 1.0 or step.weather_regime == "drought"
                    else "stable"
                )
                rows.append(
                    {
                        "features": _featurize_state_allocation(
                            state_row,
                            scenario_row,
                            allocation_row,
                            actions,
                            crop_model,
                            config.horizon_years,
                        ),
                        "labels": float(target / 1_000.0),
                        "group_id": group_id,
                        "path_index": int(scenario_row.year_index + path_offset * 10_000),
                        "step_index": int(scenario_row.year_index),
                        "time_index": int(scenario_row.year_index),
                        "history_score": float(history_score),
                        "weather_regime": str(step.weather_regime),
                        "farm_alive_next_year": bool(step.ending_state.alive),
                    }
                )
    return rows, {
        "actions": actions,
        "crop_model": crop_model,
        "initial_state": initial_state,
        "candidate_generator": candidate_generator,
        "benchmark": benchmark,
        "exploration_policies": exploration_policies,
    }


class _TabularRegressor:
    def __new__(cls, input_dim: int, hidden_dim: int) -> Any:
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, features: Any = None, labels: Any = None) -> dict[str, Any]:
                del labels
                if features is None:
                    raise ValueError("features are required.")
                logits = self.network(features)
                return {"logits": logits}

        return Model()


@dataclass(frozen=True)
class _HFPortfolioPolicy:
    model: Any
    actions: tuple[Any, ...]
    crop_model: Any
    candidate_generator: Any
    horizon_years: int
    seed_policies: tuple[Any, ...]
    seed: int

    def choose_allocation(self, state: Any, scenario: Any) -> Any:
        import torch

        seed_allocations = tuple(policy.choose_allocation(state, scenario) for policy in self.seed_policies)
        rng = random.Random(hash((self.seed, state.year, round(state.cash, 2), round(state.debt, 2), scenario.year_index)))
        candidates = self.candidate_generator.generate(
            state,
            scenario,
            rng=rng,
            seed_allocations=seed_allocations,
        )
        if not candidates:
            from ag_survival_sim import PortfolioAllocation

            return PortfolioAllocation(())

        ag = _require_ag_survival_sim()
        _featurize_state_allocation = ag["_featurize_state_allocation"]
        feature_rows = [
            _featurize_state_allocation(
                state,
                scenario,
                candidate,
                self.actions,
                self.crop_model,
                self.horizon_years,
            )
            for candidate in candidates
        ]
        with torch.no_grad():
            outputs = self.model(features=torch.tensor(feature_rows, dtype=torch.float32))
            logits = outputs["logits"].squeeze(-1).cpu().tolist()
        best_index = max(range(len(candidates)), key=lambda index: float(logits[index]))
        return candidates[best_index]


def run_hf_portfolio_benchmark(
    config: HFPortfolioBenchmarkConfig,
) -> HFPortfolioBenchmarkResult:
    import torch
    from datasets import Dataset
    from transformers import TrainingArguments

    rows, context = _build_training_rows(config)
    view = build_synthetic_mnar_view(
        rows,
        config=SyntheticMNARConfig(
            seed=config.training_seed + 17,
            view_mode=config.view_mode,
            base_observation_probability=config.base_observation_probability,
            distressed_penalty=config.distressed_penalty,
            drought_penalty=config.drought_penalty,
            exit_penalty=config.exit_penalty,
        ),
        path_key="path_index",
        step_key="step_index",
    )
    dataset = Dataset.from_list(view.rows)

    input_dim = len(view.rows[0]["features"])
    model = _TabularRegressor(input_dim=input_dim, hidden_dim=config.hidden_dim)
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        remove_unused_columns=False,
        per_device_train_batch_size=config.train_batch_size,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        logging_strategy="epoch",
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
    )
    trainer = MinimaxTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=_TabularCollator(),
        minimax_config=MinimaxHFConfig(
            task_type="regression",
            group_key="group_id",
            observed_key="label_observed",
            time_key="time_index",
            history_key="history_score",
            uncertainty_mode="knightian",
            online_mnar=True,
            assumed_observation_rate=view.result.observation_rate,
        ),
    )
    train_output = trainer.train()

    ag = _require_ag_survival_sim()
    build_portfolio_demo_policies = ag["build_portfolio_demo_policies"]
    evaluate_portfolio_policies = ag["evaluate_portfolio_policies"]
    ScenarioGenerator = ag["ScenarioGenerator"]
    PortfolioFarmSimulator = ag["PortfolioFarmSimulator"]

    benchmark_policies = build_portfolio_demo_policies(
        config.benchmark_name,
        crop_model=context["crop_model"],
        include_learned_policy=True,
        initial_state=context["initial_state"],
        learning_config=ag["LearnedPortfolioConfig"](
            horizon_years=config.horizon_years,
            training_paths=config.train_paths,
            training_seed=config.training_seed,
            epochs=240,
            bankruptcy_penalty_per_acre=config.bankruptcy_penalty_per_acre,
        ),
    )
    benchmark_policies["hf_knightian"] = _HFPortfolioPolicy(
        model=model.eval(),
        actions=context["actions"],
        crop_model=context["crop_model"],
        candidate_generator=context["candidate_generator"],
        horizon_years=config.horizon_years,
        seed_policies=tuple(
            context["exploration_policies"][name]
            for name in ("greedy_margin", "christensen_knightian")
            if name in context["exploration_policies"]
        ),
        seed=config.seed,
    )
    summary = evaluate_portfolio_policies(
        simulator=PortfolioFarmSimulator(crop_model=context["crop_model"]),
        scenario_generator=ScenarioGenerator(seed=config.seed),
        policies=benchmark_policies,
        initial_state=context["initial_state"],
        horizon_years=config.horizon_years,
        num_paths=config.test_paths,
    )
    return HFPortfolioBenchmarkResult(
        config=config,
        train_examples=len(view.rows),
        observation_rate=view.result.observation_rate,
        stable_observation_rate=view.result.stable_observation_rate,
        distressed_observation_rate=view.result.distressed_observation_rate,
        policy_metrics=summary.metrics,
        training_loss=getattr(train_output, "training_loss", None),
        learned_policy_name="hf_knightian",
    )


def format_hf_portfolio_benchmark_result(result: HFPortfolioBenchmarkResult) -> str:
    lines = [
        "HF Knightian portfolio benchmark",
        f"benchmark: {result.config.benchmark_name}",
        f"train examples: {result.train_examples}",
        f"observation rate: {result.observation_rate:.3f}",
        f"stable observation rate: {result.stable_observation_rate:.3f}",
        f"distressed observation rate: {result.distressed_observation_rate:.3f}",
        f"training loss: {result.training_loss:.6f}" if result.training_loss is not None else "training loss: unavailable",
        "",
        "policy                    mean_survival  full_horizon  bankruptcy  mean_terminal_wealth  mean_cum_profit",
        "------------------------  -------------  ------------  ----------  --------------------  ---------------",
    ]
    for policy_name, metrics in sorted(result.policy_metrics.items()):
        lines.append(
            f"{policy_name:<24}"
            f"  {metrics.mean_survival_years:>13.2f}"
            f"  {metrics.full_horizon_survival_rate:>12.2%}"
            f"  {metrics.bankruptcy_rate:>10.2%}"
            f"  {metrics.mean_terminal_wealth:>20.2f}"
            f"  {metrics.mean_cumulative_profit:>15.2f}"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> HFPortfolioBenchmarkConfig:
    parser = argparse.ArgumentParser(
        description="Train a Hugging Face-compatible Knightian regressor on portfolio data and re-run the DSSAT sim."
    )
    parser.add_argument("--benchmark", default=HFPortfolioBenchmarkConfig.benchmark_name)
    parser.add_argument("--seed", type=int, default=HFPortfolioBenchmarkConfig.seed)
    parser.add_argument("--training-seed", type=int, default=HFPortfolioBenchmarkConfig.training_seed)
    parser.add_argument("--train-paths", type=int, default=HFPortfolioBenchmarkConfig.train_paths)
    parser.add_argument("--test-paths", type=int, default=HFPortfolioBenchmarkConfig.test_paths)
    parser.add_argument("--horizon-years", type=int, default=HFPortfolioBenchmarkConfig.horizon_years)
    parser.add_argument("--workspace-root", type=str, default=HFPortfolioBenchmarkConfig.workspace_root)
    parser.add_argument("--output-dir", type=str, default=HFPortfolioBenchmarkConfig.output_dir)
    parser.add_argument("--cash", type=float, default=HFPortfolioBenchmarkConfig.initial_cash)
    parser.add_argument("--debt", type=float, default=HFPortfolioBenchmarkConfig.initial_debt)
    parser.add_argument("--credit-limit", type=float, default=HFPortfolioBenchmarkConfig.initial_credit_limit)
    parser.add_argument("--acres", type=float, default=HFPortfolioBenchmarkConfig.acres)
    parser.add_argument("--num-train-epochs", type=int, default=HFPortfolioBenchmarkConfig.num_train_epochs)
    parser.add_argument("--learning-rate", type=float, default=HFPortfolioBenchmarkConfig.learning_rate)
    parser.add_argument("--hidden-dim", type=int, default=HFPortfolioBenchmarkConfig.hidden_dim)
    parser.add_argument(
        "--view-mode",
        choices=["explicit_missing", "drop_unobserved", "truncate_after_unobserved"],
        default=HFPortfolioBenchmarkConfig.view_mode,
    )
    args = parser.parse_args(argv)
    return HFPortfolioBenchmarkConfig(
        benchmark_name=args.benchmark,
        seed=args.seed,
        training_seed=args.training_seed,
        train_paths=args.train_paths,
        test_paths=args.test_paths,
        horizon_years=args.horizon_years,
        workspace_root=args.workspace_root,
        output_dir=args.output_dir,
        initial_cash=args.cash,
        initial_debt=args.debt,
        initial_credit_limit=args.credit_limit,
        acres=args.acres,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        view_mode=args.view_mode,
    )


def main(argv: list[str] | None = None) -> None:
    result = run_hf_portfolio_benchmark(parse_args(argv))
    print(format_hf_portfolio_benchmark_result(result))


if __name__ == "__main__":
    main()
