from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
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
    initial_cash: float = 250_000.0
    initial_debt: float = 0.0
    initial_credit_limit: float = 175_000.0
    acres: float = 200.0
    land_value_per_acre: float = 5_860.0
    land_financed_fraction: float = 0.2
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
    allocation_search_rounds: int = 6
    allocation_search_samples: int = 96
    allocation_search_elite_count: int = 12
    view_mode: str = "drop_unobserved"
    base_observation_probability: float = 0.95
    distressed_penalty: float = 0.60
    drought_penalty: float = 0.10
    exit_penalty: float = 0.15
    weight_decay: float = 1.0e-4
    uncertainty_mode: str = "knightian"


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


@dataclass(frozen=True)
class HFPortfolioMultiSeedPolicySummary:
    mean_survival_years: float
    survival_years_std: float
    mean_full_horizon_survival_rate: float
    full_horizon_survival_rate_std: float
    mean_bankruptcy_rate: float
    bankruptcy_rate_std: float
    mean_terminal_wealth: float
    terminal_wealth_std: float
    mean_cumulative_profit: float
    cumulative_profit_std: float


@dataclass(frozen=True)
class HFPortfolioMultiSeedResult:
    base_config: HFPortfolioBenchmarkConfig
    seeds: tuple[int, ...]
    seed_results: dict[int, HFPortfolioBenchmarkResult]
    policy_summaries: dict[str, HFPortfolioMultiSeedPolicySummary]


@dataclass(frozen=True)
class HFPortfolioSeedGridResult:
    base_config: HFPortfolioBenchmarkConfig
    training_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    grid_results: dict[tuple[int, int], HFPortfolioBenchmarkResult]
    policy_summaries: dict[str, HFPortfolioMultiSeedPolicySummary]


class _TabularCollator:
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        return {
            "features": torch.tensor([feature["features"] for feature in features], dtype=torch.float32),
            "labels": torch.tensor([feature["labels"] for feature in features], dtype=torch.float32),
            "time_index": torch.tensor([feature["time_index"] for feature in features], dtype=torch.long),
            "history_score": torch.tensor([feature["history_score"] for feature in features], dtype=torch.float32),
            "path_index": torch.tensor([feature["path_index"] for feature in features], dtype=torch.long),
        }


def _require_ag_survival_sim() -> dict[str, Any]:
    try:
        from ag_survival_sim import (
            ContinuousAllocationOptimizer,
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
        "ContinuousAllocationOptimizer": ContinuousAllocationOptimizer,
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
    ContinuousAllocationOptimizer = ag["ContinuousAllocationOptimizer"]
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
    allocation_optimizer = ContinuousAllocationOptimizer(
        actions=actions,
        max_share_per_action=1.0,
        max_share_per_crop=1.0,
        search_rounds=config.allocation_search_rounds,
        samples_per_round=config.allocation_search_samples,
        elite_count=config.allocation_search_elite_count,
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
        "allocation_optimizer": allocation_optimizer,
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
    allocation_optimizer: Any
    horizon_years: int
    seed_policies: tuple[Any, ...]
    seed: int

    def choose_allocation(self, state: Any, scenario: Any) -> Any:
        import torch

        seed_allocations = tuple(policy.choose_allocation(state, scenario) for policy in self.seed_policies)
        rng = random.Random(hash((self.seed, state.year, round(state.cash, 2), round(state.debt, 2), scenario.year_index)))
        ag = _require_ag_survival_sim()
        _featurize_state_allocation = ag["_featurize_state_allocation"]

        def score_candidate(candidate: Any) -> float:
            feature_row = _featurize_state_allocation(
                state,
                scenario,
                candidate,
                self.actions,
                self.crop_model,
                self.horizon_years,
            )
            with torch.no_grad():
                outputs = self.model(features=torch.tensor([feature_row], dtype=torch.float32))
            return float(outputs["logits"].squeeze().item())

        return self.allocation_optimizer.optimize(
            state,
            scenario,
            score_fn=score_candidate,
            rng=rng,
            seed_allocations=seed_allocations,
        )


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return pstdev(values)


def _train_hf_portfolio_model(
    config: HFPortfolioBenchmarkConfig,
) -> tuple[Any, dict[str, Any], Any, float | None]:
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
            path_key="path_index",
            uncertainty_mode=config.uncertainty_mode,
            online_mnar=True,
            assumed_observation_rate=view.result.observation_rate,
        ),
    )
    train_output = trainer.train()
    return model.eval(), context, view, getattr(train_output, "training_loss", None)


def _build_hf_benchmark_policies(
    config: HFPortfolioBenchmarkConfig,
    *,
    trained_model: Any,
    context: dict[str, Any],
) -> dict[str, Any]:
    ag = _require_ag_survival_sim()
    build_portfolio_demo_policies = ag["build_portfolio_demo_policies"]

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
    learned_policy_name = f"hf_{config.uncertainty_mode}"
    benchmark_policies[learned_policy_name] = _HFPortfolioPolicy(
        model=trained_model,
        actions=context["actions"],
        crop_model=context["crop_model"],
        allocation_optimizer=context["allocation_optimizer"],
        horizon_years=config.horizon_years,
        seed_policies=tuple(
            context["exploration_policies"][name]
            for name in ("greedy_margin", "christensen_knightian")
            if name in context["exploration_policies"]
        ),
        seed=config.seed,
    )
    return benchmark_policies


def _evaluate_hf_benchmark_policies(
    config: HFPortfolioBenchmarkConfig,
    *,
    context: dict[str, Any],
    benchmark_policies: dict[str, Any],
) -> dict[str, Any]:
    ag = _require_ag_survival_sim()
    evaluate_portfolio_policies = ag["evaluate_portfolio_policies"]
    ScenarioGenerator = ag["ScenarioGenerator"]
    PortfolioFarmSimulator = ag["PortfolioFarmSimulator"]

    summary = evaluate_portfolio_policies(
        simulator=PortfolioFarmSimulator(crop_model=context["crop_model"]),
        scenario_generator=ScenarioGenerator(seed=config.seed),
        policies=benchmark_policies,
        initial_state=context["initial_state"],
        horizon_years=config.horizon_years,
        num_paths=config.test_paths,
    )
    return summary.metrics


def run_hf_portfolio_benchmark(
    config: HFPortfolioBenchmarkConfig,
) -> HFPortfolioBenchmarkResult:
    trained_model, context, view, training_loss = _train_hf_portfolio_model(config)
    benchmark_policies = _build_hf_benchmark_policies(
        config,
        trained_model=trained_model,
        context=context,
    )
    policy_metrics = _evaluate_hf_benchmark_policies(
        config,
        context=context,
        benchmark_policies=benchmark_policies,
    )
    return HFPortfolioBenchmarkResult(
        config=config,
        train_examples=len(view.rows),
        observation_rate=view.result.observation_rate,
        stable_observation_rate=view.result.stable_observation_rate,
        distressed_observation_rate=view.result.distressed_observation_rate,
        policy_metrics=policy_metrics,
        training_loss=training_loss,
        learned_policy_name=f"hf_{config.uncertainty_mode}",
    )


def _aggregate_multiseed_policy_metrics(
    seed_results: dict[int, HFPortfolioBenchmarkResult],
) -> dict[str, HFPortfolioMultiSeedPolicySummary]:
    if not seed_results:
        raise ValueError("seed_results must not be empty.")

    policy_names = sorted(
        {
            policy_name
            for result in seed_results.values()
            for policy_name in result.policy_metrics
        }
    )
    summaries: dict[str, HFPortfolioMultiSeedPolicySummary] = {}
    for policy_name in policy_names:
        metrics_list = [
            result.policy_metrics[policy_name]
            for result in seed_results.values()
            if policy_name in result.policy_metrics
        ]
        mean_survival_years = [float(metrics.mean_survival_years) for metrics in metrics_list]
        full_horizon_survival_rate = [float(metrics.full_horizon_survival_rate) for metrics in metrics_list]
        bankruptcy_rate = [float(metrics.bankruptcy_rate) for metrics in metrics_list]
        mean_terminal_wealth = [float(metrics.mean_terminal_wealth) for metrics in metrics_list]
        mean_cumulative_profit = [float(metrics.mean_cumulative_profit) for metrics in metrics_list]
        summaries[policy_name] = HFPortfolioMultiSeedPolicySummary(
            mean_survival_years=mean(mean_survival_years),
            survival_years_std=_std(mean_survival_years),
            mean_full_horizon_survival_rate=mean(full_horizon_survival_rate),
            full_horizon_survival_rate_std=_std(full_horizon_survival_rate),
            mean_bankruptcy_rate=mean(bankruptcy_rate),
            bankruptcy_rate_std=_std(bankruptcy_rate),
            mean_terminal_wealth=mean(mean_terminal_wealth),
            terminal_wealth_std=_std(mean_terminal_wealth),
            mean_cumulative_profit=mean(mean_cumulative_profit),
            cumulative_profit_std=_std(mean_cumulative_profit),
        )
    return summaries


def run_hf_portfolio_multiseed_benchmark(
    base_config: HFPortfolioBenchmarkConfig,
    *,
    seeds: tuple[int, ...],
) -> HFPortfolioMultiSeedResult:
    if not seeds:
        raise ValueError("seeds must not be empty.")

    seed_results: dict[int, HFPortfolioBenchmarkResult] = {}
    for seed in seeds:
        per_seed_config = HFPortfolioBenchmarkConfig(
            **{
                **base_config.__dict__,
                "seed": seed,
                "workspace_root": str(Path(base_config.workspace_root) / f"seed_{seed}"),
                "output_dir": str(Path(base_config.output_dir) / f"seed_{seed}"),
            }
        )
        seed_results[seed] = run_hf_portfolio_benchmark(per_seed_config)

    return HFPortfolioMultiSeedResult(
        base_config=base_config,
        seeds=tuple(seeds),
        seed_results=seed_results,
        policy_summaries=_aggregate_multiseed_policy_metrics(seed_results),
    )


def run_hf_portfolio_seed_grid_benchmark(
    base_config: HFPortfolioBenchmarkConfig,
    *,
    training_seeds: tuple[int, ...],
    evaluation_seeds: tuple[int, ...],
) -> HFPortfolioSeedGridResult:
    if not training_seeds:
        raise ValueError("training_seeds must not be empty.")
    if not evaluation_seeds:
        raise ValueError("evaluation_seeds must not be empty.")

    grid_results: dict[tuple[int, int], HFPortfolioBenchmarkResult] = {}
    for training_seed in training_seeds:
        training_config = HFPortfolioBenchmarkConfig(
            **{
                **base_config.__dict__,
                "training_seed": training_seed,
                "workspace_root": str(Path(base_config.workspace_root) / f"train_seed_{training_seed}"),
                "output_dir": str(Path(base_config.output_dir) / f"train_seed_{training_seed}"),
            }
        )
        trained_model, context, view, training_loss = _train_hf_portfolio_model(training_config)
        for evaluation_seed in evaluation_seeds:
            evaluation_config = HFPortfolioBenchmarkConfig(
                **{
                    **training_config.__dict__,
                    "seed": evaluation_seed,
                }
            )
            benchmark_policies = _build_hf_benchmark_policies(
                evaluation_config,
                trained_model=trained_model,
                context=context,
            )
            policy_metrics = _evaluate_hf_benchmark_policies(
                evaluation_config,
                context=context,
                benchmark_policies=benchmark_policies,
            )
            grid_results[(training_seed, evaluation_seed)] = HFPortfolioBenchmarkResult(
                config=evaluation_config,
                train_examples=len(view.rows),
                observation_rate=view.result.observation_rate,
                stable_observation_rate=view.result.stable_observation_rate,
                distressed_observation_rate=view.result.distressed_observation_rate,
                policy_metrics=policy_metrics,
                training_loss=training_loss,
                learned_policy_name=f"hf_{base_config.uncertainty_mode}",
            )

    return HFPortfolioSeedGridResult(
        base_config=base_config,
        training_seeds=tuple(training_seeds),
        evaluation_seeds=tuple(evaluation_seeds),
        grid_results=grid_results,
        policy_summaries=_aggregate_multiseed_policy_metrics(
            {index: result for index, result in enumerate(grid_results.values())}
        ),
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


def format_hf_portfolio_multiseed_result(result: HFPortfolioMultiSeedResult) -> str:
    lines = [
        "HF Knightian multi-seed portfolio benchmark",
        f"benchmark: {result.base_config.benchmark_name}",
        f"evaluation seeds: {', '.join(str(seed) for seed in result.seeds)}",
        "",
        "policy                    surv_mean  surv_std  horizon_mean  bankrupt_mean  terminal_wealth_mean  cum_profit_mean",
        "------------------------  ---------  --------  ------------  -------------  --------------------  ---------------",
    ]
    for policy_name, summary in sorted(result.policy_summaries.items()):
        lines.append(
            f"{policy_name:<24}"
            f"  {summary.mean_survival_years:>9.2f}"
            f"  {summary.survival_years_std:>8.2f}"
            f"  {summary.mean_full_horizon_survival_rate:>12.2%}"
            f"  {summary.mean_bankruptcy_rate:>13.2%}"
            f"  {summary.mean_terminal_wealth:>20.2f}"
            f"  {summary.mean_cumulative_profit:>15.2f}"
        )
    lines.append("")
    lines.append("Per-seed summaries")
    for seed in result.seeds:
        seed_result = result.seed_results[seed]
        lines.append(f"seed {seed}")
        for policy_name, metrics in sorted(seed_result.policy_metrics.items()):
            lines.append(
                f"  {policy_name:<22}"
                f" surv={metrics.mean_survival_years:>6.2f}"
                f" horizon={metrics.full_horizon_survival_rate:>7.2%}"
                f" bankrupt={metrics.bankruptcy_rate:>7.2%}"
            )
    return "\n".join(lines)


def format_hf_portfolio_seed_grid_result(result: HFPortfolioSeedGridResult) -> str:
    lines = [
        "HF Knightian training/evaluation seed grid benchmark",
        f"benchmark: {result.base_config.benchmark_name}",
        f"training seeds: {', '.join(str(seed) for seed in result.training_seeds)}",
        f"evaluation seeds: {', '.join(str(seed) for seed in result.evaluation_seeds)}",
        "",
        "policy                    surv_mean  surv_std  horizon_mean  bankrupt_mean  terminal_wealth_mean  cum_profit_mean",
        "------------------------  ---------  --------  ------------  -------------  --------------------  ---------------",
    ]
    for policy_name, summary in sorted(result.policy_summaries.items()):
        lines.append(
            f"{policy_name:<24}"
            f"  {summary.mean_survival_years:>9.2f}"
            f"  {summary.survival_years_std:>8.2f}"
            f"  {summary.mean_full_horizon_survival_rate:>12.2%}"
            f"  {summary.mean_bankruptcy_rate:>13.2%}"
            f"  {summary.mean_terminal_wealth:>20.2f}"
            f"  {summary.mean_cumulative_profit:>15.2f}"
        )
    lines.append("")
    lines.append("Per training/evaluation seed pair")
    for training_seed in result.training_seeds:
        for evaluation_seed in result.evaluation_seeds:
            run_result = result.grid_results[(training_seed, evaluation_seed)]
            hf_metrics = run_result.policy_metrics[run_result.learned_policy_name]
            lines.append(
                f"train={training_seed} eval={evaluation_seed}"
                f" surv={hf_metrics.mean_survival_years:>6.2f}"
                f" horizon={hf_metrics.full_horizon_survival_rate:>7.2%}"
                f" bankrupt={hf_metrics.bankruptcy_rate:>7.2%}"
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
    parser.add_argument("--land-value-per-acre", type=float, default=HFPortfolioBenchmarkConfig.land_value_per_acre)
    parser.add_argument("--land-financed-fraction", type=float, default=HFPortfolioBenchmarkConfig.land_financed_fraction)
    parser.add_argument("--land-mortgage-rate", type=float, default=HFPortfolioBenchmarkConfig.land_mortgage_rate)
    parser.add_argument("--land-mortgage-years", type=int, default=HFPortfolioBenchmarkConfig.land_mortgage_years)
    parser.add_argument("--land-mortgage-grace-years", type=int, default=HFPortfolioBenchmarkConfig.land_mortgage_grace_years)
    parser.add_argument("--num-train-epochs", type=int, default=HFPortfolioBenchmarkConfig.num_train_epochs)
    parser.add_argument("--learning-rate", type=float, default=HFPortfolioBenchmarkConfig.learning_rate)
    parser.add_argument("--hidden-dim", type=int, default=HFPortfolioBenchmarkConfig.hidden_dim)
    parser.add_argument(
        "--uncertainty-mode",
        choices=["group", "score", "time_varying", "knightian", "surprise", "structural_break", "adaptive_v1"],
        default=HFPortfolioBenchmarkConfig.uncertainty_mode,
    )
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
        land_value_per_acre=args.land_value_per_acre,
        land_financed_fraction=args.land_financed_fraction,
        land_mortgage_rate=args.land_mortgage_rate,
        land_mortgage_years=args.land_mortgage_years,
        land_mortgage_grace_years=args.land_mortgage_grace_years,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        view_mode=args.view_mode,
        uncertainty_mode=args.uncertainty_mode,
    )


def parse_multiseed_args(argv: list[str] | None = None) -> tuple[HFPortfolioBenchmarkConfig, tuple[int, ...]]:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the HF Knightian portfolio benchmark across multiple evaluation seeds."
    )
    parser.add_argument("--benchmark", default=HFPortfolioBenchmarkConfig.benchmark_name)
    parser.add_argument("--seed", dest="seeds", action="append", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=HFPortfolioBenchmarkConfig.seed)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--seed-step", type=int, default=1)
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
    parser.add_argument("--land-value-per-acre", type=float, default=HFPortfolioBenchmarkConfig.land_value_per_acre)
    parser.add_argument("--land-financed-fraction", type=float, default=HFPortfolioBenchmarkConfig.land_financed_fraction)
    parser.add_argument("--land-mortgage-rate", type=float, default=HFPortfolioBenchmarkConfig.land_mortgage_rate)
    parser.add_argument("--land-mortgage-years", type=int, default=HFPortfolioBenchmarkConfig.land_mortgage_years)
    parser.add_argument("--land-mortgage-grace-years", type=int, default=HFPortfolioBenchmarkConfig.land_mortgage_grace_years)
    parser.add_argument("--num-train-epochs", type=int, default=HFPortfolioBenchmarkConfig.num_train_epochs)
    parser.add_argument("--learning-rate", type=float, default=HFPortfolioBenchmarkConfig.learning_rate)
    parser.add_argument("--hidden-dim", type=int, default=HFPortfolioBenchmarkConfig.hidden_dim)
    parser.add_argument(
        "--uncertainty-mode",
        choices=["group", "score", "time_varying", "knightian", "surprise", "structural_break", "adaptive_v1"],
        default=HFPortfolioBenchmarkConfig.uncertainty_mode,
    )
    parser.add_argument(
        "--view-mode",
        choices=["explicit_missing", "drop_unobserved", "truncate_after_unobserved"],
        default=HFPortfolioBenchmarkConfig.view_mode,
    )
    args = parser.parse_args(argv)
    seeds = (
        tuple(args.seeds)
        if args.seeds
        else tuple(args.seed_start + index * args.seed_step for index in range(args.seed_count))
    )
    if not seeds:
        raise ValueError("At least one evaluation seed is required.")
    return (
        HFPortfolioBenchmarkConfig(
            benchmark_name=args.benchmark,
            seed=seeds[0],
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
            land_value_per_acre=args.land_value_per_acre,
            land_financed_fraction=args.land_financed_fraction,
            land_mortgage_rate=args.land_mortgage_rate,
            land_mortgage_years=args.land_mortgage_years,
            land_mortgage_grace_years=args.land_mortgage_grace_years,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            view_mode=args.view_mode,
            uncertainty_mode=args.uncertainty_mode,
        ),
        seeds,
    )


def parse_seed_grid_args(
    argv: list[str] | None = None,
) -> tuple[HFPortfolioBenchmarkConfig, tuple[int, ...], tuple[int, ...]]:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the HF Knightian portfolio benchmark across training and evaluation seed grids."
    )
    parser.add_argument("--benchmark", default=HFPortfolioBenchmarkConfig.benchmark_name)
    parser.add_argument("--training-seed", dest="training_seeds", action="append", type=int, default=None)
    parser.add_argument("--training-seed-start", type=int, default=HFPortfolioBenchmarkConfig.training_seed)
    parser.add_argument("--training-seed-count", type=int, default=3)
    parser.add_argument("--training-seed-step", type=int, default=1)
    parser.add_argument("--eval-seed", dest="evaluation_seeds", action="append", type=int, default=None)
    parser.add_argument("--eval-seed-start", type=int, default=HFPortfolioBenchmarkConfig.seed)
    parser.add_argument("--eval-seed-count", type=int, default=3)
    parser.add_argument("--eval-seed-step", type=int, default=1)
    parser.add_argument("--train-paths", type=int, default=HFPortfolioBenchmarkConfig.train_paths)
    parser.add_argument("--test-paths", type=int, default=HFPortfolioBenchmarkConfig.test_paths)
    parser.add_argument("--horizon-years", type=int, default=HFPortfolioBenchmarkConfig.horizon_years)
    parser.add_argument("--workspace-root", type=str, default=HFPortfolioBenchmarkConfig.workspace_root)
    parser.add_argument("--output-dir", type=str, default=HFPortfolioBenchmarkConfig.output_dir)
    parser.add_argument("--cash", type=float, default=HFPortfolioBenchmarkConfig.initial_cash)
    parser.add_argument("--debt", type=float, default=HFPortfolioBenchmarkConfig.initial_debt)
    parser.add_argument("--credit-limit", type=float, default=HFPortfolioBenchmarkConfig.initial_credit_limit)
    parser.add_argument("--acres", type=float, default=HFPortfolioBenchmarkConfig.acres)
    parser.add_argument("--land-value-per-acre", type=float, default=HFPortfolioBenchmarkConfig.land_value_per_acre)
    parser.add_argument("--land-financed-fraction", type=float, default=HFPortfolioBenchmarkConfig.land_financed_fraction)
    parser.add_argument("--land-mortgage-rate", type=float, default=HFPortfolioBenchmarkConfig.land_mortgage_rate)
    parser.add_argument("--land-mortgage-years", type=int, default=HFPortfolioBenchmarkConfig.land_mortgage_years)
    parser.add_argument("--land-mortgage-grace-years", type=int, default=HFPortfolioBenchmarkConfig.land_mortgage_grace_years)
    parser.add_argument("--num-train-epochs", type=int, default=HFPortfolioBenchmarkConfig.num_train_epochs)
    parser.add_argument("--learning-rate", type=float, default=HFPortfolioBenchmarkConfig.learning_rate)
    parser.add_argument("--hidden-dim", type=int, default=HFPortfolioBenchmarkConfig.hidden_dim)
    parser.add_argument(
        "--uncertainty-mode",
        choices=["group", "score", "time_varying", "knightian", "surprise", "structural_break", "adaptive_v1"],
        default=HFPortfolioBenchmarkConfig.uncertainty_mode,
    )
    parser.add_argument(
        "--view-mode",
        choices=["explicit_missing", "drop_unobserved", "truncate_after_unobserved"],
        default=HFPortfolioBenchmarkConfig.view_mode,
    )
    args = parser.parse_args(argv)
    training_seeds = (
        tuple(args.training_seeds)
        if args.training_seeds
        else tuple(
            args.training_seed_start + index * args.training_seed_step
            for index in range(args.training_seed_count)
        )
    )
    evaluation_seeds = (
        tuple(args.evaluation_seeds)
        if args.evaluation_seeds
        else tuple(
            args.eval_seed_start + index * args.eval_seed_step
            for index in range(args.eval_seed_count)
        )
    )
    if not training_seeds or not evaluation_seeds:
        raise ValueError("Training and evaluation seed lists must not be empty.")
    return (
        HFPortfolioBenchmarkConfig(
            benchmark_name=args.benchmark,
            seed=evaluation_seeds[0],
            training_seed=training_seeds[0],
            train_paths=args.train_paths,
            test_paths=args.test_paths,
            horizon_years=args.horizon_years,
            workspace_root=args.workspace_root,
            output_dir=args.output_dir,
            initial_cash=args.cash,
            initial_debt=args.debt,
            initial_credit_limit=args.credit_limit,
            acres=args.acres,
            land_value_per_acre=args.land_value_per_acre,
            land_financed_fraction=args.land_financed_fraction,
            land_mortgage_rate=args.land_mortgage_rate,
            land_mortgage_years=args.land_mortgage_years,
            land_mortgage_grace_years=args.land_mortgage_grace_years,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            view_mode=args.view_mode,
            uncertainty_mode=args.uncertainty_mode,
        ),
        training_seeds,
        evaluation_seeds,
    )


def main(argv: list[str] | None = None) -> None:
    result = run_hf_portfolio_benchmark(parse_args(argv))
    print(format_hf_portfolio_benchmark_result(result))


def multiseed_main(argv: list[str] | None = None) -> None:
    config, seeds = parse_multiseed_args(argv)
    result = run_hf_portfolio_multiseed_benchmark(config, seeds=seeds)
    print(format_hf_portfolio_multiseed_result(result))


def seed_grid_main(argv: list[str] | None = None) -> None:
    config, training_seeds, evaluation_seeds = parse_seed_grid_args(argv)
    result = run_hf_portfolio_seed_grid_benchmark(
        config,
        training_seeds=training_seeds,
        evaluation_seeds=evaluation_seeds,
    )
    print(format_hf_portfolio_seed_grid_result(result))


if __name__ == "__main__":
    main()
