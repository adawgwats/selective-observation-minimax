from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from minimax_core import SyntheticMNARConfig, estimate_group_snapshot
from minimax_hf import (
    DatasetSchemaError,
    MinimaxDataCollator,
    MinimaxHFConfig,
    build_synthetic_mnar_view,
    build_loss_adapter,
    prepare_training_args,
    validate_dataset_columns,
)
from minimax_hf.trainer import _apply_online_mnar_assumption, _build_adversary
from minimax_core.hf_portfolio_benchmark import (
    HFPortfolioBenchmarkConfig,
    HFPortfolioBenchmarkResult,
    _aggregate_multiseed_policy_metrics,
    parse_multiseed_args,
    parse_seed_grid_args,
)


@dataclass(frozen=True)
class StubTrainingArguments:
    output_dir: str = "outputs"
    remove_unused_columns: bool = True


class StubDataset:
    def __init__(self, rows: list[dict[str, object]], column_names: list[str] | None = None):
        self._rows = rows
        self.column_names = column_names or list(rows[0].keys())

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self._rows[index]


def test_prepare_training_args_disables_remove_unused_columns() -> None:
    training_args = StubTrainingArguments()

    prepared = prepare_training_args(training_args)

    assert prepared is not training_args
    assert prepared.remove_unused_columns is False


def test_minimax_hf_config_rejects_unknown_task_type() -> None:
    with pytest.raises(ValueError):
        MinimaxHFConfig(task_type="language_modeling")  # type: ignore[arg-type]


def test_minimax_hf_config_rejects_invalid_assumed_observation_rate() -> None:
    with pytest.raises(ValueError):
        MinimaxHFConfig(assumed_observation_rate=1.2)


def test_minimax_hf_config_rejects_invalid_uncertainty_mode() -> None:
    with pytest.raises(ValueError):
        MinimaxHFConfig(uncertainty_mode="unknown")  # type: ignore[arg-type]


def test_build_loss_adapter_supports_common_hf_tasks() -> None:
    sequence_adapter = build_loss_adapter("sequence_classification")
    regression_adapter = build_loss_adapter("regression")
    token_adapter = build_loss_adapter("token_classification")

    assert callable(sequence_adapter)
    assert callable(regression_adapter)
    assert callable(token_adapter)


def test_build_loss_adapter_rejects_unknown_task() -> None:
    with pytest.raises(ValueError):
        build_loss_adapter("language_modeling")  # type: ignore[arg-type]


def test_validate_dataset_columns_requires_group_key() -> None:
    dataset = StubDataset([{"labels": 1, "input_ids": [1, 2, 3]}])

    with pytest.raises(DatasetSchemaError):
        validate_dataset_columns(
            dataset,
            group_key="group_id",
            observed_key="label_observed",
        )


def test_validate_dataset_columns_allows_missing_observed_key_by_default() -> None:
    dataset = StubDataset([{"labels": 1, "group_id": "stable"}])

    validate_dataset_columns(
        dataset,
        group_key="group_id",
        observed_key="label_observed",
    )


def test_validate_dataset_columns_supports_extra_required_keys() -> None:
    dataset = StubDataset([{"labels": 1, "group_id": "stable", "time_index": 0}])

    validate_dataset_columns(
        dataset,
        group_key="group_id",
        observed_key="label_observed",
        extra_required_keys=("time_index",),
    )

    with pytest.raises(DatasetSchemaError):
        validate_dataset_columns(
            StubDataset([{"labels": 1, "group_id": "stable"}]),
            group_key="group_id",
            observed_key="label_observed",
            extra_required_keys=("time_index",),
        )


def test_build_adversary_supports_knightian_modes() -> None:
    assert _build_adversary(MinimaxHFConfig(uncertainty_mode="group")).__class__.__name__ == "SelectiveObservationAdversary"
    assert _build_adversary(MinimaxHFConfig(uncertainty_mode="time_varying")).__class__.__name__ == "TimeVaryingObservationAdversary"
    assert _build_adversary(MinimaxHFConfig(uncertainty_mode="knightian")).__class__.__name__ == "KnightianObservationAdversary"
    assert _build_adversary(MinimaxHFConfig(uncertainty_mode="surprise")).__class__.__name__ == "SurpriseDrivenObservationAdversary"


def test_minimax_data_collator_preserves_metadata_and_defaults_observed_mask() -> None:
    def base_collator(features: list[dict[str, object]]) -> dict[str, object]:
        return {
            "labels": [feature["labels"] for feature in features],
            "input_ids": [feature["input_ids"] for feature in features],
        }

    collator = MinimaxDataCollator(
        base_collator,
        group_key="group_id",
        observed_key="label_observed",
    )

    batch = collator(
        [
            {"input_ids": [1, 2], "labels": 0, "group_id": "stable"},
            {
                "input_ids": [3, 4],
                "labels": 1,
                "group_id": "distressed",
                "label_observed": False,
            },
        ]
    )

    assert batch["input_ids"] == [[1, 2], [3, 4]]
    assert batch["labels"] == [0, 1]
    assert batch["group_id"] == ["stable", "distressed"]
    assert batch["label_observed"] == [True, False]


def test_minimax_data_collator_preserves_multi_membership_groups() -> None:
    def base_collator(features: list[dict[str, object]]) -> dict[str, object]:
        return {
            "labels": [feature["labels"] for feature in features],
            "input_ids": [feature["input_ids"] for feature in features],
        }

    collator = MinimaxDataCollator(
        base_collator,
        group_key="group_id",
        observed_key="label_observed",
    )

    batch = collator(
        [
            {"input_ids": [1, 2], "labels": 0, "group_id": ["female", "black"]},
            {"input_ids": [3, 4], "labels": 1, "group_id": ["male"]},
        ]
    )

    assert batch["group_id"] == [["female", "black"], ["male"]]
    assert batch["label_observed"] == [True, True]


def test_build_synthetic_mnar_view_preserves_rows_for_explicit_missing() -> None:
    records = [
        {"labels": 0.2, "group_id": "stable", "path_index": 0, "step_index": 0},
        {"labels": -0.8, "group_id": "distressed", "path_index": 0, "step_index": 1},
        {"labels": -1.1, "group_id": "distressed", "path_index": 1, "step_index": 0},
    ]

    view = build_synthetic_mnar_view(
        records,
        config=SyntheticMNARConfig(
            seed=3,
            view_mode="explicit_missing",
            base_observation_probability=0.9,
            distressed_penalty=0.8,
        ),
        path_key="path_index",
        step_key="step_index",
        latent_label_key="latent_label",
    )

    assert len(view.rows) == len(records)
    assert all("label_observed" in row for row in view.rows)
    assert all("latent_label" in row for row in view.rows)
    assert any(not row["label_observed"] for row in view.rows)


def test_build_synthetic_mnar_view_drops_unobserved_rows() -> None:
    records = [
        {"labels": 0.2, "group_id": "stable", "path_index": 0, "step_index": 0},
        {"labels": -0.8, "group_id": "distressed", "path_index": 0, "step_index": 1},
        {"labels": -1.1, "group_id": "distressed", "path_index": 1, "step_index": 0},
    ]

    view = build_synthetic_mnar_view(
        records,
        config=SyntheticMNARConfig(
            seed=3,
            view_mode="drop_unobserved",
            base_observation_probability=0.9,
            distressed_penalty=0.8,
        ),
        path_key="path_index",
        step_key="step_index",
    )

    assert len(view.rows) < len(records)
    assert all(row["label_observed"] for row in view.rows)
    assert view.result.observation_rate < 1.0


def test_build_synthetic_mnar_view_rejects_multi_membership_groups() -> None:
    with pytest.raises(DatasetSchemaError):
        build_synthetic_mnar_view(
            [
                {
                    "labels": 1.0,
                    "group_id": ["stable", "region_a"],
                    "path_index": 0,
                    "step_index": 0,
                }
            ],
            config=SyntheticMNARConfig(),
            path_key="path_index",
            step_key="step_index",
        )


def test_build_synthetic_mnar_view_requires_path_for_truncation() -> None:
    with pytest.raises(ValueError):
        build_synthetic_mnar_view(
            [{"labels": 1.0, "group_id": "stable"}],
            config=SyntheticMNARConfig(view_mode="truncate_after_unobserved"),
        )


def test_build_synthetic_mnar_view_supports_custom_distressed_group_values() -> None:
    view = build_synthetic_mnar_view(
        [
            {"labels": 0.2, "group_id": "north", "path_index": 0, "step_index": 0},
            {"labels": 0.1, "group_id": "south", "path_index": 0, "step_index": 1},
        ],
        config=SyntheticMNARConfig(
            seed=4,
            view_mode="explicit_missing",
            base_observation_probability=0.95,
            distressed_penalty=0.8,
        ),
        path_key="path_index",
        step_key="step_index",
        distressed_group_values=["south"],
    )

    assert len(view.rows) == 2
    assert any(not row["label_observed"] for row in view.rows)


def test_apply_online_mnar_assumption_overrides_snapshot_rate() -> None:
    snapshot = estimate_group_snapshot(
        losses=[0.1, 0.5, 0.4],
        group_ids=["stable", "distressed", "distressed"],
        observed_mask=[True, True, False],
    )

    adjusted = _apply_online_mnar_assumption(
        snapshot,
        [True, True, False],
        MinimaxHFConfig(online_mnar=True, assumed_observation_rate=0.4),
    )

    assert snapshot.observation_rate == pytest.approx(2 / 3)
    assert adjusted.observation_rate == pytest.approx(0.4)


def test_aggregate_multiseed_policy_metrics_computes_means_and_std() -> None:
    seed_results = {
        13: HFPortfolioBenchmarkResult(
            config=HFPortfolioBenchmarkConfig(seed=13),
            train_examples=10,
            observation_rate=0.6,
            stable_observation_rate=0.7,
            distressed_observation_rate=0.4,
            policy_metrics={
                "hf_knightian": SimpleNamespace(
                    mean_survival_years=20.0,
                    full_horizon_survival_rate=0.20,
                    bankruptcy_rate=0.80,
                    mean_terminal_wealth=1_000_000.0,
                    mean_cumulative_profit=750_000.0,
                )
            },
            training_loss=1.0,
            learned_policy_name="hf_knightian",
        ),
        14: HFPortfolioBenchmarkResult(
            config=HFPortfolioBenchmarkConfig(seed=14),
            train_examples=10,
            observation_rate=0.6,
            stable_observation_rate=0.7,
            distressed_observation_rate=0.4,
            policy_metrics={
                "hf_knightian": SimpleNamespace(
                    mean_survival_years=30.0,
                    full_horizon_survival_rate=0.40,
                    bankruptcy_rate=0.60,
                    mean_terminal_wealth=1_500_000.0,
                    mean_cumulative_profit=1_000_000.0,
                )
            },
            training_loss=1.0,
            learned_policy_name="hf_knightian",
        ),
    }

    summary = _aggregate_multiseed_policy_metrics(seed_results)["hf_knightian"]

    assert summary.mean_survival_years == pytest.approx(25.0)
    assert summary.survival_years_std == pytest.approx(5.0)
    assert summary.mean_full_horizon_survival_rate == pytest.approx(0.30)
    assert summary.mean_bankruptcy_rate == pytest.approx(0.70)


def test_parse_multiseed_args_builds_seed_sequence() -> None:
    config, seeds = parse_multiseed_args(
        [
            "--seed-start",
            "20",
            "--seed-count",
            "3",
            "--seed-step",
            "2",
        ]
    )

    assert seeds == (20, 22, 24)
    assert config.seed == 20


def test_parse_seed_grid_args_builds_training_and_eval_sequences() -> None:
    config, training_seeds, evaluation_seeds = parse_seed_grid_args(
        [
            "--training-seed-start",
            "30",
            "--training-seed-count",
            "2",
            "--eval-seed-start",
            "40",
            "--eval-seed-count",
            "3",
        ]
    )

    assert training_seeds == (30, 31)
    assert evaluation_seeds == (40, 41, 42)
    assert config.training_seed == 30
    assert config.seed == 40
