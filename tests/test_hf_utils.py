from __future__ import annotations

from dataclasses import dataclass

import pytest

from minimax_hf import (
    DatasetSchemaError,
    MinimaxDataCollator,
    MinimaxHFConfig,
    build_loss_adapter,
    prepare_training_args,
    validate_dataset_columns,
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
