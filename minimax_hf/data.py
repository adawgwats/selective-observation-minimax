from __future__ import annotations

from dataclasses import is_dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence


class DatasetSchemaError(ValueError):
    pass


def prepare_training_args(training_args: Any) -> Any:
    if training_args is None or not hasattr(training_args, "remove_unused_columns"):
        return training_args
    if training_args.remove_unused_columns is False:
        return training_args
    if not is_dataclass(training_args):
        raise TypeError(
            "Training arguments must be a dataclass with remove_unused_columns support."
        )
    return replace(training_args, remove_unused_columns=False)


def validate_dataset_columns(
    dataset: Any,
    *,
    group_key: str,
    observed_key: str,
    require_observed_key: bool = False,
) -> None:
    if dataset is None:
        return

    if hasattr(dataset, "column_names"):
        columns = set(dataset.column_names)
    else:
        sample = _sample_record(dataset)
        columns = set(sample.keys())

    required = {group_key}
    if require_observed_key:
        required.add(observed_key)

    missing = sorted(required - columns)
    if missing:
        raise DatasetSchemaError(
            f"dataset is missing required minimax columns: {', '.join(missing)}"
        )


def _sample_record(dataset: Any) -> Mapping[str, Any]:
    if isinstance(dataset, Mapping):
        return dataset

    if hasattr(dataset, "__len__") and len(dataset) == 0:
        raise DatasetSchemaError("dataset must contain at least one example.")

    if hasattr(dataset, "__getitem__"):
        sample = dataset[0]
    else:
        iterator = iter(dataset)
        try:
            sample = next(iterator)
        except StopIteration as error:
            raise DatasetSchemaError("dataset must contain at least one example.") from error

    if not isinstance(sample, Mapping):
        raise DatasetSchemaError("dataset examples must behave like mappings.")
    return sample


class MinimaxDataCollator:
    def __init__(
        self,
        base_collator: Callable[[list[dict[str, Any]]], Mapping[str, Any]],
        *,
        group_key: str,
        observed_key: str,
    ) -> None:
        self.base_collator = base_collator
        self.group_key = group_key
        self.observed_key = observed_key

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        stripped_features: list[dict[str, Any]] = []
        group_ids: list[Any] = []
        observed_mask: list[bool] = []

        for feature in features:
            if self.group_key not in feature:
                raise DatasetSchemaError(
                    f"feature is missing required minimax key '{self.group_key}'."
                )
            group_ids.append(feature[self.group_key])
            observed_mask.append(bool(feature.get(self.observed_key, True)))

            stripped = dict(feature)
            stripped.pop(self.group_key, None)
            stripped.pop(self.observed_key, None)
            stripped_features.append(stripped)

        batch = dict(self.base_collator(stripped_features))
        batch[self.group_key] = group_ids
        batch[self.observed_key] = observed_mask
        return batch


def is_minimax_data_collator(
    collator: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]],
) -> bool:
    return isinstance(collator, MinimaxDataCollator)
