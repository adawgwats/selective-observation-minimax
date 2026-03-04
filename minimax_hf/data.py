from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

from minimax_core import SyntheticMNARConfig, SyntheticMNARResult, apply_synthetic_mnar


class DatasetSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class SyntheticMNARView:
    rows: list[dict[str, Any]]
    result: SyntheticMNARResult


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
    extra_required_keys: Sequence[str] = (),
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
    required.update(extra_required_keys)

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


def build_synthetic_mnar_view(
    records: Sequence[Mapping[str, Any]],
    *,
    config: SyntheticMNARConfig,
    label_key: str = "labels",
    group_key: str = "group_id",
    observed_key: str = "label_observed",
    path_key: str | None = None,
    step_key: str | None = None,
    weather_key: str | None = None,
    alive_next_key: str | None = None,
    latent_label_key: str | None = None,
    distressed_group_values: Sequence[Any] | None = None,
) -> SyntheticMNARView:
    if not records:
        return SyntheticMNARView(
            rows=[],
            result=SyntheticMNARResult(
                observed_mask=(),
                keep_mask=(),
                observed_values=(),
                observation_probabilities=(),
                observation_rate=0.0,
                stable_observation_rate=1.0,
                distressed_observation_rate=1.0,
            ),
        )

    if config.view_mode == "truncate_after_unobserved" and path_key is None:
        raise ValueError("path_key is required for truncate_after_unobserved views.")

    copied_records = [dict(record) for record in records]
    labels: list[float] = []
    group_ids: list[str] = []
    path_indices: list[int] = []
    step_indices: list[int] = []
    weather_regimes: list[str] | None = [] if weather_key is not None else None
    farm_alive_next_year: list[bool] | None = [] if alive_next_key is not None else None

    for index, record in enumerate(copied_records):
        if label_key not in record:
            raise DatasetSchemaError(f"record is missing required label key '{label_key}'.")
        if group_key not in record:
            raise DatasetSchemaError(f"record is missing required group key '{group_key}'.")

        group_id = record[group_key]
        if isinstance(group_id, (list, tuple, set, frozenset)):
            raise DatasetSchemaError(
                "build_synthetic_mnar_view requires a single group id per record."
            )

        labels.append(float(record[label_key]))
        if distressed_group_values is not None:
            group_ids.append("distressed" if group_id in distressed_group_values else "stable")
        else:
            group_ids.append(str(group_id))
        path_indices.append(int(record[path_key]) if path_key is not None else index)
        step_indices.append(int(record[step_key]) if step_key is not None else index)

        if weather_regimes is not None:
            weather_regimes.append(str(record.get(weather_key, "normal")))
        if farm_alive_next_year is not None:
            farm_alive_next_year.append(bool(record.get(alive_next_key, True)))

    result = apply_synthetic_mnar(
        labels=labels,
        group_ids=group_ids,
        path_indices=path_indices,
        step_indices=step_indices,
        weather_regimes=weather_regimes,
        farm_alive_next_year=farm_alive_next_year,
        config=config,
    )

    transformed_rows: list[dict[str, Any]] = []
    for index, record in enumerate(copied_records):
        if not result.keep_mask[index]:
            continue
        transformed = dict(record)
        if latent_label_key is not None:
            transformed[latent_label_key] = record[label_key]
        transformed[observed_key] = bool(result.observed_mask[index])
        transformed_rows.append(transformed)

    return SyntheticMNARView(rows=transformed_rows, result=result)


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
