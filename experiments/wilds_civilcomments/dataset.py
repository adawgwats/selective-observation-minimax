from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from experiments.wilds_civilcomments.common import (
    CivilCommentsExperimentConfig,
    build_observed_mask,
    extract_training_group_memberships,
)


class TrainerDependencyError(ImportError):
    pass


@dataclass(frozen=True)
class CivilCommentsSplitData:
    split_name: str
    dataset: Any
    labels: list[int]
    metadata_rows: list[list[int]]
    metadata_fields: list[str]
    group_memberships: list[list[str]]
    observed_mask: list[bool]


class StripMinimaxMetadataCollator:
    def __init__(self, base_collator: Any) -> None:
        self.base_collator = base_collator

    def __call__(self, features: Sequence[dict[str, Any]]) -> Any:
        stripped_features = []
        for feature in features:
            stripped = dict(feature)
            stripped.pop("group_id", None)
            stripped.pop("label_observed", None)
            stripped_features.append(stripped)
        return self.base_collator(stripped_features)


def require_wilds_dependencies() -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer, DataCollatorWithPadding
        from wilds import get_dataset
    except ImportError as error:
        raise TrainerDependencyError(
            "WILDS CivilComments experiments require transformers and wilds. "
            "Install minimax-optimization[wilds]."
        ) from error

    return {
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorWithPadding": DataCollatorWithPadding,
        "get_dataset": get_dataset,
    }


def load_civilcomments_splits(
    config: CivilCommentsExperimentConfig,
) -> tuple[Any, dict[str, CivilCommentsSplitData], Any]:
    deps = require_wilds_dependencies()
    AutoTokenizer = deps["AutoTokenizer"]
    DataCollatorWithPadding = deps["DataCollatorWithPadding"]
    get_dataset = deps["get_dataset"]

    dataset = get_dataset(
        "civilcomments",
        root_dir=config.dataset_root,
        download=config.download,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    collator = StripMinimaxMetadataCollator(DataCollatorWithPadding(tokenizer=tokenizer))

    split_data = {
        split_name: _build_split(
            subset=dataset.get_subset(split_name, frac=_split_fraction(config, split_name)),
            tokenizer=tokenizer,
            config=config,
            split_name=split_name,
            max_examples=_split_limit(config, split_name),
        )
        for split_name in ("train", "val", "test")
    }
    return dataset, split_data, collator


def build_training_group_summary(split: CivilCommentsSplitData) -> dict[str, dict[str, int]]:
    from experiments.wilds_civilcomments.common import summarize_memberships

    return summarize_memberships(split.group_memberships, split.observed_mask)


def _build_split(
    *,
    subset: Any,
    tokenizer: Any,
    config: CivilCommentsExperimentConfig,
    split_name: str,
    max_examples: int | None,
) -> CivilCommentsSplitData:
    metadata_fields = list(subset._metadata_fields)
    texts: list[str] = []
    labels: list[int] = []
    metadata_rows: list[list[int]] = []

    example_count = len(subset) if max_examples is None else min(len(subset), max_examples)
    for index in range(example_count):
        text, label, metadata = subset[index]
        texts.append(text)
        labels.append(_coerce_int(label))
        metadata_rows.append([_coerce_int(value) for value in metadata])

    group_memberships = [
        extract_training_group_memberships(metadata_row, metadata_fields)
        for metadata_row in metadata_rows
    ]
    observed_mask = build_observed_mask(
        metadata_rows,
        metadata_fields,
        split_name=split_name,
        config=config,
    )
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=config.max_length,
    )

    dataset = TokenizedCivilCommentsDataset(
        encodings=encodings,
        labels=labels,
        group_memberships=group_memberships,
        observed_mask=observed_mask,
    )
    return CivilCommentsSplitData(
        split_name=split_name,
        dataset=dataset,
        labels=labels,
        metadata_rows=metadata_rows,
        metadata_fields=metadata_fields,
        group_memberships=group_memberships,
        observed_mask=observed_mask,
    )


class TokenizedCivilCommentsDataset:
    def __init__(
        self,
        *,
        encodings: Any,
        labels: Sequence[int],
        group_memberships: Sequence[Sequence[str]],
        observed_mask: Sequence[bool],
    ) -> None:
        self.encodings = encodings
        self.labels = list(labels)
        self.group_memberships = [list(membership) for membership in group_memberships]
        self.observed_mask = [bool(value) for value in observed_mask]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {
            key: value[index]
            for key, value in self.encodings.items()
        }
        item["labels"] = self.labels[index]
        item["group_id"] = self.group_memberships[index]
        item["label_observed"] = self.observed_mask[index]
        return item


def _coerce_int(value: Any) -> int:
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _split_fraction(config: CivilCommentsExperimentConfig, split_name: str) -> float:
    if split_name == "train":
        return config.train_fraction
    if split_name == "val":
        return config.val_fraction
    if split_name == "test":
        return config.test_fraction
    raise ValueError(f"unsupported split: {split_name}")


def _split_limit(config: CivilCommentsExperimentConfig, split_name: str) -> int | None:
    if split_name == "train":
        return config.max_train_examples
    if split_name == "val":
        return config.max_val_examples
    if split_name == "test":
        return config.max_test_examples
    raise ValueError(f"unsupported split: {split_name}")
