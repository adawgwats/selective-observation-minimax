from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence


IDENTITY_FIELDS = (
    "male",
    "female",
    "LGBTQ",
    "christian",
    "muslim",
    "other_religions",
    "black",
    "white",
)
NON_IDENTITY_GROUP = "identity_none"


@dataclass(frozen=True)
class CivilCommentsExperimentConfig:
    method: str = "erm"
    dataset_root: str = "data/wilds"
    output_dir: str = "outputs/wilds_civilcomments"
    model_name: str = "distilbert-base-uncased"
    max_length: int = 256
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 3
    seed: int = 17
    download: bool = True
    train_fraction: float = 1.0
    val_fraction: float = 1.0
    test_fraction: float = 1.0
    max_train_examples: int | None = None
    max_val_examples: int | None = None
    max_test_examples: int | None = None
    explicit_mnar: bool = False
    base_observation_rate: float = 0.95
    toxic_penalty: float = 0.20
    identity_penalty: float = 0.10
    identity_toxic_interaction_penalty: float = 0.15
    min_observation_rate: float = 0.05

    def __post_init__(self) -> None:
        if self.method not in {"erm", "robust_group"}:
            raise ValueError("method must be one of {'erm', 'robust_group'}.")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive.")
        if self.train_batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("batch sizes must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive.")
        for name, value in (
            ("train_fraction", self.train_fraction),
            ("val_fraction", self.val_fraction),
            ("test_fraction", self.test_fraction),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1].")
        for name, value in (
            ("max_train_examples", self.max_train_examples),
            ("max_val_examples", self.max_val_examples),
            ("max_test_examples", self.max_test_examples),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided.")
        if not 0.0 < self.base_observation_rate <= 1.0:
            raise ValueError("base_observation_rate must be in (0, 1].")
        if not 0.0 < self.min_observation_rate <= 1.0:
            raise ValueError("min_observation_rate must be in (0, 1].")
        if self.min_observation_rate > self.base_observation_rate:
            raise ValueError("min_observation_rate cannot exceed base_observation_rate.")
        for name, value in (
            ("toxic_penalty", self.toxic_penalty),
            ("identity_penalty", self.identity_penalty),
            ("identity_toxic_interaction_penalty", self.identity_toxic_interaction_penalty),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")


def load_experiment_config(path: str | Path) -> CivilCommentsExperimentConfig:
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as error:
            raise ImportError(
                "Loading YAML experiment configs requires pyyaml. Install minimax-optimization[wilds]."
            ) from error
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    elif suffix == ".json":
        data = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        raise ValueError("config path must end in .yaml, .yml, or .json")

    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ValueError("experiment config must deserialize to a mapping.")
    return CivilCommentsExperimentConfig(**data)


def config_to_dict(config: CivilCommentsExperimentConfig) -> dict[str, Any]:
    return asdict(config)


def metadata_row_to_dict(
    metadata_row: Sequence[Any] | Mapping[str, Any],
    metadata_fields: Sequence[str],
) -> dict[str, int]:
    if isinstance(metadata_row, Mapping):
        return {str(key): _coerce_int(value) for key, value in metadata_row.items()}

    if len(metadata_row) != len(metadata_fields):
        raise ValueError("metadata_row and metadata_fields must have the same length.")

    return {
        field: _coerce_int(value)
        for field, value in zip(metadata_fields, metadata_row)
    }


def extract_training_group_memberships(
    metadata_row: Sequence[Any] | Mapping[str, Any],
    metadata_fields: Sequence[str],
) -> list[str]:
    metadata = metadata_row_to_dict(metadata_row, metadata_fields)
    active_groups = [
        identity
        for identity in IDENTITY_FIELDS
        if metadata.get(identity, 0) == 1
    ]
    if active_groups:
        return active_groups
    return [NON_IDENTITY_GROUP]


def synthetic_observation_probability(
    metadata_row: Sequence[Any] | Mapping[str, Any],
    metadata_fields: Sequence[str],
    config: CivilCommentsExperimentConfig,
) -> float:
    metadata = metadata_row_to_dict(metadata_row, metadata_fields)
    has_identity = any(metadata.get(identity, 0) == 1 for identity in IDENTITY_FIELDS)
    is_toxic = metadata.get("y", 0) == 1

    probability = config.base_observation_rate
    if is_toxic:
        probability -= config.toxic_penalty
    if has_identity:
        probability -= config.identity_penalty
    if is_toxic and has_identity:
        probability -= config.identity_toxic_interaction_penalty
    return min(max(probability, config.min_observation_rate), 1.0)


def build_observed_mask(
    metadata_rows: Sequence[Sequence[Any] | Mapping[str, Any]],
    metadata_fields: Sequence[str],
    *,
    split_name: str,
    config: CivilCommentsExperimentConfig,
) -> list[bool]:
    if split_name != "train" or not config.explicit_mnar:
        return [True] * len(metadata_rows)

    rng = random.Random(config.seed)
    memberships = [
        extract_training_group_memberships(row, metadata_fields)
        for row in metadata_rows
    ]
    observed_mask = [
        rng.random() < synthetic_observation_probability(row, metadata_fields, config)
        for row in metadata_rows
    ]
    _ensure_group_coverage(observed_mask, memberships)
    return observed_mask


def summarize_memberships(
    group_memberships: Sequence[Sequence[str]],
    observed_mask: Sequence[bool] | None = None,
) -> dict[str, dict[str, int]]:
    if observed_mask is not None and len(group_memberships) != len(observed_mask):
        raise ValueError("group_memberships and observed_mask must have the same length.")

    summary: dict[str, dict[str, int]] = {}
    for index, memberships in enumerate(group_memberships):
        observed = True if observed_mask is None else observed_mask[index]
        for group_id in memberships:
            group_summary = summary.setdefault(group_id, {"total": 0, "observed": 0})
            group_summary["total"] += 1
            if observed:
                group_summary["observed"] += 1
    return summary


def _ensure_group_coverage(
    observed_mask: list[bool],
    group_memberships: Sequence[Sequence[str]],
) -> None:
    group_to_indices: dict[str, list[int]] = {}
    for index, memberships in enumerate(group_memberships):
        for group_id in memberships:
            group_to_indices.setdefault(group_id, []).append(index)

    for indices in group_to_indices.values():
        if any(observed_mask[index] for index in indices):
            continue
        observed_mask[indices[0]] = True


def _coerce_int(value: Any) -> int:
    if hasattr(value, "item"):
        value = value.item()
    return int(value)
