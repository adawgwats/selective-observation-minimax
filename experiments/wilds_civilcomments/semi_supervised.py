from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

from torch.utils.data import ConcatDataset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minimax_hf import MinimaxHFConfig, MinimaxTrainer

from experiments.wilds_civilcomments.common import (
    CivilCommentsExperimentConfig,
    NON_IDENTITY_GROUP,
    config_to_dict,
    extract_training_group_memberships,
    load_experiment_config,
)
from experiments.wilds_civilcomments.dataset import (
    TokenizedCivilCommentsDataset,
    build_training_group_summary,
    load_civilcomments_splits,
    require_wilds_dependencies,
)
from experiments.wilds_civilcomments.metrics import (
    format_split_metrics,
    logits_to_predictions_and_scores,
)
from experiments.wilds_civilcomments.train import (
    _build_minimax_config,
    _build_training_arguments,
    _require_transformers,
    evaluate_split,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a two-stage WILDS CivilComments semi-supervised experiment: "
            "train teacher on labeled data, pseudo-label extra_unlabeled, then train student."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML or JSON experiment config for the labeled setup.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root directory for teacher/student outputs. Defaults to <config.output_dir>_semi_supervised.",
    )
    parser.add_argument(
        "--unlabeled-fraction",
        type=float,
        default=1.0,
        help="Fraction of the WILDS extra_unlabeled split to load before max_unlabeled_examples cap.",
    )
    parser.add_argument(
        "--max-unlabeled-examples",
        type=int,
        default=8192,
        help="Maximum number of unlabeled examples to pseudo-label.",
    )
    parser.add_argument(
        "--pseudo-label-threshold",
        type=float,
        default=0.90,
        help="Confidence threshold for pseudo-label selection in [0.5, 1.0).",
    )
    parser.add_argument(
        "--student-num-train-epochs",
        type=float,
        default=1.0,
        help="Number of student fine-tuning epochs on labeled+pseudo data.",
    )
    parser.add_argument(
        "--download-unlabeled",
        action="store_true",
        help="Allow WILDS to download the extra_unlabeled split if not present locally.",
    )
    return parser.parse_args(argv)


def run_semi_supervised_experiment(
    *,
    config_path: str | Path,
    output_root: str | Path | None,
    unlabeled_fraction: float,
    max_unlabeled_examples: int | None,
    pseudo_label_threshold: float,
    student_num_train_epochs: float,
    download_unlabeled: bool,
) -> dict[str, Any]:
    if not 0.0 < unlabeled_fraction <= 1.0:
        raise ValueError("unlabeled_fraction must be in (0, 1].")
    if max_unlabeled_examples is not None and max_unlabeled_examples <= 0:
        raise ValueError("max_unlabeled_examples must be positive when provided.")
    if not 0.5 <= pseudo_label_threshold < 1.0:
        raise ValueError("pseudo_label_threshold must be in [0.5, 1.0).")
    if student_num_train_epochs <= 0.0:
        raise ValueError("student_num_train_epochs must be positive.")

    config = load_experiment_config(config_path)
    deps = _require_transformers()
    Trainer = deps["Trainer"]
    TrainingArguments = deps["TrainingArguments"]
    AutoModelForSequenceClassification = deps["AutoModelForSequenceClassification"]
    set_seed = deps["set_seed"]

    set_seed(config.seed)
    random.seed(config.seed)

    output_dir = Path(output_root) if output_root is not None else Path(f"{config.output_dir}_semi_supervised")
    teacher_output_dir = output_dir / "teacher"
    student_output_dir = output_dir / "student"
    teacher_output_dir.mkdir(parents=True, exist_ok=True)
    student_output_dir.mkdir(parents=True, exist_ok=True)

    wilds_dataset, splits, collator = load_civilcomments_splits(config)
    train_summary = build_training_group_summary(splits["train"])

    minimax_config, effective_assumed_observation_rate = _maybe_build_minimax_config(
        config,
        train_split=splits["train"],
    )

    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
    )
    teacher_args = _build_training_arguments(
        TrainingArguments,
        output_dir=str(teacher_output_dir),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        seed=config.seed,
        remove_unused_columns=False,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        report_to=[],
    )
    teacher_trainer = _build_trainer(
        config=config,
        trainer_cls=Trainer,
        model=teacher_model,
        args=teacher_args,
        train_dataset=splits["train"].dataset,
        eval_dataset=splits["val"].dataset,
        data_collator=collator,
        minimax_config=minimax_config,
    )
    teacher_train_result = teacher_trainer.train()
    teacher_checkpoint_path = teacher_output_dir / "checkpoint-final"
    teacher_trainer.save_model(str(teacher_checkpoint_path))
    teacher_evaluated_splits = {
        split_name: evaluate_split(
            trainer=teacher_trainer,
            split=splits[split_name],
            wilds_dataset=wilds_dataset,
        )
        for split_name in ("val", "test")
    }

    unlabeled_data = load_unlabeled_for_pseudo_labels(
        config=config,
        fraction=unlabeled_fraction,
        max_examples=max_unlabeled_examples,
        download=download_unlabeled,
    )
    prediction_output = teacher_trainer.predict(unlabeled_data["predict_dataset"])
    logits = prediction_output.predictions.tolist()
    _predicted_labels, positive_scores = logits_to_predictions_and_scores(logits)
    pseudo_selection = select_pseudo_labels(
        positive_scores,
        threshold=pseudo_label_threshold,
    )
    selected_indices = pseudo_selection["indices"]
    pseudo_labels = pseudo_selection["labels"]
    if not selected_indices:
        raise RuntimeError(
            "no pseudo-labeled examples selected. Lower --pseudo-label-threshold or increase --max-unlabeled-examples."
        )

    pseudo_dataset = build_pseudo_labeled_dataset(
        encodings=unlabeled_data["encodings"],
        group_memberships=unlabeled_data["group_memberships"],
        selected_indices=selected_indices,
        pseudo_labels=pseudo_labels,
    )
    combined_train_dataset = ConcatDataset(
        [
            splits["train"].dataset,
            pseudo_dataset,
        ]
    )

    student_model = AutoModelForSequenceClassification.from_pretrained(
        str(teacher_checkpoint_path),
        num_labels=2,
    )
    student_args = _build_training_arguments(
        TrainingArguments,
        output_dir=str(student_output_dir),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=student_num_train_epochs,
        seed=config.seed,
        remove_unused_columns=False,
        save_strategy=config.save_strategy,
        logging_strategy="steps",
        logging_steps=50,
        report_to=[],
    )
    student_trainer = _build_trainer(
        config=config,
        trainer_cls=Trainer,
        model=student_model,
        args=student_args,
        train_dataset=combined_train_dataset,
        eval_dataset=splits["val"].dataset,
        data_collator=collator,
        minimax_config=minimax_config,
    )
    student_train_result = student_trainer.train()
    if config.save_final_checkpoint:
        student_trainer.save_model(str(student_output_dir / "checkpoint-final"))
    student_evaluated_splits = {
        split_name: evaluate_split(
            trainer=student_trainer,
            split=splits[split_name],
            wilds_dataset=wilds_dataset,
        )
        for split_name in ("val", "test")
    }

    unlabeled_true_labels = unlabeled_data["true_labels"]
    selected_true_labels = [int(unlabeled_true_labels[index]) for index in selected_indices]
    pseudo_label_accuracy = _accuracy(pseudo_labels, selected_true_labels)

    metrics_payload = {
        "config": config_to_dict(config),
        "semi_supervised": {
            "output_root": str(output_dir),
            "teacher_output_dir": str(teacher_output_dir),
            "student_output_dir": str(student_output_dir),
            "unlabeled_fraction": float(unlabeled_fraction),
            "max_unlabeled_examples": max_unlabeled_examples,
            "pseudo_label_threshold": float(pseudo_label_threshold),
            "student_num_train_epochs": float(student_num_train_epochs),
            "unlabeled_candidates": len(unlabeled_data["true_labels"]),
            "pseudo_selected": len(selected_indices),
            "pseudo_selection_rate": len(selected_indices) / len(unlabeled_data["true_labels"]),
            "pseudo_label_accuracy_vs_hidden_labels": pseudo_label_accuracy,
        },
        "train": {
            "teacher_runtime": float(teacher_train_result.metrics.get("train_runtime", 0.0)),
            "student_runtime": float(student_train_result.metrics.get("train_runtime", 0.0)),
            "labeled_observed_examples": sum(1 for observed in splits["train"].observed_mask if observed),
            "labeled_total_examples": len(splits["train"].observed_mask),
            "group_summary": train_summary,
            "effective_assumed_observation_rate": effective_assumed_observation_rate,
        },
        "teacher": {
            "val": teacher_evaluated_splits["val"][0],
            "test": teacher_evaluated_splits["test"][0],
        },
        "student": {
            "val": student_evaluated_splits["val"][0],
            "test": student_evaluated_splits["test"][0],
        },
    }
    metrics_path = output_dir / "semi_supervised_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")

    print("teacher " + format_split_metrics("val", teacher_evaluated_splits["val"][1]))
    print("teacher " + format_split_metrics("test", teacher_evaluated_splits["test"][1]))
    print("student " + format_split_metrics("val", student_evaluated_splits["val"][1]))
    print("student " + format_split_metrics("test", student_evaluated_splits["test"][1]))
    print(
        "pseudo-labels: "
        f"{len(selected_indices)}/{len(unlabeled_true_labels)} selected "
        f"({metrics_payload['semi_supervised']['pseudo_selection_rate']:.4f}), "
        f"agreement={pseudo_label_accuracy:.4f}"
    )
    return metrics_payload


def load_unlabeled_for_pseudo_labels(
    *,
    config: CivilCommentsExperimentConfig,
    fraction: float,
    max_examples: int | None,
    download: bool,
) -> dict[str, Any]:
    deps = require_wilds_dependencies()
    AutoTokenizer = deps["AutoTokenizer"]
    get_dataset = deps["get_dataset"]

    unlabeled_dataset = get_dataset(
        "civilcomments",
        root_dir=config.dataset_root,
        download=download,
        unlabeled=True,
    )
    subset = unlabeled_dataset.get_subset("extra_unlabeled", frac=fraction)
    metadata_fields = list(subset._metadata_fields)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    example_count = len(subset) if max_examples is None else min(len(subset), max_examples)
    texts: list[str] = []
    true_labels: list[int] = []
    metadata_rows: list[list[int]] = []
    group_memberships: list[list[str]] = []
    for index in range(example_count):
        item = subset[index]
        if len(item) == 3:
            text, label, metadata = item
            true_label = label.item() if hasattr(label, "item") else label
        elif len(item) == 2:
            text, metadata = item
            true_label = metadata[-1]
        else:
            raise ValueError(f"unexpected unlabeled sample shape: {len(item)}")
        texts.append(text)
        metadata_row = [_coerce_int(value) for value in metadata]
        true_labels.append(_coerce_int(true_label))
        metadata_rows.append(metadata_row)
        group_memberships.append(
            extract_training_group_memberships(metadata_row, metadata_fields)
        )

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=config.max_length,
    )
    predict_dataset = TokenizedCivilCommentsDataset(
        encodings=encodings,
        labels=[0 for _ in range(example_count)],
        group_memberships=group_memberships or [[NON_IDENTITY_GROUP] for _ in range(example_count)],
        observed_mask=[True for _ in range(example_count)],
    )
    return {
        "encodings": encodings,
        "group_memberships": group_memberships,
        "predict_dataset": predict_dataset,
        "true_labels": true_labels,
    }


def select_pseudo_labels(
    positive_scores: Sequence[float],
    *,
    threshold: float,
) -> dict[str, list[int]]:
    if not positive_scores:
        raise ValueError("positive_scores must contain at least one score.")
    if not 0.5 <= threshold < 1.0:
        raise ValueError("threshold must be in [0.5, 1.0).")

    indices: list[int] = []
    labels: list[int] = []
    for index, score in enumerate(positive_scores):
        score_value = float(score)
        if score_value >= threshold:
            indices.append(index)
            labels.append(1)
        elif score_value <= 1.0 - threshold:
            indices.append(index)
            labels.append(0)
    return {
        "indices": indices,
        "labels": labels,
    }


def build_pseudo_labeled_dataset(
    *,
    encodings: Mapping[str, Sequence[Any]],
    group_memberships: Sequence[Sequence[str]],
    selected_indices: Sequence[int],
    pseudo_labels: Sequence[int],
) -> TokenizedCivilCommentsDataset:
    if len(selected_indices) != len(pseudo_labels):
        raise ValueError("selected_indices and pseudo_labels must have the same length.")
    selected_encodings = {
        key: [values[index] for index in selected_indices]
        for key, values in encodings.items()
    }
    selected_memberships = [
        list(group_memberships[index]) if group_memberships[index] else [NON_IDENTITY_GROUP]
        for index in selected_indices
    ]
    return TokenizedCivilCommentsDataset(
        encodings=selected_encodings,
        labels=[int(label) for label in pseudo_labels],
        group_memberships=selected_memberships,
        observed_mask=[True] * len(selected_indices),
    )


def _maybe_build_minimax_config(
    config: CivilCommentsExperimentConfig,
    *,
    train_split: Any,
) -> tuple[MinimaxHFConfig | None, float | None]:
    if config.method in {"robust_group", "robust_auto_v1"}:
        return _build_minimax_config(config, train_split=train_split)
    return None, None


def _build_trainer(
    *,
    config: CivilCommentsExperimentConfig,
    trainer_cls: Any,
    model: Any,
    args: Any,
    train_dataset: Any,
    eval_dataset: Any,
    data_collator: Any,
    minimax_config: MinimaxHFConfig | None,
) -> Any:
    if config.method in {"robust_group", "robust_auto_v1"}:
        return MinimaxTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            minimax_config=minimax_config,
        )
    if config.method == "erm":
        return trainer_cls(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
    raise ValueError(f"unsupported method: {config.method}")


def _accuracy(predicted_labels: Sequence[int], true_labels: Sequence[int]) -> float:
    if len(predicted_labels) != len(true_labels):
        raise ValueError("predicted_labels and true_labels must have the same length.")
    if not predicted_labels:
        raise ValueError("at least one prediction is required.")
    return sum(int(int(pred) == int(true)) for pred, true in zip(predicted_labels, true_labels)) / len(
        predicted_labels
    )


def _coerce_int(value: Any) -> int:
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_semi_supervised_experiment(
        config_path=args.config,
        output_root=args.output_root,
        unlabeled_fraction=args.unlabeled_fraction,
        max_unlabeled_examples=args.max_unlabeled_examples,
        pseudo_label_threshold=args.pseudo_label_threshold,
        student_num_train_epochs=args.student_num_train_epochs,
        download_unlabeled=args.download_unlabeled,
    )


if __name__ == "__main__":
    main()
