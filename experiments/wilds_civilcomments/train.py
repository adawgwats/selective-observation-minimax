from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import random
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from minimax_hf import MinimaxHFConfig, MinimaxTrainer

from experiments.wilds_civilcomments.common import (
    CivilCommentsExperimentConfig,
    config_to_dict,
    load_experiment_config,
)
from experiments.wilds_civilcomments.dataset import (
    TrainerDependencyError,
    build_training_group_summary,
    load_civilcomments_splits,
)
from experiments.wilds_civilcomments.metrics import (
    compute_civilcomments_metrics,
    format_split_metrics,
    logits_to_predictions_and_scores,
    metrics_to_dict,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ERM or selective-observation minimax on WILDS CivilComments."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML or JSON experiment config.",
    )
    return parser.parse_args(argv)


def train_from_config(config: CivilCommentsExperimentConfig) -> dict[str, Any]:
    deps = _require_transformers()
    Trainer = deps["Trainer"]
    TrainingArguments = deps["TrainingArguments"]
    AutoModelForSequenceClassification = deps["AutoModelForSequenceClassification"]
    set_seed = deps["set_seed"]

    set_seed(config.seed)
    random.seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wilds_dataset, splits, collator = load_civilcomments_splits(config)
    train_summary = build_training_group_summary(splits["train"])

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
    )
    training_args = _build_training_arguments(
        TrainingArguments,
        output_dir=str(output_dir),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        seed=config.seed,
        remove_unused_columns=False,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        report_to=[],
    )

    trainer: Any
    if config.method == "robust_group":
        trainer = MinimaxTrainer(
            model=model,
            args=training_args,
            train_dataset=splits["train"].dataset,
            eval_dataset=splits["val"].dataset,
            data_collator=collator,
            minimax_config=MinimaxHFConfig(
                group_key="group_id",
                observed_key="label_observed",
            ),
        )
    elif config.method == "erm":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=splits["train"].dataset,
            eval_dataset=splits["val"].dataset,
            data_collator=collator,
        )
    else:
        raise ValueError(f"unsupported method: {config.method}")

    train_result = trainer.train()
    trainer.save_model(str(output_dir / "checkpoint-final"))

    evaluated_splits = {
        split_name: evaluate_split(
            trainer=trainer,
            split=splits[split_name],
            wilds_dataset=wilds_dataset,
        )
        for split_name in ("val", "test")
    }

    metrics_payload = {
        "config": config_to_dict(config),
        "train": {
            "runtime": float(train_result.metrics.get("train_runtime", 0.0)),
            "observed_examples": sum(1 for observed in splits["train"].observed_mask if observed),
            "total_examples": len(splits["train"].observed_mask),
            "group_summary": train_summary,
        },
        "val": evaluated_splits["val"][0],
        "test": evaluated_splits["test"][0],
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(format_split_metrics("val", evaluated_splits["val"][1]))
    print(format_split_metrics("test", evaluated_splits["test"][1]))
    return metrics_payload


def evaluate_split(*, trainer: Any, split: Any, wilds_dataset: Any) -> tuple[dict[str, Any], Any]:
    prediction_output = trainer.predict(split.dataset)
    logits = prediction_output.predictions.tolist()
    predicted_labels, positive_scores = logits_to_predictions_and_scores(logits)
    local_metrics = compute_civilcomments_metrics(
        labels=split.labels,
        predicted_labels=predicted_labels,
        positive_scores=positive_scores,
        metadata_rows=split.metadata_rows,
        metadata_fields=split.metadata_fields,
    )

    results = metrics_to_dict(local_metrics)

    if hasattr(wilds_dataset, "eval"):
        try:
            import torch

            wilds_results, _results_str = wilds_dataset.eval(
                y_pred=torch.tensor(predicted_labels, dtype=torch.long),
                y_true=torch.tensor(split.labels, dtype=torch.long),
                metadata=torch.tensor(split.metadata_rows, dtype=torch.long),
            )
            results["wilds_eval"] = _normalize_wilds_results(wilds_results)
        except Exception as error:  # pragma: no cover - defensive path around optional deps
            results["wilds_eval_error"] = str(error)
    return results, local_metrics


def _normalize_wilds_results(results: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in results.items():
        if hasattr(value, "item"):
            value = value.item()
        normalized[key] = value
    return normalized


def _require_transformers() -> dict[str, Any]:
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise TrainerDependencyError(
            "Training WILDS CivilComments experiments requires transformers. "
            "Install minimax-optimization[wilds]."
        ) from error

    return {
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def _build_training_arguments(TrainingArguments: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = "no"
    elif "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "no"
    return TrainingArguments(**kwargs)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_experiment_config(args.config)
    train_from_config(config)


if __name__ == "__main__":
    main()
