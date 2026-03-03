from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.wilds_civilcomments.common import CivilCommentsExperimentConfig, load_experiment_config
from experiments.wilds_civilcomments.dataset import TrainerDependencyError, load_civilcomments_splits
from experiments.wilds_civilcomments.metrics import (
    compute_civilcomments_metrics,
    format_split_metrics,
    logits_to_predictions_and_scores,
    metrics_to_dict,
)
from experiments.wilds_civilcomments.train import _normalize_wilds_results, _require_transformers


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained WILDS CivilComments checkpoint."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    return parser.parse_args(argv)


def evaluate_checkpoint(
    *,
    config: CivilCommentsExperimentConfig,
    checkpoint: str,
    split_name: str,
) -> tuple[dict[str, Any], Any]:
    deps = _require_transformers()
    AutoModelForSequenceClassification = deps["AutoModelForSequenceClassification"]
    Trainer = deps["Trainer"]
    TrainingArguments = deps["TrainingArguments"]

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wilds_dataset, splits, collator = load_civilcomments_splits(config)
    split = splits[split_name]
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir / "eval"),
            per_device_eval_batch_size=config.eval_batch_size,
            remove_unused_columns=False,
            report_to=[],
        ),
        data_collator=collator,
    )

    prediction_output = trainer.predict(split.dataset)
    logits = prediction_output.predictions.tolist()
    predicted_labels, positive_scores = logits_to_predictions_and_scores(logits)
    metrics = compute_civilcomments_metrics(
        labels=split.labels,
        predicted_labels=predicted_labels,
        positive_scores=positive_scores,
        metadata_rows=split.metadata_rows,
        metadata_fields=split.metadata_fields,
    )
    payload = metrics_to_dict(metrics)

    if hasattr(wilds_dataset, "eval"):
        try:
            import torch

            wilds_results, _results_str = wilds_dataset.eval(
                y_pred=torch.tensor(predicted_labels, dtype=torch.long),
                y_true=torch.tensor(split.labels, dtype=torch.long),
                metadata=torch.tensor(split.metadata_rows, dtype=torch.long),
            )
            payload["wilds_eval"] = _normalize_wilds_results(wilds_results)
        except Exception as error:  # pragma: no cover - defensive path around optional deps
            payload["wilds_eval_error"] = str(error)
    return payload, metrics


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_experiment_config(args.config)
    payload, metrics = evaluate_checkpoint(
        config=config,
        checkpoint=args.checkpoint,
        split_name=args.split,
    )
    output_path = Path(config.output_dir) / f"{args.split}_metrics.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(format_split_metrics(args.split, metrics))


if __name__ == "__main__":
    main()
