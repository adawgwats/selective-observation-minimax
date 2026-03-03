from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

from experiments.wilds_civilcomments.common import IDENTITY_FIELDS, metadata_row_to_dict


@dataclass(frozen=True)
class CivilCommentsSplitMetrics:
    overall_accuracy: float
    overall_auroc: float | None
    worst_group_accuracy: float | None
    worst_group_auroc: float | None
    group_accuracy: dict[str, float]
    group_accuracy_counts: dict[str, int]
    group_auroc: dict[str, float]
    group_auroc_counts: dict[str, int]


def logits_to_predictions_and_scores(
    logits: Sequence[Sequence[float]] | Sequence[float],
) -> tuple[list[int], list[float]]:
    if not logits:
        return [], []

    first = logits[0]
    if isinstance(first, (list, tuple)):
        positive_scores: list[float] = []
        predicted_labels: list[int] = []
        for row in logits:
            if len(row) == 1:
                positive_score = _sigmoid(float(row[0]))
                predicted_label = int(positive_score >= 0.5)
            else:
                probabilities = _softmax([float(value) for value in row])
                positive_score = probabilities[-1]
                predicted_label = int(max(range(len(probabilities)), key=probabilities.__getitem__))
            positive_scores.append(positive_score)
            predicted_labels.append(predicted_label)
        return predicted_labels, positive_scores

    positive_scores = [_sigmoid(float(value)) for value in logits]
    predicted_labels = [int(score >= 0.5) for score in positive_scores]
    return predicted_labels, positive_scores


def compute_civilcomments_metrics(
    labels: Sequence[int],
    predicted_labels: Sequence[int],
    positive_scores: Sequence[float],
    metadata_rows: Sequence[Sequence[Any] | Mapping[str, Any]],
    metadata_fields: Sequence[str],
) -> CivilCommentsSplitMetrics:
    if not (len(labels) == len(predicted_labels) == len(positive_scores) == len(metadata_rows)):
        raise ValueError("labels, predicted_labels, positive_scores, and metadata_rows must align.")
    if not labels:
        raise ValueError("at least one evaluation example is required.")

    label_ints = [int(label) for label in labels]
    pred_ints = [int(label) for label in predicted_labels]
    scores = [float(score) for score in positive_scores]
    metadata_dicts = [
        metadata_row_to_dict(metadata_row, metadata_fields)
        for metadata_row in metadata_rows
    ]

    group_accuracy: dict[str, float] = {}
    group_accuracy_counts: dict[str, int] = {}
    worst_group_accuracy: float | None = None

    for identity in IDENTITY_FIELDS:
        for label_value in (0, 1):
            indices = [
                index
                for index, metadata in enumerate(metadata_dicts)
                if metadata.get(identity, 0) == 1 and label_ints[index] == label_value
            ]
            if not indices:
                continue
            group_name = f"{identity}:1,y:{label_value}"
            accuracy = _accuracy(
                [label_ints[index] for index in indices],
                [pred_ints[index] for index in indices],
            )
            group_accuracy[group_name] = accuracy
            group_accuracy_counts[group_name] = len(indices)
            if worst_group_accuracy is None or accuracy < worst_group_accuracy:
                worst_group_accuracy = accuracy

    group_auroc: dict[str, float] = {}
    group_auroc_counts: dict[str, int] = {}
    worst_group_auroc: float | None = None
    for identity in IDENTITY_FIELDS:
        indices = [
            index
            for index, metadata in enumerate(metadata_dicts)
            if metadata.get(identity, 0) == 1
        ]
        if not indices:
            continue
        group_labels = [label_ints[index] for index in indices]
        group_scores = [scores[index] for index in indices]
        group_auc = binary_auroc(group_labels, group_scores)
        if group_auc is None:
            continue
        group_auroc[identity] = group_auc
        group_auroc_counts[identity] = len(indices)
        if worst_group_auroc is None or group_auc < worst_group_auroc:
            worst_group_auroc = group_auc

    return CivilCommentsSplitMetrics(
        overall_accuracy=_accuracy(label_ints, pred_ints),
        overall_auroc=binary_auroc(label_ints, scores),
        worst_group_accuracy=worst_group_accuracy,
        worst_group_auroc=worst_group_auroc,
        group_accuracy=group_accuracy,
        group_accuracy_counts=group_accuracy_counts,
        group_auroc=group_auroc,
        group_auroc_counts=group_auroc_counts,
    )


def metrics_to_dict(metrics: CivilCommentsSplitMetrics) -> dict[str, Any]:
    return asdict(metrics)


def format_split_metrics(split_name: str, metrics: CivilCommentsSplitMetrics) -> str:
    worst_group_accuracy = (
        "n/a" if metrics.worst_group_accuracy is None else f"{metrics.worst_group_accuracy:.4f}"
    )
    overall_auroc = "n/a" if metrics.overall_auroc is None else f"{metrics.overall_auroc:.4f}"
    worst_group_auroc = (
        "n/a" if metrics.worst_group_auroc is None else f"{metrics.worst_group_auroc:.4f}"
    )
    return (
        f"{split_name}: "
        f"accuracy={metrics.overall_accuracy:.4f}, "
        f"worst_group_accuracy={worst_group_accuracy}, "
        f"auroc={overall_auroc}, "
        f"worst_group_auroc={worst_group_auroc}"
    )


def binary_auroc(labels: Sequence[int], positive_scores: Sequence[float]) -> float | None:
    if len(labels) != len(positive_scores):
        raise ValueError("labels and positive_scores must have the same length.")
    if not labels:
        raise ValueError("at least one example is required.")

    positives = sum(1 for label in labels if int(label) == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    ordered_pairs = sorted(
        ((float(score), int(label)) for label, score in zip(labels, positive_scores)),
        key=lambda item: item[0],
    )
    positive_rank_sum = 0.0
    rank = 1
    index = 0
    while index < len(ordered_pairs):
        tie_end = index + 1
        while tie_end < len(ordered_pairs) and ordered_pairs[tie_end][0] == ordered_pairs[index][0]:
            tie_end += 1
        tie_count = tie_end - index
        average_rank = (2 * rank + tie_count - 1) / 2.0
        positive_count = sum(label for _score, label in ordered_pairs[index:tie_end])
        positive_rank_sum += average_rank * positive_count
        rank += tie_count
        index = tie_end

    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _accuracy(labels: Sequence[int], predicted_labels: Sequence[int]) -> float:
    if len(labels) != len(predicted_labels):
        raise ValueError("labels and predicted_labels must have the same length.")
    return sum(int(label == prediction) for label, prediction in zip(labels, predicted_labels)) / len(labels)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        denominator = 1.0 + math.exp(-value)
        return 1.0 / denominator
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _softmax(values: Sequence[float]) -> list[float]:
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]
