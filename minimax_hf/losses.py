from __future__ import annotations

from typing import Any, Callable, Literal


TaskType = Literal["sequence_classification", "regression", "token_classification"]


def _extract_logits(outputs: Any) -> Any:
    return outputs.logits if hasattr(outputs, "logits") else outputs["logits"]


def sequence_classification_loss_adapter(outputs: Any, labels: Any) -> Any:
    import torch.nn.functional as functional

    if labels is None:
        raise ValueError("labels are required for sequence classification.")

    logits = _extract_logits(outputs)
    return functional.cross_entropy(logits, labels, reduction="none")


def regression_loss_adapter(outputs: Any, labels: Any) -> Any:
    if labels is None:
        raise ValueError("labels are required for regression.")

    logits = _extract_logits(outputs)
    predictions = logits.squeeze(-1) if hasattr(logits, "squeeze") else logits
    targets = labels.squeeze(-1) if hasattr(labels, "squeeze") else labels
    errors = predictions - targets
    return errors * errors


def token_classification_loss_adapter(
    outputs: Any,
    labels: Any,
    *,
    ignore_index: int = -100,
) -> Any:
    import torch
    import torch.nn.functional as functional

    if labels is None:
        raise ValueError("labels are required for token classification.")

    logits = _extract_logits(outputs)
    batch_size, sequence_length, num_labels = logits.shape
    flat_losses = functional.cross_entropy(
        logits.view(batch_size * sequence_length, num_labels),
        labels.view(batch_size * sequence_length),
        reduction="none",
        ignore_index=ignore_index,
    ).view(batch_size, sequence_length)

    valid_mask = (labels != ignore_index).to(flat_losses.dtype)
    valid_counts = valid_mask.sum(dim=1).clamp_min(1.0)
    return (flat_losses * valid_mask).sum(dim=1) / valid_counts


def build_loss_adapter(
    task_type: TaskType,
    *,
    token_ignore_index: int = -100,
) -> Callable[[Any, Any], Any]:
    if task_type == "sequence_classification":
        return sequence_classification_loss_adapter
    if task_type == "regression":
        return regression_loss_adapter
    if task_type == "token_classification":
        return lambda outputs, labels: token_classification_loss_adapter(
            outputs,
            labels,
            ignore_index=token_ignore_index,
        )
    raise ValueError(f"unsupported task_type: {task_type}")
