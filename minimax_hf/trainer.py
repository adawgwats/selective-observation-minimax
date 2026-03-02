from __future__ import annotations

from typing import Any, Callable, Protocol

from minimax_core import (
    SelectiveObservationAdversary,
    compute_example_weights,
    estimate_group_snapshot,
)

from .config import MinimaxHFConfig
from .data import (
    MinimaxDataCollator,
    is_minimax_data_collator,
    prepare_training_args,
    validate_dataset_columns,
)
from .losses import build_loss_adapter

try:
    from transformers import Trainer
except ImportError:  # pragma: no cover - exercised by import guard tests only.
    Trainer = None


class TrainerImportError(ImportError):
    pass


class LossAdapter(Protocol):
    def __call__(self, outputs: Any, labels: Any) -> Any:
        ...


def _normalize_metadata(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    raise TypeError("metadata values must be tensors, lists, or tuples.")


if Trainer is None:

    class MinimaxTrainer:  # pragma: no cover - import guard only.
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise TrainerImportError(
                "transformers is required to use MinimaxTrainer. "
                "Install minimax-optimization[hf] to enable the Hugging Face adapter."
            )

else:

    class MinimaxTrainer(Trainer):
        def __init__(
            self,
            *args: Any,
            minimax_config: MinimaxHFConfig | None = None,
            loss_adapter: LossAdapter | None = None,
            **kwargs: Any,
        ) -> None:
            args = list(args)
            if len(args) >= 2:
                args[1] = prepare_training_args(args[1])
            if "args" in kwargs:
                kwargs["args"] = prepare_training_args(kwargs["args"])

            super().__init__(*args, **kwargs)
            self.minimax_config = minimax_config or MinimaxHFConfig()
            self.loss_adapter = loss_adapter or build_loss_adapter(
                self.minimax_config.task_type,
                token_ignore_index=self.minimax_config.token_ignore_index,
            )
            self._adversary = SelectiveObservationAdversary(self.minimax_config.q1)
            validate_dataset_columns(
                self.train_dataset,
                group_key=self.minimax_config.group_key,
                observed_key=self.minimax_config.observed_key,
                require_observed_key=self.minimax_config.require_observed_key,
            )
            validate_dataset_columns(
                self.eval_dataset,
                group_key=self.minimax_config.group_key,
                observed_key=self.minimax_config.observed_key,
                require_observed_key=self.minimax_config.require_observed_key,
            )
            if not is_minimax_data_collator(self.data_collator):
                self.data_collator = MinimaxDataCollator(
                    self.data_collator,
                    group_key=self.minimax_config.group_key,
                    observed_key=self.minimax_config.observed_key,
                )

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: int | None = None,
        ) -> Any:
            del num_items_in_batch

            model_inputs = dict(inputs)
            group_ids = _normalize_metadata(model_inputs.pop(self.minimax_config.group_key))
            observed_mask_raw = model_inputs.pop(self.minimax_config.observed_key, None)
            observed_mask = (
                _normalize_metadata(observed_mask_raw)
                if observed_mask_raw is not None
                else [True] * len(group_ids)
            )

            labels = model_inputs.get("labels")
            outputs = model(**model_inputs)
            per_example_losses = self.loss_adapter(outputs, labels)
            detached_losses = per_example_losses.detach().cpu().tolist()

            snapshot = estimate_group_snapshot(
                losses=detached_losses,
                group_ids=group_ids,
                observed_mask=[bool(item) for item in observed_mask],
            )
            q_values = (
                self._adversary.update(snapshot)
                if model.training
                else self._adversary.current_q(snapshot)
            )
            example_weights = compute_example_weights(
                snapshot=snapshot,
                group_ids=group_ids,
                observed_mask=[bool(item) for item in observed_mask],
                q_values=q_values,
            )
            weight_tensor = per_example_losses.new_tensor(example_weights)
            loss = (per_example_losses * weight_tensor).sum()
            if return_outputs:
                return loss, outputs
            return loss


def build_minimax_trainer(
    *,
    minimax_config: MinimaxHFConfig | None = None,
    loss_adapter: LossAdapter | None = None,
    **trainer_kwargs: Any,
) -> MinimaxTrainer:
    return MinimaxTrainer(
        minimax_config=minimax_config,
        loss_adapter=loss_adapter,
        **trainer_kwargs,
    )
