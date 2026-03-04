from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Protocol

from minimax_core import (
    AutoDiscoveryObservationAdversary,
    GroupSnapshot,
    KnightianObservationAdversary,
    ScoreBasedObservationAdversary,
    SelectiveObservationAdversary,
    StructuralBreakObservationAdversary,
    SurpriseDrivenObservationAdversary,
    TimeVaryingObservationAdversary,
    compute_score_based_weights,
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


def _empirical_observation_rate(observed_mask: list[bool]) -> float:
    if not observed_mask:
        raise ValueError("observed_mask must contain at least one element.")
    return sum(1.0 if observed else 0.0 for observed in observed_mask) / len(observed_mask)


def _apply_online_mnar_assumption(
    snapshot: GroupSnapshot,
    observed_mask: list[bool],
    config: MinimaxHFConfig,
) -> GroupSnapshot:
    if not config.online_mnar:
        return snapshot
    assumed_rate = config.assumed_observation_rate or _empirical_observation_rate(observed_mask)
    return replace(snapshot, observation_rate=assumed_rate)


def _required_metadata_keys(config: MinimaxHFConfig) -> tuple[str, ...]:
    if config.uncertainty_mode == "time_varying":
        return (config.time_key,)
    if config.uncertainty_mode in {"knightian", "surprise"}:
        return (config.time_key, config.history_key)
    if config.uncertainty_mode == "structural_break":
        return (config.time_key, config.history_key, config.path_key)
    return ()


def _build_adversary(config: MinimaxHFConfig) -> Any:
    if config.uncertainty_mode == "group":
        return SelectiveObservationAdversary(config.q1)
    if config.uncertainty_mode == "score":
        return ScoreBasedObservationAdversary(config.q1)
    if config.uncertainty_mode == "time_varying":
        return TimeVaryingObservationAdversary(config.q1)
    if config.uncertainty_mode == "knightian":
        return KnightianObservationAdversary(config.q1)
    if config.uncertainty_mode == "surprise":
        return SurpriseDrivenObservationAdversary(config.q1)
    if config.uncertainty_mode == "structural_break":
        return StructuralBreakObservationAdversary(config.q1)
    if config.uncertainty_mode == "adaptive_v1":
        return AutoDiscoveryObservationAdversary(config.q1)
    raise ValueError(f"unsupported uncertainty_mode: {config.uncertainty_mode}")


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
            self._adversary = _build_adversary(self.minimax_config)
            validate_dataset_columns(
                self.train_dataset,
                group_key=self.minimax_config.group_key,
                observed_key=self.minimax_config.observed_key,
                require_observed_key=self.minimax_config.require_observed_key,
                extra_required_keys=_required_metadata_keys(self.minimax_config),
            )
            validate_dataset_columns(
                self.eval_dataset,
                group_key=self.minimax_config.group_key,
                observed_key=self.minimax_config.observed_key,
                require_observed_key=self.minimax_config.require_observed_key,
                extra_required_keys=_required_metadata_keys(self.minimax_config),
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
            time_indices_raw = model_inputs.pop(self.minimax_config.time_key, None)
            history_scores_raw = model_inputs.pop(self.minimax_config.history_key, None)
            path_ids_raw = model_inputs.pop(self.minimax_config.path_key, None)
            time_indices = (
                _normalize_metadata(time_indices_raw)
                if time_indices_raw is not None
                and self.minimax_config.uncertainty_mode in {"time_varying", "knightian", "surprise", "structural_break"}
                else None
            )
            history_scores = (
                _normalize_metadata(history_scores_raw)
                if history_scores_raw is not None
                and self.minimax_config.uncertainty_mode in {"knightian", "surprise", "structural_break"}
                else None
            )
            path_ids = (
                _normalize_metadata(path_ids_raw)
                if path_ids_raw is not None and self.minimax_config.uncertainty_mode == "structural_break"
                else None
            )

            labels = model_inputs.get("labels")
            outputs = model(**model_inputs)
            per_example_losses = self.loss_adapter(outputs, labels)
            detached_losses = per_example_losses.detach().cpu().tolist()

            bool_observed_mask = [bool(item) for item in observed_mask]
            if self.minimax_config.uncertainty_mode == "group":
                snapshot = estimate_group_snapshot(
                    losses=detached_losses,
                    group_ids=group_ids,
                    observed_mask=bool_observed_mask,
                )
                snapshot = _apply_online_mnar_assumption(
                    snapshot,
                    bool_observed_mask,
                    self.minimax_config,
                )
                q_values = (
                    self._adversary.update(snapshot)
                    if model.training
                    else self._adversary.current_q(snapshot)
                )
                example_weights = compute_example_weights(
                    snapshot=snapshot,
                    group_ids=group_ids,
                    observed_mask=bool_observed_mask,
                    q_values=q_values,
                )
            else:
                observation_rate = self.minimax_config.assumed_observation_rate or _empirical_observation_rate(
                    bool_observed_mask
                )
                if self.minimax_config.uncertainty_mode == "score":
                    q_values = (
                        self._adversary.update(detached_losses, observation_rate)
                        if model.training
                        else self._adversary.current_q(detached_losses, observation_rate)
                    )
                elif self.minimax_config.uncertainty_mode == "adaptive_v1":
                    q_values = (
                        self._adversary.update(detached_losses, observation_rate, bool_observed_mask)
                        if model.training
                        else self._adversary.current_q(detached_losses, observation_rate, bool_observed_mask)
                    )
                elif self.minimax_config.uncertainty_mode == "time_varying":
                    if time_indices is None:
                        raise ValueError("time metadata is required for time_varying uncertainty.")
                    q_values = (
                        self._adversary.update(detached_losses, observation_rate, [int(v) for v in time_indices])
                        if model.training
                        else self._adversary.current_q(detached_losses, observation_rate, [int(v) for v in time_indices])
                    )
                elif self.minimax_config.uncertainty_mode == "structural_break":
                    if time_indices is None or history_scores is None or path_ids is None:
                        raise ValueError(
                            "time, history, and path metadata are required for structural_break uncertainty."
                        )
                    time_values = [int(v) for v in time_indices]
                    history_values = [float(v) for v in history_scores]
                    path_values = list(path_ids)
                    q_values = (
                        self._adversary.update(
                            detached_losses,
                            observation_rate,
                            time_values,
                            history_values,
                            path_values,
                        )
                        if model.training
                        else self._adversary.current_q(
                            detached_losses,
                            observation_rate,
                            time_values,
                            history_values,
                            path_values,
                        )
                    )
                else:
                    if time_indices is None or history_scores is None:
                        raise ValueError("time and history metadata are required for Knightian-style uncertainty.")
                    time_values = [int(v) for v in time_indices]
                    history_values = [float(v) for v in history_scores]
                    q_values = (
                        self._adversary.update(detached_losses, observation_rate, time_values, history_values)
                        if model.training
                        else self._adversary.current_q(detached_losses, observation_rate, time_values, history_values)
                    )
                example_weights = compute_score_based_weights(
                    bool_observed_mask,
                    q_values,
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
