from experiments.wilds_civilcomments.train import _build_training_arguments


class _LegacyTrainingArguments:
    def __init__(self, *, output_dir: str, eval_strategy: str, save_strategy: str) -> None:
        self.output_dir = output_dir
        self.eval_strategy = eval_strategy
        self.save_strategy = save_strategy


class _ModernTrainingArguments:
    def __init__(self, *, output_dir: str, evaluation_strategy: str, save_strategy: str) -> None:
        self.output_dir = output_dir
        self.evaluation_strategy = evaluation_strategy
        self.save_strategy = save_strategy


def test_build_training_arguments_supports_eval_strategy_name() -> None:
    args = _build_training_arguments(
        _LegacyTrainingArguments,
        output_dir="outputs",
        save_strategy="epoch",
    )

    assert args.eval_strategy == "no"


def test_build_training_arguments_supports_evaluation_strategy_name() -> None:
    args = _build_training_arguments(
        _ModernTrainingArguments,
        output_dir="outputs",
        save_strategy="epoch",
    )

    assert args.evaluation_strategy == "no"
