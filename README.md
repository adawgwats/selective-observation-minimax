# Minimax Optimization

`minimax-optimization` packages Christensen-consistent robustness under selective observation and exposes a thin Hugging Face adapter.

## Hugging Face quickstart

HF users should only need to do three things:

1. Add a `group_id` column to the dataset.
2. Add a `label_observed` column if labels can be missing or censored.
3. Swap `Trainer` for `MinimaxTrainer`.

```python
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

from minimax_hf import MinimaxHFConfig, MinimaxTrainer

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=8,
    remove_unused_columns=True,  # MinimaxTrainer will disable this automatically.
)

def preprocess(example):
    encoded = tokenizer(example["text"], truncation=True)
    encoded["labels"] = example["label"]
    encoded["group_id"] = example["region"]
    encoded["label_observed"] = example.get("label_observed", True)
    return encoded

train_dataset = Dataset.from_list(raw_train_examples).map(preprocess)
eval_dataset = Dataset.from_list(raw_eval_examples).map(preprocess)

trainer = MinimaxTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    minimax_config=MinimaxHFConfig(
        group_key="group_id",
        observed_key="label_observed",
    ),
)

trainer.train()
```

## What the adapter does for you

- wraps the active HF data collator so `group_id` and `label_observed` survive batching
- disables `remove_unused_columns` automatically so HF does not strip minimax metadata
- validates that `group_id` exists on the train and eval datasets before training starts
- includes built-in loss adapters for sequence classification, regression, and token classification

## Built-in task support

Set `task_type` on `MinimaxHFConfig` when you are not doing sequence classification:

```python
from minimax_hf import MinimaxHFConfig

config = MinimaxHFConfig(task_type="regression")
```

Supported values:

- `sequence_classification`
- `regression`
- `token_classification`

## When you still need a custom loss adapter

Pass `loss_adapter=` when:

- your model output does not expose `logits`
- you need a nonstandard reduction
- you are training a task outside the three built-in adapters

The loss adapter contract is:

```python
def loss_adapter(outputs, labels):
    # return one loss per example, not a reduced scalar
    return per_example_losses
```

## Dataset contract

Required:

- `group_id`: metadata used by the adversary

Optional:

- `label_observed`: boolean flag for whether the label was observed

If `label_observed` is omitted, the trainer assumes all labels are observed.

## Current scope

`v0` implements one adversary family:

- selective observation / non-ignorable missingness

The long-term architecture is broader, but the current implemented HF method is intentionally narrow.
