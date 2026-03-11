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

When the training set is survivor-biased and dropped rows are no longer explicitly marked, enable the first hidden-selection baseline directly in the HF adapter:

```python
trainer = MinimaxTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    minimax_config=MinimaxHFConfig(
        group_key="group_id",
        online_mnar=True,
        assumed_observation_rate=0.67,
    ),
)
```

When users do not have task-specific Knightian metadata but still want a versioned auto-discovery ambiguity controller, use `adaptive_v1` with an assumed observation-rate prior:

```python
trainer = MinimaxTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    minimax_config=MinimaxHFConfig(
        group_key="group_id",
        uncertainty_mode="adaptive_v1",
        assumed_observation_rate=0.85,
    ),
)
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

- `group_id`: metadata used by the adversary; this can be a single group id or a list of group ids for overlapping groups

Optional:

- `label_observed`: boolean flag for whether the label was observed

If `label_observed` is omitted, the trainer assumes all labels are observed.

## Current scope

`v0` still centers on selective-observation robustness, but the HF surface now exposes both explicit-metadata and auto-discovery variants:

- grouped selective observation / non-ignorable missingness
- score-based ambiguity from per-example losses
- `adaptive_v1`, a versioned auto-discovery controller that derives time/history-like ambiguity signals online from the score stream

The long-term architecture is broader, but the current implemented HF method is intentionally narrow.

## Uncertainty-set architecture

`minimax_core` now exposes explicit uncertainty-set abstractions so the Christensen-style selective-observation objective can be extended toward Knightian ambiguity without changing the HF surface area.

Available set families:

- `SelectiveObservationSet`: grouped observation ambiguity, the current default
- `ScoreBasedObservationSet`: per-example observation ambiguity from proxy scores
- `TimeVaryingObservationSet`: a time-indexed extension where later observations can be given a different ambiguity budget than earlier ones
- `KnightianObservationSet`: a history-aware extension where ambiguity can grow with both time and accumulated hidden/distress history
- `SurpriseDrivenObservationSet`: a surprise-aware extension that can amplify ambiguity after unexpected residual shocks

The grouped and score adversaries currently drive the main benchmarks. The time-varying, Knightian, and surprise-aware sets are the package-level seams for dynamic ambiguity over observation. `adaptive_v1` in the HF adapter uses those same ideas without requiring downstream users to supply custom time/history metadata.

The DSSAT benchmark also now includes a `robust_time_varying` baseline. It uses per-example time indices to let later observations carry a different ambiguity budget than earlier ones, which is useful when selection bias compounds over time.

The benchmark now also includes `robust_knightian`, which adds a simple path-history score on top of the time index so later examples with more cumulative distress or hidden outcomes can be treated as more ambiguous during training.

## Synthetic MNAR tooling

`minimax_core` now owns the reusable synthetic MNAR layer used by the agriculture benchmark.

Supported training-view modes:

- `explicit_missing`: keep rows in the dataset and mark labels as unobserved
- `drop_unobserved`: remove unobserved rows entirely, which simulates hidden sample-selection bias
- `truncate_after_unobserved`: drop the first unobserved row in a path and all later rows, which is a simple survivorship-bias / panel-attrition model

This lets downstream benchmarks separate two cases cleanly:

- MNAR is present and the learner is told which labels are missing
- MNAR is present but the learner only sees the survivor-biased dataset

The agriculture benchmark now also exposes a training-time baseline, `robust_group_online`, that keeps the visible dataset fixed but lets the adversary impose an assumed observation-rate prior during optimization. This is the first step toward handling hidden survivorship bias when the dataset does not carry an explicit missingness flag.

HF users can reuse the same MNAR machinery directly on generic record datasets:

```python
from minimax_core import SyntheticMNARConfig
from minimax_hf import build_synthetic_mnar_view

view = build_synthetic_mnar_view(
    raw_records,
    config=SyntheticMNARConfig(
        view_mode="drop_unobserved",
        distressed_penalty=0.60,
    ),
    label_key="labels",
    group_key="group_id",
    path_key="path_index",
    step_key="step_index",
    distressed_group_values=["south_region", "late_season_failure"],
)

train_dataset = Dataset.from_list(view.rows)
```

Notes:

- `explicit_missing` keeps rows and adds `label_observed`
- `drop_unobserved` hides dropped rows completely and is the closer survivorship-bias model
- `truncate_after_unobserved` needs a `path_key` because it drops the rest of a path after the first hidden failure

## DSSAT-backed agriculture benchmark

If `ag-survival-sim` is installed and DSSAT is available locally, you can run the benchmark-ready DSSAT crop bundles directly from this package:

```bash
minimax-ag-benchmark --benchmark georgia_soybean --trials 3 --train-paths 6 --test-paths 4 --horizon-years 4
```

That benchmark currently:

- pulls DSSAT-backed trajectory data from `ag-survival-sim`
- trains on selectively observed labels
- evaluates on held-out latent outcomes
- evaluates learned action-selection policies on paired DSSAT paths using survival, bankruptcy, and terminal-wealth metrics
- compares `ERM`, common reweighting baselines, `group_dro`, and the selective-observation minimax estimator
- supports the benchmark-ready DSSAT bundles currently exposed by `ag-survival-sim`
  - `iowa_maize`
  - `georgia_maize_management`
  - `georgia_soybean`
  - `kansas_wheat`
  - `dtsp_rice`
  - `georgia_peanut`
  - `uafd_sunflower`

To run all currently benchmark-ready DSSAT bundles in sequence:

```bash
minimax-ag-benchmark --all-benchmarks --trials 1 --train-paths 2 --test-paths 1 --horizon-years 2
```

To switch the agriculture benchmark into hidden sample-selection mode:

```bash
minimax-ag-benchmark --benchmark georgia_peanut --mnar-mode drop_unobserved
```

To enable an explicit assumed observation-rate prior for the online MNAR baseline:

```bash
minimax-ag-benchmark --benchmark georgia_peanut --mnar-mode drop_unobserved --assumed-observation-rate 0.67
```

The benchmark output now also includes static action references like `static_corn_low` and `static_corn_medium`. This makes it obvious when all learned estimators collapse to the same action policy even if their label-fit metrics differ.

Useful agriculture targets now include:

- `net_income`: one-step supervised target
- `survival_years`: remaining survival time from the current decision point
- `cumulative_profit_to_go`: suffix profit from the current decision point to the end of the simulated path

For example, the Georgia maize management bundle is now expressive enough to answer both:

- which policies stay in business longer than `ERM`
- which policies match or beat the strongest static competitor

The benchmark summary reports:

- mean survival and bankruptcy rate
- pathwise outlast rate versus `ERM`
- pathwise outlast rate versus the best static policy on the evaluation paths
- dominant action share, so it is easy to see whether a learned policy is genuinely dynamic or just replicating one static management choice

The current ag policy benchmark is also stricter than the first version:

- learned policies no longer get realized weather regime as a decision-time feature
- action features are benchmark-specific, so multi-action bundles are represented explicitly instead of collapsing into a single low/medium flag
- the simulator now fails obviously unaffordable actions, which creates a real interaction between farm balance sheet and policy choice
- learned policies now use decision-time price context (`t-1`, `t-2`, ... lags) with action-price interaction features, so crop choice can respond to changing price regimes

Install helper dependency:

```bash
pip install ".[ag]"
```

Optional price-dynamics backend (open source `statsmodels`, used when `--price-dynamics-model statsmodels_arima`):

```bash
pip install ".[ag,ag_price]"
```

Price knobs available on `minimax-ag-benchmark`:

- `--disable-price-features`
- `--price-history-lags 3`
- `--price-dynamics-model ema|autoregressive|statsmodels_arima`
- `--price-spot-weight 0.65`
- `--price-ema-alpha 0.35`
- `--use-fred-price-history`
- `--fred-price-lookback-years 100`
- `--fred-price-end-year 2025` (optional; defaults to latest completed year)
- `--fred-price-cache-dir data/fred_cache`

FRED history initialization only uses completed past years (for example, with current date March 4, 2026, default end year is 2025).

Current FRED crop proxies used by the simulator for historical initialization:

- corn: `WPU012202`
- soy: `WPU01830131`
- wheat: `WPU0121`
- rice: `WPU0123`
- peanut: `WPU01830111`
- sunflower: `WPU01830161`

Example with FRED-backed historical initialization:

```bash
minimax-ag-benchmark --benchmark georgia_soybean --trials 3 --train-paths 6 --test-paths 4 --horizon-years 4 --use-fred-price-history --fred-price-lookback-years 100
```

The turn-based agriculture game, browser UI, local API, and Godot prototype were extracted into the sibling repo `../minimax-ag-game` so this repository stays focused on the minimax estimator and benchmark code.
