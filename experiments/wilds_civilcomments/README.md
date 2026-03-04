# WILDS CivilComments Experiments

This directory contains the CivilComments-specific experiment layer that sits on top of the core selective-observation package.

Current scope:

- overlapping identity groups are represented as multi-membership `group_id` values
- explicit MNAR masking utilities are available for the `train` split
- configuration files are tracked alongside the experiment
- `train.py` and `eval.py` provide the first runnable CivilComments experiment entrypoints
- training configs can disable checkpoint writes with `save_strategy: "no"` and `save_final_checkpoint: false`

The intended experiment tracks are:

- `vanilla`: original WILDS supervision
- `explicit_mnar`: shared synthetic MNAR masking on the training split
- `latent_mnar`: unchanged training data with an internal latent-missingness adversary
- `adaptive_v1`: versioned auto-discovery ambiguity updates from the score stream, without custom dataset metadata

Install dependencies with:

```bash
pip install ".[wilds]"
```

Run training with:

```bash
python experiments/wilds_civilcomments/train.py --config experiments/wilds_civilcomments/configs/base_erm.yaml
python experiments/wilds_civilcomments/train.py --config experiments/wilds_civilcomments/configs/robust_group.yaml
python experiments/wilds_civilcomments/train.py --config experiments/wilds_civilcomments/configs/pilot_auto_v1.yaml
```

Run leaderboard-style 5-seed sweeps with:

```bash
python experiments/wilds_civilcomments/multiseed.py --config experiments/wilds_civilcomments/configs/midscale_auto_v1_rate_0p75.yaml
python experiments/wilds_civilcomments/multiseed.py --config experiments/wilds_civilcomments/configs/midscale_erm.yaml
```

Each sweep writes a `multiseed_summary.json` artifact under `<config.output_dir>_multiseed/`.

Evaluate a saved checkpoint with:

```bash
python experiments/wilds_civilcomments/eval.py --config experiments/wilds_civilcomments/configs/robust_group.yaml --checkpoint outputs/wilds_civilcomments/robust_group/checkpoint-final --split val
```

Compare saved metrics artifacts with:

```bash
python experiments/wilds_civilcomments/report.py --metrics outputs/wilds_civilcomments/smoke_erm/metrics.json outputs/wilds_civilcomments/smoke_robust_group/metrics.json
```

Tracked benchmark summaries live in `experiments/wilds_civilcomments/RESULTS.md`.
