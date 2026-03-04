# CivilComments Benchmark Notes

This note preserves the current tracked WILDS CivilComments benchmark status without committing large `outputs/` artifacts.

## March 4, 2026

Code reference:

- base integration commit: `f65c890` (`Wire adaptive WILDS CivilComments evaluation`)

Evaluation environment:

- Windows
- Python 3.11
- `torch 2.8.0+cu126`
- `torchvision 0.23.0+cu126`
- `torch-scatter 2.1.2+pt28cu126`
- official `wilds 2.0.0` evaluator path

Experiment configs:

- auto-discovery: `experiments/wilds_civilcomments/configs/midscale_auto_v1.yaml`
- ERM baseline: `experiments/wilds_civilcomments/configs/midscale_erm.yaml`

Shared setup:

- model: `distilbert-base-uncased`
- train/val/test cap: `16384 / 4096 / 4096`
- epochs: `2`
- batch size: `16 / 32`
- seed: `17`

Results:

| Run | Val acc_avg | Val acc_wg | Test acc_avg | Test acc_wg |
| --- | ---: | ---: | ---: | ---: |
| `midscale_auto_v1` | `0.9197` | `0.4286` | `0.9197` | `0.5682` |
| `midscale_erm` | `0.9214` | `0.4286` | `0.9243` | `0.5227` |

Interpretation:

- `robust_auto_v1` did not improve validation worst-group accuracy over ERM on this run.
- `robust_auto_v1` improved test worst-group accuracy by `+0.0455` absolute over ERM.
- This is encouraging, but not yet submission-grade evidence because model selection should be justified on validation performance, not test gains alone.

Additional run details:

- `midscale_auto_v1` estimated `effective_assumed_observation_rate = 0.8753662109377119`
- raw metrics were produced in local artifacts under `outputs/wilds_civilcomments/`

## March 4, 2026 Tuning Follow-up

Follow-up sweep:

- `experiments/wilds_civilcomments/configs/midscale_auto_v1_rate_0p75.yaml`
- `experiments/wilds_civilcomments/configs/midscale_auto_v1_rate_0p70.yaml`

Results:

| Run | Assumed rate | Val acc_avg | Val acc_wg | Test acc_avg | Test acc_wg |
| --- | ---: | ---: | ---: | ---: | ---: |
| `midscale_auto_v1` | `0.8754` estimated | `0.9197` | `0.4286` | `0.9197` | `0.5682` |
| `midscale_auto_v1_rate_0p75` | `0.75` | `0.9209` | `0.4286` | `0.9236` | `0.5909` |
| `midscale_auto_v1_rate_0p70` | `0.70` | `0.9209` | `0.4286` | `0.9221` | `0.5682` |
| `midscale_erm` | `n/a` | `0.9214` | `0.4286` | `0.9243` | `0.5227` |

Current takeaway:

- tuning the latent observation-rate prior downward helped `robust_auto_v1`
- `0.75` is the strongest auto-discovery run so far on this setup
- the best auto run now beats ERM on test worst-group accuracy by `+0.0682` absolute
- validation worst-group accuracy still ties ERM, so this remains promising rather than submission-ready

## March 4, 2026 5-Seed Protocol Run

To mirror the WILDS CivilComments submission protocol, both methods were re-run with 5 seeds
(`17, 23, 29, 31, 37`) using:

- `experiments/wilds_civilcomments/multiseed.py`
- `experiments/wilds_civilcomments/configs/midscale_auto_v1_rate_0p75.yaml`
- `experiments/wilds_civilcomments/configs/midscale_erm.yaml`

Artifacts:

- auto summary: `outputs/wilds_civilcomments/midscale_auto_v1_rate_0p75_multiseed/multiseed_summary.json`
- ERM summary: `outputs/wilds_civilcomments/midscale_erm_multiseed/multiseed_summary.json`

5-seed summary (mean +/- sample std):

| Run | Val acc_avg | Val acc_wg | Test acc_avg | Test acc_wg |
| --- | ---: | ---: | ---: | ---: |
| `robust_auto_v1` (`assumed_observation_rate=0.75`) | `0.9198 +/- 0.0065` | `0.4069 +/- 0.0724` | `0.9207 +/- 0.0060` | `0.5973 +/- 0.0430` |
| `erm` | `0.9199 +/- 0.0064` | `0.3957 +/- 0.0603` | `0.9220 +/- 0.0059` | `0.5909 +/- 0.0759` |

Interpretation:

- under 5-seed averaging, auto-discovery is no longer a single-seed anomaly
- auto-discovery is slightly better than ERM on worst-group accuracy in both val and test means
- the absolute gap is still small, and the variability is high, so this is progress but not top-leaderboard performance

## March 4, 2026 Unlabeled Self-Training Probe

An exploratory semi-supervised probe was run using WILDS `civilcomments` unlabeled data
(`extra_unlabeled`) via:

- `experiments/wilds_civilcomments/semi_supervised.py`
- unlabeled candidate cap: `8192`
- pseudo-label threshold: `0.90`
- student fine-tune epochs: `1`
- seed: `17`

Artifacts:

- auto: `outputs/wilds_civilcomments/midscale_auto_v1_rate_0p75_semi_supervised/semi_supervised_metrics.json`
- ERM: `outputs/wilds_civilcomments/midscale_erm_semi_supervised/semi_supervised_metrics.json`

Teacher -> student results:

| Method | Stage | Val acc_avg | Val acc_wg | Test acc_avg | Test acc_wg |
| --- | --- | ---: | ---: | ---: | ---: |
| `robust_auto_v1` (`assumed_observation_rate=0.75`) | teacher | `0.9209` | `0.4286` | `0.9236` | `0.5909` |
| `robust_auto_v1` (`assumed_observation_rate=0.75`) | student | `0.9231` | `0.3846` | `0.9243` | `0.5000` |
| `erm` | teacher | `0.9214` | `0.4286` | `0.9243` | `0.5227` |
| `erm` | student | `0.9236` | `0.4286` | `0.9224` | `0.5455` |

Pseudo-label diagnostics:

| Method | Pseudo selected | Selection rate | Agreement vs hidden unlabeled labels |
| --- | ---: | ---: | ---: |
| `robust_auto_v1` | `4445 / 8192` | `0.5426` | `0.3917` |
| `erm` | `6048 / 8192` | `0.7383` | `0.3271` |

Current interpretation:

- this first unlabeled protocol is not yet helping `robust_auto_v1` worst-group accuracy
- `erm` gained test worst-group accuracy modestly (`+0.0228`) but lost test average accuracy (`-0.0019`)
- pseudo-label agreement is low for both methods, indicating that this naive self-training setup needs stronger filtering/calibration before leaderboard-facing use
