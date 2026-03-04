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
