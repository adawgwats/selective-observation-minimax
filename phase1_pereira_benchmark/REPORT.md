# Phase 1 Report — Christensen Minimax vs MICE under MNAR Labels

**Path A benchmark**: Pereira et al. 2024 MNAR mechanisms applied to label column for regression tasks on 10 UCI medical datasets. See PROTOCOL.md for full spec and declared deviations from Pereira's original imputation-quality benchmark.

**Seeds completed**: 10 per cell (median across cells)
**Total rows**: 31,500
**Methods**: ['complete_case', 'erm_sgd', 'heckman', 'ipw_estimated', 'knn_impute', 'mean_impute', 'mice', 'minimax_score', 'oracle']
**Mechanisms**: ['MBIR_Bayesian', 'MBIR_Frequentist', 'MBOV_Centered', 'MBOV_Higher', 'MBOV_Lower', 'MBOV_Stochastic', 'MBUV']
**Datasets**: ['bc-coimbra', 'cleveland', 'cmc', 'ctg', 'pima', 'saheart', 'thyroid', 'transfusion', 'vertebral', 'wisconsin']

## Headline: Christensen minimax vs MICE

Across 350 (dataset, mechanism, rate) cells:

- **Wins** (95% CI strictly below MICE): **42** (12.0%)
- **Ties** (95% CIs overlap MICE): 137 (39.1%)
- **Losses** (95% CI strictly above MICE): 171 (48.9%)

## Headline: minimax vs ERM (same SGD engine, no adversary)

This is the apples-to-apples algorithm comparison — same SGD, same learning schedule, only difference is the adversary.

- Wins: **64** (18.3%)
- Ties: 198 (56.6%)
- Losses: 88 (25.1%)

## Win/loss vs MICE by MNAR mechanism

| mechanism | wins | ties | losses | total | win_rate | mean_mse_diff_pct |
| --- | --- | --- | --- | --- | --- | --- |
| MBIR_Bayesian | 1 | 21 | 28 | 50 | 0.020 | 18706.294 |
| MBIR_Frequentist | 1 | 21 | 28 | 50 | 0.020 | 18706.294 |
| MBOV_Centered | 6 | 17 | 27 | 50 | 0.120 | 9001.507 |
| MBOV_Higher | 3 | 33 | 14 | 50 | 0.060 | 8993.227 |
| MBOV_Lower | 19 | 9 | 22 | 50 | 0.380 | 3264.102 |
| MBOV_Stochastic | 12 | 14 | 24 | 50 | 0.240 | 13334.999 |
| MBUV | 0 | 22 | 28 | 50 | 0.000 | 12584.843 |

## Win/loss vs MICE by missing rate

| missing_rate_pct | wins | ties | losses | total | win_rate | mean_mse_diff_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 10.000 | 3.000 | 20.000 | 47.000 | 70.000 | 0.043 | 11908.751 |
| 20.000 | 7.000 | 22.000 | 41.000 | 70.000 | 0.100 | 13177.793 |
| 40.000 | 6.000 | 33.000 | 31.000 | 70.000 | 0.086 | 5959.059 |
| 60.000 | 10.000 | 32.000 | 28.000 | 70.000 | 0.143 | 10889.205 |
| 80.000 | 16.000 | 30.000 | 24.000 | 70.000 | 0.229 | 18487.526 |

## Win/loss vs MICE by dataset

| dataset | wins | ties | losses | total | win_rate | mean_mse_diff_pct |
| --- | --- | --- | --- | --- | --- | --- |
| bc-coimbra | 5 | 30 | 0 | 35 | 0.143 | -10.629 |
| cleveland | 5 | 30 | 0 | 35 | 0.143 | 2.283 |
| cmc | 4 | 9 | 22 | 35 | 0.114 | 6.470 |
| ctg | 7 | 0 | 28 | 35 | 0.200 | 120755.792 |
| pima | 2 | 7 | 26 | 35 | 0.057 | 19.089 |
| saheart | 5 | 18 | 12 | 35 | 0.143 | 6.860 |
| thyroid | 0 | 5 | 30 | 35 | 0.000 | 26.736 |
| transfusion | 1 | 9 | 25 | 35 | 0.029 | 18.837 |
| vertebral | 13 | 20 | 2 | 35 | 0.371 | -4.688 |
| wisconsin | 0 | 9 | 26 | 35 | 0.000 | 23.918 |

## Most favorable cells (minimax beats MICE by largest %)

| dataset | mechanism | missing_rate_pct | minimax_mse | mice_mse | diff_% |
| --- | --- | --- | --- | --- | --- |
| vertebral | MBIR_Frequentist | 80.000 | 0.140 | 0.181 | -23.077 |
| vertebral | MBIR_Bayesian | 80.000 | 0.140 | 0.181 | -23.077 |
| bc-coimbra | MBOV_Stochastic | 20.000 | 0.220 | 0.280 | -21.512 |
| bc-coimbra | MBOV_Stochastic | 10.000 | 0.225 | 0.280 | -19.479 |
| bc-coimbra | MBOV_Lower | 20.000 | 0.225 | 0.278 | -19.031 |
| bc-coimbra | MBOV_Lower | 10.000 | 0.230 | 0.282 | -18.357 |
| vertebral | MBOV_Centered | 40.000 | 0.120 | 0.146 | -17.444 |
| vertebral | MBOV_Higher | 40.000 | 0.120 | 0.146 | -17.444 |
| pima | MBOV_Centered | 80.000 | 0.542 | 0.649 | -16.492 |
| pima | MBOV_Lower | 80.000 | 0.542 | 0.649 | -16.492 |

## Least favorable cells (MICE beats minimax by largest %)

| dataset | mechanism | missing_rate_pct | minimax_mse | mice_mse | diff_% |
| --- | --- | --- | --- | --- | --- |
| ctg | MBIR_Bayesian | 80.000 | 0.006 | 0.000 | 529427.533 |
| ctg | MBIR_Frequentist | 80.000 | 0.006 | 0.000 | 529427.533 |
| ctg | MBOV_Stochastic | 20.000 | 0.005 | 0.000 | 525694.833 |
| ctg | MBUV | 80.000 | 0.005 | 0.000 | 228030.076 |
| ctg | MBOV_Centered | 60.000 | 0.002 | 0.000 | 189667.562 |
| ctg | MBOV_Higher | 60.000 | 0.002 | 0.000 | 189667.562 |
| ctg | MBOV_Lower | 10.000 | 0.002 | 0.000 | 162669.793 |
| ctg | MBOV_Stochastic | 10.000 | 0.001 | 0.000 | 140517.730 |
| ctg | MBUV | 60.000 | 0.001 | 0.000 | 131414.966 |
| ctg | MBIR_Bayesian | 60.000 | 0.001 | 0.000 | 125527.716 |

## Interpretation notes

1. **This is NOT a replication of Pereira et al.'s benchmark**. They measure imputation MAE on feature values; we measure test-set prediction MSE with MNAR injected on the label. See PROTOCOL.md §Deviation.
2. The minimax estimator run here is SGD-with-online-score-based-adversary, not the closed-form β̂ = M·(1/n Σ XᵢỸᵢ) + m from Christensen's 2020 write-up. A follow-up comparing the two algorithms is warranted if this result is encouraging.
3. Binary-labeled datasets with extreme MNAR (high rate + strong selection) can produce training splits with all-one or all-zero labels, causing SGD-based methods to diverge from the trivial mean-predictor. This is reflected in some LOSS cells.
4. MBUV is near-MCAR in label-only setting (see mnar_injection.py). Differences vs MICE there are expected to be small; the interesting signal is under MBOV_Lower/Higher.
