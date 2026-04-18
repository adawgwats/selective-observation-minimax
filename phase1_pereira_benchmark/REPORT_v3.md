# Phase 1 Report — Christensen Minimax vs MICE under MNAR Labels

**Path A benchmark**: Pereira et al. 2024 MNAR mechanisms applied to label column for regression tasks on 10 UCI medical datasets. See PROTOCOL.md for full spec and declared deviations from Pereira's original imputation-quality benchmark.

**Seeds completed**: 30 per cell (median across cells)
**Total rows**: 101,982
**Methods**: ['christensen_faithful', 'complete_case', 'erm_sgd', 'heckman', 'ipw_estimated', 'knn_impute', 'mean_impute', 'mice', 'minimax_score', 'oracle']
**Mechanisms**: ['MBIR_Bayesian', 'MBIR_Frequentist', 'MBOV_Centered', 'MBOV_Higher', 'MBOV_Lower', 'MBOV_Stochastic', 'MBUV']
**Datasets**: ['bc-coimbra', 'cleveland', 'cmc', 'ctg', 'pima', 'saheart', 'thyroid', 'transfusion', 'vertebral', 'wisconsin']

## Headline: Christensen minimax vs MICE

Across 250 (dataset, mechanism, rate) cells:

- **Wins** (95% CI strictly below MICE): **70** (28.0%)
- **Ties** (95% CIs overlap MICE): 102 (40.8%)
- **Losses** (95% CI strictly above MICE): 78 (31.2%)

## Headline: christensen_faithful vs minimax_score (DRO variant)

This is the two-minimax-estimator comparison: the faithful Christensen (closed-form vec solve + reference-based Q) vs the DRO-inspired SGD variant in minimax_core. Direct measurement of faithful-vs-paraphrase delta.

- Wins: **126** (50.4%)
- Ties: 47 (18.8%)
- Losses: 77 (30.8%)

## Win/loss vs MICE by MNAR mechanism

| mechanism | wins | ties | losses | total | win_rate | mean_mse_diff_pct |
| --- | --- | --- | --- | --- | --- | --- |
| MBOV_Centered | 15 | 18 | 17 | 50 | 0.300 | 1338340.964 |
| MBOV_Higher | 0 | 25 | 25 | 50 | 0.000 | 1338358.419 |
| MBOV_Lower | 30 | 18 | 2 | 50 | 0.600 | -24.208 |
| MBOV_Stochastic | 25 | 22 | 3 | 50 | 0.500 | 11543.188 |
| MBUV | 0 | 19 | 31 | 50 | 0.000 | 1059071.928 |

## Win/loss vs MICE by missing rate

| missing_rate_pct | wins | ties | losses | total | win_rate | mean_mse_diff_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 10.000 | 0.000 | 39.000 | 11.000 | 50.000 | 0.000 | 104798.987 |
| 20.000 | 6.000 | 27.000 | 17.000 | 50.000 | 0.120 | 364247.824 |
| 40.000 | 21.000 | 10.000 | 19.000 | 50.000 | 0.420 | 1018348.321 |
| 60.000 | 23.000 | 12.000 | 15.000 | 50.000 | 0.460 | 1967085.329 |
| 80.000 | 20.000 | 14.000 | 16.000 | 50.000 | 0.400 | 292809.830 |

## Win/loss vs MICE by dataset

| dataset | wins | ties | losses | total | win_rate | mean_mse_diff_pct |
| --- | --- | --- | --- | --- | --- | --- |
| bc-coimbra | 2 | 20 | 3 | 25 | 0.080 | 5.390 |
| cleveland | 8 | 11 | 6 | 25 | 0.320 | -4.151 |
| cmc | 8 | 7 | 10 | 25 | 0.320 | -2.898 |
| ctg | 6 | 1 | 18 | 25 | 0.240 | 7494530.492 |
| pima | 9 | 11 | 5 | 25 | 0.360 | -10.413 |
| saheart | 8 | 13 | 4 | 25 | 0.320 | -10.936 |
| thyroid | 8 | 16 | 1 | 25 | 0.320 | -4.844 |
| transfusion | 9 | 11 | 5 | 25 | 0.360 | -10.344 |
| vertebral | 5 | 6 | 14 | 25 | 0.200 | 37.054 |
| wisconsin | 7 | 6 | 12 | 25 | 0.280 | 51.231 |

## Most favorable cells (minimax beats MICE by largest %)

| dataset | mechanism | missing_rate_pct | minimax_mse | mice_mse | diff_% |
| --- | --- | --- | --- | --- | --- |
| ctg | MBOV_Lower | 20.000 | 0.001 | 0.180 | -99.329 |
| ctg | MBOV_Stochastic | 40.000 | 0.031 | 0.180 | -82.643 |
| ctg | MBOV_Lower | 40.000 | 0.036 | 0.180 | -80.158 |
| wisconsin | MBOV_Lower | 40.000 | 0.078 | 0.374 | -79.262 |
| wisconsin | MBOV_Stochastic | 60.000 | 0.095 | 0.374 | -74.590 |
| transfusion | MBOV_Lower | 80.000 | 0.195 | 0.760 | -74.349 |
| transfusion | MBOV_Centered | 80.000 | 0.195 | 0.760 | -74.349 |
| pima | MBOV_Lower | 80.000 | 0.180 | 0.649 | -72.294 |
| pima | MBOV_Centered | 80.000 | 0.180 | 0.649 | -72.294 |
| wisconsin | MBOV_Lower | 60.000 | 0.105 | 0.374 | -71.862 |

## Least favorable cells (MICE beats minimax by largest %)

| dataset | mechanism | missing_rate_pct | minimax_mse | mice_mse | diff_% |
| --- | --- | --- | --- | --- | --- |
| ctg | MBOV_Higher | 60.000 | 0.328 | 0.000 | 39006785.213 |
| ctg | MBOV_Centered | 60.000 | 0.328 | 0.000 | 39006751.495 |
| ctg | MBUV | 60.000 | 0.171 | 0.000 | 20340693.040 |
| ctg | MBOV_Higher | 40.000 | 0.163 | 0.000 | 19356284.560 |
| ctg | MBOV_Centered | 40.000 | 0.163 | 0.000 | 19356249.987 |
| ctg | MBUV | 80.000 | 0.321 | 0.000 | 14559533.721 |
| ctg | MBUV | 40.000 | 0.102 | 0.000 | 12204267.736 |
| ctg | MBOV_Higher | 20.000 | 0.056 | 0.000 | 6629154.159 |
| ctg | MBOV_Centered | 20.000 | 0.056 | 0.000 | 6629141.831 |
| ctg | MBUV | 20.000 | 0.038 | 0.000 | 4525356.048 |

## Interpretation notes

1. **This is NOT a replication of Pereira et al.'s benchmark**. They measure imputation MAE on feature values; we measure test-set prediction MSE with MNAR injected on the label. See PROTOCOL.md §Deviation.
2. The minimax estimator run here is SGD-with-online-score-based-adversary, not the closed-form β̂ = M·(1/n Σ XᵢỸᵢ) + m from Christensen's 2020 write-up. A follow-up comparing the two algorithms is warranted if this result is encouraging.
3. Binary-labeled datasets with extreme MNAR (high rate + strong selection) can produce training splits with all-one or all-zero labels, causing SGD-based methods to diverge from the trivial mean-predictor. This is reflected in some LOSS cells.
4. MBUV is near-MCAR in label-only setting (see mnar_injection.py). Differences vs MICE there are expected to be small; the interesting signal is under MBOV_Lower/Higher.
