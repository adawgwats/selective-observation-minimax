# Phase 1 Protocol: Pereira et al. 2024 MNAR Mechanisms Applied to Regression Under MNAR Labels

## Source of truth

- **Paper**: Pereira, R.C., Abreu, P.H., Rodrigues, P.P., & Figueiredo, M.A.T. (2024). *Imputation of data Missing Not at Random: Artificial generation and benchmark analysis*. Expert Systems with Applications, 249, 123654. [DOI:10.1016/j.eswa.2024.123654](https://doi.org/10.1016/j.eswa.2024.123654)
- **Primary spec**: Chapter 9 of Pereira's 2023 PhD thesis (University of Coimbra). Saved at [../docs/pereira_2024/chapter9_mnar.txt](../docs/pereira_2024/chapter9_mnar.txt). The 2024 ESWA paper is this chapter.
- **Canonical MNAR code**: `mdatagen` Python library, co-authored by Pereira and Abreu. GitHub: https://github.com/ArthurMangussi/pymdatagen. Installed via `pip install mdatagen`. Authoritative because it is the authors' own implementation, peer-reviewed (Mangussi et al. 2025, Neurocomputing).

## Deviation from Pereira et al. 2024 — important and declared upfront

**Pereira et al. 2024 benchmarks *imputation quality***: they inject MNAR into feature values X, impute, and measure MAE between imputed X̂ and ground-truth X.

**This study benchmarks *regression under MNAR labels***: we inject MNAR into the label column Y, fit a regression model, and measure held-out prediction MSE.

**Reason for deviation**: Christensen's selective-observation minimax estimator (minimax_core in this repo) is defined for regression under selective observation of Y, not imputation of X. Plugging it into the Pereira imputation harness would be a category error. Instead, we adopt the authors' MNAR generation mechanisms — including the authors' own code via mdatagen's `missTarget=True` flag — and apply them to Y. The MNAR-generation methodology is unchanged; the task is regression, not imputation.

**Framing for a paper**: *"We adopt the MNAR generation mechanisms of Pereira et al. 2024 — using the authors' published mdatagen library — and apply them to the label variable in a regression task on the ten UCI medical datasets from their benchmark. We then compare Christensen's selective-observation minimax estimator against standard methods for regression under MNAR labels."* This is a methodology transfer, not a direct replication, and MUST be labeled as such in any public write-up.

## Datasets (Pereira Table 9.3, p.124)

All from UCI Machine Learning Repository. Ten public medical datasets, complete (no missing values pre-injection).

| Dataset      | # Instances | # Categorical | # Continuous |
|--------------|-------------|---------------|--------------|
| bc-coimbra   |         116 |             1 |            9 |
| cleveland    |         303 |             7 |            7 |
| cmc          |        1473 |             8 |            2 |
| ctg          |        2126 |             2 |           23 |
| pima         |         768 |             1 |            8 |
| saheart      |         462 |             2 |            8 |
| thyroid      |        7200 |            16 |            6 |
| transfusion  |         748 |             1 |            4 |
| vertebral    |         310 |             1 |            5 |
| wisconsin    |         569 |             1 |           30 |

Target variable: last column (the class label) for each dataset, treated as a regression target via Linear Probability Model (multiclass datasets binarized to "healthy vs. any pathology" — see datasets.py for exact mapping per dataset).

## MNAR Mechanisms (Pereira §9.1)

All four novel mechanisms introduced in the paper, implemented in mdatagen. For label-only injection, we use `mMNAR(X, y, missTarget=True)` and invoke methods with `columns=['target']` where applicable.

| Mechanism             | mdatagen method                                        | Notes |
|-----------------------|--------------------------------------------------------|-------|
| MBOV (Lower)          | `MBOV_randomness(rate, randomness=0.0, ['target'])`    | default MBOV behavior, p=0 |
| MBOV (Higher)         | see adapter — negate target, call MBOV_randomness      | authors describe but no direct flag in mdatagen |
| MBOV (Stochastic)     | `MBOV_randomness(rate, randomness=0.25, ['target'])`   | p=0.25 per paper |
| MBOV (Centered)       | `MBOV_median(rate, ['target'])`                        | |
| MBUV                  | `uMNAR.run(method='MBUV_Normal', ...)`                 | generates N(0,1) unobserved feature; univariate |
| MBIR (Frequentist)    | `MBIR(rate, ['target'], 'Mann-Whitney')`               | α=0.05 |
| MBIR (Bayesian)       | `MBIR(rate, ['target'], 'Bayesian')`                   | BF ≥ 10 |
| MBOUV (Multivariate)  | **excluded** — multivariate by design, cannot target-only | see DEVIATIONS |

For binary targets, MBOV (Lower) is equivalent to preferentially removing Y=0 observations. This IS meaningful MNAR (it creates selection bias toward Y=1), but note the behavior when designing plots.

## Missingness Rates (Pereira §9.2)

**10%, 20%, 40%, 60%, 80%** — five rates, exactly as in the paper.

## Baselines

Paper baselines (Table 9.1) are imputation methods: DAE, VAE, kNN, Mean/Mode, MICE, SoftImpute. Here they are re-used as **impute-then-regress pipelines**, supplemented with selection-bias-specific methods:

| Method            | Description |
|-------------------|-------------|
| Oracle-OLS        | OLS on full pre-injection data (upper bound on achievable MSE) |
| Complete-case OLS | Drop observations with missing Y, then OLS on rest |
| Mean+OLS          | Impute missing Y with mean of observed Y, then OLS (mirrors Pereira Mean) |
| MICE+OLS          | `sklearn.IterativeImputer` on [X, Y], then OLS (mirrors Pereira MICE — **baseline to beat**) |
| kNN+OLS           | `sklearn.KNNImputer(k=5)` on [X, Y], then OLS (mirrors Pereira kNN) |
| IPW-OLS (est.)    | Estimate response probability q(X) via logistic regression; weighted OLS on observed |
| Heckman 2-step    | Selection + outcome equations (statsmodels) — classical selection-bias correction |
| **Christensen minimax** | Method under test — `train_robust_score` from `minimax_core.gradient_validation` |

DAE/VAE/SoftImpute omitted from initial slice for scope control; can be added as ablation if the core result is interesting.

## Experimental Protocol (Pereira §9.2)

- **Split**: 50% train / 20% validation / 30% test, stratified by class label.
- **Normalization**: features scaled to [0, 1] per column.
- **Categorical encoding**: one-hot **after** missing-data injection.
- **Seeds**: 30 independent runs, dataset reshuffled per seed. Final result = mean over 30 seeds with 95% CI.
- **CI computation**: `μ ± 1.96·σ/√30`. Methods with overlapping CIs are declared "not significantly different."
- **Evaluation metric**: held-out test MSE (prediction MSE on the 30% test split, using the model trained on observed Y in the training split). Note: this differs from Pereira's MAE on imputed X — we use test-set prediction MSE because our task is regression, not imputation.

## Deviations, explicitly declared

1. **Task**: imputation → regression under MNAR labels (justified above).
2. **Metric**: imputation MAE → prediction MSE on held-out test (consequence of task change).
3. **MBOUV excluded** from single-target injection (multivariate-by-design, not applicable to label-only).
4. **DAE/VAE/SoftImpute omitted** from initial slice; MICE retained as the primary imputation-based baseline.
5. **Categorical encoding timing**: Pereira applies one-hot after injection. We follow.
6. **Multiclass → binary**: targets with >2 classes are binarized (healthy vs. any pathology) because minimax_core's linear estimator is designed for scalar Y. This is a real deviation from a general multi-output framework and is declared per-dataset in datasets.py.

## Reproducibility

- Branch: `phase1-pereira-benchmark`
- Python: 3.11 (system installation)
- Dependencies: see `requirements.txt` in this directory
- Random state: seeded via numpy global RNG in harness, per-trial seed = `base_seed + trial_index`
