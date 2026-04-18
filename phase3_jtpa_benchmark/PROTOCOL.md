# Phase 3 Protocol — JTPA Program Evaluation under Selective Outcome Observation

## Motivation

Phases 1 and 2 evaluated Christensen's selective-observation minimax framework on synthetic MNAR mechanisms (Pereira et al. 2024; Ipsen et al. 2021). The benchmarks are reproducible but introduce MNAR artificially. Real-world outcome-correlated non-response in a canonical econometrics benchmark is a stronger test and brings the work into Christensen's own applied territory (labor economics, program evaluation).

**JTPA (Job Training Partnership Act)** is the natural choice. The JTPA Study (1987-1993) was a randomized training-program evaluation with a documented follow-up survey non-response problem. Published baseline treatment effect estimates span 30 years of econometrics literature.

## Task specification

**Causal question**: what is the effect of JTPA training on 30-month post-assignment earnings?

**Estimands of interest**:
1. **ITT (Intent-to-treat)**: `E[Y(assigned_to_training) - Y(control)]`
2. **ATE on treated**: `E[Y(1) - Y(0) | D=1]`
3. **LATE**: local average treatment effect among compliers

**Data structure**:
- `D` — treatment assignment (binary; random)
- `Y` — 30-month earnings (continuous, outcome, subject to MNAR)
- `X` — baseline covariates (age, gender, race, education, prior earnings, etc.)
- `R` — response indicator for the 30-month follow-up survey (binary)

**MNAR structure**: `R` depends on both observed `X` and unobserved post-treatment earnings `Y`. Specifically, individuals with worse post-treatment outcomes are less likely to respond to the follow-up survey. This is the classical **outcome-correlated MNAR** regime Christensen's framework is designed for.

## Why JTPA satisfies all 5 Christensen-applicability conditions

| # | Condition | JTPA status |
|---|---|---|
| 1 | Outcome-correlated MNAR | ✓ Documented in Heckman-Smith (1999); unemployed/low-earners respond less |
| 2 | Direction of selection known | ✓ q is decreasing in post-treatment earnings — classic Heckman setup |
| 3 | Training data has class balance | ✓ Continuous earnings outcome; no binary class collapse |
| 4 | Moderate-to-high missing rate | ✓ Follow-up response ~65-80%, so 20-35% missing |
| 5 | Linear outcome model reasonable | ✓ Log-earnings regression is a standard econometric specification |

This is the first benchmark in our program where all 5 conditions are satisfied NATURALLY, not by injection.

## Published baseline estimates (targets to match or beat)

These are 30-month earnings effects from the JTPA literature:

| Method | ATT estimate | Source | Notes |
|---|---|---|---|
| Experimental ITT (full compliance) | ~$1,200-1,800 | Bloom, Orr & Bell (1993) | Original evaluation report |
| Experimental adjusted for non-compliance | ~$1,593 | Orr et al. (1996) | Treatment-on-treated |
| Heckman two-step | ~$1,800-2,200 | Heckman, Smith & Clements (1997) | With selection correction |
| IV LATE (Angrist-Imbens) | varies | Abadie, Angrist & Imbens (2002) | Quantile IV |
| Matching-based | ~$1,500-2,400 | Heckman, Ichimura & Todd (1997) | Propensity score |
| Double ML | ~$1,700-2,000 | Chernozhukov et al. (2018) appendix | DML with RF |

**Note**: these range from ~$1,200 to ~$2,400 depending on method + sample definition. There's no single "true" target — the spread ITSELF is the research question. Christensen's value-add would be providing **sensitivity-bounded estimates**: what's the range of plausible treatment effects under different Q-class assumptions about the non-response mechanism?

## Comparison methods to implement

1. **Complete-case OLS** (naive, bias unknown direction)
2. **Experimental ITT** (ignores non-response; biased if response-correlated with outcome)
3. **Heckman two-step** (selection equation + outcome equation with IMR)
4. **IPW Horvitz-Thompson** (with estimated response propensity)
5. **AIPW / doubly-robust** (Robins-Rotnitzky-Zhao)
6. **Double ML** (Chernozhukov et al. 2018)
7. **Matching estimator** (Abadie-Imbens nearest-neighbor)
8. **Christensen minimax** with ConstantQ, MonotoneInY("decreasing"), and a structured Q family

## Evaluation

Since the outcome is partially missing in real data, we cannot compute a traditional test RMSE. Instead:

1. **Point estimates with confidence intervals** for each method (ATT estimand)
2. **Sensitivity bounds** for Christensen: report the [min, max] ATT across a plausible Q family (following Christensen-Connault 2023 style)
3. **Comparison to Bloom-Orr-Bell experimental ITT** as the "ground truth-ish" baseline for full-response case
4. **Identification of Christensen's unique contribution**: where does its bounds disagree with the other methods, and why?

## Data access

**JTPA Public Use File (JPU)** is available via:

- **Upjohn Institute for Employment Research**: `https://www.upjohn.org/data-tools/employment-research-data-center/national-jtpa-study`
- **ICPSR**: `https://www.icpsr.umich.edu/web/ICPSR/studies/8997` (Study 8997)

Access requirements:
- Free for academic/research use
- May require registration with Upjohn or ICPSR (standard data-use agreement)
- Download format: Stata `.dta`, SAS `.sas7bdat`, or CSV

**This protocol document is scaffold; data acquisition is a manual step the user must complete before the benchmark can run.**

Once downloaded, place the files in `phase3_jtpa_benchmark/data_cache/jtpa/` with these expected filenames:
- `baseline.csv` — pre-treatment covariates
- `outcomes_30m.csv` — 30-month follow-up earnings
- `assignment.csv` — treatment assignment

The loader (`datasets.py`) will handle different file formats and documented schema variants.

## Published schema (what we expect in the raw data)

Key variables (from the Abadie-Angrist-Imbens 2002 replication package):

| Variable | Type | Description |
|---|---|---|
| `earnings_30m` | float | 30-month post-assignment earnings (outcome, may be NaN) |
| `treatment` | 0/1 | Assignment indicator |
| `respondent_30m` | 0/1 | Responded to 30-month follow-up survey |
| `prior_earnings_quarters` | float[] | Earnings in up to 12 quarters prior to assignment |
| `age` | float | |
| `female` | 0/1 | |
| `race_black`, `race_hispanic` | 0/1 | |
| `education_years` | int | |
| `married` | 0/1 | |
| `has_dependents` | 0/1 | |
| `in_school` | 0/1 | |
| `welfare_received` | 0/1 | |
| `site` | categorical | 16 JTPA sites (Butte MT, Corpus Christi TX, etc.) |
| `target_group` | categorical | Youth, Adult Men, Adult Women, etc. |

Sample size: roughly 9,700 after restricting to valid treatment assignment.

## Non-goals for Phase 3 v1

Explicitly OUT OF SCOPE:
- Full Abadie-Angrist-Imbens replication (IV LATE quantile analysis)
- Site-level heterogeneity analysis
- Long-term (60+ month) outcomes
- Any claim of "solving" JTPA — we're presenting a sensitivity analysis, not a new causal estimate

## Expected paper framing

*"We apply Christensen's (2020) selective-observation minimax framework to the JTPA program-evaluation benchmark, where outcome non-response is documented to be correlated with post-treatment earnings. We compare against standard correction methods (Heckman two-step, IPW, doubly-robust, DML) and provide sensitivity bounds for the average treatment effect across a family of plausible non-response mechanisms. The 95% ITT interval from Bloom, Orr & Bell (1993) is $[x, y]$; our minimax-sensitivity interval is $[x', y']$, with the width indicating how much the treatment effect depends on non-response assumptions."*

This is a different kind of paper from Phases 1 and 2 — it's an **applied econometrics sensitivity analysis** rather than an ML benchmark win-rate table. Target venue: Journal of Econometrics, Journal of Applied Econometrics, or Review of Economics and Statistics.

## Effort estimate

- Scaffolding (this commit + next 2 commits): ~4-6 hours
- Data loading + preprocessing (after user downloads data): ~8 hours
- Baseline implementations (Heckman, IPW, DML, matching): ~15 hours
- Christensen sensitivity analysis with multiple Q classes: ~8 hours
- Comparison + write-up: ~10 hours

Total: **~50 hours** (a week of focused work). Do this AFTER the job search stabilizes; it's a publishable piece, not a quick sprint.
