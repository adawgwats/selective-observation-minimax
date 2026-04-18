# Phase 3 Implementation Plan

Scaffold is stub-only; this doc sequences the build-out.

## Milestone 1 — Data acquisition and schema verification (~8h, BLOCKING user action)

**User action required**: download JTPA data from Upjohn Institute or ICPSR.

Steps:
1. Register at `https://www.upjohn.org/data-tools/employment-research-data-center/national-jtpa-study` or `https://www.icpsr.umich.edu/web/ICPSR/studies/8997`
2. Download the JTPA Public Use File
3. Place files in `phase3_jtpa_benchmark/data_cache/jtpa/`
4. Notify Claude that data is available

Once data is available:
- Flesh out `datasets.load_jtpa()` with actual parsing
- Document the schema we observe in `DATA_SCHEMA.md`
- Run diagnostic: `summarize_non_response` — verify outcome-correlated non-response

## Milestone 2 — Baselines (~15h)

Order:
1. `CompleteCaseOLS` (trivial, 1h)
2. `ExperimentalITT` (trivial, 1h)
3. `IPWHorvitzThompson` (sklearn LogisticRegression + weighted OLS, 2h)
4. `HeckmanTwoStep` (statsmodels Probit + IMR + OLS, 3h)
5. `AIPWDoublyRobust` (4h)
6. `DoubleML` (use `doubleml` package, 2h)
7. `AbadieImbensMatching` (3h)

Verify each method's output against published baseline estimates where available. Record deviations — they're real science (different specification choices).

## Milestone 3 — Christensen sensitivity analysis (~8h)

1. Implement `ChristensenTreatmentEffect.estimate` per the algorithm in the adapter module's docstring
2. Choose a plausible Q family: ConstantQ, MonotoneInY_decreasing with δ ∈ {0.1, 0.3, 0.5}
3. Bootstrap CI computation (500 resamples)
4. Produce sensitivity band plot: τ̂ as a function of δ, direction, Q class

## Milestone 4 — Comparison + sensitivity (~8h)

1. Run `harness.run_comparison` across all methods
2. Run `leave_one_site_out` sensitivity on each method
3. Generate REPORT.md with:
   - Main comparison table (method × τ̂ × SE × CI)
   - Christensen sensitivity band
   - Site-robustness check
   - Discussion: under what assumptions does each method's estimate hold?
   - Comparison to Bloom-Orr-Bell (1993) experimental ITT as partial ground truth

## Milestone 5 — Paper draft (~10h)

Structure:
- Intro: motivate JTPA as canonical program evaluation with documented MNAR
- Methods: Christensen framework + all baselines  
- Empirical: the big comparison table
- Discussion: what Christensen adds vs what it doesn't — sensitivity bounds vs point estimates
- Practical implication: when should an applied researcher prefer Christensen?

Target venue options:
- Journal of Econometrics (most rigorous, longest turnaround)
- Journal of Applied Econometrics (faster, good fit)
- Economics and Statistics (broader reach)
- Labour Economics (domain-specific)

## Critical-path tasks

If Phase 3 is to be a real paper (not just a scaffold), these are critical path:

1. Data access (user action, undetermined)
2. Schema verification (data-dependent, 2-4h once data is in hand)
3. Heckman + DoubleML implementations (these are the strongest comparisons)
4. Clear decision on estimand: ATT vs ITT vs LATE — matters for which comparison numbers we target

## Explicit non-goals

- Not trying to prove Christensen is superior to DML. The finding may be that DML is better; that's fine — our paper is about characterizing when each is appropriate.
- Not reimplementing IV LATE (Abadie-Angrist-Imbens 2002). That's a different paper.
- Not site-level heterogeneity analysis. Different paper.
- Not dynamic treatment effects (60-month outcomes). Different paper.

## Effort summary

- Milestone 1 (data): 8h + user acquisition time (days to weeks)
- Milestone 2 (baselines): 15h
- Milestone 3 (Christensen sensitivity): 8h
- Milestone 4 (comparison): 8h
- Milestone 5 (paper): 10h

**Total: 49 hours ≈ 1.5 focused weeks.** Not on critical path for job search — this is a post-employment research investment.
