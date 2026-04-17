# Implementation plan: filling in the christensen_core scaffold

Scaffold is stub-only. Every function raises `NotImplementedError` with a specific
instruction in its docstring. This plan gives the build order, estimated effort,
and the specific tests that must pass at each milestone before proceeding.

Total estimated effort: **~2 focused days** (16–20 hours) for v1 coverage of
ConstantQ and Parametric2ParamForBinary, including re-running the Pereira
benchmark with the faithful estimator. MBIR and MBOV_Centered stay as
"Q-mismatch" and are deferred to v2.

## Build order

### Milestone 1 — moments.py (~2h)

**Goal**: sample moments from corrupted data.

Implement in order:
1. `compute_W_n` (one-liner: `X.T @ X / n`)
2. `compute_b_n` (one-liner: `X.T @ Y_tilde / n`)
3. `compute_r_n` (loop or vectorized; skip non-respondents)
4. `compute_moments` (convenience wrapper)

**Acceptance**: tests/test_moments.py passes.

Key test: `test_r_n_ipw_identity_large_sample` — with the true q, the sample
r_n must converge to E[X Y] computed on uncorrupted data. This is the IPW
identity and it is what makes the entire framework work. If this fails,
something is fundamentally wrong.

### Milestone 2 — inner_solver.py (~3h)

**Goal**: closed-form (M, m) given fixed q.

Implement:
1. `solve_inner` — the vec trick from PDF §1.5 with pinv
2. `inner_objective_value` — evaluate the quadratic at a given (M, m)
3. `predict_from_M_m` — apply estimator to new data

**Critical subtlety**: the pinv system `(b⊗I)W(b'⊗I) a = (b⊗I) r` is rank
deficient. Use `np.linalg.lstsq` with `rcond=None` rather than explicit pinv
to avoid numerical instability on ill-conditioned W. The PDF explicitly
acknowledges the rank deficiency: "any such solution gives the same minimum
squared prediction error."

**Acceptance**: tests/test_inner_solver.py passes, including
`test_reduction_to_OLS_when_q_is_constant_and_full_response` — the most
important regression test.

### Milestone 3 — q_classes.py (~2h)

**Goal**: implement `ConstantQ` and `Parametric2ParamForBinary` fully;
`MonotoneInY` remains stubbed for v2.

Both v1 classes are tiny (ConstantQ is 1D, Parametric is 2D with a binary
branch). Parameter bound routines and q_values methods are each a few lines.

**Acceptance**: tests/test_q_classes.py passes.

### Milestone 4 — outer_solver.py (~4h)

**Goal**: maximize inner objective over Q for the two v1 classes.

Order:
1. `_solve_outer_constant` — 1D via scipy.optimize.minimize_scalar (bounded).
   Inner is called once per outer evaluation; ~50 evaluations total.
2. `_solve_outer_2param_binary` — 2D grid + L-BFGS-B polish.
   Handle monotone constraint via an explicit linear inequality in the
   scipy.optimize call, or by rejecting infeasible grid points before polish.

**Critical**: evaluate inner carefully. Cache `W_n` and `b_n` (they don't
depend on q — recompute them once per fit(), not per outer-eval). Only
`r_n(q)` must be recomputed each outer iteration.

**Acceptance**: tests/test_outer_solver.py passes. Also verify that for
ConstantQ on MAR-like data, θ* ≈ empirical observation rate q̂.

### Milestone 5 — estimator.py (~1h)

**Goal**: sklearn-style ChristensenEstimator tying the pieces together.

Thin wrapper. Main work is defensive Y_tilde handling (zero-out
non-respondent entries) and intercept-column management.

**Acceptance**: tests/test_reduction_to_ols.py passes
(`test_constant_q_recovers_beta_true_on_large_sample` is the key check).

### Milestone 6 — pereira_q.py (~1h)

**Goal**: dispatch from mechanism name to correct QClass.

Trivial lookup table. Raise `NotImplementedError` for MBIR_* and
`MBOV_Centered` — those are v2.

**Acceptance**: mechanism names in the Pereira benchmark that we've marked
"high" or "medium" fidelity return a valid QClass; "low" fidelity ones raise.

### Milestone 7 — adapter + harness wiring (~1h)

**Goal**: make `christensen_faithful` a method in the phase1 harness.

In phase1_pereira_benchmark/christensen_adapter.py, implement `fit`/`predict`
that dispatches on `mechanism_name`. In phase1_pereira_benchmark/harness.py,
add `christensen_faithful` to `METHOD_ORDER` and `METHOD_FACTORIES`. For the
3 deferred mechanisms (MBIR_*, MBOV_Centered), the harness should catch
`NotImplementedError` and record that cell as "N/A" (not NaN — distinguish
"method doesn't apply" from "method ran and failed").

**Acceptance**: harness runs one cell end-to-end with the faithful estimator
on wisconsin × MBOV_Lower × 20% × seed=0, producing a finite test MSE.

### Milestone 8 — integration test (~2h)

**Goal**: end-to-end sanity check of the whole pipeline before any benchmark
re-run.

Run the synthetic MBOV scenario from tests/test_integration.py. Confirm:
- Christensen beats OLS-on-observed on known-Q MBOV_Lower.
- Christensen ties OLS/q̂ on MAR.
- Christensen degrades gracefully when Q is misspecified.

If any of these fail, DO NOT run the Pereira benchmark. Something is wrong
in the core pipeline and running full benchmark will just produce noise.

### Milestone 9 — re-run Phase 1 (~2h wall clock, ~15min active)

**Goal**: faithful estimator vs MICE baseline on Pereira's 10 datasets.

Resume-logic in the harness means the re-run will process only the new
`christensen_faithful` column; existing columns (minimax_score, mice, etc.)
are already cached. Effective runtime is roughly 1/9 of the original benchmark
since we're adding one method.

Generate REPORT_v2.md with three sections:
1. Faithful Christensen vs MICE (the new headline claim)
2. Faithful Christensen vs DRO variant minimax_score (the old method)
3. Which Pereira mechanisms are "high/medium/low" Q-fidelity and how the
   results break down by that axis

Only the "high fidelity" slice (MBOV_Lower, MBOV_Higher, MBOV_Stochastic)
represents a clean test of Christensen's framework; the others are either
MAR-approximation (ConstantQ for MBUV) or deferred (MBIR, MBOV_Centered).

## What we are deliberately NOT building in v1

- **MBIR Q class** (missingness depends on an unobserved covariate) — needs
  a `DependentOnUnobservedScore` QClass with more thought about what
  constraints to put on the unobserved score.
- **MBOV_Centered Q class** — non-monotone; would need a piecewise-linear
  U-shape parameterization. Not a natural fit to Christensen's monotone
  example.
- **Problem 3 extension** (unobserved X for non-respondents) — our benchmark
  has full X observation, so this doesn't matter for v1.
- **Closed-form outer max** — we use numerical optimization. For some Q
  classes a closed-form bang-bang solution may exist but deriving it is
  research, not engineering.
- **Multiple local optima in the outer problem** — the outer max over some
  Q classes may be nonconvex. For Parametric2ParamForBinary (2D) the grid +
  polish pattern handles this robustly; for higher-dim Q we'd need more care.

## Dependencies

Pure numpy + scipy. No torch, no sklearn, no pandas in the core. The adapter
imports pandas only for DataFrame handling.

## Relationship to minimax_core/

christensen_core does NOT import from minimax_core. The two packages are
independent. The phase1 benchmark harness is the only place both are loaded
together, purely for head-to-head comparison.

The Knightian / surprise-driven / structural-break extensions in minimax_core
are orthogonal to this work. They extend the DRO variant's Q class but don't
map cleanly onto Christensen's primary-source framework. If any of those
extensions turn out to be useful empirically, they can be ported into
christensen_core as new QClass implementations (e.g., StructuralBreakInY).
