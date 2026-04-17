# christensen_core — Faithful implementation of Christensen's minimax estimator

**Primary source**: [../docs/christensen_minimax.pdf](../docs/christensen_minimax.pdf) — "Regression example for prediction," 7 pages. Text extract at [../docs/christensen_minimax_pdf.txt](../docs/christensen_minimax_pdf.txt).

**Purpose**: Implement the estimator derived on PDF pages 1–7 as literally as possible. This package exists alongside `minimax_core/` (which is a DRO-inspired variant with known divergences from Christensen's theory — see [../phase1_pereira_benchmark/AUDIT_v2.md](../phase1_pereira_benchmark/AUDIT_v2.md)). The two packages cohabit intentionally: `minimax_core` is the engineering artifact of the last five years; `christensen_core` is what Christensen actually wrote down.

## What this package does (in one paragraph)

Given training data `(Xᵢ, Ỹᵢ)` with `Ỹᵢ = εᵢ·Yᵢ` and `εᵢ ~ Bernoulli(q(Xᵢ,Yᵢ))` for some unknown response function `q` lying in an uncertainty set `Q`, estimate `β` of the linear regression `Y = X'β + error` such that worst-case mean-square prediction error is minimized. Concretely: find `(M*, m*)` solving

```
min_{M, m}  max_{q ∈ Q}  L(M, m; q)
```

where `β̂ = M·bₙ + m`, `bₙ = (1/n)ΣXᵢỸᵢ`, and `L(M, m; q)` is the approximate prediction MSE expression from PDF Problem 2. For fixed `q`, `(M, m)` has a closed form via the vec trick + pinv. Outer max over `Q` depends on the structure of `Q`.

## Module layout

- `moments.py` — compute `bₙ`, `Wₙ`, `rₙ(q)` from corrupted training data. Pure numpy. Corresponds directly to PDF Problem 2 terms.
- `inner_solver.py` — closed-form `(M, m)` given `q` via the vec trick on the FOC `(b⊗Iₐ) W (b'⊗Iₐ) a = (b⊗Iₐ) r`. Corresponds directly to PDF §1.5.
- `q_classes.py` — structured uncertainty-set classes. Abstract base `QClass` plus concrete implementations starting with the ones Christensen exemplifies and the ones needed for Pereira's MNAR mechanisms. See PDF page 5 final paragraph and `q_classes.py` docstring.
- `outer_solver.py` — maximize the inner objective over a given `QClass`. Closed form where possible, grid/convex-opt otherwise.
- `estimator.py` — sklearn-style `ChristensenEstimator` that wraps everything with `fit(X, Y_tilde, response_mask, q_class)` / `predict(X_test)`.
- `pereira_q.py` — mapping from Pereira MNAR mechanism names to concrete `QClass` instances. Documents which mechanisms are well-matched to which Q structures.

## Tests (acceptance criteria)

Before declaring any faithful-to-Christensen claim:

1. **Reduces to OLS under MAR**: for `Q = {constant q}`, the estimator should equal `(1/q̂)·β̂_OLS` (the classical MAR correction) up to numerical tolerance.
2. **Identity under no corruption**: for `εᵢ ≡ 1` (no missingness), the estimator should equal OLS on full data.
3. **Matches analytical result on toy problem with known optimum**: construct a small problem where the saddle can be computed by hand or by a high-accuracy solver; verify agreement.
4. **Beats OLS/q̂ on synthetic MNAR**: generate data with a known `q(x,y)` in `Q`, verify Christensen's estimator has lower test MSE than MAR-corrected OLS.
5. **Reduces correctly for binary Y**: document the simplification (for `Y ∈ {0,1}`, `Ỹᵢ = 1` only if responded AND `Yᵢ = 1`), verify implementation handles this degenerate case.

Tests live in `tests/`. Each test file has a specific purpose documented in its module docstring.

## Non-goals

- **Not** a general minimax library. This package implements one specific estimator from one 7-page note.
- **Not** a replacement for `minimax_core/`. The two coexist; the existing benchmark in `phase1_pereira_benchmark/` will be extended to compare both side by side.
- **Not** policy learning or downstream decision-making. This package implements the supervised regression estimator (PDF §1). Downstream extensions are a separate research question.
- **Not** a closed-form for the OUTER max. For arbitrary `Q`, the outer problem may be nonconvex. We provide correct solvers for specific structured `Q` classes; arbitrary `Q` is out of scope for v1.

## Dependencies

numpy, scipy (for pinv/lstsq and optimization routines if needed). No torch, no sklearn in the core. A separate adapter in `phase1_pereira_benchmark/christensen_adapter.py` provides the sklearn-style interface for the benchmark harness.
