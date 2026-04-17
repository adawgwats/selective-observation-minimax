# Phase 1 Methodology Audit: Divergences Between Our Implementation and Christensen's Theory

**Purpose**: Before claiming any result from the benchmark, enumerate where our implementation diverges from the theoretical framework in [context.md](../context.md) (§§*v0 uncertainty set Q1* and *Exact v0 minimax objective*, lines 336–398). Rate each divergence as **recoverable with a code refactor** or **requires framework-level rework**.

**Source of truth for theory**: [context.md](../context.md) lines 336–398, which distill Christensen's *Minimax optimization.pdf* (not in this repo). Theoretical specification quoted verbatim below.

**Source of truth for implementation**: [minimax_core/gradient_validation.py:529–559](../minimax_core/gradient_validation.py) (`train_robust_score`), [minimax_core/adversary.py:84–129](../minimax_core/adversary.py) (`ScoreBasedObservationAdversary`), [minimax_core/uncertainty.py:265–292](../minimax_core/uncertainty.py) (`ScoreBasedObservationSet`), [phase1_pereira_benchmark/minimax_adapter.py](minimax_adapter.py).

---

## The theoretical v0 specification (verbatim from context.md)

From context.md lines 336–365:

> **v0 uncertainty set Q1**
> - nature chooses an observation probability `q(x, y)` or `q(x, a, y)`
> - low-probability observations are more likely in difficult, low-performing, or distressed cases
>
> **v0 implementation approximation:**
> - discretize `q(x, y)` into groupwise observation probabilities `q_g`
> - compute group priors from all examples, observed and unobserved
> - estimate group losses from observed labels only
> - optimize against the worst feasible `q_g` subject to:
>   - `q_min <= q_g <= q_max`
>   - weighted average observation rate matches the empirical observation rate

From context.md lines 376–398 (*Exact v0 minimax objective*):

> For groups `g = 1, ..., G`, let:
> - `pi_g` be the empirical group prior based on all examples
> - `L_g` be the mean observed loss in group `g`
> - `q_g` be the adversarial groupwise observation probability
>
> The v0 robust objective is:
> - minimize over model parameters
> - maximize over feasible `q_g`
> - objective value `sum_g pi_g * L_g / q_g`
>
> subject to:
> - `q_min <= q_g <= q_max` for each group
> - `sum_g pi_g * q_g = q_bar`, where `q_bar` is the empirical overall observation rate

---

## Divergence 1 — Score-based adversary instead of groupwise (major conceptual)

**Theory (context.md line 377)**: *"For groups `g = 1, ..., G`"* — `q_g` is per-group, and priors `pi_g` are group masses summing to 1.

**Implementation**: In [minimax_adapter.py:108](minimax_adapter.py), the benchmark's minimax method hard-codes `adversary_mode="score"` → [gradient_validation.py:529 `train_robust_score`](../minimax_core/gradient_validation.py) → [adversary.py:84 `ScoreBasedObservationAdversary`](../minimax_core/adversary.py) → [uncertainty.py:265 `ScoreBasedObservationSet`](../minimax_core/uncertainty.py). Per-example `q_i`, uniform weights `[1.0 for _ in range(count)]` in the projection (uncertainty.py:281).

**What the code actually computes**: The score-based mode is equivalent to the group-based mode in the limit where every example is its own group (uniform priors 1/n). The objective `Σ_i 1/n · ℓ_i / q_i` is a valid relaxation of `Σ_g π_g · L_g / q_g` but it is NOT the v0 spec — v0 is explicitly groupwise.

**Why it matters for the Pereira benchmark**: Per-example q gives the adversary maximum freedom, including the freedom to isolate single outliers. This differs from the intended "realistic selection regime" picture, where q varies at the regime level (e.g., `distressed_farms`, `drought_years`).

**Recoverability**: **Recoverable via code refactor, no framework change needed.** The codebase already has `train_robust_group` ([gradient_validation.py:490](../minimax_core/gradient_validation.py)) and `SelectiveObservationAdversary` ([adversary.py:34](../minimax_core/adversary.py)). The fix is:

1. Define groups based on MNAR mechanism semantics (e.g., quantile bins of observed y; or "response probability quantile" from a pilot estimator). For MBOV_Lower, low-y and high-y groups capture the selection.
2. Replace `ScoreMinimaxRegressor` in [minimax_adapter.py](minimax_adapter.py) with a `GroupMinimaxRegressor` wrapping `train_robust_group`.
3. Re-run the benchmark. Effort: ~2 hours.

---

## Divergence 2 — Proxy labels for unobserved examples (major theoretical)

**Theory (context.md line 380)**: *"`L_g` be the mean observed loss in group `g`"* — losses are computed **only from observed examples**. Unobserved examples contribute only to the group prior `π_g`, not to the loss.

**Implementation**: [gradient_validation.py:540–551](../minimax_core/gradient_validation.py) introduces `proxy_losses` and `effective_scores`:
```python
proxy_losses = [(prediction - proxy_label)**2 for ...]
effective_scores = [actual_loss if observed else proxy_loss for ...]
```

And our [minimax_adapter.py:79](minimax_adapter.py) sets `proxy_labels = np.where(response_mask, y_filled, observed_mean)` — so unobserved rows get a mean-imputed pseudo-label, and their squared-error-against-mean contributes to the effective score used by the adversary.

**What this means**: The adversary's per-example score for unobserved rows is computed against an imputed label — this is a hybrid between pure IPW (theory) and imputation (deviation). It is specifically **not** the mechanism in the v0 spec, which uses only observed losses.

**Why it matters**: (a) Unobserved rows influence adversary q-updates even though the theory says they shouldn't. (b) The proxy-label choice (mean-observed) creates a dependency on an imputation strategy that isn't part of the minimax formulation. (c) The hybrid is closer to an augmented-loss Lagrangian than to the saddle point in context.md §*Exact v0*.

**Recoverability**: **Recoverable via code refactor.** Modify `train_robust_score` (or `train_robust_group`) to compute scores and gradients only over observed examples. Concretely:

- In [gradient_validation.py:537–558](../minimax_core/gradient_validation.py), replace `effective_scores = [actual_loss if observed else proxy_loss for ...]` with `effective_scores = [loss for loss, observed in zip(losses, observed_mask) if observed]` and guard index bookkeeping.
- Remove the `train_proxy_labels` field from `LinearDataset` or keep it purely for diagnostic purposes.
- Effort: ~4 hours. This is a breaking change for the existing gradient_validation tests; they'll need updating.

---

## Divergence 3 — Score normalization inside the adversary (undocumented scaling)

**Theory**: No normalization of losses before the adversary gradient step.

**Implementation**: [adversary.py:109](../minimax_core/adversary.py): `scaled_scores = self._normalize_scores(scores)` where [adversary.py:122–129](../minimax_core/adversary.py) divides each score by the mean absolute score:
```python
scale = sum(abs(score) for score in scores) / len(scores)
return [score / scale for score in scores]
```

**What this means**: The adversary gradient `-score / q²` is computed on per-epoch-renormalized scores, not raw squared errors. The fixed point of the saddle iteration is shifted relative to the theoretical one.

**Recoverability**: **Trivially recoverable.** Drop `_normalize_scores` or document as a stability heuristic. The side effect is that raw loss magnitudes no longer matter for the adversary step size — only relative rankings do. This might actually be fine empirically but it IS a theory divergence.

**Effort**: 15 minutes to remove; longer to re-validate that SGD doesn't diverge without it.

---

## Divergence 4 — SGD stochastic saddle approximation vs. exact inner max

**Theory (context.md lines 383–398)**: The inner problem is `max over q_g subject to constraints` for fixed β. This is a convex optimization problem (objective is convex in q when we treat 1/q as the variable via substitution, feasible region is a box + linear equality) and has a closed-form solution via KKT conditions: at the optimum, high-loss groups hit `q_min` and low-loss groups hit `q_max` subject to the mean constraint.

**Implementation**: [adversary.py:103–117](../minimax_core/adversary.py) takes **one** gradient step on q per epoch:
```python
gradient = -score / max(q_value, self.config.epsilon) ** 2
proposed_q.append(q_value + self.config.adversary_step_size * gradient)
```
then projects. This is online minimax / stochastic saddle iteration, not exact inner optimization.

**What this means**: At any point during training, the current `q` is not `argmax_q L(β,q)` for the current β. The algorithm converges to the saddle only jointly, which is a different convergence guarantee than "for each β, compute worst-case L."

**Why it matters**: For β's that aren't yet close to the saddle, the adversary's q isn't the true worst case. This can under-state the worst-case loss that the learner is training against and slow convergence.

**Recoverability**: **Recoverable with a moderate refactor.** Two options:

1. **Exact inner max per epoch**: Solve the convex inner problem analytically each epoch. The v0 problem with equality + box constraints has a bang-bang solution (ordered by loss, assign q_min to high-loss groups first until the budget constraint forces mixing at one boundary group). Implement in a new `train_robust_group_exact_inner` that replaces the gradient step with a sort-and-fill solve.
2. **Multiple inner steps per outer step**: Run N adversary updates for each learner step. Already partially supported by the `adversary_step_size` hyperparameter; increasing it and looping is a small change.

Option 1 is more theoretically aligned; option 2 is easier.

**Effort**: Option 1 ~6 hours including tests. Option 2 ~2 hours but no convergence guarantee.

---

## Divergence 5 — Adaptive learning rate scaling (I introduced this)

**Theory**: Constant learning rate; no dataset-size dependence.

**Implementation**: I added `lr = learning_rate / sqrt(n/200)` in [minimax_adapter.py:108–111](minimax_adapter.py) to prevent SGD divergence on thyroid (7200 rows). This was a hotfix, not a theoretical move.

**Why I added it**: Pure-minimax SGD with `lr=0.05` produced parameter explosion on n=7200 datasets (MSE > 1e126). The minimax_core defaults are tuned to the repo's own small synthetic tests (~200 rows per trial).

**What this means**: The benchmark result for large-n datasets (thyroid especially) is confounded by my stability hotfix. The adaptive lr changes the effective step size by a factor that grows with dataset size. This is not what Christensen's theory prescribes.

**Recoverability**: **Trivially recoverable, but has knock-on effects.**

- Removing the adaptive lr brings back thyroid divergence. The real fix is (a) gradient normalization in minimax_core itself, (b) smarter initialization, or (c) per-parameter learning rates. None of these are in Christensen's theory either.
- **Honest reporting**: The main finding — MBOV_Lower 38% wins — is on mostly small-to-medium datasets where adaptive lr barely kicks in (n < 200 or just above). For thyroid specifically (n=3600 train, adaptive lr ≈ 0.012), the "loss to MICE" is partially a consequence of my hotfix slowing SGD convergence.

**Effort**: Removing is trivial. Replacing with a principled gradient normalization in minimax_core is ~4 hours of careful work + re-validation.

---

## Divergence 6 — Silent observation-rate clipping changes the optimization problem at high missingness

**Theory (context.md line 363)**: `q_min <= q_g <= q_max`, and `sum_g pi_g * q_g = q_bar` where `q_bar` is the **empirical overall observation rate**.

**Implementation**: [config.py:8](../minimax_core/config.py) defaults `q_min=0.25`. At 80% missing rate, empirical `q_bar = 0.2 < q_min`, making the box + mean-equality constraint INFEASIBLE. I initially wrote in this audit that the infeasibility would surface as a `ValueError` caught by the harness and recorded as NaN. **That was wrong — I verified by checking the results CSV and all 700 minimax cells at 80% rate returned finite MSEs, not NaN.**

**What actually happens** (verified by reading [gradient_validation.py:214–215](../minimax_core/gradient_validation.py)):
```python
def _clip_observation_rate(rate: float, config: GradientValidationConfig) -> float:
    return min(max(rate, config.q1.q_min), config.q1.q_max)
```
The observation_rate is **silently clamped into `[q_min, q_max]`** before being passed to the projection. When true `q_bar = 0.2` and `q_min = 0.25`, the code uses `q_bar_effective = 0.25`. The projection then returns a uniform `q ≈ 0.25`, which is feasible — but the constraint `sum_g pi_g · q_g = q_bar` is being enforced against a **fictitious budget**, not the empirical one.

**Why this is worse than a noisy failure**: An infeasibility `ValueError` would have made the problem visible and the cell would've been excluded. Silent clipping produces plausible-looking results that violate the theoretical specification. At 80% rate in particular, the adversary is working against an observation-rate constraint that overstates the true budget by 25%, which **uniformly softens the adversary's ability to downweight high-loss examples**. The minimax estimator's wins at 80% (22.9% overall, highest of any rate in our results) are thus partially explained by the constraint relaxation, not purely by the adversarial objective.

Call sites of the clipping: [gradient_validation.py:496, 509, 535, 568, 604, 648, 692](../minimax_core/gradient_validation.py) — used in every `train_robust_*` mode including group, score, time-varying, Knightian, etc. Every minimax mode in the codebase has this silent clamping.

**Recoverability**: **Trivially recoverable in code, but needs theoretical thought.**

1. Short-term: surface the infeasibility. Remove `_clip_observation_rate` and let the feasibility error propagate; log which cells fail.
2. Theoretically principled: make `q_min` dynamic: `q_min_effective = min(q_min_config, q_bar * 0.5)`. This preserves theory (box constraints widened to include the empirical budget) at the cost of weaker worst-case guarantees at high missingness. Document this as a decision, not a hack.
3. Report separately: results at rates where `q_bar < q_min` should be segregated in the main table and labeled as "constraint-relaxed minimax," not as a test of the v0 objective as written.

**Effort**: 30 minutes for code, but requires decision on which option. **Results at 60% and 80% missing rates in the current REPORT.md should be caveated as "constraint-relaxed" until this is resolved.**

---

## Divergence 7 — Uncertainty set has no directional/sign encoding of selection

**Theory (context.md line 338–340)**: *"low-probability observations are more likely in difficult, low-performing, or distressed cases"* — the selection is **directional** (outcome-correlated). The v0 Q1 captures this implicitly via the adversarial structure (high loss → low q).

**Implementation**: Our `ScoreBasedObservationSet` is symmetric in q — the box `[q_min, q_max]` and the equality constraint don't encode which direction of selection is more plausible. The adversary learns directionality from the scores alone.

**What this means**: For Pereira's MBOV_Lower (the case where minimax wins 38%), selection is directionally known: Y=0 is more likely to be hidden. Our Q1 doesn't USE this prior. A theoretically-aligned Q1 for this case would restrict q to be monotone in y (or at least monotone in some pilot score).

**Recoverability**: **Requires framework-level work, not just a refactor.** This is the biggest open issue. Options:

1. **Monotonicity constraints**: Add `q_i ≤ q_j` if `ŷ_i < ŷ_j` to encode "low-y-is-hidden" selection. Requires modifying the projection routine to enforce monotonicity on top of the box + linear constraints — nontrivial convex-optimization work.
2. **Shape-restricted Q**: Parametric class of q(x,y) restricted to (e.g.) logistic in y, then learn the logistic parameters adversarially. This is a different Q entirely and would require new uncertainty set classes.
3. **Diagnostic-only**: Keep the unrestricted Q, but add a diagnostic showing what q-pattern emerged and whether it matched the true MNAR mechanism.

Option 3 is cheap (~1 day); options 1–2 are weeks of research + code.

**This is the divergence that most affects the claim**: without directional structure, the adversary is "fighting at half strength" — it has to infer direction from losses rather than prior knowledge.

---

## Divergence 8 — Loss function = squared error, but target is binary

**Theory**: Christensen's note works with any prediction loss. The squared error choice is common for regression.

**Implementation**: [gradient_validation.py:503, 539](../minimax_core/gradient_validation.py) uses `(prediction - label) ** 2`. Our targets are binary {0,1} (Linear Probability Model for classification datasets).

**What this means**: The LPM regression gives us a real-valued prediction for a {0,1} target. Squared error is valid (LPM is a standard regression approach) but it's not aligned with logistic regression, which is what most of Pereira's comparison methods (MICE, kNN, etc.) implicitly support through their classification mode.

Our baselines (`OracleRegressor`, `MICERegressor`, etc.) also use OLS on binary y — so at least the comparison is apples-to-apples within our harness. But a cross-comparison to Pereira's own reported numbers requires the same LPM setup.

**Recoverability**: Not a divergence from theory — it's a scope choice for v0. **Recoverable if needed**: swap to logistic loss by modifying the per-example loss in `train_robust_*`. Effort: 2 hours.

---

## Divergence 9 — No validation of "converged to saddle"

**Theory**: Claim is "minimizes worst-case MSE over feasible q." Validation that we reached the saddle requires checking `max_q L(β,q)` against `L(β, q_adv)` at the end of training.

**Implementation**: We report test MSE after fixed-epoch training. We never verify the final β actually achieves its worst-case guarantee, nor that the adversary's q is close to `argmax`.

**Recoverability**: **Recoverable as a diagnostic.** After training, solve the inner max exactly (see Divergence 4) and compare to the adversary's final q. Report per-trial. Effort: ~3 hours.

---

## Summary table

| # | Divergence | Severity | Where | Recoverable? |
|---|---|---|---|---|
| 1 | Score-based adversary, not groupwise | Major | adversary.py:84, minimax_adapter.py:108 | **Yes, code refactor (~2h)** — use `train_robust_group` |
| 2 | Proxy labels for unobserved examples | Major | gradient_validation.py:540–551, minimax_adapter.py:79 | **Yes, code refactor (~4h)** — pure IPW over observed only |
| 3 | Score normalization inside adversary | Minor | adversary.py:122–129 | **Trivially (~15m)** — delete `_normalize_scores` |
| 4 | SGD saddle, not exact inner max | Moderate | adversary.py:103–117 | **Yes (~6h)** — bang-bang closed-form inner solve |
| 5 | Adaptive lr scaling (my hotfix) | Moderate | minimax_adapter.py:108–111 | **Yes, but needs stability fix first** — gradient norm or per-param lr in minimax_core |
| 6 | **Silent obs-rate clipping** changes budget constraint at 60%+ rates | **Major** — invalidates v0 interpretation of those cells | config.py:8, gradient_validation.py:214–215 | **Yes (~30m) but needs principled decision** on which alternative — all existing 60%/80% results should be re-caveated |
| 7 | No directional encoding in Q | Major | uncertainty.py:265 | **Framework rework (weeks)** — monotone or shape-restricted Q classes |
| 8 | Squared error on binary targets (LPM) | Scope choice, not divergence | gradient_validation.py:503 | **Yes (~2h)** — swap to logistic loss |
| 9 | No saddle-convergence diagnostic | Documentation gap | — | **Yes (~3h)** — add post-hoc max-q solve |

---

## What this means for the 10-seed benchmark result

Four of the nine divergences materially affect the headline number (42 wins / 350 cells = 12%):

- **#1 (score-based not groupwise)**: We tested a per-example generalization of v0, not v0 itself. The theoretical claim "minimax beats MICE" is about the groupwise formulation. A re-run with `train_robust_group` on meaningful groups (e.g., observed-y quartiles) might shift the number in either direction.
- **#5 (adaptive lr hotfix)**: Large-dataset losses are partially attributable to my hotfix slowing SGD, not to the minimax objective itself. Without the hotfix the method diverges; with it, we're benchmarking a different algorithm than pure minimax SGD.
- **#6 (silent obs-rate clipping)**: At 60% and 80% rates (which is where minimax shows the highest win rates — 14.3% and 22.9%), the adversary's budget constraint has been silently loosened. The "wins at 80%" result is partially an artifact of constraint relaxation, not a test of the v0 objective. These rows need to be re-run with a principled `q_min` policy before being interpreted as evidence for Christensen's framework.
- **#7 (no directional Q)**: The adversary isn't using the directional prior that would match MBOV's structure. A directional Q would likely increase the MBOV_Lower win rate above 38% and decrease it for MBUV.

**Recommendation before scaling to 30 seeds or writing any paper**: address divergences 1, 2, 5, 6 (the recoverable ones). Then re-run Phase 1. Only at that point does the 12% / 38% number represent a test of Christensen's v0 objective rather than a test of "a specific implementation approximation to v0."

### Quantifying the clipping impact on the headline MBOV_Lower result

Empirical observation rate = 1 − missing_rate. Clipping to `q_min = 0.25` is inactive when `q_bar ≥ 0.25`, i.e., at missing rates ≤ 75%. Among our five rates (10, 20, 40, 60, 80), clipping is **only active at 80%**.

Computed from [results/raw_results.csv](results/raw_results.csv):

- **MBOV_Lower total wins: 19 out of 50 cells (38%)**
- MBOV_Lower wins at 80% (clipping active): **7 of those 19** (37% of wins come from the constraint-relaxed regime)
- MBOV_Lower wins at rates 10–60% (clipping inactive): **12 of 40 cells (30%)**

**Interpretation**: Stripping the constraint-relaxed 80%-rate cells, the MBOV_Lower win rate drops from 38% to 30%. The niche claim survives — minimax is still winning on outcome-correlated MNAR at moderate rates — but the "minimax wins more at higher missing rates" story is weaker than it first looked, because the 80% rate is exactly where the budget constraint is silently relaxed.

---

## What we cannot claim right now

1. *"Christensen-style selective-observation minimax helps under outcome-correlated MNAR labels."* — We tested a score-based relaxation with proxy labels and an adaptive lr. These are three distinct departures from v0.
2. *"We replicate the Pereira 2024 benchmark for the minimax estimator."* — We adapted Pereira's MNAR mechanisms to label-column injection (documented Path A deviation in PROTOCOL.md). Our baseline comparisons reuse Pereira's baseline names but are configured for regression-under-MNAR-labels, not imputation quality.
3. *"This validates the closed-form β̂ = M·(1/n Σ XᵢỸᵢ) + m from Christensen's 2020 write-up."* — That estimator is not what we ran. `train_robust_score` is SGD with an online adversary.

## What we can claim

1. An SGD-with-online-score-adversary variant of Christensen-style robust estimation, combined with a mean-imputation proxy for unobserved rows and an adaptive learning rate for numerical stability, produces test MSE statistically lower than MICE+OLS in roughly 12% of label-MNAR regression cells across 10 UCI medical datasets, with wins concentrated on MBOV_Lower at rates ≥40%.

That's a real empirical finding. It is not, as stated, a test of the v0 framework in context.md.
