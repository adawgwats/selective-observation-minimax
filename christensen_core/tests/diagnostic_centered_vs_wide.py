"""Diagnostic: head-to-head wide-box vs centered-Q on the 4 previously-failing scenarios.

Run directly:
    python christensen_core/tests/diagnostic_centered_vs_wide.py

This is NOT a pytest file — it's a reproducible diagnostic that prints numbers so
we can decide honestly whether the centered-Q approach fixes anything, partially fixes
anything, or doesn't help at all.
"""
from __future__ import annotations

import numpy as np

from christensen_core.estimator import ChristensenEstimator
from christensen_core.q_classes import (
    ConstantQ,
    Parametric2ParamForBinary,
    QClassConfig,
)
from christensen_core.reference_based_q import (
    adaptive_centered_q_for,
    centered_config,
    compute_q_hat,
    mechanism_to_delta,
)


def _scenario_mar_constant_q(seed: int, n: int = 500, q_true: float = 0.6):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    X_aug = np.concatenate([np.ones((n, 1)), X], axis=1)
    beta_true = np.array([0.5, 1.0, -0.5, 0.3])
    Y = X_aug @ beta_true + 0.1 * rng.standard_normal(n)
    response = rng.random(n) < q_true
    Y_tilde = np.where(response, Y, 0.0)

    X_test = rng.standard_normal((500, 3))
    X_test_aug = np.concatenate([np.ones((500, 1)), X_test], axis=1)
    Y_test = X_test_aug @ beta_true + 0.1 * rng.standard_normal(500)

    q_hat = compute_q_hat(response)

    def fit_mse(q_class):
        est = ChristensenEstimator(q_class=q_class, fit_intercept=True).fit(X, Y_tilde, response)
        y_pred = est.predict(X_test)
        return float(np.mean((y_pred - Y_test) ** 2)), est._theta_star

    # Wide box
    mse_wide, theta_wide = fit_mse(ConstantQ(config=QClassConfig(q_min=0.05, q_max=1.0)))
    # Centered δ=0.30
    mse_c30, theta_c30 = fit_mse(ConstantQ(config=centered_config(q_hat, 0.30)))
    # Centered δ=0.10
    mse_c10, theta_c10 = fit_mse(ConstantQ(config=centered_config(q_hat, 0.10)))
    # Centered δ=0.02 (near-point ball)
    mse_c02, theta_c02 = fit_mse(ConstantQ(config=centered_config(q_hat, 0.02)))

    # OLS on respondents (baseline)
    X_resp = X_aug[response]
    Y_resp = Y[response]
    beta_ols = np.linalg.lstsq(X_resp, Y_resp, rcond=None)[0]
    y_pred_ols = X_test_aug @ beta_ols
    mse_ols = float(np.mean((y_pred_ols - Y_test) ** 2))

    return dict(q_hat=q_hat, mse_wide=mse_wide, theta_wide=theta_wide,
                mse_c30=mse_c30, theta_c30=theta_c30,
                mse_c10=mse_c10, theta_c10=theta_c10,
                mse_c02=mse_c02, theta_c02=theta_c02,
                mse_ols=mse_ols)


def _scenario_mbov_lower(seed: int, n: int = 500, q0: float = 0.3, q1: float = 0.9):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    X_aug = np.concatenate([np.ones((n, 1)), X], axis=1)
    beta_true = np.array([0.3, 0.8, -0.4, 0.2])
    y_cont = X_aug @ beta_true + 0.1 * rng.standard_normal(n)
    Y = (y_cont > 0).astype(float)
    q_true = np.where(Y == 0, q0, q1)
    response = rng.random(n) < q_true
    Y_tilde = np.where(response, Y, 0.0)

    X_test = rng.standard_normal((500, 3))
    X_test_aug = np.concatenate([np.ones((500, 1)), X_test], axis=1)
    y_cont_test = X_test_aug @ beta_true + 0.1 * rng.standard_normal(500)
    Y_test = (y_cont_test > 0).astype(float)

    q_hat = compute_q_hat(response)

    def fit_mse(q_class):
        est = ChristensenEstimator(q_class=q_class, fit_intercept=True).fit(X, Y_tilde, response)
        y_pred = est.predict(X_test)
        return float(np.mean((y_pred - Y_test) ** 2)), est._theta_star

    mse_wide, theta_wide = fit_mse(Parametric2ParamForBinary(monotone="increasing",
                                                              config=QClassConfig(q_min=0.05, q_max=1.0)))
    mse_c30, theta_c30 = fit_mse(Parametric2ParamForBinary(monotone="increasing",
                                                            config=centered_config(q_hat, 0.30)))
    mse_c10, theta_c10 = fit_mse(Parametric2ParamForBinary(monotone="increasing",
                                                            config=centered_config(q_hat, 0.10)))

    # OLS on respondents
    X_resp = X_aug[response]
    Y_resp = Y[response]
    beta_ols = np.linalg.lstsq(X_resp, Y_resp, rcond=None)[0]
    y_pred_ols = X_test_aug @ beta_ols
    mse_ols = float(np.mean((y_pred_ols - Y_test) ** 2))

    return dict(q_hat=q_hat, mse_wide=mse_wide, theta_wide=theta_wide,
                mse_c30=mse_c30, theta_c30=theta_c30,
                mse_c10=mse_c10, theta_c10=theta_c10,
                mse_ols=mse_ols)


def _scenario_wrong_q(seed: int, n: int = 500, q0: float = 0.9, q1: float = 0.3):
    """MBOV_HIGHER truth, but we fit with monotone='increasing' (wrong direction)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    X_aug = np.concatenate([np.ones((n, 1)), X], axis=1)
    beta_true = np.array([0.3, 0.8, -0.4, 0.2])
    y_cont = X_aug @ beta_true + 0.1 * rng.standard_normal(n)
    Y = (y_cont > 0).astype(float)
    q_true = np.where(Y == 0, q0, q1)  # DECREASING: q(0)=0.9, q(1)=0.3
    response = rng.random(n) < q_true
    Y_tilde = np.where(response, Y, 0.0)

    X_test = rng.standard_normal((500, 3))
    X_test_aug = np.concatenate([np.ones((500, 1)), X_test], axis=1)
    y_cont_test = X_test_aug @ beta_true + 0.1 * rng.standard_normal(500)
    Y_test = (y_cont_test > 0).astype(float)

    q_hat = compute_q_hat(response)

    def fit_mse(q_class):
        est = ChristensenEstimator(q_class=q_class, fit_intercept=True).fit(X, Y_tilde, response)
        y_pred = est.predict(X_test)
        return float(np.mean((y_pred - Y_test) ** 2))

    # Wrong direction: increasing when truth is decreasing
    mse_wide = fit_mse(Parametric2ParamForBinary(monotone="increasing",
                                                  config=QClassConfig(q_min=0.05, q_max=1.0)))
    mse_c30 = fit_mse(Parametric2ParamForBinary(monotone="increasing",
                                                 config=centered_config(q_hat, 0.30)))
    mse_c10 = fit_mse(Parametric2ParamForBinary(monotone="increasing",
                                                 config=centered_config(q_hat, 0.10)))

    X_resp = X_aug[response]
    Y_resp = Y[response]
    beta_ols = np.linalg.lstsq(X_resp, Y_resp, rcond=None)[0]
    y_pred_ols = X_test_aug @ beta_ols
    mse_ols = float(np.mean((y_pred_ols - Y_test) ** 2))

    return dict(q_hat=q_hat, mse_wide=mse_wide, mse_c30=mse_c30,
                mse_c10=mse_c10, mse_ols=mse_ols)


def _mean(metrics, key):
    return float(np.mean([m[key] for m in metrics]))


def _scenario_adaptive_by_mechanism(mechanism: str, n_seeds: int = 20):
    """For each seed, build the data matching the named mechanism, then fit
    Christensen with the adaptive-delta dispatch (mechanism_to_delta). Return
    the mean MSE and delta used."""
    results = []
    delta_used = mechanism_to_delta(mechanism)
    for seed in range(n_seeds):
        if mechanism == "MBUV-like-MAR":
            # Represent MBUV-on-label as MAR data (independent unobserved feature)
            rng = np.random.default_rng(seed)
            n = 500
            X = rng.standard_normal((n, 3))
            X_aug = np.concatenate([np.ones((n, 1)), X], axis=1)
            beta_true = np.array([0.5, 1.0, -0.5, 0.3])
            Y = X_aug @ beta_true + 0.1 * rng.standard_normal(n)
            response = rng.random(n) < 0.6
            Y_tilde = np.where(response, Y, 0.0)
            X_test = rng.standard_normal((500, 3))
            X_test_aug = np.concatenate([np.ones((500, 1)), X_test], axis=1)
            Y_test = X_test_aug @ beta_true + 0.1 * rng.standard_normal(500)
            mech_for_dispatch = "MBUV"
        elif mechanism == "MBOV_Lower":
            sc = _scenario_mbov_lower(seed)
            continue  # handled below via direct calls
        else:
            continue

        from christensen_core.estimator import ChristensenEstimator
        q_cls = adaptive_centered_q_for(mech_for_dispatch, response)
        est = ChristensenEstimator(q_class=q_cls, fit_intercept=True).fit(X, Y_tilde, response)
        y_pred = est.predict(X_test)
        results.append({
            "mse": float(np.mean((y_pred - Y_test) ** 2)),
            "delta": delta_used,
            "theta_star": est._theta_star.copy(),
        })
    return results


def main():
    print("=" * 78)
    print("ADAPTIVE DELTA TABLE (from christensen_core.reference_based_q.MECHANISM_DELTA)")
    print("=" * 78)
    for mech in ("MBOV_Lower", "MBOV_Higher", "MBOV_Stochastic", "MBOV_Centered", "MBUV"):
        print(f"  {mech:20} -> delta = {mechanism_to_delta(mech):.2f}")
    print(f"  DEFAULT (fallback)   -> delta = {mechanism_to_delta(None):.2f}")

    print()
    print("=" * 78)
    print("Scenario 1: MAR with q_true=0.6 (20 seeds)")
    print("=" * 78)
    mar = [_scenario_mar_constant_q(s) for s in range(20)]
    print(f"  mean q_hat:           {_mean(mar, 'q_hat'):.3f}")
    print(f"  OLS-on-respondents:   {_mean(mar, 'mse_ols'):.4f}")
    print(f"  Christ wide [.05,1]:  {_mean(mar, 'mse_wide'):.4f}  "
          f"(theta*≈{np.mean([m['theta_wide'][0] for m in mar]):.3f})")
    print(f"  Christ centered δ=.30:{_mean(mar, 'mse_c30'):.4f}  "
          f"(theta*≈{np.mean([m['theta_c30'][0] for m in mar]):.3f})")
    print(f"  Christ centered δ=.10:{_mean(mar, 'mse_c10'):.4f}  "
          f"(theta*≈{np.mean([m['theta_c10'][0] for m in mar]):.3f})")
    print(f"  Christ centered δ=.02:{_mean(mar, 'mse_c02'):.4f}  "
          f"(theta*≈{np.mean([m['theta_c02'][0] for m in mar]):.3f})")

    # Adaptive lookup: for MAR-like data (no mechanism metadata), fallback delta=0.30
    # For MBUV-labeled data (MAR on label), adaptive delta=0.05 — tight ball.
    print("  -- adaptive dispatch --")
    adaptive_mbuv = _scenario_adaptive_by_mechanism("MBUV-like-MAR", n_seeds=20)
    if adaptive_mbuv:
        print(f"  Christ adaptive(MBUV): {_mean(adaptive_mbuv, 'mse'):.4f}  "
              f"(delta={adaptive_mbuv[0]['delta']:.2f}, "
              f"theta*≈{np.mean([m['theta_star'][0] for m in adaptive_mbuv]):.3f})")

    print()
    print("=" * 78)
    print("Scenario 2: MBOV_Lower q(0)=0.3, q(1)=0.9, binary Y (30 seeds)")
    print("=" * 78)
    mb = [_scenario_mbov_lower(s) for s in range(30)]
    print(f"  mean q_hat:           {_mean(mb, 'q_hat'):.3f}")
    print(f"  ERM-on-respondents:   {_mean(mb, 'mse_ols'):.4f}")
    print(f"  Christ wide [.05,1]:  {_mean(mb, 'mse_wide'):.4f}  "
          f"(theta*≈{np.mean([m['theta_wide'][0] for m in mb]):.3f},"
          f" {np.mean([m['theta_wide'][1] for m in mb]):.3f})")
    print(f"  Christ centered δ=.30:{_mean(mb, 'mse_c30'):.4f}  "
          f"(theta*≈{np.mean([m['theta_c30'][0] for m in mb]):.3f},"
          f" {np.mean([m['theta_c30'][1] for m in mb]):.3f})")
    print(f"  Christ centered δ=.10:{_mean(mb, 'mse_c10'):.4f}  "
          f"(theta*≈{np.mean([m['theta_c10'][0] for m in mb]):.3f},"
          f" {np.mean([m['theta_c10'][1] for m in mb]):.3f})")

    # Adaptive for MBOV_Lower: delta = mechanism_to_delta("MBOV_Lower") = 0.30
    print("  -- adaptive dispatch --")
    from christensen_core.estimator import ChristensenEstimator as _CE
    mb_adapt_mses = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        n = 500
        X = rng.standard_normal((n, 3))
        X_aug = np.concatenate([np.ones((n, 1)), X], axis=1)
        beta_true = np.array([0.3, 0.8, -0.4, 0.2])
        y_cont = X_aug @ beta_true + 0.1 * rng.standard_normal(n)
        Y = (y_cont > 0).astype(float)
        q_true = np.where(Y == 0, 0.3, 0.9)
        response = rng.random(n) < q_true
        Y_tilde = np.where(response, Y, 0.0)
        X_test = rng.standard_normal((500, 3))
        X_test_aug = np.concatenate([np.ones((500, 1)), X_test], axis=1)
        y_cont_test = X_test_aug @ beta_true + 0.1 * rng.standard_normal(500)
        Y_test = (y_cont_test > 0).astype(float)

        q_cls = adaptive_centered_q_for("MBOV_Lower", response)
        est = _CE(q_class=q_cls, fit_intercept=True).fit(X, Y_tilde, response)
        y_pred = est.predict(X_test)
        mb_adapt_mses.append(float(np.mean((y_pred - Y_test) ** 2)))
    print(f"  Christ adaptive(MBOV_Lower): {float(np.mean(mb_adapt_mses)):.4f}  "
          f"(delta={mechanism_to_delta('MBOV_Lower'):.2f})")

    print()
    print("=" * 78)
    print("Scenario 3: Wrong Q (fit increasing, truth decreasing) (15 seeds)")
    print("=" * 78)
    wq = [_scenario_wrong_q(s) for s in range(15)]
    print(f"  mean q_hat:           {_mean(wq, 'q_hat'):.3f}")
    print(f"  ERM-on-respondents:   {_mean(wq, 'mse_ols'):.4f}")
    print(f"  Christ wide [.05,1]:  {_mean(wq, 'mse_wide'):.4f}  "
          f"(ratio vs ERM: {_mean(wq, 'mse_wide') / _mean(wq, 'mse_ols'):.2f}x)")
    print(f"  Christ centered δ=.30:{_mean(wq, 'mse_c30'):.4f}  "
          f"(ratio vs ERM: {_mean(wq, 'mse_c30') / _mean(wq, 'mse_ols'):.2f}x)")
    print(f"  Christ centered δ=.10:{_mean(wq, 'mse_c10'):.4f}  "
          f"(ratio vs ERM: {_mean(wq, 'mse_c10') / _mean(wq, 'mse_ols'):.2f}x)")


if __name__ == "__main__":
    main()
