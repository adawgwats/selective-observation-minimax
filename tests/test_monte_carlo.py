from minimax_core import MonteCarloConfig, Q1ObjectiveConfig, run_monte_carlo, run_validation_suite


def test_monte_carlo_robust_beats_erm_on_fixed_seed() -> None:
    _trials, summary = run_monte_carlo(
        MonteCarloConfig(
            seed=11,
            trials=40,
            adversary_iterations=100,
            theta_step=0.02,
            q1=Q1ObjectiveConfig(q_min=0.25, q_max=1.0, adversary_step_size=0.05),
        )
    )

    assert summary.mean_robust_latent_risk < summary.mean_erm_latent_risk
    assert summary.mean_robust_theta_error < summary.mean_erm_theta_error
    assert summary.robust_beats_erm_rate > 0.75
    assert summary.robust_closer_to_latent_theta_rate > 0.75


def test_validation_suite_includes_negative_control_behavior() -> None:
    summaries = run_validation_suite(
        MonteCarloConfig(
            seed=11,
            trials=30,
            adversary_iterations=100,
            theta_step=0.02,
            q1=Q1ObjectiveConfig(q_min=0.25, q_max=1.0, adversary_step_size=0.05),
        )
    )

    aligned = summaries["aligned_selective"]
    agnostic = summaries["group_agnostic"]
    label_dependent = summaries["label_dependent"]

    assert aligned.mean_latent_risk_improvement > 0.02
    assert aligned.robust_beats_erm_rate > 0.9

    assert agnostic.mean_latent_risk_improvement < 0.002
    assert agnostic.robust_beats_erm_rate < 0.6

    assert label_dependent.mean_latent_risk_improvement > 0.05
    assert label_dependent.robust_beats_erm_rate > 0.9


def test_score_based_monte_carlo_improves_over_erm_in_aligned_setting() -> None:
    _trials, summary = run_monte_carlo(
        MonteCarloConfig(
            seed=11,
            trials=6,
            scenario="aligned_selective",
            adversary_mode="score",
            adversary_iterations=30,
            theta_step=0.04,
            q1=Q1ObjectiveConfig(q_min=0.25, q_max=1.0, adversary_step_size=0.05),
        )
    )

    assert summary.mean_robust_latent_risk < summary.mean_erm_latent_risk
    assert summary.robust_beats_erm_rate > 0.6
