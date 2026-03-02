from minimax_core import (
    GradientValidationConfig,
    Q1ObjectiveConfig,
    run_gradient_validation_suite,
)


def test_gradient_validation_suite_shows_expected_pattern() -> None:
    summaries = run_gradient_validation_suite(
        GradientValidationConfig(
            seed=17,
            trials=12,
            epochs=120,
            learning_rate=0.05,
            q1=Q1ObjectiveConfig(q_min=0.25, q_max=1.0, adversary_step_size=0.05),
        )
    )

    aligned = summaries["aligned_selective"]
    agnostic = summaries["group_agnostic"]
    label_dependent = summaries["label_dependent"]

    assert aligned.mean_test_mse_improvement > 0.001
    assert aligned.robust_beats_erm_rate > 0.75

    assert agnostic.mean_test_mse_improvement < 0.001

    assert label_dependent.mean_test_mse_improvement > 0.01
    assert label_dependent.robust_beats_erm_rate > 0.9


def test_score_based_gradient_validation_improves_in_aligned_setting() -> None:
    summaries = run_gradient_validation_suite(
        GradientValidationConfig(
            seed=17,
            trials=8,
            epochs=100,
            learning_rate=0.05,
            adversary_mode="score",
            q1=Q1ObjectiveConfig(q_min=0.25, q_max=1.0, adversary_step_size=0.05),
        ),
        scenarios=("aligned_selective",),
    )

    aligned = summaries["aligned_selective"]

    assert aligned.mean_test_mse_improvement > 0.001
    assert aligned.robust_beats_erm_rate > 0.75
