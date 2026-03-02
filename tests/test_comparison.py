from minimax_core import BaselineComparisonConfig, run_baseline_comparison


def test_baseline_comparison_shows_robust_gain_in_aligned_setting() -> None:
    _trials, summary = run_baseline_comparison(
        BaselineComparisonConfig(
            seed=23,
            trials=8,
            scenario="aligned_selective",
            epochs=100,
        )
    )

    assert summary.methods["robust_group"].mean_test_mse < summary.methods["erm"].mean_test_mse
    assert summary.methods["robust_score"].mean_test_mse < summary.methods["erm"].mean_test_mse


def test_label_dependent_comparison_favors_grouped_minimax_over_simple_reweighting() -> None:
    _trials, summary = run_baseline_comparison(
        BaselineComparisonConfig(
            seed=23,
            trials=8,
            scenario="label_dependent",
            epochs=100,
        )
    )

    assert summary.methods["robust_group"].mean_test_mse < summary.methods["group_prior"].mean_test_mse
    assert summary.methods["robust_group"].mean_test_mse < summary.methods["group_dro"].mean_test_mse
    assert summary.methods["robust_group"].mean_test_mse < summary.methods["robust_score"].mean_test_mse
