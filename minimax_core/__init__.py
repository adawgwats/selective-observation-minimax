from .adversary import ScoreBasedObservationAdversary, SelectiveObservationAdversary
from .config import Q1ObjectiveConfig
from .metrics import normalized_group_weights, worst_group_loss
from .objectives import (
    GroupSnapshot,
    compute_example_weights,
    compute_score_based_weights,
    empirical_risk,
    estimate_group_snapshot,
    observed_empirical_risk,
    robust_risk,
    score_based_risk,
)
from .uncertainty import ScoreBasedObservationSet, SelectiveObservationSet

__all__ = [
    "BaselineComparisonConfig",
    "GradientTrialResult",
    "GradientValidationConfig",
    "GradientValidationSummary",
    "GroupSnapshot",
    "MethodSummary",
    "MonteCarloConfig",
    "MonteCarloSummary",
    "Q1ObjectiveConfig",
    "ScenarioComparisonSummary",
    "ScoreBasedObservationAdversary",
    "ScoreBasedObservationSet",
    "SelectiveObservationAdversary",
    "SelectiveObservationSet",
    "TrialResult",
    "compute_example_weights",
    "compute_score_based_weights",
    "empirical_risk",
    "estimate_group_snapshot",
    "normalized_group_weights",
    "observed_empirical_risk",
    "robust_risk",
    "score_based_risk",
    "run_monte_carlo",
    "run_gradient_validation",
    "run_gradient_validation_suite",
    "run_baseline_comparison",
    "run_baseline_comparison_suite",
    "run_validation_suite",
    "worst_group_loss",
]


def __getattr__(name: str):
    if name in {
        "BaselineComparisonConfig",
        "GradientTrialResult",
        "GradientValidationConfig",
        "GradientValidationSummary",
        "MethodSummary",
        "MonteCarloConfig",
        "MonteCarloSummary",
        "ScenarioComparisonSummary",
        "TrialResult",
        "run_baseline_comparison",
        "run_baseline_comparison_suite",
        "run_gradient_validation",
        "run_gradient_validation_suite",
        "run_monte_carlo",
        "run_validation_suite",
    }:
        from .comparison import (
            BaselineComparisonConfig,
            MethodSummary,
            ScenarioComparisonSummary,
            run_baseline_comparison,
            run_baseline_comparison_suite,
        )
        from .gradient_validation import (
            GradientTrialResult,
            GradientValidationConfig,
            GradientValidationSummary,
            run_gradient_validation,
            run_gradient_validation_suite,
        )
        from .monte_carlo import (
            MonteCarloConfig,
            MonteCarloSummary,
            TrialResult,
            run_monte_carlo,
            run_validation_suite,
        )

        mapping = {
            "BaselineComparisonConfig": BaselineComparisonConfig,
            "GradientTrialResult": GradientTrialResult,
            "GradientValidationConfig": GradientValidationConfig,
            "GradientValidationSummary": GradientValidationSummary,
            "MethodSummary": MethodSummary,
            "MonteCarloConfig": MonteCarloConfig,
            "MonteCarloSummary": MonteCarloSummary,
            "ScenarioComparisonSummary": ScenarioComparisonSummary,
            "TrialResult": TrialResult,
            "run_baseline_comparison": run_baseline_comparison,
            "run_baseline_comparison_suite": run_baseline_comparison_suite,
            "run_gradient_validation": run_gradient_validation,
            "run_gradient_validation_suite": run_gradient_validation_suite,
            "run_monte_carlo": run_monte_carlo,
            "run_validation_suite": run_validation_suite,
        }
        return mapping[name]
    raise AttributeError(f"module 'minimax_core' has no attribute {name!r}")
