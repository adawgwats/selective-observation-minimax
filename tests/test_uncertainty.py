from minimax_core import (
    KnightianObservationSet,
    Q1ObjectiveConfig,
    SurpriseDrivenObservationSet,
    TimeVaryingObservationSet,
)
from minimax_core.uncertainty import project_to_boxed_weighted_mean, weighted_mean


def test_projection_preserves_box_and_mean() -> None:
    projected = project_to_boxed_weighted_mean(
        values=[0.1, 0.9, 1.4],
        weights=[0.2, 0.3, 0.5],
        lower_bounds=[0.25, 0.25, 0.25],
        upper_bounds=[1.0, 1.0, 1.0],
        target_mean=0.65,
    )

    assert all(0.25 <= value <= 1.0 for value in projected)
    assert abs(weighted_mean(projected, [0.2, 0.3, 0.5]) - 0.65) < 1e-8


def test_time_varying_projection_weights_decay_over_time() -> None:
    uncertainty_set = TimeVaryingObservationSet(
        Q1ObjectiveConfig(q_min=0.25, q_max=1.0),
        time_strength=0.8,
        min_projection_weight=0.2,
    )

    weights = uncertainty_set.projection_weights([0, 1, 2, 3])

    assert weights[0] > weights[-1]
    assert all(0.2 <= weight <= 1.0 for weight in weights)


def test_time_varying_projection_preserves_temporal_weighted_mean() -> None:
    uncertainty_set = TimeVaryingObservationSet(
        Q1ObjectiveConfig(q_min=0.25, q_max=1.0),
        time_strength=0.6,
    )
    time_indices = [0, 1, 2, 3]
    projected = uncertainty_set.project(
        proposed_q=[0.2, 0.25, 0.9, 1.2],
        observation_rate=0.6,
        time_indices=time_indices,
    )
    weights = uncertainty_set.projection_weights(time_indices)

    assert all(0.25 <= value <= 1.0 for value in projected)
    assert abs(weighted_mean(projected, weights) - 0.6) < 1e-8


def test_knightian_projection_weights_increase_with_history() -> None:
    uncertainty_set = KnightianObservationSet(
        Q1ObjectiveConfig(q_min=0.25, q_max=1.0),
        time_strength=0.4,
        history_strength=1.0,
    )

    weights = uncertainty_set.projection_weights(
        time_indices=[0, 1, 2, 3],
        history_scores=[0.0, 0.0, 1.0, 2.0],
    )

    assert weights[0] > weights[-1]
    assert all(0.2 <= weight <= 1.0 for weight in weights)


def test_surprise_projection_weights_increase_with_surprise() -> None:
    uncertainty_set = SurpriseDrivenObservationSet(
        Q1ObjectiveConfig(q_min=0.25, q_max=1.0),
        surprise_strength=1.4,
    )

    weights = uncertainty_set.projection_weights(
        time_indices=[0, 1, 2, 3],
        history_scores=[0.0, 0.0, 0.0, 0.0],
        surprise_scores=[0.0, 0.1, 0.4, 1.2],
    )

    assert weights[0] > weights[-1]
    assert all(0.15 <= weight <= 1.0 for weight in weights)
