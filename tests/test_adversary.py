from minimax_core import (
    AutoDiscoveryObservationAdversary,
    KnightianObservationAdversary,
    Q1ObjectiveConfig,
    RupturesStructuralBreakDetector,
    ScoreBasedObservationAdversary,
    SelectiveObservationAdversary,
    StructuralBreakObservationAdversary,
    SurpriseDrivenObservationAdversary,
    TimeVaryingObservationAdversary,
    estimate_group_snapshot,
)


def test_high_loss_group_gets_lower_observation_probability() -> None:
    snapshot = estimate_group_snapshot(
        losses=[0.1, 0.1, 0.9, 0.9],
        group_ids=["stable", "stable", "distressed", "distressed"],
        observed_mask=[True, True, True, False],
    )
    adversary = SelectiveObservationAdversary(
        Q1ObjectiveConfig(q_min=0.25, q_max=1.0, adversary_step_size=0.1)
    )

    initial_q = adversary.current_q(snapshot)
    updated_q = adversary.update(snapshot)

    assert updated_q["distressed"] < initial_q["distressed"]
    assert updated_q["stable"] > initial_q["stable"]
    assert abs(
        0.5 * updated_q["stable"] + 0.5 * updated_q["distressed"] - snapshot.observation_rate
    ) < 1e-8


def test_q_values_stay_within_bounds_after_many_updates() -> None:
    snapshot = estimate_group_snapshot(
        losses=[0.05, 0.05, 1.2, 1.1],
        group_ids=["stable", "stable", "distressed", "distressed"],
        observed_mask=[True, True, True, False],
    )
    config = Q1ObjectiveConfig(q_min=0.3, q_max=0.95, adversary_step_size=0.1)
    adversary = SelectiveObservationAdversary(config)

    q_values = adversary.current_q(snapshot)
    for _ in range(100):
        q_values = adversary.update(snapshot)

    assert all(config.q_min - 1e-8 <= value <= config.q_max + 1e-8 for value in q_values.values())


def test_score_based_adversary_downweights_high_score_examples() -> None:
    config = Q1ObjectiveConfig(q_min=0.25, q_max=1.0, adversary_step_size=0.1)
    adversary = ScoreBasedObservationAdversary(config)
    scores = [0.1, 0.2, 1.0, 1.2]
    observation_rate = 0.6

    initial_q = adversary.current_q(scores, observation_rate)
    updated_q = adversary.update(scores, observation_rate)

    assert updated_q[3] < initial_q[3]
    assert updated_q[0] > initial_q[0]
    assert abs(sum(updated_q) / len(updated_q) - observation_rate) < 1e-8


def test_time_varying_adversary_downweights_later_high_risk_examples_more() -> None:
    config = Q1ObjectiveConfig(q_min=0.25, q_max=1.0, adversary_step_size=0.1)
    adversary = TimeVaryingObservationAdversary(config)
    scores = [0.6, 0.6, 0.6, 0.6]
    time_indices = [0, 1, 2, 3]
    observation_rate = 0.6

    initial_q = adversary.current_q(scores, observation_rate, time_indices)
    updated_q = adversary.update(scores, observation_rate, time_indices)
    weights = adversary.uncertainty_set.projection_weights(time_indices)

    assert updated_q[-1] < updated_q[0]
    assert updated_q[-1] < initial_q[-1]
    assert abs(sum(weight * q for weight, q in zip(weights, updated_q)) / sum(weights) - observation_rate) < 1e-8


def test_knightian_adversary_downweights_high_history_examples_more() -> None:
    config = Q1ObjectiveConfig(q_min=0.25, q_max=1.0, adversary_step_size=0.1)
    adversary = KnightianObservationAdversary(config)
    scores = [0.6, 0.6, 0.6, 0.6]
    time_indices = [0, 1, 2, 3]
    history_scores = [0.0, 0.0, 1.0, 2.0]
    observation_rate = 0.6

    updated_q = adversary.update(scores, observation_rate, time_indices, history_scores)
    weights = adversary.uncertainty_set.projection_weights(time_indices, history_scores)

    assert updated_q[-1] < updated_q[0]
    assert abs(sum(weight * q for weight, q in zip(weights, updated_q)) / sum(weights) - observation_rate) < 1e-8


def test_surprise_adversary_accumulates_residual_shocks() -> None:
    config = Q1ObjectiveConfig(q_min=0.25, q_max=1.0, adversary_step_size=0.1)
    adversary = SurpriseDrivenObservationAdversary(config, surprise_decay=0.5)
    time_indices = [0, 1, 2, 3]
    history_scores = [0.0, 0.0, 0.0, 0.0]
    observation_rate = 0.6

    adversary.update([0.2, 0.2, 0.2, 0.2], observation_rate, time_indices, history_scores)
    updated_q = adversary.update([0.2, 0.2, 0.2, 1.2], observation_rate, time_indices, history_scores)
    surprise_scores = adversary.current_surprise_scores()

    assert surprise_scores[-1] > surprise_scores[0]
    assert updated_q[-1] < updated_q[0]


def test_auto_discovery_adversary_accumulates_online_state() -> None:
    config = Q1ObjectiveConfig(q_min=0.25, q_max=1.0, adversary_step_size=0.1)
    adversary = AutoDiscoveryObservationAdversary(config, score_decay=0.5, history_decay=0.5)
    observation_rate = 0.6

    initial_q = adversary.current_q([0.2, 0.2, 0.2, 0.2], observation_rate, [True, True, True, True])
    updated_q = adversary.update([0.2, 0.2, 0.2, 1.2], observation_rate, [True, True, True, False])

    assert adversary._seen_examples == 4
    assert adversary._surprise_state > 0.0
    assert adversary._history_state > 0.0
    assert updated_q[-1] < initial_q[-1]


def test_structural_break_detector_marks_post_break_examples() -> None:
    detector = RupturesStructuralBreakDetector(min_size=2, min_normalized_shift=0.1, break_decay=0.9)
    result = detector.detect(
        scores=[0.1, 0.1, 0.1, 1.0, 1.0, 1.0],
        time_indices=[0, 1, 2, 3, 4, 5],
        path_ids=[0, 0, 0, 0, 0, 0],
    )

    assert max(result.break_scores[:3]) == 0.0
    assert max(result.break_scores[3:]) > 0.0
    assert result.breakpoints


def test_structural_break_adversary_downweights_post_break_examples_more() -> None:
    config = Q1ObjectiveConfig(q_min=0.25, q_max=1.0, adversary_step_size=0.1)
    adversary = StructuralBreakObservationAdversary(config)
    scores = [0.2, 0.2, 0.2, 1.1, 1.1, 1.1]
    time_indices = [0, 1, 2, 3, 4, 5]
    history_scores = [0.0, 0.0, 0.2, 0.4, 0.6, 0.8]
    path_ids = [0, 0, 0, 0, 0, 0]
    observation_rate = 0.6

    updated_q = adversary.update(scores, observation_rate, time_indices, history_scores, path_ids)
    break_scores = adversary.current_break_scores()

    assert max(break_scores[:3]) == 0.0
    assert max(break_scores[3:]) > 0.0
    assert updated_q[-1] < updated_q[0]
