from minimax_core import (
    Q1ObjectiveConfig,
    ScoreBasedObservationAdversary,
    SelectiveObservationAdversary,
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
