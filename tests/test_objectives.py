from minimax_core import (
    Q1ObjectiveConfig,
    SelectiveObservationAdversary,
    compute_example_weights,
    estimate_group_snapshot,
    robust_risk,
)


def test_example_weights_reconstruct_robust_objective() -> None:
    losses = [0.2, 0.4, 0.8, 1.0]
    group_ids = ["stable", "stable", "distressed", "distressed"]
    observed_mask = [True, True, True, True]
    snapshot = estimate_group_snapshot(losses, group_ids, observed_mask)
    adversary = SelectiveObservationAdversary(
        Q1ObjectiveConfig(q_min=0.25, q_max=1.0, adversary_step_size=0.05)
    )

    q_values = adversary.update(snapshot)
    example_weights = compute_example_weights(snapshot, group_ids, observed_mask, q_values)
    weighted_sum = sum(weight * loss for weight, loss in zip(example_weights, losses))

    assert abs(weighted_sum - robust_risk(snapshot, q_values)) < 1e-10
