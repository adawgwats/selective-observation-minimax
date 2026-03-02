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
