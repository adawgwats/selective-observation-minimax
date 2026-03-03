from __future__ import annotations

from minimax_core.mnar import SyntheticMNARConfig, apply_synthetic_mnar, build_proxy_labels


def test_explicit_missing_keeps_rows_but_marks_unobserved() -> None:
    result = apply_synthetic_mnar(
        labels=[1.0, -1.0, -2.0],
        group_ids=["stable", "distressed", "distressed"],
        path_indices=[0, 0, 1],
        step_indices=[0, 1, 0],
        weather_regimes=["normal", "drought", "normal"],
        farm_alive_next_year=[True, False, True],
        config=SyntheticMNARConfig(
            seed=3,
            view_mode="explicit_missing",
            base_observation_probability=0.0,
            min_observation_probability=0.0,
            max_observation_probability=1.0,
        ),
    )

    assert result.keep_mask == (True, True, True)
    assert any(not observed for observed in result.observed_mask)


def test_drop_unobserved_hides_missing_rows_from_training_view() -> None:
    result = apply_synthetic_mnar(
        labels=[1.0, -1.0, -2.0],
        group_ids=["stable", "distressed", "distressed"],
        path_indices=[0, 0, 1],
        step_indices=[0, 1, 0],
        config=SyntheticMNARConfig(
            seed=5,
            view_mode="drop_unobserved",
            base_observation_probability=0.0,
            min_observation_probability=0.0,
            max_observation_probability=1.0,
        ),
    )

    assert result.keep_mask == result.observed_mask
    assert sum(1 for keep in result.keep_mask if keep) >= 1


def test_truncate_after_unobserved_drops_future_rows_on_same_path() -> None:
    result = apply_synthetic_mnar(
        labels=[1.0, -1.0, 1.0],
        group_ids=["stable", "distressed", "stable"],
        path_indices=[0, 0, 0],
        step_indices=[0, 1, 2],
        config=SyntheticMNARConfig(
            seed=7,
            view_mode="truncate_after_unobserved",
            base_observation_probability=1.0,
            distressed_penalty=1.0,
            min_observation_probability=0.0,
            max_observation_probability=1.0,
        ),
    )

    assert result.keep_mask[0] is True
    assert result.keep_mask[1] is False
    assert result.keep_mask[2] is False


def test_build_proxy_labels_uses_group_then_global_means() -> None:
    proxies = build_proxy_labels(
        observed_values=[1.0, None, None],
        group_ids=["stable", "stable", "distressed"],
        observed_mask=[True, False, False],
        label_scale=1.0,
    )

    assert proxies == [1.0, 1.0, 1.0]
