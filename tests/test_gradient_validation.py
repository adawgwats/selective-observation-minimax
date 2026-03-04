import random

import pytest

from minimax_core import (
    GradientValidationConfig,
    Q1ObjectiveConfig,
    run_gradient_validation_suite,
)
from minimax_core.gradient_validation import (
    LinearDataset,
    _clip_observation_rate,
    _mse,
    _predict,
    generate_linear_dataset,
    train_erm,
    train_robust_group,
    train_robust_group_online,
    train_robust_knightian,
    train_robust_structural_break,
    train_robust_surprise,
    train_robust_time_varying,
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


def test_online_mnar_group_training_improves_after_hidden_row_drops() -> None:
    rng = random.Random(29)
    dataset, _true_parameters = generate_linear_dataset(
        rng,
        GradientValidationConfig(
            seed=29,
            scenario="aligned_selective",
            epochs=120,
        ),
    )

    retained_indices = [
        index for index, observed in enumerate(dataset.train_observed_mask) if observed
    ]
    dropped_dataset = LinearDataset(
        train_features=[dataset.train_features[index] for index in retained_indices],
        train_labels=[dataset.train_labels[index] for index in retained_indices],
        train_proxy_labels=[dataset.train_proxy_labels[index] for index in retained_indices],
        train_group_ids=[dataset.train_group_ids[index] for index in retained_indices],
        train_observed_mask=[True for _ in retained_indices],
        train_time_indices=[dataset.train_time_indices[index] for index in retained_indices],
        train_history_scores=[dataset.train_history_scores[index] for index in retained_indices],
        train_path_ids=[dataset.train_path_ids[index] for index in retained_indices],
        test_features=dataset.test_features,
        test_labels=dataset.test_labels,
        stable_observation_probability=dataset.stable_observation_probability,
        distressed_observation_probability=dataset.distressed_observation_probability,
    )
    assumed_rate = sum(1 for observed in dataset.train_observed_mask if observed) / len(dataset.train_observed_mask)
    config = GradientValidationConfig(
        seed=29,
        scenario="aligned_selective",
        epochs=120,
        assumed_observation_rate=assumed_rate,
    )

    erm_parameters = train_erm(dropped_dataset, config)
    robust_parameters = train_robust_group(dropped_dataset, config)
    online_parameters = train_robust_group_online(dropped_dataset, config)

    erm_mse = _mse(_predict(erm_parameters, dropped_dataset.test_features), dropped_dataset.test_labels)
    robust_mse = _mse(_predict(robust_parameters, dropped_dataset.test_features), dropped_dataset.test_labels)
    online_mse = _mse(_predict(online_parameters, dropped_dataset.test_features), dropped_dataset.test_labels)

    assert robust_mse == pytest.approx(erm_mse)
    assert online_mse < robust_mse


def test_clip_observation_rate_respects_q1_bounds() -> None:
    config = GradientValidationConfig(
        q1=Q1ObjectiveConfig(q_min=0.25, q_max=0.75, adversary_step_size=0.05)
    )

    assert _clip_observation_rate(0.10, config) == pytest.approx(0.25)
    assert _clip_observation_rate(0.90, config) == pytest.approx(0.75)


def test_time_varying_training_improves_with_later_hidden_risk() -> None:
    dataset = LinearDataset(
        train_features=[[1.0, 0.0], [1.0, 0.2], [1.0, 0.8], [1.0, 1.0]],
        train_labels=[0.2, 0.3, 1.2, 1.4],
        train_proxy_labels=[0.2, 0.3, 1.2, 1.4],
        train_group_ids=["stable", "stable", "distressed", "distressed"],
        train_observed_mask=[True, True, True, False],
        train_time_indices=[0, 1, 2, 3],
        train_history_scores=[0.0, 0.0, 1.0, 2.0],
        train_path_ids=[0, 0, 0, 0],
        test_features=[[1.0, 0.1], [1.0, 0.9]],
        test_labels=[0.25, 1.3],
        stable_observation_probability=1.0,
        distressed_observation_probability=0.5,
    )
    config = GradientValidationConfig(
        seed=41,
        epochs=120,
        learning_rate=0.05,
        adversary_mode="time_varying",
    )

    erm_parameters = train_erm(dataset, config)
    time_parameters = train_robust_time_varying(dataset, config)

    erm_mse = _mse(_predict(erm_parameters, dataset.test_features), dataset.test_labels)
    time_mse = _mse(_predict(time_parameters, dataset.test_features), dataset.test_labels)

    assert time_mse < erm_mse


def test_knightian_training_improves_with_compounding_history() -> None:
    dataset = LinearDataset(
        train_features=[[1.0, 0.0], [1.0, 0.2], [1.0, 0.8], [1.0, 1.0]],
        train_labels=[0.2, 0.3, 1.2, 1.4],
        train_proxy_labels=[0.2, 0.3, 1.2, 1.4],
        train_group_ids=["stable", "stable", "distressed", "distressed"],
        train_observed_mask=[True, True, True, False],
        train_time_indices=[0, 1, 2, 3],
        train_history_scores=[0.0, 0.0, 1.0, 2.0],
        train_path_ids=[0, 0, 0, 0],
        test_features=[[1.0, 0.1], [1.0, 0.9]],
        test_labels=[0.25, 1.3],
        stable_observation_probability=1.0,
        distressed_observation_probability=0.5,
    )
    config = GradientValidationConfig(
        seed=53,
        epochs=120,
        learning_rate=0.05,
        adversary_mode="knightian",
    )

    erm_parameters = train_erm(dataset, config)
    knightian_parameters = train_robust_knightian(dataset, config)

    erm_mse = _mse(_predict(erm_parameters, dataset.test_features), dataset.test_labels)
    knightian_mse = _mse(_predict(knightian_parameters, dataset.test_features), dataset.test_labels)

    assert knightian_mse < erm_mse


def test_surprise_training_improves_after_late_residual_shift() -> None:
    dataset = LinearDataset(
        train_features=[[1.0, 0.0], [1.0, 0.2], [1.0, 0.8], [1.0, 1.0]],
        train_labels=[0.2, 0.3, 1.2, 1.8],
        train_proxy_labels=[0.2, 0.3, 1.2, 0.9],
        train_group_ids=["stable", "stable", "distressed", "distressed"],
        train_observed_mask=[True, True, True, False],
        train_time_indices=[0, 1, 2, 3],
        train_history_scores=[0.0, 0.0, 0.5, 1.5],
        train_path_ids=[0, 0, 0, 0],
        test_features=[[1.0, 0.1], [1.0, 0.9], [1.0, 1.1]],
        test_labels=[0.25, 1.35, 1.75],
        stable_observation_probability=1.0,
        distressed_observation_probability=0.5,
    )
    config = GradientValidationConfig(
        seed=61,
        epochs=140,
        learning_rate=0.05,
        adversary_mode="surprise",
    )

    erm_parameters = train_erm(dataset, config)
    surprise_parameters = train_robust_surprise(dataset, config)

    erm_mse = _mse(_predict(erm_parameters, dataset.test_features), dataset.test_labels)
    surprise_mse = _mse(_predict(surprise_parameters, dataset.test_features), dataset.test_labels)

    assert surprise_mse < erm_mse


def test_structural_break_training_improves_after_regime_shift() -> None:
    dataset = LinearDataset(
        train_features=[
            [1.0, -0.2], [1.0, 0.0], [1.0, 0.1], [1.0, 0.2],
            [1.0, 0.6], [1.0, 0.8], [1.0, 1.0], [1.0, 1.2],
        ],
        train_labels=[0.1, 0.2, 0.2, 0.3, 1.0, 1.2, 1.3, 1.5],
        train_proxy_labels=[0.1, 0.2, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0],
        train_group_ids=["stable"] * 4 + ["distressed"] * 4,
        train_observed_mask=[True, True, True, True, True, False, False, False],
        train_time_indices=[0, 1, 2, 3, 4, 5, 6, 7],
        train_history_scores=[0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0],
        train_path_ids=[0, 0, 0, 0, 0, 0, 0, 0],
        test_features=[[1.0, 0.0], [1.0, 0.9], [1.0, 1.1]],
        test_labels=[0.2, 1.25, 1.45],
        stable_observation_probability=1.0,
        distressed_observation_probability=0.4,
    )
    config = GradientValidationConfig(
        seed=71,
        epochs=140,
        learning_rate=0.05,
        adversary_mode="structural_break",
    )

    erm_parameters = train_erm(dataset, config)
    structural_parameters = train_robust_structural_break(dataset, config)

    erm_mse = _mse(_predict(erm_parameters, dataset.test_features), dataset.test_labels)
    structural_mse = _mse(_predict(structural_parameters, dataset.test_features), dataset.test_labels)

    assert structural_mse < erm_mse
