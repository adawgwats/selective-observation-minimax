import json

from experiments.wilds_civilcomments.common import (
    CivilCommentsExperimentConfig,
    NON_IDENTITY_GROUP,
    build_observed_mask,
    load_experiment_config,
    extract_training_group_memberships,
    metadata_row_to_dict,
    summarize_memberships,
    synthetic_observation_probability,
)


METADATA_FIELDS = [
    "male",
    "female",
    "LGBTQ",
    "christian",
    "muslim",
    "other_religions",
    "black",
    "white",
    "identity_any",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
    "y",
]


def test_extract_training_group_memberships_uses_all_active_identities() -> None:
    metadata_row = [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1]

    memberships = extract_training_group_memberships(metadata_row, METADATA_FIELDS)

    assert memberships == ["female", "black"]


def test_extract_training_group_memberships_falls_back_to_identity_none() -> None:
    metadata_row = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    memberships = extract_training_group_memberships(metadata_row, METADATA_FIELDS)

    assert memberships == [NON_IDENTITY_GROUP]


def test_metadata_row_to_dict_supports_mapping_input() -> None:
    metadata = metadata_row_to_dict({"male": True, "y": 1}, ["male", "y"])

    assert metadata == {"male": 1, "y": 1}


def test_synthetic_observation_probability_penalizes_toxic_identity_examples() -> None:
    config = CivilCommentsExperimentConfig(
        explicit_mnar=True,
        base_observation_rate=0.95,
        toxic_penalty=0.20,
        identity_penalty=0.10,
        identity_toxic_interaction_penalty=0.15,
        min_observation_rate=0.05,
    )
    toxic_identity = [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1]
    neutral = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    toxic_probability = synthetic_observation_probability(
        toxic_identity,
        METADATA_FIELDS,
        config,
    )
    neutral_probability = synthetic_observation_probability(
        neutral,
        METADATA_FIELDS,
        config,
    )

    assert toxic_probability == 0.5
    assert neutral_probability == 0.95


def test_build_observed_mask_defaults_to_all_observed_without_explicit_mnar() -> None:
    metadata_rows = [
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    observed_mask = build_observed_mask(
        metadata_rows,
        METADATA_FIELDS,
        split_name="train",
        config=CivilCommentsExperimentConfig(explicit_mnar=False),
    )

    assert observed_mask == [True, True]


def test_build_observed_mask_preserves_group_coverage_after_sampling() -> None:
    config = CivilCommentsExperimentConfig(
        explicit_mnar=True,
        base_observation_rate=0.05,
        toxic_penalty=0.0,
        identity_penalty=0.0,
        identity_toxic_interaction_penalty=0.0,
        min_observation_rate=0.05,
        seed=3,
    )
    metadata_rows = [
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    ]

    observed_mask = build_observed_mask(
        metadata_rows,
        METADATA_FIELDS,
        split_name="train",
        config=config,
    )
    memberships = [
        extract_training_group_memberships(row, METADATA_FIELDS)
        for row in metadata_rows
    ]
    summary = summarize_memberships(memberships, observed_mask)

    assert summary["female"]["observed"] >= 1
    assert summary["black"]["observed"] >= 1


def test_load_experiment_config_supports_json(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "method": "robust_group",
                "explicit_mnar": True,
                "seed": 19,
            }
        ),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config == CivilCommentsExperimentConfig(
        method="robust_group",
        explicit_mnar=True,
        seed=19,
    )
