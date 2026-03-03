from experiments.wilds_civilcomments.metrics import (
    binary_auroc,
    compute_civilcomments_metrics,
    format_split_metrics,
    logits_to_predictions_and_scores,
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


def test_binary_auroc_matches_expected_rank_statistic() -> None:
    auc = binary_auroc(
        labels=[0, 1, 1, 0],
        positive_scores=[0.1, 0.3, 0.9, 0.8],
    )

    assert auc == 0.75


def test_logits_to_predictions_and_scores_uses_softmax_for_two_class_logits() -> None:
    predicted_labels, positive_scores = logits_to_predictions_and_scores(
        [
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )

    assert predicted_labels == [0, 1]
    assert positive_scores[0] < 0.5
    assert positive_scores[1] > 0.5


def test_compute_civilcomments_metrics_tracks_worst_group_accuracy_and_auroc() -> None:
    metadata_rows = [
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    ]

    metrics = compute_civilcomments_metrics(
        labels=[0, 1, 1, 0],
        predicted_labels=[0, 0, 1, 1],
        positive_scores=[0.1, 0.3, 0.9, 0.8],
        metadata_rows=metadata_rows,
        metadata_fields=METADATA_FIELDS,
    )

    assert metrics.overall_accuracy == 0.5
    assert metrics.overall_auroc == 0.75
    assert metrics.group_accuracy["female:1,y:0"] == 1.0
    assert metrics.group_accuracy["female:1,y:1"] == 0.0
    assert metrics.group_accuracy["black:1,y:1"] == 0.5
    assert metrics.group_accuracy["male:1,y:0"] == 0.0
    assert metrics.worst_group_accuracy == 0.0
    assert metrics.group_auroc["female"] == 1.0
    assert metrics.worst_group_auroc == 1.0


def test_format_split_metrics_renders_compact_summary() -> None:
    metadata_rows = [
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    ]
    metrics = compute_civilcomments_metrics(
        labels=[0, 1],
        predicted_labels=[0, 1],
        positive_scores=[0.2, 0.8],
        metadata_rows=metadata_rows,
        metadata_fields=METADATA_FIELDS,
    )

    rendered = format_split_metrics("val", metrics)

    assert rendered.startswith("val: accuracy=1.0000")
    assert "worst_group_accuracy=" in rendered
