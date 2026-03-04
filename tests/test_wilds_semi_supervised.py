import pytest

from experiments.wilds_civilcomments.semi_supervised import (
    build_pseudo_labeled_dataset,
    select_pseudo_labels,
)


def test_select_pseudo_labels_filters_by_threshold() -> None:
    selected = select_pseudo_labels(
        [0.99, 0.88, 0.08, 0.54, 0.03],
        threshold=0.90,
    )
    assert selected["indices"] == [0, 2, 4]
    assert selected["labels"] == [1, 0, 0]


def test_select_pseudo_labels_validates_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        select_pseudo_labels([0.9], threshold=0.49)


def test_build_pseudo_labeled_dataset_selects_requested_rows() -> None:
    dataset = build_pseudo_labeled_dataset(
        encodings={
            "input_ids": [[101, 1], [101, 2], [101, 3]],
            "attention_mask": [[1, 1], [1, 1], [1, 1]],
        },
        group_memberships=[["group_a"], [], ["group_b"]],
        selected_indices=[2, 0],
        pseudo_labels=[1, 0],
    )

    assert len(dataset) == 2
    first = dataset[0]
    second = dataset[1]
    assert first["labels"] == 1
    assert first["group_id"] == ["group_b"]
    assert second["labels"] == 0
    assert second["group_id"] == ["group_a"]
