from experiments.wilds_civilcomments.report import render_metrics_report


def test_render_metrics_report_formats_methods_and_splits() -> None:
    report = render_metrics_report(
        [
            {
                "config": {"method": "erm"},
                "train": {"observed_examples": 64, "total_examples": 64},
                "val": {
                    "overall_accuracy": 0.9,
                    "worst_group_accuracy": 0.2,
                    "overall_auroc": 0.7,
                    "worst_group_auroc": 0.4,
                },
                "test": {
                    "overall_accuracy": 0.8,
                    "worst_group_accuracy": 0.1,
                    "overall_auroc": 0.6,
                    "worst_group_auroc": 0.3,
                },
            },
            {
                "config": {"method": "robust_group"},
                "train": {"observed_examples": 54, "total_examples": 64},
                "val": {
                    "overall_accuracy": 0.91,
                    "worst_group_accuracy": 0.25,
                    "overall_auroc": 0.71,
                    "worst_group_auroc": 0.45,
                },
                "test": {
                    "overall_accuracy": 0.82,
                    "worst_group_accuracy": 0.12,
                    "overall_auroc": 0.61,
                    "worst_group_auroc": 0.31,
                },
            },
        ]
    )

    assert "method        split" in report
    assert "erm           val" in report
    assert "robust_group  test" in report
    assert "64/64" in report
    assert "54/64" in report
