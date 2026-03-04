import pytest

from experiments.wilds_civilcomments.multiseed import (
    aggregate_multiseed_metrics,
    render_multiseed_summary,
)


def test_aggregate_multiseed_metrics_computes_mean_and_std() -> None:
    artifacts = [
        {
            "val": {
                "overall_accuracy": 0.9,
                "worst_group_accuracy": 0.4,
                "overall_auroc": 0.8,
                "worst_group_auroc": 0.6,
                "wilds_eval": {"acc_avg": 0.9, "acc_wg": 0.4},
            },
            "test": {
                "overall_accuracy": 0.91,
                "worst_group_accuracy": 0.45,
                "overall_auroc": 0.81,
                "worst_group_auroc": 0.61,
                "wilds_eval": {"acc_avg": 0.91, "acc_wg": 0.45},
            },
        },
        {
            "val": {
                "overall_accuracy": 0.8,
                "worst_group_accuracy": 0.3,
                "overall_auroc": 0.7,
                "worst_group_auroc": 0.5,
                "wilds_eval": {"acc_avg": 0.8, "acc_wg": 0.3},
            },
            "test": {
                "overall_accuracy": 0.82,
                "worst_group_accuracy": 0.35,
                "overall_auroc": 0.72,
                "worst_group_auroc": 0.52,
                "wilds_eval": {"acc_avg": 0.82, "acc_wg": 0.35},
            },
        },
    ]

    summary = aggregate_multiseed_metrics(
        artifacts=artifacts,
        base_config={"method": "robust_auto_v1"},
        seeds=[17, 23],
        output_root="outputs/demo_multiseed",
        config_path="demo.yaml",
        seed_runs=[
            {"seed": 17, "output_dir": "outputs/demo_multiseed/seed_17"},
            {"seed": 23, "output_dir": "outputs/demo_multiseed/seed_23"},
        ],
    )

    assert summary["method"] == "robust_auto_v1"
    assert summary["num_seeds"] == 2
    assert summary["val"]["overall_accuracy"]["mean"] == pytest.approx(0.85)
    assert round(float(summary["val"]["overall_accuracy"]["std"]), 4) == 0.0707
    assert summary["test"]["wilds_acc_wg"]["mean"] == pytest.approx(0.4)


def test_render_multiseed_summary_formats_expected_fields() -> None:
    summary = {
        "method": "erm",
        "seeds": [17, 23, 29, 31, 37],
        "output_root": "outputs/demo_multiseed",
        "val": {
            "overall_accuracy": {"mean": 0.91, "std": 0.01},
            "worst_group_accuracy": {"mean": 0.43, "std": 0.02},
            "overall_auroc": {"mean": 0.92, "std": 0.01},
            "worst_group_auroc": {"mean": 0.74, "std": 0.03},
            "wilds_acc_avg": {"mean": 0.91, "std": 0.01},
            "wilds_acc_wg": {"mean": 0.43, "std": 0.02},
        },
        "test": {
            "overall_accuracy": {"mean": 0.92, "std": 0.01},
            "worst_group_accuracy": {"mean": 0.52, "std": 0.04},
            "overall_auroc": {"mean": 0.96, "std": 0.01},
            "worst_group_auroc": {"mean": 0.88, "std": 0.02},
            "wilds_acc_avg": {"mean": 0.92, "std": 0.01},
            "wilds_acc_wg": {"mean": 0.52, "std": 0.04},
        },
    }

    rendered = render_multiseed_summary(summary)

    assert "method: erm" in rendered
    assert "17, 23, 29, 31, 37" in rendered
    assert "worst_group_accuracy" in rendered
    assert "wilds_acc_wg" in rendered
