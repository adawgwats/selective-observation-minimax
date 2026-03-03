from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUMMARY_KEYS = (
    "overall_accuracy",
    "worst_group_accuracy",
    "overall_auroc",
    "worst_group_auroc",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a compact comparison table from CivilComments metrics artifacts."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="One or more metrics.json paths emitted by train.py or eval.py.",
    )
    return parser.parse_args(argv)


def load_metrics_artifact(path: str | Path) -> dict[str, Any]:
    metrics_path = Path(path)
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def render_metrics_report(artifacts: list[dict[str, Any]]) -> str:
    lines = [
        "method        split  accuracy  worst_acc  auroc   worst_auroc  observed/total",
        "------------  -----  --------  ---------  ------  -----------  --------------",
    ]
    for artifact in artifacts:
        method = artifact.get("config", {}).get("method", "unknown")
        observed = artifact.get("train", {}).get("observed_examples")
        total = artifact.get("train", {}).get("total_examples")
        observed_summary = "n/a" if observed is None or total is None else f"{observed}/{total}"
        for split in ("val", "test"):
            split_metrics = artifact.get(split, {})
            lines.append(
                f"{method:<12}"
                f"  {split:<5}"
                f"  {_format_metric(split_metrics.get('overall_accuracy')):>8}"
                f"  {_format_metric(split_metrics.get('worst_group_accuracy')):>9}"
                f"  {_format_metric(split_metrics.get('overall_auroc')):>6}"
                f"  {_format_metric(split_metrics.get('worst_group_auroc')):>11}"
                f"  {observed_summary:>14}"
            )
    return "\n".join(lines)


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    artifacts = [load_metrics_artifact(path) for path in args.metrics]
    print(render_metrics_report(artifacts))


if __name__ == "__main__":
    main()
