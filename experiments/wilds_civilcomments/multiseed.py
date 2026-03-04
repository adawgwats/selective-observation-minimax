from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.wilds_civilcomments.common import config_to_dict, load_experiment_config
from experiments.wilds_civilcomments.train import train_from_config


DEFAULT_SEEDS = (17, 23, 29, 31, 37)
SUMMARY_METRICS = (
    ("overall_accuracy", ("overall_accuracy",)),
    ("worst_group_accuracy", ("worst_group_accuracy",)),
    ("overall_auroc", ("overall_auroc",)),
    ("worst_group_auroc", ("worst_group_auroc",)),
    ("wilds_acc_avg", ("wilds_eval", "acc_avg")),
    ("wilds_acc_wg", ("wilds_eval", "acc_wg")),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WILDS CivilComments experiments across multiple seeds and aggregate metrics."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the base YAML or JSON experiment config.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="One or more integer seeds. Defaults to 5 fixed seeds for WILDS-style evaluation.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional directory for per-seed runs and the multiseed summary. Defaults to <config.output_dir>_multiseed.",
    )
    return parser.parse_args(argv)


def run_multiseed_experiment(
    *,
    config_path: str | Path,
    seeds: Sequence[int],
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    base_config = load_experiment_config(config_path)
    if not seeds:
        raise ValueError("at least one seed is required.")

    root_dir = Path(output_root) if output_root is not None else Path(f"{base_config.output_dir}_multiseed")
    root_dir.mkdir(parents=True, exist_ok=True)

    seed_artifacts: list[dict[str, Any]] = []
    seed_runs: list[dict[str, Any]] = []
    for seed in seeds:
        run_output_dir = root_dir / f"seed_{seed}"
        run_config = replace(
            base_config,
            seed=int(seed),
            output_dir=str(run_output_dir),
        )
        artifact = train_from_config(run_config)
        metrics_path = run_output_dir / "metrics.json"
        seed_artifacts.append(artifact)
        seed_runs.append(
            {
                "seed": int(seed),
                "output_dir": str(run_output_dir),
                "metrics_path": str(metrics_path),
            }
        )
        _release_accelerator_memory()

    summary = aggregate_multiseed_metrics(
        artifacts=seed_artifacts,
        base_config=config_to_dict(base_config),
        seeds=[int(seed) for seed in seeds],
        output_root=str(root_dir),
        config_path=str(config_path),
        seed_runs=seed_runs,
    )
    summary_path = root_dir / "multiseed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(render_multiseed_summary(summary))
    return summary


def aggregate_multiseed_metrics(
    *,
    artifacts: Sequence[Mapping[str, Any]],
    base_config: Mapping[str, Any],
    seeds: Sequence[int],
    output_root: str,
    config_path: str,
    seed_runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(artifacts) != len(seeds):
        raise ValueError("artifacts and seeds must have the same length.")

    return {
        "config_path": config_path,
        "output_root": output_root,
        "method": base_config.get("method", "unknown"),
        "base_config": dict(base_config),
        "seeds": [int(seed) for seed in seeds],
        "seed_runs": [dict(seed_run) for seed_run in seed_runs],
        "num_seeds": len(seeds),
        "val": _summarize_split(artifacts, "val"),
        "test": _summarize_split(artifacts, "test"),
    }


def render_multiseed_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        f"method: {summary.get('method', 'unknown')}",
        f"seeds: {', '.join(str(seed) for seed in summary.get('seeds', []))}",
        f"output_root: {summary.get('output_root', 'n/a')}",
        "",
        "split  metric                mean     std",
        "-----  ------------------  -------  ------",
    ]
    for split in ("val", "test"):
        split_summary = summary.get(split, {})
        for metric_name in (
            "overall_accuracy",
            "worst_group_accuracy",
            "overall_auroc",
            "worst_group_auroc",
            "wilds_acc_avg",
            "wilds_acc_wg",
        ):
            metric_summary = split_summary.get(metric_name, {})
            lines.append(
                f"{split:<5}"
                f"  {metric_name:<18}"
                f"  {_format_number(metric_summary.get('mean')):>7}"
                f"  {_format_number(metric_summary.get('std')):>6}"
            )
    return "\n".join(lines)


def _summarize_split(
    artifacts: Sequence[Mapping[str, Any]],
    split_name: str,
) -> dict[str, dict[str, float | int | None]]:
    split_summary: dict[str, dict[str, float | int | None]] = {}
    for metric_name, path in SUMMARY_METRICS:
        values = [
            float(value)
            for artifact in artifacts
            if (value := _get_nested_value(artifact.get(split_name, {}), path)) is not None
        ]
        split_summary[metric_name] = _summarize_numeric_values(values)
    return split_summary


def _summarize_numeric_values(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
    }


def _get_nested_value(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
        if value is None:
            return None
    return value


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_multiseed_experiment(
        config_path=args.config,
        seeds=args.seeds,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
