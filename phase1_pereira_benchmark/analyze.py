"""Analysis and reporting for Phase 1 results.

Reads `results/raw_results.csv` (produced by run_benchmark.py), aggregates over seeds,
and produces:
  - A wide CSV with one row per (dataset, mechanism, rate) and one column per method
  - A win-loss table vs MICE (Pereira's primary baseline)
  - A markdown REPORT.md with summary statistics and honest commentary
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PRIMARY_BASELINE = "mice"
METHOD_UNDER_TEST = "minimax_score"


def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, mechanism, rate, method): mean, std, 95% CI over seeds."""
    grouped = df.groupby(["dataset", "mechanism", "missing_rate_pct", "method"])["test_mse"]
    n = grouped.count()
    mean = grouped.mean()
    std = grouped.std(ddof=1)
    se = std / np.sqrt(n.clip(lower=1))
    return pd.DataFrame({
        "mean_mse": mean, "std_mse": std, "n_seeds": n,
        "ci_lower": mean - 1.96 * se, "ci_upper": mean + 1.96 * se,
    }).reset_index()


def pivot_methods(agg: pd.DataFrame, metric: str = "mean_mse") -> pd.DataFrame:
    return agg.pivot_table(
        index=["dataset", "mechanism", "missing_rate_pct"],
        columns="method",
        values=metric,
    ).reset_index()


def win_loss_vs_baseline(agg: pd.DataFrame, baseline: str = PRIMARY_BASELINE, method: str = METHOD_UNDER_TEST) -> pd.DataFrame:
    """For each cell, compare `method` to `baseline` and classify: WIN / TIE / LOSS.

    WIN: method CI strictly below baseline CI (method.ci_upper < baseline.ci_lower)
    LOSS: method CI strictly above baseline CI (method.ci_lower > baseline.ci_upper)
    TIE: CIs overlap (no statistically significant difference at 95%)
    """
    b = agg[agg.method == baseline].set_index(["dataset", "mechanism", "missing_rate_pct"])
    m = agg[agg.method == method].set_index(["dataset", "mechanism", "missing_rate_pct"])
    aligned = b.join(m, lsuffix="_baseline", rsuffix="_method", how="inner")

    def classify(row):
        if np.isnan(row.mean_mse_method) or np.isnan(row.mean_mse_baseline):
            return "N/A"
        if row.ci_upper_method < row.ci_lower_baseline:
            return "WIN"
        if row.ci_lower_method > row.ci_upper_baseline:
            return "LOSS"
        return "TIE"

    aligned["outcome"] = aligned.apply(classify, axis=1)
    aligned["mse_diff"] = aligned.mean_mse_method - aligned.mean_mse_baseline
    aligned["mse_diff_pct"] = 100.0 * aligned.mse_diff / aligned.mean_mse_baseline
    return aligned.reset_index()


def summarize_outcomes(win_loss: pd.DataFrame) -> dict:
    counts = win_loss.outcome.value_counts().to_dict()
    total = sum(counts.values())
    return {
        "win": counts.get("WIN", 0),
        "tie": counts.get("TIE", 0),
        "loss": counts.get("LOSS", 0),
        "na": counts.get("N/A", 0),
        "total": total,
        "win_rate": counts.get("WIN", 0) / total if total else 0.0,
        "loss_rate": counts.get("LOSS", 0) / total if total else 0.0,
        "tie_rate": counts.get("TIE", 0) / total if total else 0.0,
    }


def slice_by_mechanism(win_loss: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mech, sub in win_loss.groupby("mechanism"):
        c = sub.outcome.value_counts().to_dict()
        total = len(sub)
        rows.append({
            "mechanism": mech,
            "wins": c.get("WIN", 0),
            "ties": c.get("TIE", 0),
            "losses": c.get("LOSS", 0),
            "total": total,
            "win_rate": c.get("WIN", 0) / total if total else 0.0,
            "mean_mse_diff_pct": sub.mse_diff_pct.mean(),
        })
    return pd.DataFrame(rows).sort_values("mechanism").reset_index(drop=True)


def slice_by_rate(win_loss: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rate, sub in win_loss.groupby("missing_rate_pct"):
        c = sub.outcome.value_counts().to_dict()
        total = len(sub)
        rows.append({
            "missing_rate_pct": rate,
            "wins": c.get("WIN", 0),
            "ties": c.get("TIE", 0),
            "losses": c.get("LOSS", 0),
            "total": total,
            "win_rate": c.get("WIN", 0) / total if total else 0.0,
            "mean_mse_diff_pct": sub.mse_diff_pct.mean(),
        })
    return pd.DataFrame(rows).sort_values("missing_rate_pct").reset_index(drop=True)


def slice_by_dataset(win_loss: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, sub in win_loss.groupby("dataset"):
        c = sub.outcome.value_counts().to_dict()
        total = len(sub)
        rows.append({
            "dataset": ds,
            "wins": c.get("WIN", 0),
            "ties": c.get("TIE", 0),
            "losses": c.get("LOSS", 0),
            "total": total,
            "win_rate": c.get("WIN", 0) / total if total else 0.0,
            "mean_mse_diff_pct": sub.mse_diff_pct.mean(),
        })
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def format_markdown_table(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    """Simple markdown table renderer."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if np.isnan(v):
                    vals.append("—")
                else:
                    vals.append(float_fmt.format(v))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def generate_report(raw_path: Path, out_path: Path) -> None:
    df = load_raw(raw_path)
    if len(df) == 0:
        out_path.write_text("# Phase 1 Report\n\nNo results yet.\n")
        return

    agg = aggregate(df)
    wl_mice = win_loss_vs_baseline(agg, baseline="mice", method="minimax_score")
    wl_cc = win_loss_vs_baseline(agg, baseline="complete_case", method="minimax_score")
    wl_erm = win_loss_vs_baseline(agg, baseline="erm_sgd", method="minimax_score")

    summary_mice = summarize_outcomes(wl_mice)
    summary_cc = summarize_outcomes(wl_cc)
    summary_erm = summarize_outcomes(wl_erm)

    mech_slice = slice_by_mechanism(wl_mice)
    rate_slice = slice_by_rate(wl_mice)
    ds_slice = slice_by_dataset(wl_mice)

    lines = []
    lines.append("# Phase 1 Report — Christensen Minimax vs MICE under MNAR Labels")
    lines.append("")
    lines.append("**Path A benchmark**: Pereira et al. 2024 MNAR mechanisms applied to label column for regression tasks on 10 UCI medical datasets. See PROTOCOL.md for full spec and declared deviations from Pereira's original imputation-quality benchmark.")
    lines.append("")
    lines.append(f"**Seeds completed**: {df.groupby(['dataset','mechanism','missing_rate_pct']).seed.nunique().median():.0f} per cell (median across cells)")
    lines.append(f"**Total rows**: {len(df):,}")
    lines.append(f"**Methods**: {sorted(df.method.unique())}")
    lines.append(f"**Mechanisms**: {sorted(df.mechanism.unique())}")
    lines.append(f"**Datasets**: {sorted(df.dataset.unique())}")
    lines.append("")

    lines.append("## Headline: Christensen minimax vs MICE")
    lines.append("")
    lines.append(f"Across {summary_mice['total']} (dataset, mechanism, rate) cells:")
    lines.append("")
    lines.append(f"- **Wins** (95% CI strictly below MICE): **{summary_mice['win']}** ({summary_mice['win_rate']*100:.1f}%)")
    lines.append(f"- **Ties** (95% CIs overlap MICE): {summary_mice['tie']} ({summary_mice['tie_rate']*100:.1f}%)")
    lines.append(f"- **Losses** (95% CI strictly above MICE): {summary_mice['loss']} ({summary_mice['loss_rate']*100:.1f}%)")
    if summary_mice.get("na", 0):
        lines.append(f"- Failed cells (NaN): {summary_mice['na']}")
    lines.append("")

    lines.append("## Headline: minimax vs ERM (same SGD engine, no adversary)")
    lines.append("")
    lines.append(f"This is the apples-to-apples algorithm comparison — same SGD, same learning schedule, only difference is the adversary.")
    lines.append("")
    lines.append(f"- Wins: **{summary_erm['win']}** ({summary_erm['win_rate']*100:.1f}%)")
    lines.append(f"- Ties: {summary_erm['tie']} ({summary_erm['tie_rate']*100:.1f}%)")
    lines.append(f"- Losses: {summary_erm['loss']} ({summary_erm['loss_rate']*100:.1f}%)")
    lines.append("")

    lines.append("## Win/loss vs MICE by MNAR mechanism")
    lines.append("")
    lines.append(format_markdown_table(mech_slice))
    lines.append("")

    lines.append("## Win/loss vs MICE by missing rate")
    lines.append("")
    lines.append(format_markdown_table(rate_slice))
    lines.append("")

    lines.append("## Win/loss vs MICE by dataset")
    lines.append("")
    lines.append(format_markdown_table(ds_slice))
    lines.append("")

    # Highlight most favorable and least favorable cells
    lines.append("## Most favorable cells (minimax beats MICE by largest %)")
    lines.append("")
    top_wins = wl_mice[wl_mice.outcome == "WIN"].sort_values("mse_diff_pct").head(10)
    cols = ["dataset", "mechanism", "missing_rate_pct", "mean_mse_method", "mean_mse_baseline", "mse_diff_pct"]
    if len(top_wins):
        lines.append(format_markdown_table(top_wins[cols].rename(columns={
            "mean_mse_method": "minimax_mse",
            "mean_mse_baseline": "mice_mse",
            "mse_diff_pct": "diff_%",
        })))
    else:
        lines.append("(no wins yet)")
    lines.append("")

    lines.append("## Least favorable cells (MICE beats minimax by largest %)")
    lines.append("")
    top_losses = wl_mice[wl_mice.outcome == "LOSS"].sort_values("mse_diff_pct", ascending=False).head(10)
    if len(top_losses):
        lines.append(format_markdown_table(top_losses[cols].rename(columns={
            "mean_mse_method": "minimax_mse",
            "mean_mse_baseline": "mice_mse",
            "mse_diff_pct": "diff_%",
        })))
    else:
        lines.append("(no losses yet)")
    lines.append("")

    lines.append("## Interpretation notes")
    lines.append("")
    lines.append("1. **This is NOT a replication of Pereira et al.'s benchmark**. They measure imputation MAE on feature values; we measure test-set prediction MSE with MNAR injected on the label. See PROTOCOL.md §Deviation.")
    lines.append("2. The minimax estimator run here is SGD-with-online-score-based-adversary, not the closed-form β̂ = M·(1/n Σ XᵢỸᵢ) + m from Christensen's 2020 write-up. A follow-up comparing the two algorithms is warranted if this result is encouraging.")
    lines.append("3. Binary-labeled datasets with extreme MNAR (high rate + strong selection) can produce training splits with all-one or all-zero labels, causing SGD-based methods to diverge from the trivial mean-predictor. This is reflected in some LOSS cells.")
    lines.append("4. MBUV is near-MCAR in label-only setting (see mnar_injection.py). Differences vs MICE there are expected to be small; the interesting signal is under MBOV_Lower/Higher.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path,
                   default=Path(__file__).parent / "results" / "raw_results.csv")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).parent / "REPORT.md")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.raw.exists():
        print(f"No raw results at {args.raw}. Run run_benchmark.py first.")
        return 1
    generate_report(args.raw, args.out)
    print(f"Report written to {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
