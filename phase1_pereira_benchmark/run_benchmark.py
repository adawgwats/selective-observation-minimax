"""Entry point: run the full Phase 1 benchmark.

Saves incremental CSV every 100 cells to phase1_pereira_benchmark/results/raw_results.csv.

Usage:
    python -m phase1_pereira_benchmark.run_benchmark              # full run, 30 seeds
    python -m phase1_pereira_benchmark.run_benchmark --seeds 5    # quick run, 5 seeds
    python -m phase1_pereira_benchmark.run_benchmark --datasets wisconsin,pima   # subset
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

from .datasets import REGISTRY as DATASET_REGISTRY
from .harness import run_benchmark, aggregate, MISSING_RATES
from .mnar_injection import ALL_MECHANISMS


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=30, help="Number of seeds per cell.")
    p.add_argument("--datasets", type=str, default="all",
                   help="Comma-separated dataset names or 'all'.")
    p.add_argument("--mechanisms", type=str, default="all",
                   help="Comma-separated mechanism names or 'all'.")
    p.add_argument("--rates", type=str, default="all",
                   help="Comma-separated missing rates or 'all'.")
    p.add_argument("--output", type=Path,
                   default=Path(__file__).parent / "results" / "raw_results.csv")
    p.add_argument("--agg-output", type=Path,
                   default=Path(__file__).parent / "results" / "aggregated.csv")
    return p.parse_args(argv)


def main(argv=None):
    warnings.filterwarnings("ignore")
    args = parse_args(argv)

    datasets = tuple(DATASET_REGISTRY.keys()) if args.datasets == "all" else tuple(args.datasets.split(","))
    mechanisms = ALL_MECHANISMS if args.mechanisms == "all" else tuple(args.mechanisms.split(","))
    rates = MISSING_RATES if args.rates == "all" else tuple(float(r) for r in args.rates.split(","))
    seeds = tuple(range(args.seeds))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_cells = len(datasets) * len(mechanisms) * len(rates) * len(seeds)
    print(f"[run_benchmark] datasets={len(datasets)}  mechanisms={len(mechanisms)}  "
          f"rates={len(rates)}  seeds={len(seeds)}")
    print(f"[run_benchmark] total cells: {total_cells}")
    print(f"[run_benchmark] output: {args.output}")
    print(f"[run_benchmark] aggregated: {args.agg_output}")

    t0 = time.perf_counter()
    df = run_benchmark(
        datasets=datasets,
        mechanisms=mechanisms,
        rates=rates,
        seeds=seeds,
        out_csv=args.output,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"[run_benchmark] done in {elapsed/60:.1f} minutes; {len(df)} rows written.")

    agg = aggregate(df)
    agg.to_csv(args.agg_output, index=False)
    print(f"[run_benchmark] aggregation saved to {args.agg_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
