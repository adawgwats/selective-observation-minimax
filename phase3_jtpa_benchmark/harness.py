"""Phase 3 harness: run all treatment effect estimators on JTPA.

Unlike Phases 1 and 2 which were per-cell benchmarks (dataset × mechanism × seed
× method), Phase 3 has a single dataset and produces a single estimate per
method. The harness is therefore much simpler:

    for estimator in all_estimators:
        result = estimator.fit_estimate(Y, D, X, response_mask)
        collect(result)

The output is a comparison table of (method, τ̂, SE, 95% CI, notes) that
maps to Table 1-style presentation in the published paper.

Plus: bootstrap stability analysis, leave-one-site-out sensitivity, and the
Christensen sensitivity range across Q families.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ComparisonRow:
    method: str
    att_estimate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    notes: str = ""


def run_comparison(estimators: Iterable, dataset) -> pd.DataFrame:
    """Run all estimators on the dataset and return a comparison table.

    Not yet implemented. Scaffolded signature.
    """
    raise NotImplementedError("Implement after baselines + christensen_adapter are ready")


def leave_one_site_out(estimator, dataset) -> pd.DataFrame:
    """Sensitivity: leave-one-training-site-out. Reveals whether the
    treatment effect is driven by a single anomalous site (a known JTPA
    concern from Heckman et al.).
    """
    raise NotImplementedError("Implement")


def publish_table(comparison_df: pd.DataFrame, out_path: Path | None = None) -> str:
    """Format as a publication-ready markdown table for the final report."""
    raise NotImplementedError("Implement")
