"""JTPA data loading and preprocessing.

The JTPA Public Use File (JPU) is not automatically downloadable — it requires
registration with Upjohn Institute or ICPSR. This module provides:

1. `load_jtpa()` — assumes user has downloaded files to data_cache/jtpa/; raises
   a descriptive error with download instructions if missing.
2. `summarize_non_response()` — diagnostic reporting of response rates by
   treatment group and covariate strata.
3. `apply_jtpa_schema()` — normalizes variable names across different format
   variants (ICPSR, Upjohn, Abadie-Angrist-Imbens replication package).

Expected directory layout after user-driven download:
    phase3_jtpa_benchmark/data_cache/jtpa/
        baseline.csv               # covariates at time of assignment
        outcomes_30m.csv           # 30-month follow-up earnings
        assignment.csv             # treatment assignment indicator

These files can be derived from ICPSR Study 8997 or the Upjohn JPU distribution.
See PROTOCOL.md §"Data access" for specifics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).parent / "data_cache" / "jtpa"


@dataclass(frozen=True)
class JTPADataset:
    """Unified JTPA dataset with naturally-occurring outcome MNAR."""

    covariates: pd.DataFrame     # (n, d_covariates) — baseline X
    treatment: np.ndarray         # (n,) — 0/1 assignment
    earnings_30m: np.ndarray      # (n,) — outcome Y; NaN where non-respondent
    response_mask: np.ndarray     # (n,) — True where respondent (earnings observed)
    site_ids: np.ndarray          # (n,) — training-site identifiers
    target_group: np.ndarray      # (n,) — Adult Women, Adult Men, Youth, etc.
    notes: str = ""


DOWNLOAD_INSTRUCTIONS = """
JTPA data is NOT included in this repo and must be downloaded manually.

Option A (recommended): Upjohn Institute
  URL: https://www.upjohn.org/data-tools/employment-research-data-center/national-jtpa-study
  Steps:
    1. Register for an account (free, academic/research)
    2. Download the JTPA Public Use File (JPU)
    3. Extract to phase3_jtpa_benchmark/data_cache/jtpa/

Option B: ICPSR
  URL: https://www.icpsr.umich.edu/web/ICPSR/studies/8997
  Study number: 8997
  Download as CSV or Stata format

Option C: Abadie-Angrist-Imbens (2002) replication package
  Their QE paper replication files are on Angrist's MIT page.
  Smaller subset focused on IV quantile estimation.

Expected files under data_cache/jtpa/:
  baseline.csv
  outcomes_30m.csv
  assignment.csv

Once downloaded, rerun to verify schema with: python -m phase3_jtpa_benchmark.datasets
"""


def load_jtpa() -> JTPADataset:
    """Load the JTPA dataset from local cache, raising a clear error with
    download instructions if data is not present.

    Not yet implemented: the schema normalization logic. This is a stub that
    checks for file existence and returns a placeholder. Real implementation
    happens once data is acquired.
    """
    baseline_path = DATA_DIR / "baseline.csv"
    outcomes_path = DATA_DIR / "outcomes_30m.csv"
    assignment_path = DATA_DIR / "assignment.csv"

    missing = [p.name for p in (baseline_path, outcomes_path, assignment_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing JTPA data files: {missing}\n"
            f"Expected location: {DATA_DIR.resolve()}\n\n"
            f"{DOWNLOAD_INSTRUCTIONS}"
        )

    raise NotImplementedError(
        "Data files found but parsing not yet implemented. This is Phase 3 scaffold. "
        "Flesh out parsing after user confirms data schema."
    )


def summarize_non_response(ds: JTPADataset) -> pd.DataFrame:
    """Diagnostic: response rate by treatment × covariate strata.

    Not yet implemented. Once data is loaded, this will produce a table like:
        | subgroup | treatment=0 response_rate | treatment=1 response_rate |
        | overall  | X                         | Y                         |
        | high prior earnings | ... | ... |
        | low prior earnings  | ... | ... |

    This supports the audit step: verifying that non-response is indeed
    outcome-correlated in the dataset before fitting Christensen's framework.
    """
    raise NotImplementedError("Implement after data is loaded")


if __name__ == "__main__":
    try:
        ds = load_jtpa()
        print(f"JTPA loaded. n={len(ds.treatment)}")
        print(f"Response rate overall: {ds.response_mask.mean():.3f}")
    except FileNotFoundError as e:
        print(str(e))
