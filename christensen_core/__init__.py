"""christensen_core: faithful implementation of the estimator in Christensen's
`Minimax optimization.pdf` (see ../docs/christensen_minimax.pdf and ../docs/christensen_minimax_pdf.txt).

This package is intentionally separate from `minimax_core` because they are
different estimators. `minimax_core` implements a DRO-inspired variant (see
phase1_pereira_benchmark/AUDIT_v2.md for divergence catalog); `christensen_core`
implements what Christensen actually wrote down.

Top-level exports are intentionally minimal — the package surface is small and
matches PDF terminology. Deeper imports are available via submodule paths for
tests and advanced use.
"""

from __future__ import annotations

# Symbols are re-exported from submodules once those submodules are implemented.
# Keeping this __init__ empty until each submodule lands avoids premature API commitments.

__all__: list[str] = []
