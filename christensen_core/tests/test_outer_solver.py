"""Tests for christensen_core.outer_solver.

Acceptance criteria:
    - ConstantQ outer max returns q* near the empirical non-observation rate
      complement on pure MAR data (where the adversary has no bite).
    - Parametric2ParamForBinary outer max returns q_0 < q_1 on data generated
      with MBOV_Lower-style selection.
    - Objective value at θ* is at least as high as at any initial grid point.
"""

from __future__ import annotations

import numpy as np


def test_outer_respects_monotone_constraint() -> None:
    """For Parametric2ParamForBinary(monotone='increasing') on MBOV_Lower data,
    the returned θ* should satisfy θ_0 <= θ_1 up to tolerance."""
    raise NotImplementedError


def test_outer_returns_nontrivial_theta_under_MNAR() -> None:
    """On data with strong MBOV_Lower selection, θ* should be far from [q̂, q̂]
    (the trivial MAR solution), reflecting the adversary's freedom to bias low
    observation rates at Y=0."""
    raise NotImplementedError


def test_outer_objective_improves_over_initial() -> None:
    """Final inner value at θ* >= inner value at the midpoint of the box."""
    raise NotImplementedError
