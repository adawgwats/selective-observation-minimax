"""Tests for christensen_core.q_classes.

Verify:
    - dim_theta and theta_bounds match the expected dimension
    - q_values returns correct per-example probabilities for a range of θ
    - monotonicity flag is respected (for Parametric2ParamForBinary)
    - q values are always clipped into [q_min, q_max]
"""

from __future__ import annotations

import numpy as np


def test_constant_q_broadcasts_scalar() -> None:
    """ConstantQ with θ = [0.3] should give q_values = [0.3, 0.3, ..., 0.3]."""
    raise NotImplementedError


def test_parametric2param_distinguishes_y0_from_y1() -> None:
    """With θ = [q_0, q_1] and Y_tilde = [0, 1, 0, 1, 0], q_values should be
    [q_0, q_1, q_0, q_1, q_0]."""
    raise NotImplementedError


def test_theta_bounds_are_inside_q_min_q_max() -> None:
    """For any class, theta_bounds should lie within [q_min, q_max] per entry."""
    raise NotImplementedError


def test_q_values_clipped() -> None:
    """If a caller sneaks θ outside [q_min, q_max], q_values should clip."""
    raise NotImplementedError
