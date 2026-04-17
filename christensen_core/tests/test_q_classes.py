"""Tests for christensen_core.q_classes.

Verify:
    - dim_theta and theta_bounds match the expected dimension
    - q_values returns correct per-example probabilities for a range of θ
    - monotonicity flag is respected (for Parametric2ParamForBinary)
    - q values are always clipped into [q_min, q_max]
"""

from __future__ import annotations

import numpy as np

from christensen_core.q_classes import (
    ConstantQ,
    Parametric2ParamForBinary,
    QClassConfig,
)


def test_constant_q_broadcasts_scalar() -> None:
    """ConstantQ with θ = [0.3] should give q_values = [0.3, 0.3, ..., 0.3]."""
    q_class = ConstantQ()
    theta = np.array([0.3])
    Y_tilde = np.array([0, 1, 0, 1, 0])
    X = np.zeros((len(Y_tilde), 2))

    q = q_class.q_values(theta, X, Y_tilde)

    assert q.shape == (5,)
    np.testing.assert_allclose(q, np.full(5, 0.3))
    assert q_class.dim_theta() == 1


def test_parametric2param_distinguishes_y0_from_y1() -> None:
    """With θ = [q_0, q_1] and Y_tilde = [0, 1, 0, 1, 0], q_values should be
    [q_0, q_1, q_0, q_1, q_0]."""
    q_class = Parametric2ParamForBinary()
    theta = np.array([0.4, 0.7])
    Y_tilde = np.array([0, 1, 0, 1, 0])
    X = np.zeros((len(Y_tilde), 2))

    q = q_class.q_values(theta, X, Y_tilde)

    expected = np.array([0.4, 0.7, 0.4, 0.7, 0.4])
    np.testing.assert_allclose(q, expected)
    assert q_class.dim_theta() == 2


def test_theta_bounds_are_inside_q_min_q_max() -> None:
    """For any class, theta_bounds should lie within [q_min, q_max] per entry."""
    config = QClassConfig(q_min=0.1, q_max=0.9)
    for q_class in (
        ConstantQ(config=config),
        Parametric2ParamForBinary(config=config),
        Parametric2ParamForBinary(monotone="increasing", config=config),
        Parametric2ParamForBinary(monotone="decreasing", config=config),
    ):
        low, high = q_class.theta_bounds()
        assert low.shape == (q_class.dim_theta(),)
        assert high.shape == (q_class.dim_theta(),)
        assert np.all(low >= config.q_min - 1e-12)
        assert np.all(high <= config.q_max + 1e-12)
        assert np.all(low <= high)


def test_q_values_clipped() -> None:
    """If a caller sneaks θ outside [q_min, q_max], q_values should clip."""
    config = QClassConfig(q_min=0.05, q_max=1.0)

    # ConstantQ above q_max → clipped down to q_max
    q_class = ConstantQ(config=config)
    Y_tilde = np.array([0, 1, 0])
    X = np.zeros((len(Y_tilde), 2))
    q_high = q_class.q_values(np.array([2.0]), X, Y_tilde)
    np.testing.assert_allclose(q_high, np.full(3, config.q_max))

    # ConstantQ below q_min → clipped up to q_min
    q_low = q_class.q_values(np.array([-0.5]), X, Y_tilde)
    np.testing.assert_allclose(q_low, np.full(3, config.q_min))

    # Parametric2ParamForBinary mixed: theta[0] too low, theta[1] too high
    p_class = Parametric2ParamForBinary(config=config)
    Y_tilde = np.array([0, 1, 0, 1])
    X = np.zeros((len(Y_tilde), 2))
    q = p_class.q_values(np.array([-0.5, 2.0]), X, Y_tilde)
    expected = np.array([config.q_min, config.q_max, config.q_min, config.q_max])
    np.testing.assert_allclose(q, expected)
