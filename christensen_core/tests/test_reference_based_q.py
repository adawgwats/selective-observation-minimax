"""Tests for christensen_core.reference_based_q: q_hat + centered QClassConfig."""
from __future__ import annotations

import numpy as np
import pytest

from christensen_core.reference_based_q import (
    centered_config,
    centered_q_for,
    compute_q_hat,
)
from christensen_core.q_classes import (
    ConstantQ,
    Parametric2ParamForBinary,
    QClassConfig,
)


def test_compute_q_hat() -> None:
    """Basic correctness: response_mask=[T,T,F,T,F] -> q_hat=0.6."""
    mask = np.array([True, True, False, True, False])
    assert compute_q_hat(mask) == pytest.approx(0.6)

    # All-true / all-false boundaries
    assert compute_q_hat(np.array([True, True, True])) == pytest.approx(1.0)
    assert compute_q_hat(np.array([False, False, False])) == pytest.approx(0.0)

    # Empty raises
    with pytest.raises(ValueError):
        compute_q_hat(np.array([], dtype=bool))


def test_centered_config_basic() -> None:
    """q_hat=0.6, delta=0.3 -> QClassConfig(q_min=0.30, q_max=0.90)."""
    cfg = centered_config(q_hat=0.6, delta=0.3)
    assert isinstance(cfg, QClassConfig)
    assert cfg.q_min == pytest.approx(0.30)
    assert cfg.q_max == pytest.approx(0.90)


def test_centered_config_clamps_at_floor() -> None:
    """q_hat=0.1, delta=0.3 -> q_min=0.01 (floor), q_max=0.4."""
    cfg = centered_config(q_hat=0.1, delta=0.3)
    assert cfg.q_min == pytest.approx(0.01)
    assert cfg.q_max == pytest.approx(0.4)


def test_centered_config_clamps_at_ceiling() -> None:
    """q_hat=0.9, delta=0.3 -> q_min=0.60, q_max=1.00."""
    cfg = centered_config(q_hat=0.9, delta=0.3)
    assert cfg.q_min == pytest.approx(0.60)
    assert cfg.q_max == pytest.approx(1.00)


def test_centered_config_rejects_bad_inputs() -> None:
    """q_hat out of (0,1] and negative delta raise ValueError."""
    with pytest.raises(ValueError):
        centered_config(q_hat=0.0, delta=0.1)
    with pytest.raises(ValueError):
        centered_config(q_hat=-0.1, delta=0.1)
    with pytest.raises(ValueError):
        centered_config(q_hat=1.1, delta=0.1)
    with pytest.raises(ValueError):
        centered_config(q_hat=0.5, delta=-0.01)


def test_centered_q_for_mechanisms() -> None:
    """Each Pereira mechanism returns the right QClass with config centered on q_hat."""
    rng = np.random.default_rng(42)
    # Construct a response_mask with q_hat approximately 0.6
    response_mask = rng.random(1000) < 0.6
    q_hat = compute_q_hat(response_mask)
    # sanity: q_hat close to 0.6 by law of large numbers
    assert abs(q_hat - 0.6) < 0.05

    delta = 0.30
    expected_q_min = max(0.01, q_hat - delta)
    expected_q_max = min(1.0, q_hat + delta)

    # MBOV_Lower -> Parametric2ParamForBinary, increasing
    q = centered_q_for("MBOV_Lower", response_mask, delta=delta)
    assert isinstance(q, Parametric2ParamForBinary)
    assert q.monotone == "increasing"
    assert q.config.q_min == pytest.approx(expected_q_min)
    assert q.config.q_max == pytest.approx(expected_q_max)

    # MBOV_Stochastic -> same as MBOV_Lower
    q = centered_q_for("MBOV_Stochastic", response_mask, delta=delta)
    assert isinstance(q, Parametric2ParamForBinary)
    assert q.monotone == "increasing"
    assert q.config.q_min == pytest.approx(expected_q_min)
    assert q.config.q_max == pytest.approx(expected_q_max)

    # MBOV_Higher -> decreasing
    q = centered_q_for("MBOV_Higher", response_mask, delta=delta)
    assert isinstance(q, Parametric2ParamForBinary)
    assert q.monotone == "decreasing"
    assert q.config.q_min == pytest.approx(expected_q_min)
    assert q.config.q_max == pytest.approx(expected_q_max)

    # MBUV -> ConstantQ
    q = centered_q_for("MBUV", response_mask, delta=delta)
    assert isinstance(q, ConstantQ)
    assert q.config.q_min == pytest.approx(expected_q_min)
    assert q.config.q_max == pytest.approx(expected_q_max)

    # MBOV_Centered -> ConstantQ
    q = centered_q_for("MBOV_Centered", response_mask, delta=delta)
    assert isinstance(q, ConstantQ)
    assert q.config.q_min == pytest.approx(expected_q_min)
    assert q.config.q_max == pytest.approx(expected_q_max)


def test_centered_q_for_deferred_raises() -> None:
    """MBIR_Frequentist/Bayesian still raise NotImplementedError."""
    mask = np.array([True, True, False, True, False])
    for mech in ("MBIR_Frequentist", "MBIR_Bayesian"):
        with pytest.raises(NotImplementedError):
            centered_q_for(mech, mask)


def test_centered_q_for_unknown_raises_valueerror() -> None:
    """Unknown mechanism names raise ValueError."""
    mask = np.array([True, False, True])
    with pytest.raises(ValueError):
        centered_q_for("not_a_real_mechanism", mask)
