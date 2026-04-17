"""Tests for christensen_core.pereira_q.q_class_for dispatch.

Verify:
    - High/medium fidelity mechanisms return the correct QClass subclass and
      the correct monotone flag (when applicable).
    - MBIR_* mechanisms raise NotImplementedError (deferred to v2).
    - Unknown mechanism names raise ValueError.
    - A caller-provided QClassConfig flows through to the returned QClass.
"""

from __future__ import annotations

import pytest

from christensen_core.pereira_q import FIDELITY, q_class_for
from christensen_core.q_classes import (
    ConstantQ,
    Parametric2ParamForBinary,
    QClassConfig,
)


def test_known_mechanisms_return_correct_class() -> None:
    """MBOV_Lower / MBOV_Higher / MBOV_Stochastic / MBUV / MBOV_Centered all
    dispatch to the expected QClass subclass with the expected monotone flag."""
    expected = {
        "MBOV_Lower": (Parametric2ParamForBinary, "increasing"),
        "MBOV_Higher": (Parametric2ParamForBinary, "decreasing"),
        "MBOV_Stochastic": (Parametric2ParamForBinary, "increasing"),
        "MBUV": (ConstantQ, None),
        "MBOV_Centered": (ConstantQ, None),
    }
    for mechanism, (cls, monotone) in expected.items():
        q = q_class_for(mechanism)
        assert isinstance(q, cls), f"{mechanism}: expected {cls}, got {type(q)}"
        if monotone is not None:
            assert q.monotone == monotone, (
                f"{mechanism}: expected monotone={monotone!r}, got {q.monotone!r}"
            )
        # Sanity-check that the mechanism is also listed in the FIDELITY table.
        assert mechanism in FIDELITY


def test_deferred_mechanisms_raise() -> None:
    """MBIR_Frequentist and MBIR_Bayesian are deferred to v2."""
    for mechanism in ("MBIR_Frequentist", "MBIR_Bayesian"):
        with pytest.raises(NotImplementedError):
            q_class_for(mechanism)


def test_unknown_mechanism_raises_valueerror() -> None:
    """Bogus mechanism names raise ValueError (not NotImplementedError)."""
    with pytest.raises(ValueError):
        q_class_for("not_a_real_mechanism")


def test_config_passed_through() -> None:
    """A custom QClassConfig is forwarded to the returned QClass."""
    cfg = QClassConfig(q_min=0.1, q_max=0.9)

    # Covers both dispatch branches: Parametric2ParamForBinary and ConstantQ.
    q_param = q_class_for("MBOV_Lower", config=cfg)
    assert q_param.config is cfg

    q_const = q_class_for("MBUV", config=cfg)
    assert q_const.config is cfg
