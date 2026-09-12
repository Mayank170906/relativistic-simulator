"""Tests for special-relativistic kinematics and energy relations."""
from __future__ import annotations

from decimal import Decimal, getcontext

import numpy as np
import pytest

from relativistic_simulator import C, C_SQUARED
from relativistic_simulator.exceptions import (
    InvalidMassError,
    InvalidSimulationParameterError,
    VelocityLimitError,
)
from relativistic_simulator.relativity import (
    beta_from_velocity,
    energy_from_momentum,
    energy_from_velocity,
    gamma_from_beta,
    gamma_from_momentum,
    gamma_from_velocity,
    kinetic_energy_from_momentum,
    kinetic_energy_from_velocity,
    momentum_from_velocity,
    rest_energy,
    velocity_from_beta,
    velocity_from_momentum,
)

getcontext().prec = 50


def _decimal_gamma(beta: float) -> Decimal:
    """Independent (decimal-arithmetic) Lorentz factor for IBC verification.

    ``Decimal.from_float`` captures the *exact* binary value of the double,
    so the reference is the true gamma of the float the code receives.
    """
    b = Decimal.from_float(float(beta))
    one_minus = 1 - b * b
    assert one_minus > 0
    return (1 / one_minus).sqrt()


@pytest.mark.parametrize("beta", [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999])
def test_gamma_from_beta_matches_decimal(beta: float) -> None:
    got = gamma_from_beta(beta)
    want = float(_decimal_gamma(beta))
    assert got == pytest.approx(want, rel=1e-14, abs=1e-14)


@pytest.mark.parametrize("beta", [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999])
def test_gamma_from_velocity_matches_decimal(beta: float) -> None:
    """gamma(v) vs an arbitrary-precision oracle.

    Near c the mapping v -> beta -> gamma amplifies rounding by ~gamma^2,
    so the oracle uses the *exactly reconstructed* beta = fl(v)/fl(c)
    (50-digit division) as its input; the float result must then match to
    a few ulps.
    """
    velocity = beta * C
    # The exact float the function receives is fl(fl(beta*c)/c).
    b_rec = Decimal.from_float(float(velocity / C))
    one_minus = 1 - b_rec * b_rec
    want = float((1 / one_minus).sqrt())
    got = gamma_from_velocity(velocity)
    assert got == pytest.approx(want, rel=1e-13, abs=1e-13)


def test_gamma_at_rest_is_one() -> None:
    assert gamma_from_beta(0.0) == 1.0
    assert gamma_from_velocity(0.0) == 1.0


def test_beta_velocity_roundtrip(beta_grid: np.ndarray) -> None:
    for b in beta_grid:
        v = velocity_from_beta(b)
        assert beta_from_velocity(v) == pytest.approx(b, rel=1e-15, abs=1e-15)
        assert beta_from_velocity(-v) == pytest.approx(-b, rel=1e-15, abs=1e-15)


def test_velocity_from_beta_is_c_times_beta() -> None:
    b = 0.42
    v = velocity_from_beta(b)
    assert v == pytest.approx(b * C, rel=1e-16, abs=1e-16)


def test_momentum_velocity_inverse(beta_grid: np.ndarray) -> None:
    m = 25.0
    for b in beta_grid:
        v = velocity_from_beta(b)
        p = momentum_from_velocity(m, v)
        v_back = velocity_from_momentum(m, p)
        assert v_back == pytest.approx(v, rel=1e-12, abs=1e-12)


def test_momentum_is_gamma_m_v(beta_grid: np.ndarray) -> None:
    m = 42.0
    for b in beta_grid:
        v = velocity_from_beta(b)
        p = momentum_from_velocity(m, v)
        g = gamma_from_beta(b)
        assert p == pytest.approx(g * m * v, rel=1e-12, abs=1e-12)


def test_energy_from_velocity_is_gamma_m_c2(beta_grid: np.ndarray) -> None:
    m = 3.0
    for b in beta_grid:
        v = velocity_from_beta(b)
        assert energy_from_velocity(m, v) == pytest.approx(
            gamma_from_beta(b) * m * C_SQUARED, rel=1e-12, abs=1e-12
        )
def test_energy_from_momentum_equals_energy_from_velocity(beta_grid: np.ndarray) -> None:
    m = 3.0
    for b in beta_grid:
        v = velocity_from_beta(b)
        p = momentum_from_velocity(m, v)
        assert energy_from_momentum(m, p) == pytest.approx(energy_from_velocity(m, v), rel=1e-12, abs=1e-12)


def test_invariant_energy_momentum(beta_grid: np.ndarray) -> None:
    """E^2 - p^2 c^2 == m^2 c^4."""
    m = 7.0
    for b in beta_grid:
        v = velocity_from_beta(b)
        p = momentum_from_velocity(m, v)
        E = energy_from_velocity(m, v)
        lhs = E * E - (p * C) ** 2
        rhs = (m * C_SQUARED) ** 2
        assert lhs / rhs == pytest.approx(1.0, rel=1e-10, abs=1e-10)


def test_kinetic_energy_velocity_momentum_agree(beta_grid: np.ndarray) -> None:
    m = 11.0
    for b in beta_grid:
        v = velocity_from_beta(b)
        p = momentum_from_velocity(m, v)
        assert kinetic_energy_from_velocity(m, v) == pytest.approx(
            kinetic_energy_from_momentum(m, p), rel=1e-10, abs=1e-10
        )


def test_kinetic_energy_plus_rest_is_total() -> None:
    m = 2.5
    v = 0.9 * C
    E = energy_from_velocity(m, v)
    K = kinetic_energy_from_velocity(m, v)
    assert K == pytest.approx(E - rest_energy(m), rel=1e-14, abs=1e-14)


def test_kinetic_energy_low_velocity_classical() -> None:
    m = 1.0
    v = 1.0  # 1 m/s: totally non-relativistic
    K = kinetic_energy_from_velocity(m, v)
    assert K == pytest.approx(0.5 * m * v * v, rel=1e-14, abs=1e-14)


def test_rest_energy() -> None:
    m = 4.0
    assert rest_energy(m) == pytest.approx(m * C_SQUARED, rel=1e-16, abs=1e-16)


def test_near_c_value_is_accepted() -> None:
    """The largest double below 1 must give a huge but finite gamma."""
    beta = 1.0 - 2.0**-53  # 0.9999999999999999: nearest double below 1
    gamma = gamma_from_beta(beta)
    assert np.isfinite(gamma)
    # Exact: (1-beta) = 2^-53 -> gamma = 1/sqrt(2^-53 * (2 - 2^-53)) ~= 2^26
    assert gamma == pytest.approx(2.0**26, rel=1e-9)
    # round-trip through momentum must stay sub-luminal
    m = 1.0
    p = momentum_from_velocity(m, velocity_from_beta(beta))
    v_back = velocity_from_momentum(m, p)
    assert 0.0 < v_back <= C
    assert beta_from_velocity(v_back) == pytest.approx(beta, rel=1e-12, abs=1e-12)


def test_vectorized_matches_scalar(beta_grid: np.ndarray) -> None:
    m = np.array([1.0, 2.0, 3.0])
    v = beta_grid[:3] * C
    p_loop = np.array([momentum_from_velocity(mi, vi) for mi, vi in zip(m, v)])
    p_vec = momentum_from_velocity(m, v)
    np.testing.assert_allclose(p_vec, p_loop, rtol=1e-14, atol=1e-14)


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [1.0, -1.0, 1.5, np.nan, np.inf, -np.inf])
def test_beta_at_or_above_c_rejected(bad: float) -> None:
    for fn in (gamma_from_beta, velocity_from_beta):
        with pytest.raises(VelocityLimitError):
            fn(bad)


@pytest.mark.parametrize("bad", [C, -C, C + 1, -C - 1, np.nan, np.inf, -np.inf])
def test_velocity_at_or_above_c_rejected(bad: float) -> None:
    for fn in (beta_from_velocity, gamma_from_velocity, momentum_from_velocity):
        with pytest.raises(VelocityLimitError):
            if fn is momentum_from_velocity:
                fn(1.0, bad)
            else:
                fn(bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_bad_mass_rejected(bad: float) -> None:
    for fn in (momentum_from_velocity, velocity_from_momentum, energy_from_velocity,
               energy_from_momentum, kinetic_energy_from_velocity,
               kinetic_energy_from_momentum, rest_energy, gamma_from_momentum):
        with pytest.raises(InvalidMassError):
            if "momentum" in fn.__name__:
                fn(bad, 1.0)
            elif "rest_energy" in fn.__name__:
                fn(bad)
            else:
                fn(bad, 0.0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_bad_momentum_rejected(bad: float) -> None:
    for fn in (velocity_from_momentum, energy_from_momentum,
               kinetic_energy_from_momentum, gamma_from_momentum):
        with pytest.raises(InvalidSimulationParameterError):
            fn(1.0, bad)


def test_complex_input_rejected() -> None:
    with pytest.raises(InvalidSimulationParameterError):
        beta_from_velocity(1 + 2j)


def test_string_input_rejected() -> None:
    with pytest.raises(InvalidSimulationParameterError):
        beta_from_velocity("fast")


def test_broadcast_mismatch_rejected() -> None:
    with pytest.raises(InvalidSimulationParameterError):
        momentum_from_velocity(np.array([1.0, 2.0]), np.array([0.1, 0.2, 0.3]) * C)