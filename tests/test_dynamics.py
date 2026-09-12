"""Tests for the constant-force relativistic dynamics solution."""
from __future__ import annotations

import numpy as np
import pytest

from relativistic_simulator import C, C_SQUARED
from relativistic_simulator.dynamics import analyze_constant_force, integrate_rk4
from relativistic_simulator.exceptions import (
    InvalidSimulationParameterError,
    InvalidTimeStepError,
    VelocityLimitError,
)
from relativistic_simulator.relativity import energy_from_velocity

MASS = 1000.0
FORCE = 1e6
DT = 0.01


def _rel_bounds(a: np.ndarray, b: np.ndarray) -> float:
    """Max relative error between two arrays with a tiny absolute floor."""
    denom = np.maximum(np.abs(b), 1e-12)
    return float(np.max(np.abs(a - b) / denom))


def test_zero_force_constant_velocity() -> None:
    t = np.linspace(0, 10, 101)
    state = analyze_constant_force(2.0, 0.0, 0.9 * C, 5.0, t)
    assert _rel_bounds(state["velocity"], 0.9 * C) < 1e-12
    # x = x0 + v0 t
    assert _rel_bounds(state["position"], 5.0 + 0.9 * C * t) < 1e-12
    # tau = t / gamma0, with gamma0 = 1/sqrt(1 - beta0^2)
    gamma0 = 1.0 / np.sqrt(1.0 - 0.9**2)
    assert _rel_bounds(state["proper_time"], t / gamma0) < 1e-12


def test_zero_force_rest_stays_rest() -> None:
    t = np.linspace(0, 10, 11)
    state = analyze_constant_force(1.0, 0.0, 0.0, 0.0, t)
    assert np.all(state["velocity"] == 0.0)
    assert np.all(state["position"] == 0.0)
    assert np.all(state["proper_time"] == t)  # gamma = 1 -> tau = t


def test_momentum_is_linear_in_time() -> None:
    t = np.linspace(0, 100, 1001)
    p0 = 0.0
    state = analyze_constant_force(MASS, FORCE, 0.0, 0.0, t)
    assert _rel_bounds(state["momentum"], p0 + FORCE * t) < 1e-12


def test_analytic_matches_rk4_across_regimes() -> None:
    """The closed-form solution must agree with an independent RK4 integrator."""
    cases = [
        dict(mass=1000.0, force=1e6, v0=0.0),
        dict(mass=1000.0, force=-1e6, v0=0.9 * C),
        dict(mass=1e-3, force=1e-6, v0=0.99 * C),
        dict(mass=1e6, force=1e12, v0=0.0),
        dict(mass=2.0, force=0.0, v0=0.9 * C),
    ]
    for case in cases:
        t = np.linspace(0, 50, 2001)
        ana = analyze_constant_force(case["mass"], case["force"], case["v0"], 0.0, t)
        num = integrate_rk4(case["mass"], case["force"], case["v0"], 0.0, time=t)
        for key in ("position", "velocity", "momentum", "gamma", "proper_time", "total_energy"):
            err = _rel_bounds(np.asarray(ana[key]), np.asarray(num[key]))
            assert err < 1e-8, f"{key}: rel err {err:.2e} for {case}"


def test_position_work_energy_theorem() -> None:
    """F * dx == dK (work-energy theorem, non-relativistic check of the solution)."""
    t = np.linspace(0, 50, 501)
    state = analyze_constant_force(MASS, FORCE, 0.0, 0.0, t)
    dx = np.diff(state["position"])
    dK = np.diff(state["kinetic_energy"])
    assert _rel_bounds(FORCE * dx, dK) < 1e-8


def test_position_stationary_at_low_velocity_matches_classical() -> None:
    """For beta << 1 the relativistic position should approach x = 0.5 a t^2."""
    m, F, T = 1e6, 1.0, 10.0
    t = np.linspace(0, T, 101)
    state = analyze_constant_force(m, F, 0.0, 0.0, t)
    x_classical = 0.5 * (F / m) * t**2
    assert _rel_bounds(state["position"], x_classical) < 1e-3


def test_turning_point_negative_force() -> None:
    """A particle decelerating to rest and turning around."""
    m = 1.0
    F = -1e-3
    v0 = 0.5 * C
    t = np.linspace(0, 1e12, 20001)
    state = analyze_constant_force(m, F, v0, 0.0, t)
    v = state["velocity"]
    assert np.sign(v[0]) > 0
    assert np.sign(v[-1]) < 0  # turned around
    crossing = np.where(np.diff(np.sign(v)) != 0)[0]
    assert crossing.size >= 1
    # near the turning point gamma is minimal (velocity ~ 0)
    i = crossing[0]
    assert np.min(state["gamma"]) == pytest.approx(1.0, rel=1e-6)


def test_proper_time_leq_coordinate_time(beta_grid: np.ndarray) -> None:
    t = np.linspace(0, 50, 501)
    for rate in (beta_grid[1], beta_grid[-1]):
        state = analyze_constant_force(MASS, FORCE, rate * C, 0.0, t)
        assert np.all(state["proper_time"] <= t + 1e-9)
        assert np.all(np.diff(state["proper_time"]) >= 0.0)


def test_negative_initial_velocity_decelerates_first() -> None:
    t = np.linspace(0, 100, 1001)
    state = analyze_constant_force(MASS, 1e10, -0.5 * C, 0.0, t)
    v = state["velocity"]
    assert v[0] < 0
    assert v[-1] > 0  # strong force eventually accelerates into +x


def test_point_mass_with_large_force_raises_on_overflow() -> None:
    """An absurd force x time must raise, never silently propagate inf/FTL."""
    with pytest.raises((InvalidSimulationParameterError, VelocityLimitError)):
        analyze_constant_force(1e-30, 1e60, 0.0, 0.0, 10.0)


def test_rk4_rejects_bad_grid() -> None:
    with pytest.raises(InvalidSimulationParameterError):
        integrate_rk4(1.0, 1.0, 0.0, 0.0, time=np.array([0.0, 1.0, 0.5]))
    with pytest.raises(InvalidSimulationParameterError):
        integrate_rk4(1.0, 1.0, 0.0, 0.0)  # nothing given
    with pytest.raises(InvalidSimulationParameterError):
        integrate_rk4(1.0, 1.0, 0.0, 0.0, duration=1.0)  # dt missing


def test_analyze_scalar_inputs_return_scalars() -> None:
    state = analyze_constant_force(3.0, 2.0, 0.1 * C, 0.0, 5.0)
    for v in state.values():
        assert isinstance(v, float), f"{v!r} is not a scalar"


def test_analyze_zero_time_returns_initial_state() -> None:
    state = analyze_constant_force(3.0, 2.0, 0.1 * C, 7.0, 0.0)
    assert state["position"] == 7.0
    assert state["velocity"] == pytest.approx(0.1 * C, rel=1e-12)
    assert state["proper_time"] == 0.0
    assert state["kinetic_energy"] >= 0.0


def test_energy_conservation_consistent_with_work() -> None:
    """total_energy - rest_energy == kinetic_energy along the whole trajectory."""
    t = np.linspace(0, 100, 1001)
    state = analyze_constant_force(MASS, FORCE, 0.5 * C, 0.0, t)
    rest = MASS * C_SQUARED
    assert _rel_bounds(state["total_energy"] - rest, state["kinetic_energy"]) < 1e-12