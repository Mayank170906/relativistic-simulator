"""Tests for the public simulate() API and the Trajectory container."""
from __future__ import annotations

import numpy as np
import pytest

from relativistic_simulator import C
from relativistic_simulator.exceptions import (
    InvalidMassError,
    InvalidSimulationParameterError,
    InvalidTimeStepError,
    VelocityLimitError,
)
from relativistic_simulator.simulator import simulate
from relativistic_simulator.trajectory import TIME_VARYING_COLUMNS

MASS, FORCE = 1000.0, 1e6


def test_simulate_basic_contract() -> None:
    traj = simulate(mass=MASS, force=FORCE, duration=100.0, dt=0.01)
    assert len(traj) == 10001
    assert traj.time[0] == 0.0
    assert traj.time[-1] == pytest.approx(100.0)
    for name in TIME_VARYING_COLUMNS:
        arr = getattr(traj, name)
        assert arr.shape == (10001,)
        assert np.issubdtype(arr.dtype, np.floating)


def test_simulate_arrays_match_analyze() -> None:
    traj = simulate(mass=MASS, force=FORCE, initial_velocity=0.5 * C, duration=10.0, dt=0.01)
    assert traj.momentum[-1] == pytest.approx(traj.momentum[0] + FORCE * traj.time[-1], rel=1e-12)
    assert traj.beta[-1] == pytest.approx(traj.velocity[-1] / C, rel=1e-12)


def test_simulate_zero_duration_single_point() -> None:
    traj = simulate(mass=MASS, force=FORCE, duration=0.0, dt=0.01)
    assert len(traj) == 1
    assert traj.position[0] == pytest.approx(0.0)
    assert traj.proper_time[0] == 0.0


def test_simulate_dt_larger_than_duration() -> None:
    traj = simulate(mass=MASS, force=FORCE, duration=0.5, dt=10.0)
    assert len(traj) == 2
    assert traj.time[-1] == pytest.approx(0.5)


def test_final_state_dict() -> None:
    traj = simulate(mass=MASS, force=FORCE, duration=10.0, dt=0.1)
    final = traj.final_state()
    assert set(final) == set(TIME_VARYING_COLUMNS)
    assert final["gamma"] == pytest.approx(traj.gamma[-1], rel=1e-15)


def test_to_numpy_shape_and_columns() -> None:
    traj = simulate(mass=MASS, force=FORCE, duration=10.0, dt=0.1)
    arr = traj.to_numpy()
    assert arr.shape == (101, 9)
    assert list(traj.columns) == list(TIME_VARYING_COLUMNS)


def test_to_dict_includes_parameters() -> None:
    traj = simulate(mass=MASS, force=FORCE, duration=10.0, dt=0.1)
    d = traj.to_dict()
    assert d["mass"] == MASS
    assert d["force"] == FORCE
    assert d["method"] == "analytic"
    assert d["time"].shape == (101,)


def test_to_dataframe_pandas() -> None:
    pd = pytest.importorskip("pandas")
    traj = simulate(mass=MASS, force=FORCE, duration=10.0, dt=0.1)
    df = traj.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == list(TIME_VARYING_COLUMNS)
    assert len(df) == 101


def test_numerical_method_matches_analytic() -> None:
    ana = simulate(mass=MASS, force=FORCE, initial_velocity=0.9 * C, duration=100.0, dt=0.1)
    num = simulate(mass=MASS, force=FORCE, initial_velocity=0.9 * C, duration=100.0, dt=0.1, method="numerical")
    # relative agreement (RK4 truncation on a 2.7e10 m trajectory)
    rel = np.max(np.abs(ana.position - num.position)) / np.max(np.abs(ana.position))
    assert rel < 1e-9
    assert num.method == "numerical"


def test_reproducible_deterministic() -> None:
    a = simulate(mass=MASS, force=FORCE, duration=10.0, dt=0.01)
    b = simulate(mass=MASS, force=FORCE, duration=10.0, dt=0.01)
    np.testing.assert_array_equal(a.position, b.position)
    np.testing.assert_array_equal(a.momentum, b.momentum)


# --------------------------------------------------------------------------- #
# Invalid input handling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_mass(bad: float) -> None:
    with pytest.raises(InvalidMassError):
        simulate(mass=bad, force=FORCE, duration=10.0, dt=0.1)


@pytest.mark.parametrize("v0", [C, -C, C + 1, np.nan, np.inf])
def test_invalid_initial_velocity(v0: float) -> None:
    with pytest.raises(VelocityLimitError):
        simulate(mass=MASS, force=FORCE, initial_velocity=v0, duration=10.0, dt=0.1)


@pytest.mark.parametrize("dt", [0.0, -0.01, np.nan, np.inf])
def test_invalid_dt(dt: float) -> None:
    with pytest.raises(InvalidTimeStepError):
        simulate(mass=MASS, force=FORCE, duration=10.0, dt=dt)


@pytest.mark.parametrize("dur", [-1.0, np.nan, np.inf])
def test_invalid_duration(dur: float) -> None:
    with pytest.raises(InvalidSimulationParameterError):
        simulate(mass=MASS, force=FORCE, duration=dur, dt=0.1)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_invalid_force(bad: float) -> None:
    with pytest.raises(InvalidSimulationParameterError):
        simulate(mass=MASS, force=bad, duration=10.0, dt=0.1)


def test_invalid_method() -> None:
    with pytest.raises(InvalidSimulationParameterError):
        simulate(mass=MASS, force=FORCE, duration=10.0, dt=0.1, method="leapfrog")


def test_high_beta_stability() -> None:
    """Starting at 0.99999c, the force must never push the particle to c."""
    traj = simulate(mass=MASS, force=1e6, initial_velocity=0.99999 * C, duration=100.0, dt=0.05)
    assert np.max(np.abs(traj.beta)) < 1.0
    assert np.all(np.isfinite(traj.gamma))
    assert traj.validate().passed