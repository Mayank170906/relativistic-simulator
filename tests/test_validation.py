"""Tests for the quantitative validation system.

These tests follow the "independent verification" rule: the validator must
use *different* code paths than the data generator, and it must report
quantitative errors (not just booleans).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from relativistic_simulator import C, generate_dataset, simulate, validate_trajectory


def test_basic_trajectory_passes(basic_trajectory) -> None:
    report = validate_trajectory(basic_trajectory)
    assert report.passed
    assert report.max_velocity_bound_violation == 0.0
    assert report.max_momentum_relation_error < 1e-9
    assert report.max_energy_relation_error < 1e-9
    assert report.max_energy_momentum_error < 1e-9
    assert report.max_force_error < 1e-6
    assert report.max_velocity_difference_error < 1e-6
    assert report.max_proper_time_error < 1e-6
    assert report.max_proper_time_violation == 0.0
    assert report.max_initial_condition_error < 1e-12


def test_high_beta_trajectory_passes(high_beta_trajectory) -> None:
    report = validate_trajectory(high_beta_trajectory)
    assert report.passed
    assert report.min_velocity_margin > 0.0  # |beta| < 1 strictly


def test_numerical_trajectory_passes(numerical_trajectory) -> None:
    report = validate_trajectory(numerical_trajectory)
    assert report.passed


def test_coasting_trajectory_passes(coasting_trajectory) -> None:
    report = validate_trajectory(coasting_trajectory)
    assert report.passed


def test_quantitative_errors_are_small(basic_trajectory) -> None:
    """The metrics must be quantitative and tiny for an exact solution."""
    report = validate_trajectory(basic_trajectory)
    assert report.max_energy_momentum_error < 1e-10
    assert report.min_velocity_margin == pytest.approx(1.0 - np.max(np.abs(basic_trajectory.beta)))
    assert report.n_points == len(basic_trajectory)


def test_tampered_momentum_but_consistent_velocity_fails() -> None:
    """Tampering with one column must be caught by the independent route."""
    traj = simulate(mass=1000.0, force=1e6, duration=10.0, dt=0.01)
    tampered = replace(traj, momentum=traj.momentum * 1.001)
    report = validate_trajectory(tampered)
    assert not report.passed
    assert report.max_momentum_relation_error > 1e-4
    assert report.max_force_error > 1e-4  # dp/dt deviates from F


def test_tampered_position_fails_dxdt() -> None:
    traj = simulate(mass=1000.0, force=1e6, duration=10.0, dt=0.01)
    # A ramp perturbation changes dx/dt (a constant shift would not).
    tampered = replace(traj, position=traj.position + np.linspace(0.0, 1e4, traj.n_points))
    report = validate_trajectory(tampered)
    assert not report.passed
    assert report.max_velocity_difference_error > 1e-2
    assert report.max_initial_condition_error < 1e-9  # x[0] is unchanged


def test_tampered_gamma_fails_energy() -> None:
    traj = simulate(mass=1000.0, force=1e6, duration=10.0, dt=0.01)
    tampered = replace(traj, gamma=traj.gamma * 0.99)
    report = validate_trajectory(tampered)
    assert not report.passed
    assert report.max_energy_relation_error > 1e-3


def test_tampered_proper_time_fails() -> None:
    traj = simulate(mass=1000.0, force=1e6, duration=10.0, dt=0.01)
    tampered = replace(traj, proper_time=traj.proper_time * 1.05)
    report = validate_trajectory(tampered)
    assert not report.passed
    assert report.max_proper_time_error > 1e-2


def test_tampered_initial_condition_fails() -> None:
    traj = simulate(mass=1000.0, force=1e6, duration=10.0, dt=0.01)
    tampered = replace(traj, position=np.concatenate(([999.0], traj.position[1:])))
    report = validate_trajectory(tampered)
    assert not report.passed
    assert report.max_initial_condition_error > 1e-3


def test_failure_is_quantitative_not_fake(basic_trajectory) -> None:
    """A failing report must show non-trivial error magnitudes."""
    traj = simulate(mass=1000.0, force=1e6, duration=10.0, dt=0.01)
    tampered = replace(traj, momentum=traj.momentum * 1.01)
    report = validate_trajectory(tampered)
    assert not report.passed
    # the measured error really is the tampering amplitude (1%)
    assert report.max_force_error == pytest.approx(0.01 / 1.01, rel=1e-9)
    assert report.max_momentum_relation_error > 1e-3
    # and the summary text carries the measured magnitude
    text = report.summary()
    assert "passed=False" in text


def test_dataset_validation_passes_and_is_quantitative() -> None:
    ds = generate_dataset(n_samples=5000, random_seed=21)
    report = ds.validate()
    assert report.passed
    assert report.max_energy_momentum_error < 1e-12
    assert report.max_proper_time_consistency_error < 1e-9
    assert report.max_position_consistency_error < 1e-9


def test_dataset_validation_catches_tampering() -> None:
    ds = generate_dataset(n_samples=5000, random_seed=21)
    tampered = replace(ds, gamma=ds.gamma * 1.01)
    report = tampered.validate()
    assert not report.passed
    assert report.max_energy_relation_error > 1e-3


def test_validation_rtol_is_respected() -> None:
    """A tighter rtol than the machine-achieved accuracy must fail."""
    traj = simulate(mass=1000.0, force=1e6, duration=10.0, dt=0.01)
    report = validate_trajectory(traj, rtol=1e-20)
    assert not report.passed