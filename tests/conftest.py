"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pytest

from relativistic_simulator import C

BETA_GRID = np.array(
    [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999]
)

MASS = 1000.0
FORCE = 1e6
DURATION = 100.0
DT = 0.01


@pytest.fixture
def beta_grid() -> np.ndarray:
    """High-beta test grid: 0.5c, 0.7c, 0.8c, 0.9c, ..., 0.99999c."""
    return BETA_GRID


@pytest.fixture
def basic_trajectory():
    """Rest particle under 1e6 N for 100 s."""
    from relativistic_simulator import simulate

    return simulate(mass=MASS, force=FORCE, duration=DURATION, dt=DT)


@pytest.fixture
def coasting_trajectory():
    """Zero force, moving at 0.9c (uniform relativistic motion)."""
    from relativistic_simulator import simulate

    return simulate(mass=MASS, force=0.0, initial_velocity=0.9 * C, duration=DURATION, dt=DT)


@pytest.fixture
def high_beta_trajectory():
    """Particle starting at 0.9c under a constant force."""
    from relativistic_simulator import simulate

    return simulate(
        mass=MASS,
        force=FORCE,
        initial_velocity=0.9 * C,
        duration=DURATION,
        dt=DT,
    )


@pytest.fixture
def numerical_trajectory():
    """Same physical case as basic_trajectory but via RK4."""
    from relativistic_simulator import simulate

    return simulate(mass=MASS, force=FORCE, duration=DURATION, dt=DT, method="numerical")