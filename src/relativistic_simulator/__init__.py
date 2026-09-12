"""relativistic-simulator: 1D special-relativistic constant-force particle simulator.

This package is a **ground-truth physics engine** for machine-learning
research.  It generates exact synthetic observations from known physics
(1D special-relativistic constant-force massive-particle dynamics in flat
Minkowski spacetime).  It does **not** discover or learn physics: it is the
teacher, not the student.  A separate ML experiment may later study whether
a model can learn the mappings generated here and generalise toward the
ultra-relativistic regime.

Core API
--------
>>> from relativistic_simulator import simulate, generate_dataset, C
>>> result = simulate(mass=1000.0, force=1e6, initial_velocity=0.9 * C,
...                   duration=100.0, dt=0.01)
>>> validation = result.validate()
>>> validation.passed
True

Dataset generation
------------------
>>> ds = generate_dataset(n_samples=100_000, beta_range=(0.0, 0.8), random_seed=42)
>>> ds.beta.max() < 1.0
True
"""

from __future__ import annotations

from . import classical, plotting
from ._common import (
    as_float_array,
    require_finite,
    require_positive_mass,
    require_subluminal,
    validate_duration,
    validate_time_step,
)
from .classical import (  # noqa: F401
    classical_kinetic_energy,
    classical_momentum,
    classical_position,
    classical_velocity,
)
from .constants import C, C_SQUARED
from .dataset import RelativisticDataset, generate_dataset
from .dynamics import analyze_constant_force, integrate_rk4
from .exceptions import (
    InvalidMassError,
    InvalidSimulationParameterError,
    InvalidTimeStepError,
    RelativisticSimulatorError,
    VelocityLimitError,
)
from .relativity import (
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
from .simulator import simulate
from .trajectory import Trajectory
from .validation import (
    DatasetValidation,
    TrajectoryValidation,
    validate_dataset,
    validate_trajectory,
)

__version__ = "0.1.0"

__all__ = [
    # constants
    "C",
    "C_SQUARED",
    # errors
    "RelativisticSimulatorError",
    "InvalidMassError",
    "VelocityLimitError",
    "InvalidTimeStepError",
    "InvalidSimulationParameterError",
    # kinematics / energy
    "beta_from_velocity",
    "velocity_from_beta",
    "gamma_from_beta",
    "gamma_from_velocity",
    "gamma_from_momentum",
    "momentum_from_velocity",
    "velocity_from_momentum",
    "energy_from_velocity",
    "energy_from_momentum",
    "kinetic_energy_from_velocity",
    "kinetic_energy_from_momentum",
    "rest_energy",
    # simulation
    "simulate",
    "analyze_constant_force",
    "integrate_rk4",
    "Trajectory",
    # datasets
    "generate_dataset",
    "RelativisticDataset",
    # validation
    "validate_trajectory",
    "validate_dataset",
    "TrajectoryValidation",
    "DatasetValidation",
    # classical comparison
    "classical_velocity",
    "classical_position",
    "classical_kinetic_energy",
    "classical_momentum",
]