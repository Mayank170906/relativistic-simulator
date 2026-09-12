"""Special-relativistic kinematics and energy-momentum relations (1 spatial dimension).

All relations here are derived for *massive* particles in *flat Minkowski
spacetime* (special relativity) with rest mass ``m > 0`` and one spatial
dimension.  The full set of assumptions is listed in ``docs/physics.md``.

Every function:

* accepts a Python scalar or a NumPy array (vectorised where practical),
* works in SI units (``m`` in kg, ``v`` in m/s, ``p`` in kg·m/s, ``E`` in J),
* validates inputs and raises :mod:`relativistic_simulator.exceptions` for
  unphysical states — it **never** silently clips ``|v| >= c``.

Numerical stability notes
-------------------------
Several textbook formulas suffer catastrophic cancellation near the
relativistic limit or for small momenta.  The iterated forms used here are:

* :math:`1 - \\beta^2 = (1-\\beta)(1+\\beta)` (accurate when ``beta -> 1``),
* :math:`\\gamma - 1 = \\gamma^2\\beta^2/(\\gamma+1)` (accurate when ``beta -> 0``),
* :math:`E - m c^2 = p^2 c^2/(E + m c^2)` (accurate when ``p -> 0``),
* ``np.hypot`` for ``sqrt(a^2 + b^2)`` (avoids overflow/underflow).
"""
from __future__ import annotations

import numpy as np

from ._common import (
    broadcast,
    require_finite,
    require_positive_mass,
    require_subluminal,
    require_valid_beta,
    scalar_or_array,
)
from .constants import C, C_SQUARED

__all__ = [
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
]


def beta_from_velocity(velocity: float | np.ndarray) -> float | np.ndarray:
    """Speed ratio ``beta = v / c`` from an SI velocity.

    Parameters
    ----------
    velocity : float | np.ndarray
        Velocity in m/s with ``|v| < c`` for every element.

    Returns
    -------
    float | np.ndarray
        Dimensionless ``beta`` in ``[-1, 1)``.

    Raises
    ------
    VelocityLimitError
        If any ``|v| >= c`` or the input is non-finite.
    """
    v, was_scalar = require_subluminal(velocity)
    return scalar_or_array(v / C, was_scalar)


def velocity_from_beta(beta: float | np.ndarray) -> float | np.ndarray:
    """Velocity ``v = beta * c`` in m/s from a speed ratio.

    Parameters
    ----------
    beta : float | np.ndarray
        Dimensionless speed ratio with ``|beta| < 1``.

    Returns
    -------
    float | np.ndarray
        Velocity in m/s.

    Raises
    ------
    VelocityLimitError
        If any ``|beta| >= 1`` (a massive particle cannot move at ``c``).
    """
    b, was_scalar = require_valid_beta(beta)
    return scalar_or_array(b * C, was_scalar)


def gamma_from_beta(beta: float | np.ndarray) -> float | np.ndarray:
    r"""Lorentz factor :math:`\gamma = (1-\beta^2)^{-1/2}`.

    Uses the cancellation-free form :math:`1-\beta^2 \to (1-\beta)(1+\beta)`.

    Parameters
    ----------
    beta : float | np.ndarray
        Dimensionless speed ratio with ``|beta| < 1``.

    Returns
    -------
    float | np.ndarray
        Lorentz factor, ``gamma >= 1``.

    Raises
    ------
    VelocityLimitError
        If any ``|beta| >= 1`` or the input is non-finite.
    """
    b, was_scalar = require_valid_beta(beta)
    gamma = 1.0 / np.sqrt((1.0 - b) * (1.0 + b))
    return scalar_or_array(gamma, was_scalar)


def gamma_from_velocity(velocity: float | np.ndarray) -> float | np.ndarray:
    """Lorentz factor from an SI velocity ``v`` (m/s)."""
    v, was_scalar = require_subluminal(velocity)
    beta = v / C
    gamma = 1.0 / np.sqrt((1.0 - beta) * (1.0 + beta))
    return scalar_or_array(gamma, was_scalar)


def gamma_from_momentum(mass: float | np.ndarray, momentum: float | np.ndarray) -> float | np.ndarray:
    r"""Lorentz factor from momentum: :math:`\gamma = \sqrt{1 + (p/(m c))^2}`.

    This inversion is exact (no loss of precision) and well-defined for any
    finite momentum, since ``|p/(m c)|`` can be arbitrarily large but never
    causes ``gamma`` to overflow (``np.hypot`` is used).

    Parameters
    ----------
    mass : float | np.ndarray
        Rest mass in kg, ``m > 0``.
    momentum : float | np.ndarray
        Relativistic momentum in kg·m/s (any sign, finite).

    Returns
    -------
    float | np.ndarray
        Lorentz factor ``gamma >= 1``.

    Raises
    ------
    InvalidMassError
        If ``mass <= 0`` or non-finite.
    InvalidSimulationParameterError
        If ``momentum`` is non-finite.
    """
    m, _ = require_positive_mass(mass)
    p, was_scalar = require_finite(momentum, "momentum")
    m, p = broadcast(m, p)
    gamma = np.hypot(1.0, p / (m * C))
    return scalar_or_array(gamma, was_scalar)


def momentum_from_velocity(mass: float | np.ndarray, velocity: float | np.ndarray) -> float | np.ndarray:
    r"""Relativistic momentum :math:`p = \gamma m v`.

    Parameters
    ----------
    mass : float | np.ndarray
        Rest mass in kg, ``m > 0``.
    velocity : float | np.ndarray
        Velocity in m/s with ``|v| < c``.

    Returns
    -------
    float | np.ndarray
        Momentum in kg·m/s, same sign as velocity.

    Raises
    ------
    InvalidMassError, VelocityLimitError
        For unphysical mass or velocity.
    """
    m, _ = require_positive_mass(mass)
    v, was_scalar = require_subluminal(velocity)
    m, v = broadcast(m, v)
    beta = v / C
    gamma = 1.0 / np.sqrt((1.0 - beta) * (1.0 + beta))
    return scalar_or_array(gamma * m * v, was_scalar)
def velocity_from_momentum(mass: float | np.ndarray, momentum: float | np.ndarray) -> float | np.ndarray:
    r"""Velocity from momentum: :math:`v = p c^2 / \sqrt{m^2 c^4 + p^2 c^2}`.

    This is the exact relativistic inversion, equivalent to
    :math:`v = c\, \tanh(\operatorname{arsinh}(p/(m c)))`.  The form
    ``c * p / hypot(m * c, p)`` is numerically stable for all ``|p|``.

    Parameters
    ----------
    mass : float | np.ndarray
        Rest mass in kg, ``m > 0``.
    momentum : float | np.ndarray
        Relativistic momentum in kg·m/s (any sign, finite).

    Returns
    -------
    float | np.ndarray
        Velocity in m/s, always satisfying ``|v| < c``.

    Raises
    ------
    InvalidMassError, InvalidSimulationParameterError
        For unphysical mass or momentum.
    """
    m, _ = require_positive_mass(mass)
    p, was_scalar = require_finite(momentum, "momentum")
    m, p = broadcast(m, p)
    v = C * p / np.hypot(m * C, p)
    return scalar_or_array(v, was_scalar)


def energy_from_velocity(mass: float | np.ndarray, velocity: float | np.ndarray) -> float | np.ndarray:
    r"""Total energy :math:`E = \gamma m c^2` from velocity.

    Parameters
    ----------
    mass : float | np.ndarray
        Rest mass in kg, ``m > 0``.
    velocity : float | np.ndarray
        Velocity in m/s with ``|v| < c``.

    Returns
    -------
    float | np.ndarray
        Total energy in joules (J).

    Raises
    ------
    InvalidMassError, VelocityLimitError
        For unphysical mass or velocity.
    """
    m, _ = require_positive_mass(mass)
    v, was_scalar = require_subluminal(velocity)
    m, v = broadcast(m, v)
    beta = v / C
    gamma = 1.0 / np.sqrt((1.0 - beta) * (1.0 + beta))
    return scalar_or_array(gamma * m * C_SQUARED, was_scalar)


def energy_from_momentum(mass: float | np.ndarray, momentum: float | np.ndarray) -> float | np.ndarray:
    r"""Total energy :math:`E = \sqrt{m^2 c^4 + p^2 c^2}` from momentum.

    Uses ``np.hypot`` to avoid overflow when ``|p c|`` is large.

    Parameters
    ----------
    mass : float | np.ndarray
        Rest mass in kg, ``m > 0``.
    momentum : float | np.ndarray
        Relativistic momentum in kg·m/s (any sign, finite).

    Returns
    -------
    float | np.ndarray
        Total energy in joules (J).

    Raises
    ------
    InvalidMassError, InvalidSimulationParameterError
        For unphysical mass or momentum.
    """
    m, _ = require_positive_mass(mass)
    p, was_scalar = require_finite(momentum, "momentum")
    m, p = broadcast(m, p)
    energy = C * np.hypot(m * C, p)
    return scalar_or_array(energy, was_scalar)
def kinetic_energy_from_velocity(mass: float | np.ndarray, velocity: float | np.ndarray) -> float | np.ndarray:
    r"""Relativistic kinetic energy :math:`K = (\gamma - 1) m c^2` from velocity.

    Uses the cancellation-free form :math:`\gamma - 1 = \gamma^2\beta^2/(\gamma+1)`
    so that ``K`` is accurate both for ``beta -> 0`` (where ``gamma - 1``
    loses digits) and for ``beta -> 1``.

    Parameters
    ----------
    mass : float | np.ndarray
        Rest mass in kg, ``m > 0``.
    velocity : float | np.ndarray
        Velocity in m/s with ``|v| < c``.

    Returns
    -------
    float | np.ndarray
        Kinetic energy in joules (J), ``K >= 0``.

    Raises
    ------
    InvalidMassError, VelocityLimitError
        For unphysical mass or velocity.
    """
    m, _ = require_positive_mass(mass)
    v, was_scalar = require_subluminal(velocity)
    m, v = broadcast(m, v)
    beta = v / C
    gamma = 1.0 / np.sqrt((1.0 - beta) * (1.0 + beta))
    kinetic = m * C_SQUARED * gamma * gamma * beta * beta / (gamma + 1.0)
    return scalar_or_array(kinetic, was_scalar)


def kinetic_energy_from_momentum(mass: float | np.ndarray, momentum: float | np.ndarray) -> float | np.ndarray:
    r"""Kinetic energy from momentum: :math:`K = p^2 c^2 / (E + m c^2)`.

    Algebraically equal to :math:`E - m c^2` but stable as ``p -> 0``
    (the direct subtraction ``E - m c^2`` loses all digits when the kinetic
    energy is small compared with the rest energy).

    Parameters
    ----------
    mass : float | np.ndarray
        Rest mass in kg, ``m > 0``.
    momentum : float | np.ndarray
        Relativistic momentum in kg·m/s (any sign, finite).

    Returns
    -------
    float | np.ndarray
        Kinetic energy in joules (J), ``K >= 0``.

    Raises
    ------
    InvalidMassError, InvalidSimulationParameterError
        For unphysical mass or momentum.
    """
    m, _ = require_positive_mass(mass)
    p, was_scalar = require_finite(momentum, "momentum")
    m, p = broadcast(m, p)
    rest = m * C_SQUARED
    energy = C * np.hypot(m * C, p)
    kinetic = p * p * C_SQUARED / (energy + rest)
    return scalar_or_array(kinetic, was_scalar)


def rest_energy(mass: float | np.ndarray) -> float | np.ndarray:
    r"""Rest (invariant-mass) energy :math:`E_0 = m c^2`.

    Parameters
    ----------
    mass : float | np.ndarray
        Rest mass in kg, ``m > 0``.

    Returns
    -------
    float | np.ndarray
        Rest energy in joules (J).

    Raises
    ------
    InvalidMassError
        If ``mass <= 0`` or non-finite.
    """
    m, was_scalar = require_positive_mass(mass)
    return scalar_or_array(m * C_SQUARED, was_scalar)